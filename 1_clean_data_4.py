# filename: core_analysis.py
# 作用：
# - 串流讀取 amazon_reviews_multilingual_US_v1_00.tsv（支援 .tsv/.tsv.gz/.csv/.csv.gz 或 CLI 指定）
# - 過濾：US、2010–2015、星等 1..5、排除 Vine
# - 同步輸出：
#   1) data_processed/amazon_reviews_2010_2015_filtered.csv（中繼基礎欄，逐塊 append）
#   2) data_processed/reviews_clean.csv（完整清理後逐筆，逐塊 append）
#   3) data_processed/monthly_agg_all.csv、monthly_agg_verified.csv（線上月彙整）
#   4) data_processed/data_dictionary.json、clean_meta.json
#
# 僅輸出 CSV，不需任何 parquet 依賴。

import os, sys, re, csv, json, warnings, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
pd.options.mode.copy_on_write = True

# ---------------- 基本設定 ----------------
DEFAULT_IN_NAME = "amazon_reviews_multilingual_US_v1_00.tsv"
MARKET    = "US"
YEAR_MIN  = 2010
YEAR_MAX  = 2015
CHUNKSIZE = 200_000     # 依機器調整；低記憶體可降 100_000
MAX_CHUNKS = None       # 僅測試可設 2；正式請用 None
ENCODING  = "utf-8-sig" # 輸出 CSV 讓 Excel 直接讀

# 路徑
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_RAW        = BASE_DIR / "data_raw"
DATA_PROCESSED  = BASE_DIR / "data_processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# 輸出檔
OUT_FILTERED = DATA_PROCESSED / "amazon_reviews_2010_2015_filtered.csv"
OUT_CLEAN    = DATA_PROCESSED / "reviews_clean.csv"
OUT_AGG_ALL  = DATA_PROCESSED / "monthly_agg_all.csv"
OUT_AGG_VER  = DATA_PROCESSED / "monthly_agg_verified.csv"
OUT_DICT     = DATA_PROCESSED / "data_dictionary.json"
OUT_META     = DATA_PROCESSED / "clean_meta.json"

# 欄位字典
DATA_DICT = {
    # 原始欄位（依你提供的中譯）
    "marketplace": "市場",
    "customer_id": "顧客編號",
    "review_id": "評論編號",
    "product_id": "商品編號",
    "product_parent": "產品主編號",
    "product_title": "產品標題",
    "product_category": "產品類別",
    "star_rating": "星等",
    "helpful_votes": "有幫助的票數",
    "total_votes": "總票數",
    "vine": "Vine 計畫標記",
    "verified_purchase": "已驗證購買",
    "review_headline": "評論標題",
    "review_body": "評論內容",
    "review_date": "評論日期",

    # 內部衍生欄位（保留）
    "review_year": "評論年份",
    "review_month": "評論月份",
    "is_verified": "是否已驗證購買",
    "is_vine": "是否 Vine",
    "helpful_ratio": "有幫助票比例",
    "weight": "評價權重(1+log1p(helpful))，封頂8",
    "review_body_clean": "清理後評論文本",
    "sentiment_label": "粗情緒標籤(星等→Neg/Neu/Pos)",
    "cs_issue": "客服問題",
    "delivery_issue": "物流問題",
    "return_issue": "退換/退款問題",
    "fulfillment_issue": "出貨/品項問題",
    "service_any": "任一服務問題",
    "is_neg": "是否負評(<=2★)",
    "is_service_neg": "服務相關負評"
}

# ---------------- 文本清理與服務關鍵字 ----------------
def clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"http\S+|www\.\S+", " ", regex=True)
    s = s.str.replace(r"<[^>]+>", " ", regex=True)
    s = s.str.replace(r"[^\w\s]", " ", regex=True)  # 移除標點與符號
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

CS_KW  = ['customer service','no reply','unhelpful','rude','support','agent']
DLV_KW = ['late delivery','missed delivery','never arrived','tracking','took forever','slow delivery']
RET_KW = ['return process','no refund','refund','exchange issue','return label','restocking fee']
FUL_KW = ['wrong item','damaged','broken','empty box','missing parts','packaging']

def build_service_flags(s: pd.Series) -> pd.DataFrame:
    s = s.fillna("").astype(str).str.lower()
    esc = lambda ks: "|".join(map(re.escape, ks))
    out = pd.DataFrame({
        "cs_issue":            s.str.contains(esc(CS_KW),  regex=True, na=False),
        "delivery_issue":      s.str.contains(esc(DLV_KW), regex=True, na=False),
        "return_issue":        s.str.contains(esc(RET_KW), regex=True, na=False),
        "fulfillment_issue":   s.str.contains(esc(FUL_KW), regex=True, na=False),
    })
    out["service_any"] = out.any(axis=1)
    return out

# ---------------- 型別與衍生欄位 ----------------
def memory_shrink(df: pd.DataFrame) -> pd.DataFrame:
    # 數值降精度
    for c, t in (("star_rating","Int32"), ("helpful_votes","Int32"), ("total_votes","Int32")):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(t)
    # 分類型（節省記憶體）
    for c in ("marketplace","vine","verified_purchase","product_category"):
        if c in df.columns:
            try:
                if df[c].nunique(dropna=True) <= max(50, len(df)//50):
                    df[c] = df[c].astype("category")
            except Exception:
                pass
    return df

TRUTHY = {"Y","YES","TRUE","T","VERIFIED PURCHASE"}

def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 時間
    dt = pd.to_datetime(df["review_date"], errors="coerce")
    df["review_year"]  = dt.dt.year.astype("Int16")
    df["review_month"] = dt.dt.month.astype("Int8")

    # 已驗證/Vine
    vp = df["verified_purchase"].fillna("").astype(str).str.strip().str.upper()
    df["is_verified"] = vp.isin(TRUTHY)
    vn = df["vine"].fillna("").astype(str).str.strip().str.upper()
    df["is_vine"]     = vn.isin(TRUTHY)

    # 權重與比例
    tv = pd.to_numeric(df["total_votes"], errors="coerce").fillna(0).astype("float32")
    hv = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype("float32")
    df["helpful_ratio"] = np.where(tv>0, hv/tv, 0.0).astype("float32")
    w = (1 + np.log1p(hv)).astype("float32")
    w[w>8] = 8
    df["weight"] = w

    # 文本與服務 flags
    df["review_body_clean"] = clean_text(df["review_body"])
    svc = build_service_flags(df["review_body"])
    df = pd.concat([df, svc], axis=1)

    # 星等→粗情緒標籤
    sr = pd.to_numeric(df["star_rating"], errors="coerce")
    lab = pd.Series("Unknown", index=df.index, dtype="string")
    lab.loc[sr.le(2)] = "Negative"
    lab.loc[sr.eq(3)] = "Neutral"
    lab.loc[sr.ge(4)] = "Positive"
    df["sentiment_label"] = lab

    # 便於彙整
    df["is_neg"] = sr.le(2)
    df["is_service_neg"] = df["service_any"] & df["is_neg"]

    return df

# ---------------- 月聚合（線上累加） ----------------
class OnlineMonthAgg:
    def __init__(self):
        self.all  = {}
        self.ver  = {}

    @staticmethod
    def _update_bucket(bucket: dict, key, n, sum_star, neg, svc_neg, w_sum, w_neg):
        if key not in bucket:
            bucket[key] = {"n":0, "sum_star":0.0, "neg":0, "svc_neg":0, "w_sum":0.0, "w_neg":0.0}
        b = bucket[key]
        b["n"]      += int(n)
        b["sum_star"] += float(sum_star)
        b["neg"]    += int(neg)
        b["svc_neg"]+= int(svc_neg)
        b["w_sum"]  += float(w_sum)
        b["w_neg"]  += float(w_neg)

    def ingest(self, df: pd.DataFrame):
        if df.empty: return
        sr = pd.to_numeric(df["star_rating"], errors="coerce")

        # 全量
        grp = df.groupby(["review_year","review_month"], dropna=True, as_index=False)
        a = grp.agg(
            n=("review_id","count"),
            sum_star=("star_rating", "sum"),
            neg=("review_id", lambda s: int((sr.loc[s.index] <= 2).sum())),
            svc_neg=("review_id", lambda s: int(((df.loc[s.index,"service_any"]) & (sr.loc[s.index] <= 2)).sum())),
            w_sum=("weight","sum"),
            w_neg=("review_id", lambda s: float(np.sum((sr.loc[s.index] <= 2) * df.loc[s.index,"weight"])))
        )
        for _, r in a.iterrows():
            self._update_bucket(self.all, (int(r.review_year), int(r.review_month)),
                                r.n, r.sum_star, r.neg, r.svc_neg, r.w_sum, r.w_neg)

        # Verified
        v = df[df["is_verified"]].copy()
        if not v.empty:
            vsr = pd.to_numeric(v["star_rating"], errors="coerce")
            grp = v.groupby(["review_year","review_month"], dropna=True, as_index=False)
            a = grp.agg(
                n=("review_id","count"),
                sum_star=("star_rating","sum"),
                neg=("review_id", lambda s: int((vsr.loc[s.index] <= 2).sum())),
                svc_neg=("review_id", lambda s: int(((v.loc[s.index,"service_any"]) & (vsr.loc[s.index] <= 2)).sum())),
                w_sum=("weight","sum"),
                w_neg=("review_id", lambda s: float(np.sum((vsr.loc[s.index] <= 2) * v.loc[s.index,"weight"])))
            )
            for _, r in a.iterrows():
                self._update_bucket(self.ver, (int(r.review_year), int(r.review_month)),
                                    r.n, r.sum_star, r.neg, r.svc_neg, r.w_sum, r.w_neg)

    @staticmethod
    def _to_frame(bucket: dict) -> pd.DataFrame:
        if not bucket:
            return pd.DataFrame(columns=["review_year","review_month","n","avg_star","neg_rate","service_neg_rate","w_neg_rate"])
        rows = []
        for (y,m), b in bucket.items():
            avg_star = (b["sum_star"]/b["n"]) if b["n"] else np.nan
            neg_rate = (b["neg"]/b["n"]) if b["n"] else 0.0
            svc_neg_rate = (b["svc_neg"]/b["n"]) if b["n"] else 0.0
            w_neg_rate = (b["w_neg"]/b["w_sum"]) if b["w_sum"]>0 else 0.0
            rows.append({
                "review_year": y, "review_month": m, "n": int(b["n"]),
                "avg_star": round(float(avg_star), 4) if b["n"] else np.nan,
                "neg_rate": round(float(neg_rate), 4),
                "service_neg_rate": round(float(svc_neg_rate), 4),
                "w_neg_rate": round(float(w_neg_rate), 4)
            })
        return pd.DataFrame(rows).sort_values(["review_year","review_month"])

    def frames(self):
        return self._to_frame(self.all), self._to_frame(self.ver)

# ---------------- 讀檔工具 ----------------
def find_input_path() -> Path:
    # 1) CLI 指定
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
        else:
            print("[WARN] CLI 指定路徑不存在：", p)

    # 2) 常見位置
    candidates = []
    for root in [DATA_RAW, BASE_DIR, Path.cwd()]:
        for pat in [DEFAULT_IN_NAME, "amazon_reviews_multilingual_US_v1_00.tsv.gz",
                    "amazon_reviews_multilingual_US_v1_00.csv", "amazon_reviews_multilingual_US_v1_00.csv.gz"]:
            p = root / pat
            if p.exists():
                candidates.append(p)
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        f"找不到輸入檔（{DEFAULT_IN_NAME} / .tsv.gz / .csv / .csv.gz）。"
        f"\n請將檔案放至 {DATA_RAW} 或以完整路徑當參數傳入。"
    )

def open_reader(path: Path):
    name = path.name.lower()
    sep = "\t" if (name.endswith(".tsv") or name.endswith(".tsv.gz")) else ","
    comp = "infer"
    dtype = {
        "marketplace":"string","customer_id":"string","review_id":"string","product_id":"string",
        "product_parent":"string","product_title":"string","product_category":"string",
        "star_rating":"string","helpful_votes":"string","total_votes":"string",
        "vine":"string","verified_purchase":"string","review_headline":"string",
        "review_body":"string","review_date":"string",
    }
    usecols = list(dtype.keys())
    try:
        reader = pd.read_csv(path, sep=sep, dtype=dtype, usecols=usecols,
                             chunksize=CHUNKSIZE, engine="c", quoting=csv.QUOTE_MINIMAL,
                             on_bad_lines="skip", low_memory=False, compression=comp)
        engine_used = "c"
    except Exception as e:
        print(f"[WARN] C 引擎失敗，改用 python：{e}")
        reader = pd.read_csv(path, sep=sep, dtype=dtype, usecols=usecols,
                             chunksize=CHUNKSIZE, engine="python", quoting=csv.QUOTE_MINIMAL,
                             on_bad_lines="skip", low_memory=False, compression=comp)
        engine_used = "python"
    return reader, sep, engine_used

# ---------------- 主流程 ----------------
def main():
    src = find_input_path()
    print("[INFO] 使用輸入檔：", src.resolve())

    # 準備輸出（若存在就先刪）
    for p in [OUT_FILTERED, OUT_CLEAN, OUT_AGG_ALL, OUT_AGG_VER]:
        if p.exists():
            p.unlink()

    # 旗標：首塊寫表頭
    filtered_first = True
    clean_first = True

    # 統計
    meta = {
        "engine": None, "chunks": 0, "rows_read": 0,
        "rows_us": 0, "rows_star_ok": 0, "rows_year_ok": 0, "rows_non_vine": 0,
        "rows_kept": 0, "verified_ratio": 0.0
    }
    seen_ids = set()
    agg = OnlineMonthAgg()
    vp_counter = Counter()

    reader, sep, engine_used = open_reader(src)
    meta["engine"] = engine_used

    for chunk in reader:
        meta["chunks"] += 1
        meta["rows_read"] += len(chunk)

        # 以 review_id 去重（跨 chunk）
        if "review_id" in chunk.columns:
            before = len(chunk)
            chunk = chunk[~chunk["review_id"].isin(seen_ids)].copy()
            seen_ids.update(chunk["review_id"].dropna().tolist())

        # 市場 + 星等篩選
        chunk = chunk[chunk["marketplace"] == MARKET]
        meta["rows_us"] += len(chunk)

        # 星等合法化
        chunk["star_rating"] = pd.to_numeric(chunk["star_rating"], errors="coerce")
        chunk = chunk[(chunk["star_rating"]>=1) & (chunk["star_rating"]<=5)].copy()
        meta["rows_star_ok"] += len(chunk)

        # 型別降載與衍生欄位
        chunk = memory_shrink(chunk)
        chunk = derive_columns(chunk)

        # 年分 + 排除 Vine
        mask_year = (chunk["review_year"]>=YEAR_MIN) & (chunk["review_year"]<=YEAR_MAX)
        chunk = chunk[mask_year].copy()
        meta["rows_year_ok"] += len(chunk)
        chunk = chunk[~chunk["is_vine"]].copy()
        meta["rows_non_vine"] += len(chunk)

        if chunk.empty:
            print(f"[{meta['chunks']:>4}] kept=0（過濾後為空）")
            if MAX_CHUNKS and meta["chunks"] >= MAX_CHUNKS:
                break
            continue

        # 更新 verified 統計
        vp_counter.update(chunk["is_verified"].astype(bool).map({True:"T", False:"F"}))

        # --- 產出「中繼基礎欄」：amazon_reviews_2010_2015_filtered.csv ---
        base_cols = [
            "marketplace","customer_id","review_id","product_id","product_parent",
            "product_title","product_category","star_rating","helpful_votes","total_votes",
            "verified_purchase","review_headline","review_body","review_date",
            "review_year","review_month"
        ]
        keep = [c for c in base_cols if c in chunk.columns]

        # ✅ 第一次用 'w' 且 header=True；之後用 'a' 且 header=False
        mode = "w" if filtered_first else "a"
        chunk.loc[:, keep].to_csv(
            OUT_FILTERED,
            mode=mode, index=False, encoding=ENCODING,
            header=filtered_first, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        # 首塊寫完立即自檢表頭
        if filtered_first:
            _hdr = list(pd.read_csv(OUT_FILTERED, nrows=0, encoding=ENCODING).columns)
            if _hdr != keep:
                raise RuntimeError(f"[HEADER ERROR] {OUT_FILTERED.name} 表頭不一致：{_hdr} != {keep}")
        filtered_first = False

        # --- 產出「清理後逐筆」：reviews_clean.csv ---
        clean_cols = list(dict.fromkeys(keep + [
            "is_verified","is_vine","helpful_ratio","weight","review_body_clean",
            "sentiment_label","cs_issue","delivery_issue","return_issue",
            "fulfillment_issue","service_any","is_neg","is_service_neg"
        ]))
        mode = "w" if clean_first else "a"
        chunk.loc[:, [c for c in clean_cols if c in chunk.columns]].to_csv(
            OUT_CLEAN,
            mode=mode, index=False, encoding=ENCODING,
            header=clean_first, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        if clean_first:
            _hdr2 = list(pd.read_csv(OUT_CLEAN, nrows=0, encoding=ENCODING).columns)
            if _hdr2 != [c for c in clean_cols if c in chunk.columns]:
                raise RuntimeError(f"[HEADER ERROR] {OUT_CLEAN.name} 表頭不一致")
        clean_first = False

        # --- 月彙整線上累加 ---
        agg.ingest(chunk)

        kept_now = len(chunk)
        meta["rows_kept"] += kept_now
        print(f"[{meta['chunks']:>4}] kept={kept_now:,}  total_kept={meta['rows_kept']:,}")

        if MAX_CHUNKS and meta["chunks"] >= MAX_CHUNKS:
            break

    # 彙整輸出
    df_all, df_ver = agg.frames()
    df_all.to_csv(OUT_AGG_ALL, index=False, encoding=ENCODING, lineterminator="\n")
    df_ver.to_csv(OUT_AGG_VER, index=False, encoding=ENCODING, lineterminator="\n")

    # 資料字典
    with open(OUT_DICT, "w", encoding="utf-8") as f:
        json.dump(DATA_DICT, f, ensure_ascii=False, indent=2)

    # 簡易 meta
    meta["verified_ratio"] = round(vp_counter["T"] / max(1, (vp_counter["T"]+vp_counter["F"])), 4)
    meta["input"] = str(src.resolve())
    meta["sep"] = "\\t" if sep == "\t" else ","
    meta["outputs"] = {
        "filtered": str(OUT_FILTERED.resolve()),
        "clean": str(OUT_CLEAN.resolve()),
        "monthly_all": str(OUT_AGG_ALL.resolve()),
        "monthly_verified": str(OUT_AGG_VER.resolve()),
        "dict": str(OUT_DICT.resolve())
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[DONE] core_analysis 完成 ✅")
    print(" - 中繼：", OUT_FILTERED.resolve())
    print(" - 逐筆：", OUT_CLEAN.resolve())
    print(" - 月彙整（全量）：", OUT_AGG_ALL.resolve())
    print(" - 月彙整（Verified）：", OUT_AGG_VER.resolve())
    print(" - 資料字典：", OUT_DICT.resolve())
    print(" - Meta：", OUT_META.resolve())
    print(f" - Verified 比例（估）：{meta['verified_ratio']*100:.2f}%")
    print(f" - 使用引擎：{meta['engine']}，處理 chunks：{meta['chunks']}")
    print(f" - 保留筆數：{meta['rows_kept']:,}（US={meta['rows_us']:,}，星等OK={meta['rows_star_ok']:,}，年分OK={meta['rows_year_ok']:,}，非Vine={meta['rows_non_vine']:,}）")

if __name__ == "__main__":
    main()
