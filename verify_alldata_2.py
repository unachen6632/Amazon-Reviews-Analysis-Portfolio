# filename: validate_all.py
# 作用：
# - 總驗證腳本，用於檢查 1_clean_data.py (core_analysis.py) 的所有 6 個輸出檔案。
# - 驗證 (1) 檔案是否存在 (2) 結構/表頭是否正確 (3) 大型檔案的資料完整性 (行數)。
# - (v2 更新：配合最新的中文資料字典定義)

import sys
import json
import csv
import pandas as pd
from pathlib import Path

# --- 1. 基本設定（必須與 1_clean_data.py 保持一致）---

# 預期路徑
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_PROCESSED = BASE_DIR / "data_processed"
ENCODING = "utf-8-sig" # 匹配 ETL 輸出

# 預期檔案名稱 (來自 meta.json)
PATH_FILTERED = DATA_PROCESSED / "amazon_reviews_2010_2015_filtered.csv"
PATH_CLEAN = DATA_PROCESSED / "reviews_clean.csv"
PATH_AGG_ALL = DATA_PROCESSED / "monthly_agg_all.csv"
PATH_AGG_VER = DATA_PROCESSED / "monthly_agg_verified.csv"
PATH_DICT = DATA_PROCESSED / "data_dictionary.json"
PATH_META = DATA_PROCESSED / "clean_meta.json"


# --- 2. 預期結構（來自 1_clean_data.py 的程式邏輯）---

# 【修正】這裡更新為您指定的最新簡化版翻譯
EXPECTED_DATA_DICT = {
    # 基礎欄位 (根據您最新的翻譯要求)
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
    
    # 基礎衍生欄位 (原本的定義)
    "review_year": "評論年份",
    "review_month": "評論月份",
    
    # 進階衍生欄位 (原本的定義)
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

# 來自 1_clean_data.py -> OnlineMonthAgg._to_frame 函式
EXPECTED_COLS_AGG = [
    "review_year","review_month","n","avg_star","neg_rate",
    "service_neg_rate","w_neg_rate"
]

# 來自 1_clean_data.py -> base_cols 變數
EXPECTED_COLS_FILTERED = [
    "marketplace","customer_id","review_id","product_id","product_parent",
    "product_title","product_category","star_rating","helpful_votes","total_votes",
    "verified_purchase","review_headline","review_body","review_date",
    "review_year","review_month"
]

# 來自 1_clean_data.py -> clean_cols 變數
EXPECTED_COLS_CLEAN = [
    "marketplace","customer_id","review_id","product_id","product_parent",
    "product_title","product_category","star_rating","helpful_votes","total_votes",
    "verified_purchase","review_headline","review_body","review_date",
    "review_year","review_month",
    "is_verified","is_vine","helpful_ratio","weight","review_body_clean",
    "sentiment_label","cs_issue","delivery_issue","return_issue",
    "fulfillment_issue","service_any","is_neg","is_service_neg"
]

# --- 3. 驗證輔助函式 ---

def check_path(p: Path) -> bool:
    """檢查檔案是否存在"""
    if not p.exists():
        print(f"❌ [失敗] 檔案不存在: {p.name}")
        return False
    return True

def validate_json(filepath: Path, expected_content: dict = None) -> bool:
    """驗證 JSON 檔案：(1) 存在 (2) 可讀 (3) [可選] 內容匹配"""
    print(f"🔬 驗證 (JSON): {filepath.name}")
    if not check_path(filepath): return False
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if expected_content:
            # 為了容錯，我們可以檢查 key 是否都存在，或者完全匹配
            # 這裡使用完全匹配
            if data == expected_content:
                print(f"✅ [成功] {filepath.name} 內容完全匹配。")
                return True
            else:
                print(f"❌ [失敗] {filepath.name} 內容不匹配！")
                # 顯示差異 (Debug 用)
                diff_keys = [k for k in expected_content if k not in data or data[k] != expected_content[k]]
                if diff_keys:
                    print(f"   - 差異欄位範例 (前3個): {diff_keys[:3]}")
                    print(f"   - 預期: {expected_content[diff_keys[0]]}")
                    print(f"   - 實際: {data.get(diff_keys[0], '不存在')}")
                return False
        
        # 如果只是檢查可讀性
        print(f"✅ [成功] {filepath.name} 格式正確。")
        return True
        
    except Exception as e:
        print(f"❌ [失敗] {filepath.name} 讀取或解析失敗: {e}")
        return False

def validate_csv_header(filepath: Path, expected_cols: list) -> bool:
    """(適用小型 CSV) 驗證表頭"""
    print(f"🔬 驗證 (CSV Header): {filepath.name}")
    if not check_path(filepath): return False
    
    try:
        with open(filepath, "r", encoding=ENCODING, newline='') as f:
            header = next(csv.reader(f))
        
        if header == expected_cols:
            print(f"✅ [成功] {filepath.name} 表頭 (欄位) OK。")
            return True
        else:
            print(f"❌ [失敗] {filepath.name} 表頭 (欄位) 不匹配！")
            print(f"   - 預期: {expected_cols}")
            print(f"   - 實際: {header}")
            return False
            
    except Exception as e:
        print(f"❌ [失敗] {filepath.name} 讀取失敗: {e}")
        return False

def validate_large_csv(filepath: Path, expected_cols: list, expected_rows: int) -> bool:
    """(適用大型 CSV) 串流驗證表頭和總行數"""
    print(f"🔬 驗證 (Large CSV): {filepath.name}")
    if not check_path(filepath): return False

    try:
        # 1. 驗證表頭
        header_df = pd.read_csv(filepath, encoding=ENCODING, nrows=0, low_memory=False)
        header = list(header_df.columns)
            
        if header != expected_cols:
            print(f"❌ [失敗] {filepath.name} 表頭 (欄位) 不匹配！")
            print(f"   - 預期: {expected_cols}")
            print(f"   - 實際: {header}")
            return False
        
        # 2. 驗證行數 (Pandas 串流計數)
        row_count = 0
        reader_col = expected_cols[0] # 優化：只讀一欄
        
        chunk_iter = pd.read_csv(
            filepath, 
            encoding=ENCODING, 
            chunksize=200_000,
            low_memory=False,
            usecols=[reader_col]
        )
        
        for chunk in chunk_iter:
            row_count += len(chunk)
        
        if row_count != expected_rows:
            print(f"❌ [失敗] {filepath.name} 總行數不匹配！")
            print(f"   - 預期 (來自 meta.json): {expected_rows:,} 行")
            print(f"   - 實際 (檔案內): {row_count:,} 行")
            return False
            
        print(f"✅ [成功] {filepath.name} 表頭 (欄位) OK。")
        print(f"✅ [成功] {filepath.name} 總行數 OK ({row_count:,} 行)。")
        return True
            
    except Exception as e:
        print(f"❌ [失敗] {filepath.name} 驗證時發生程式錯誤: {e}")
        return False

# --- 4. 主執行流程 ---

def main():
    print("=" * 60)
    print("開始執行 ETL (1_clean_data.py) 總輸出驗證...")
    print(f"資料夾: {DATA_PROCESSED}")
    print("=" * 60)
    
    results = {}
    
    # 步驟 0: 讀取 Meta - 這是後續驗證的基礎
    print(f"🔬 驗證 (JSON): {PATH_META.name}")
    if not check_path(PATH_META):
        print("❌ [致命錯誤] meta.json 不存在。無法繼續驗證。")
        sys.exit()
    try:
        with open(PATH_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        EXPECTED_ROWS = meta["rows_kept"]
        print(f"✅ [成功] {PATH_META.name} 讀取成功。")
        print(f"   ...預期資料行數: {EXPECTED_ROWS:,} 行")
        results["meta"] = True
    except Exception as e:
        print(f"❌ [致命錯誤] {PATH_META.name} 讀取或解析失敗: {e}")
        sys.exit()

    print("-" * 60)
    
    # 步驟 1: 驗證 Data Dictionary
    results["dict"] = validate_json(PATH_DICT, EXPECTED_DATA_DICT)
    
    print("-" * 60)
    
    # 步驟 2: 驗證月彙總表 (小型 CSV)
    results["agg_all"] = validate_csv_header(PATH_AGG_ALL, EXPECTED_COLS_AGG)
    results["agg_ver"] = validate_csv_header(PATH_AGG_VER, EXPECTED_COLS_AGG)

    print("-" * 60)

    # 步驟 3: 驗證大型資料檔案 (大型 CSV)
    results["filtered"] = validate_large_csv(PATH_FILTERED, EXPECTED_COLS_FILTERED, EXPECTED_ROWS)
    results["clean"] = validate_large_csv(PATH_CLEAN, EXPECTED_COLS_CLEAN, EXPECTED_ROWS)

    print("=" * 60)
    
    # 總結
    total_checks = len(results)
    success_checks = sum(results.values())
    
    if success_checks == total_checks:
        print(f"✅✅✅ 總結論：全部 {total_checks} 項檢查均已通過！")
        print("ETL 輸出資料已 100% 驗證正確。")
    else:
        print(f"❌❌❌ 總結論：{total_checks} 項檢查中，有 {total_checks - success_checks} 項失敗。")
        print("請檢查上方的 [失敗] 訊息。")
    
    print("=" * 60)

if __name__ == "__main__":
    main()