# filename: generate_all_plots.py
# 功能：合併「2_generate_plots」與「3_generate_service_plots」的圖表邏輯為單一可執行腳本
# 產出圖檔：
#   1_2_combined_volume_avgstar.png
#   3_monthly_neg_rate_heatmap.png
#   4_bar_top_categories.png
#   5_stacked_bar_sentiment_dist.png
#   6_bar_service_issues.png
#   7_grouped_bar_issues_by_verified.png
#   9_service_specific_keyword_ranking.png
#   10_service_neg_trends_combo.png（目前在 main 裡預設註解，不會輸出）

from matplotlib import colors as mcolors
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from pathlib import Path
import json, sys, re
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# =========================
# 基本路徑與環境設定
# =========================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_PROCESSED = BASE_DIR / "data_processed"
DATA_FIGURES   = BASE_DIR / "data_figures"
DATA_FIGURES.mkdir(parents=True, exist_ok=True)

PATH_CLEAN_CSV = DATA_PROCESSED / "reviews_clean.csv"
PATH_AGG_ALL   = DATA_PROCESSED / "monthly_agg_all.csv"
PATH_AGG_VER   = DATA_PROCESSED / "monthly_agg_verified.csv"
PATH_DICT      = DATA_PROCESSED / "data_dictionary.json"
PATH_META      = DATA_PROCESSED / "clean_meta.json"

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 顏色與對照
# =========================
AMAZON_ORANGE = "#FF9900"
AMAZON_NAVY   = "#232F3E"
AMAZON_LINK   = "#007185"
AMAZON_GRAY   = "#6B7280"
ANNOT_RED     = "#D00000"
ANNOT_GREEN   = "#2E7D32"

FALLBACK_CAT_ZH = {
    "Wireless": "無線通訊產品", "Mobile_Apps": "APP", "PC": "電腦與周邊",
    "Digital_Ebook_Purchase": "數位電子書",
    "Home": "居家生活用品", "Home_Improvement": "居家修繕/裝修",
    "Health_&_Personal_Care": "健康與個人護理", "Automotive": "汽機車配件",
    "Sports": "運動用品", "Books": "書籍", "Lawn_and_Garden": "園藝與草坪",
    "Beauty": "美妝美容", "Kitchen": "廚房用品", "Apparel": "服飾", "Toys": "玩具",
    "Digital_Video_Download": "數位影音下載", "Video DVD": "DVD",
    "Home Entertainment": "家庭娛樂", "Camera": "相機", "Video Games": "電玩遊戲",
    "Electronics": "消費性電子", "Musical Instruments": "樂器",
    "Digital_Music_Purchase": "數位音樂",
    "Digital_Music": "數位音樂",
    "Music": "音樂",
}

# 服務關鍵字（延伸分析用）
CS_KW  = ['customer service','no reply','unhelpful','rude','support','agent']
DLV_KW = ['late delivery','missed delivery','never arrived','tracking','took forever','slow delivery']
RET_KW = ['return process','no refund','refund','exchange issue','return label','restocking fee']
FUL_KW = ['wrong item','damaged','broken','empty box','missing parts','packaging']
ALL_KW = CS_KW + DLV_KW + RET_KW + FUL_KW

KEYWORD_TRANSLATIONS = {
    'customer service': '客服', 'no reply': '未回覆', 'unhelpful': '無幫助',
    'rude': '態度粗魯', 'support': '(技術)支援', 'agent': '客服專員',
    'late delivery': '延遲送達', 'missed delivery': '錯過送達', 'never arrived': '包裹未到',
    'tracking': '物流追蹤', 'took forever': '等太久', 'slow delivery': '運送緩慢',
    'return process': '退貨流程', 'no refund': '沒有退款', 'refund': '退款',
    'exchange issue': '換貨問題', 'return label': '退貨標籤', 'restocking fee': '重整費',
    'wrong item': '商品錯誤', 'damaged': '損壞', 'broken': '破損', 'empty box': '空盒',
    'missing parts': '缺少零件', 'packaging': '包裝'
}

# =========================
# 輔助工具
# =========================
def save_plot(fig, filename: str):
    try:
        filepath = DATA_FIGURES / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"✅ 已儲存圖表: {filepath.name}")
    except Exception as e:
        print(f"❌ 儲存圖表 {filename} 失敗: {e}")
    plt.close(fig)

def load_data():
    print("... 正在載入彙總檔案/字典 ...")
    try:
        agg_all = pd.read_csv(PATH_AGG_ALL)
        agg_ver = pd.read_csv(PATH_AGG_VER)
        agg_all["date"] = pd.to_datetime(
            agg_all["review_year"].astype(str) + "-" + agg_all["review_month"].astype(str)
        )
        agg_ver["date"] = pd.to_datetime(
            agg_ver["review_year"].astype(str) + "-" + agg_ver["review_month"].astype(str)
        )
        with open(PATH_DICT, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        return {"agg_all": agg_all, "agg_ver": agg_ver, "dict": data_dict}
    except FileNotFoundError as e:
        print(f"❌ 找不到必要的 CSV/JSON。{e}")
        print("請先完成清理與彙總：1_clean_data.py / monthly_agg*.csv")
        sys.exit()

def fmt_k(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    return f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"

def robust_bool_converter(series: pd.Series) -> pd.Series:
    try:
        if series.dtype == "bool":
            return series
        map_ = {
            True: True, False: False, 1: True, 0: False,
            "True": True, "False": False, "TRUE": True, "FALSE": False,
            "1": True, "0": False, "Y": True, "N": False
        }
        return series.map(map_).fillna(False).astype(bool)
    except Exception:
        return series.astype(str).str.upper().isin(["TRUE", "1", "T", "Y"])

def to_zh_category(name, data_dict):
    key = str(name)
    return data_dict.get(key, FALLBACK_CAT_ZH.get(key, key))

def translate_sentiment_to_zh(s, data_dict):
    return data_dict.get(s, {"Positive": "正向", "Neutral": "中立", "Negative": "負向"}.get(s, s))

# —— 水平/垂直通用：標柱的最高/最低（智慧位置，避重疊）——
def _outside_or_inside_for_hbar(ax, val):
    xmin, xmax = ax.get_xlim()
    return "inside" if val >= xmax * 0.94 else "outside"

def annotate_min_max_bars(ax):
    patches = [p for p in ax.patches if isinstance(p, Rectangle)]
    if not patches:
        return
    lengths = [max(p.get_width(), p.get_height()) for p in patches]
    i_max = int(np.argmax(lengths))
    i_min = int(np.argmin(lengths))

    def _place_label(p, is_max, dy_stack=0):
        color = ANNOT_RED if is_max else ANNOT_GREEN
        txt = "最高 " if is_max else "最低 "
        val = max(p.get_width(), p.get_height())

        # 垂直條
        if p.get_height() >= p.get_width():
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + p.get_height()
            _, ymax = ax.get_ylim()
            if y >= ymax * 0.96:  # 太頂：放柱內
                ax.text(
                    x, y - (p.get_height()*0.05) - 3, f"{txt}{fmt_k(val)}",
                    ha="center", va="top", fontsize=10, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
                    zorder=5, clip_on=False
                )
            else:
                ax.text(
                    x, y + 3 + dy_stack, f"{txt}{fmt_k(val)}",
                    ha="center", va="bottom", fontsize=10, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
                    zorder=5, clip_on=False
                )
        # 水平條
        else:
            x = p.get_x() + p.get_width()
            y = p.get_y() + p.get_height() / 2
            pos = _outside_or_inside_for_hbar(ax, x)
            if pos == "inside":
                ax.text(
                    x - 4, y + dy_stack, f"{txt}{fmt_k(val)}",
                    ha="right", va="center", fontsize=10, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
                    zorder=5, clip_on=False
                )
            else:
                ax.text(
                    x + 4, y + dy_stack, f"{txt}{fmt_k(val)}",
                    ha="left", va="center", fontsize=10, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
                    zorder=5, clip_on=False
                )

    patches[i_max].set_facecolor(ANNOT_RED)
    patches[i_min].set_facecolor(ANNOT_GREEN)
    _place_label(patches[i_max], True, dy_stack=6)
    _place_label(patches[i_min], False, dy_stack=-6)

# —— 折線：依指定顏色做極值標註（用於雙軸圖）——
def annotate_extrema_line_by_color(ax, x, y, color, txt_fmt, dy=10):
    if len(y) == 0 or pd.Series(y).isnull().all():
        return
    y = pd.Series(y).astype(float)
    idx_max = int(np.nanargmax(y))
    idx_min = int(np.nanargmin(y))
    x_max, y_max = x.iloc[idx_max], float(y.iloc[idx_max])
    x_min, y_min = x.iloc[idx_min], float(y.iloc[idx_min])

    _, y_top = ax.get_ylim()
    dy_max = -abs(dy) if y_max >= y_top * 0.95 else abs(dy)
    dy_min = abs(dy)

    ax.scatter([x_max, x_min], [y_max, y_min], s=54, color=color, edgecolor="white", zorder=5)
    ax.annotate(
        f"最高 {txt_fmt(y_max)}", xy=(x_max, y_max),
        xytext=(0, dy_max), textcoords="offset points",
        ha="center", va="bottom" if dy_max > 0 else "top",
        fontsize=10, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
        zorder=6, clip_on=False
    )
    ax.annotate(
        f"最低 {txt_fmt(y_min)}", xy=(x_min, y_min),
        xytext=(0, dy_min), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=9, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
        zorder=6, clip_on=False
    )

def annotate_extrema_bar_by_color(ax, x, h, color, txt_fmt):
    h = pd.Series(h).astype(float)
    if h.empty:
        return
    i_max = int(np.nanargmax(h))
    i_min = int(np.nanargmin(h))
    x_max, v_max = x.iloc[i_max], float(h.iloc[i_max])
    x_min, v_min = x.iloc[i_min], float(h.iloc[i_min])
    _, y_top = ax.get_ylim()
    pad = max(y_top * 0.01, 6)
    ax.annotate(
        f"最高 {txt_fmt(v_max)}", xy=(x_max, v_max),
        xytext=(0, pad if v_max < y_top*0.96 else -pad),
        textcoords="offset points",
        ha="center", va="bottom" if v_max < y_top*0.96 else "top",
        fontsize=10, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
        zorder=6, clip_on=False
    )
    ax.annotate(
        f"最低 {txt_fmt(v_min)}", xy=(x_min, v_min),
        xytext=(0, pad), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=9, color=color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
        zorder=6, clip_on=False
    )

def annotate_extrema_line_red_green(ax, x, y, txt_fmt=lambda v: f"{v:.1f}", dy=10):
    s = pd.Series(y).astype(float)
    if s.empty or s.isna().all():
        return

    imax = int(np.nanargmax(s))
    imin = int(np.nanargmin(s))
    x_max, y_max = x.iloc[imax], float(s.iloc[imax])
    x_min, y_min = x.iloc[imin], float(s.iloc[imin])

    _, y_top = ax.get_ylim()
    dy_max = -abs(dy) if y_max >= y_top * 0.95 else abs(dy)
    dy_min = abs(dy)

    ax.scatter([x_max, x_min], [y_max, y_min], s=54,
               color=[ANNOT_RED, ANNOT_GREEN], edgecolor="white", zorder=5)

    ax.annotate(
        f"最高 {txt_fmt(y_max)}", xy=(x_max, y_max),
        xytext=(0, dy_max), textcoords="offset points",
        ha="center", va="bottom" if dy_max > 0 else "top",
        fontsize=10, color=ANNOT_RED,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ANNOT_RED, lw=0.9),
        zorder=6, clip_on=False
    )

    ax.annotate(
        f"最低 {txt_fmt(y_min)}", xy=(x_min, y_min),
        xytext=(0, dy_min), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=9, color=ANNOT_GREEN,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ANNOT_GREEN, lw=0.9),
        zorder=6, clip_on=False
    )

# =========================
# 圖表 (1)～(3)
# =========================
def plot_1_2_combined_volume_and_star(df_all, df_ver, data_dict):
    print("... (Extra) 評論數(長條) + 平均星等(折線) ...")
    x = pd.to_datetime(df_all["date"])
    n = df_all["n"].astype(float)
    star_all = df_all["avg_star"].astype(float)
    star_ver = df_ver["avg_star"].astype(float)

    star_label     = data_dict.get("star_rating", "星等")
    verified_label = data_dict.get("verified_purchase", "已驗證購買")
    date_label     = data_dict.get("review_date", "日期")

    LIGHT_GRAY = "#BEBAB7"
    ORANGE     = "#FF9900"
    LINE_ALL   = "#232F3E"
    LINE_VER   = "#2E6F40"

    def _fmt_k_spaced(v: float) -> str:
        return f"{int(round(float(v) / 1000.0))} k"

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.92)
    ax1.set_axisbelow(True)
    ax1.grid(axis='y', zorder=0, linestyle='--', alpha=0.7)
    ax1.margins(x=0)

    # 左軸：評論數
    bars = ax1.bar(x, n, color=LIGHT_GRAY, width=25, alpha=0.95, zorder=2, label="評論數")

    years = x.dt.year.values
    idx_global_max = int(np.argmax(n.values))
    bars[idx_global_max].set_color(ORANGE)

    idx_2013 = None
    mask2013 = (years == 2013)
    if mask2013.any():
        i_in_2013 = np.where(mask2013)[0]
        idx_2013 = int(i_in_2013[np.argmax(n.values[i_in_2013])])
        bars[idx_2013].set_color(ORANGE)

    def _label_bar(i: int, text: str, color: str):
        b = bars[i]
        xi = b.get_x() + b.get_width() / 2
        yi = b.get_height()
        ax1.annotate(
            text, xy=(xi, yi), xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.9),
            clip_on=True, zorder=5
        )

    _label_bar(idx_global_max, _fmt_k_spaced(n.iloc[idx_global_max]), ANNOT_RED)
    if idx_2013 is not None and idx_2013 != idx_global_max:
        _label_bar(idx_2013, _fmt_k_spaced(n.iloc[idx_2013]), ANNOT_RED)

    idx_min = int(np.argmin(n.values))
    bars[idx_min].set_color(ORANGE)
    _label_bar(idx_min, _fmt_k_spaced(n.iloc[idx_min]), ANNOT_GREEN)

    ax1.set_xlabel(date_label, fontsize=12, color="black")
    ax1.set_ylabel("評論數", fontsize=12, color="black")
    ax1.tick_params(axis="x", labelcolor="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: fmt_k(v)))
    ax1.set_ylim(0, float(n.max()) * 1.15)

    # 右軸：平均星等
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(x, star_all, color=LINE_ALL, linewidth=2.0,
             label="平均星等(已驗證+未驗證購買)", zorder=3)
    ax2.plot(x, star_ver, color=LINE_VER, linewidth=2.0, linestyle="--",
             label=f"平均{star_label}（{verified_label}）", zorder=3)
    ax2.set_ylabel(f"平均{star_label}", fontsize=12, color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(4.0, 5.0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left",
               frameon=True, facecolor="white", edgecolor="gray", framealpha=0.9)

    plt.title("評論數 與 平均星等趨勢", fontsize=16, pad=6)
    save_plot(fig, "1_2_combined_volume_avgstar.png")

def plot_3_heatmap_neg_rate(df_all, data_dict):
    print("... (3/9) 每月負面評論率 (Heatmap 風格折線) ...")

    df = df_all.rename(columns={
        "review_year": "year",
        "review_month": "month",
        "neg_rate": "neg_rate"
    }).copy()

    df = df[df["month"].between(1, 12)]

    months = list(range(1, 13))
    pivot = (
        df.pivot_table(index="year", columns="month", values="neg_rate", aggfunc="mean")
          .sort_index()
          .reindex(columns=months)
    )

    years_sorted = list(pivot.index)[-5:]

    COLORS = ["#FF0000", "#FF7F00", "#FFD400", "#0072FF", "#7B1FA2"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for i, yr in enumerate(years_sorted):
        yvals = pivot.loc[yr, months].values.astype(float)
        ax.plot(
            months, yvals,
            label=str(int(yr)),
            color=COLORS[i],
            linewidth=2.2, marker="o", markersize=3, zorder=2
        )

    ax.set_title("每月負面評論率", fontsize=16, pad=12)
    ax.set_xlabel("月份", fontsize=12)
    ax.set_ylabel("負評率", fontsize=12)
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}月" for m in months])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    ax.legend(
        title="年份",
        frameon=True, facecolor="white", edgecolor="gray", framealpha=0.9,
        ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02)
    )

    ax.margins(x=0)
    save_plot(fig, "3_monthly_neg_rate_heatmap.png")

# =========================
# 圖 4：Top 6 + 其他
# =========================
def plot_4_bar_top_categories(data_dict):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    ORANGE = "#FF9900"
    LIGHT_GRAY = "#BEBAB7"

    var_map = dict(FALLBACK_CAT_ZH)
    var_map.update({k.replace("_", " "): v for k, v in FALLBACK_CAT_ZH.items()})
    var_map.update({k.replace("_&_", " & "): v for k, v in FALLBACK_CAT_ZH.items()})
    var_map["Video_DVD"] = "DVD"
    var_map["Mobile Apps"] = "APP"
    var_map["Lawn and Garden"] = "園藝與草坪"
    var_map["Health & Personal Care"] = "健康與個人護理"
    var_map["Home Improvement"] = "居家修繕/裝修"

    df = pd.read_csv(PATH_CLEAN_CSV, usecols=["product_category"], low_memory=False)
    df["product_category"] = df["product_category"].astype(str)

    counts_en = df["product_category"].value_counts().sort_values(ascending=False)
    top15_en = counts_en.index.tolist()[:15]

    def to_display(name: str) -> str:
        key = str(name)
        label = data_dict.get(key, var_map.get(key, key))
        if label == key:
            k1 = key.replace("_", " ")
            k2 = key.replace("_&_", " & ")
            label = var_map.get(k1, var_map.get(k2, label))
        if key in ("Mobile_Apps", "Mobile Apps") or label == "行動應用程式 (App)":
            label = "APP"
        if key in ("Video DVD", "Video_DVD") or label == "Video DVD":
            label = "DVD"
        return label

    disp = df["product_category"].map(to_display)
    counts_zh = disp.value_counts().sort_values(ascending=False)

    if len(counts_zh) > 6:
        top6 = counts_zh.iloc[:6]
        tail = counts_zh.iloc[6:]
        others_avg = tail.mean() if len(tail) > 0 else 0
        counts_plot = pd.concat([top6, pd.Series({"其他": others_avg})])
    else:
        counts_plot = counts_zh.copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

    y = np.arange(len(counts_plot))
    colors = [ORANGE if i < 4 else LIGHT_GRAY for i in range(len(counts_plot))]

    bars = ax.barh(y, counts_plot.values, color=colors, edgecolor="none", zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(counts_plot.index.tolist(), fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("評論數", fontsize=12)
    ax.set_title("評論聲量 Top 6 產品類別 + 其他", fontsize=16, pad=10)

    ax.set_xlim(0, 1_600_000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{int(v/1000)}k"))

    xmax = ax.get_xlim()[1]
    pad = max(xmax * 0.008, 8)
    for b in bars[:min(4, len(bars))]:
        val_txt = f"{int(round(b.get_width() / 1000.0))} k"
        ax.text(
            b.get_x() + b.get_width() + pad,
            b.get_y() + b.get_height() / 2,
            val_txt,
            ha="left",
            va="center",
            fontsize=11,
            color=ANNOT_RED,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=ANNOT_RED,
                lw=1.2,
            ),
            zorder=5,
            clip_on=False,
        )

    save_plot(fig, "4_bar_top_categories.png")
    return top15_en

# =========================
# 圖 5：各品類情緒佔比（Top15 中文）
# =========================
def plot_5_stacked_bar_sentiment_dist(top_cats_en, data_dict):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    var_map = dict(FALLBACK_CAT_ZH)
    var_map.update({k.replace("_", " "): v for k, v in FALLBACK_CAT_ZH.items()})
    var_map.update({k.replace("_&_", " & "): v for k, v in FALLBACK_CAT_ZH.items()})
    var_map["Video_DVD"] = "DVD"
    var_map["Mobile Apps"] = "APP"
    var_map["Lawn and Garden"] = "園藝與草坪"
    var_map["Health & Personal Care"] = "健康與個人護理"
    var_map["Home Improvement"] = "居家修繕/裝修"

    def to_display(name: str) -> str:
        key = str(name)
        label = data_dict.get(key, var_map.get(key, key))
        if label == key:
            k1 = key.replace("_", " ")
            k2 = key.replace("_&_", " & ")
            label = var_map.get(k1, var_map.get(k2, label))
        if key in ("Mobile_Apps", "Mobile Apps") or label == "行動應用程式 (App)":
            label = "APP"
        if key in ("Video DVD", "Video_DVD") or label == "Video DVD":
            label = "DVD"
        return label

    use_cols = ["product_category", "sentiment_label"]
    df = pd.read_csv(PATH_CLEAN_CSV, usecols=use_cols, low_memory=False)
    df["product_category"] = df["product_category"].astype(str)
    df["cat_zh"] = df["product_category"].map(to_display)

    if top_cats_en:
        zh_list = [to_display(en) for en in top_cats_en]
        df = df[df["cat_zh"].isin(zh_list)].copy()
    else:
        zh_list = df["cat_zh"].value_counts().nlargest(15).index.tolist()
        df = df[df["cat_zh"].isin(zh_list)].copy()

    df["sentiment_label"] = df["sentiment_label"].map(
        {"Positive": "正向", "Neutral": "中立", "Negative": "負向"}
    ).fillna(df["sentiment_label"])

    ct = pd.crosstab(df["cat_zh"], df["sentiment_label"])
    for col in ("正向", "中立", "負向"):
        if col not in ct.columns:
            ct[col] = 0

    ct = ct[["正向", "中立", "負向"]]
    pct = (ct.T / ct.sum(axis=1)).T * 100.0
    pct = pct.fillna(0.0)
    pct["負向"] = (100.0 - (pct["正向"] + pct["中立"])).clip(lower=0.0)
    pct = pct[["正向", "中立", "負向"]]

    pos_rank = pct["正向"].sort_values(ascending=False).index.tolist()
    neg_rank = pct["負向"].sort_values(ascending=False).index.tolist()

    top3_pos = pos_rank[:3]
    top3_neg = [c for c in neg_rank if c not in top3_pos][:3]
    remaining = [c for c in pos_rank if c not in top3_pos + top3_neg]

    ordered = top3_pos + top3_neg + remaining
    pct = pct.loc[ordered]

    ORANGE      = "#FF9900"
    AMAZON_LINK = "#007185"
    AMAZON_NAVY = "#232F3E"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="x", zorder=0, linestyle="--", alpha=0.7)

    pct.plot(
        kind="barh",
        stacked=True,
        color=[ORANGE, AMAZON_LINK, AMAZON_NAVY],
        ax=ax,
        width=0.8,
        zorder=2,
    )

    ax.set_title("各品類情緒評價佔比", fontsize=18, pad=10)
    ax.set_ylabel("產品類別", fontsize=12)
    ax.set_xlabel("情緒佔比 (%)", fontsize=12)
    ax.set_xlim(0, 100)
    ax.margins(x=0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_xticks(np.arange(0, 101, 20))

    ax.legend(
        ["正向", "中立", "負向"],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )

    save_plot(fig, "5_stacked_bar_sentiment_dist.png")
    plt.close(fig)

# =========================
# 圖 6：服務負評數
# =========================
def plot_6_bar_service_issues(data_dict):
    import matplotlib.ticker as mticker
    print("... (6/7) 負面評價數 ...")

    issue_cols = ["cs_issue", "delivery_issue", "return_issue", "fulfillment_issue", "service_any"]
    try:
        use_cols = issue_cols + ["review_year", "review_month", "is_neg"]
        df = pd.read_csv(PATH_CLEAN_CSV, usecols=use_cols, low_memory=False)

        df["date"] = pd.to_datetime(
            df["review_year"].astype(str) + "-" + df["review_month"].astype(str),
            errors="coerce"
        )

        is_neg = (
            (df["is_neg"] == True)
            | (df["is_neg"].astype(str).str.upper().eq("TRUE"))
            | (df["is_neg"].astype(str).eq("1"))
        )
        df = df[(df["service_any"] == 1) & is_neg]

        monthly = df.groupby("date").size().reset_index(name="n")
        monthly["year"] = monthly["date"].dt.year

        LIGHT_GRAY = "#BEBAB7"
        ORANGE     = "#FF9900"

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axisbelow(True)
        ax.grid(axis="y", zorder=0, linestyle="--", alpha=0.7)

        bar_width_days = 25
        bars = ax.bar(
            monthly["date"], monthly["n"],
            color=LIGHT_GRAY, width=bar_width_days, zorder=2
        )

        for yr in (2013, 2014):
            m_in_year = monthly[monthly["year"] == yr]
            if not m_in_year.empty:
                idx = int(m_in_year["n"].idxmax())
                target_dt = monthly.loc[idx, "date"]
                for b, dt in zip(bars, monthly["date"]):
                    if dt == target_dt:
                        b.set_facecolor(ORANGE)
                        break

        idx_max = int(monthly["n"].idxmax())
        idx_second = int(monthly.drop(index=idx_max)["n"].idxmax())

        index_to_bar = {i: b for i, b in zip(monthly.index, bars)}

        def label_bar_at_index(ix: int, text: str, edge_color: str):
            b = index_to_bar[ix]
            x = b.get_x() + b.get_width() / 2.0
            y = b.get_height()
            ax.annotate(
                text, xy=(x, y), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=11, color=edge_color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=edge_color, lw=1.2),
                zorder=5, clip_on=True
            )

        edge_red = ANNOT_RED
        label_bar_at_index(idx_max,    f"{int(monthly.loc[idx_max, 'n'])}", edge_red)
        label_bar_at_index(idx_second, f"{int(monthly.loc[idx_second, 'n'])}", edge_red)

        ax.set_title("負面評價數", fontsize=16, pad=12)
        ax.set_xlabel(data_dict.get("review_date", "評論日期"), fontsize=12)
        ax.set_ylabel("每月服務負評數", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.set_ylim(0, 1100)

        half = pd.to_timedelta(bar_width_days / 2, unit="D")
        ax.set_xlim(monthly["date"].min() - half, monthly["date"].max() + half)
        ax.margins(x=0)

        save_plot(fig, "6_bar_service_issues.png")
        return issue_cols

    except Exception as e:
        print(f"❌ 繪製圖 6 失敗: {e}")
        return None

# =========================
# 圖 7：已驗證 vs 未驗證
# =========================
def plot_7_grouped_bar_issues_by_verified(issue_cols, data_dict):
    if not issue_cols:
        return
    print("... (7/7) 負面評價來源分析 ...")

    issue_cols = [c for c in issue_cols if c != "service_any"]
    try:
        use_cols = issue_cols + ["is_verified", "is_neg"]
        df = pd.read_csv(PATH_CLEAN_CSV, usecols=use_cols, low_memory=False)

        df["is_neg"] = (
            (df["is_neg"] == True)
            | (df["is_neg"].astype(str).str.upper().eq("TRUE"))
            | (df["is_neg"].astype(str).eq("1"))
        )
        df = df[df["is_neg"] == True]

        agg = df.groupby("is_verified")[issue_cols].sum()
        df_plot = (
            agg.T.reset_index()
               .melt(id_vars="index", var_name="is_verified", value_name="count")
        )

        verified_label   = data_dict.get("verified_purchase", "已驗證購買")
        unverified_label = "未驗證購買"
        df_plot["is_verified"] = df_plot["is_verified"].map(
            {True: verified_label, False: unverified_label}
        )
        df_plot["index"] = df_plot["index"].map(lambda x: data_dict.get(x, x))

        ORANGE     = "#FF9900"
        LIGHT_GRAY = "#BEBAB7"
        palette    = {verified_label: ORANGE, unverified_label: LIGHT_GRAY}

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_axisbelow(True)
        ax.grid(axis="y", zorder=0, linestyle="--", alpha=0.7)

        sns.barplot(
            data=df_plot, x="index", y="count", hue="is_verified",
            hue_order=[verified_label, unverified_label],
            palette=palette, ax=ax, zorder=2,
            saturation=1, edgecolor="none"
        )

        ax.set_title("負面評價來源分析", fontsize=16, pad=12)
        ax.set_xlabel("問題類別", fontsize=12)
        ax.set_ylabel("提及次數", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.tick_params(axis="x", rotation=0)

        if ax.get_legend() is not None:
            ax.get_legend().remove()
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=ORANGE, edgecolor="none"),
            Patch(facecolor=LIGHT_GRAY, edgecolor="none"),
        ]
        labels = ["橘色：已驗證購買", "淺灰色：未驗證購買"]
        ax.legend(handles, labels, loc="upper right",
                  frameon=True, facecolor="white", edgecolor="gray", framealpha=0.9)

        save_plot(fig, "7_grouped_bar_issues_by_verified.png")
    except Exception as e:
        print(f"❌ 繪製圖 7 失敗: {e}")

# =========================
# 圖 8：服務負評趨勢（雙軸）
# =========================
def plot_8_service_neg_trends_combo(data_dict):
    print("... (8/9) 服務負評趨勢（雙軸） ...")
    try:
        use_cols = ["review_year", "review_month", "is_neg", "service_any"]
        df = pd.read_csv(PATH_CLEAN_CSV, usecols=use_cols, low_memory=False)

        df["date"] = pd.to_datetime(
            df["review_year"].astype(str) + "-" + df["review_month"].astype(str),
            errors="coerce"
        )

        df["is_neg"] = robust_bool_converter(df["is_neg"])
        df["service_any"] = df["service_any"].fillna(0).astype(int)

        neg_monthly = (
            df.groupby("date")["is_neg"]
              .sum()
              .rename("neg_n")
              .reset_index()
        )

        service_neg_monthly = (
            df.assign(service_neg=(df["is_neg"] & (df["service_any"] == 1)))
              .groupby("date")["service_neg"]
              .sum()
              .rename("service_neg_n")
              .reset_index()
        )

        m = pd.merge(neg_monthly, service_neg_monthly, on="date", how="outer").fillna(0)
        m = m.sort_values("date")
        m["rate"] = np.where(m["neg_n"] > 0, m["service_neg_n"] / m["neg_n"] * 100.0, 0.0)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_axisbelow(True)
        ax1.grid(axis="y", zorder=0, linestyle="--", alpha=0.7)

        bar_w_days = 25
        ax1.bar(m["date"], m["service_neg_n"], color=AMAZON_ORANGE, width=bar_w_days, zorder=2)
        ax1.set_xlabel(data_dict.get("review_date", "評論日期"), fontsize=12)
        ax1.set_ylabel("每月服務相關負評數", fontsize=12, color=AMAZON_GRAY)
        ax1.tick_params(axis="y", labelcolor=AMAZON_GRAY)
        ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

        half = pd.to_timedelta(bar_w_days/2, unit="D")
        ax1.set_xlim(m["date"].min() - half, m["date"].max() + half)
        ax1.margins(x=0)

        annotate_min_max_bars(ax1)

        ax2 = ax1.twinx()
        ax2.grid(False)
        ax2.plot(m["date"], m["rate"], color=AMAZON_LINK, linewidth=2.0,
                 zorder=3, label="服務負評占全部負評 (%)")
        ax2.set_ylabel("服務負評占全部負評 (%)", fontsize=12, color=AMAZON_GRAY)
        ax2.tick_params(axis="y", labelcolor=AMAZON_GRAY)
        ymax = float(np.nanmax(m["rate"])) if len(m) else 0.0
        ax2.set_ylim(0, max(5.0, ymax * 1.2))

        annotate_extrema_line_by_color(ax2, m["date"], m["rate"],
                                       AMAZON_LINK, lambda v: f"{v:.1f}%", dy=14)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left",
                   frameon=True, facecolor="white", edgecolor="gray", framealpha=0.9)

        plt.title("服務負評趨勢（數量 + 占比）", fontsize=16, pad=12)
        save_plot(fig, "10_service_neg_trends_combo.png")
    except Exception as e:
        print(f"❌ 圖 8 失敗: {e}")

# =========================
# 圖 9：服務痛點關鍵字排行
# =========================
def plot_9_service_specific_keyword_ranking(data_dict):
    print("... (9/9) Top 15 服務痛點關鍵字排行（合併最少五項→其他） ...")
    try:
        word_counts = Counter()
        for chunk in pd.read_csv(
            PATH_CLEAN_CSV, usecols=["review_body_clean"], chunksize=50_000, low_memory=False
        ):
            text = chunk["review_body_clean"].astype(str)
            for kw in ALL_KW:
                pattern = r"\b" + re.escape(kw) + r"\b"
                word_counts[kw] += int(text.str.contains(pattern, regex=True, na=False).sum())

        if not word_counts:
            print("❌ 找不到任何指定的服務關鍵字。")
            return

        df_all = pd.DataFrame(list(word_counts.items()), columns=["keyword_en", "count"])
        df_all["keyword_zh"] = df_all["keyword_en"].map(KEYWORD_TRANSLATIONS).fillna(df_all["keyword_en"])

        df15 = (
            df_all.groupby("keyword_zh", as_index=False)["count"].sum()
                  .sort_values("count", ascending=False)
                  .head(15)
                  .reset_index(drop=True)
        )

        tail_k = min(5, len(df15))
        tail_sum = df15.tail(tail_k)["count"].sum()
        df_head = df15.head(len(df15) - tail_k).copy()
        if tail_k > 0:
            df_head = pd.concat(
                [df_head, pd.DataFrame([{"keyword_zh": "其他", "count": tail_sum}])],
                ignore_index=True
            )
        df_plot = df_head.sort_values("count", ascending=False).reset_index(drop=True)

        ORANGE, LIGHT_GRAY = "#FF9900", "#BEBAB7"
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_axisbelow(True)
        ax.grid(axis="x", zorder=0, linestyle="--", alpha=0.7)

        sns.barplot(
            x="count", y="keyword_zh",
            data=df_plot, ax=ax,
            color=LIGHT_GRAY, orient="h", zorder=2
        )

        yticks = ax.get_yticks()
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        highlights = ("支援", "破損", "包裝")

        for p in ax.patches:
            cy = p.get_y() + p.get_height() / 2
            yi = int(np.argmin(np.abs(np.array(yticks) - cy)))
            label = ylabels[yi] if 0 <= yi < len(ylabels) else ""
            if any(h in label for h in highlights):
                p.set_facecolor(ORANGE)
            else:
                p.set_facecolor(LIGHT_GRAY)

        patches = [p for p in ax.patches if isinstance(p, Rectangle)]
        if patches:
            widths = np.array([p.get_width() for p in patches])
            xmax = 30_000
            pad = max(xmax * 0.01, 6)

            def fmt_k_spaced(v: float) -> str:
                return f"{int(round(float(v) / 1000.0))} k"

            def badge(p, text):
                x = p.get_x() + p.get_width()
                y = p.get_y() + p.get_height() / 2
                ax.text(
                    x + pad, y, text,
                    ha="left", va="center", fontsize=11, color=ANNOT_RED,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ANNOT_RED, lw=1.2),
                    zorder=5, clip_on=False
                )

            i_max = int(np.argmax(widths))
            badge(patches[i_max], fmt_k_spaced(widths[i_max]))

            label_to_patch = {}
            for p in ax.patches:
                cy = p.get_y() + p.get_height() / 2
                yi = int(np.argmin(np.abs(np.array(yticks) - cy)))
                if 0 <= yi < len(ylabels):
                    label_to_patch[ylabels[yi]] = p

            for target in ("破損", "包裝"):
                if target in label_to_patch:
                    p = label_to_patch[target]
                    badge(p, fmt_k_spaced(p.get_width()))

        ax.set_title("Top 15 服務痛點關鍵字排行", fontsize=16, pad=12)
        ax.set_xlabel("送出次數", fontsize=12)
        ax.set_ylabel("服務痛點關鍵字", fontsize=12)
        ax.set_xlim(0, 30_000)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: fmt_k(x)))

        save_plot(fig, "9_service_specific_keyword_ranking.png")
    except MemoryError:
        print("❌ 記憶體不足！")
    except Exception as e:
        print(f"❌ 圖 9 失敗: {e}")

# =========================
# 主程式
# =========================
def main():
    print("===== 開始執行 generate_all_plots.py =====")
    data = load_data()
    data_dict = data["dict"]

    plot_1_2_combined_volume_and_star(data["agg_all"], data["agg_ver"], data_dict)
    plot_3_heatmap_neg_rate(data["agg_all"], data_dict)

    print("\n--- 開始處理大型檔案 (reviews_clean.csv) ---")
    try:
        top_cats_en = plot_4_bar_top_categories(data_dict)
        plot_5_stacked_bar_sentiment_dist(top_cats_en, data_dict)
        issue_cols = plot_6_bar_service_issues(data_dict)
        plot_7_grouped_bar_issues_by_verified(issue_cols, data_dict)
    except Exception as e:
        print(f"❌ 處理大型檔案時發生未預期錯誤: {e}")

    try:
        # 若要輸出 10_service_neg_trends_combo.png，解除下一行註解
        # plot_8_service_neg_trends_combo(data_dict)
        plot_9_service_specific_keyword_ranking(data_dict)
    except Exception as e:
        print(f"❌ 服務深掘繪圖失敗: {e}")

    print("==========================================")
    print(f"✅✅✅ 完成：圖片輸出至 {DATA_FIGURES.resolve()}")
    print("==========================================")

if __name__ == "__main__":
    main()
