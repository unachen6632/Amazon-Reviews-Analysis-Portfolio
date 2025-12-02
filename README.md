# 專題：2010–2015 Amazon 以顧客評價資料提供降低負面評價之關鍵策略
*數據驅動的服務痛點挖掘與決策建議（2010-2015 US Market）*

## 專案目標與商業價值
本專案的目標是透過巨量顧客評論數據（**約 693 萬筆**），建立一套自動化的分析流程，**明確識別導致客戶負面評價的服務與產品關鍵痛點**，並提供可執行、數據驗證的改善策略，以提升客戶滿意度與品牌忠誠度。

---

## 資料來源與處理 (Data Source)
**由於原始檔案過大（Gigabytes 級別），本儲存庫未包含完整數據。**

* **資料集名稱：** Amazon US Customer Reviews Dataset
* **資料集連結：** https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?resource=download
* **處理重點：** 處理大型 `amazon_reviews_multilingual_US_v1_00.tsv` 檔案。

---

## 技術流程與程式碼亮點
| 階段 | 腳本 (Python) | 技術內容與成果 |
| :--- | :--- | :--- | :--- |
| **I. 資料概覽** | `read_data.py` | 讀取大型 `.tsv` 檔案，快速檢查欄位與資料總筆數。 |
| **II. ETL 核心** | `1_clean_data_4.py` | **ETL Pipeline 實作**：數據清洗、資料過濾（2010-2015/US Market）、執行 **NLP 情緒分類（Sentiment Labeling）**。 |
| **III. 數據驗證** | `verify_alldata_2.py` | 專案資料品質保證（Data Quality Assurance）模組，確保 ETL 輸出檔案的結構與筆數正確無誤。 | 
| **IV. 洞察報告** | `picture_test_n_all_5.py` | 產出分析圖表，如：每月評價量/平均星等趨勢、**服務痛點關鍵字排名**。 |

## 關鍵策略洞察（基於分析結果）
依據專案分析結果，我們鎖定以下負評關鍵字排行前三名資訊，提出對策建議：

1.  **技術支援**：技術支援自助化，可以設置FAQ / 聊天機器人。
2.  **破損**：為了降低破損率，可以風險評估後設計不同包材。
3.  **包裝**：為了包裝保護提升，商品包裝作業流程化。

---

## 執行環境與使用方式
* **語言/環境**: Python 3.8+
* **核心套件**: `pandas`, `numpy`, `matplotlib`, `seaborn`
* **如何執行**:
    1.  下載本儲存庫代碼。
    2.  從上方 [Kaggle Dataset Link] 下載完整數據，並命名為 `amazon_reviews_multilingual_US_v1_00.tsv` 放入專案根目錄。
    3.  依序執行 “read_data.py > 1_clean_data_4.py > verify_alldata_2.py > picture_test_n_all_5.py” 的 Python 腳本。
