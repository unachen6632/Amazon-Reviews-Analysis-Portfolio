#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize the Amazon reviews TSV:
- Column count & names
- Total row count (excluding header)
Usage:
    python summarize_amazon_tsv.py [PATH]
Default PATH = ./amazon_reviews_multilingual_US_v1_00.tsv
"""

import os
import sys
import pandas as pd

def summarize_tsv(path: str, chunksize: int = 1_000_000):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # 先讀少量資料以取得欄位名稱（保守處理壞行）
    head = pd.read_csv(
        path,
        sep="\t",
        nrows=5,
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )
    columns = list(head.columns)
    col_count = len(columns)

    # 計算資料筆數（不載入全部欄位，節省記憶體）
    total_rows = 0
    try:
        if path.endswith(".tsv"):  # 非壓縮檔：用行數最快
            with open(path, "rb") as f:
                for i, _ in enumerate(f, 1):
                    pass
            total_rows = max(0, i - 1)  # 扣掉表頭
        else:
            # 壓縮檔（.tsv.gz 等）：以最少欄位分塊累加
            first_col = columns[0] if columns else None
            for chunk in pd.read_csv(
                path,
                sep="\t",
                usecols=[first_col] if first_col else None,
                dtype=str,
                chunksize=chunksize,
                low_memory=True,
                on_bad_lines="skip",
            ):
                total_rows += len(chunk)
    except Exception:
        # 若上面方法失敗，改用分塊讀取全部欄位當後備方案
        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=chunksize,
            low_memory=True,
            on_bad_lines="skip",
        ):
            total_rows += len(chunk)

    return columns, col_count, total_rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "amazon_reviews_multilingual_US_v1_00.tsv"
    cols, n_cols, n_rows = summarize_tsv(path)

    print("檔案：", os.path.abspath(path))
    print("欄位數：", n_cols)
    print("欄位名稱：")
    for i, c in enumerate(cols, 1):
        print(f"  {i:>2}. {c}")
    print("資料筆數（不含表頭）：", f"{n_rows:,}")

if __name__ == "__main__":
    main()
