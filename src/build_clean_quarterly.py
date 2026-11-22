"""
从原始未差分季度 Excel（../季度数据/*.xlsx）重建干净的 31 行业单季面板数据与关联矩阵目录。

背景（D0.1 审计发现）：
- 上级目录 `季度数据/*.xlsx` 中，29 个非金融行业本身已是米筐导出的**单季值**（dup_ratio=1.00，Q2/Q1 中位数 0.93~1.32），
  此前被错误地再次执行跨季差分（Q2-Q1, Q3-Q2, Q4-Q3），导致 32.5% 样本变为负营收、4 个行业 2021q4 整窗清空。
- 仅 `银行I`、`非银金融I` 原始表为年初至今累计值（YTD）且每公司每季有 4 条重复行，现目录中其已完成正确去重与单季差分。
- 全部 577 张 `*_stock_matrix.csv`（含煤炭/环保/石油石化/美容护理 2021q4）与原始表企业代码 100% 一致（0 失配），直接保留复用。

用法（在 重建实验方案/ 目录下运行）：
    python src/build_clean_quarterly.py
    python src/build_clean_quarterly.py --raw-xlsx-dir "../../季度数据" --existing-root ".." --out-dir "data/clean"
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

VALID_QUARTERS: List[str] = [
    "2021q1", "2021q2", "2021q3", "2021q4",
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3",
]

# 仅这两个金融行业在原始 Excel 中为 YTD 累计值，沿用现目录已正确去重差分的单季 CSV
KEEP_EXISTING_CSV_INDUSTRIES = {"银行I", "非银金融I"}


def _compute_audit_stats(df_all: pd.DataFrame) -> dict:
    """计算单个行业重建后的核心审计指标。"""
    if df_all.empty:
        return {"rows": 0, "q_cnt": 0, "neg_q1_pct": np.nan, "neg_q2_pct": np.nan, "med_q2_q1": np.nan}
    q1 = df_all[df_all["quarter"].astype(str).str.endswith("q1")]
    q2 = df_all[df_all["quarter"].astype(str).str.endswith("q2")]
    neg_q1 = float((q1["revenue"] < 0).mean() * 100.0) if len(q1) else np.nan
    neg_q2 = float((q2["revenue"] < 0).mean() * 100.0) if len(q2) else np.nan

    tmp = df_all[["order_book_id", "quarter", "revenue"]].drop_duplicates(
        subset=["order_book_id", "quarter"]
    ).copy()
    tmp["yr"] = tmp["quarter"].astype(str).str[:4]
    tmp["q"] = tmp["quarter"].astype(str).str[4:]
    piv = tmp.pivot(index=["order_book_id", "yr"], columns="q", values="revenue")
    if "q1" in piv.columns and "q2" in piv.columns:
        valid = piv[(piv["q1"] > 0) & piv["q2"].notna()]
        med_q2_q1 = float((valid["q2"] / valid["q1"]).median()) if len(valid) else np.nan
    else:
        med_q2_q1 = np.nan

    return {
        "rows": int(len(df_all)),
        "q_cnt": int(df_all["quarter"].nunique()),
        "neg_q1_pct": round(neg_q1, 2) if not np.isnan(neg_q1) else np.nan,
        "neg_q2_pct": round(neg_q2, 2) if not np.isnan(neg_q2) else np.nan,
        "med_q2_q1": round(med_q2_q1, 3) if not np.isnan(med_q2_q1) else np.nan,
    }


def rebuild_clean_dataset(
    raw_xlsx_dir: Path,
    existing_root: Path,
    out_dir: Path,
    clean_out_first: bool = True,
) -> pd.DataFrame:
    if not raw_xlsx_dir.exists():
        raise FileNotFoundError(f"未找到原始 Excel 目录: {raw_xlsx_dir}")
    if not existing_root.exists():
        raise FileNotFoundError(f"未找到现有行业根目录: {existing_root}")

    industries = sorted(
        d.name for d in existing_root.iterdir()
        if d.is_dir() and d.name.endswith("I")
    )
    if not industries:
        raise RuntimeError(f"在 {existing_root} 下未发现任何以 'I' 结尾的行业文件夹")

    if clean_out_first and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for ind in industries:
        src_ind_dir = existing_root / ind
        dst_ind_dir = out_dir / ind
        dst_ind_dir.mkdir(parents=True, exist_ok=True)

        # 1. 复制所有已有的 stock_matrix.csv（经核验 539/539 0 失配）
        mat_copied = 0
        for q in VALID_QUARTERS:
            src_mat = src_ind_dir / f"{q}_{ind}_stock_matrix.csv"
            if src_mat.exists():
                shutil.copy2(src_mat, dst_ind_dir / src_mat.name)
                mat_copied += 1

        # 2. 生成/复制干净的季度财务 CSV
        ind_frames = []
        csv_written = 0

        if ind in KEEP_EXISTING_CSV_INDUSTRIES:
            source_mode = "existing_ytd_diff_csv"
            for q in VALID_QUARTERS:
                src_csv = src_ind_dir / f"{q}_{ind}.csv"
                if src_csv.exists():
                    shutil.copy2(src_csv, dst_ind_dir / src_csv.name)
                    df_q = pd.read_csv(src_csv)
                    ind_frames.append(df_q)
                    csv_written += 1
        else:
            source_mode = "raw_single_quarter_xlsx"
            xlsx_path = raw_xlsx_dir / f"{ind}.xlsx"
            if not xlsx_path.exists():
                raise FileNotFoundError(f"缺少原始行业文件: {xlsx_path}")

            df_raw = pd.read_excel(xlsx_path)
            df_raw["quarter"] = df_raw["quarter"].astype(str).str.strip().str.lower()
            df_raw["order_book_id"] = df_raw["order_book_id"].astype(str).str.strip()

            # 仅保留 2021q1 - 2025q3，若极个别存在重复代码则保留最后一条
            df_raw = df_raw[df_raw["quarter"].isin(VALID_QUARTERS)].copy()
            df_raw = df_raw.drop_duplicates(subset=["order_book_id", "quarter"], keep="last")

            for q in VALID_QUARTERS:
                grp = df_raw[df_raw["quarter"] == q].copy()
                if grp.empty:
                    continue

                # 数值列标准化
                for col in ["n_income", "revenue", "operating_expense", "total_assets", "total_liab"]:
                    grp[col] = pd.to_numeric(grp[col], errors="coerce")

                # 与对应季度 stock_matrix.csv 的行顺序对齐（如果存在矩阵）
                mat_path = dst_ind_dir / f"{q}_{ind}_stock_matrix.csv"
                if mat_path.exists():
                    mat_codes = pd.read_csv(mat_path, index_col=0, usecols=[0]).index.astype(str).tolist()
                    order_map = {c: i for i, c in enumerate(mat_codes)}
                    grp["_ord"] = grp["order_book_id"].map(order_map).fillna(10**9)
                    grp = grp.sort_values(["_ord", "order_book_id"]).drop(columns=["_ord"])
                else:
                    grp = grp.sort_values("order_book_id")

                # 兼容字段 tao, k（仅作参考列，模拟引擎仍按原始四列现场计算）
                rev = grp["revenue"]
                grp["tao"] = np.where(rev > 0, grp["n_income"] / rev, np.nan)
                grp["k"] = np.where(rev > 0, grp["operating_expense"] / rev, np.nan)

                out_cols = [
                    "quarter", "industry", "name", "order_book_id",
                    "n_income", "tao", "revenue", "operating_expense", "k",
                    "total_assets", "total_liab",
                ]
                if "end_date" in grp.columns:
                    out_cols = ["end_date"] + out_cols

                grp_out = grp[out_cols].reset_index(drop=True)
                grp_out.to_csv(dst_ind_dir / f"{q}_{ind}.csv", index=False, encoding="utf-8-sig")
                ind_frames.append(grp_out)
                csv_written += 1

        df_ind_all = pd.concat(ind_frames, ignore_index=True) if ind_frames else pd.DataFrame()
        stats = _compute_audit_stats(df_ind_all)
        summary_rows.append({
            "industry": ind,
            "source_mode": source_mode,
            "csv_written": csv_written,
            "mat_copied": mat_copied,
            **stats,
        })
        print(
            f"[OK] {ind:6s} | mode={source_mode:23s} | csv={csv_written:2d} mat={mat_copied:2d} "
            f"| rows={stats['rows']:4d} | negQ1={stats['neg_q1_pct']:4.1f}% negQ2={stats['neg_q2_pct']:4.1f}% "
            f"| med(Q2/Q1)={stats['med_q2_q1']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "build_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n完成！共处理 {len(summary_df)} 个行业，汇总表已保存至: {summary_path}")
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="从原始未差分 Excel 重建 31 行业干净单季面板数据")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--raw-xlsx-dir",
        type=Path,
        default=project_root.parent.parent / "季度数据",
        help="原始未差分 Excel 目录（默认 ../../季度数据）",
    )
    parser.add_argument(
        "--existing-root",
        type=Path,
        default=project_root.parent,
        help="现有 31 行业文件夹所在目录（用于复用 stock_matrix.csv 与银行/非银 CSV，默认 ..）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "data" / "clean",
        help="输出干净数据目录（默认 data/clean）",
    )
    args = parser.parse_args()

    rebuild_clean_dataset(
        raw_xlsx_dir=args.raw_xlsx_dir.resolve(),
        existing_root=args.existing_root.resolve(),
        out_dir=args.out_dir.resolve(),
    )


if __name__ == "__main__":
    main()
