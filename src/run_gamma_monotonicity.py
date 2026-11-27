# -*- coding: utf-8 -*-
"""
run_gamma_monotonicity.py —— 修订第 11 条：γ -> θ_k -> BR0 单调性与负 θ_k 核查
================================================================================
无需运行网络传播仿真，直接对全量 29 主样本行业 × 19 季度计算多组高危校准分位 γ 下的
θ_k 与初始破产率 BR0，检查：
  1) BR0 随 γ 增大是否单调非降；
  2) θ_k 为负的窗口数量与占比；
  3) 各行业均值对照表。

用法：
    python src/run_gamma_monotonicity.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import discover_industries, read_stock_matrix_safe
from engine_legacy_d1 import _to_num, _z_score, fit_huber_theta

QUARTERS: List[str] = [
    "2021q1", "2021q2", "2021q3", "2021q4",
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3",
]
GAMMA_LIST: List[float] = [0.05, 0.10, 0.15, 0.25, 0.35, 0.45]
EXCLUDED = {"银行", "非银金融"}


def main() -> None:
    proj_root = Path(__file__).resolve().parent.parent
    raw_root = str(proj_root / "data" / "clean")
    out_dir = proj_root / "版本归档" / "V2_辅助诊断_gamma单调性检验"
    out_dir.mkdir(parents=True, exist_ok=True)

    inds = [x for x in discover_industries(raw_root) if x not in EXCLUDED]
    rows = []

    for ind in inds:
        for q in QUARTERS:
            ind_dir = os.path.join(raw_root, f"{ind}I")
            f_csv = os.path.join(ind_dir, f"{q}_{ind}I.csv")
            f_mat = os.path.join(ind_dir, f"{q}_{ind}I_stock_matrix.csv")
            if not (os.path.exists(f_csv) and os.path.exists(f_mat)):
                continue
            df = pd.read_csv(f_csv, encoding="utf-8-sig")
            if df.empty:
                continue
            for c in ["n_income", "total_assets", "total_liab", "revenue", "operating_expense"]:
                if c in df.columns:
                    df[c] = df[c].apply(_to_num)
            df = df[(df["revenue"] > 0) & (df["operating_expense"] > 0)].copy()
            if len(df) < 2:
                continue
            df["order_book_id"] = df["order_book_id"].astype(str).str.strip()
            W, _, _ = read_stock_matrix_safe(f_mat, strip_mode="auto_by_name")
            valid = [cid for cid in df["order_book_id"] if cid in W.index and cid in W.columns]
            if len(valid) < 2:
                continue
            df = df[df["order_book_id"].isin(valid)].drop_duplicates(subset=["order_book_id"]).copy()

            liab = df["total_liab"].fillna(0.0)
            assets = df["total_assets"].fillna(0.0).replace(0, np.nan)
            income = df["n_income"].fillna(0.0)
            exp = df["operating_expense"].fillna(0.0)
            rv = df["revenue"].replace(0, np.nan)

            df["L"] = liab / assets
            df["k_ratio"] = exp / rv
            df["tau_raw"] = income / rv
            df["IA"] = income / assets
            for c in ["L", "k_ratio", "tau_raw", "IA"]:
                s = df[c].fillna(df[c].median()).fillna(0.0)
                lo, hi = s.quantile(0.01), s.quantile(0.99)
                df["z_" + c] = _z_score(s.clip(lo, hi))
            df["VS_score"] = df["z_L"] + df["z_k_ratio"] - df["z_tau_raw"] - df["z_IA"]
            df["survival_value"] = income
            is_insolvent = (df["total_assets"].fillna(0.0) - liab) <= 0

            for g in GAMMA_LIST:
                theta = fit_huber_theta(df, g)
                br0_gamma = float((liab * theta > income).mean())
                br0_total = float(((liab * theta > income) | is_insolvent).mean())
                rows.append({
                    "Industry": ind,
                    "Quarter": q,
                    "gamma": g,
                    "theta": theta,
                    "theta_is_neg": int(theta < 0),
                    "BR0_gamma_only": br0_gamma,
                    "BR0_with_insolvent": br0_total,
                    "N_firms": len(df),
                })

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_dir / "gamma_theta_br0_full.csv", index=False, encoding="utf-8-sig")

    # 汇总各 γ 的均值与单调性检查
    summary = res_df.groupby("gamma").agg(
        Valid_Windows=("theta", "count"),
        Theta_mean=("theta", "mean"),
        Theta_median=("theta", "median"),
        Theta_neg_cnt=("theta_is_neg", "sum"),
        Theta_neg_pct=("theta_is_neg", lambda s: round(float(s.mean()) * 100, 2)),
        BR0_gamma_mean=("BR0_gamma_only", "mean"),
        BR0_total_mean=("BR0_with_insolvent", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "gamma_summary.csv", index=False, encoding="utf-8-sig")

    # 逐窗口单调性核验（检查每个 (Industry, Quarter) 内，BR0 与 theta 是否随 γ 单调非降）
    piv_br = res_df.pivot(index=["Industry", "Quarter"], columns="gamma", values="BR0_total_mean" if "BR0_total_mean" in res_df else "BR0_with_insolvent")
    piv_th = res_df.pivot(index=["Industry", "Quarter"], columns="gamma", values="theta")
    br_diffs = piv_br.diff(axis=1).iloc[:, 1:]
    th_diffs = piv_th.diff(axis=1).iloc[:, 1:]
    n_nonmono_br = int((br_diffs < -1e-12).any(axis=1).sum())
    n_nonmono_th = int((th_diffs < -1e-12).any(axis=1).sum())

    print("=== γ -> θ_k -> BR0 全量汇总（29 主样本行业 × 19 季）===")
    print(summary.to_string(index=False))
    print(f"\n窗口级单调性核验（共 {len(piv_br)} 个有效窗口）：")
    print(f"  BR0 随 γ 非单调窗口数 = {n_nonmono_br} ({n_nonmono_br / len(piv_br) * 100:.2f}%)")
    print(f"  θ_k 随 γ 非单调窗口数 = {n_nonmono_th} ({n_nonmono_th / len(piv_th) * 100:.2f}%)")

    # 生成 README.md
    readme = f"""# V2 辅助诊断 — 第 11 条 `γ → θ_k → BR0` 单调性与负 `θ_k` 核查

- **复现命令**：`python src/run_gamma_monotonicity.py`
- **样本范围**：29 个非金融主样本行业 × 19 季度 = {len(piv_br)} 个有效窗口（严格剔除 `rev <= 0` 或 `cost <= 0`）
- **单调性核验**：
  - `BR0` 随 `γ` 增大非单调窗口数：**{n_nonmono_br} / {len(piv_br)}**（{n_nonmono_br / len(piv_br) * 100:.2f}%）
  - `θ_k` 随 `γ` 增大非单调窗口数：**{n_nonmono_th} / {len(piv_th)}**（{n_nonmono_th / len(piv_th) * 100:.2f}%，出现负斜率即 Huber 拟合中低 VS 企业反而利润/负债比更低的反常窗口）

## 各 `γ` 分位汇总表

| `γ` | 有效窗 | `θ_k` 均值 | `θ_k` 中位数 | `θ_k < 0` 窗口数 | `θ_k < 0` 占比(%) | `BR0`（仅 γ 规则） | `BR0`（含初始资不抵债） |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in summary.iterrows():
        readme += (
            f"| {r['gamma']:.2f} | {int(r['Valid_Windows'])} | {r['Theta_mean']:.5f} | {r['Theta_median']:.5f} | "
            f"{int(r['Theta_neg_cnt'])} | {r['Theta_neg_pct']:.2f}% | {r['BR0_gamma_mean']:.4f} | {r['BR0_total_mean']:.4f} |\n"
        )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"\n[OK] 结果与 README.md 已保存至: {out_dir}")


if __name__ == "__main__":
    main()
