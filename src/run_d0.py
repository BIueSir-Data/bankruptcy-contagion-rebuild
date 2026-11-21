# -*- coding: utf-8 -*-
"""
run_d0.py —— 先行检验 D0：数据口径审计、样本流转表、全窗口覆盖与单季 SRI 核对
========================================================================
用法示例：
  # 1) 快速试跑 5 个行业（推荐先验证）：
  python src/run_d0.py --config configs/d0_config.yaml --inds 交通运输,公用事业,国防军工,石油石化,银行 --workers 4

  # 2) 全量运行全部 31 行业（建议本机 4 或 8 进程）：
  python src/run_d0.py --config configs/d0_config.yaml --workers 8
"""
import os
import sys
import time
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_config, get_git_commit, resolve_paths, discover_industries,
    create_run_dir, save_seeds, read_stock_matrix_safe, compute_sri_tables,
)
from engine_legacy_d1 import run_single_cell

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _sim_worker(task):
    raw_root, ind, q, gamma_q, params, strip_mode = task
    return run_single_cell(
        raw_root, ind, q, gamma_q, params,
        strip_mode=strip_mode, propagation_enabled=True,
    )


def audit_data(raw_root: str, industries: list, quarters: list,
               excluded_inds: set, strip_mode: str, logger):
    """执行 D0.1 - D0.5 数据口径审计与样本流转统计（只读，绝不修改原始数据）。"""
    window_rows = []
    # 记录各步样本流转（企业-季度 & 去重企业集合）
    flow = {
        "step0_raw_all": {"rows": 0, "firms": set()},
        "step1_excl_fin": {"rows": 0, "firms": set()},
        "step1_main_raw": {"rows": 0, "firms": set()},
        "step2_main_clean": {"rows": 0, "firms": set()},
        "step3_main_aligned": {"rows": 0, "firms": set()},
        "step4_main_valid_win": {"rows": 0, "firms": set()},
    }
    cost_zero_main_rows = 0
    cost_zero_main_firms = set()
    no_summary_matrices = 0
    total_matrices = 0

    # 收集利润表季度值（用于 D0.1 单季 vs 累计检验）
    rev_records = []

    for ind in industries:
        is_fin = ind in excluded_inds
        ind_dir = os.path.join(raw_root, f"{ind}I")
        for q in quarters:
            raw_csv = os.path.join(ind_dir, f"{q}_{ind}I.csv")
            mat_csv = os.path.join(ind_dir, f"{q}_{ind}I_stock_matrix.csv")
            has_raw = os.path.exists(raw_csv)
            has_mat = os.path.exists(mat_csv)

            if not has_raw or not has_mat:
                reason = "missing_both" if (not has_raw and not has_mat) else (
                    "missing_financial_csv" if not has_raw else "missing_stock_matrix"
                )
                window_rows.append({
                    "Industry": ind, "Quarter": q, "Sample_Group": "financial_excluded" if is_fin else "main_29",
                    "Status": "NA", "Reason": reason,
                    "N_raw": np.nan, "N_clean": np.nan, "N_aligned": np.nan,
                    "Had_Summary_Row": np.nan, "Had_Summary_Col": np.nan,
                })
                continue

            df = pd.read_csv(raw_csv, encoding="utf-8-sig")
            n_raw = len(df)
            if n_raw == 0:
                window_rows.append({
                    "Industry": ind, "Quarter": q, "Sample_Group": "financial_excluded" if is_fin else "main_29",
                    "Status": "NA", "Reason": "empty_financial_csv",
                    "N_raw": 0, "N_clean": 0, "N_aligned": 0,
                    "Had_Summary_Row": np.nan, "Had_Summary_Col": np.nan,
                })
                continue

            df["order_book_id"] = df["order_book_id"].astype(str).str.strip()
            raw_ids = set(df["order_book_id"].dropna())
            flow["step0_raw_all"]["rows"] += n_raw
            flow["step0_raw_all"]["firms"].update(raw_ids)

            if is_fin:
                flow["step1_excl_fin"]["rows"] += n_raw
                flow["step1_excl_fin"]["firms"].update(raw_ids)
            else:
                flow["step1_main_raw"]["rows"] += n_raw
                flow["step1_main_raw"]["firms"].update(raw_ids)

            rev_col = "operating_revenue" if "operating_revenue" in df.columns else "revenue"
            cost_col = "operating_costs" if "operating_costs" in df.columns else "operating_expense"
            rev = pd.to_numeric(df[rev_col], errors="coerce")
            cost = pd.to_numeric(df[cost_col], errors="coerce")
            ninc = pd.to_numeric(df["n_income"], errors="coerce") if "n_income" in df.columns else pd.Series(np.nan, index=df.index)

            # 记录少量列用于单季口径检验
            sub_rev = pd.DataFrame({
                "order_book_id": df["order_book_id"],
                "name": df["name"] if "name" in df.columns else "",
                "Industry": ind,
                "Quarter": q,
                "revenue": rev,
                "cost": cost,
                "n_income": ninc,
            })
            rev_records.append(sub_rev)

            mask_clean = (rev > 0) & (cost >= 0)
            df_clean = df[mask_clean].copy()
            n_clean = len(df_clean)

            if not is_fin:
                flow["step2_main_clean"]["rows"] += n_clean
                flow["step2_main_clean"]["firms"].update(df_clean["order_book_id"].dropna())
                c0_mask = mask_clean & (cost == 0)
                cost_zero_main_rows += int(c0_mask.sum())
                cost_zero_main_firms.update(df.loc[c0_mask, "order_book_id"].dropna())

            W, had_row, had_col = read_stock_matrix_safe(mat_csv, strip_mode=strip_mode)
            total_matrices += 1
            if not (had_row or had_col):
                no_summary_matrices += 1

            valid_ids = [cid for cid in df_clean["order_book_id"] if cid in W.index and cid in W.columns]
            n_aligned = len(valid_ids)
            if not is_fin:
                flow["step3_main_aligned"]["rows"] += n_aligned
                flow["step3_main_aligned"]["firms"].update(valid_ids)

            if n_clean < 2:
                status, reason = "NA", f"lt2_after_cleaning(n={n_clean})"
            elif n_aligned < 2:
                status, reason = "NA", f"lt2_after_matrix_align(n={n_aligned})"
            else:
                status, reason = "valid", "ok"
                if not is_fin:
                    flow["step4_main_valid_win"]["rows"] += n_aligned
                    flow["step4_main_valid_win"]["firms"].update(valid_ids)

            window_rows.append({
                "Industry": ind, "Quarter": q, "Sample_Group": "financial_excluded" if is_fin else "main_29",
                "Status": status, "Reason": reason,
                "N_raw": n_raw, "N_clean": n_clean, "N_aligned": n_aligned,
                "Had_Summary_Row": int(had_row), "Had_Summary_Col": int(had_col),
            })

    df_win = pd.DataFrame(window_rows)

    # 汇总样本流转表
    flow_table = pd.DataFrame([
        {"Step": "0_原始全部记录(所选行业)", "Firm_Quarters": flow["step0_raw_all"]["rows"],
         "Unique_Firms": len(flow["step0_raw_all"]["firms"]), "Note": f"涵盖 {len(industries)} 个行业 × {len(quarters)} 季度"},
        {"Step": "1a_移出金融行业(银行/非银金融,单独保存)", "Firm_Quarters": flow["step1_excl_fin"]["rows"],
         "Unique_Firms": len(flow["step1_excl_fin"]["firms"]), "Note": "移出主样本，单独存档"},
        {"Step": "1b_主样本原始记录", "Firm_Quarters": flow["step1_main_raw"]["rows"],
         "Unique_Firms": len(flow["step1_main_raw"]["firms"]), "Note": "不含银行、非银金融"},
        {"Step": "2_清洗(营收>0且成本>=0)", "Firm_Quarters": flow["step2_main_clean"]["rows"],
         "Unique_Firms": len(flow["step2_main_clean"]["firms"]),
         "Note": f"其中成本==0占 {cost_zero_main_rows} 条 ({len(cost_zero_main_firms)} 家企业)，后续第5条需严格剔除"},
        {"Step": "3_与stock_matrix节点对齐", "Firm_Quarters": flow["step3_main_aligned"]["rows"],
         "Unique_Firms": len(flow["step3_main_aligned"]["firms"]),
         "Note": f"矩阵裁剪模式={strip_mode}；检查 {total_matrices} 个矩阵中 {no_summary_matrices} 个无'合计'行/列"},
        {"Step": "4_有效窗口最终主样本(窗口节点>=2)", "Firm_Quarters": flow["step4_main_valid_win"]["rows"],
         "Unique_Firms": len(flow["step4_main_valid_win"]["firms"]), "Note": "进入 D0/D1 仿真的最终主样本"},
    ])

    # D0.1 单季口径定量核验
    df_rev = pd.concat(rev_records, ignore_index=True) if rev_records else pd.DataFrame()
    q_check_rows = []
    if not df_rev.empty:
        df_rev["Year"] = df_rev["Quarter"].str[:4]
        df_rev["Qtr"] = df_rev["Quarter"].str[4:]
        piv = df_rev.pivot_table(index=["order_book_id", "Year"], columns="Qtr", values="revenue", aggfunc="first")
        for q_pair in [("q1", "q2"), ("q2", "q3"), ("q3", "q4")]:
            a, b = q_pair
            if a in piv.columns and b in piv.columns:
                sub = piv[[a, b]].dropna()
                sub = sub[(sub[a] > 0) & (sub[b] > 0)]
                ratio = sub[b] / sub[a]
                q_check_rows.append({
                    "Pair": f"{b.upper()}/{a.upper()}",
                    "N_firm_years": len(sub),
                    "Median_Ratio": float(ratio.median()) if len(sub) else np.nan,
                    "Pct_Strictly_Increasing": float((sub[b] > sub[a]).mean()) if len(sub) else np.nan,
                    "Pct_Ratio_gt_1_5": float((ratio > 1.5).mean()) if len(sub) else np.nan,
                    "Interpretation": "若为单季值中位数应接近1.0；若为年初累计值Q2/Q1应接近2.0且递增率>95%",
                })
    df_qcheck = pd.DataFrame(q_check_rows)
    logger.info("D0.1 利润表季度口径核验完成：%d 组相邻季度比对", len(df_qcheck))
    return flow_table, df_win, df_qcheck


def plot_d0_sri(sri_main: pd.DataFrame, sri_all: pd.DataFrame, fig_path: str, caption: str):
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    x = np.arange(len(sri_main.index))
    qs = list(sri_main.index)

    ax.plot(x, sri_main["SRI_raw_19q"].values, marker="o", lw=2.0, label="主样本(不含金融) 19季原值 SRI")
    if sri_main["SRI_ma4_16q"].notna().any():
        ax.plot(x, sri_main["SRI_ma4_16q"].values, marker="s", lw=1.6, ls="--", label="主样本(不含金融) 旧MA4 SRI(附注)")
    if sri_all is not None and not sri_all.empty:
        sri_all_al = sri_all.reindex(qs)
        ax.plot(x, sri_all_al["SRI_raw_19q"].values, marker="^", lw=1.4, alpha=0.75, label="全样本(含金融) 19季原值 SRI")

    for target_q, col in [("2022q4", "#C00000"), ("2023q4", "#B76E00")]:
        if target_q in qs:
            idx = qs.index(target_q)
            ax.axvline(idx, color=col, ls=":", lw=1.3, alpha=0.8)
            ax.text(idx, ax.get_ylim()[1] * 0.96, f" {target_q.upper()}", color=col, fontsize=8.5, va="top")

    ax.set_xticks(x)
    ax.set_xticklabels(qs, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("SRI (原值, 无Z变换)")
    ax.set_title("D0.6 单季口径基准 SRI 走势与 2022Q4 / 2023Q4 核查")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8.5)
    fig.text(0.01, 0.01, caption, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="执行先行检验 D0（数据口径审计 + 单季基准 SRI 核对）")
    ap.add_argument("--config", default=os.path.join("configs", "d0_config.yaml"))
    ap.add_argument("--inds", default=None, help="可选：逗号分隔行业名试跑，如 交通运输,公用事业,银行（默认全部31行业）")
    ap.add_argument("--workers", type=int, default=None, help="并行进程数（覆盖配置文件）")
    ap.add_argument("--skip-sim", action="store_true", help="仅做 D0.1-D0.5 纯数据审计，跳过 D0.6 基准仿真")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proj_root, raw_root, out_root = resolve_paths(cfg, args.config)
    run_dir, logger = create_run_dir(out_root, cfg.get("experiment_name", "D0_data_audit"), args.config)
    commit = get_git_commit(proj_root)

    all_inds = discover_industries(raw_root)
    if args.inds:
        req = [x.strip().rstrip("I") for x in args.inds.split(",") if x.strip()]
        industries = [x for x in req if x in all_inds]
    else:
        industries = all_inds

    quarters = list(cfg["quarters"])
    excluded_inds = set(cfg.get("excluded_industries", ["银行", "非银金融"]))
    strip_mode = cfg.get("strip_last_row_mode", "auto_by_name")
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 4))

    logger.info("=== 开始运行 D0 | commit=%s | config=%s ===", commit, os.path.basename(args.config))
    logger.info("行业数=%d | 季度数=%d | 移出金融=%s | strip_mode=%s | workers=%d",
                len(industries), len(quarters), sorted(excluded_inds), strip_mode, workers)

    t0 = time.time()
    flow_table, df_win, df_qcheck = audit_data(raw_root, industries, quarters, excluded_inds, strip_mode, logger)

    clean_dir = os.path.join(out_root, "data", "clean")
    os.makedirs(clean_dir, exist_ok=True)
    for dst_dir in (run_dir, clean_dir):
        flow_table.to_csv(os.path.join(dst_dir, "d0_sample_flow.csv"), index=False, encoding="utf-8-sig", na_rep="NA")
        df_win.to_csv(os.path.join(dst_dir, "d0_window_matrix.csv"), index=False, encoding="utf-8-sig", na_rep="NA")
        piv_win = df_win.pivot(index="Industry", columns="Quarter", values="Status")
        piv_win.to_csv(os.path.join(dst_dir, "d0_window_pivot.csv"), encoding="utf-8-sig", na_rep="NA")
        df_qcheck.to_csv(os.path.join(dst_dir, "d0_quarterly_check.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

    n_valid = int((df_win["Status"] == "valid").sum())
    n_na = int((df_win["Status"] != "valid").sum())
    logger.info("D0.1-D0.5 完成 (%.1fs) | 有效窗口=%d | 缺失窗口(NA)=%d", time.time() - t0, n_valid, n_na)
    logger.info("\n%s", flow_table.to_string(index=False))

    if args.skip_sim:
        logger.info("已指定 --skip-sim，跳过 D0.6 基准仿真。")
        return

    # D0.6 单季口径基准复跑（沿用 11.16 现有 8.31 参数）
    params = dict(cfg["baseline_sim"])
    gamma_q = float(params["gamma_q"])
    tasks = [(raw_root, ind, q, gamma_q, params, strip_mode) for ind in industries for q in quarters]
    logger.info("开始 D0.6 单季口径基准仿真：%d 格 × workers=%d", len(tasks), workers)

    t1 = time.time()
    rows = []
    seed_records = []
    if workers > 1 and len(tasks) > 4:
        with Pool(workers) as pool:
            for (ind, q), (res, seed, reason) in zip([(t[1], t[2]) for t in tasks], pool.map(_sim_worker, tasks)):
                seed_records.append({"Industry": ind, "Quarter": q, "gamma": gamma_q, "seed": seed, "status": reason})
                if res is not None:
                    rows.append(res)
    else:
        for t in tasks:
            res, seed, reason = _sim_worker(t)
            seed_records.append({"Industry": t[1], "Quarter": t[2], "gamma": gamma_q, "seed": seed, "status": reason})
            if res is not None:
                rows.append(res)

    save_seeds(run_dir, seed_records)
    df_sim = pd.DataFrame(rows)
    df_sim.to_csv(os.path.join(run_dir, "d0_baseline_panel.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

    df_main = df_sim[~df_sim["Industry"].isin(excluded_inds)].copy()
    df_fin = df_sim[df_sim["Industry"].isin(excluded_inds)].copy()
    df_main.to_csv(os.path.join(run_dir, "d0_baseline_panel_main.csv"), index=False, encoding="utf-8-sig", na_rep="NA")
    if not df_fin.empty:
        df_fin.to_csv(os.path.join(run_dir, "d0_baseline_panel_financial.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

    sri_main = compute_sri_tables(df_main) if not df_main.empty else pd.DataFrame()
    sri_all = compute_sri_tables(df_sim) if not df_sim.empty else pd.DataFrame()
    sri_comp = sri_main.add_prefix("Main_").join(sri_all.add_prefix("All_"), how="outer")
    sri_comp.to_csv(os.path.join(run_dir, "d0_sri_comparison.csv"), encoding="utf-8-sig", na_rep="NA")

    if not sri_main.empty:
        fig_dir = os.path.join(out_root, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        caption = f"Config: {os.path.basename(args.config)} | commit: {commit} | gamma={gamma_q} | inds={len(industries)} | strip={strip_mode}"
        fig_path = os.path.join(fig_dir, "D0_SRI_comparison.png")
        plot_d0_sri(sri_main, sri_all if not df_fin.empty else None, fig_path, caption)
        plot_d0_sri(sri_main, sri_all if not df_fin.empty else None, os.path.join(run_dir, "D0_SRI_comparison.png"), caption)

    logger.info("D0.6 仿真完成 (%.1fs) | 产出 %d 行 | 结果目录: %s", time.time() - t1, len(df_sim), run_dir)
    if not sri_comp.empty:
        for q_chk in ["2022q3", "2022q4", "2023q3", "2023q4"]:
            if q_chk in sri_comp.index:
                logger.info("SRI 核查 %s: %s", q_chk, sri_comp.loc[q_chk].round(6).to_dict())


if __name__ == "__main__":
    main()
