# -*- coding: utf-8 -*-
"""
run_v2.py —— 渐进式冻结模型统一运行入口（V2.0 -> V2.1 -> V2.2）
================================================================================
所有版本共用同一套主程序，通过切换 YAML 配置实现渐进累积升级：
  - V2.0（会计与社会力核心）:     python src/run_v2.py --config configs/v2_0_accounting_core.yaml --workers 8
  - V2.1（覆盖 V2.0 + 动力学同步）: python src/run_v2.py --config configs/v2_1_dynamics_sync.yaml --workers 8
  - V2.2（覆盖 V2.1 + 50/200 重复）: python src/run_v2.py --config configs/v2_2_mc_convergence.yaml --workers 8

每次运行自动产出：
  - panel_all.csv / panel_main.csv / panel_financial.csv
  - industry_summary.csv / gamma_theta_br0_diag.csv / seeds.json / run.log
  - sri_index.csv + figs_ma4/ (5张论文标准图) + figs_raw_19q/ (5张单季19Q标准图)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    create_run_dir,
    discover_industries,
    get_git_commit,
    load_config,
    resolve_paths,
    save_seeds,
)
from engine_v2 import run_single_cell_v2
from plotting import generate_standard_5figs


def _worker(task):
    raw_root, ind, q, gamma_q, params, strip_mode = task
    return run_single_cell_v2(
        raw_root, ind, q, gamma_q, params, strip_mode=strip_mode, propagation_enabled=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="渐进式冻结模型统一运行脚本 (V2.0 / V2.1 / V2.2)")
    parser.add_argument("--config", required=True, help="版本配置文件路径（如 configs/v2_0_accounting_core.yaml）")
    parser.add_argument("--inds", default=None, help="可选：逗号分隔行业名用于快速冒烟测试（如 煤炭,食品饮料,美容护理）")
    parser.add_argument("--workers", type=int, default=None, help="覆盖配置中的并行进程数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    proj_root, raw_root, out_root = resolve_paths(cfg, args.config)
    exp_name = cfg.get("experiment_name", "V2_model")
    run_dir, logger = create_run_dir(out_root, exp_name, args.config)
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
    params = dict(cfg["model_params"])
    gamma_q = float(params["gamma_q"])
    version_tag = cfg.get("version_tag", exp_name)

    logger.info("=== 开始运行 %s | commit=%s | config=%s ===", version_tag, commit, os.path.basename(args.config))
    logger.info(
        "行业数=%d | 季度数=%d | gamma=%.2f | buffer=%s | k=%s | margin=%s | ego=%s | sec=%s | stop=%s | R=%d | workers=%d",
        len(industries), len(quarters), gamma_q,
        params.get("buffer_mode"), params.get("k_mode"), params.get("margin_mode"),
        params.get("ego_mode"), params.get("secondary_impact_mode"), params.get("stop_mode"),
        int(params.get("n_repeats", 1)), workers,
    )

    tasks = [(raw_root, ind, q, gamma_q, params, strip_mode) for ind in industries for q in quarters]
    t0 = time.time()
    rows = []
    seed_records = []
    na_rows = []

    if workers > 1 and len(tasks) > 4:
        with Pool(workers) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = [_worker(t) for t in tasks]

    for (raw_root_, ind, q, _, _, _), (res, seeds, reason) in zip(tasks, results):
        seed_records.append({
            "Industry": ind,
            "Quarter": q,
            "gamma": gamma_q,
            "n_seeds": len(seeds),
            "seed_first": seeds[0] if seeds else None,
            "status": reason,
        })
        if res is not None:
            rows.append(res)
        else:
            na_rows.append({"Industry": ind, "Quarter": q, "gamma": gamma_q, "Status": "NA", "Reason": reason})

    save_seeds(run_dir, seed_records)

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(run_dir, "panel_all.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

    df_main = df_all[~df_all["Industry"].isin(excluded_inds)].copy()
    df_fin = df_all[df_all["Industry"].isin(excluded_inds)].copy()
    df_main.to_csv(os.path.join(run_dir, "panel_main.csv"), index=False, encoding="utf-8-sig", na_rep="NA")
    if not df_fin.empty:
        df_fin.to_csv(os.path.join(run_dir, "panel_financial.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

    if na_rows:
        pd.DataFrame(na_rows).to_csv(
            os.path.join(run_dir, "missing_windows.csv"), index=False, encoding="utf-8-sig", na_rep="NA"
        )

    # 第 11 条：输出 γ -> θ_k -> BR0 诊断表（检查 θ_k 为负的窗口）
    if not df_main.empty:
        diag_cols = [
            "Industry", "Quarter", "gamma", "theta", "BR0", "BRT", "dBR",
            "N_firms", "N_init_gamma", "N_init_insolvent", "N_tau_clipped",
            "T_last_activation", "T_loss_stop", "T_run", "All_Converged",
        ]
        df_main[diag_cols].to_csv(
            os.path.join(run_dir, "gamma_theta_br0_diag.csv"), index=False, encoding="utf-8-sig", na_rep="NA"
        )
        n_neg_theta = int((df_main["theta"] < 0).sum())
        n_unconv = int((df_main["All_Converged"] == 0).sum())

        ind_sum = df_main.groupby("Industry").agg(
            Valid_Quarters=("Quarter", "count"),
            BR0_mean=("BR0", "mean"),
            BRT_mean=("BRT", "mean"),
            dBR_mean=("dBR", "mean"),
            Loss_Ratio_mean=("Loss_Ratio", "mean"),
            Max_Prop_Gen_mean=("Max_Prop_Gen", "mean"),
            Max_Prop_Gen_max=("Max_Prop_Gen_max", "max"),
            WCC_mean=("WCC", "mean"),
            Leverage_mean=("Leverage", "mean"),
            Tau_Clipped_total=("N_tau_clipped", "sum"),
        ).reset_index()
        ind_sum.to_csv(os.path.join(run_dir, "industry_summary.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

        # 自动输出论文标准五图（MA4 与单季 19Q 两套）+ sri_index.csv
        sri_comb, sri_raw, sri_ma4 = generate_standard_5figs(
            df_main, run_dir, note_prefix=f"{version_tag}, gamma={gamma_q}"
        )

        logger.info(
            "=== %s 完成 (%.1fs) | 主样本有效窗=%d, 缺失窗=%d | θ<0窗口=%d | 未收敛窗=%d ===",
            version_tag, time.time() - t0, len(df_main), len(na_rows), n_neg_theta, n_unconv,
        )
        logger.info(
            "主样本均值 | BR0=%.4f | BRT=%.4f | dBR=%.4f | Loss_Ratio=%.4f | MaxGen_mean=%.2f (max=%d) | SRI_raw均值=%.4f | SRI_ma4均值=%.4f",
            df_main["BR0"].mean(), df_main["BRT"].mean(), df_main["dBR"].mean(),
            df_main["Loss_Ratio"].mean(), df_main["Max_Prop_Gen"].mean(), int(df_main["Max_Prop_Gen_max"].max()),
            float(sri_raw.mean()), float(sri_ma4.mean()) if len(sri_ma4) else float("nan"),
        )
        logger.info("结果目录（含 sri_index.csv 与 figs_ma4/ figs_raw_19q/ 标准五图）: %s", run_dir)
    else:
        logger.warning("主样本产出为空，请检查配置与数据源！")


if __name__ == "__main__":
    main()
