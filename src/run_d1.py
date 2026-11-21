# -*- coding: utf-8 -*-
"""
run_d1.py —— 先行检验 D1：无传播基准对照（传播开启 vs 传播关闭）
========================================================================
遵循导师指南 D1 要求：
  1. 用 D0 后主样本（移出银行、非银金融），γ = 0.25 与 0.45 各跑一次，沿用 11.16 现有网络与代码；
  2. 同一初始激活集合 B(0)、同一组种子，分别跑“传播开启”和“传播关闭”；
  3. 逐行业-季度输出：BR0、BRT、ΔBR、λ、直接损失、传播新增损失、传播代数；
  4. 绘制各行业 BR0 与 ΔBR 的散点图。

用法示例：
  # 1) 快速试跑 5 个行业（推荐先验证）：
  python src/run_d1.py --config configs/d1_config.yaml --inds 交通运输,公用事业,国防军工,石油石化,美容护理 --workers 4

  # 2) 全量运行 29 个主样本行业（建议本机 4 或 8 进程）：
  python src/run_d1.py --config configs/d1_config.yaml --workers 8
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
    create_run_dir, save_seeds,
)
from engine_legacy_d1 import run_single_cell

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _pair_worker(task):
    """同一行业-季度、同一 gamma、同一 B(0) 与种子，成对跑传播关(off)与传播开(on)。"""
    raw_root, ind, q, gamma_q, params, strip_mode = task
    res_off, seed_off, reason_off = run_single_cell(
        raw_root, ind, q, gamma_q, params, strip_mode=strip_mode, propagation_enabled=False
    )
    res_on, seed_on, reason_on = run_single_cell(
        raw_root, ind, q, gamma_q, params, strip_mode=strip_mode, propagation_enabled=True
    )
    assert seed_off == seed_on, f"Seed mismatch: {seed_off} != {seed_on}"

    if res_off is None or res_on is None:
        return {
            "Quarter": q, "Industry": ind, "gamma": gamma_q, "Status": "NA", "Reason": reason_on,
            "BR0": np.nan, "BRT_off": np.nan, "BRT_on": np.nan, "dBR": np.nan, "lambda": np.nan,
            "Direct_Loss_Ratio": np.nan, "Loss_Ratio_off": np.nan, "Loss_Ratio_on": np.nan,
            "Prop_Added_Loss_Ratio": np.nan, "NonInitial_Loss_Ratio": np.nan,
            "Max_Prop_Gen": np.nan, "N_firms": np.nan, "N_initial": np.nan,
            "N_final_on": np.nan, "N_secondary_on": np.nan,
            "T_last": np.nan, "T_run": np.nan, "WCC": np.nan, "Leverage": np.nan, "theta": np.nan,
            "seed": seed_on,
        }

    # 校验同一初始激活集 B(0)
    assert res_off["N_initial"] == res_on["N_initial"], "Initial bankruptcy set B(0) mismatch!"
    loss_off = res_off["Loss_Ratio"]
    loss_on = res_on["Loss_Ratio"]
    prop_added_loss = (loss_on - loss_off) if (pd.notna(loss_on) and pd.notna(loss_off)) else np.nan

    return {
        "Quarter": q,
        "Industry": ind,
        "gamma": gamma_q,
        "Status": "valid",
        "Reason": "ok",
        "BR0": res_on["BR0"],
        "BRT_off": res_off["BRT"],
        "BRT_on": res_on["BRT"],
        "dBR": res_on["dBR"],
        "lambda": res_on["lambda"],
        "Direct_Loss_Ratio": res_on["Direct_Loss_Ratio"],
        "Loss_Ratio_off": loss_off,
        "Loss_Ratio_on": loss_on,
        "Prop_Added_Loss_Ratio": prop_added_loss,
        "NonInitial_Loss_Ratio": res_on["NonInitial_Loss_Ratio"],
        "Max_Prop_Gen": res_on["Max_Prop_Gen"],
        "N_firms": res_on["N_firms"],
        "N_initial": res_on["N_initial"],
        "N_final_on": res_on["N_final"],
        "N_secondary_on": res_on["N_secondary"],
        "T_last": res_on["T_last"],
        "T_run": res_on["T_run"],
        "WCC": res_on["WCC"],
        "Leverage": res_on["Leverage"],
        "theta": res_on["theta"],
        "seed": seed_on,
    }


def plot_d1_scatter(df_ind_sum: pd.DataFrame, gammas: list, fig_path: str, caption: str):
    ncols = len(gammas)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.2), dpi=160, squeeze=False)
    for idx, g in enumerate(gammas):
        ax = axes[0, idx]
        sub = df_ind_sum[df_ind_sum["gamma"] == g].dropna(subset=["BR0", "dBR"])
        ax.scatter(sub["BR0"], sub["dBR"], s=55, alpha=0.85, edgecolors="#1F4E79", facecolors="#4F81BD")
        for _, r in sub.iterrows():
            ax.annotate(r["Industry"], (r["BR0"], r["dBR"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("初始破产率 BR0 (行业跨期均值)")
        ax.set_ylabel("传播新增破产率 ΔBR = BRT - BR0 (行业跨期均值)")
        ax.set_title(f"D1 各行业 BR0 vs ΔBR (γ = {g})")
        ax.grid(True, ls="--", alpha=0.35)
    fig.text(0.01, 0.01, caption, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="执行先行检验 D1（无传播基准对照）")
    ap.add_argument("--config", default=os.path.join("configs", "d1_config.yaml"))
    ap.add_argument("--inds", default=None, help="可选：逗号分隔行业名试跑，如 交通运输,公用事业,国防军工")
    ap.add_argument("--workers", type=int, default=None, help="并行进程数（覆盖配置文件）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proj_root, raw_root, out_root = resolve_paths(cfg, args.config)
    run_dir, logger = create_run_dir(out_root, cfg.get("experiment_name", "D1_no_propagation"), args.config)
    commit = get_git_commit(proj_root)

    all_inds = discover_industries(raw_root)
    excluded_inds = set(cfg.get("excluded_industries", ["银行", "非银金融"]))
    if args.inds:
        req = [x.strip().rstrip("I") for x in args.inds.split(",") if x.strip()]
        industries = [x for x in req if x in all_inds]
    else:
        industries = [x for x in all_inds if x not in excluded_inds]
        if cfg.get("also_run_excluded", False):
            industries = all_inds

    quarters = list(cfg["quarters"])
    gammas = [float(g) for g in cfg["gamma_list"]]
    params = dict(cfg["sim_params"])
    strip_mode = cfg.get("strip_last_row_mode", "auto_by_name")
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 4))

    logger.info("=== 开始运行 D1 | commit=%s | config=%s ===", commit, os.path.basename(args.config))
    logger.info("行业数=%d | 季度数=%d | gammas=%s | strip_mode=%s | workers=%d",
                len(industries), len(quarters), gammas, strip_mode, workers)

    t0 = time.time()
    all_frames = []
    seed_records = []

    for g in gammas:
        tg = time.time()
        tasks = [(raw_root, ind, q, g, params, strip_mode) for ind in industries for q in quarters]
        logger.info("运行 γ=%.2f：%d 格 (每格传播关+传播开成对) × workers=%d", g, len(tasks), workers)
        if workers > 1 and len(tasks) > 4:
            with Pool(workers) as pool:
                rows = pool.map(_pair_worker, tasks)
        else:
            rows = [_pair_worker(t) for t in tasks]

        df_g = pd.DataFrame(rows)
        df_g.to_csv(os.path.join(run_dir, f"d1_panel_g{g:g}.csv"), index=False, encoding="utf-8-sig", na_rep="NA")
        all_frames.append(df_g)
        for r in rows:
            seed_records.append({"Industry": r["Industry"], "Quarter": r["Quarter"],
                                 "gamma": g, "seed": r["seed"], "status": r["Status"]})

        n_val = int((df_g["Status"] == "valid").sum())
        n_na = int((df_g["Status"] != "valid").sum())
        valid_sub = df_g[df_g["Status"] == "valid"]
        logger.info("γ=%.2f 完成 (%.1fs) | 有效=%d, 缺失(NA)=%d | 均值 BR0=%.4f, BRT_on=%.4f, dBR=%.4f, Max_Prop_Gen(max)=%s",
                    g, time.time() - tg, n_val, n_na,
                    valid_sub["BR0"].mean(), valid_sub["BRT_on"].mean(), valid_sub["dBR"].mean(),
                    valid_sub["Max_Prop_Gen"].max() if len(valid_sub) else "NA")

    save_seeds(run_dir, seed_records)
    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.to_csv(os.path.join(run_dir, "d1_panel_all.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

    # 按行业跨期均值汇总（供散点图与汇报）
    df_valid = df_all[df_all["Status"] == "valid"].copy()
    ind_sum = (
        df_valid.groupby(["gamma", "Industry"], as_index=False)[
            ["BR0", "BRT_off", "BRT_on", "dBR", "lambda",
             "Direct_Loss_Ratio", "Loss_Ratio_on", "Prop_Added_Loss_Ratio", "Max_Prop_Gen"]
        ].mean()
    )
    ind_sum.to_csv(os.path.join(run_dir, "d1_industry_summary.csv"), index=False, encoding="utf-8-sig", na_rep="NA")

    fig_dir = os.path.join(out_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    caption = f"Config: {os.path.basename(args.config)} | commit: {commit} | inds={len(industries)} | strip={strip_mode}"
    plot_d1_scatter(ind_sum, gammas, os.path.join(fig_dir, "D1_BR0_vs_dBR_scatter.png"), caption)
    plot_d1_scatter(ind_sum, gammas, os.path.join(run_dir, "D1_BR0_vs_dBR_scatter.png"), caption)

    logger.info("=== D1 全部完成 (总耗时 %.1fs) | 输出目录: %s ===", time.time() - t0, run_dir)


if __name__ == "__main__":
    main()
