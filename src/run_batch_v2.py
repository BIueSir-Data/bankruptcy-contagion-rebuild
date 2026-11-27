# -*- coding: utf-8 -*-
"""
run_batch_v2.py —— 无人值守一键批量运行 + 自动版本归档 + 总对比表生成脚本
================================================================================
专为“上课/离机挂机跑全量”设计：只需在终端运行**一行命令**，程序会自动按顺序完成：
  1) 跑通单元测试 tests/test_engine_v2.py；
  2) 依次全量运行 V2.1b -> V2.1c -> V2.1d -> V2.1e（每版自动产出标准五图与 sri_index.csv）；
  3) 自动将每版结果归档至 `版本归档/<版本号_名称>/` 并写入该版本的 README.md；
  4) 自动生成跨版本总对比表 `版本归档/V2_suite_comparison.csv`。

用法（在 重建实验方案/ 目录下运行一行即可离开）：
    # 全量 29 行业无人值守运行（8 进程，约 2-3 分钟全部跑完）
    python src/run_batch_v2.py --workers 8

    # 3 行业快速冒烟验证（不覆盖正式版本归档）
    python src/run_batch_v2.py --inds 煤炭,食品饮料,美容护理 --workers 3 --no-archive
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# 默认批量执行的渐进版本队列：(配置文件相对路径, 归档目录名, 一句话版本说明)
DEFAULT_QUEUE: List[Tuple[str, str, str]] = [
    (
        "configs/v2_1b_gross_margin.yaml",
        "V2.1b_毛利率版",
        "完全覆盖 V2.1，将第 4 条利润率由 V1 净利润率 max(τ,0) 切换为 V2 毛利率 max(1-E/R,0)",
    ),
    (
        "configs/v2_1c_sparse_network.yaml",
        "V2.1c_稀疏骨干网版",
        "完全覆盖 V2.1b（含毛利率 V2），将仿真网络由稠密全连接切换为 α=0.05 稀疏骨干网",
    ),
    (
        "configs/v2_1d_weighted_all_pass.yaml",
        "V2.1d_全下游按权下传版",
        "完全覆盖 V2.1c，将第 7 条健康节点下传由 V1 随机单下游切换为 V2 按边权分摊全部下游",
    ),
    (
        "configs/v2_1e_delta_stress_50pct.yaml",
        "V2.1e_外生缓冲冲击50pct版",
        "完全覆盖 V2.1d，按第 11 条对全系统净资产缓冲施加 δ=50% 宏观压力扣减",
    ),
]


def _find_latest_run_dir(runs_dir: Path, exp_name: str, after_ts: float) -> Path:
    cands = [
        p for p in runs_dir.iterdir()
        if p.is_dir() and exp_name in p.name and p.stat().st_mtime >= after_ts - 5.0
    ]
    if not cands:
        raise FileNotFoundError(f"未在 {runs_dir} 中找到本次运行产生的 {exp_name} 目录")
    return sorted(cands, key=lambda p: p.stat().st_mtime)[-1]


def _summarize_run(run_dir: Path, ver_label: str, desc: str) -> dict:
    p_main = pd.read_csv(run_dir / "panel_main.csv")
    sri = pd.read_csv(run_dir / "sri_index.csv")
    has_prop = float((p_main["dBR"] > 0).mean() * 100.0)
    gen_ge2 = float((p_main["Max_Prop_Gen_max"] >= 2).mean() * 100.0) if "Max_Prop_Gen_max" in p_main else 0.0
    return {
        "Version": ver_label,
        "Description": desc,
        "Run_Dir": run_dir.name,
        "Valid_Windows": int(len(p_main)),
        "BR0_mean": round(float(p_main["BR0"].mean()), 4),
        "BRT_mean": round(float(p_main["BRT"].mean()), 4),
        "dBR_mean": round(float(p_main["dBR"].mean()), 4),
        "Prop_Window_Pct": round(has_prop, 2),
        "Gen_ge2_Pct": round(gen_ge2, 2),
        "Max_Gen_max": int(p_main["Max_Prop_Gen_max"].max()) if "Max_Prop_Gen_max" in p_main else 0,
        "Loss_Ratio_mean": round(float(p_main["Loss_Ratio"].mean()), 4),
        "SRI_raw_mean": round(float(sri["SRI_raw_19q"].mean()), 4),
        "SRI_ma4_mean": round(float(sri["SRI_ma4_16q"].dropna().mean()), 4) if sri["SRI_ma4_16q"].notna().any() else float("nan"),
        "Tau_Clipped_total": int(p_main["N_tau_clipped"].sum()) if "N_tau_clipped" in p_main else 0,
        "Unconverged_Windows": int((p_main["All_Converged"] == 0).sum()) if "All_Converged" in p_main else 0,
    }


def _write_version_readme(dst_dir: Path, cfg_rel: str, stats: dict) -> None:
    cfg_name = os.path.basename(cfg_rel)
    readme = f"""# {stats['Version']}

## 1. 本版说明（怎么跑出来的）
- **改动摘要**：{stats['Description']}
- **复现命令**：
  ```powershell
  python tests/test_engine_v2.py
  python src/run_v2.py --config {cfg_rel} --workers 8
  ```
- **配置文件**：`{cfg_name}`（原始运行目录：`runs/{stats['Run_Dir']}`）

## 2. 核心结果摘要（29 主样本行业 × 19 季度）
- **有效窗口数**：`{stats['Valid_Windows']}`（未收敛窗口：`{stats['Unconverged_Windows']}`）
- **破产率与传染**：`BR0 = {stats['BR0_mean']:.4f}` → `BRT = {stats['BRT_mean']:.4f}`，净传染增量 **`ΔBR = {stats['dBR_mean']:+.4f}`**
- **传染覆盖面**：发生传染窗口占比 **`{stats['Prop_Window_Pct']:.2f}%`**，多代传染 (`Gen≥2`) 占比 **`{stats['Gen_ge2_Pct']:.2f}%`**，最大传染代数 **`{stats['Max_Gen_max']}`**
- **损失率与系统性风险**：`Loss_Ratio = {stats['Loss_Ratio_mean']:.4f}`，`SRI_raw 均值 = {stats['SRI_raw_mean']:.4f}`，`SRI_ma4 均值 = {stats['SRI_ma4_mean']:.4f}`
- **利润率非负截断数**：`{stats['Tau_Clipped_total']}` 条

## 3. 目录文件索引
- `panel_main.csv` / `panel_financial.csv` / `panel_all.csv` / `missing_windows.csv`：全量面板与缺失清单
- `industry_summary.csv` / `gamma_theta_br0_diag.csv`：分行业汇总与 `γ → θ_k → BR0` 诊断表
- `sri_index.csv` + `figs_ma4/`（论文 MA4 标准五图）+ `figs_raw_19q/`（单季 19Q 标准五图）
- `{cfg_name}` / `seeds.json` / `run.log`：配置副本、种子清单与运行日志
"""
    (dst_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="无人值守批量运行 V2.1b–V2.1e 并自动归档")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数（默认 8）")
    parser.add_argument("--inds", default=None, help="可选：逗号分隔行业名用于冒烟测试")
    parser.add_argument("--no-archive", action="store_true", help="仅运行不写入版本归档/目录（冒烟测试用）")
    args = parser.parse_args()

    proj = Path(__file__).resolve().parent.parent
    runs_dir = proj / "runs"
    archive_dir = proj / "版本归档"
    archive_dir.mkdir(exist_ok=True)

    t_all = time.time()
    print("=" * 72)
    print(f"[STEP 0] 先运行单元测试 tests/test_engine_v2.py ...")
    subprocess.run([sys.executable, str(proj / "tests" / "test_engine_v2.py")], cwd=str(proj), check=True)

    summary_rows = []

    # 先把已归档的 V2.0 和 V2.1 也读进来，便于生成完整总对比表
    if not args.no_archive:
        for existing_ver, existing_desc in [
            ("V2.0_会计与社会力核心", "覆盖 V1.0 干净数据 + 第 1,3,4,5,12 条会计与社会力核心修复"),
            ("V2.1_动力学与同步更新", "完全覆盖 V2.0 + 第 2,6,7,8,9,11 条动力学与同步更新修复"),
        ]:
            ex_dir = archive_dir / existing_ver
            if (ex_dir / "panel_main.csv").exists() and (ex_dir / "sri_index.csv").exists():
                summary_rows.append(_summarize_run(ex_dir, existing_ver, existing_desc))

    for idx, (cfg_rel, ver_folder, desc) in enumerate(DEFAULT_QUEUE, start=1):
        cfg_path = proj / cfg_rel
        # 读取 experiment_name
        exp_name = None
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("experiment_name:"):
                exp_name = line.split(":", 1)[1].strip().strip('"').strip("'")
                break
        if not exp_name:
            raise ValueError(f"{cfg_rel} 缺少 experiment_name")

        print("\n" + "=" * 72)
        print(f"[STEP {idx}/{len(DEFAULT_QUEUE)}] 运行 {ver_folder} ({cfg_rel}) | workers={args.workers}")
        t_start = time.time()
        cmd = [sys.executable, str(proj / "src" / "run_v2.py"), "--config", str(cfg_path), "--workers", str(args.workers)]
        if args.inds:
            cmd += ["--inds", args.inds]
        subprocess.run(cmd, cwd=str(proj), check=True)

        run_dir = _find_latest_run_dir(runs_dir, exp_name, t_start)
        stats = _summarize_run(run_dir, ver_folder, desc)
        summary_rows.append(stats)

        if not args.no_archive:
            dst_dir = archive_dir / ver_folder
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(run_dir, dst_dir)
            _write_version_readme(dst_dir, cfg_rel, stats)
            print(f"[ARCHIVED] 已自动归档至: 版本归档/{ver_folder}/ 并生成 README.md")

    comp_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 72)
    print("=== V2 渐进版本总对比表 ===")
    cols_show = ["Version", "BR0_mean", "BRT_mean", "dBR_mean", "Prop_Window_Pct", "Gen_ge2_Pct", "Max_Gen_max", "Loss_Ratio_mean", "SRI_ma4_mean"]
    print(comp_df[cols_show].to_string(index=False))

    if not args.no_archive:
        comp_path = archive_dir / "V2_suite_comparison.csv"
        comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 全部任务完成！总耗时 {time.time() - t_all:.1f}s，总对比表已保存至: {comp_path}")
    else:
        print(f"\n[OK] 冒烟测试全部通过！总耗时 {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
