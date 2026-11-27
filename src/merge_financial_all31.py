# -*- coding: utf-8 -*-
"""
merge_financial_all31.py —— 将已仿真完成的银行/非银金融与 29 主样本合并为 31 全行业持久结果
================================================================================
说明：
  `run_d0.py` 与 `run_v2.py` 在全量运行时已经对全部 31 个行业（含银行、非银金融共 38 个窗口）
  完成了仿真，并保存在各版本目录的 `panel_all.csv` / `d0_baseline_panel.csv` 中；此前仅作图
  默认使用了剔除金融后的 29 行业主样本 `panel_main.csv`。

本脚本无需重复仿真，直接为 `版本归档/` 下的全部 7 个版本生成持久化 31 全行业目录 `含金融31行业/`：
  - panel_all31.csv                 （31 行业完整面板）
  - industry_summary_all31.csv      （31 行业分行业汇总，含银行与非银金融）
  - sri_index.csv                   （31 行业 SRI_raw_19q 与 SRI_ma4_16q）
  - figs_ma4/                       （含金融 31 行业 MA4 五张论文标准图）
  - figs_raw_19q/                   （含金融 31 行业单季 19Q 五张论文标准图）
并汇总生成两张持久化总对比 CSV：
  - 版本归档/V1_V2_all31_vs_main29_comparison.csv
  - 版本归档/V1_V2_SRI_all31_by_quarter.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotting import generate_standard_5figs

VERSIONS: List[Tuple[str, str, str]] = [
    ("V1.0_D0_干净单季基线", "V1.0_D0", "d0_baseline_panel.csv"),
    ("V2.0_会计与社会力核心", "V2.0", "panel_all.csv"),
    ("V2.1_动力学与同步更新", "V2.1", "panel_all.csv"),
    ("V2.1b_毛利率版", "V2.1b", "panel_all.csv"),
    ("V2.1c_稀疏骨干网版", "V2.1c", "panel_all.csv"),
    ("V2.1d_全下游按权下传版", "V2.1d", "panel_all.csv"),
    ("V2.1e_外生缓冲冲击50pct版", "V2.1e", "panel_all.csv"),
]

FIN_INDS = {"银行", "非银金融"}


def _subset_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "BR0": float("nan"), "BRT": float("nan"), "dBR": float("nan"), "Loss": float("nan"), "MaxGen": 0}
    max_col = "Max_Prop_Gen_max" if "Max_Prop_Gen_max" in df.columns else "Max_Prop_Gen"
    return {
        "rows": int(len(df)),
        "BR0": round(float(df["BR0"].mean()), 4),
        "BRT": round(float(df["BRT"].mean()), 4),
        "dBR": round(float(df["dBR"].mean()), 4),
        "Loss": round(float(df["Loss_Ratio"].mean()), 4),
        "MaxGen": int(df[max_col].max()),
    }


def main() -> None:
    proj = Path(__file__).resolve().parent.parent
    arch = proj / "版本归档"

    comp_rows = []
    sri_ma4_cols = {}
    sri_raw_cols = {}

    for folder, tag, panel_fname in VERSIONS:
        ver_dir = arch / folder
        p_all = ver_dir / panel_fname
        if not p_all.exists():
            print(f"[SKIP] 缺少 {p_all}")
            continue

        df_all = pd.read_csv(p_all)
        df_main = df_all[~df_all["Industry"].isin(FIN_INDS)].copy()
        df_fin = df_all[df_all["Industry"].isin(FIN_INDS)].copy()
        df_bank = df_all[df_all["Industry"] == "银行"].copy()
        df_nonbank = df_all[df_all["Industry"] == "非银金融"].copy()

        out_dir = ver_dir / "含金融31行业"
        out_dir.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_dir / "panel_all31.csv", index=False, encoding="utf-8-sig", na_rep="NA")

        # 31 行业分行业汇总
        max_col = "Max_Prop_Gen_max" if "Max_Prop_Gen_max" in df_all.columns else "Max_Prop_Gen"
        ind_sum = df_all.groupby("Industry").agg(
            Valid_Quarters=("Quarter", "count"),
            BR0_mean=("BR0", "mean"),
            BRT_mean=("BRT", "mean"),
            dBR_mean=("dBR", "mean"),
            Loss_Ratio_mean=("Loss_Ratio", "mean"),
            Max_Prop_Gen_mean=("Max_Prop_Gen", "mean"),
            Max_Prop_Gen_max=(max_col, "max"),
            WCC_mean=("WCC", "mean"),
            Leverage_mean=("Leverage", "mean"),
        ).reset_index()
        ind_sum["Is_Financial"] = ind_sum["Industry"].isin(FIN_INDS).astype(int)
        ind_sum.to_csv(out_dir / "industry_summary_all31.csv", index=False, encoding="utf-8-sig", na_rep="NA")

        # 生成含金融 31 行业两套标准五图 + sri_index.csv
        sri_all31, sri_raw, sri_ma4 = generate_standard_5figs(
            df_all, str(out_dir), note_prefix=f"{tag} All 31 Inds (incl. Financial)"
        )
        sri_raw_cols[f"{tag}_raw19q"] = sri_raw
        sri_ma4_cols[f"{tag}_ma4_16q"] = sri_ma4

        # 读取原 29 行业 SRI
        sri_main_p = ver_dir / "sri_index.csv"
        if sri_main_p.exists():
            sm = pd.read_csv(sri_main_p)
            main_sri_raw = round(float(sm["SRI_raw_19q"].mean()), 4)
            main_sri_ma4 = round(float(sm["SRI_ma4_16q"].dropna().mean()), 4)
        else:
            main_sri_raw = float("nan")
            main_sri_ma4 = float("nan")

        st_main = _subset_stats(df_main)
        st_all = _subset_stats(df_all)
        st_bank = _subset_stats(df_bank)
        st_nb = _subset_stats(df_nonbank)

        comp_rows.append({
            "Version": tag,
            "Main29_BR0": st_main["BR0"],
            "Main29_BRT": st_main["BRT"],
            "Main29_dBR": st_main["dBR"],
            "Main29_Loss": st_main["Loss"],
            "Main29_SRI_ma4": main_sri_ma4,
            "All31_BR0": st_all["BR0"],
            "All31_BRT": st_all["BRT"],
            "All31_dBR": st_all["dBR"],
            "All31_Loss": st_all["Loss"],
            "All31_SRI_raw": round(float(sri_raw.mean()), 4),
            "All31_SRI_ma4": round(float(sri_ma4.mean()), 4),
            "Bank_BRT": st_bank["BRT"],
            "Bank_dBR": st_bank["dBR"],
            "Bank_Loss": st_bank["Loss"],
            "Bank_MaxGen": st_bank["MaxGen"],
            "NonBank_BRT": st_nb["BRT"],
            "NonBank_dBR": st_nb["dBR"],
            "NonBank_Loss": st_nb["Loss"],
            "NonBank_MaxGen": st_nb["MaxGen"],
        })
        print(
            f"[OK] {tag:6s} | Main29 SRI_ma4={main_sri_ma4:.4f} -> All31 SRI_ma4={float(sri_ma4.mean()):.4f} "
            f"| 银行 dBR={st_bank['dBR']:+.4f} (Gen={st_bank['MaxGen']}) | 非银 dBR={st_nb['dBR']:+.4f} (Gen={st_nb['MaxGen']})"
        )

    comp_df = pd.DataFrame(comp_rows)
    comp_path = arch / "V1_V2_all31_vs_main29_comparison.csv"
    comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")

    sri_q_df = pd.DataFrame({**sri_raw_cols, **sri_ma4_cols})
    sri_q_df.index.name = "Quarter"
    sri_q_path = arch / "V1_V2_SRI_all31_by_quarter.csv"
    sri_q_df.to_csv(sri_q_path, encoding="utf-8-sig", na_rep="NA")

    print("\n=== 29 主样本 vs 31 含金融全行业汇总对照 ===")
    print(comp_df.to_string(index=False))
    print(f"\n[OK] 持久化总对比 CSV 已保存：\n  1) {comp_path}\n  2) {sri_q_path}")


if __name__ == "__main__":
    main()
