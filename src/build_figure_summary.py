# -*- coding: utf-8 -*-
"""
build_figure_summary.py —— 汇总全部版本的热力图与 SRI 指标至统一目录
================================================================================
在 `版本归档/结果图与SRI总汇总/` 下生成：
  1) 1_SRI_All_Versions_Comparison_MA4.png      （左：29 主样本；右：31 含金融，全部 7 版同图对照）
  2) 2_SRI_All_Versions_Comparison_Raw19Q.png   （19 季单季原始口径，左 29 主样本 / 右 31 含金融）
  3) 3_Industry_IRCI_Comparison_Heatmap.png     （各行业跨版本平均 IRCI 热力图，一眼看清结构变化）
  4) SRI_All_Versions_Master.csv                （全部 7 版 29 主样本 & 31 含金融的逐季 SRI 总表）
  5) 代表版本五图合集/                          （将 V1.0_D0、V2.1、V2.1c、V2.1e 的标准五图按版本号平铺归集，免去逐层翻文件夹）
  6) README.md                                  （图文索引与关键数字说明）
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.sans-serif"] = ["Times New Roman", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

VERSIONS: List[Tuple[str, str, str, str]] = [
    ("V1.0_D0_干净单季基线", "V1.0 (Legacy 8.31)", "#1976D2", "--"),
    ("V2.0_会计与社会力核心", "V2.0 (Acct+Force)", "#7B1FA2", ":"),
    ("V2.1_动力学与同步更新", "V2.1 (Sync+Dyn)", "#388E3C", "-"),
    ("V2.1b_毛利率版", "V2.1b (Gross Margin)", "#00897B", "-."),
    ("V2.1c_稀疏骨干网版", "V2.1c (Sparse Net)", "#F57C00", "-"),
    ("V2.1d_全下游按权下传版", "V2.1d (Weighted Pass)", "#E64A19", "--"),
    ("V2.1e_外生缓冲冲击50pct版", "V2.1e (Delta=50%)", "#D32F2F", "-"),
]

# 需要平铺归集标准五图的代表性版本
REP_VERSIONS = [
    ("V1.0_D0_干净单季基线", "V1.0_D0"),
    ("V2.1_动力学与同步更新", "V2.1"),
    ("V2.1c_稀疏骨干网版", "V2.1c"),
    ("V2.1e_外生缓冲冲击50pct版", "V2.1e"),
]

INDUSTRY_EN = {
    "交通运输": "Transportation", "休闲服务": "Leisure Services", "传媒": "Media",
    "公用事业": "Utilities", "农林牧渔": "Agriculture", "化工": "Chemicals",
    "医药生物": "Medicine & Biology", "商业贸易": "Commercial Trade",
    "国防军工": "Defense & Military", "家用电器": "Home Appliances",
    "建筑材料": "Building Materials", "建筑装饰": "Construction & Decoration",
    "房地产": "Real Estate", "有色金属": "Non-ferrous Metals",
    "机械设备": "Machinery", "汽车": "Automotive", "煤炭": "Coal",
    "环保": "Environmental", "电子": "Electronics", "电气设备": "Power Equipment",
    "石油石化": "Oil & Petrochemicals", "纺织服装": "Textiles", "综合": "Conglomerates",
    "美容护理": "Beauty & Personal Care", "计算机": "Computer",
    "轻工制造": "Light Manufacturing", "通信": "Telecommunications",
    "钢铁": "Steel", "银行": "Banking", "非银金融": "Non-bank Financial",
    "食品饮料": "Food & Beverages",
}


def plot_sri_two_panel(
    df_main_dict: dict,
    df_all_dict: dict,
    col: str,
    title_main: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    panels = [
        (axes[0], df_main_dict, "Main Sample (29 Non-Financial Industries)"),
        (axes[1], df_all_dict, "All 31 Industries (incl. Banking & Non-bank Fin.)"),
    ]
    markers = ["o", "^", "s", "d", "v", "P", "X"]

    for ax, d_map, sub_title in panels:
        quarters = None
        for idx, (folder, label, color, ls) in enumerate(VERSIONS):
            if folder not in d_map:
                continue
            s = d_map[folder].set_index("Quarter")[col].dropna()
            if quarters is None:
                quarters = list(s.index)
            ax.plot(
                range(len(s)), s.values,
                label=label, color=color, linestyle=ls,
                linewidth=2.0 if "V2.1" in label or "V1.0" in label else 1.5,
                marker=markers[idx % len(markers)], markersize=5, alpha=0.9,
            )
        if quarters:
            ax.set_xticks(range(len(quarters)))
            ax.set_xticklabels(quarters, rotation=45, ha="right", fontsize=10, family="Times New Roman")
        ax.set_title(sub_title, fontsize=13, fontweight="bold", family="Times New Roman")
        ax.set_xlabel("Quarter", fontsize=11, fontweight="bold", family="Times New Roman")
        ax.set_ylabel("Systemic Risk Index (SRI)", fontsize=11, fontweight="bold", family="Times New Roman")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=8.5, frameon=True, prop={"family": "Times New Roman", "size": 8.5})

    fig.suptitle(title_main, fontsize=15, fontweight="bold", family="Times New Roman", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def plot_industry_cross_version_irci(arch: Path, out_path: Path) -> None:
    """绘制 31 行业 × 7 版本的平均 IRCI（BRT × Loss_Ratio）汇总热力图。"""
    mat_cols = {}
    for folder, label, _, _ in VERSIONS:
        p = arch / folder / "含金融31行业" / "panel_all31.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["IRCI"] = df["BRT"] * df["Loss_Ratio"]
        df["Industry_EN"] = df["Industry"].map(INDUSTRY_EN).fillna(df["Industry"])
        short_lbl = label.split(" ")[0]
        mat_cols[short_lbl] = df.groupby("Industry_EN")["IRCI"].mean()

    piv = pd.DataFrame(mat_cols)
    # 按 V2.1e 降序排列，便于直观识别高风险行业
    sort_col = "V2.1e" if "V2.1e" in piv.columns else piv.columns[-1]
    piv = piv.sort_values(sort_col, ascending=False)

    fig, ax = plt.subplots(figsize=(11, 11))
    data = piv.values.astype(float)
    vmax = float(np.nanmax(data)) if np.any(np.isfinite(data)) else 1.0
    # 针对非金融与金融跨度大，采用温和色阶上限（第 95 分位）避免银行单格把其余压白
    vmax_disp = float(np.nanpercentile(data, 96)) if np.any(np.isfinite(data)) else vmax
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(vmax_disp, 1e-4))

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                s, c, fw = "/", "#999999", "normal"
            else:
                s = f"{v:.4f}"
                c = "white" if v > vmax_disp * 0.55 else "black"
                fw = "bold" if v > vmax_disp * 0.40 else "normal"
            ax.text(j, i, s, ha="center", va="center", fontsize=8.5, color=c, fontweight=fw, family="Times New Roman")

    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, fontsize=11, fontweight="bold", family="Times New Roman")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9.5, family="Times New Roman")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cb.set_label("Mean Industry Risk Value (IRCI = BRT × Loss_Ratio)", rotation=270, labelpad=16, fontsize=11, fontweight="bold", family="Times New Roman")
    ax.set_title("Cross-Version Mean IRCI Heatmap (All 31 Industries, 2021Q1–2025Q3)", fontsize=13.5, fontweight="bold", pad=12, family="Times New Roman")
    ax.set_xlabel("Model Version", fontsize=11, fontweight="bold", family="Times New Roman")
    ax.set_ylabel("Industry (Sorted by V2.1e IRCI)", fontsize=11, fontweight="bold", family="Times New Roman")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def main() -> None:
    proj = Path(__file__).resolve().parent.parent
    arch = proj / "版本归档"
    out = arch / "结果图与SRI总汇总"
    out.mkdir(parents=True, exist_ok=True)

    df_main_map = {}
    df_all_map = {}
    master_cols = {}

    for folder, label, _, _ in VERSIONS:
        short = label.split(" ")[0]
        p_m = arch / folder / "sri_index.csv"
        p_a = arch / folder / "含金融31行业" / "sri_index.csv"
        if p_m.exists():
            dm = pd.read_csv(p_m)
            df_main_map[folder] = dm
            master_cols[f"{short}_Main29_raw19q"] = dm.set_index("Quarter")["SRI_raw_19q"]
            master_cols[f"{short}_Main29_ma4"] = dm.set_index("Quarter")["SRI_ma4_16q"]
        if p_a.exists():
            da = pd.read_csv(p_a)
            df_all_map[folder] = da
            master_cols[f"{short}_All31_raw19q"] = da.set_index("Quarter")["SRI_raw_19q"]
            master_cols[f"{short}_All31_ma4"] = da.set_index("Quarter")["SRI_ma4_16q"]

    # 1. 输出合并 SRI 总表 CSV
    master_df = pd.DataFrame(master_cols)
    master_df.index.name = "Quarter"
    master_df.to_csv(out / "SRI_All_Versions_Master.csv", encoding="utf-8-sig", na_rep="NA")

    # 2. 绘制两张多版本 SRI 总对比图（MA4 与 Raw 19Q）
    plot_sri_two_panel(
        df_main_map, df_all_map, "SRI_ma4_16q",
        "Systemic Risk Index (SRI, MA4) Across Progressive Model Versions",
        out / "1_SRI_All_Versions_Comparison_MA4.png",
    )
    plot_sri_two_panel(
        df_main_map, df_all_map, "SRI_raw_19q",
        "Systemic Risk Index (SRI, Raw Single-Quarter 19Q) Across Progressive Model Versions",
        out / "2_SRI_All_Versions_Comparison_Raw19Q.png",
    )

    # 3. 绘制 31 行业 × 7 版本平均 IRCI 汇总热力图
    plot_industry_cross_version_irci(arch, out / "3_Industry_IRCI_Comparison_Heatmap.png")

    # 4. 将代表版本（V1.0_D0, V2.1, V2.1c, V2.1e）的五张标准图平铺复制到统一子目录，方便集中翻阅
    for scope_sub, src_rel in [
        ("代表版本五图_29主样本_MA4", "figs_ma4"),
        ("代表版本五图_31含金融_MA4", os.path.join("含金融31行业", "figs_ma4")),
    ]:
        dst_sub = out / scope_sub
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        dst_sub.mkdir(parents=True, exist_ok=True)
        for folder, tag in REP_VERSIONS:
            src_dir = arch / folder / src_rel
            if not src_dir.exists():
                continue
            for png in sorted(src_dir.glob("*.png")):
                shutil.copy2(png, dst_sub / f"{tag}_{png.name}")

    # 5. 生成汇总 README.md
    readme_lines = [
        "# 结果图与 SRI 指标总汇总（V1.0_D0 → V2.1e）",
        "",
        "本目录将所有核心可视化图表与 SRI 指标汇总到一处，方便汇报与直接嵌入论文：",
        "",
        "## 一、跨版本总览图与总表",
        "- **`1_SRI_All_Versions_Comparison_MA4.png`**：7 个渐进版本（`V1.0_D0 ~ V2.1e`）的 **MA4 滚动平均 SRI** 同图对照（左：29 非金融主样本；右：31 含金融全行业）。",
        "- **`2_SRI_All_Versions_Comparison_Raw19Q.png`**：7 个渐进版本的 **19 季单季原始 SRI** 同图对照（左：29 主样本；右：31 含金融）。",
        "- **`3_Industry_IRCI_Comparison_Heatmap.png`**：**31 个行业 × 7 个版本**的平均行业风险值（`IRCI = BRT × Loss_Ratio`）汇总热力图（按 V2.1e 降序）。",
        "- **`SRI_All_Versions_Master.csv`**：全部 7 个版本的 29 主样本与 31 含金融逐季 SRI 数值总表（含 `raw_19q` 与 `ma4`）。",
        "",
        "## 二、代表版本五张标准热力图/SRI 图集中目录（平铺免翻层）",
        "- **`代表版本五图_29主样本_MA4/`**：集中存放 `V1.0_D0`、`V2.1`、`V2.1c`、`V2.1e` 四个代表版本的 29 主样本标准五图（共 20 张 300-DPI PNG）。",
        "- **`代表版本五图_31含金融_MA4/`**：集中存放同上 4 个代表版本的 31 含金融标准五图（共 20 张 300-DPI PNG）。",
        "",
        "## 三、各版本 SRI 均值速查表",
        "",
        "| 版本 | 29 主样本 `SRI_raw` | 29 主样本 `SRI_ma4` | 31 含金融 `SRI_raw` | 31 含金融 `SRI_ma4` |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for folder, label, _, _ in VERSIONS:
        short = label.split(" ")[0]
        m_r = master_df[f"{short}_Main29_raw19q"].mean()
        m_4 = master_df[f"{short}_Main29_ma4"].dropna().mean()
        a_r = master_df[f"{short}_All31_raw19q"].mean()
        a_4 = master_df[f"{short}_All31_ma4"].dropna().mean()
        readme_lines.append(f"| **{label}** | `{m_r:.4f}` | `{m_4:.4f}` | `{a_r:.4f}` | `{a_4:.4f}` |")

    (out / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"[OK] 结果图与 SRI 总汇总已生成至: {out}")


if __name__ == "__main__":
    main()
