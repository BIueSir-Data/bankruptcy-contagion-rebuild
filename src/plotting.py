"""
plotting.py —— 论文实证标准五图与 SRI 指标统一出图模块
================================================================================
继承自论文原学术作图规范（Times New Roman、DPI=300、季度轴 45°、热力图三态标注：
NaN→灰色 "/"；真零→"0.00%"/"0.0000"；极小→"<0.01%"/"<0.0001"）。

每次基础破产模型运行后统一产出：
  - sri_index.csv                          （含 19 季单季原始 SRI 与 16 季 MA4 滚动平均 SRI）
  - figs_ma4/                              （论文原口径：4 季度滚动平均 MA4，16 季）
      * sri_index.csv
      * 1_Final_Bankruptcy_Ratio_Heatmap.png
      * 2_Net_Profit_Loss_Ratio_Heatmap.png
      * 3_Industry_Risk_Value_Heatmap.png
      * 4_Weighted_Industry_Risk_Heatmap.png
      * 5_Systemic_Risk_Index_Evolution.png
  - figs_raw_19q/                          （修订指导 D0.6 口径：单季原始值无滚动平均，19 季）
      * sri_index.csv
      * 1_Final_Bankruptcy_Ratio_Heatmap.png
      * 2_Net_Profit_Loss_Ratio_Heatmap.png
      * 3_Industry_Risk_Value_Heatmap.png
      * 4_Weighted_Industry_Risk_Heatmap.png
      * 5_Systemic_Risk_Index_Evolution.png

独立用法（对任意已跑完的 panel CSV 补出五图）：
    python src/plotting.py --panel runs/20261121_215414_D0_data_audit/d0_baseline_panel_main.csv
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

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

INDUSTRY_EN: Dict[str, str] = {
    "交通运输": "Transportation",
    "休闲服务": "Leisure Services",
    "传媒": "Media",
    "公用事业": "Utilities",
    "农林牧渔": "Agriculture",
    "化工": "Chemicals",
    "医药生物": "Medicine & Biology",
    "商业贸易": "Commercial Trade",
    "国防军工": "Defense & Military",
    "家用电器": "Home Appliances",
    "建筑材料": "Building Materials",
    "建筑装饰": "Construction & Decoration",
    "房地产": "Real Estate",
    "有色金属": "Non-ferrous Metals",
    "机械设备": "Machinery",
    "汽车": "Automotive",
    "煤炭": "Coal",
    "环保": "Environmental",
    "电子": "Electronics",
    "电气设备": "Power Equipment",
    "石油石化": "Oil & Petrochemicals",
    "纺织服装": "Textiles",
    "综合": "Conglomerates",
    "美容护理": "Beauty & Personal Care",
    "计算机": "Computer",
    "轻工制造": "Light Manufacturing",
    "通信": "Telecommunications",
    "钢铁": "Steel",
    "银行": "Banking",
    "非银金融": "Non-bank Financial",
    "食品饮料": "Food & Beverages",
}


def heatmap_academic(
    df: pd.DataFrame,
    val_col: str,
    title: str,
    cmap_name: str,
    out_path: str,
    is_percentage: bool = True,
    note_suffix: str = "Cleaned Sample, MA4",
) -> None:
    """绘制论文标准学术热力图（三态单元格标注）。"""
    pivot_df = df.pivot(index="Industry_EN", columns="Quarter", values=val_col)
    pivot_df = pivot_df.reindex(sorted(pivot_df.index), axis=0)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    fig, ax = plt.subplots(
        figsize=(max(12, pivot_df.shape[1] * 0.8), max(8, pivot_df.shape[0] * 0.45))
    )
    data = pivot_df.values.astype(float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#EEEEEE")
    finite_vals = data[np.isfinite(data)]
    vmax = float(np.nanmax(finite_vals)) if finite_vals.size > 0 and np.nanmax(finite_vals) > 0 else 1.0
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            v = pivot_df.values[i, j]
            if pd.isna(v):
                s, c, fw = "/", "#999999", "normal"
            elif v == 0:
                s, c, fw = ("0.00%" if is_percentage else "0.0000"), "black", "normal"
            elif 0 < v < 0.00005:
                s, c, fw = ("<0.01%" if is_percentage else "<0.0001"), "black", "normal"
            else:
                s = f"{v:.2%}" if is_percentage else f"{v:.4f}"
                c = "white" if v > vmax * 0.55 else "black"
                fw = "bold" if v > vmax * 0.4 else "normal"
            ax.text(
                j, i, s, ha="center", va="center", fontsize=8,
                color=c, fontweight=fw, family="Times New Roman",
            )

    ax.set_xticks(range(pivot_df.shape[1]))
    ax.set_xticklabels(
        pivot_df.columns, rotation=45, ha="right", fontsize=10, family="Times New Roman"
    )
    ax.set_yticks(range(pivot_df.shape[0]))
    ax.set_yticklabels(pivot_df.index, fontsize=10, family="Times New Roman")
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    cb.set_label(
        title, rotation=270, labelpad=15, fontsize=12, fontweight="bold", family="Times New Roman"
    )
    ax.set_title(
        f"{title} Heatmap ({note_suffix})",
        fontsize=14, fontweight="bold", pad=12, family="Times New Roman",
    )
    plt.xlabel("Quarter", fontsize=12, fontweight="bold", family="Times New Roman")
    plt.ylabel("Industry", fontsize=12, fontweight="bold", family="Times New Roman")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def sri_academic(
    sri_series: pd.Series,
    out_path: str,
    title: str = "Time Series of Systemic Risk Index (SRI)",
) -> None:
    """绘制论文标准 SRI 演化折线图（红方块 + 顶部数值标注）。"""
    q = list(sri_series.index)
    v = sri_series.values.astype(float)
    fig = plt.figure(figsize=(14, 6))
    plt.plot(
        range(len(v)), v, linewidth=2.5, marker="s", markersize=8,
        color="#D32F2F", label="Systemic Risk Index (SRI)",
    )
    plt.xlabel("Quarter", fontsize=14, fontweight="bold", family="Times New Roman")
    plt.ylabel("Systemic Risk Value", fontsize=14, fontweight="bold", family="Times New Roman")
    plt.title(title, fontsize=18, fontweight="bold", family="Times New Roman")
    plt.xticks(range(len(q)), q, rotation=45, ha="right", fontsize=12, family="Times New Roman")
    plt.yticks(fontsize=12, family="Times New Roman")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper left", prop={"family": "Times New Roman", "size": 12})

    span = float(np.nanmax(v) - np.nanmin(v)) if len(v) > 0 else 0.0
    y_off = span * 0.03 if span > 0 else 0.0005
    for i, val in enumerate(v):
        if np.isfinite(val):
            plt.text(
                i, val + y_off, f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", family="Times New Roman",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)


def _prepare_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """标准化输入面板列名（兼容 BRT / BRT_on / Bankrupt_Ratio，Loss_Ratio / Loss_Ratio_on）。"""
    df = raw_df.copy()
    df["Quarter"] = df["Quarter"].astype(str).str.strip().str.lower()
    if "Status" in df.columns:
        df = df[df["Status"] == "valid"].copy()

    if "Bankrupt_Ratio" not in df.columns:
        if "BRT" in df.columns:
            df["Bankrupt_Ratio"] = df["BRT"]
        elif "BRT_on" in df.columns:
            df["Bankrupt_Ratio"] = df["BRT_on"]
        else:
            raise KeyError("输入面板缺少 BRT / BRT_on / Bankrupt_Ratio 列")

    if "Loss_Ratio" not in df.columns:
        if "Loss_Ratio_on" in df.columns:
            df["Loss_Ratio"] = df["Loss_Ratio_on"]
        else:
            raise KeyError("输入面板缺少 Loss_Ratio / Loss_Ratio_on 列")

    df["Industry_Clean"] = df["Industry"].astype(str).str.rstrip("I")
    df["Industry_EN"] = df["Industry_Clean"].map(INDUSTRY_EN).fillna(df["Industry_Clean"])
    return df


def build_master_ma4(raw_df: pd.DataFrame) -> pd.DataFrame:
    """按第四章 MA4 口径构建主表：4 季度滚动平均 → IRCI → 季度内权重归一化 → Weighted_Risk。"""
    df = _prepare_panel(raw_df)
    out = []
    for ind in df["Industry_EN"].unique():
        g = df[df["Industry_EN"] == ind].sort_values("Quarter").reset_index(drop=True)
        g["Metric_Bankrupt"] = g["Bankrupt_Ratio"].rolling(4).mean()
        g["Metric_Loss"] = g["Loss_Ratio"].rolling(4).mean()
        g["Metric_WCC"] = g["WCC"].rolling(4).mean()
        g["Metric_Leverage"] = g["Leverage"].rolling(4).mean()
        out.append(g)
    d = pd.concat(out, ignore_index=True).dropna(subset=["Metric_Bankrupt"]).copy()
    d["IRCI"] = d["Metric_Bankrupt"] * d["Metric_Loss"]
    d["Raw_Weight"] = d["Metric_Leverage"] * d["Metric_WCC"]
    d["Total_Weight_Current_Quarter"] = d.groupby("Quarter")["Raw_Weight"].transform("sum")
    d["Normalized_Weight"] = np.where(
        d["Total_Weight_Current_Quarter"] > 0,
        d["Raw_Weight"] / d["Total_Weight_Current_Quarter"],
        0.0,
    )
    d["Weighted_Risk"] = d["Normalized_Weight"] * d["IRCI"]
    return d


def build_master_raw_19q(raw_df: pd.DataFrame) -> pd.DataFrame:
    """按修订指导 D0.6 单季原始口径（无滚动平均，19 季）构建主表。"""
    d = _prepare_panel(raw_df)
    d["Metric_Bankrupt"] = d["Bankrupt_Ratio"].astype(float)
    d["Metric_Loss"] = d["Loss_Ratio"].astype(float)
    d["Metric_WCC"] = d["WCC"].astype(float)
    d["Metric_Leverage"] = d["Leverage"].astype(float)
    d["IRCI"] = d["Metric_Bankrupt"] * d["Metric_Loss"]
    d["Raw_Weight"] = d["Metric_Leverage"] * d["Metric_WCC"]
    d["Total_Weight_Current_Quarter"] = d.groupby("Quarter")["Raw_Weight"].transform("sum")
    d["Normalized_Weight"] = np.where(
        d["Total_Weight_Current_Quarter"] > 0,
        d["Raw_Weight"] / d["Total_Weight_Current_Quarter"],
        0.0,
    )
    d["Weighted_Risk"] = d["Normalized_Weight"] * d["IRCI"]
    return d


def _render_5figs(
    master_df: pd.DataFrame,
    out_dir: str,
    note_suffix: str,
    sri_title: str,
) -> pd.Series:
    """输出五张标准论文图及 sri_index.csv。"""
    os.makedirs(out_dir, exist_ok=True)
    heatmap_academic(
        master_df, "Metric_Bankrupt", "Final Bankruptcy Ratio", "YlOrRd",
        os.path.join(out_dir, "1_Final_Bankruptcy_Ratio_Heatmap.png"),
        is_percentage=True, note_suffix=note_suffix,
    )
    heatmap_academic(
        master_df, "Metric_Loss", "Net Profit Loss Ratio", "YlOrRd",
        os.path.join(out_dir, "2_Net_Profit_Loss_Ratio_Heatmap.png"),
        is_percentage=True, note_suffix=note_suffix,
    )
    heatmap_academic(
        master_df, "IRCI", "Industry Risk Value (IRCI)", "Reds",
        os.path.join(out_dir, "3_Industry_Risk_Value_Heatmap.png"),
        is_percentage=False, note_suffix=note_suffix,
    )
    heatmap_academic(
        master_df, "Weighted_Risk", "Weighted Industry Risk Value", "Reds",
        os.path.join(out_dir, "4_Weighted_Industry_Risk_Heatmap.png"),
        is_percentage=False, note_suffix=note_suffix,
    )
    sri = master_df.groupby("Quarter")["Weighted_Risk"].sum().sort_index()
    sri_academic(
        sri,
        os.path.join(out_dir, "5_Systemic_Risk_Index_Evolution.png"),
        title=sri_title,
    )
    sri_df = pd.DataFrame({"Quarter": sri.index, "SRI": sri.values})
    sri_df.to_csv(os.path.join(out_dir, "sri_index.csv"), index=False, encoding="utf-8-sig")
    return sri


def generate_standard_5figs(
    raw_df: pd.DataFrame,
    out_dir: str,
    note_prefix: str = "Cleaned Sample",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    同时生成：
      1) figs_ma4/     （4 季度滚动平均 MA4，论文原实证风格五图 + sri_index.csv）
      2) figs_raw_19q/ （19 季度原始单季口径五图 + sri_index.csv）
      3) 顶层 sri_index.csv（合并 SRI_raw_19q 与 SRI_ma4_16q）
    """
    os.makedirs(out_dir, exist_ok=True)
    m_ma4 = build_master_ma4(raw_df)
    sri_ma4 = _render_5figs(
        m_ma4,
        os.path.join(out_dir, "figs_ma4"),
        note_suffix=f"{note_prefix}, MA4",
        sri_title="Time Series of Systemic Risk Index (SRI, MA4)",
    )

    m_raw = build_master_raw_19q(raw_df)
    sri_raw = _render_5figs(
        m_raw,
        os.path.join(out_dir, "figs_raw_19q"),
        note_suffix=f"{note_prefix}, Raw Single-Quarter 19Q",
        sri_title="Time Series of Systemic Risk Index (SRI, Single-Quarter 19Q)",
    )

    sri_combined = pd.DataFrame({"SRI_raw_19q": sri_raw, "SRI_ma4_16q": sri_ma4})
    sri_combined.index.name = "Quarter"
    sri_combined.to_csv(
        os.path.join(out_dir, "sri_index.csv"), encoding="utf-8-sig", na_rep="NA"
    )
    return sri_combined, sri_raw, sri_ma4


def main() -> None:
    parser = argparse.ArgumentParser(description="从破产模型面板 CSV 生成论文标准五图与 sri_index.csv")
    parser.add_argument("--panel", required=True, help="输入面板 CSV 路径（如 d0_baseline_panel_main.csv）")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认与面板同目录）")
    parser.add_argument("--note", default="Cleaned Sample", help="图标题副注前缀")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.panel))
    raw_df = pd.read_csv(args.panel)
    sri_comb, _, _ = generate_standard_5figs(raw_df, out_dir, note_prefix=args.note)
    print(f"[OK] 已在 {out_dir} 生成 sri_index.csv 以及 figs_ma4/、figs_raw_19q/ 各 5 张论文标准图。")
    print(sri_comb.round(6).to_string())


if __name__ == "__main__":
    main()
