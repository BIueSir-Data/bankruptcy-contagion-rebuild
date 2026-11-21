# -*- coding: utf-8 -*-
"""
common.py —— 通用工程规范工具（配置加载、运行目录、日志、种子、安全矩阵读取、Git commit）
"""
import os
import json
import shutil
import hashlib
import logging
import subprocess
from datetime import datetime

import yaml
import numpy as np
import pandas as pd


SUMMARY_TOKENS = ("合计", "总计", "小计", "total", "sum")


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_git_commit(repo_dir: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "NO_GIT"
    except Exception:
        return "NO_GIT"


def resolve_paths(cfg: dict, cfg_path: str):
    """基于配置文件所在位置（重建实验方案/configs/xxx.yaml）解析项目根与原始数据根。"""
    cfg_abs = os.path.abspath(cfg_path)
    proj_root = os.path.dirname(os.path.dirname(cfg_abs))  # 重建实验方案/
    raw_root = os.path.abspath(os.path.join(proj_root, cfg.get("raw_data_root", "..")))
    out_root = os.path.abspath(os.path.join(proj_root, cfg.get("output_root", ".")))
    return proj_root, raw_root, out_root


def discover_industries(raw_root: str) -> list:
    """扫描原始数据目录下的全部申万一级行业（去掉末尾 'I' 返回行业名列表）。"""
    inds = []
    for d in sorted(os.listdir(raw_root)):
        full = os.path.join(raw_root, d)
        if os.path.isdir(full) and d.endswith("I"):
            inds.append(d[:-1])
    return inds


def create_run_dir(out_root: str, exp_name: str, cfg_path: str):
    """在 runs/<YYYYMMDD_HHMMSS>_<exp_name>/ 下创建本次运行目录并保存配置副本与日志器。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, "runs", f"{ts}_{exp_name}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(cfg_path, os.path.join(run_dir, os.path.basename(cfg_path)))

    logger = logging.getLogger(f"{exp_name}_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return run_dir, logger


def save_seeds(run_dir: str, seed_records: list):
    path = os.path.join(run_dir, "seeds.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seed_records, f, ensure_ascii=False, indent=2)
    return path


def quarter_seed(industry: str, quarter: str, gamma_q: float, p0: float, phi: float, salt: int = 0) -> int:
    """与现行 8.31 基准代码保持一致的确定性种子生成函数。"""
    s = f"{industry}|{quarter}|{gamma_q}|{p0}|{phi}|{salt}"
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "big")


def _is_summary_label(label) -> bool:
    s = str(label).strip().lower()
    return any(tok in s for tok in SUMMARY_TOKENS)


def read_stock_matrix_safe(mat_path: str, strip_mode: str = "auto_by_name"):
    """
    安全读取 <quarter>_<行业>I_stock_matrix.csv。
    返回: (W_df, had_summary_row, had_summary_col)
      - strip_mode == "auto_by_name": 仅当末行/末列标签包含“合计/总计/Total”时才剔除（避免误删真实企业）
      - strip_mode == "legacy_always": 沿用原代码无条件 W.iloc[:-1, :-1]
    """
    W = pd.read_csv(mat_path, encoding="utf-8-sig", index_col=0)
    had_row = False
    had_col = False
    if W.shape[0] > 0 and _is_summary_label(W.index[-1]):
        had_row = True
    if W.shape[1] > 0 and _is_summary_label(W.columns[-1]):
        had_col = True

    if strip_mode == "legacy_always":
        if W.shape[0] > 1 and W.shape[1] > 1:
            W = W.iloc[:-1, :-1]
    else:
        if had_row:
            W = W.iloc[:-1, :]
        if had_col:
            W = W.iloc[:, :-1]

    W.index = W.index.astype(str).str.strip()
    W.columns = W.columns.astype(str).str.strip()
    return W, had_row, had_col


def compute_sri_tables(df_panel: pd.DataFrame):
    """
    从行业-季度面板计算两套 SRI：
      1) sri_raw_19q: 19 季原值加权 SRI（不做 MA4、不做 Z 变换，遵循导师最新口径）
      2) sri_ma4_16q: 4 期移动平均加权 SRI（16 季，仅作与 11.16/第 5 章旧口径的附注对照）
    缺失一律保留 NaN（导出时为 NA）。
    """
    d = df_panel.sort_values(["Industry", "Quarter"]).copy()
    d["IRCI_raw"] = d["BRT"] * d["Loss_Ratio"]
    d["RawW_raw"] = d["Leverage"] * d["WCC"]

    # 1) 19 季原值 SRI
    w1 = d.dropna(subset=["IRCI_raw", "RawW_raw"]).copy()
    w1["TotW"] = w1.groupby("Quarter")["RawW_raw"].transform("sum")
    w1["Nw"] = np.where(w1["TotW"] > 0, w1["RawW_raw"] / w1["TotW"], np.nan)
    w1["WR"] = w1["Nw"] * w1["IRCI_raw"]
    sri_raw = w1.groupby("Quarter", as_index=True)["WR"].sum(min_count=1).rename("SRI_raw_19q")

    # 2) 旧口径 MA4 16 季 SRI（附注对照）
    for src, dst in [("BRT", "R_BR"), ("Loss_Ratio", "R_LS"),
                     ("Leverage", "R_LEV"), ("WCC", "R_WCC")]:
        d[dst] = d.groupby("Industry")[src].transform(lambda s: s.rolling(4).mean())
    d["IRCI_ma4"] = d["R_BR"] * d["R_LS"]
    w2 = d.dropna(subset=["IRCI_ma4", "R_LEV", "R_WCC"]).copy()
    w2["RawW"] = w2["R_LEV"] * w2["R_WCC"]
    w2["TotW"] = w2.groupby("Quarter")["RawW"].transform("sum")
    w2["Nw"] = np.where(w2["TotW"] > 0, w2["RawW"] / w2["TotW"], np.nan)
    w2["WR"] = w2["Nw"] * w2["IRCI_ma4"]
    sri_ma4 = w2.groupby("Quarter", as_index=True)["WR"].sum(min_count=1).rename("SRI_ma4_16q")

    all_q = sorted(d["Quarter"].dropna().unique())
    out = pd.DataFrame(index=all_q)
    out.index.name = "Quarter"
    out = out.join(sri_raw).join(sri_ma4)
    return out
