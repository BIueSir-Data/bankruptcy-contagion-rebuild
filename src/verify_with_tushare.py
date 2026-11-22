# -*- coding: utf-8 -*-
"""
verify_with_tushare.py —— 使用 Tushare Pro 权威财报数据对原始单季表与旧版差分表做第三方双重核验
================================================================================
安全规范：Token 仅通过环境变量 TUSHARE_TOKEN 或命令行参数 --token 传入，绝不硬编码进代码库。

用法示例：
    $env:TUSHARE_TOKEN="你的token"
    python src/verify_with_tushare.py
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 抽检代表企业（覆盖非金融大行业、原2021q4缺失行业、金融行业）
SAMPLE_FIRMS: List[Tuple[str, str, str, str]] = [
    ("食品饮料I", "000529.XSHE", "000529.SZ", "广弘控股"),
    ("食品饮料I", "000568.XSHE", "000568.SZ", "泸州老窖"),
    ("农林牧渔I", "000019.XSHE", "000019.SZ", "深粮控股"),
    ("煤炭I",     "000552.XSHE", "000552.SZ", "甘肃能化"),
    ("银行I",     "600015.XSHG", "600015.SH", "华夏银行"),
]

END_DATE_TO_Q: Dict[str, str] = {
    "20240331": "2024q1",
    "20240630": "2024q2",
    "20240930": "2024q3",
    "20241231": "2024q4",
    "20250331": "2025q1",
    "20250630": "2025q2",
    "20250930": "2025q3",
}


def query_tushare(api_name: str, token: str, params: dict, fields: str) -> pd.DataFrame:
    """直接调用 Tushare Pro 官方 HTTP 接口（无需额外部署 tushare 包，与 pro_api 完全等价）。"""
    payload = json.dumps({
        "api_name": api_name,
        "token": token,
        "params": params,
        "fields": fields,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.tushare.pro",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("code") != 0:
        raise RuntimeError(f"Tushare API error ({api_name}): code={data.get('code')}, msg={data.get('msg')}")
    cols = data["data"]["fields"]
    items = data["data"]["items"]
    return pd.DataFrame(items, columns=cols)


def fetch_tushare_single_quarter(ts_code: str, token: str) -> pd.DataFrame:
    """
    从 Tushare 拉取合并报表(report_type=1)累计利润表(income)，并换算为标准单季营业收入：
      Q1 = 一季报；Q2 = 半年报 - 一季报；Q3 = 三季报 - 半年报；Q4 = 年报 - 三季报。
    """
    df = query_tushare(
        "income",
        token=token,
        params={"ts_code": ts_code, "start_date": "20240101", "end_date": "20251031", "report_type": "1"},
        fields="ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,revenue,total_revenue,oper_cost,n_income",
    )
    if df.empty:
        return pd.DataFrame()

    # 同一报告期若存在更正/重述，保留最新公告(f_ann_date 最大)
    df = df.sort_values(["end_date", "f_ann_date"]).drop_duplicates(subset=["end_date"], keep="last")
    df["quarter"] = df["end_date"].map(END_DATE_TO_Q)
    df = df[df["quarter"].notna()].sort_values("quarter").reset_index(drop=True)

    # 金融企业部分字段在 revenue，部分在 total_revenue；优先用 revenue，空则回退 total_revenue
    df["ts_rev_ytd"] = pd.to_numeric(df["revenue"], errors="coerce")
    tot_rev = pd.to_numeric(df["total_revenue"], errors="coerce")
    df["ts_rev_ytd"] = df["ts_rev_ytd"].fillna(tot_rev)
    df["ts_ni_ytd"] = pd.to_numeric(df["n_income"], errors="coerce")

    # 累计转单季
    df["yr"] = df["quarter"].str[:4]
    df["q_num"] = df["quarter"].str[-1].astype(int)
    df = df.sort_values(["yr", "q_num"]).reset_index(drop=True)
    df["ts_rev_sq"] = df.groupby("yr")["ts_rev_ytd"].diff()
    df.loc[df["q_num"] == 1, "ts_rev_sq"] = df.loc[df["q_num"] == 1, "ts_rev_ytd"]
    df["ts_ni_sq"] = df.groupby("yr")["ts_ni_ytd"].diff()
    df.loc[df["q_num"] == 1, "ts_ni_sq"] = df.loc[df["q_num"] == 1, "ts_ni_ytd"]
    return df[["quarter", "ts_rev_ytd", "ts_rev_sq", "ts_ni_sq"]]


def load_local_series(
    ind: str,
    ob_id: str,
    clean_root: Path,
    old_diff_root: Path,
    quarters: List[str],
) -> pd.DataFrame:
    rows = []
    for q in quarters:
        clean_p = clean_root / ind / f"{q}_{ind}.csv"
        old_p = old_diff_root / ind / f"{q}_{ind}.csv"
        v_clean = np.nan
        v_old = np.nan
        if clean_p.exists():
            d = pd.read_csv(clean_p)
            sub = d[d["order_book_id"].astype(str) == ob_id]
            if len(sub):
                v_clean = float(pd.to_numeric(sub["revenue"].iloc[0], errors="coerce"))
        if old_p.exists():
            d = pd.read_csv(old_p)
            sub = d[d["order_book_id"].astype(str) == ob_id]
            if len(sub):
                v_old = float(pd.to_numeric(sub["revenue"].iloc[0], errors="coerce"))
        rows.append({"quarter": q, "clean_rev": v_clean, "old_diff_rev": v_old})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Tushare Pro 第三方核验单季营收口径")
    parser.add_argument("--token", default=os.environ.get("TUSHARE_TOKEN", ""), help="Tushare Pro Token")
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("请通过 --token 或环境变量 TUSHARE_TOKEN 提供 Tushare Token。")

    proj_root = Path(__file__).resolve().parent.parent
    clean_root = proj_root / "data" / "clean"
    old_diff_root = proj_root.parent

    all_records = []
    quarters = sorted(END_DATE_TO_Q.values())

    for ind, ob_id, ts_code, name in SAMPLE_FIRMS:
        ts_df = fetch_tushare_single_quarter(ts_code, args.token)
        loc_df = load_local_series(ind, ob_id, clean_root, old_diff_root, quarters)
        m = ts_df.merge(loc_df, on="quarter", how="inner")
        m["industry"] = ind
        m["name"] = name
        m["ts_code"] = ts_code
        m["clean_vs_ts_sq_relerr"] = np.abs(m["clean_rev"] - m["ts_rev_sq"]) / np.maximum(np.abs(m["ts_rev_sq"]), 1.0)
        m["old_vs_ts_sq_relerr"] = np.abs(m["old_diff_rev"] - m["ts_rev_sq"]) / np.maximum(np.abs(m["ts_rev_sq"]), 1.0)
        all_records.append(m)

        print(f"\n=== {ind} | {name} ({ts_code}) ===")
        disp = m[["quarter", "ts_rev_ytd", "ts_rev_sq", "clean_rev", "old_diff_rev", "clean_vs_ts_sq_relerr", "old_vs_ts_sq_relerr"]].copy()
        for col in ["ts_rev_ytd", "ts_rev_sq", "clean_rev", "old_diff_rev"]:
            disp[col] = (disp[col] / 1e8).round(3)  # 单位：亿元
        disp["clean_vs_ts_sq_relerr"] = (disp["clean_vs_ts_sq_relerr"] * 100).round(2)
        disp["old_vs_ts_sq_relerr"] = (disp["old_vs_ts_sq_relerr"] * 100).round(2)
        disp.columns = ["Quarter", "TS_YTD(亿)", "TS_单季(亿)", "新版clean(亿)", "旧版差分(亿)", "新版误差%", "旧版误差%"]
        print(disp.to_string(index=False))

    out_df = pd.concat(all_records, ignore_index=True)
    out_path = proj_root / "data" / "clean" / "tushare_double_check.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 第三方核验明细已保存至: {out_path}")


if __name__ == "__main__":
    main()
