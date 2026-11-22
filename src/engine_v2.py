# -*- coding: utf-8 -*-
"""
engine_v2.py —— 渐进式冻结模型引擎（V2.0 -> V2.1 -> V2.2 累积配置驱动）
================================================================================
严格对照《11.16版重建实验指导第一批_修订》第三节 12 条修正，采用单引擎 + YAML
累积开关设计，确保 V2.1 完全覆盖 V2.0 并叠加动力学修正，V2.2 完全覆盖 V2.1 并叠加
蒙特卡洛重复与置信区间：

  - V2.0（会计与社会力核心，第 1, 3, 4, 5, 12 条）：
      1) 缓冲与阈值：C_i(0) = total_assets - total_liab, U_i = 0（可切 total_assets / legacy_income）
      3) 可视度项：alpha * exp(-gamma_v * vis_b) + (1 - alpha)，主设定 alpha=0.5, gamma_v=1.0
      4) 利润率非负截断：V1=max(tau, 0) 并记录截断数；V2=max(1 - cost/rev, 0)
      5) 避免除零与 k_b 取倒数：k_b = revenue / operating_expense（>1）；ΔS 直接相乘不除 τ；
         严格剔除 revenue <= 0 或 cost <= 0（VS 评分中的成本率 cost/rev 保持不变）
      12) 指标口径：损失率 = Σ(C_i(0)-C_i(T))^+ / ΣC_i(0)^+（分母为0记 NA）；
          WCC 采用 Fagiolo (2007) 有向加权聚类系数；BAR_end / ΔBAR / λ = ΔBAR/(1-BAR_start)
          遵循 Zhang & Xia (2020, IEEE Access) 定义。

  - V2.1（在 V2.0 基础上叠加动力学与同步更新，第 2, 6, 7, 8, 9, 11 条）：
      2) 边方向明确为 a->b（a 为 b 的供应商）；修复旧版 Ego 查反邻居方向的 Bug（全邻居破产占比）
      6) 二次影响几何衰减：上步受冲击则以概率 p_b 再受 phi * ||F||（理论期望 phi*||F||*p_b/(1-p_b)）
      7) 健康节点下传守恒：V1 随机选一下游 / V2 按边权分摊全部下游，均强制下传总量 <= ΔS
      8) 同步更新：第 s 步统一按第 s-1 步状态算力、统一扣减、统一判定激活
      9) 双重停止规则：连续 3 步无新增激活且待执行二次影响总额 < 1e-6 * ΣC_i(0)^+，或达到 S_max=1000
      11) γ -> θ_k -> BR0 诊断表 + 外生缓冲冲击参数 δ（全部企业缓冲扣减 δ%）

  - V2.2（在 V2.1 基础上叠加重复次数，第 10 条）：
      10) 每设定至少 R=50 次独立种子重复；若 BRT 均值标准误 > 0.005 自动扩至 R=200，
          报告均值与 5%–95% 分位区间。
"""
from __future__ import annotations

import math
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from common import quarter_seed, read_stock_matrix_safe
from engine_legacy_d1 import (
    EPS,
    _to_num,
    _z_score,
    disparity_filter_directed,
    fit_huber_theta,
    node_visibility,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 核心纯公式函数（便于 2-3 节点单元测试直接调用）
# ---------------------------------------------------------------------------
def compute_vis_factor(vis_b: float, vis_formula: str = "paper",
                       alpha_p: float = 0.5, gamma_v: float = 1.0) -> float:
    """第 3 条：可视度缓冲项。"""
    if vis_formula == "legacy_831":
        return math.exp(-vis_b)
    return alpha_p * math.exp(-gamma_v * vis_b) + (1.0 - alpha_p)


def compute_ego_broken_ratio(
    node_idx: int,
    up_neighbors: List[int],
    dn_neighbors: List[int],
    is_broken_flags: List[bool],
    ego_mode: str = "all_neighbors",
    side: str = "1a",
) -> float:
    """
    计算节点 b 的破产邻居占比（含节点自身构成的 1-ego 网络中的破产比例，
    或严格邻居中已破产比例）。
    - all_neighbors (V2.1 修复): 所有上下游唯一邻居集合中处于破产状态的比例。
    - legacy_inverted (V2.0 兼容旧机制): 旧版 1a 查 dn、1b 查 up。
    """
    if ego_mode == "legacy_inverted":
        nbrs = dn_neighbors if side == "1a" else up_neighbors
    else:
        nbrs = list(set(up_neighbors) | set(dn_neighbors))
    if not nbrs:
        return 0.0
    broken_cnt = sum(1 for j in nbrs if is_broken_flags[j])
    return broken_cnt / float(len(nbrs))


def compute_force_1a_and_delta_s(
    v_ab: float,
    k_b: float,
    tau_b: float,
    vis_fac: float,
    ego_ratio: float,
    share_up: float,
    direct_delta_s: bool = True,
) -> Tuple[float, float]:
    """
    公式 1a（供应商 a 破产 -> 客户 b 受冲击）：
      ΔS_ab = V_ab * k_b * vis_fac * ego_ratio * share_up
      |F_ab| = ΔS_ab * tau_b
    第 5 条：直接计算 ΔS 再乘以 tau_b，完全避免 F/tau 除零问题，代数上等价。
    """
    if v_ab <= 0 or k_b <= 0 or ego_ratio <= 0 or share_up <= 0:
        return 0.0, 0.0
    delta_s = v_ab * k_b * vis_fac * ego_ratio * share_up
    if direct_delta_s:
        force = delta_s * max(tau_b, 0.0)
    else:
        force = delta_s * tau_b
    return float(force), float(delta_s)


def compute_force_1b(
    v_ba: float,
    tau_b: float,
    vis_fac: float,
    ego_ratio: float,
    share_dn: float,
) -> float:
    """
    公式 1b（客户 a 破产 -> 供应商 b 受冲击）：
      |F_ba| = V_ba * tau_b * vis_fac * ego_ratio * share_dn
    """
    if v_ba <= 0 or tau_b <= 0 or ego_ratio <= 0 or share_dn <= 0:
        return 0.0
    return float(v_ba * tau_b * vis_fac * ego_ratio * share_dn)


def compute_loss_ratio_v2(init_buffers: np.ndarray, final_buffers: np.ndarray) -> float:
    """
    第 12 条：损失率 = Σ(C_i(0) - C_i(T))^+ / ΣC_i(0)^+，分母为 0 记 np.nan (NA)。
    """
    init_pos = np.maximum(init_buffers, 0.0)
    denom = float(np.sum(init_pos))
    if denom <= 0.0:
        return float("nan")
    # 缓冲消耗不超过其初始正缓冲部分，也不允许负消耗（收益）抵消损失
    loss = np.maximum(init_pos - np.maximum(final_buffers, 0.0), 0.0)
    return float(np.sum(loss) / denom)


def compute_fagiolo_wcc(W_df: pd.DataFrame) -> float:
    """
    第 12 条：Fagiolo (2007) 有向加权聚类系数（NetworkX DiGraph weighted clustering
    采用 Fagiolo 2007 total directed triangles 几何平均定义，边权自动按 max(W) 归一化）。
    """
    es = W_df.stack().reset_index()
    es.columns = ["s", "t", "w"]
    es = es[(es["w"] > 0) & (es["s"] != es["t"])]
    if es.empty:
        return 0.0
    G = nx.from_pandas_edgelist(es, "s", "t", edge_attr="w", create_using=nx.DiGraph)
    cl = nx.clustering(G, weight="weight")
    return float(np.mean(list(cl.values()))) if cl else 0.0


# ---------------------------------------------------------------------------
# 单行业-单季度数据与网络准备
# ---------------------------------------------------------------------------
def prepare_quarter_v2(
    raw_root: str,
    ind_name: str,
    quarter: str,
    gamma_q: float,
    params: dict,
    strip_mode: str = "auto_by_name",
) -> Tuple[Optional[dict], str]:
    """准备单行业-单季度数据与网络（遵循 V2 清洗与缓冲口径）。"""
    ind_dir = os.path.join(raw_root, f"{ind_name}I")
    raw = os.path.join(ind_dir, f"{quarter}_{ind_name}I.csv")
    mat = os.path.join(ind_dir, f"{quarter}_{ind_name}I_stock_matrix.csv")
    if not os.path.exists(raw):
        return None, "missing_financial_csv"
    if not os.path.exists(mat):
        return None, "missing_stock_matrix"

    df = pd.read_csv(raw, encoding="utf-8-sig")
    n_raw = int(len(df))
    if n_raw == 0:
        return None, "empty_financial_csv"

    for c in ["n_income", "total_assets", "total_liab", "revenue",
              "operating_revenue", "operating_costs", "operating_expense"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_num)

    rev = df.get("operating_revenue") if "operating_revenue" in df.columns else df.get("revenue")
    cost = df.get("operating_costs") if "operating_costs" in df.columns else df.get("operating_expense")
    if rev is None or cost is None:
        return None, "missing_rev_or_cost_col"

    # 第 5 条：严格剔除 revenue <= 0 或 cost <= 0
    if params.get("require_cost_gt_zero", True):
        mask = (rev > 0) & (cost > 0)
    else:
        mask = (rev > 0) & (cost >= 0)
    n_dropped_rev_cost = int((~mask).sum())
    df = df[mask].copy()
    if len(df) < 2:
        return None, f"lt2_after_cleaning(n={len(df)})"

    df["n_income"] = df["n_income"].fillna(0.0)
    df["order_book_id"] = df["order_book_id"].astype(str).str.strip()

    W, had_row, had_col = read_stock_matrix_safe(mat, strip_mode=strip_mode)
    valid = [cid for cid in df["order_book_id"] if cid in W.index and cid in W.columns]
    if len(valid) < 2:
        return None, f"lt2_after_matrix_align(n={len(valid)})"

    df = df[df["order_book_id"].isin(valid)].drop_duplicates(subset=["order_book_id"]).set_index("order_book_id")
    ids = list(df.index)
    W = W.loc[ids, ids].astype(float)

    liab = _to_num(df["total_liab"]).fillna(0.0)
    assets = _to_num(df["total_assets"]).fillna(0.0)
    assets_safe = assets.replace(0, np.nan)
    income = _to_num(df["n_income"]).fillna(0.0)
    exp = _to_num(df["operating_costs"] if "operating_costs" in df.columns else df["operating_expense"]).fillna(0.0)
    rv = _to_num(df["operating_revenue"] if "operating_revenue" in df.columns else df["revenue"]).replace(0, np.nan)

    # VS 评分（第 5 条明确：VS 评分中的营业成本率 cost/rev 保持不变）
    df["L"] = liab / assets_safe
    df["k_ratio"] = exp / rv
    df["tau_raw"] = income / rv
    df["IA"] = income / assets_safe

    for c in ["L", "k_ratio", "tau_raw", "IA"]:
        s = df[c].fillna(df[c].median()).fillna(0.0)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lower=lo, upper=hi)
        df["z_" + c] = _z_score(s)
    df["VS_score"] = df["z_L"] + df["z_k_ratio"] - df["z_tau_raw"] - df["z_IA"]

    # 第 5 条：公式 1a 中的 k_b = revenue / operating_expense（>1）
    k_mode = params.get("k_mode", "paper_rev_over_cost")
    if k_mode == "paper_rev_over_cost":
        df["k_agent"] = (rv / exp.replace(0, np.nan)).fillna(1.0)
    else:
        df["k_agent"] = (exp / rv).fillna(0.0)

    # 第 4 条：利润率符号（V1 净利润率 max(tau, 0)；V2 毛利率 max(1 - cost/rev, 0)）
    margin_mode = params.get("margin_mode", "v1_net_nonneg")
    if margin_mode == "v2_gross_nonneg":
        raw_margin = 1.0 - (exp / rv)
        n_tau_clipped = int((raw_margin < 0).sum())
        df["tau_agent"] = raw_margin.clip(lower=0.0).fillna(0.0)
    elif margin_mode == "legacy_raw":
        n_tau_clipped = 0
        df["tau_agent"] = df["tau_raw"].fillna(0.0)
    else:  # v1_net_nonneg
        n_tau_clipped = int((df["tau_raw"] < 0).sum())
        df["tau_agent"] = df["tau_raw"].clip(lower=0.0).fillna(0.0)

    # 第 11 条：高危校准分位 γ -> θ_k -> 初始破产集合 B(0)
    df["survival_value"] = income.copy()  # 供 Huber 拟合 n_income / total_liab 使用
    theta = fit_huber_theta(df, gamma_q)
    is_initial_gamma = (liab * theta > income)

    # 第 1 条：缓冲与阈值设定 + 第 11 条外生缓冲冲击 δ
    buffer_mode = params.get("buffer_mode", "equity")
    delta_shock = float(params.get("delta_shock", 0.0))
    if buffer_mode == "equity":
        # 主设定：C_i(0) = 所有者权益 = total_assets - total_liab，U_i = 0
        raw_buf = assets - liab
        init_buf = np.where(raw_buf > 0, raw_buf * (1.0 - delta_shock), raw_buf)
        threshold = np.zeros(len(df), dtype=float)
    elif buffer_mode == "total_assets":
        # 稳健性设定②：C_i(0) = total_assets, U_i = asset_threshold_ratio * total_assets
        ratio_u = float(params.get("asset_threshold_ratio", 0.5))
        init_buf = assets.values * (1.0 - delta_shock)
        threshold = assets.values * ratio_u
    else:
        # 兼容旧版：C_i(0) = n_income, U_i = liab * theta
        init_buf = income.values * (1.0 - delta_shock)
        threshold = (liab * theta).values

    df["init_buffer"] = init_buf
    df["threshold_buf"] = threshold
    is_insolvent = (df["init_buffer"] <= df["threshold_buf"])
    df["is_bankrupt"] = (is_initial_gamma | is_insolvent).astype(int)

    n_init_gamma = int(is_initial_gamma.sum())
    n_init_insolvent = int(is_insolvent.sum())
    n_bankrupt0 = int(df["is_bankrupt"].sum())
    br0 = float(n_bankrupt0 / len(df))

    # 稀疏化网络用于 WCC（Fagiolo）与可视度 vis_b
    disparity_alpha = float(params.get("disparity_alpha", 0.05))
    Ws = disparity_filter_directed(W, disparity_alpha)
    wcc = compute_fagiolo_wcc(Ws)

    vis_mode = params.get("vis_mode", "betweenness")
    if vis_mode == "betweenness":
        es = Ws.stack().reset_index()
        es.columns = ["s", "t", "w"]
        es = es[(es["w"] > 0) & (es["s"] != es["t"])]
        if not es.empty:
            Gs = nx.from_pandas_edgelist(es, "s", "t", edge_attr="w", create_using=nx.DiGraph)
            vis_d = node_visibility(Gs)
            df["vis"] = df.index.map(vis_d).fillna(0.0)
        else:
            df["vis"] = 0.0
    elif vis_mode == "zero":
        df["vis"] = 0.0
    else:
        df["vis"] = 1.0

    # 第 2 条：边方向 a->b 表示 a 向 b 供货；可选是否反转矩阵
    A_raw = Ws.values.astype(float) if params.get("sim_network", "full") == "sparse" else W.values.astype(float)
    if params.get("transpose_edges", False):
        A_sim = A_raw.T.copy()
    else:
        A_sim = A_raw.copy()
    np.fill_diagonal(A_sim, 0.0)

    lev = float((liab / assets_safe).mean())
    return {
        "df": df,
        "A": A_sim,
        "ids": ids,
        "wcc": wcc,
        "br0": br0,
        "n_bankrupt0": n_bankrupt0,
        "n_init_gamma": n_init_gamma,
        "n_init_insolvent": n_init_insolvent,
        "n_tau_clipped": n_tau_clipped,
        "n_dropped_rev_cost": n_dropped_rev_cost,
        "theta": theta,
        "lev": lev,
        "n_raw": n_raw,
        "n": len(df),
        "had_summary_row": had_row,
        "had_summary_col": had_col,
    }, "ok"


# ---------------------------------------------------------------------------
# 核心仿真器（纯矩阵/数组状态，支持同步更新与旧版异步更新，便于单元测试）
# ---------------------------------------------------------------------------
def simulate_network_v2(
    A: np.ndarray,
    k_arr: np.ndarray,
    tau_arr: np.ndarray,
    vis_arr: np.ndarray,
    init_buf: np.ndarray,
    threshold: np.ndarray,
    is_initial: np.ndarray,
    params: dict,
    seed: int = 42,
    propagation_enabled: bool = True,
) -> dict:
    """
    在给定有向加权矩阵 A（A[a, b] = V_ab，a 为 b 的供应商）上执行一次破产传染仿真。
    直接接收 numpy 数组，方便 2-3 节点单元测试精确核对。
    """
    rng = random.Random(seed)
    N = int(len(init_buf))

    p0 = float(params.get("p0", 0.5))
    phi = float(params.get("phi", 0.5))
    s_max = int(params.get("tmax", 1000))
    vis_formula = params.get("vis_formula", "paper")
    alpha_p = float(params.get("alpha_p", 0.5))
    gamma_v = float(params.get("gamma_v", 1.0))
    direct_delta_s = bool(params.get("direct_delta_s", True))
    ego_mode = params.get("ego_mode", "all_neighbors")
    secondary_mode = params.get("secondary_impact_mode", "geometric_pb")
    healthy_pass_mode = params.get("healthy_pass_mode", "v1_random_single")
    update_mode = params.get("update_mode", "synchronous")
    stop_mode = params.get("stop_mode", "dual_criterion")
    stop_tol = float(params.get("stop_tol", 1e-6))

    # 预计算上下游邻居表与总入/出边权
    up_nbrs: List[List[int]] = [[] for _ in range(N)]
    dn_nbrs: List[List[int]] = [[] for _ in range(N)]
    sum_in = np.zeros(N, dtype=float)
    sum_out = np.zeros(N, dtype=float)
    for a in range(N):
        for b in range(N):
            w = float(A[a, b])
            if a != b and w > 0.0:
                dn_nbrs[a].append(b)
                up_nbrs[b].append(a)
                sum_out[a] += w
                sum_in[b] += w

    vis_fac_arr = np.array(
        [compute_vis_factor(float(vis_arr[i]), vis_formula, alpha_p, gamma_v) for i in range(N)],
        dtype=float,
    )

    buf = init_buf.astype(float).copy()
    broken = is_initial.astype(bool).copy()
    new_broken = is_initial.astype(bool).copy()
    gen = np.where(broken, 0, -1)

    # 初始破产企业：在 t=0 标记为破产，其缓冲视为已耗尽（用于与传播损失区分）
    # 注意：第 12 条直接损失与传播损失均可通过 init_buf 与 final_buf 严格拆分
    total_pos_init_buf = float(np.sum(np.maximum(init_buf, 0.0)))
    direct_loss_ratio = (
        float(np.sum(np.maximum(init_buf[broken], 0.0)) / total_pos_init_buf)
        if total_pos_init_buf > 0.0 else float("nan")
    )

    # 若传播关闭，直接返回 t=0 状态
    if not propagation_enabled:
        final_buf = np.where(broken, np.minimum(buf, threshold), buf)
        n_init = int(broken.sum())
        br0 = n_init / float(N)
        loss_r = compute_loss_ratio_v2(init_buf, final_buf)
        return {
            "BR0": br0,
            "BRT": br0,
            "dBR": 0.0,
            "lambda": 0.0 if br0 < 1.0 else float("nan"),
            "Direct_Loss_Ratio": direct_loss_ratio,
            "Loss_Ratio": loss_r,
            "Max_Prop_Gen": 0,
            "N_firms": N,
            "N_initial": n_init,
            "N_final": n_init,
            "N_secondary": 0,
            "T_last_activation": 0,
            "T_loss_stop": 0,
            "T_run": 0,
            "converged": True,
            "final_buffers": final_buf,
            "final_broken": broken,
        }

    # 待执行的二次影响队列：每个节点记录上一步受到的有效冲击列表 [(magnitude, origin_gen), ...]
    # 第 6 条：若上一步受到该冲击（或其二次影响），则本步以概率 p_b 再次受到 phi * ||F||
    prev_impacts: List[List[Tuple[float, int]]] = [[] for _ in range(N)]
    # 健康节点上一步因公式 1a 产生的待下传销售损失 ΔS：[(delta_s, origin_gen), ...]
    prev_delta_s: List[List[Tuple[float, int]]] = [[] for _ in range(N)]

    t_last_act = 0
    t_loss_stop = 0
    t_run = 0
    no_new_streak = 0
    converged = False

    for step in range(1, s_max + 1):
        t_run = step
        broken_prev = broken.copy()
        new_broken_prev = new_broken.copy()
        broken_prev_list = broken_prev.tolist()

        step_force = np.zeros(N, dtype=float)
        step_hit_gen = np.full(N, -1, dtype=int)
        next_impacts: List[List[Tuple[float, int]]] = [[] for _ in range(N)]
        next_delta_s: List[List[Tuple[float, int]]] = [[] for _ in range(N)]

        def _record_hit(target: int, force_val: float, src_gen: int, add_to_secondary: bool = True) -> None:
            if force_val <= 0.0:
                return
            step_force[target] += force_val
            if step_hit_gen[target] < 0 or src_gen < step_hit_gen[target]:
                step_hit_gen[target] = src_gen
            if add_to_secondary:
                next_impacts[target].append((force_val, src_gen))

        # 1) 来自上一步新破产节点的直接冲击（公式 1a 与 1b）
        for a in range(N):
            if not new_broken_prev[a]:
                continue
            src_g = int(gen[a])
            # 1a: 供应商 a 破产 -> 下游客户 b 受冲击
            for b in dn_nbrs[a]:
                if broken_prev[b]:
                    continue
                v_ab = float(A[a, b])
                share_up = v_ab / sum_in[b] if sum_in[b] > 0 else 0.0
                ego_r = compute_ego_broken_ratio(b, up_nbrs[b], dn_nbrs[b], broken_prev_list, ego_mode, "1a")
                f_ab, ds_ab = compute_force_1a_and_delta_s(
                    v_ab, float(k_arr[b]), float(tau_arr[b]),
                    float(vis_fac_arr[b]), ego_r, share_up, direct_delta_s,
                )
                if f_ab > 0.0:
                    _record_hit(b, f_ab, src_g, add_to_secondary=True)
                if ds_ab > 0.0:
                    next_delta_s[b].append((ds_ab, src_g))

            # 1b: 客户 a 破产 -> 上游供应商 b 受冲击
            for b in up_nbrs[a]:
                if broken_prev[b]:
                    continue
                v_ba = float(A[b, a])
                share_dn = v_ba / sum_out[b] if sum_out[b] > 0 else 0.0
                ego_r = compute_ego_broken_ratio(b, up_nbrs[b], dn_nbrs[b], broken_prev_list, ego_mode, "1b")
                f_ba = compute_force_1b(
                    v_ba, float(tau_arr[b]), float(vis_fac_arr[b]), ego_r, share_dn,
                )
                if f_ba > 0.0:
                    _record_hit(b, f_ba, src_g, add_to_secondary=True)

        # 2) 健康节点下传（第 7 条：上一步收到 ΔS 且仍健康的节点，向下游传递销售损失，总量 <= ΔS）
        for b in range(N):
            if broken_prev[b] or not prev_delta_s[b]:
                continue
            healthy_dns = [c for c in dn_nbrs[b] if not broken_prev[c]]
            if not healthy_dns:
                continue
            for ds_val, src_g in prev_delta_s[b]:
                if ds_val <= 0.0:
                    continue
                if healthy_pass_mode == "v2_weighted_all":
                    # V2：按边权把 ΔS 分配给全部健康下游，下传总量严格等于 ΔS（<= ΔS）
                    w_tot = sum(float(A[b, c]) for c in healthy_dns)
                    if w_tot <= 0.0:
                        continue
                    for c in healthy_dns:
                        share_c = float(A[b, c]) / w_tot
                        ds_c = ds_val * share_c
                        f_c = ds_c * max(float(tau_arr[c]), 0.0)
                        if f_c > 0.0:
                            _record_hit(c, f_c, src_g, add_to_secondary=False)
                else:
                    # V1（论文版）：随机选一个健康下游，下传量 = min(ds_val, ds_val * (A[b,c]/sum_out[b])) <= ds_val
                    c = rng.choice(healthy_dns)
                    w_bc = float(A[b, c])
                    frac = min(1.0, w_bc / sum_out[b]) if sum_out[b] > 0 else 1.0
                    ds_c = ds_val * frac  # 严格保证 ds_c <= ds_val
                    f_c = ds_c * max(float(tau_arr[c]), 0.0)
                    if f_c > 0.0:
                        _record_hit(c, f_c, src_g, add_to_secondary=False)

        # 3) 二次影响（第 6 条）：上步受冲击的节点，以概率 p_b = p0 * ego_ratio 再受 phi * ||F||
        pending_secondary_total = 0.0
        for b in range(N):
            if broken_prev[b] or not prev_impacts[b]:
                continue
            if secondary_mode == "geometric_pb":
                ego_r = compute_ego_broken_ratio(b, up_nbrs[b], dn_nbrs[b], broken_prev_list, ego_mode, "1a")
                p_b = min(max(p0 * ego_r, 0.0), 1.0)
            else:
                p_b = min(max(p0, 0.0), 1.0)

            for f_orig, src_g in prev_impacts[b]:
                sec_mag = phi * f_orig
                if sec_mag <= 0.0:
                    continue
                pending_secondary_total += sec_mag
                if p_b > 0.0 and rng.random() < p_b:
                    # 第 6 条：每次存活均受 φ·||F||（保持原始 ||F||），故理论期望严格为 φ·||F||·p_b/(1 - p_b)
                    _record_hit(b, sec_mag, src_g, add_to_secondary=False)
                    next_impacts[b].append((f_orig, src_g))

        # 4) 统一扣减与统一判定激活（第 8 条：同步更新）
        step_loss_applied = float(np.sum(step_force[~broken_prev]))
        if step_loss_applied > 0.0:
            t_loss_stop = step

        buf[~broken_prev] -= step_force[~broken_prev]
        newly_activated = (~broken_prev) & (buf <= threshold)

        new_broken = newly_activated.copy()
        broken = broken_prev | newly_activated
        for i in range(N):
            if newly_activated[i]:
                gen[i] = int(step_hit_gen[i]) + 1 if step_hit_gen[i] >= 0 else 1

        n_new = int(newly_activated.sum())
        if n_new > 0:
            t_last_act = step
            no_new_streak = 0
        else:
            no_new_streak += 1

        # 更新下一步待传播状态（已破产节点不再向下传递健康下传或接收二次影响）
        prev_impacts = [next_impacts[i] if not broken[i] else [] for i in range(N)]
        prev_delta_s = [next_delta_s[i] if not broken[i] else [] for i in range(N)]

        # 5) 停止规则（第 9 条：连 3 步无新增激活 且 待执行二次影响总额 < 1e-6 * 总缓冲）
        next_pending_total = sum(
            phi * f_val for i in range(N) for (f_val, _) in prev_impacts[i]
        ) + sum(
            ds_val for i in range(N) for (ds_val, _) in prev_delta_s[i]
        )

        if stop_mode == "dual_criterion":
            tol_thresh = stop_tol * max(total_pos_init_buf, 1.0)
            if no_new_streak >= 3 and next_pending_total < tol_thresh and n_new == 0:
                converged = True
                break
        else:
            if n_new == 0 and next_pending_total == 0.0:
                converged = True
                break

    # 汇总指标（第 12 条）
    # 对初始即破产企业，缓冲消耗按初始正缓冲全部损失计；对非初始破产或受损健康企业，按实际剩余缓冲计
    final_buf = buf.copy()
    final_buf[is_initial] = np.minimum(final_buf[is_initial], threshold[is_initial])

    n_init = int(is_initial.sum())
    n_final = int(broken.sum())
    n_sec = n_final - n_init
    br0 = n_init / float(N)
    brt = n_final / float(N)
    dbr = brt - br0
    lam = (dbr / (1.0 - br0)) if br0 < 1.0 else float("nan")
    loss_ratio = compute_loss_ratio_v2(init_buf, final_buf)
    max_gen = int(np.max(gen)) if n_final > 0 else 0

    return {
        "BR0": br0,
        "BRT": brt,
        "dBR": dbr,
        "lambda": lam,
        "Direct_Loss_Ratio": direct_loss_ratio,
        "Loss_Ratio": loss_ratio,
        "Max_Prop_Gen": max_gen,
        "N_firms": N,
        "N_initial": n_init,
        "N_final": n_final,
        "N_secondary": n_sec,
        "T_last_activation": t_last_act,
        "T_loss_stop": t_loss_stop,
        "T_run": t_run,
        "converged": converged,
        "final_buffers": final_buf,
        "final_broken": broken,
    }


def run_single_cell_v2(
    raw_root: str,
    ind_name: str,
    quarter: str,
    gamma_q: float,
    params: dict,
    strip_mode: str = "auto_by_name",
    propagation_enabled: bool = True,
) -> Tuple[Optional[dict], List[int], str]:
    """
    运行单个 (行业, 季度) 单元（支持 n_repeats >= 1，以及 V2.2 自动扩充至 200 次）。
    """
    prep, reason = prepare_quarter_v2(raw_root, ind_name, quarter, gamma_q, params, strip_mode)
    if prep is None:
        return None, [], reason

    df = prep["df"]
    A = prep["A"]
    k_arr = df["k_agent"].values.astype(float)
    tau_arr = df["tau_agent"].values.astype(float)
    vis_arr = df["vis"].values.astype(float)
    init_buf = df["init_buffer"].values.astype(float)
    threshold = df["threshold_buf"].values.astype(float)
    is_initial = (df["is_bankrupt"].values == 1)

    n_repeats = int(params.get("n_repeats", 1))
    auto_expand_se = float(params.get("auto_expand_se", 0.0))
    max_repeats = int(params.get("max_repeats", 200))
    seed_salt = int(params.get("seed_salt", 0))

    seeds: List[int] = []
    rep_rows: List[dict] = []

    p0 = float(params.get("p0", 0.5))
    phi = float(params.get("phi", 0.5))

    def _run_batch(start_r: int, end_r: int) -> None:
        for r in range(start_r, end_r):
            sd = quarter_seed(ind_name, quarter, gamma_q, p0, phi, salt=seed_salt + r * 10007)
            seeds.append(sd)
            res = simulate_network_v2(
                A, k_arr, tau_arr, vis_arr, init_buf, threshold, is_initial,
                params=params, seed=sd, propagation_enabled=propagation_enabled,
            )
            rep_rows.append(res)

    _run_batch(0, n_repeats)
    if auto_expand_se > 0.0 and n_repeats >= 2 and max_repeats > n_repeats:
        brt_vals = np.array([x["BRT"] for x in rep_rows], dtype=float)
        se = float(np.std(brt_vals, ddof=1) / math.sqrt(len(brt_vals)))
        if se > auto_expand_se:
            _run_batch(n_repeats, max_repeats)

    brt_arr = np.array([x["BRT"] for x in rep_rows], dtype=float)
    dbr_arr = np.array([x["dBR"] for x in rep_rows], dtype=float)
    loss_arr = np.array([x["Loss_Ratio"] for x in rep_rows], dtype=float)
    lam_arr = np.array([x["lambda"] for x in rep_rows], dtype=float)
    gen_arr = np.array([x["Max_Prop_Gen"] for x in rep_rows], dtype=float)

    se_brt = float(np.std(brt_arr, ddof=1) / math.sqrt(len(brt_arr))) if len(brt_arr) > 1 else 0.0

    out = {
        "Quarter": quarter,
        "Industry": ind_name,
        "gamma": gamma_q,
        "Propagation": int(propagation_enabled),
        "N_repeats": len(rep_rows),
        "BR0": prep["br0"],
        "BRT": float(np.mean(brt_arr)),
        "BRT_p05": float(np.percentile(brt_arr, 5)),
        "BRT_p95": float(np.percentile(brt_arr, 95)),
        "BRT_SE": se_brt,
        "dBR": float(np.mean(dbr_arr)),
        "lambda": float(np.nanmean(lam_arr)) if np.any(np.isfinite(lam_arr)) else float("nan"),
        "Direct_Loss_Ratio": float(rep_rows[0]["Direct_Loss_Ratio"]),
        "Loss_Ratio": float(np.nanmean(loss_arr)) if np.any(np.isfinite(loss_arr)) else float("nan"),
        "Loss_Ratio_p05": float(np.nanpercentile(loss_arr, 5)) if np.any(np.isfinite(loss_arr)) else float("nan"),
        "Loss_Ratio_p95": float(np.nanpercentile(loss_arr, 95)) if np.any(np.isfinite(loss_arr)) else float("nan"),
        "Max_Prop_Gen": float(np.mean(gen_arr)),
        "Max_Prop_Gen_max": int(np.max(gen_arr)),
        "N_firms": prep["n"],
        "N_initial": prep["n_bankrupt0"],
        "N_init_gamma": prep["n_init_gamma"],
        "N_init_insolvent": prep["n_init_insolvent"],
        "N_tau_clipped": prep["n_tau_clipped"],
        "N_dropped_rev_cost": prep["n_dropped_rev_cost"],
        "N_final": float(np.mean([x["N_final"] for x in rep_rows])),
        "N_secondary": float(np.mean([x["N_secondary"] for x in rep_rows])),
        "T_last_activation": float(np.mean([x["T_last_activation"] for x in rep_rows])),
        "T_loss_stop": float(np.mean([x["T_loss_stop"] for x in rep_rows])),
        "T_run": float(np.mean([x["T_run"] for x in rep_rows])),
        "All_Converged": int(all(x["converged"] for x in rep_rows)),
        "WCC": prep["wcc"],
        "Leverage": prep["lev"],
        "theta": prep["theta"],
        "seed_first": seeds[0],
    }
    return out, seeds, "ok"
