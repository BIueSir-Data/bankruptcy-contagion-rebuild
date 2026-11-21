# -*- coding: utf-8 -*-
"""
engine_legacy_d1.py —— D0.6 与 D1 专用仿真引擎
================================================
严格遵循导师指南要求：“D1、D2 先沿用 11.16 版现有的网络与代码，企业级网络完成后再在新网络上重跑。”
在保持 8.31/11.16 现有核心算法不变的前提下，按通用规范做了以下工程重构：
  1. 去除全部硬编码全局常量，所有参数由 YAML 配置字典传入；
  2. 新增 propagation_enabled (True/False) 开关，确保传播开/关共享同一 B(0) 与同一种子；
  3. 新增传播代数追踪 (Max_Prop_Gen：最远被激活企业距 B(0) 的跳数)；
  4. 拆出 Direct_Loss_Ratio（初始破产组正缓冲占比）与 Prop_Added_Loss_Ratio（传播新增损失率）；
  5. 分母为 0 或缺失值统一记 np.nan（CSV 导出为 NA），不以 0 替代。
"""
import os
import math
import random
import warnings

import numpy as np
import pandas as pd
import networkx as nx

from common import quarter_seed, read_stock_matrix_safe

warnings.filterwarnings("ignore")

EPS = 1e-8


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _z_score(s):
    std = s.std()
    if std == 0 or pd.isna(std):
        return np.zeros(len(s))
    return (s - s.mean()) / std


def _huber_irls(x, y, eps=1.35, iters=60):
    X = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        resid = y - X @ beta
        s = max(float(np.median(np.abs(resid))) * 1.4826, 1e-9)
        z = resid / s
        w = np.where(np.abs(z) <= eps, 1.0, eps / np.maximum(np.abs(z), 1e-12))
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X + 1e-8 * np.eye(2), X.T @ W @ y)
    return float(beta[0]), float(beta[1])


def fit_huber_theta(df, gamma_q, epsilon=1.35):
    liab = pd.to_numeric(df["total_liab"], errors="coerce").replace(0, np.nan)
    r = pd.to_numeric(df["survival_value"], errors="coerce") / liab
    d = pd.DataFrame({"VS": pd.to_numeric(df["VS_score"], errors="coerce"), "r": r})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return float(d["r"].median()) if len(d) else 0.05
    d = d.sort_values("VS", ascending=False).reset_index(drop=True)
    N = len(d)
    x = (np.arange(N) + 1.0) / float(N)
    y = d["r"].values
    try:
        from sklearn.linear_model import HuberRegressor
        m = HuberRegressor(epsilon=epsilon, alpha=1e-4, max_iter=300)
        m.fit(x.reshape(-1, 1), y)
        beta, alpha0 = float(m.coef_[0]), float(m.intercept_)
    except Exception:
        beta, alpha0 = _huber_irls(x, y, epsilon)
    return float(beta * float(gamma_q) + alpha0)


def disparity_filter_directed(W: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    Wm = W.values
    N = Wm.shape[0]
    k_out = (Wm > 0).sum(1)
    k_in = (Wm > 0).sum(0)
    s_out = Wm.sum(1)
    s_in = Wm.sum(0)
    Ws = np.zeros_like(Wm)
    for i in range(N):
        for j in range(N):
            if Wm[i, j] == 0 or i == j:
                continue
            keep = False
            if s_out[i] > 0 and k_out[i] > 1:
                f = Wm[i, j] / s_out[i]
                if 0 < f < 1 and (1 - f) ** (k_out[i] - 1) < alpha:
                    keep = True
            if s_in[j] > 0 and k_in[j] > 1:
                f = Wm[i, j] / s_in[j]
                if 0 < f < 1 and (1 - f) ** (k_in[j] - 1) < alpha:
                    keep = True
            if keep:
                Ws[i, j] = Wm[i, j]
    return pd.DataFrame(Ws, index=W.index, columns=W.columns)


def node_visibility(G, epsilon=EPS):
    if G is None or G.number_of_nodes() == 0:
        return {}
    for _, _, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        d["distance"] = 1.0 / w if (w and w > 0) else float("inf")
    cb = nx.betweenness_centrality(G, weight="distance", normalized=False)
    vals = list(cb.values())
    med = float(np.median(vals)) if vals else 0.0
    return {n: (c / (med + epsilon)) for n, c in cb.items()}


def vis_factor(v: float, params: dict) -> float:
    if params.get("force_mode", "8.31") == "8.31":
        return math.exp(-v)
    alpha_p = float(params.get("alpha_p", 0.5))
    gamma_v = float(params.get("gamma_v", 3.0))
    return alpha_p * math.exp(-gamma_v * v) + (1.0 - alpha_p)


def prepare_quarter(raw_root: str, ind_name: str, quarter: str, gamma_q: float,
                    params: dict, strip_mode: str = "auto_by_name"):
    """准备单行业-单季度数据与网络；无效窗口返回 (None, reason)。"""
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

    mask = (rev > 0) & (cost >= 0)
    df = df[mask].copy()
    if len(df) < 2:
        return None, f"lt2_after_cleaning(n={len(df)})"

    # D0/D1 沿用现行 8.31 逻辑
    df["n_income"] = df["n_income"].fillna(0)
    df["order_book_id"] = df["order_book_id"].astype(str).str.strip()

    W, had_row, had_col = read_stock_matrix_safe(mat, strip_mode=strip_mode)
    valid = [cid for cid in df["order_book_id"] if cid in W.index and cid in W.columns]
    if len(valid) < 2:
        return None, f"lt2_after_matrix_align(n={len(valid)})"

    df = df[df["order_book_id"].isin(valid)].drop_duplicates(subset=["order_book_id"]).set_index("order_book_id")
    ids = list(df.index)
    W = W.loc[ids, ids].astype(float)

    liab = _to_num(df["total_liab"]).fillna(0)
    assets = _to_num(df["total_assets"]).fillna(0).replace(0, np.nan)
    income = _to_num(df["n_income"]).fillna(0)
    exp = _to_num(df.get("operating_costs", df.get("operating_expense"))).fillna(0)
    rv = _to_num(df.get("revenue")).replace(0, np.nan)

    df["L"] = liab / assets
    df["k_ratio"] = exp / rv
    tau_col = "tao" if "tao" in df.columns else ("tau" if "tau" in df.columns else None)
    df["tau"] = _to_num(df[tau_col]) if tau_col else (income / rv)
    df["IA"] = income / assets
    k_col = "k" if "k" in df.columns else "k_ratio"
    df["k_agent"] = _to_num(df[k_col])

    for c in ["L", "k_ratio", "tau", "IA"]:
        df[c] = df[c].fillna(df[c].median()).fillna(0)
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        df[c] = df[c].clip(lower=lo, upper=hi)
        df["z_" + c] = _z_score(df[c])
    df["VS_score"] = df["z_L"] + df["z_k_ratio"] - df["z_tau"] - df["z_IA"]

    df["survival_value"] = df["n_income"].copy()
    if params.get("theta_mode", "huber") == "huber":
        theta = fit_huber_theta(df, gamma_q)
    else:
        vs_thr = df["VS_score"].quantile(1 - gamma_q)
        top = df[df["VS_score"] >= vs_thr]
        sp = float(top["n_income"].sum())
        sl = float(top["total_liab"].sum())
        theta = sp / sl if (sl != 0 and not pd.isna(sl)) else 0.05

    df["threshold"] = liab * theta
    df["is_bankrupt"] = (df["threshold"] > df["survival_value"]).astype(int)
    br0 = float(df["is_bankrupt"].mean())
    n_bankrupt0 = int(df["is_bankrupt"].sum())

    disparity_alpha = float(params.get("disparity_alpha", 0.05))
    Ws = disparity_filter_directed(W, disparity_alpha)
    es = Ws.stack().reset_index()
    es.columns = ["s", "t", "w"]
    es = es[es["w"] > 0]
    wcc = 0.0
    vis_d = {}
    vis_mode = params.get("vis_mode", "zero")
    if not es.empty:
        Gs = nx.from_pandas_edgelist(es, "s", "t", edge_attr="w", create_using=nx.DiGraph)
        cl = nx.clustering(Gs, weight="weight")
        wcc = float(np.mean(list(cl.values()))) if cl else 0.0
        if vis_mode == "betweenness":
            vis_d = node_visibility(Gs)

    if vis_mode == "betweenness":
        df["vis"] = df.index.map(vis_d).fillna(0.0)
    elif vis_mode == "zero":
        df["vis"] = 0.0
    else:
        df["vis"] = 1.0

    A_sim = Ws.values.astype(float) if params.get("sim_network", "full") == "sparse" else W.values.astype(float)
    lev = float((liab / assets).mean())

    return {
        "df": df, "A": A_sim, "ids": ids, "W": W, "Ws": Ws, "wcc": wcc,
        "br0": br0, "n_bankrupt0": n_bankrupt0, "theta": theta, "lev": lev,
        "n_raw": n_raw, "n": len(df), "had_summary_row": had_row, "had_summary_col": had_col,
    }, "ok"


class Agent:
    __slots__ = (
        "pos", "up", "dn", "k", "tau", "vis", "survival", "threshold",
        "healthy", "new_broken", "broken", "is_initial",
        "recv_up", "recv_dn", "recv_hl", "prev_up", "imp_up", "imp_dn",
        "gen", "impact_gen", "pending_gen", "last_hit_gen",
    )

    def __init__(self, pos, up, dn, k, tau, vis, survival, threshold, is_initial=False):
        self.pos = pos
        self.up = up
        self.dn = dn
        self.k = k
        self.tau = tau
        self.vis = vis
        self.survival = survival
        self.threshold = threshold
        self.healthy = True
        self.new_broken = False
        self.broken = False
        self.is_initial = is_initial
        self.recv_up = {}
        self.recv_dn = {}
        self.recv_hl = {}
        self.prev_up = {}
        self.imp_up = {}
        self.imp_dn = {}
        # 传播代数追踪：初始破产节点 gen=0；未激活为 None
        self.gen = 0 if is_initial else None
        self.impact_gen = None
        self.pending_gen = None
        self.last_hit_gen = None

    @property
    def bs(self):
        return self.broken or self.new_broken


def build_agents(P: dict) -> dict:
    df, A = P["df"], P["A"]
    n = len(df)
    ids = P["ids"]
    ags = {}
    for i, cid in enumerate(ids):
        dn = [j for j in range(n) if j != i and A[i, j] > 0]
        up = [j for j in range(n) if j != i and A[j, i] > 0]
        r = df.loc[cid]
        is_init = bool(int(r["is_bankrupt"]) == 1)
        ags[i] = Agent(
            i, up, dn,
            float(r["k_agent"]) if pd.notna(r["k_agent"]) else 1.0,
            float(r["tau"]) if pd.notna(r["tau"]) else 0.0,
            float(r["vis"]),
            float(r["survival_value"]),
            float(r["threshold"]),
            is_initial=is_init,
        )
    return ags


def _share_broken(ags, pos_list):
    if not pos_list:
        return 0.0
    c = sum(1 for j in pos_list if ags[j].bs)
    return c / len(pos_list)


def _update_pending_gen(target: Agent, src_gen):
    if src_gen is None:
        return
    cand = src_gen + 1
    if target.pending_gen is None or cand < target.pending_gen:
        target.pending_gen = cand


def simulate_quarter(P: dict, ags: dict, params: dict, propagation_enabled: bool = True) -> dict:
    """
    执行单季度仿真：
      - propagation_enabled=True : 沿用现行 8.31 传播逻辑
      - propagation_enabled=False: 仅激活初始集合 B(0)，不发生任何力传播
    """
    A = P["A"]
    n = len(ags)
    p0 = float(params.get("p0", 0.5))
    phi = float(params.get("phi", 0.5))
    tmax = int(params.get("tmax", 1000))
    step_order = params.get("step_order", "710")
    healthy_pass_mode = params.get("healthy_pass_mode", "original")

    init_vals = {i: max(0.0, ag.survival) for i, ag in ags.items()}
    total_init = sum(init_vals.values())
    direct_init_loss = sum(init_vals[i] for i, ag in ags.items() if ag.is_initial)
    direct_loss_ratio = (direct_init_loss / total_init) if total_init > 0 else np.nan

    if not propagation_enabled:
        # 传播关闭：仅 B(0) 破产，无二次激活、无传播损失、最大传播代数 = 0
        for ag in ags.values():
            if ag.survival < ag.threshold:
                ag.healthy = False
                ag.broken = True
                ag.gen = 0
        n_bs = sum(1 for a in ags.values() if a.bs)
        return {
            "brt": n_bs / n,
            "loss": 0.0 if total_init > 0 else np.nan,
            "direct_loss_ratio": direct_loss_ratio,
            "non_initial_loss_ratio": 0.0 if total_init > 0 else np.nan,
            "t_last": 0,
            "t_run": 0,
            "n_total": n,
            "n_bs": n_bs,
            "n_initial": int(P["n_bankrupt0"]),
            "n_secondary": 0,
            "max_prop_gen": 0,
        }

    step = 0
    last_new = 0

    def phase_apply():
        for a in ags.values():
            tot = sum(a.recv_up.values()) + sum(a.recv_dn.values()) + sum(a.recv_hl.values())
            if tot:
                a.survival -= tot
                if a.pending_gen is not None:
                    a.last_hit_gen = a.pending_gen
            if a.recv_up and a.pending_gen is not None:
                a.impact_gen = a.pending_gen
            elif not a.recv_up:
                a.impact_gen = None
            a.prev_up = dict(a.recv_up)
            a.recv_up.clear()
            a.recv_dn.clear()
            a.recv_hl.clear()
            a.pending_gen = None

    def phase_check():
        nonlocal new_this, last_new
        for a in ags.values():
            if a.survival < a.threshold:
                if a.healthy:
                    a.healthy = False
                    a.new_broken = True
                    new_this += 1
                    if a.is_initial:
                        a.gen = 0
                    elif a.gen is None:
                        a.gen = a.last_hit_gen if a.last_hit_gen is not None else 1
                elif a.new_broken:
                    a.new_broken = False
                    a.broken = True
            else:
                a.healthy = True
                a.new_broken = False
                a.broken = False
        if new_this > 0:
            last_new = step

    def phase_propagate():
        for i in list(ags.keys()):
            a = ags[i]
            if a.new_broken:
                imp = {}
                for j in a.dn:
                    b = ags[j]
                    if not b.healthy:
                        continue
                    colsum = float(A[:, j].sum())
                    v = A[i, j]
                    if colsum <= 0 or v <= 0 or b.tau == 0:
                        continue
                    F = v * b.k * b.tau * vis_factor(b.vis, params) * _share_broken(ags, b.dn) * (v / colsum)
                    if F > 0:
                        b.recv_up[j] = b.recv_up.get(j, 0.0) + F
                        _update_pending_gen(b, a.gen)
                        imp[j] = F
                a.imp_dn = {t: f for t, f in imp.items() if ags[t].healthy}

                imp = {}
                for j in a.up:
                    b = ags[j]
                    if not b.healthy:
                        continue
                    rowsum = float(A[j, :].sum())
                    v = A[j, i]
                    if rowsum <= 0 or v <= 0 or b.tau == 0:
                        continue
                    F = v * b.tau * vis_factor(b.vis, params) * _share_broken(ags, b.up) * (v / rowsum)
                    if F > 0:
                        b.recv_dn[j] = b.recv_dn.get(j, 0.0) + F
                        _update_pending_gen(b, a.gen)
                        imp[j] = F
                a.imp_up = {t: f for t, f in imp.items() if ags[t].healthy}

            elif a.broken:
                for t, f in list(a.imp_dn.items()):
                    if ags[t].healthy and random.random() < p0:
                        ags[t].recv_up[t] = ags[t].recv_up.get(t, 0.0) + f * phi
                        _update_pending_gen(ags[t], a.gen)
                    else:
                        a.imp_dn.pop(t, None)
                for t, f in list(a.imp_up.items()):
                    if ags[t].healthy and random.random() < p0:
                        ags[t].recv_dn[t] = ags[t].recv_dn.get(t, 0.0) + f * phi
                        _update_pending_gen(ags[t], a.gen)
                    else:
                        a.imp_up.pop(t, None)

            elif a.healthy and a.prev_up:
                total_up = sum(a.prev_up.values())
                if a.tau != 0 and total_up > 0 and a.dn:
                    delta_s = total_up / a.tau
                    if healthy_pass_mode == "single":
                        targets = [random.choice(a.dn)]
                    else:
                        pr = 0.8 / len(a.dn)
                        targets = [j for j in a.dn if random.random() < pr]
                    for j in targets:
                        d = ags[j]
                        if not (d.healthy and d.tau != 0):
                            continue
                        colsum = float(A[:, j].sum())
                        v = A[a.pos, j]
                        if colsum > 0 and v > 0 and delta_s > 0:
                            F = delta_s * d.k * d.tau * vis_factor(d.vis, params) * _share_broken(ags, d.dn) * (v / colsum)
                            if F > 0:
                                d.recv_hl[j] = d.recv_hl.get(j, 0.0) + F
                                _update_pending_gen(d, a.impact_gen if a.impact_gen is not None else 1)
                a.prev_up = {}

    while step < tmax:
        step += 1
        new_this = 0
        if step_order == "710":
            phase_propagate()
            phase_apply()
            phase_check()
        else:
            phase_apply()
            phase_check()
            phase_propagate()

    n_bs = sum(1 for a in ags.values() if a.bs)
    cur = {i: max(0.0, a.survival) for i, a in ags.items()}
    loss_sum = sum(max(0.0, init_vals[i] - cur[i]) for i in ags)
    non_init_loss_sum = sum(max(0.0, init_vals[i] - cur[i]) for i, a in ags.items() if not a.is_initial)

    loss_ratio = (loss_sum / total_init) if total_init > 0 else np.nan
    non_init_loss_ratio = (non_init_loss_sum / total_init) if total_init > 0 else np.nan
    gens = [a.gen for a in ags.values() if a.bs and a.gen is not None]
    max_prop_gen = int(max(gens)) if gens else 0

    return {
        "brt": n_bs / n,
        "loss": loss_ratio,
        "direct_loss_ratio": direct_loss_ratio,
        "non_initial_loss_ratio": non_init_loss_ratio,
        "t_last": last_new,
        "t_run": step,
        "n_total": n,
        "n_bs": n_bs,
        "n_initial": int(P["n_bankrupt0"]),
        "n_secondary": n_bs - int(P["n_bankrupt0"]),
        "max_prop_gen": max_prop_gen,
    }


def run_single_cell(raw_root: str, ind_name: str, quarter: str, gamma_q: float,
                    params: dict, strip_mode: str = "auto_by_name",
                    propagation_enabled: bool = True):
    """单格确定性仿真入口（多进程安全）。返回 (row_dict | None, seed, status_reason)。"""
    p0 = float(params.get("p0", 0.5))
    phi = float(params.get("phi", 0.5))
    salt = int(params.get("seed_salt", 0))
    seed = quarter_seed(ind_name, quarter, gamma_q, p0, phi, salt=salt)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))

    P, reason = prepare_quarter(raw_root, ind_name, quarter, gamma_q, params, strip_mode=strip_mode)
    if P is None:
        return None, seed, reason

    ags = build_agents(P)
    res = simulate_quarter(P, ags, params, propagation_enabled=propagation_enabled)
    br0 = P["br0"]
    brt = res["brt"]
    dbr = brt - br0
    lam = (dbr / (1.0 - br0)) if (1.0 - br0) > 1e-12 else np.nan

    return {
        "Quarter": quarter,
        "Industry": ind_name,
        "gamma": gamma_q,
        "Propagation": "on" if propagation_enabled else "off",
        "BR0": br0,
        "BRT": brt,
        "dBR": dbr,
        "lambda": lam,
        "Direct_Loss_Ratio": res["direct_loss_ratio"],
        "Loss_Ratio": res["loss"],
        "NonInitial_Loss_Ratio": res["non_initial_loss_ratio"],
        "Max_Prop_Gen": res["max_prop_gen"],
        "N_firms": P["n"],
        "N_initial": res["n_initial"],
        "N_final": res["n_bs"],
        "N_secondary": res["n_secondary"],
        "T_last": res["t_last"],
        "T_run": res["t_run"],
        "WCC": P["wcc"],
        "Leverage": P["lev"],
        "theta": P["theta"],
        "seed": seed,
    }, seed, "ok"
