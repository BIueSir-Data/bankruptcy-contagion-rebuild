# -*- coding: utf-8 -*-
"""
tests/test_engine_v2.py —— 冻结模型 2-3 节点手算单元测试（对应通用规范第 8 条 + 第 6 条要求）
================================================================================
运行方式（在 重建实验方案/ 目录下）：
    python tests/test_engine_v2.py
"""
from __future__ import annotations

import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from engine_v2 import (
    compute_ego_broken_ratio,
    compute_force_1a_and_delta_s,
    compute_force_1b,
    compute_loss_ratio_v2,
    compute_vis_factor,
    simulate_network_v2,
)


def test_vis_factor():
    # 第 3 条：alpha * exp(-gamma_v * vis) + (1 - alpha)
    v = compute_vis_factor(1.0, "paper", alpha_p=0.5, gamma_v=1.0)
    expected = 0.5 * math.exp(-1.0) + 0.5
    assert abs(v - expected) < 1e-12, f"vis_factor mismatch: {v} vs {expected}"
    assert abs(compute_vis_factor(0.0, "paper", 0.5, 1.0) - 1.0) < 1e-12


def test_force_1a_1b_and_no_div_zero():
    # 第 5 条：tau_b = 0 时不除零，ΔS > 0 且 |F_ab| = 0
    f0, ds0 = compute_force_1a_and_delta_s(
        v_ab=100.0, k_b=2.0, tau_b=0.0, vis_fac=0.8, ego_ratio=0.5, share_up=1.0, direct_delta_s=True
    )
    assert f0 == 0.0 and abs(ds0 - 80.0) < 1e-12

    # 手算 1a: V=100, k=1.5 (>1), tau=0.2, vis_fac=0.75, ego=0.5, share=0.6
    # ΔS = 100 * 1.5 * 0.75 * 0.5 * 0.6 = 33.75; |F| = 33.75 * 0.2 = 6.75
    f1, ds1 = compute_force_1a_and_delta_s(100.0, 1.5, 0.2, 0.75, 0.5, 0.6, True)
    assert abs(ds1 - 33.75) < 1e-12 and abs(f1 - 6.75) < 1e-12

    # 手算 1b: V=100, tau=0.2, vis_fac=0.75, ego=0.5, share=0.6 -> 100*0.2*0.75*0.5*0.6 = 4.5
    f2 = compute_force_1b(100.0, 0.2, 0.75, 0.5, 0.6)
    assert abs(f2 - 4.5) < 1e-12


def test_loss_ratio_v2():
    # 第 12 条：Σ(C_i(0) - C_i(T))^+ / ΣC_i(0)^+，分母为 0 记 NaN
    r = compute_loss_ratio_v2(np.array([100.0, 200.0, -50.0]), np.array([40.0, 250.0, -80.0]))
    # init_pos = [100, 200, 0] -> sum=300; loss = [60, 0, 0] -> sum=60 -> 0.2
    assert abs(r - 0.2) < 1e-12
    assert math.isnan(compute_loss_ratio_v2(np.array([-10.0, 0.0]), np.array([-20.0, -5.0])))


def test_2node_synchronous_contagion():
    # 2 节点手算：0 -> 1 (V_01 = 100)，0 初始破产，1 健康
    # 1 的参数：k=2.0, tau=0.25, vis=0 (vis_fac=1.0), init_buf=40, threshold=0
    # step 1: ego_ratio(1) = 1/1 = 1.0, share_up = 1.0
    #   ΔS = 100 * 2.0 * 1.0 * 1.0 * 1.0 = 200
    #   |F_01| = 200 * 0.25 = 50 -> buf(1) = 40 - 50 = -10 <= 0 -> 1 在第 1 步激活破产！
    A = np.array([[0.0, 100.0], [0.0, 0.0]])
    params = {
        "p0": 0.0, "phi": 0.5, "tmax": 50, "vis_formula": "paper",
        "alpha_p": 0.5, "gamma_v": 1.0, "direct_delta_s": True,
        "ego_mode": "all_neighbors", "secondary_impact_mode": "geometric_pb",
        "healthy_pass_mode": "v1_random_single", "update_mode": "synchronous",
        "stop_mode": "dual_criterion", "stop_tol": 1e-6,
    }
    res = simulate_network_v2(
        A=A,
        k_arr=np.array([1.5, 2.0]),
        tau_arr=np.array([0.1, 0.25]),
        vis_arr=np.array([0.0, 0.0]),
        init_buf=np.array([100.0, 40.0]),
        threshold=np.array([0.0, 0.0]),
        is_initial=np.array([True, False]),
        params=params,
        seed=42,
    )
    assert res["N_initial"] == 1 and res["N_final"] == 2
    assert res["Max_Prop_Gen"] == 1 and res["T_last_activation"] == 1
    assert abs(res["Loss_Ratio"] - 1.0) < 1e-12  # 两家缓冲全部耗尽


def test_3node_healthy_pass_bound():
    # 3 节点链 0 -> 1 -> 2，0 初始破产，1 缓冲很大保持健康，2 接收来自健康节点 1 的下传
    # 验证 V1 与 V2 下传总量严格 <= ΔS，且 2 正确扣减缓冲
    A = np.array([
        [0.0, 100.0, 0.0],
        [0.0, 0.0, 50.0],
        [0.0, 0.0, 0.0],
    ])
    for mode in ("v1_random_single", "v2_weighted_all"):
        params = {
            "p0": 0.0, "phi": 0.0, "tmax": 50, "vis_formula": "paper",
            "alpha_p": 0.5, "gamma_v": 1.0, "direct_delta_s": True,
            "ego_mode": "all_neighbors", "secondary_impact_mode": "geometric_pb",
            "healthy_pass_mode": mode, "update_mode": "synchronous",
            "stop_mode": "dual_criterion", "stop_tol": 1e-6,
        }
        res = simulate_network_v2(
            A=A,
            k_arr=np.array([1.5, 2.0, 1.5]),
            tau_arr=np.array([0.1, 0.1, 0.2]),
            vis_arr=np.array([0.0, 0.0, 0.0]),
            init_buf=np.array([100.0, 10000.0, 1000.0]),
            threshold=np.array([0.0, 0.0, 0.0]),
            is_initial=np.array([True, False, False]),
            params=params,
            seed=42,
        )
        # step 1: 1 有 2 个邻居 (0 破产, 2 健康) -> ego_ratio = 0.5
        # ΔS_01 = 100 * 2.0 * 1.0 * 0.5 * 1.0 = 100.0; F_01 = 10.0
        # step 2: 1 健康，下传给唯一健康下游 2，share=1.0 -> ds_2 = 100.0 <= ΔS_01 -> F_2 = 100 * 0.2 = 20.0
        assert abs(res["final_buffers"][1] - 9990.0) < 1e-9, f"{mode}: buf1={res['final_buffers'][1]}"
        assert abs(res["final_buffers"][2] - 980.0) < 1e-9, f"{mode}: buf2={res['final_buffers'][2]}"


def test_secondary_impact_100k_mc():
    # 第 6 条明确要求：固定 p_b，模拟 10 万次，均值与理论期望 φ*||F||*p_b/(1-p_b) 差距小于 1%
    rng = random.Random(42)
    F0 = 100.0
    phi = 0.5
    p_b = 0.4
    theory = phi * F0 * p_b / (1.0 - p_b)  # = 33.3333...

    n_sim = 100_000
    total_sec = 0.0
    for _ in range(n_sim):
        acc = 0.0
        while rng.random() < p_b:
            acc += phi * F0
        total_sec += acc
    emp = total_sec / n_sim
    rel_err = abs(emp - theory) / theory
    assert rel_err < 0.01, f"MC secondary impact error too large: emp={emp:.4f}, theory={theory:.4f}, err={rel_err:.4%}"

    # 同步核验引擎 simulate_network_v2 实际二次影响期望（2 节点，F0=100, p0=0.4, ego=1.0 -> p_b=0.4）
    A = np.array([[0.0, 100.0], [0.0, 0.0]])
    params = {
        "p0": 0.4, "phi": 0.5, "tmax": 200, "vis_formula": "paper",
        "alpha_p": 0.5, "gamma_v": 1.0, "direct_delta_s": True,
        "ego_mode": "all_neighbors", "secondary_impact_mode": "geometric_pb",
        "healthy_pass_mode": "v1_random_single", "update_mode": "synchronous",
        "stop_mode": "dual_criterion", "stop_tol": 1e-9,
    }
    n_eng = 5000
    sec_losses = []
    for sd in range(n_eng):
        res = simulate_network_v2(
            A=A,
            k_arr=np.array([1.0, 1.0]),
            tau_arr=np.array([1.0, 1.0]),
            vis_arr=np.array([0.0, 0.0]),
            init_buf=np.array([100.0, 1e6]),
            threshold=np.array([0.0, 0.0]),
            is_initial=np.array([True, False]),
            params=params,
            seed=sd,
        )
        # 节点 1 总损失 = 初始冲击 100 + 二次影响累计
        sec_losses.append((1e6 - res["final_buffers"][1]) - 100.0)
    emp_eng = float(np.mean(sec_losses))
    rel_err_eng = abs(emp_eng - theory) / theory
    assert rel_err_eng < 0.05, f"Engine secondary impact mismatch: emp={emp_eng:.4f}, theory={theory:.4f}"


if __name__ == "__main__":
    tests = [
        test_vis_factor,
        test_force_1a_1b_and_no_div_zero,
        test_loss_ratio_v2,
        test_2node_synchronous_contagion,
        test_3node_healthy_pass_bound,
        test_secondary_impact_100k_mc,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n全部 {len(tests)} 项单元测试通过！")
