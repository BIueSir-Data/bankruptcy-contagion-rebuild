# V2.1e_外生缓冲冲击50pct版

## 1. 本版说明（怎么跑出来的）
- **改动摘要**：完全覆盖 V2.1d，按第 11 条对全系统净资产缓冲施加 δ=50% 宏观压力扣减
- **复现命令**：
  ```powershell
  python tests/test_engine_v2.py
  python src/run_v2.py --config configs/v2_1e_delta_stress_50pct.yaml --workers 8
  ```
- **配置文件**：`v2_1e_delta_stress_50pct.yaml`（原始运行目录：`runs/20261123_170515_V2_1e_delta_stress_50pct`）

## 2. 核心结果摘要（29 主样本行业 × 19 季度）
- **有效窗口数**：`539`（未收敛窗口：`0`）
- **破产率与传染**：`BR0 = 0.2196` → `BRT = 0.2196`，净传染增量 **`ΔBR = +0.0000`**
- **传染覆盖面**：发生传染窗口占比 **`0.19%`**，多代传染 (`Gen≥2`) 占比 **`0.00%`**，最大传染代数 **`1`**
- **损失率与系统性风险**：`Loss_Ratio = 0.1176`，`SRI_raw 均值 = 0.0264`，`SRI_ma4 均值 = 0.0257`
- **利润率非负截断数**：`2728` 条

## 3. 目录文件索引
- `panel_main.csv` / `panel_financial.csv` / `panel_all.csv` / `missing_windows.csv`：全量面板与缺失清单
- `industry_summary.csv` / `gamma_theta_br0_diag.csv`：分行业汇总与 `γ → θ_k → BR0` 诊断表
- `sri_index.csv` + `figs_ma4/`（论文 MA4 标准五图）+ `figs_raw_19q/`（单季 19Q 标准五图）
- `v2_1e_delta_stress_50pct.yaml` / `seeds.json` / `run.log`：配置副本、种子清单与运行日志
