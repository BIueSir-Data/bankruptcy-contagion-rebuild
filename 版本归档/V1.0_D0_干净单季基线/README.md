# V1.0 — D0 干净单季基线与数据审计

## 1. 本版说明（怎么跑出来的）
- **版本定位**：第一批先行检验 **D0（数据口径审计 + 单季基线复跑）**，作为后续所有 V2.x 改写的干净数据基准锚点。
- **数据修复**：通过 `src/build_clean_quarterly.py` 从上级未差分原始表 `../../季度数据/*.xlsx` 重建干净单季面板至 `data/clean/`（已通过广弘控股 `000529.SZ` 误差 `0.00 元`、泸州老窖 `000568.SZ` `71.02 亿` 官方定期报告双重核验）：
  - 29 个非金融行业恢复原始单季值（修复此前误做的二次差分）；
  - `银行I`、`非银金融I` 保留已去重并正确累计转单季的 CSV，移出主样本并单独存档；
  - 全部 577 张 `*_stock_matrix.csv` 无“合计”行/列，采用 `auto_by_name` 避免旧版 `iloc[:-1, :-1]` 误删末位企业。
- **模型引擎**：沿用旧版 8.31/11.16 动力学逻辑（`src/engine_legacy_d1.py`，`γ = 0.25`，单次运行 `R = 1`）。
- **复现命令**：
  ```powershell
  python src/build_clean_quarterly.py
  python src/run_d0.py --workers 8
  ```
- **对应配置文件**：`d0_config.yaml`（原运行目录：`runs/20261121_215414_D0_data_audit`）。

## 2. 核心结果摘要
- **样本流转**：原始 `91,939` 条（31 行业） → 移出金融 `1,843` 条 → 主样本 `90,096` 条 → 清洗后保留 **`89,909` 条（99.79% 保留率**，修复前仅 `60,832` 条 / 67.5%）。
- **D0.1 季度比核验**：`Q2/Q1 = 1.128`、`Q3/Q2 = 1.000`、`Q4/Q3 = 1.082`（100% 通过单季口径核验）。
- **窗口覆盖**：有效窗 `539` 个，缺失窗 `12` 个（仅剩申万 2021 分类切换前客观缺失的 `煤炭/环保/石油石化/美容护理` `2021q1–q3`，原丢失的 4 个 `2021q4` 窗口全部恢复）。
- **基线指标（`γ = 0.25`）**：`BR0 = 0.2187`，`BRT = 0.2223`，`ΔBR = +0.0035`（仅 8.3% 窗口发生传染），`Loss_Ratio = 0.0271`，`SRI_raw 均值 = 0.0084`，`SRI_ma4 均值 = 0.0074`。

## 3. 目录文件索引
- `d0_sample_flow.csv` / `d0_quarterly_check.csv` / `d0_window_matrix.csv` / `d0_window_pivot.csv`：D0.1–D0.4 审计表
- `d0_baseline_panel_main.csv` / `d0_baseline_panel_financial.csv`：主样本（29 行业）与金融（2 行业）面板
- `sri_index.csv` / `d0_sri_comparison.csv` / `D0_SRI_comparison.png`：SRI 指标与对比图
- `figs_ma4/`：论文原口径（MA4 滚动平均）五张标准图 + `sri_index.csv`
- `figs_raw_19q/`：单季原始 19Q 五张标准图 + `sri_index.csv`
- `d0_config.yaml` / `seeds.json` / `run.log`：配置副本、种子清单与运行日志
