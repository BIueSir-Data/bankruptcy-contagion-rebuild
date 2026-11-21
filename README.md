# 重建实验方案（第一阶段：先行检验 D0 & D1）

本目录严格遵循导师《11.16版重建实验指导第一批_修订》的通用规范构建，**只读引用**上一级工作区的 31 个行业原始 CSV 与交易矩阵，绝不覆盖原始数据。

## 目录结构

```text
重建实验方案/
├── configs/
│   ├── d0_config.yaml        # D0 数据口径审计与单季基准复跑配置
│   └── d1_config.yaml        # D1 无传播基准对照配置 (γ=0.25, 0.45)
├── data/clean/               # D0 产出：样本流转表、全窗口覆盖表、单季口径检验表
├── runs/<时间戳>_<实验名>/    # 每次运行自动归档：配置副本、seeds.json、run.log、原始输出 CSV
├── figures/                  # 最新生成的 D0 SRI 走势图 & D1 BR0-ΔBR 散点图
└── src/
    ├── common.py             # 配置、日志、种子、安全矩阵读取（按名称去“合计”，不盲切）、SRI 计算
    ├── engine_legacy_d1.py   # 沿用 11.16 现有网络与 8.31 基准口径，新增传播开关与传播代数追踪
    ├── run_d0.py             # D0 一键脚本
    └── run_d1.py             # D1 一键脚本
```

## 运行命令（在 `重建实验方案/` 目录下执行）

### 1. 先行检验 D0（数据口径审计 + 样本流转 + 单季 SRI 复核）

```powershell
# 抽样 5 个行业快速验证（约 15 秒）：
python src/run_d0.py --config configs/d0_config.yaml --inds 交通运输,公用事业,国防军工,石油石化,银行 --workers 4

# 全量运行全部 31 行业（本机建议 4 或 8 进程）：
python src/run_d0.py --config configs/d0_config.yaml --workers 8
```

**核心产出物**（同时保存在 `runs/<时间戳>_D0_data_audit/` 与 `data/clean/`）：
- `d0_quarterly_check.csv`：利润表相邻季度比值核验（单季 vs 年初累计证据）
- `d0_sample_flow.csv`：样本流转表（原始 → 移出银行/非银金融 → 清洗 → 矩阵对齐 → 有效窗口）
- `d0_window_matrix.csv` / `d0_window_pivot.csv`：全部行业—季度有效/缺失（`NA`）清单及原因
- `d0_sri_comparison.csv` & `figures/D0_SRI_comparison.png`：主样本(29行业)与全样本(31行业)的 19 季原值 SRI 与旧 MA4 SRI 对比（标注 2022Q4、2023Q4）

### 2. 先行检验 D1（无传播基准对照，γ=0.25 & 0.45）

```powershell
# 抽样 5 个行业快速验证（约 20 秒）：
python src/run_d1.py --config configs/d1_config.yaml --inds 交通运输,公用事业,国防军工,石油石化,美容护理 --workers 4

# 全量运行 29 个主样本行业（移出银行、非银金融，建议 4 或 8 进程）：
python src/run_d1.py --config configs/d1_config.yaml --workers 8
```

**核心产出物**（保存在 `runs/<时间戳>_D1_no_propagation_benchmark/`）：
- `d1_panel_g0.25.csv` / `d1_panel_g0.45.csv` / `d1_panel_all.csv`：逐行业—季度对照结果（含 `BR0, BRT_off, BRT_on, dBR, lambda, Direct_Loss_Ratio, Loss_Ratio_on, Prop_Added_Loss_Ratio, NonInitial_Loss_Ratio, Max_Prop_Gen`）
- `d1_industry_summary.csv`：各行业跨期均值汇总
- `figures/D1_BR0_vs_dBR_scatter.png`：各行业 `BR0 vs ΔBR` 双子图散点图
