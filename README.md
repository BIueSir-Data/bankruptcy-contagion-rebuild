# 破产风险传染模型重建实验（bankruptcy-contagion-rebuild）

本仓库严格遵循《11.16版重建实验指导第一批_修订》构建，涵盖 **第一阶段先行检验（D0 数据审计、D1 无传播基准对照）** 与 **第二阶段渐进式冻结模型（V2.0 → V2.1e，覆盖第三节 12 条修正中的全部可运行机制 + 单元测试 + ODD 模型说明文档）**。

> **暂缓项说明**：经课题组沟通，D2（ST/Altman Z 外部困境事件验证）与第四部分（企业级前五大客户/供应商网络）暂待后续数据到位后再接入；敏感性分析（多参数网格扫描）暂缓，当前全部版本采用单组主设定渐进消融。

---

## 一、项目分类目录结构

```text
bankruptcy-contagion-rebuild/
├── 1. 核心代码 (src/)
│   ├── build_clean_quarterly.py   # [数据重建] 从原始未差分 Excel 重建干净 31 行业单季面板（修复 29 行业二次差分）
│   ├── verify_with_tushare.py     # [第三方核验] Tushare Pro 财报口径双重核验脚本（Token 从环境变量读取）
│   ├── common.py                  # [公共工具] YAML 加载、日志、SHA-1 独立种子、安全矩阵读取（不盲删末行）、SRI 计算
│   ├── engine_legacy_d1.py        # [V1.0 引擎] 沿用旧版 8.31 逻辑 + 传播开关与传播代数追踪（专供 D0/D1 基线）
│   ├── engine_v2.py               # [V2.x 冻结引擎] 渐进累积式配置驱动引擎（完整实现第 1–12 条修订）
│   ├── plotting.py                # [统一出图] 论文标准五图（Times New Roman, 300 DPI, 三态标注；同时出 MA4 与 19Q）
│   ├── run_d0.py / run_d1.py      # [D0/D1 入口] 先行检验运行脚本
│   ├── run_v2.py                  # [V2.x 入口] 渐进版本统一运行脚本（自动产出面板、诊断表、标准五图与 SRI）
│   ├── run_batch_v2.py            # [批量无人值守] 依次跑单测 + V2.1b–V2.1e + 自动归档与生成总对比表
│   ├── run_gamma_monotonicity.py  # [第 11 条诊断] 全量 539 窗 γ → θ_k → BR0 单调性与负 θ_k 核查
│   ├── merge_financial_all31.py   # [含金融合并] 将 29 主样本与银行/非银金融合并生成持久化 31 行业结果
│   └── build_figure_summary.py    # [图表总汇总] 生成 7 版同图 SRI 对比图与 31 行业跨版本 IRCI 热力图
│
├── 2. 渐进版本配置 (configs/)
│   ├── d0_config.yaml             # V1.0_D0：数据口径审计与单季基准
│   ├── d1_config.yaml             # V1.0_D1：γ=0.25 & 0.45 传播开/关成对对照
│   ├── v2_0_accounting_core.yaml  # V2.0：覆盖 V1.0 + 第 1,3,4,5,12 条（净资产缓冲、k=R/E、可视度、max(τ,0)、新指标）
│   ├── v2_1_dynamics_sync.yaml    # V2.1：覆盖 V2.0 + 第 2,6,7,8,9,11 条（全邻居 Ego、几何二次影响、同步更新、双重早停）
│   ├── v2_1b_gross_margin.yaml    # V2.1b：覆盖 V2.1 + 第 4 条 V2 毛利率 max(1-E/R, 0)
│   ├── v2_1c_sparse_network.yaml  # V2.1c：覆盖 V2.1b + 稀疏骨干网 (disparity_alpha=0.05)
│   ├── v2_1d_weighted_all_pass.yaml # V2.1d：覆盖 V2.1c + 第 7 条 V2 全下游按边权下传
│   ├── v2_1e_delta_stress_50pct.yaml # V2.1e：覆盖 V2.1d + 第 11 条外生缓冲扣减 δ=50%
│   └── v2_2_mc_convergence.yaml   # (预备) V2.2：覆盖定稿 V2.1x + 第 10 条 R=50/200 蒙特卡洛重复
│
├── 3. 单元测试 (tests/)
│   └── test_engine_v2.py          # 6 项 2–3 节点手算单测 + 第 6 条 10 万次二次影响期望误差 <1% 校验
│
├── 4. 模型规范文档 (docs/)
│   └── ODD.md                     # 通用规范第 7 条：ODD 协议七部分完整模型说明文档
│
└── 5. 版本归档与汇总结果 (版本归档/)
    ├── README.md                  # 全版本特性演变与核心指标总览
    ├── V1_V2_all31_vs_main29_comparison.csv  # 29 主样本 vs 31 含金融 vs 银行/非银总对比表
    ├── V1_V2_SRI_all31_by_quarter.csv        # 7 版 31 行业逐季 SRI 总表
    ├── V2_suite_comparison.csv               # V2.0–V2.1e 主样本消融总表
    ├── V1.0_D0_干净单季基线/                 # 含样本流转表、季度比核验、配置与 README
    ├── V1.0_D1_无传播对照/                   # 含分行业汇总、BR0 vs ΔBR 散点图、配置与 README
    ├── V2.0_会计与社会力核心/ ~ V2.1e_外生缓冲冲击50pct版/
    ├── V2_辅助诊断_gamma单调性检验/          # 第 11 条 539 窗单调性核查表与 README
    └── 结果图与SRI总汇总/                    # 7 版 SRI 对比图(MA4/19Q)、31 行业 IRCI 热力图与总 SRI CSV
```

> 注：为保持 GitHub 仓库轻量，原始数据 `data/clean/`、时间戳日志目录 `runs/` 以及各版本内大体积逐窗面板与单版散图已通过 `.gitignore` 忽略，仓库中保留了全部可复现代码、配置、单元测试、ODD 文档、各版 `README.md` 以及核心汇总 CSV/总览图。

---

## 二、核心阶段性发现速览（`γ = 0.25`，2021Q1–2025Q3）

1. **D0 数据口径重大修复（经官方财报精确到“分”核验）**：
   - 发现原始米筐数据中仅 `银行I`/`非银金融I` 为 YTD 累计值，其余 29 个非金融行业本已是单季值；此前统一差分导致二次差分（广弘控股 `000529.SZ` 2025Q2 官方与新版均为 `638,071,536.27 元`，误差 `0.00 元`；泸州老窖 `000568.SZ` 官方与新版 `71.02 亿`，旧版被减成 `-22.51 亿`）。
   - 修复后主样本保留率由 `67.5%` 升至 **`99.79%`（`89,909 / 90,096`）**，缺失窗由 `16` 降至客观缺失的 `12`，`Q2/Q1=1.128, Q3/Q2=1.000, Q4/Q3=1.082`。
2. **D0.2 金融分离的实证必要性**：
   - 旧版 `V1.0_D0` 下银行每季 100% 破产，仅金融 2 行业就把 31 行业 `SRI_ma4` 从 `0.0074` 暴拉至 `0.0722`；新版净资产缓冲下银行不再虚假全灭，且高杠杆的**非银金融**在稀疏网下呈现清晰传染（`V2.1c: +13.20%` → `V2.1e: +19.70%`）。
3. **V2.0 → V2.1e 渐进消融核心机理发现**：
   - 29 实体行业的 `Loss_Ratio`（`0.1136 → 0.1176`）与 `SRI_ma4`（`0.0249 → 0.0257`）跨版本严格单调响应，但实体行业二次破产率 `ΔBR` 几乎为 `0`（仅 `0~1/539` 窗），即使切换毛利率 V2、稀疏网、全下游按权下传乃至砍半净资产（`δ=50%`）依然如此。
   - 根本原因在于**量纲错配**：单季流量利润冲击（中位数千万级）相对多年累积所有者权益归零 `U_i = 0`（中位数约 30 亿元）小约 2 个数量级。
4. **第 11 条 `γ → θ_k → BR0` 核验**：
   - `γ ∈ [0.05, 0.45]` 在 539 个窗口上 `BR0` 与 `θ_k` **100% 单调（0 个反常窗）**；`γ = 0.25` 时 `67.16%` 窗口 `θ_k < 0`。

---

## 三、本地复现命令

```powershell
# 0. 运行全部 6 项单元测试
python tests/test_engine_v2.py

# 1. 重建干净单季面板（需具备上级未差分 Excel）
python src/build_clean_quarterly.py

# 2. 先行检验 D0 & D1
python src/run_d0.py --workers 8
python src/run_d1.py --workers 8

# 3. 渐进版本 V2.0 / V2.1 与一键批量 V2.1b–V2.1e
python src/run_v2.py --config configs/v2_0_accounting_core.yaml --workers 8
python src/run_v2.py --config configs/v2_1_dynamics_sync.yaml --workers 8
python src/run_batch_v2.py --workers 8

# 4. 辅助诊断、含金融 31 行业合并与总汇总出图
python src/run_gamma_monotonicity.py
python src/merge_financial_all31.py
python src/build_figure_summary.py
```
