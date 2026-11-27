# 模型说明文档（ODD 协议规范）

> 本文档严格按照《11.16版重建实验指导第一批_修订》通用规范第 7 条要求，以 **ODD 协议（Overview, Design concepts, Details）** 七部分结构记录当前冻结版本（V2.1 系列）的完整数学定义、数据结构与调度规则。每次核心代码修改时同步更新本文档。

---

## 1. 目的与检验模式（Purpose and Patterns）
- **研究目的**：在行业内企业资金关联网络上，基于社会力（Social Force）框架刻画中国 A 股上市公司（2021Q1–2025Q3，29 个申万一级非金融主样本行业）在财务困境冲击下的破产风险传染机制、级联深度与系统性风险演化。
- **检验模式**：
  1. **样本口径检验（D0）**：验证单季利润表口径（`Q2/Q1 ≈ 1.0`）、金融两行业独立分离、全窗口覆盖率与缺失显式标注 `NA`。
  2. **传播机制检验（D1）**：在同一初始激活集合 `B(0)` 与同一种子下对比传播开/关，剥离直接损失与传播新增损失及传染代数。
  3. **渐进消融与并列对照（V2.0 → V2.1 → V2.1b–e）**：分别对比净利润率 V1 / 毛利率 V2、稠密网 / 稀疏网、随机单下游 V1 / 按边权分摊 V2、以及外生缓冲压力冲击 `δ`。
  4. **单元手算校验**：通过 2–3 节点手算算例及 10 万次二次影响蒙特卡洛测试（`tests/test_engine_v2.py`）确保公式实现无误。

---

## 2. 实体、状态变量与尺度（Entities, State Variables, and Scales）
- **核心实体**：申万一级行业内的 A 股上市公司节点 $i \in \{1, \dots, N\}$，边 $(a \to b)$ 表示企业 $a$ 向企业 $b$ 供货（$a$ 为供应商，$b$ 为客户），边权 $V_{ab} > 0$ 为交易金额（元）。
- **节点静态属性（单行业-单季度窗口内不变）**：
  - 财务基础量：营业收入 $R_i$、营业成本/费用 $E_i$、净利润 $NI_i$、总资产 $A_i$、总负债 $L_i$（单位统一为人民币元）。
  - 投入产出系数：$k_i = R_i / E_i$（用于公式 1a，通常 $> 1$）；VS 评分中的营业成本率保持 $E_i / R_i$ 不变。
  - 利润率 $\tau_i$：V1（论文版）取 $\max(NI_i / R_i, 0)$ 并记录被截断企业数；V2（毛利率版）取 $\max(1 - E_i / R_i, 0)$。
  - 网络可视度 $\text{vis}_i$：在显著性过滤稀疏骨干网 $W_s$（`disparity_alpha = 0.05`）上，以边距 $d_{ab} = 1 / V_{ab}$ 计算未归一化介数中心性 $C_B(i)$，再按行业中位数归一化：$\text{vis}_i = C_B(i) / (\text{median}(C_B) + 10^{-8})$。
- **节点动态状态变量（随仿真步 $s = 0, 1, \dots, S_{\max}$ 演化）**：
  - 存活缓冲 $C_i(s)$：主设定 $C_i(0) = (A_i - L_i) \cdot (1 - \delta)$（所有者权益，含外生压力扣减 $\delta \in [0, 1)$），破产阈值 $U_i = 0$。
  - 破产状态 $\text{broken}_i(s) \in \{0, 1\}$、本步新破产标记 $\text{new\_broken}_i(s)$、破产代数 $\text{gen}_i \in \{-1, 0, 1, 2, \dots\}$（初始破产为 0）。
  - 待执行队列：上步所受冲击列表（用于二次影响）与上步由公式 1a 产生的待下传销售损失 $\Delta S$。
- **时空尺度**：时间截面覆盖 2021Q1–2025Q3（19 个单季，剔除 2020Q4 基期）；空间单元为 29 个非金融行业（`银行`、`非银金融` 单独保存）；单季内仿真为离散步 $s \le S_{\max} = 1000$。

---

## 3. 过程概述与调度（Process Overview and Scheduling）
采用**严格同步更新（Synchronous Update，第 8 条）**。在每个离散步 $s \ge 1$，所有力与下传均基于 $s - 1$ 步快照状态计算：
1. **直接破产冲击（公式 1a / 1b）**：遍历 $s-1$ 步新破产节点 $a$，向下游客户 $b$ 施加销售损失 $\Delta S_{ab}$ 与利润冲击 $|F_{ab}|$，向上游供应商 $b$ 施加利润冲击 $|F_{ba}|$。
2. **健康节点下传（第 7 条）**：遍历 $s-1$ 步仍健康且在 $s-1$ 步收到 $\Delta S$ 的节点 $b$，按 V1（随机单健康下游）或 V2（按边权分摊给全部健康下游）向下传，严格保证下传总量 $\le \Delta S$。
3. **二次影响（第 6 条）**：遍历 $s-1$ 步仍健康且受到冲击（或其二次影响）的节点 $b$，以概率 $p_b = p_0 \cdot \text{ego\_ratio}_b$ 再受一次 $\varphi \cdot \|F_{\text{orig}}\|$ 冲击。
4. **统一扣减与统一激活**：将本步累计冲击统一从健康节点的 $C_i(s-1)$ 中扣减，得到 $C_i(s)$；若 $C_i(s) \le U_i$ 则标记为第 $s$ 步新破产，记录其传播代数 $\text{gen}_i = \min(\text{src\_gen}) + 1$。
5. **双重停止判定（第 9 条）**：连续 3 步无新增激活且待执行二次影响与下传总额 $< 10^{-6} \cdot \sum C_i(0)^+$，或达到 $S_{\max} = 1000$ 时终止。

---

## 4. 设计概念（Design Concepts）
- **涌现（Emergence）**：行业最终破产率 `BRT`、净传染增量 `ΔBR`、级联代数 `Max_Prop_Gen` 与全市场系统性风险指数 `SRI` 均由企业异质性财务缓冲、边权集中度与多步同步级联动态涌现。
- **交互与局部感知（Interaction & Sensing）**：企业通过有向加权交易边 $(a \to b)$ 感知邻居破产状态；破产邻居占比 $\text{ego\_ratio}_b$ 反映局部供应链恶化对声誉与二次影响概率 $p_b$ 的放大作用。
- **随机性（Stochasticity）**：随机性仅来自（1）二次影响伯努利抽样（概率 $p_b$）与（2）V1 模式下健康节点下传的随机下游选择；每个（行业、季度、重复）均由 SHA-1 确定性派生独立随机种子并存档于 `seeds.json`。
- **观测指标（Observation，第 12 条）**：
  - 损失率：$\text{Loss\_Ratio} = \sum_i (C_i(0)^+ - C_i(T)^+)^+ / \sum_i C_i(0)^+$（分母 $\le 0$ 记 `NA`）。
  - 行业风险值：$\text{IRCI}_k = \text{BRT}_k \times \text{Loss\_Ratio}_k$。
  - 系统性风险指数：$\text{SRI}_t = \sum_k w_{k,t} \cdot \text{IRCI}_{k,t}$（原值不作 Z 变换，其中 $w_{k,t} \propto \text{Leverage}_{k,t} \times \text{WCC}_{k,t}$ 在季度内归一化；同时输出 19 季原始单季口径与 16 季 MA4 滚动平均口径）。
  - 加权聚类系数 `WCC`：Fagiolo (2007) 有向加权全三角形聚类系数。
  - IEEE Access 2020 对照指标：$\text{BAR}_{\text{end}} = \text{BRT}$，$\Delta\text{BAR} = \text{BRT} - \text{BR0}$，$\lambda = \Delta\text{BAR} / (1 - \text{BR0})$。

---

## 5. 初始化（Initialization）
- 按单行业-单季度窗口清洗数据：严格剔除 $R_i \le 0$ 或 $E_i \le 0$ 的企业-季度，并与 `stock_matrix.csv` 企业代码对齐（不盲删末行）。
- 计算脆弱性综合评分 $VS_i = z(L_i/A_i) + z(E_i/R_i) - z(NI_i/R_i) - z(NI_i/A_i)$（各分项先按 1%–99% 缩尾）。
- 按高危校准分位 $\gamma$（主设定 `0.25`）对 $(x_i, NI_i / L_i)$ 做 Huber 稳健回归得到行业阈值 $\theta_k$；满足 $L_i \cdot \theta_k > NI_i$ 或初始权益 $C_i(0) \le U_i$ 的企业进入初始破产集合 $B(0)$，记 $\text{gen}_i = 0$。

---

## 6. 输入数据（Input Data）
- 干净单季财务面板：`data/clean/<行业>I/<quarter>_<行业>I.csv`（由 `src/build_clean_quarterly.py` 从未差分原始 Excel 生成，已通过上市公司 2025 一季报/半年报官方公告精确到分核验）。
- 行业关联矩阵：`data/clean/<行业>I/<quarter>_<行业>I_stock_matrix.csv`（全部 577 张矩阵与财务表企业代码 0 失配）。
- 全部超参数与版本开关均外置于 `configs/*.yaml`，代码中无硬编码参数。

---

## 7. 子模型（Submodels）
1. **可视度缓冲项（第 3 条）**：
   $$\text{vis\_fac}(b) = \alpha \cdot e^{-\gamma_v \cdot \text{vis}_b} + (1 - \alpha), \quad \text{主设定 } \alpha = 0.5,\; \gamma_v = 1.0$$
2. **公式 1a（供应商 $a$ 破产 → 客户 $b$，第 5 条防除零形式）**：
   $$\Delta S_{ab} = V_{ab} \cdot k_b \cdot \text{vis\_fac}(b) \cdot \text{ego\_ratio}_b \cdot \frac{V_{ab}}{\sum_{j \in US_b} V_{jb}}, \quad |F_{ab}| = \Delta S_{ab} \cdot \tau_b$$
3. **公式 1b（客户 $a$ 破产 → 供应商 $b$）**：
   $$|F_{ba}| = V_{ba} \cdot \tau_b \cdot \text{vis\_fac}(b) \cdot \text{ego\_ratio}_b \cdot \frac{V_{ba}}{\sum_{j \in DS_b} V_{bj}}$$
4. **二次影响子模型（第 6 条）**：上一步受冲击 $\|F_{\text{orig}}\|$（或其二次影响）的健康节点，本步以概率 $p_b = p_0 \cdot \text{ego\_ratio}_b$ 再受 $\varphi \cdot \|F_{\text{orig}}\|$，其理论总期望严格等于 $\varphi \|F_{\text{orig}}\| \cdot \frac{p_b}{1 - p_b}$。
5. **健康节点下传子模型（第 7 条）**：
   - V1（随机单下游）：随机选取健康下游 $c \in DS_b^{\text{healthy}}$，传递 $\Delta S_c = \Delta S \cdot \min(1, V_{bc} / \sum_{j \in DS_b} V_{bj}) \le \Delta S$，施加利润冲击 $\Delta S_c \cdot \tau_c$。
   - V2（全下游按权分摊）：对全部 $c \in DS_b^{\text{healthy}}$ 按 $w_{bc} / \sum_{j \in DS_b^{\text{healthy}}} w_{bj}$ 分配 $\Delta S$，总下传量严格等于 $\Delta S$。
