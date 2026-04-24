# 多智能体光网络自治系统 (OODA)

基于 **OODA 环**（Observe–Orient–Decide–Act）的多智能体光网络自治运维系统。系统对光网络进行「感知 → 理解 → 决策 → 执行 → 评估」的闭环管理，自动发现风险（低 OSNR、拥塞等）并执行修复动作（功率调节、重路由），决策层采用 **Q-Learning 强化学习**。

---

## 功能概览

- **感知层 (Perception)**：基于 TEFNET 拓扑与流量矩阵，计算光路 OSNR、统计链路负载，识别低 OSNR 与拥塞风险。
- **意图层 (Intent)**：将风险报告转化为结构化意图（业务等级、问题类型、优化目标）。支持 **Qwen 大模型** 语义解析，未配置时自动使用规则引擎兜底。
- **证据层 (Evidence)**：从本地知识库 (RAG) 检索与意图相关的 SOP 规则，为决策提供依据。
- **决策层 (Decision)**：支持四类高层动作决策器：**规则法 / Q-Learning / 轻量 DQN / VDN-lite 价值分解**。默认使用轻量 DQN，根据低维光网络特征选择 **保持 / 调功率 / 重路由**。
- **执行层 (Action)**：执行功率提升或业务重路由（含拓扑剪枝、K-最短路径、物理指标重算）。
- **评估层 (Evaluation)**：根据执行结果计算奖励，更新 Q 表，形成闭环学习。

---

## 项目结构

```
code/
├── main.py                 # 入口：组装 OODA 流水线并运行一轮
├── requirements.txt        # Python 依赖
├── README.md
├── OPTIMIZATION.md          # 优化建议与已做修改说明
├── core/
│   └── context.py          # 系统上下文 SystemContext、BaseAgent 基类
├── agents/
│   ├── perception.py      # Step 1: 态势感知
│   ├── intent.py           # Step 2: 意图解析 (Qwen / 规则兜底)
│   ├── evidence.py        # Step 3: 证据检索 (RAG)
│   ├── decision_rl.py     # Step 4: RL 决策
│   ├── action.py           # Step 5: 动作执行
│   └── evaluation.py      # Step 6: 评估与 Q 表更新
├── utils/
│   ├── tefnet_loader.py    # 拓扑与流量数据加载
│   └── visualizer.py      # 重路由等可视化
└── data/
    ├── tefnet_nodes.csv    # 节点数据
    ├── tefnet_links.csv    # 链路数据
    ├── tefnet_traffic.csv  # 流量需求矩阵
    ├── knowledge_base.json # 专家规则知识库
    └── q_table_memory.json  # (可选) RL Q 表持久化
```

---

## 环境要求

- Python 3.8+
- 见 `requirements.txt`

---

## 安装与运行

### 1. 克隆或进入项目目录

```bash
cd /path/to/code
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 运行主流程

```bash
python main.py
```

如需切换决策方式，可设置环境变量：

```bash
DECISION_MODE=rule python main.py
DECISION_MODE=rl python main.py
DECISION_MODE=dqn python main.py
DECISION_MODE=vdn python main.py
```

每轮会从全部业务中随机抽取 15 条作为当前活跃业务，依次执行 Perception → Intent → Evidence → Decision → Action → Evaluation，并在控制台输出风险报告、决策与修复日志；若有重路由，会触发可视化。

---

## 配置说明

### 可选：使用 Qwen 大模型做意图解析

1. 安装 DashScope SDK：

   ```bash
   pip install dashscope
   ```

2. 设置环境变量（**不要将 API Key 写进代码**）：

   ```bash
   export DASHSCOPE_API_KEY=sk-你的密钥
   export DASHSCOPE_MODEL=qwen-plus   # 可选，默认 qwen-plus
   ```

未配置或未安装 `dashscope` 时，系统会自动使用规则引擎生成意图，无需大模型即可运行。

### 数据文件

- 拓扑与流量：`data/tefnet_nodes.csv`、`data/tefnet_links.csv`、`data/tefnet_traffic.csv` 需存在且格式符合 `TefnetLoader` 约定。
- 知识库：`data/knowledge_base.json` 缺失时证据层无法检索规则，决策仍可进行。
- Q 表：`data/q_table_memory.json` 为 RL 存档，首次运行会自动生成；存在则会读档继续学习。

---

## 输出说明

- 控制台：各 Agent 的步骤说明、风险列表、决策动作、奖励与 Q 值更新、本轮修复报告。
- 若发生重路由：`utils.visualizer` 会生成重路由相关图示（具体输出路径见 `visualizer.py`）。

---

## 论文对照实验（聚焦版本）

为响应“聚焦具体网络算法”的要求，项目提供了一个可直接用于论文结果表的对照脚本：

```bash
python benchmark_focus.py
```

脚本会在同一批业务样本下比较：

- Rule Baseline（固定规则决策）
- Q-Learning Baseline（传统表格强化学习）
- DQN Multi-Agent（轻量神经网络决策）
- VDN-lite Multi-Agent（路由/功率双智能体价值分解）

并输出关键指标（可直接写入实验章节）：

- 执行前/后风险数 (`risk_before`, `risk_after`)
- 风险修复率 (`risk_repair_rate`)
- 拥塞/低 QoT 风险变化
- 平均 OSNR、平均重路由次数、平均功率提升次数

可选：运行负载分层实验（低/中/高负载）：

```bash
python benchmark_load_levels.py
```

可选：运行 DQN 消融实验（Replay/Target 与探索策略）：

```bash
python benchmark_ablation.py
```

可选：将负载分层结果画成论文图：

```bash
python plot_load_level_results.py
```

可选：持续训练 DQN 直到收益趋稳，并生成训练曲线：

```bash
python train_dqn_until_stable.py
```

---

## 其他说明

- 更多优化建议与已做修改（如 API Key 环境变量、意图兜底与动作名统一等）见 [OPTIMIZATION.md](OPTIMIZATION.md)。
- 收口与论文交付建议见 [docs/final_scope_and_delivery.md](docs/final_scope_and_delivery.md)。
- 实验与绘图脚本：`experiment_runner.py`、`draw_ooda_arch.py`、`draw_osnr_schematic_v2.py` 等可按需单独运行。
