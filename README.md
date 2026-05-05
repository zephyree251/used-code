# 基于Agent的光网络智能控制研究与设计

本项目为本科毕业设计“基于Agent的光网络智能控制研究与设计”的代码实现。系统面向光网络运行过程中的低QoT和链路拥塞风险，构建了基于多智能体协同机制与强化学习决策方法的闭环控制流程。

核心流程为：

`Perception -> Intent -> Evidence -> Decision -> Action -> Evaluation`

系统聚焦两类风险：
- `Low_QoT`（低信号质量，OSNR不足）
- `High_Congestion`（链路拥塞）

决策层支持 `Rule / Q-Learning / DQN / VDN-lite`，当前论文主线为 `Rule / Q-Learning / DQN(7动作参数化)` 三类方法对比。

---

## 当前版本要点

- 决策状态：8维连续状态特征（见 `docs/decision_state_8d_features.md`）
- DQN动作空间：7类离散参数化动作
  - `Action_Maintain`
  - `Action_Power_Boost_1p0dB / 2p0dB / 3p0dB`
  - `Action_Reroute_K1 / K3 / K5`
- 奖励机制：基于动作执行前后KPI变化（OSNR、链路利用率、路径长度、动作成本）的连续奖励
- 实验框架：主对照实验、负载分层实验、DQN消融实验与多随机种子稳健复验
- 论文指标：风险修复率、剩余风险数、OSNR不达标业务数、拥塞未缓解业务数、平均OSNR、控制消息数等

---

## 目录结构（核心）

```text
code/
├── main.py
├── agents/
│   ├── perception.py
│   ├── intent.py
│   ├── evidence.py
│   ├── decision_rl.py
│   ├── action.py
│   └── evaluation.py
├── benchmark_focus.py
├── benchmark_load_levels.py
├── benchmark_ablation.py
├── train_dqn_until_stable.py
├── train_dqn_dense_curves.py
├── plot_dqn_dense_smoothed.py
├── docs/
│   ├── latest_benchmark_results.md
│   ├── robust_revalidation_7actions.json
│   ├── thesis_draft_main_body.md
│   └── decision_state_8d_features.md
└── output_images/
```

---

## 快速开始

### 1) 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 运行单轮闭环

```bash
python main.py
```

可选模式：

```bash
DECISION_MODE=rule python main.py
DECISION_MODE=rl python main.py
DECISION_MODE=dqn python main.py
DECISION_MODE=vdn python main.py
```

---

## 实验命令

### 主对照实验（推荐）

```bash
python benchmark_focus.py
```

### 负载分层实验

```bash
python benchmark_load_levels.py
python plot_load_level_results.py
```

### DQN消融实验

```bash
python benchmark_ablation.py
```

### 训练曲线

```bash
python train_dqn_until_stable.py
python train_dqn_dense_curves.py
python plot_dqn_dense_smoothed.py
```

---

## 结果与图表位置

- 结果汇总：`docs/latest_benchmark_results.md`
- 稳健复验（5 seeds）：`docs/robust_revalidation_7actions.json`
- 训练曲线数据：`docs/dqn_training_curve*.csv`
- 训练图和整理图：`output_images/`、`output_images/_organized/`
- 论文方法与实验说明：`docs/decision_state_8d_features.md`、`docs/final_scope_and_delivery.md`

---

## 当前结论（论文口径）

- DQN(7动作)在风险修复率、OSNR不达标业务数和平均OSNR等关键指标上优于Q-Learning
- DQN的重路由次数、平均路径长度和控制消息数略高于Q-Learning，但增幅有限，体现出可控开销下的综合优化能力
- 负载分层结果表明，DQN更适合中低负载、同时关注风险修复和QoT提升的光网络控制场景
- 在高负载、拥塞较强且可替代路径受限的场景下，Q-Learning或更保守的策略仍具有稳定性优势

---

## 说明

- `dashscope` 未配置时，Intent层自动使用规则兜底，不影响整体运行
- 项目收口计划见 `docs/final_scope_and_delivery.md`
