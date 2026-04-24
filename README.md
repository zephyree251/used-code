# 多智能体光网络自治系统（OODA + 轻量RL）

本项目实现了一个面向光网络风险处置的多智能体闭环控制系统，流程为：

`Perception -> Intent -> Evidence -> Decision -> Action -> Evaluation`

系统聚焦两类风险：
- `Low_QoT`（低信号质量，OSNR不足）
- `High_Congestion`（链路拥塞）

决策层支持 `Rule / Q-Learning / DQN / VDN-lite`，当前论文主线为 `Rule / Q-Learning / DQN(7动作参数化)`。

---

## 当前版本要点

- 决策状态：8维连续特征（见 `docs/decision_state_8d_features.md`）
- DQN动作：7个参数化动作
  - `Action_Maintain`
  - `Action_Power_Boost_1p0dB / 2p0dB / 3p0dB`
  - `Action_Reroute_K1 / K3 / K5`
- 奖励机制：基于执行前后KPI变化（OSNR、利用率、路径代价、动作成本）的连续奖励
- 实验框架：主对照、负载分层、消融、训练曲线与复现实验

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

---

## 当前结论（论文口径）

- DQN(7动作)在平均修复能力与QoT指标上优于Q-Learning
- 代价是控制开销略有上升，体现“性能-开销”权衡
- 在高负载拥塞场景下，Q-Learning仍有稳定性优势

---

## 说明

- `dashscope` 未配置时，Intent层自动使用规则兜底，不影响整体运行
- 项目收口计划见 `docs/final_scope_and_delivery.md`
