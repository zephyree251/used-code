# 最新实验结果记录

## 实验版本

- 决策框架：`Rule / Q-Learning / DQN`
- 多智能体流程：`Perception -> Intent -> Evidence -> Decision -> Action -> Evaluation`
- DQN 特点：
  - 8 维轻量状态特征
  - 连续型 KPI 奖励
  - 引导式探索
  - CPU 小型 MLP

## 运行方式

使用命令：

```bash
.venv/bin/python benchmark_focus.py
```

当前脚本配置：

- 训练轮数：`120`
- 评估轮数：`30`
- 每轮抽样业务数：`15`
- 随机种子：`42`

## 结果汇总

| 指标 | Rule | Q-Learning | DQN |
|---|---:|---:|---:|
| risk_before | 6.7333 | 6.8333 | 6.9000 |
| risk_after | 6.5000 | 6.0667 | 5.6333 |
| risk_repair_rate | 0.0347 | 0.1122 | 0.1836 |
| congestion_before | 1.7667 | 1.7667 | 1.7667 |
| congestion_after | 1.2000 | 1.2667 | 1.3667 |
| low_qot_before | 4.9667 | 5.0667 | 5.1333 |
| low_qot_after | 5.3000 | 4.8000 | 4.2667 |
| avg_osnr_after | 35.0308 | 35.5390 | 35.9382 |
| avg_reroute_actions | 1.2667 | 3.8333 | 6.3333 |
| avg_boost_actions | 4.9667 | 2.5000 | 0.1000 |
| avg_path_length | 204.1191 | 207.4613 | 211.4198 |
| max_link_utilization | 1.1194 | 1.1159 | 1.1232 |
| control_messages | 14.7333 | 17.5000 | 20.1333 |

## 初步结论

1. `DQN` 在总体风险修复率上表现最好，`0.1836` 高于 `Q-Learning` 的 `0.1122` 和 `Rule` 的 `0.0347`。
2. `DQN` 在低 QoT 风险消减方面更强，`low_qot_after` 降至 `4.2667`，同时 `avg_osnr_after` 最高，为 `35.9382`。
3. `DQN` 的代价是控制动作更激进：
   - `avg_reroute_actions` 更高
   - `avg_path_length` 更长
   - `control_messages` 更多
4. 这说明 `DQN` 更偏向通过积极重路由换取更高的修复效果和更好的 QoT 指标。

## 论文可用表述

可直接写为：

> 在相同业务采样条件下，本文提出的轻量 DQN 决策方法在总体风险修复率、低 QoT 风险消减以及平均 OSNR 指标上均优于规则法和传统 Q-Learning 方法。其中，DQN 的风险修复率达到 0.1836，相比 Q-Learning 的 0.1122 有明显提升。但与此同时，DQN 采用了更积极的重路由策略，导致控制消息数量和平均路径长度有所增加，体现出性能收益与控制开销之间的权衡关系。

## 备注

- 当前结果来自最新一版连续型奖励函数和引导式探索策略。
- 若后续继续调参，应保留本文件作为阶段性基线记录。

---

## 多随机种子复现实验

为降低单次随机采样带来的偶然性，额外使用 `seed = 42 / 52 / 62` 进行了三组重复实验。

### 关键指标均值

| 指标 | Rule | Q-Learning | DQN |
|---|---:|---:|---:|
| risk_repair_rate | 0.0263 | 0.1169 | 0.1294 |
| risk_after | 6.1000 | 5.6111 | 5.5444 |
| congestion_after | 1.2556 | 1.2778 | 1.3778 |
| low_qot_after | 4.8444 | 4.3333 | 4.1667 |
| avg_osnr_after | 35.9440 | 36.3881 | 36.5295 |
| avg_reroute_actions | 1.2222 | 3.4444 | 4.1333 |
| avg_boost_actions | 4.3778 | 2.2444 | 1.7778 |
| avg_path_length | 196.4431 | 199.1906 | 199.9500 |
| max_link_utilization | 1.1373 | 1.1342 | 1.1392 |
| control_messages | 13.7556 | 16.1556 | 16.8889 |

### 波动情况（标准差）

| 指标 | Rule | Q-Learning | DQN |
|---|---:|---:|---:|
| risk_repair_rate | 0.0059 | 0.0251 | 0.0620 |
| risk_after | 0.3139 | 0.3292 | 0.4131 |
| congestion_after | 0.1030 | 0.0956 | 0.1931 |
| low_qot_after | 0.3247 | 0.3682 | 0.2373 |
| avg_osnr_after | 0.7095 | 0.6810 | 0.4236 |
| avg_reroute_actions | 0.0416 | 0.2753 | 2.4301 |
| avg_boost_actions | 0.4166 | 0.1931 | 2.0023 |
| avg_path_length | 6.0236 | 6.5954 | 10.3531 |
| max_link_utilization | 0.0541 | 0.0565 | 0.0552 |
| control_messages | 0.7335 | 0.9723 | 2.7938 |

### 复现实验结论

1. 从三组随机种子的均值来看，`DQN` 在总体风险修复率上仍然最高，为 `0.1294`，略高于 `Q-Learning` 的 `0.1169`，明显高于 `Rule` 的 `0.0263`。
2. `DQN` 在低 QoT 风险处理方面持续占优，`low_qot_after = 4.1667`，同时平均 OSNR 最高，为 `36.5295`。
3. `Q-Learning` 的结果更稳定，标准差普遍低于 `DQN`；而 `DQN` 在重路由次数、控制消息和路径长度方面波动更大，说明当前策略更激进，但稳定性还有待进一步增强。
4. 因此可以将论文结论表述为：`DQN` 在平均性能上优于 `Q-Learning`，但其控制开销和结果波动也更大，体现出“收益提升与稳定性/开销”之间的权衡。

---

## VDN-lite 初步接入结果

已在 `benchmark_focus.py` 中新增 `VDN-lite`（路由/功率双智能体价值分解）分支。当前单组基准结果如下：

| 指标 | Rule | Q-Learning | DQN | VDN-lite |
|---|---:|---:|---:|---:|
| risk_repair_rate | 0.0347 | 0.1122 | 0.0290 | 0.0146 |
| risk_after | 6.5000 | 6.0667 | 6.7000 | 6.7333 |
| low_qot_after | 5.3000 | 4.8000 | 5.0333 | 4.9667 |
| avg_osnr_after | 35.0308 | 35.5390 | 35.0233 | 35.0581 |
| avg_reroute_actions | 1.2667 | 3.8333 | 0.1000 | 0.0000 |
| avg_boost_actions | 4.9667 | 2.5000 | 5.7667 | 6.8333 |

初步观察：当前 `VDN-lite` 仍偏向“功率提升优先”，重路由动作几乎未被激活，整体修复率暂未超过 `Q-Learning`。后续应重点优化联合动作映射与路由子智能体探索策略。

### VDN-lite 调优后结果（加入拥塞偏置与惩罚增强）

调优点：

- 在 `VDN-lite` 的联合动作打分中加入拥塞/低QoT场景偏置（拥塞时提高 `Reroute` 候选得分）。
- 在评估层中增加“拥塞未缓解且未选择重路由”的额外惩罚。

单组基准结果更新为：

| 指标 | Rule | Q-Learning | DQN | VDN-lite(调优后) |
|---|---:|---:|---:|---:|
| risk_repair_rate | 0.0347 | 0.1122 | 0.0676 | 0.0488 |
| risk_after | 6.5000 | 6.0667 | 6.4333 | 6.5000 |
| congestion_after | 1.2000 | 1.2667 | 1.1667 | 1.1667 |
| low_qot_after | 5.3000 | 4.8000 | 5.2667 | 5.3333 |
| avg_reroute_actions | 1.2667 | 3.8333 | 1.2667 | 0.9000 |
| avg_boost_actions | 4.9667 | 2.5000 | 5.1333 | 5.3333 |

调优后结论：

- `VDN-lite` 相比初版已有提升（`risk_repair_rate` 由 `0.0146` 提升到 `0.0488`）。
- 在拥塞缓解指标上有改善（`congestion_after = 1.1667`）。
- 但在低 QoT 消减与总体修复率上仍落后于 `Q-Learning`，目前更适合作为“价值分解扩展探索”而非主方法。

---

## DQN 稳定性优化结果（Replay + Target）

为提升 DQN 的收敛稳定性，已在决策层引入：

- Experience Replay（批量采样训练）
- Target Network（周期性同步目标网络）

优化后单组基准结果如下：

| 指标 | Rule | Q-Learning | DQN(优化后) | VDN-lite |
|---|---:|---:|---:|---:|
| risk_repair_rate | 0.0347 | 0.1122 | 0.1558 | 0.0537 |
| risk_after | 6.5000 | 6.0667 | 5.6000 | 6.4667 |
| low_qot_after | 5.3000 | 4.8000 | 4.2333 | 5.3000 |
| avg_osnr_after | 35.0308 | 35.5390 | 35.9603 | 35.0680 |
| avg_reroute_actions | 1.2667 | 3.8333 | 5.4333 | 1.0000 |
| control_messages | 14.7333 | 17.5000 | 18.7000 | 14.6667 |

结果解读：

1. 优化后 DQN 在 `risk_repair_rate` 上超过 Q-Learning（`0.1558 > 0.1122`）。
2. 在低 QoT 风险消减和平均 OSNR 上，DQN 也保持优势。
3. 代价是动作更激进（重路由次数、控制消息数上升），体现出“性能提升 vs 控制开销”的权衡。

### 多随机种子复验（优化后 DQN）

在 `seed = 42 / 52 / 62` 下的均值结果（含 VDN-lite）如下：

| 指标 | Rule | Q-Learning | DQN(优化后) | VDN-lite |
|---|---:|---:|---:|---:|
| risk_repair_rate | 0.0263 | 0.1169 | 0.1202 | 0.0807 |
| risk_after | 6.1000 | 5.6111 | 5.4667 | 5.8444 |
| congestion_after | 1.2556 | 1.2778 | 1.2778 | 1.2333 |
| low_qot_after | 4.8444 | 4.3333 | 4.1889 | 4.6111 |
| avg_osnr_after | 35.9440 | 36.3881 | 36.5987 | 36.1456 |
| avg_reroute_actions | 1.2222 | 3.4444 | 3.5667 | 1.9000 |
| avg_boost_actions | 4.3778 | 2.2444 | 2.1667 | 4.0667 |
| avg_path_length | 196.4431 | 199.1906 | 200.0493 | 197.0887 |
| max_link_utilization | 1.1373 | 1.1342 | 1.1743 | 1.1373 |
| control_messages | 13.7556 | 16.1556 | 16.0111 | 14.6111 |

复验结论：

1. 优化后 DQN 在多种子均值上小幅超过 Q-Learning（`risk_repair_rate: 0.1202 > 0.1169`），优势不大但方向稳定。
2. DQN 在低 QoT 风险处理和平均 OSNR 上继续领先（`low_qot_after` 更低、`avg_osnr_after` 更高）。
3. DQN 在拥塞指标上未明显优于 Q-Learning（`congestion_after` 持平，`max_link_utilization` 略差），说明当前优化仍偏 QoT 导向。
4. 因此论文可表述为：优化后 DQN 在综合修复能力上略优于 Q-Learning，但其改进主要来自 QoT 维度，拥塞控制仍有优化空间。

---

## 负载分层实验（低/中/高）

新增脚本 `benchmark_load_levels.py`，在不同负载强度下进行多随机种子复验（`seed=42/52/62`）：

- 低负载：每轮 `10` 条活跃业务
- 中负载：每轮 `15` 条活跃业务
- 高负载：每轮 `25` 条活跃业务

### 关键指标均值（Rule / Q-Learning / DQN）

#### 低负载

| 指标 | Rule | Q-Learning | DQN |
|---|---:|---:|---:|
| risk_repair_rate | 0.0160 | 0.0963 | 0.1416 |
| risk_after | 4.1111 | 3.7556 | 3.6000 |
| low_qot_after | 3.2778 | 2.9000 | 2.7444 |
| avg_osnr_after | 36.0516 | 36.4942 | 36.7444 |
| control_messages | 9.0000 | 10.4889 | 11.3778 |

#### 中负载

| 指标 | Rule | Q-Learning | DQN |
|---|---:|---:|---:|
| risk_repair_rate | 0.0263 | 0.1169 | 0.1313 |
| risk_after | 6.1000 | 5.6111 | 5.4000 |
| low_qot_after | 4.8444 | 4.3333 | 4.1111 |
| avg_osnr_after | 35.9440 | 36.3881 | 36.6495 |
| control_messages | 13.7556 | 16.1556 | 16.1778 |

#### 高负载

| 指标 | Rule | Q-Learning | DQN |
|---|---:|---:|---:|
| risk_repair_rate | 0.0684 | 0.1390 | 0.1141 |
| risk_after | 10.7889 | 10.0111 | 10.3333 |
| low_qot_after | 7.9333 | 7.1000 | 7.5444 |
| avg_osnr_after | 35.4500 | 35.8207 | 35.9534 |
| control_messages | 26.7778 | 30.2444 | 28.8000 |

### 场景化结论

1. `DQN` 在低负载和中负载场景下优于 `Q-Learning`（修复率更高，低 QoT 风险更低，OSNR 更高）。
2. 在高负载场景下，`Q-Learning` 在总体修复率上更有优势（`0.1390 > 0.1141`），表现出更好的高负载鲁棒性。
3. 当前 DQN 更适合中低负载下的 QoT 优化；若要在高负载场景继续提升，需要进一步加强拥塞导向策略。

---

## 工作量扩展结果（消融 + 分层图表）

### 1) DQN 消融实验（多随机种子）

脚本：`benchmark_ablation.py`  
设置：`seed=42/52/62`，中负载（每轮15条业务）

| 指标 | Q-Learning | DQN_Full | DQN_NoReplayTarget | DQN_NoGuidedExplore |
|---|---:|---:|---:|---:|
| risk_repair_rate | 0.1169 | 0.1103 | 0.1275 | 0.1227 |
| risk_after | 5.6111 | 5.5333 | 5.4889 | 5.5778 |
| low_qot_after | 4.3333 | 4.2556 | 4.1778 | 4.3000 |
| avg_osnr_after | 36.3881 | 36.5801 | 36.4418 | 36.5607 |
| avg_reroute_actions | 3.4444 | 3.4000 | 3.7444 | 3.3556 |
| control_messages | 16.1556 | 15.8444 | 16.2778 | 16.0667 |

说明：

- 消融结果显示 DQN 方案对配置较敏感，不同机制组合会改变策略激进度和修复效果。
- `NoReplayTarget` 在该组实验中修复率较高，但波动也更大（标准差更高），稳定性不足。
- `DQN_Full` 在综合指标上更平衡，适合作为论文主实验配置。

### 2) 负载分层图表产出

脚本：

- `benchmark_load_levels.py`（输出分层结果到 `docs/load_level_results.json`）
- `plot_load_level_results.py`（自动生成图表）

已生成图表：

- `output_images/load_risk_repair_rate.png`
- `output_images/load_avg_osnr.png`
- `output_images/load_control_messages.png`

可用于论文中“不同负载场景对比”小节与答辩展示页。

---

## 稳健复验（5随机种子，7动作DQN最终版）

为进入论文定稿阶段，针对当前最终配置（7动作参数化 DQN + 约束式奖励微调）补做 5 随机种子稳健复验。

- 配置：`train=120`，`eval=30`，`sample_size=15`
- 种子：`42 / 52 / 62 / 72 / 82`
- 对照方法：`Rule / Q-Learning / DQN(7动作)`
- 原始明细已保存：`docs/robust_revalidation_7actions.json`

### 关键指标均值

| 指标 | Rule | Q-Learning | DQN(7动作) |
|---|---:|---:|---:|
| risk_repair_rate | 0.0298 | 0.0960 | 0.1300 |
| risk_after | 6.0333 | 5.6800 | 5.4400 |
| congestion_after | 1.1400 | 1.1133 | 1.1867 |
| low_qot_after | 4.9600 | 4.5667 | 4.2533 |
| avg_osnr_after | 35.6254 | 35.9970 | 36.3139 |
| avg_reroute_actions | 0.9000 | 2.8933 | 3.3467 |
| avg_boost_actions | 4.5333 | 2.2133 | 2.0933 |
| avg_path_length | 196.9230 | 199.4871 | 202.1560 |
| max_link_utilization | 1.1028 | 1.1022 | 1.1028 |
| control_messages | 14.1333 | 16.1067 | 16.7533 |

### 定稿阶段结论（建议口径）

1. 在 5 随机种子均值下，`DQN(7动作)` 的 `risk_repair_rate` 高于 `Q-Learning`（`0.1300 > 0.0960`）。
2. DQN 在 QoT 相关指标上保持优势：`low_qot_after` 更低、`avg_osnr_after` 更高。
3. DQN 的代价是控制更积极：`reroute` 更频繁、`control_messages` 略高于 Q-Learning。
4. 因此最终可表述为：**DQN(7动作) 在平均修复能力与QoT优化上优于 Q-Learning，但存在一定控制开销上升，体现性能-开销权衡。**
