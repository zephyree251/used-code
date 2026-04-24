import json
import os
import random
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.context import BaseAgent
from utils.decision_features import encode_state, find_service, snapshot_service_metrics


def intent_to_label_state(intent: Dict) -> Tuple[str, str]:
    level = "Core_VIP" if "Core_VIP" in intent.get("user_level", "") else "Access_Aggregation"
    issue = intent.get("issue_type", "Low_OSNR")
    if issue == "Congestion":
        issue = "High_Congestion"
    else:
        issue = "Low_QoT"
    return level, issue


def default_q_table():
    return {
        ("Core_VIP", "Low_QoT"): [-10.0, 50.0, -100.0],
        ("Access_Aggregation", "Low_QoT"): [-10.0, 40.0, 30.0],
        ("Core_VIP", "High_Congestion"): [-10.0, -20.0, 40.0],
        ("Access_Aggregation", "High_Congestion"): [-10.0, -20.0, 40.0],
    }


class BaseDecisionAgent(BaseAgent):
    actions = ["Action_Maintain", "Action_Power_Boost", "Action_Reroute"]

    def _build_decision(self, context, intent: Dict, action_idx: int, state_snapshot, confidence: float):
        service = find_service(context, intent.get("target_service", intent.get("service_id", "Unknown")))
        pre_metrics = snapshot_service_metrics(context, service) if service is not None else None
        return {
            "intent_id": intent.get("intent_id", ""),
            "service_id": intent.get("target_service", intent.get("service_id", "Unknown")),
            "action": self.actions[action_idx],
            "action_idx": action_idx,
            "state_snapshot": state_snapshot,
            "confidence": float(confidence),
            "pre_metrics": pre_metrics,
        }


class RuleDecisionAgent(BaseDecisionAgent):
    """固定规则基线。"""

    def learn(self, state, action_idx, reward, next_state=None, done=True):
        return None

    def process(self, context):
        if not context.intents:
            return

        print("\n   -> [决策层 (Rule)] 使用启发式规则进行动作选择...")
        context.decisions = []

        for intent in context.intents:
            state = intent_to_label_state(intent)
            _, issue = state
            if issue == "High_Congestion":
                action_idx = 2
            elif issue == "Low_QoT":
                action_idx = 1
            else:
                action_idx = 0

            context.decisions.append(self._build_decision(context, intent, action_idx, state, 1.0))


class RLDecisionAgent(BaseDecisionAgent):
    """
    传统表格 Q-Learning，作为轻量基线保留。
    """

    def __init__(self, name, memory_file: str = "data/q_table_memory.json", load_existing: bool = True):
        super().__init__(name)
        self.memory_file = memory_file
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.1

        if load_existing and os.path.exists(self.memory_file):
            self._load_memory()
        else:
            self.q_table = default_q_table()

    def get_state(self, intent):
        return intent_to_label_state(intent)

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

        if random.uniform(0, 1) < self.epsilon:
            print("      │  [Q-Learning] 探索随机动作...")
            return random.choice(range(len(self.actions)))

        q_values = self.q_table[state]
        print(f"      │  [Q-Learning] 当前 Q 值: {q_values}")
        return int(np.argmax(q_values))

    def learn(self, state, action_idx, reward, next_state=None, done=True):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

        old_val = self.q_table[state][action_idx]
        new_val = old_val + self.learning_rate * (reward - old_val)
        self.q_table[state][action_idx] = new_val

        print(f"      [Q-Learning] State={state}, Action={self.actions[action_idx]}")
        print(f"                   └── Q值更新: {old_val:.2f} -> {new_val:.2f}")
        self._save_memory()

    def process(self, context):
        if not context.intents:
            return

        print("\n   -> [决策层 (Q-Learning)] 智能体正在查阅 Q-Table...")
        context.decisions = []

        for intent in context.intents:
            state = self.get_state(intent)
            print(f"      ┌─ 当前状态: {state}")
            action_idx = self.choose_action(state)
            print(f"      └─ 选择动作: {self.actions[action_idx]}")
            confidence = self.q_table[state][action_idx]
            context.decisions.append(self._build_decision(context, intent, action_idx, state, confidence))

    def _save_memory(self):
        save_data = {}
        for key, value in self.q_table.items():
            save_data[f"{key[0]},{key[1]}"] = value

        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as handle:
                json.dump(save_data, handle, indent=2)
        except Exception as exc:
            print(f"      [Warning] Q 表存档失败: {exc}")

    def _load_memory(self):
        try:
            with open(self.memory_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for key_str, value in data.items():
                p0, p1 = key_str.split(",")
                self.q_table[(p0, p1)] = value
            print(f"      [Q-Learning] 已加载 {len(self.q_table)} 条历史经验。")
        except Exception as exc:
            print(f"      [Q-Learning] 读取存档出错: {exc}")
            self.q_table = default_q_table()


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(32, 16)):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNDecisionAgent(BaseDecisionAgent):
    """
    轻量 DQN：只负责高层动作选择，不做候选路径级别的复杂决策。
    """
    actions = [
        "Action_Maintain",
        "Action_Power_Boost_1p0dB",
        "Action_Power_Boost_2p0dB",
        "Action_Power_Boost_3p0dB",
        "Action_Reroute_K1",
        "Action_Reroute_K3",
        "Action_Reroute_K5",
    ]

    def __init__(
        self,
        name,
        state_dim: int = 8,
        memory_file: str = "data/dqn_policy_memory.pt",
        load_existing: bool = True,
        use_replay_target: bool = True,
        use_guided_exploration: bool = True,
        hidden_dims=(32, 16),
    ):
        super().__init__(name)
        self.state_dim = state_dim
        self.memory_file = memory_file
        self.use_replay_target = use_replay_target
        self.use_guided_exploration = use_guided_exploration
        self.hidden_dims = tuple(hidden_dims)
        self.learning_rate = 1e-3
        self.gamma = 0.9
        self.epsilon = 0.8
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.device = torch.device("cpu")
        self.batch_size = 32
        self.warmup_steps = 64
        self.target_update_interval = 100
        self.train_steps = 0

        self.model = DQNNetwork(self.state_dim, len(self.actions), hidden_dims=self.hidden_dims).to(self.device)
        self.target_model = DQNNetwork(self.state_dim, len(self.actions), hidden_dims=self.hidden_dims).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=4000)

        if load_existing:
            self._load_memory()

    def get_state(self, context, intent):
        service = find_service(context, intent.get("target_service", intent.get("service_id", "")))
        if service is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        return encode_state(context, service, intent)

    @staticmethod
    def _guided_exploration_action(state):
        risk_type = float(state[1])           # 1=Congestion, 0=Low_QoT
        osnr_margin = float(state[5])
        max_link_util = float(state[6])
        alt_path_count = float(state[7])

        if risk_type >= 0.5:
            if alt_path_count > 0.0 and max_link_util > 0.85:
                probs = [0.05, 0.03, 0.04, 0.03, 0.15, 0.35, 0.35]
            else:
                probs = [0.65, 0.05, 0.08, 0.07, 0.08, 0.04, 0.03]
        else:
            if osnr_margin < 0.0:
                probs = [0.06, 0.18, 0.48, 0.20, 0.04, 0.02, 0.02]
            else:
                probs = [0.80, 0.07, 0.06, 0.03, 0.02, 0.01, 0.01]

        return int(np.random.choice(list(range(len(DQNDecisionAgent.actions))), p=probs))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            if self.use_guided_exploration:
                print("      │  [DQN] 引导式探索动作...")
                return self._guided_exploration_action(state), 0.0
            print("      │  [DQN] 随机探索动作...")
            return random.choice(range(len(self.actions))), 0.0

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
        action_idx = int(np.argmax(q_values))
        print(f"      │  [DQN] 当前 Q 估计: {np.round(q_values, 3).tolist()}")
        return action_idx, float(q_values[action_idx])

    def learn(self, state, action_idx, reward, next_state=None, done=True):
        if next_state is None:
            next_state = np.zeros(self.state_dim, dtype=np.float32)

        self.replay_buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action_idx),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )

        if len(self.replay_buffer) < self.warmup_steps:
            return

        if self.use_replay_target:
            batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        else:
            batch = [self.replay_buffer[-1]]
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        current_q = self.model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_model = self.target_model if self.use_replay_target else self.model
            next_q = next_model(next_states).max(dim=1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.use_replay_target and self.train_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        print(f"      [DQN] Action={self.actions[action_idx]}, Reward={reward:.2f}, Loss={loss.item():.4f}")
        self._save_memory()

    def process(self, context):
        if not context.intents:
            return

        print("\n   -> [决策层 (DQN)] 轻量神经网络正在选择动作...")
        context.decisions = []

        for intent in context.intents:
            state = self.get_state(context, intent)
            action_idx, confidence = self.choose_action(state)
            print(f"      └─ 选择动作: {self.actions[action_idx]}")
            context.decisions.append(self._build_decision(context, intent, action_idx, state, confidence))

    def set_eval_mode(self):
        self.epsilon = 0.0

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "target_model_state_dict": self.target_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "train_steps": self.train_steps,
                },
                self.memory_file,
            )
        except Exception as exc:
            print(f"      [Warning] DQN 存档失败: {exc}")

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return
        try:
            checkpoint = torch.load(self.memory_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "target_model_state_dict" in checkpoint:
                self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            else:
                self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.train_steps = checkpoint.get("train_steps", 0)
            print("      [DQN] 已加载历史策略参数。")
        except Exception as exc:
            print(f"      [DQN] 读取模型失败，将使用随机初始化: {exc}")


class VDNLiteDecisionAgent(BaseDecisionAgent):
    """
    VDN-lite：将决策拆成两个可学习子智能体（Route / Power），并做价值相加。
    """

    def __init__(
        self,
        name,
        state_dim: int = 8,
        memory_file: str = "data/vdn_lite_policy_memory.pt",
        load_existing: bool = True,
    ):
        super().__init__(name)
        self.state_dim = state_dim
        self.memory_file = memory_file
        self.learning_rate = 1e-3
        self.gamma = 0.9
        self.epsilon = 0.7
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.10
        self.device = torch.device("cpu")

        self.route_feature_idx = [0, 1, 2, 3, 4, 6, 7]
        self.power_feature_idx = [0, 1, 2, 3, 4, 5]

        self.route_net = DQNNetwork(len(self.route_feature_idx), 2).to(self.device)
        self.power_net = DQNNetwork(len(self.power_feature_idx), 2).to(self.device)
        self.optimizer = optim.Adam(
            list(self.route_net.parameters()) + list(self.power_net.parameters()),
            lr=self.learning_rate,
        )
        self.loss_fn = nn.MSELoss()

        if load_existing:
            self._load_memory()

    def get_state(self, context, intent):
        service = find_service(context, intent.get("target_service", intent.get("service_id", "")))
        if service is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        return encode_state(context, service, intent)

    def _split_state(self, state):
        route_state = np.array([state[i] for i in self.route_feature_idx], dtype=np.float32)
        power_state = np.array([state[i] for i in self.power_feature_idx], dtype=np.float32)
        return route_state, power_state

    @staticmethod
    def _joint_to_global(route_action: int, power_action: int) -> int:
        if route_action == 1:
            return 2
        if power_action == 1:
            return 1
        return 0

    @staticmethod
    def _global_to_joint(global_action: int):
        if global_action == 2:
            return 1, 0
        if global_action == 1:
            return 0, 1
        return 0, 0

    @staticmethod
    def _guided_route_action(state):
        risk_type = float(state[1])
        max_link_util = float(state[6])
        alt_path_count = float(state[7])
        if risk_type >= 0.5 and alt_path_count > 0.0 and max_link_util > 0.85:
            return 1 if random.random() < 0.8 else 0
        return 1 if random.random() < 0.2 else 0

    @staticmethod
    def _guided_power_action(state):
        risk_type = float(state[1])
        osnr_margin = float(state[5])
        if risk_type < 0.5 and osnr_margin < 0.0:
            return 1 if random.random() < 0.8 else 0
        return 1 if random.random() < 0.2 else 0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            print("      │  [VDN-lite] 引导式探索动作...")
            route_action = self._guided_route_action(state)
            power_action = self._guided_power_action(state)
            action_idx = self._joint_to_global(route_action, power_action)
            return action_idx, 0.0

        route_state, power_state = self._split_state(state)
        route_tensor = torch.tensor(route_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        power_tensor = torch.tensor(power_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            route_q = self.route_net(route_tensor).squeeze(0).cpu().numpy()
            power_q = self.power_net(power_tensor).squeeze(0).cpu().numpy()

        candidates = [
            (0, 0, route_q[0] + power_q[0]),  # Maintain
            (0, 1, route_q[0] + power_q[1]),  # Power_Boost
            (1, 0, route_q[1] + power_q[0]),  # Reroute
        ]
        # 轻量上下文偏置：拥塞高时鼓励重路由；低QoT时鼓励功率提升
        risk_type = float(state[1])   # 1=Congestion
        osnr_margin = float(state[5])
        max_link_util = float(state[6])
        alt_path_count = float(state[7])
        if risk_type >= 0.5 and max_link_util > 0.85 and alt_path_count > 0.0:
            candidates = [
                (0, 0, candidates[0][2] - 0.20),
                (0, 1, candidates[1][2] - 0.40),
                (1, 0, candidates[2][2] + 0.60),
            ]
        elif risk_type < 0.5 and osnr_margin < 0.0:
            candidates = [
                (0, 0, candidates[0][2] - 0.10),
                (0, 1, candidates[1][2] + 0.40),
                (1, 0, candidates[2][2] - 0.20),
            ]

        best_route, best_power, best_q = max(candidates, key=lambda x: x[2])
        action_idx = self._joint_to_global(best_route, best_power)
        print(
            f"      │  [VDN-lite] RouteQ={np.round(route_q,3).tolist()} "
            f"PowerQ={np.round(power_q,3).tolist()}"
        )
        return action_idx, float(best_q)

    def learn(self, state, action_idx, reward, next_state=None, done=True):
        route_state, power_state = self._split_state(state)
        route_action, power_action = self._global_to_joint(action_idx)

        route_tensor = torch.tensor(route_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        power_tensor = torch.tensor(power_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        route_q_all = self.route_net(route_tensor)
        power_q_all = self.power_net(power_tensor)
        predicted_q_total = route_q_all[0, route_action] + power_q_all[0, power_action]

        with torch.no_grad():
            if next_state is None:
                target_total = reward
            else:
                next_route_state, next_power_state = self._split_state(next_state)
                next_route_tensor = torch.tensor(next_route_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_power_tensor = torch.tensor(next_power_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_route_q = self.route_net(next_route_tensor).squeeze(0)
                next_power_q = self.power_net(next_power_tensor).squeeze(0)
                next_candidates = torch.stack(
                    [
                        next_route_q[0] + next_power_q[0],  # Maintain
                        next_route_q[0] + next_power_q[1],  # Power_Boost
                        next_route_q[1] + next_power_q[0],  # Reroute
                    ]
                )
                next_q_max = float(torch.max(next_candidates).item())
                target_total = reward if done else reward + self.gamma * next_q_max

        target_tensor = torch.tensor(target_total, dtype=torch.float32, device=self.device)
        loss = self.loss_fn(predicted_q_total, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        print(f"      [VDN-lite] Action={self.actions[action_idx]}, Reward={reward:.2f}, Loss={loss.item():.4f}")
        self._save_memory()

    def process(self, context):
        if not context.intents:
            return

        print("\n   -> [决策层 (VDN-lite)] 路由/功率双智能体正在协同选择动作...")
        context.decisions = []
        for intent in context.intents:
            state = self.get_state(context, intent)
            action_idx, confidence = self.choose_action(state)
            print(f"      └─ 选择动作: {self.actions[action_idx]}")
            context.decisions.append(self._build_decision(context, intent, action_idx, state, confidence))

    def set_eval_mode(self):
        self.epsilon = 0.0

    def _save_memory(self):
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            torch.save(
                {
                    "route_state_dict": self.route_net.state_dict(),
                    "power_state_dict": self.power_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                },
                self.memory_file,
            )
        except Exception as exc:
            print(f"      [Warning] VDN-lite 存档失败: {exc}")

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return
        try:
            checkpoint = torch.load(self.memory_file, map_location=self.device)
            self.route_net.load_state_dict(checkpoint["route_state_dict"])
            self.power_net.load_state_dict(checkpoint["power_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            print("      [VDN-lite] 已加载历史策略参数。")
        except Exception as exc:
            print(f"      [VDN-lite] 读取模型失败，将使用随机初始化: {exc}")