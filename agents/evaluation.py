from core.context import BaseAgent
from utils.decision_features import (
    OSNR_THRESHOLD_DB,
    compute_max_link_utilization,
    encode_state,
    find_service,
    snapshot_service_metrics,
)


class EvaluationAgent(BaseAgent):
    """
    Step 7: 评估与训练智能体
    职责：
    1. 基于执行后的真实网络状态计算奖励。
    2. 调用决策智能体的 learn 方法更新策略。
    """

    def __init__(self, name, rl_agent):
        super().__init__(name)
        self.rl_agent = rl_agent

    @staticmethod
    def _parse_action(action: str):
        if action.startswith("Action_Power_Boost_"):
            tail = action.replace("Action_Power_Boost_", "").replace("dB", "").replace("p", ".")
            try:
                return "Action_Power_Boost", {"boost_db": float(tail)}
            except ValueError:
                return "Action_Power_Boost", {"boost_db": 2.0}

        if action.startswith("Action_Reroute_K"):
            tail = action.replace("Action_Reroute_K", "")
            try:
                return "Action_Reroute", {"k_paths": max(1, int(tail))}
            except ValueError:
                return "Action_Reroute", {"k_paths": 3}

        if action == "Action_Power_Boost":
            return "Action_Power_Boost", {"boost_db": 2.0}
        if action == "Action_Reroute":
            return "Action_Reroute", {"k_paths": 5}
        if action in ("Action_Maintain", "Action_Do_Nothing"):
            return "Action_Maintain", {}
        return action, {}

    @staticmethod
    def _find_intent(context, decision):
        service_id = str(decision.get("service_id", ""))
        for intent in context.intents:
            target = str(intent.get("target_service", intent.get("service_id", "")))
            if target == service_id:
                return intent
        return None

    def _compute_reward(self, context, decision, intent):
        action = decision["action"]
        base_action, action_params = self._parse_action(action)
        service = find_service(context, decision["service_id"])
        if service is None or intent is None:
            return -10.0, None, True

        issue_type = intent.get("issue_type", "Low_OSNR")
        user_level = intent.get("user_level", "Access_Aggregation")
        before = decision.get("pre_metrics") or snapshot_service_metrics(context, service)
        after = snapshot_service_metrics(context, service)

        osnr_before = float(before.get("osnr", 0.0))
        osnr_after = float(after.get("osnr", 0.0))
        util_before = float(before.get("max_util", compute_max_link_utilization(context, service, include_self=True)))
        util_after = float(after.get("max_util", compute_max_link_utilization(context, service, include_self=True)))
        path_before = float(before.get("path_length", 0.0))
        path_after = float(after.get("path_length", 0.0))
        path_growth_ratio = 0.0 if path_before <= 1e-6 else max(0.0, (path_after - path_before) / path_before)
        reroute_failed = base_action == "Action_Reroute" and before.get("path", []) == after.get("path", [])
        boost_db = float(action_params.get("boost_db", 0.0))
        k_paths = int(action_params.get("k_paths", 1))
        reroute_succeeded = base_action == "Action_Reroute" and (not reroute_failed)

        resolved_qot = osnr_after >= OSNR_THRESHOLD_DB
        resolved_congestion = util_after <= 0.80

        reward = 0.0
        if issue_type == "Congestion":
            reward += 28.0 * (util_before - util_after)
            reward += 10.0 if resolved_congestion else -5.0
            if base_action == "Action_Reroute":
                reward -= (0.9 + 0.22 * k_paths)
                if k_paths >= 5:
                    # 对大候选集重路由施加额外开销惩罚，抑制过度探索
                    reward -= 1.2
            reward -= (2.5 + 1.0 * boost_db) if base_action == "Action_Power_Boost" else 0.0
            reward -= 12.0 if reroute_failed else 0.0
            if not resolved_congestion and base_action != "Action_Reroute":
                # 拥塞场景下如果不尝试重路由，增加额外惩罚
                reward -= 10.0
            if resolved_congestion and reroute_succeeded:
                # 拥塞已解决后，继续积极重路由会引入额外控制开销
                reward -= 1.0
            done = resolved_congestion
        else:
            reward += 2.0 * (osnr_after - osnr_before)
            reward += 8.0 if resolved_qot else -4.0
            reward -= (1.0 + 0.7 * boost_db) if base_action == "Action_Power_Boost" else 0.0
            reward -= (2.2 + 0.3 * k_paths) if base_action == "Action_Reroute" else 0.0
            reward -= 5.0 * path_growth_ratio
            reward -= 6.0 if reroute_failed else 0.0
            done = resolved_qot

        if base_action == "Action_Maintain" and not done:
            reward -= 2.0
        if "Core_VIP" in user_level and base_action == "Action_Reroute":
            reward -= 5.0

        next_state = encode_state(context, service, intent)
        return reward, next_state, done

    def process(self, context):
        if not context.decisions:
            return

        print("\n   -> [评估层] 正在基于执行后网络状态计算奖励...")

        for decision in context.decisions:
            action_idx = decision["action_idx"]
            state = decision["state_snapshot"]
            intent = self._find_intent(context, decision)
            reward, next_state, done = self._compute_reward(context, decision, intent)

            print(
                f"      └─ {decision['service_id']} | {decision['action']} | "
                f"Reward={reward:.2f} | Done={done}"
            )
            self.rl_agent.learn(state, action_idx, reward, next_state=next_state, done=done)