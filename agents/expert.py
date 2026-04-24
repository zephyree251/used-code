from core.context import BaseAgent


class ExpertAgent(BaseAgent):
    """
    Step 5: 专家会诊
    职责：对 RL 做出的决策进行风险评估和规则校验。
    重路由人工检测
    重路由路径生成

    """

    def process(self, context):
        print(f"\n   -> [专家会诊层] 正在审查决策风险...")

        # 遍历所有决策
        for decision in context.decisions:
            risk_score = 0
            expert_comments = []

            action = decision.get('action_type')
            confidence = decision.get('confidence', 0)

            # --- 专家规则 1: 动作风险评估 ---
            if action == "Action_Reroute":
                risk_score += 50
                expert_comments.append("⚠️ [风险] 重路由操作可能导致瞬断，需检查波长冲突。")
            elif action == "Action_Power_Boost":
                risk_score += 10
                expert_comments.append("ℹ️ [提示] 功率调节需注意非线性效应积累。")

            # --- 专家规则 2: 信心评估 ---
            # 如果 Q-Learning 的分数很低（刚开始训练），专家要预警
            if confidence < 5.0:
                risk_score += 30
                expert_comments.append("⚠️ [警告] AI 决策信心不足(Q值低)，建议人工复核。")

            # --- 判定最终结果 ---
            decision['risk_level'] = "HIGH" if risk_score > 40 else "LOW"
            decision['expert_opinion'] = "; ".join(expert_comments) if expert_comments else "✅ 方案符合规范，无明显风险。"

            print(f"      └─ 决策ID: {decision['intent_id']} | 动作: {action}")
            print(f"         评级: {decision['risk_level']} (Risk: {risk_score})")
            print(f"         意见: {decision['expert_opinion']}")