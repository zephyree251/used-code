# 文件名: intent.py
class IntentAgent:
    def __init__(self):
        # 定义意图模板
        self.intent_templates = {
            "Low_QoT": {
                "goal": "Restore_Service_Quality",
                "target_metric": "OSNR > Threshold",
                "constraints": ["Latency_Increase < 5ms", "Priority_High"],
                "strategy_hint": "Try_Power_Adjustment_First"
            },
            "Congestion": {
                "goal": "Balance_Traffic_Load",
                "target_metric": "Link_Util < 80%",
                "constraints": ["No_Packet_Loss"],
                "strategy_hint": "Reroute_To_Longer_Path"
            }
        }

    def parse_intent(self, risk_report):
        """
        输入：Step 1 的风险报告
        输出：Step 2 的治理意图
        """
        intents = []
        for risk in risk_report:
            r_type = risk.get('risk_type')

            # 匹配模板
            if r_type in self.intent_templates:
                template = self.intent_templates[r_type]
                intent = {
                    "intent_id": f"INT-{risk['service_id']}",
                    "target_obj": risk['service_id'],
                    "source_risk": risk,  # 把原始风险也带上
                    "generated_goal": template['goal'],
                    "required_constraints": template['constraints']
                }
                intents.append(intent)
        return intents