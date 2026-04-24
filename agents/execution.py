from core.context import BaseAgent
import datetime


class ExecutionAgent(BaseAgent):
    """
    Step 6: 执行编排
    职责：将技术参数转化为可执行的运维工单，并实施 Gate 门控。
    """

    def process(self, context):
        print(f"\n   -> [执行编排层] 正在生成运维工单...")

        final_plan = []
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for decision in context.decisions:
            # 1. Gate 门控机制
            # 如果专家评级为 HIGH，则触发 GATE，暂停自动化
            gate_status = "AUTO_EXECUTE"
            if decision.get('risk_level') == "HIGH":
                gate_status = "MANUAL_APPROVAL_REQUIRED"

            # 2. 生成自然语言步骤 (SOP)
            steps = []
            steps.append(f"Step 1: 登录网管系统，定位业务 ID {decision['intent_id']}.")
            action_name = decision.get('action', decision.get('action_type', 'Action_Maintain'))
            
            if action_name == "Action_Power_Boost":
                steps.append(f"Step 2: 调节光放增益，目标值 +2dBm.")
                steps.append(f"Step 3: 观察 OSNR 是否恢复至阈值以上.")
            elif action_name == "Action_Reroute":
                steps.append(f"Step 2: 计算备用路由，检查波长资源.")
                steps.append(f"Step 3: 下发交叉连接配置，执行业务割接.")
            else:
                steps.append(f"Step 2: 持续监控，暂不执行主动操作.")

            # 3. 组装最终工单
            ticket = {
                "ticket_id": f"TKT-{decision['intent_id']}",
                "time": current_time,
                "action": action_name,
                "risk": decision.get('risk_level'),
                "gate": gate_status,
                "workflow": steps,
                "expert_note": decision.get('expert_opinion')
            }
            final_plan.append(ticket)

            # 打印漂亮的输出
            print(f"      📄 [工单生成] {ticket['ticket_id']}")
            print(f"         状态: {gate_status}")
            print(f"         步骤: {steps[-1]}")

        # 将工单存回上下文，供 Step 7 评估使用（虽然 Step 7 还是看原始 Decision）
        context.execution_plan = final_plan