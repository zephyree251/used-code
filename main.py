import os
import random
from utils.tefnet_loader import TefnetLoader
from utils.visualizer import ProcessVisualizer
from core.context import SystemContext

# 导入所有 Agents
from agents.perception import PerceptionAgent
from agents.intent import IntentAgent
from agents.evidence import EvidenceAgent
from agents.decision_rl import DQNDecisionAgent, RLDecisionAgent, RuleDecisionAgent, VDNLiteDecisionAgent
from agents.action import ActionAgent  # <--- 新增：导入 ActionAgent
from agents.evaluation import EvaluationAgent


class OODA_System:
    def __init__(self):
        self.context = SystemContext()
        self.decision_mode = os.environ.get("DECISION_MODE", "dqn").lower()
        self.reset_dqn = os.environ.get("RESET_DQN", "0") == "1"

        print(">>> [System] 正在初始化数据底座...")
        loader = TefnetLoader('data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
        self.context.graph = loader.load_topology()
        self.context.all_demands = loader.load_traffic_demands()

        if self.decision_mode == "rule":
            self.decision_agent = RuleDecisionAgent("Decision(Rule)")
        elif self.decision_mode in ("rl", "qlearning"):
            self.decision_agent = RLDecisionAgent("Decision(Q-Learning)")
        elif self.decision_mode in ("vdn", "vdn-lite"):
            self.decision_agent = VDNLiteDecisionAgent("Decision(VDN-lite)", load_existing=not self.reset_dqn)
        else:
            self.decision_agent = DQNDecisionAgent("Decision(DQN)", load_existing=not self.reset_dqn)

        # 组装完整流水线 (严格遵循 OODA 闭环)
        self.agents = [
            PerceptionAgent("Perception"),  # Step 1: 发现问题
            IntentAgent("Intent"),  # Step 2: 理解意图
            EvidenceAgent("Evidence"),  # Step 3: 收集证据
            self.decision_agent,  # Step 4: 做出决策
            ActionAgent("Action"),  # Step 6: 执行切换 <--- 新增插在这里！
            EvaluationAgent("Evaluation", self.decision_agent)  # Step 7: 评估结果
        ]

    def run_cycle(self):
        print(f"\n🚀 多智能体光网络自治系统启动 (Decision={self.decision_mode})")

        self.context.active_services = random.sample(self.context.all_demands, 15)

        # 清空状态
        self.context.risk_report = []
        self.context.intents = []
        self.context.evidence = []
        self.context.decisions = []
        self.context.action_logs = []  # 新增一个记录动作日志的列表

        for agent in self.agents:
            print(f"\n🔹 执行 [{agent.name}]...")
            agent.process(self.context)

        print("\n✅ 闭环结束。")

        # 打印一下修复成果
        if hasattr(self.context, 'action_logs') and self.context.action_logs:
            print("\n📊 --- 本轮网络修复报告 ---")
            for log in self.context.action_logs:
                print(log)
        visualizer = ProcessVisualizer()
        visualizer.visualize_reroute_from_logs(self.context)

if __name__ == "__main__":
    system = OODA_System()
    system.run_cycle()