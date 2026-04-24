import operator
import random
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END

# 引入核心模块
from core.context import SystemContext
from utils.tefnet_loader import TefnetLoader

# 引入所有 Agent (Step 1-7 全家福)
from agents.perception import PerceptionAgent
from agents.intent import IntentAgent
from agents.evidence import EvidenceAgent
from agents.decision_rl import RLDecisionAgent
from agents.expert import ExpertAgent  # <--- [新增] Step 5
from agents.execution import ExecutionAgent  # <--- [新增] Step 6
from agents.evaluation import EvaluationAgent


# ==========================================
# 1. 定义图的状态 (State)
# ==========================================
class GraphState(TypedDict):
    context_obj: SystemContext
    step_log: Annotated[List[str], operator.add]


# ==========================================
# 2. 定义节点包装器 (Node Wrappers)
# ==========================================

def node_perception(state: GraphState):
    ctx = state['context_obj']
    agent = PerceptionAgent("Perception")
    agent.process(ctx)
    return {"context_obj": ctx, "step_log": ["Step 1 Done"]}


def node_intent(state: GraphState):
    ctx = state['context_obj']
    agent = IntentAgent("Intent")
    agent.process(ctx)
    return {"context_obj": ctx, "step_log": ["Step 2 Done"]}


def node_evidence(state: GraphState):
    ctx = state['context_obj']
    agent = EvidenceAgent("Evidence")
    agent.process(ctx)
    return {"context_obj": ctx, "step_log": ["Step 3 Done"]}


# 全局 RL 实例，保证记忆(Q表)延续
rl_agent_instance = RLDecisionAgent("Decision(RL)")


def node_decision(state: GraphState):
    ctx = state['context_obj']
    rl_agent_instance.process(ctx)
    return {"context_obj": ctx, "step_log": ["Step 4 Done"]}


# --- [新增] Step 5 节点 ---
def node_expert(state: GraphState):
    ctx = state['context_obj']
    agent = ExpertAgent("Expert")
    agent.process(ctx)
    return {"context_obj": ctx, "step_log": ["Step 5 Done"]}


# --- [新增] Step 6 节点 ---
def node_execution(state: GraphState):
    ctx = state['context_obj']
    agent = ExecutionAgent("Execution")
    agent.process(ctx)
    return {"context_obj": ctx, "step_log": ["Step 6 Done"]}


def node_evaluation(state: GraphState):
    ctx = state['context_obj']
    # 传入 rl_agent_instance 以便更新 Q 表
    agent = EvaluationAgent("Evaluation", rl_agent_instance)
    agent.process(ctx)
    return {"context_obj": ctx, "step_log": ["Step 7 Done"]}


# ==========================================
# 3. 构建图 (Build the Graph)
# ==========================================
def build_ooda_graph():
    # 初始化图
    workflow = StateGraph(GraphState)

    # (A) 添加所有节点
    workflow.add_node("perception", node_perception)
    workflow.add_node("intent", node_intent)
    workflow.add_node("evidence", node_evidence)
    workflow.add_node("decision", node_decision)
    workflow.add_node("expert", node_expert)  # <--- [新增]
    workflow.add_node("execution", node_execution)  # <--- [新增]
    workflow.add_node("evaluation", node_evaluation)

    # (B) 定义连线 (串联 Step 1 -> 7)
    workflow.set_entry_point("perception")

    workflow.add_edge("perception", "intent")
    workflow.add_edge("intent", "evidence")
    workflow.add_edge("evidence", "decision")

    # --- [关键修改] 插入 Expert 和 Execution ---
    workflow.add_edge("decision", "expert")  # 决策后 -> 找专家
    workflow.add_edge("expert", "execution")  # 专家审完 -> 生成执行方案
    workflow.add_edge("execution", "evaluation")  # 执行完 -> 评估效果
    # ------------------------------------------

    workflow.add_edge("evaluation", END)

    # 编译图
    app = workflow.compile()
    return app


# ==========================================
# 4. 运行测试
# ==========================================
if __name__ == "__main__":
    print("🚀 [LangGraph] 正在启动全流程 OODA 自治系统 (Step 1-7)...")

    # 1. 准备数据
    ctx = SystemContext()
    try:
        loader = TefnetLoader('data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
        ctx.graph = loader.load_topology()
        ctx.all_demands = loader.load_traffic_demands()
        print(f">>> 数据加载成功。节点数: {ctx.graph.number_of_nodes()}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        exit()

    ctx.active_services = random.sample(ctx.all_demands, 15)

    # 2. 初始化状态
    initial_state = {
        "context_obj": ctx,
        "step_log": ["System Start"]
    }

    # 3. 构建并运行
    app = build_ooda_graph()

    try:
        print("\n[系统架构图]")
        print(app.get_graph().draw_ascii())
    except Exception:
        print("(Graph visualization skipped)")

    print("\n>>> 开始执行完整工作流...")
    final_state = app.invoke(initial_state)

    print("\n✅ 全流程执行完毕！请检查日志中的 Expert 和 Execution 输出。")