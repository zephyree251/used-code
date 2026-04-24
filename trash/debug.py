import random
import os
import sys

# 确保 Python 能找到 core, agents, utils 文件夹
sys.path.append(os.getcwd())

from core.context import SystemContext

# 假设你的 loader 在 utils 文件夹下，如果还在根目录，改成 from tefnet_loader import TefnetLoader
try:
    from utils.tefnet_loader import TefnetLoader
except ImportError:
    from tefnet_loader import TefnetLoader

# 只导入前三个 Agent
from agents.perception import PerceptionAgent
from agents.intent import IntentAgent
from agents.evidence import EvidenceAgent


def run_debug():
    print("🧪 [测试模式] 正在启动 Step 1-3 联合调试...")

    # 1. 初始化上下文
    context = SystemContext()

    # 2. 加载数据 (注意路径，如果你的csv在data下，要写 'data/xxx.csv')
    print(">>> 正在加载数据...")
    try:
        loader = TefnetLoader('data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
        context.graph = loader.load_topology()
        context.all_demands = loader.load_traffic_demands()
        print(f">>> 数据加载成功。节点数: {context.graph.number_of_nodes()}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("💡 提示: 请检查 CSV 文件是否在 'data' 文件夹下，且文件名正确。")
        return

    # 3. 组装仅包含 Step 1-3 的流水线
    agents = [
        PerceptionAgent("Perception"),
        IntentAgent("Intent"),
        EvidenceAgent("Evidence")
    ]

    # 4. 准备测试数据
    # 随机抽样，但如果想必现故障，Loader 里可能需要包含 ID 248/452
    # 或者我们依赖 PerceptionAgent 里的强制故障注入逻辑
    if len(context.all_demands) > 15:
        context.active_services = random.sample(context.all_demands, 15)
    else:
        context.active_services = context.all_demands

    # 5. 运行循环
    print("\n🚀 开始流水线执行...")
    for agent in agents:
        print(f"\n🔹 执行 [{agent.name}]...")
        agent.process(context)

    # 6. 【核心】结果验证 (Validation)
    print("\n" + "=" * 30)
    print("📊 验证报告 (Verification Report)")
    print("=" * 30)

    # 验证 Step 1
    if len(context.risk_report) > 0:
        print(f"✅ Step 1 感知层: 通过! 发现了 {len(context.risk_report)} 个风险。")
        print(f"   - 第一个风险 ID: {context.risk_report[0]['id']}")
        print(f"   - 物理数值: {context.risk_report[0]['current_val']} dB")
    else:
        print("❌ Step 1 感知层: 未发现风险 (可能故障注入没触发，或者阈值太低)。")

    # 验证 Step 2
    if len(context.intents) > 0:
        print(f"✅ Step 2 意图层: 通过! 生成了 {len(context.intents)} 条意图。")
        first_intent = context.intents[0]
        print(f"   - 意图 ID: {first_intent.get('intent_id')}")
        print(f"   - 判定等级: {first_intent.get('user_level')} (关键检查点!)")

        # 深度检查: 如果是 248，必须是 Core_VIP
        if "248" in str(first_intent.get('intent_id')) and "Core_VIP" not in first_intent.get('user_level', ''):
            print("   ⚠️ 警告: ID 248 被识别为了 Access，请检查 Prompt 规则或路径数据。")
    else:
        print("❌ Step 2 意图层: 未生成意图 (API 调用失败或逻辑错误)。")

    # 验证 Step 3
    if len(context.evidence) > 0:
        print(f"✅ Step 3 检索层: 通过! 检索到了 {len(context.evidence)} 条证据链。")
        print(f"   - 检索内容: {context.evidence[0]['related_rules'][:60]}...")
    else:
        print("❌ Step 3 检索层: 未匹配到规则 (请检查 knowledge_base.json 路径)。")

    print("=" * 30)


if __name__ == "__main__":
    run_debug()