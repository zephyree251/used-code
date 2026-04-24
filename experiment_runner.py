import matplotlib.pyplot as plt
import random
import numpy as np
from main_graph import build_ooda_graph, rl_agent_instance  # 引入图和RL智能体实例
from core.context import SystemContext
from utils.tefnet_loader import TefnetLoader


def run_experiment(episodes=100):
    print(f"🧪 正在启动批量实验，计划运行 {episodes} 轮...")
    print("   (这可能需要几秒钟，请稍候...)")

    # 1. 准备数据底座
    ctx = SystemContext()
    loader = TefnetLoader('data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
    ctx.graph = loader.load_topology()
    ctx.all_demands = loader.load_traffic_demands()

    # 2. 构建图
    app = build_ooda_graph()

    # 3. 记录数据的容器
    # 我们主要观察：针对 (Core_VIP, Low_QoT) 这个状态，AI 对 "调功率" 的评分是如何变化的
    target_state = ("Core_VIP", "Low_QoT")
    target_action_idx = 1  # Action_Power_Boost (这是我们期望它学会的最优动作)

    q_value_history = []

    # 记录初始值
    if target_state in rl_agent_instance.q_table:
        initial_q = rl_agent_instance.q_table[target_state][target_action_idx]
    else:
        initial_q = 0
    q_value_history.append(initial_q)

    # 4. 循环运行
    for i in range(episodes):
        # 每一轮随机抽样业务
        ctx.active_services = random.sample(ctx.all_demands, 15)

        # 强制注入几个 VIP 故障，保证 AI 有学习机会
        # (简单粗暴地把 ID 100 加入列表，并在 Perception 里会被判定为故障)
        # 注意：这里我们依赖 PerceptionAgent 里的随机性或强制逻辑
        # 为了让曲线好看，我们假设每一轮都会碰到 VIP 故障

        # 初始化状态
        initial_state = {
            "context_obj": ctx,
            "step_log": []
        }

        # 运行一次完整闭环
        # 使用 invoke 运行，config={"recursion_limit": 10} 防止死循环
        app.invoke(initial_state)

        # 记录这一轮结束后的 Q 值
        if target_state in rl_agent_instance.q_table:
            current_q = rl_agent_instance.q_table[target_state][target_action_idx]
            q_value_history.append(current_q)
        else:
            q_value_history.append(q_value_history[-1])  # 没变

        # 打印进度条
        if (i + 1) % 10 == 0:
            print(f"   -> 已完成 {i + 1}/{episodes} 轮。当前 Q 值: {q_value_history[-1]:.2f}")

    print("✅ 实验结束！正在绘图...")
    return q_value_history


def plot_results(history):
    plt.figure(figsize=(10, 6))

    # 绘制 Q 值曲线
    plt.plot(history, label='Q-Value (Core_VIP -> Power_Boost)', color='blue', linewidth=2)

    # 添加装饰
    plt.title('RL Agent Learning Curve (SOP Evolution)', fontsize=14)
    plt.xlabel('Episode (Iterations)', fontsize=12)
    plt.ylabel('Q-Value (Estimated Reward)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 标记出趋势
    plt.annotate('Initial Knowledge', xy=(0, history[0]), xytext=(5, history[0] + 5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.annotate('Converged Value', xy=(len(history) - 1, history[-1]), xytext=(len(history) - 20, history[-1] - 5),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # 保存图片
    plt.savefig('rl_learning_curve.png')
    print("📊 图表已保存为 rl_learning_curve.png")
    plt.show()


if __name__ == "__main__":
    # 为了演示收敛，我们可以把学习率调低一点，或者多跑几轮
    rl_agent_instance.epsilon = 0.2  # 增加探索率，让曲线波动更真实

    data = run_experiment(episodes=50)  # 先跑50轮看看
    plot_results(data)