import matplotlib.pyplot as plt
import numpy as np
import json
import os

# 1. 模拟读取当前的 Q-Table 数据 (来自最新日志)
# 为了演示，直接使用硬编码的最新值
q_data = {
    "Congestion (Core VIP)": [-10.0, -20.0, 32.8],  # Maintain, Boost, Reroute
    "Congestion (Access)": [-10.0, -20.0, 32.5],
    "Low QoT (Access)": [-6.9, 10.0, 6.99]
}

actions = ["Maintain", "Power Boost", "Reroute"]
states = list(q_data.keys())

# 设置绘图风格
plt.figure(figsize=(10, 6))
# plt.style.use('seaborn-v0_8-whitegrid') # matplotlib 3.6+ style name changed, use default if unsure or check version
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 准备数据
bar_width = 0.25
index = np.arange(len(states))

# 绘制柱状图
for i, action in enumerate(actions):
    values = [q_data[state][i] for state in states]
    plt.bar(index + i * bar_width, values, bar_width, label=action)

# 添加标签和标题
plt.xlabel('Network State & Service Level', fontsize=12)
plt.ylabel('Q-Value (Expected Reward)', fontsize=12)
plt.title('RL Agent Decision Preference Analysis', fontsize=14)
plt.xticks(index + bar_width, states)
plt.legend()

# 添加数值标签
for i, action in enumerate(actions):
    values = [q_data[state][i] for state in states]
    for j, val in enumerate(values):
        plt.text(index[j] + i * bar_width, val + (1 if val >= 0 else -2), 
                 f'{val:.1f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

# 保存图片
plt.tight_layout()
output_path = 'rl_decision_preference.png'
plt.savefig(output_path, dpi=300)
print(f"Chart saved to {os.path.abspath(output_path)}")
