import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 确保输出目录存在
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置中文字体 (优先尝试 Windows 常见中文字体)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 15)
ax.axis('off')

# --- 绘图工具函数 ---

def draw_box(x, y, width, height, text, color='#E6E6FA', edge_color='black', linestyle='-'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                  linewidth=2, edgecolor=edge_color, facecolor=color, linestyle=linestyle, zorder=2)
    ax.add_patch(rect)
    # 自动换行处理
    wrapped_text = text.replace(' ', '\n')
    ax.text(x + width/2, y + height/2, wrapped_text, ha='center', va='center', fontsize=12, fontweight='normal', zorder=3)
    return (x + width/2, y + height/2) # 返回中心点

def draw_cylinder(x, y, width, height, text, color='#D3D3D3'):
    """绘制圆柱体代表数据库/环境"""
    # 底部椭圆
    ellipse_bot = patches.Ellipse((x + width/2, y), width, height*0.3, facecolor=color, edgecolor='black', linewidth=2, zorder=2)
    ax.add_patch(ellipse_bot)
    # 顶部椭圆
    ellipse_top = patches.Ellipse((x + width/2, y + height), width, height*0.3, facecolor=color, edgecolor='black', linewidth=2, zorder=4)
    ax.add_patch(ellipse_top)
    # 中间矩形
    rect = patches.Rectangle((x, y), width, height, facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(rect)
    # 侧边线
    ax.plot([x, x], [y, y + height], color='black', linewidth=2, zorder=4)
    ax.plot([x + width, x + width], [y, y + height], color='black', linewidth=2, zorder=4)
    
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12, fontweight='normal', zorder=5)
    return (x + width/2, y + height) # 返回顶部中心点

def draw_arrow(start, end, text="", style='->', linestyle='-', color='black', curvature=0.0):
    """绘制带文字的箭头"""
    arrow = patches.FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={curvature}", 
                                    arrowstyle=style, color=color, linestyle=linestyle, linewidth=2, mutation_scale=20, zorder=5)
    ax.add_patch(arrow)
    
    # 计算文字位置 (简单的中点)
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2
    
    # 根据曲率调整文字位置
    if curvature != 0:
        mid_y += curvature * 2 # 简单偏移
        
    if text:
        # 使用白色背景框遮挡线条，防止重叠
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2), zorder=6)

# --- 1. 绘制底层环境 (TEFNET24) ---
env_x, env_y = 8, 1
env_w, env_h = 4, 2
# 移除 Emoji: 🌐
env_center_top = draw_cylinder(env_x, env_y, env_w, env_h, "TEFNET24\n数字孪生底层\n(拓扑 & 流量)")

# --- 2. 绘制控制层外框 ---
# 调整位置和大小以更紧凑居中
# 智能体包围盒范围大概是 x=[2, 14.5], y=[6, 12.5]
# 设置 rect: x=1.5, y=5.5, w=13.5, h=8.0
control_layer_rect = patches.Rectangle((1.5, 5.5), 13.5, 8.0, linewidth=2, edgecolor='#4682B4', facecolor='#F0F8FF', linestyle='--', zorder=0)
ax.add_patch(control_layer_rect)
# 去掉 "OODA 多智能体闭环控制层" 文字
# ax.text(2, 12.5, "OODA 多智能体闭环控制层", fontsize=14, fontweight='bold', color='#4682B4')

# --- 3. 绘制 6 个智能体 ---
# 布局坐标
agent_w, agent_h = 2.5, 1.5
y_row1 = 9
y_row2 = 6 # 暂时没用到双行布局，用环形布局更合适

# 环形布局坐标
# 感知 (左下)
p_x, p_y = 2, 8
# 意图 (左上)
i_x, i_y = 2, 11
# 证据 (中上)
e_x, e_y = 7, 11
# 决策 (右上)
d_x, d_y = 12, 11
# 执行 (右下)
a_x, a_y = 12, 8
# 评估 (中下)
eval_x, eval_y = 7, 6 # 放在中间下方

# 绘制盒子 (移除 Emoji，统一使用商务蓝风格，避免彩虹色)
# 使用统一的极淡蓝灰色 (#F0F4F8) 作为背景，既显专业又护眼
p_center = draw_box(p_x, p_y, agent_w, agent_h, "感知智能体\n(Perception)", color='#E8F1F5') 
i_center = draw_box(i_x, i_y, agent_w, agent_h, "意图智能体\n(Intent/Qwen)", color='#E8F1F5') 
e_center = draw_box(e_x, e_y, agent_w, agent_h, "证据智能体\n(Evidence/RAG)", color='#E8F1F5') 
d_center = draw_box(d_x, d_y, agent_w, agent_h, "决策智能体\n(Decision/RL)", color='#E8F1F5') 
a_center = draw_box(a_x, a_y, agent_w, agent_h, "执行智能体\n(Action)", color='#E8F1F5') 
eval_center = draw_box(eval_x, eval_y, agent_w, agent_h, "评估智能体\n(Evaluation)", color='#E8F1F5')

# --- 4. 绘制连接线 ---

# 1. 环境 -> 感知 (虚线)
draw_arrow(env_center_top, (p_center[0], p_y), "实时状态扫描", linestyle='--', color='#555555')

# 2. 感知 -> 意图 (实线)
# 修正：感知在(2,8)，意图在(2,11)，应该是从感知顶部连到意图底部
draw_arrow((p_center[0], p_y + agent_h), (i_center[0], i_y), "传递风险报告")

# 3. 意图 -> 证据
draw_arrow((i_center[0] + agent_w/2, i_center[1]), (e_center[0] - agent_w/2, e_center[1]), "提取标准意图")

# 4. 证据 -> 决策
draw_arrow((e_center[0] + agent_w/2, e_center[1]), (d_center[0] - agent_w/2, d_center[1]), "匹配专家 SOP")

# 5. 决策 -> 执行
# 修正：决策在(12,11)，执行在(12,8)，应该是从决策底部连到执行顶部
draw_arrow((d_center[0], d_y), (a_center[0], a_y + agent_h), "下发指令\n(Reroute等)")

# 6. 执行 -> 环境 (虚线，向下)
# 从执行智能体底部连到底层环境顶部右侧
draw_arrow((a_center[0], a_y), (env_x + env_w*0.8, env_y + env_h), "配置覆写", linestyle='--', color='#555555')

# 7. 执行 -> 评估
# 这里稍微绕一下，从执行左侧连到评估右侧
draw_arrow((a_center[0] - agent_w/2, a_center[1]), (eval_center[0] + agent_w/2, eval_center[1]), "执行日志", curvature=-0.2)

# 8. 评估 -> 决策 (反馈闭环，彩色虚线)
# 从评估顶部连回决策底部 (跨越中间区域)
# 使用 connectionstyle="arc3,rad=..." 来画大弧线
# 这里用红色虚线表示核心反馈
fb_arrow = patches.FancyArrowPatch((eval_center[0], eval_center[1] + agent_h/2), (d_center[0] - agent_w/4, d_center[1] - agent_h/2),
                                   connectionstyle="arc3,rad=-0.3", arrowstyle='->', 
                                   color='#DC143C', linestyle='--', linewidth=2.5, mutation_scale=20, zorder=10)
ax.add_patch(fb_arrow)
# 调整位置，避免与中间的文字重叠
ax.text(9.5, 9.2, "计算 Reward 奖励\n更新模型", ha='center', va='center', fontsize=11, fontweight='normal', color='#DC143C',
        bbox=dict(facecolor='white', edgecolor='#DC143C', alpha=0.9, boxstyle='round,pad=0.3'), zorder=11)


# 标题
ax.text(10, 14.5, "图 3：基于 OODA 环的多智能体光网络自治架构", ha='center', va='center', fontsize=16, fontweight='normal')

# 保存
save_path = os.path.join(output_dir, 'ooda_multi_agent_arch.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Architecture diagram saved to {os.path.abspath(save_path)}")
