import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# 确保输出目录存在
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 设置中文字体 (优先尝试 Windows 常见中文字体)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
# 稍微减小高度，移除底部标题的空间
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0.5, 4.5) # 调整Y轴范围，让图形居中
ax.axis('off')  # 关闭坐标轴

# --- 绘图函数定义 ---

def draw_block(x, y, width, height, text, color='#E6E6FA', text_color='black'):
    """绘制矩形模块 (Tx/Rx)"""
    # 使用圆角矩形，更现代
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor='#555555', facecolor=color, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=14, fontweight='normal', color=text_color, zorder=3)

def draw_fiber(x_start, x_end, y):
    """绘制光纤 (螺旋线表示)"""
    # 绘制直线作为光纤主体
    # 使用深灰色 (#666666) 而不是亮橙色
    ax.plot([x_start, x_end], [y, y], color='#666666', linewidth=1.5, zorder=1)
    
    # 在中间画个螺旋圈圈表示光纤盘
    mid = (x_start + x_end) / 2
    t = np.linspace(0, 4*np.pi, 50)
    # 简单的螺旋参数方程
    x_coil = mid + (t - 2*np.pi) / (4*np.pi) * 0.8
    y_coil = y + 0.3 * np.sin(t)
    ax.plot(x_coil, y_coil, color='#666666', linewidth=1.2, zorder=1)
    
    # 标注 (字体稍微调小，颜色淡化)
    ax.text(mid, y + 0.5, "Fiber Span\n(0.2dB/km)", ha='center', va='bottom', fontsize=9, color='#444444')

def draw_edfa(x, y, size=1.0):
    """绘制 EDFA (三角形)"""
    # 顶点坐标
    points = [[x, y - size/2], [x, y + size/2], [x + size, y]]
    # 使用淡绿色 (#F0FFF0 蜜瓜色 或 #E8F8F5)
    triangle = patches.Polygon(points, closed=True, linewidth=1.5, edgecolor='#555555', facecolor='#E8F8F5', zorder=2)
    ax.add_patch(triangle)
    
    # 内部文字
    ax.text(x + size*0.4, y, "EDFA", ha='center', va='center', fontsize=9, fontweight='normal', rotation=0, zorder=3)
    
    # 底部标注 NF
    ax.text(x + size/2, y - size/2 - 0.2, "NF=4.5dB", ha='center', va='top', fontsize=9, color='#444444')

# --- 开始绘制 ---

# 统一配色方案 (莫兰迪色系 / 商务蓝灰)
COLOR_TX = '#E8F1F5'  # 淡蓝灰 (同 OODA 智能体)
COLOR_RX = '#E8F1F5'  # 淡蓝灰
# COLOR_EDFA = '#E8F8F5' # 在函数里硬编码了

# 1. 发射机 Tx
tx_x, tx_y = 1, 2.5
draw_block(tx_x, tx_y - 0.5, 1, 1, "Tx", color=COLOR_TX)

# 2. 第一段光纤
fiber1_start = tx_x + 1
fiber1_end = fiber1_start + 2.5
draw_fiber(fiber1_start, fiber1_end, tx_y)

# 3. 第一个 EDFA
edfa1_x = fiber1_end
draw_edfa(edfa1_x, tx_y, size=1.2)

# 4. 第二段光纤
fiber2_start = edfa1_x + 1.2
fiber2_end = fiber2_start + 2.5
draw_fiber(fiber2_start, fiber2_end, tx_y)

# 5. 第二个 EDFA
edfa2_x = fiber2_end
draw_edfa(edfa2_x, tx_y, size=1.2)

# 6. 第三段光纤
fiber3_start = edfa2_x + 1.2
fiber3_end = fiber3_start + 2.5
draw_fiber(fiber3_start, fiber3_end, tx_y)

# 7. 接收机 Rx
rx_x = fiber3_end
draw_block(rx_x, tx_y - 0.5, 1, 1, "Rx", color=COLOR_RX)

# --- 装饰与标注 ---

# 信号流向箭头 (更细致，深灰色)
arrow_params = dict(head_width=0.1, head_length=0.2, fc='#333333', ec='#333333', zorder=5)
ax.arrow(tx_x + 1, tx_y, 0.5, 0, **arrow_params)
ax.arrow(fiber1_end + 1.2, tx_y, 0.5, 0, **arrow_params)
ax.arrow(fiber2_end + 1.2, tx_y, 0.5, 0, **arrow_params)

# 移除内部标题文字，交给外部排版软件处理

# 保存图片
save_path = os.path.join(output_dir, 'osnr_model_schematic_v2.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Schematic saved to {os.path.abspath(save_path)}")
