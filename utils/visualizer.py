import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


class ProcessVisualizer:
    """
    光网络运行过程可视化工具
    自动解析 ActionAgent 的执行日志，并生成重路由对比图。
    """

    def __init__(self, output_dir="output_images"):
        self.output_dir = output_dir
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def visualize_reroute_from_logs(self, context):
        """解析上下文中成功重路由的日志，并绘制拓扑对比图"""
        if not hasattr(context, 'action_logs') or not context.action_logs:
            return

        # 过滤出所有包含了 "Reroute" 的成功日志
        reroute_logs = [log for log in context.action_logs if "Reroute" in log]

        if not reroute_logs:
            print("   📊 [可视化] 本轮无重路由操作，跳过绘图。")
            return

        print(f"\n   📊 [可视化] 正在生成 {len(reroute_logs)} 张重路由拓扑对比图...")

        for log in reroute_logs:
            try:
                # 解析字符串: "🔄 INT-219 执行 Reroute | 旧: N201->N203 | 新: N201->N202->N203 | 新OSNR: 25.78dB"
                parts = log.split(" | ")
                service_id = parts[0].split(" ")[1]
                old_path_str = parts[1].replace("旧: ", "").strip()
                new_path_str = parts[2].replace("新: ", "").strip()

                old_path = old_path_str.split("->")
                new_path = new_path_str.split("->")

                # 调用绘图核心函数
                self._draw_single_comparison(context.graph, service_id, old_path, new_path)
            except Exception as e:
                print(f"      ❌ [可视化错误] 无法解析并绘制日志 '{log}': {e}")

    def _draw_single_comparison(self, G, service_id, old_path, new_path):
        """核心绘图逻辑"""
        plt.figure(figsize=(12, 8))

        # 使用 Kamada-Kawai 物理力学布局，这种布局画出来的光网络拓扑最舒展、最漂亮
        pos = nx.kamada_kawai_layout(G)

        # 1. 画底层网络灰度背景 (106个节点，200多条边)
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='lightgray')
        nx.draw_networkx_edges(G, pos, edge_color='whitesmoke', alpha=0.5, arrows=False)

        # 2. 画旧路径 (红色虚线：代表拥塞、被抛弃的劣质路径)
        old_edges = [(old_path[i], old_path[i + 1]) for i in range(len(old_path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=old_edges, edge_color='red', style='dashed', width=2)
        nx.draw_networkx_nodes(G, pos, nodelist=old_path, node_size=50, node_color='salmon')

        # 3. 画新路径 (绿色实线：代表新开辟的安全/高 OSNR 路径)
        new_edges = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=new_edges, edge_color='green', width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=new_path, node_size=60, node_color='lightgreen')

        # 4. 高亮起点 (Source) 和 终点 (Target)
        src, dst = new_path[0], new_path[-1]
        nx.draw_networkx_nodes(G, pos, nodelist=[src, dst], node_size=120, node_color='gold', edgecolors='black')
        nx.draw_networkx_labels(G, pos, labels={src: 'Src', dst: 'Dst'}, font_size=10, font_weight='bold')

        # 5. 添加标题和图例，保存为高清图片
        plt.title(f"Autonomous Rerouting Execution: {service_id}", fontsize=16, fontweight='bold')

        # 伪造图例句柄
        import matplotlib.lines as mlines
        red_line = mlines.Line2D([], [], color='red', linestyle='dashed', linewidth=2,
                                 label='Old Path (Congested/Low QoT)')
        green_line = mlines.Line2D([], [], color='green', linewidth=3, label='New Path (Safe Rerouted)')
        plt.legend(handles=[red_line, green_line], loc='best')

        plt.axis('off')
        plt.tight_layout()

        # 自动保存到 output_images 文件夹中
        filepath = os.path.join(self.output_dir, f"reroute_{service_id}.png")
        plt.savefig(filepath, dpi=300)  # dpi=300 保证论文打印高清
        plt.close()  # 关掉画布，防止批量运行时内存溢出

        print(f"      📸 已生成拓扑对比图: {filepath}")