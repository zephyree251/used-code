import networkx as nx
import matplotlib.pyplot as plt
from utils.tefnet_loader import TefnetLoader  # 确保路径正确


def draw_reroute_comparison():
    # 1. 加载底图
    loader = TefnetLoader('data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
    G = loader.load_topology()

    # 2. 从刚才的日志里提取新旧路径 (INT-219)
    old_path = ['N201', 'N203', 'N331', 'N332', 'N451', 'N491', 'N472', 'N474']
    new_path = ['N201', 'N202', 'N206', 'N322', 'N401', 'N402', 'N442', 'N471', 'N472', 'N474']

    # 提取所有路径上的边
    old_edges = [(old_path[i], old_path[i + 1]) for i in range(len(old_path) - 1)]
    new_edges = [(new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1)]

    # 3. 开始绘图
    plt.figure(figsize=(12, 8))

    # 尝试使用 Kamada-Kawai 布局（如果你有经纬度坐标更好，如果没有就用算法布局）
    pos = nx.kamada_kawai_layout(G)

    # 画底层灰色拓扑
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray')
    nx.draw_networkx_edges(G, pos, edge_color='whitesmoke', arrows=False)

    # 画旧路径 (红色虚线：拥塞/被抛弃的路)
    nx.draw_networkx_edges(G, pos, edgelist=old_edges, edge_color='red', style='dashed', width=2,
                           label='Old Path (Congested)')
    nx.draw_networkx_nodes(G, pos, nodelist=old_path, node_size=60, node_color='salmon')

    # 画新路径 (绿色实线：安全绕行路)
    nx.draw_networkx_edges(G, pos, edgelist=new_edges, edge_color='green', width=3, label='New Path (Safe Reroute)')
    nx.draw_networkx_nodes(G, pos, nodelist=new_path, node_size=80, node_color='lightgreen')

    # 标出起点和终点
    nx.draw_networkx_nodes(G, pos, nodelist=['N201', 'N474'], node_size=150, node_color='gold', edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels={'N201': 'Source', 'N474': 'Target'}, font_size=12, font_weight='bold')

    # 图例与标题
    plt.title("Multi-Agent Autonomous Rerouting on TEFNET24 (INT-219)", fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_reroute_comparison()