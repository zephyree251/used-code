import networkx as nx
from itertools import islice
from utils.tefnet_loader import TefnetLoader
import random


def calculate_path_metrics(graph, path):
    """辅助函数：计算某条路径的总长度和预估OSNR"""
    total_length = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = graph.get_edge_data(u, v)
        # 读取真实长度
        link_len = edge_data.get('length_km', 50.0)
        total_length += link_len

    # OSNR 估算公式
    osnr = 32.0 - (total_length / 100.0) * 2.2 - random.uniform(0, 0.5)
    return total_length, osnr


def demo_k_paths():
    print(f"\n{'=' * 60}")
    print(f" 🗺️  多路径发现演示 (K-Shortest Paths Demo)")
    print(f"{'=' * 60}")

    # 1. 加载环境
    print("⏳ 正在加载拓扑...")
    loader = TefnetLoader('data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
    G = loader.load_topology()  # 必须是有向图 DiGraph
    print(f"✅ 拓扑就绪: {G.number_of_nodes()} 节点, {G.number_of_edges()} 链路")

    # 2. 挑选一个"复杂"的节点对 (N022 -> N071)
    # 我们知道这一对之间肯定有多条路
    src = 'N022'
    dst = 'N071'

    print(f"\n🔍 正在寻找 {src} 到 {dst} 之间的所有可行道路...")
    print(f"   (使用算法: nx.shortest_simple_paths, Weight='length_km')")

    try:
        # === 核心算法：KSP (K-Shortest Paths) ===
        # 它会按照总长度从小到大，依次吐出路径
        # 我们只取前 5 条 (K=5)
        k = 5
        k_paths = list(islice(nx.shortest_simple_paths(G, src, dst, weight='length_km'), k))

        print(f"\n🎉 找到了 {len(k_paths)} 条备选路径！详情如下：\n")

        for i, path in enumerate(k_paths):
            length, osnr = calculate_path_metrics(G, path)

            # 格式化输出
            rank = i + 1
            path_str = " -> ".join(path)

            # 第一条通常是主用路由 (Primary)，后面是备用 (Backup)
            tag = "🌟 [主用 Primary]" if i == 0 else f"🛡️ [备用 Backup-{i}]"

            print(f"{tag}")
            print(f"   路由跳数: {len(path) - 1} 跳")
            print(f"   物理距离: {length:.2f} km")
            print(f"   预估OSNR: {osnr:.2f} dB")
            print(f"   路径走向: {path_str}")
            print("-" * 60)

    except nx.NetworkXNoPath:
        print(f"❌ {src} 和 {dst} 之间没有任何通路！")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    demo_k_paths()