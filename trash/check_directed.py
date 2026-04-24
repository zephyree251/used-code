import networkx as nx
from utils.tefnet_loader import TefnetLoader


def check_directionality():
    loader = TefnetLoader('../data/tefnet_nodes.csv', 'data/tefnet_links.csv', 'data/tefnet_traffic.csv')
    G = loader.load_topology()

    print("\n🔍 [图类型检查]")
    if G.is_directed():
        print("✅ 当前图是: Directed (有向图)")
    else:
        print("❌ 当前图是: Undirected (无向图) -> 请立刻修改代码！")
        return

    # 随机抽查一条边，看看反向边是否存在
    edges = list(G.edges(data=True))
    u, v, data_forward = edges[0]  # 取第一条边 u->v

    print(f"\n🔍 [双向性检查] 检查链路 {u} <--> {v}")
    print(f"   ➡️ 正向 {u}->{v}: 长度={data_forward.get('length_km')} km")

    if G.has_edge(v, u):
        data_backward = G.get_edge_data(v, u)
        print(f"   ⬅️ 反向 {v}->{u}: 长度={data_backward.get('length_km')} km")
        print("✅ 确认：双向链路独立存在。")
    else:
        print(f"⚠️ 警告：{v}->{u} 不存在！这条路是单行道。")


if __name__ == "__main__":
    check_directionality()