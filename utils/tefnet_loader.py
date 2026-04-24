import pandas as pd
import networkx as nx
import numpy as np


class TefnetLoader:
    def __init__(self, node_file, link_file, traffic_file):
        self.node_file = node_file
        self.link_file = link_file
        self.traffic_file = traffic_file
        # 【关键修改】必须是 DiGraph (有向图)，否则 A->B 和 B->A 会混在一起
        self.graph = nx.DiGraph()

    def load_topology(self):
        print(f"正在读取节点文件: {self.node_file}")
        df_nodes = pd.read_csv(self.node_file)

        # 1. 加载节点
        for _, row in df_nodes.iterrows():
            n_id = row.get('Site Ref.')
            if pd.notna(n_id):
                self.graph.add_node(n_id, **row.to_dict())

        print(f"正在读取链路文件: {self.link_file}")
        df_links = pd.read_csv(self.link_file)

        # 2. 加载链路
        count = 0
        for _, row in df_links.iterrows():
            # CSV 里明确区分了 Origin 和 Destination
            u = row.get('Origin')
            v = row.get('Destination')

            # 读取物理属性
            length = row.get('Link length (km)', 50.0)
            spans = row.get('# spans', 1)

            try:
                length = float(length)
            except:
                length = 50.0

            if u and v:
                # 【关键】因为是 DiGraph，这里添加的是 u->v 的单向边
                # 如果 CSV 里有 v->u 的行，循环读到那一行时会自动添加反向边
                self.graph.add_edge(u, v, length_km=length, spans=spans)
                self.graph.add_edge(v, u, length_km=length, spans=spans)
                count += 1

        print(f"✅ 拓扑构建完成 (有向图): {self.graph.number_of_nodes()} 节点, {count} 链路。")
        return self.graph

    def load_traffic_demands(self):
        # ... (这部分代码保持不变，照抄之前的即可) ...
        print(f"正在读取流量矩阵: {self.traffic_file}")
        df_traffic = pd.read_csv(self.traffic_file, index_col=0)
        demands = []
        demand_id = 1
        for src_raw in df_traffic.index:
            for dst_raw in df_traffic.columns:
                bandwidth = df_traffic.loc[src_raw, dst_raw]
                if pd.notna(bandwidth) and bandwidth > 0:
                    src = src_raw.split('_')[0]
                    dst = dst_raw.split('_')[0]
                    sla = src_raw.split('_')[1] if '_' in src_raw else 'HL3'
                    if src in self.graph and dst in self.graph:
                        demands.append({
                            "id": f"INT-{demand_id:03d}",
                            "source": src,
                            "target": dst,
                            "bandwidth": float(bandwidth),
                            "sla": sla,
                            "path": [],
                            "osnr": 0.0
                        })
                        demand_id += 1
        return demands

# --- 测试代码 ---
if __name__ == "__main__":
    # 请确保这三个文件在同级目录下，且名字改对了
    loader = TefnetLoader(
        'tefnet_nodes.csv',
        'tefnet_links.csv',
        'tefnet_traffic.csv'
    )

    topology = loader.load_topology()
    services = loader.load_traffic_demands()

    # 打印一条链路看看有没有长度
    first_link = list(topology.edges(data=True))[0]
    print(f"\n【验证】链路数据示例: {first_link}")