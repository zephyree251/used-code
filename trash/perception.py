import networkx as nx
import numpy as np
import json
from tefnet_loader import TefnetLoader  # 引用你刚才跑通的那个文件


class PerceptionAgent:
    def __init__(self, topology_graph, active_services):
        self.graph = topology_graph
        self.services = active_services
        # 设定故障红线：OSNR 低于 14dB 就认为是高风险
        self.osnr_threshold = 28.0

    def _calculate_theoretical_osnr(self, length_km):
        """
        物理层仿真核心公式 (替代 Fraunhofer)
        这里模拟的是：距离越长，损耗越大，OSNR 越低
        经验公式：基准 30dB - (0.02 * 距离)
        """
        return 30.0 - (0.02 * length_km)

    def sense(self):
        """
        核心功能：全网扫描，输出带“不确定性”的体检报告
        """
        print(f"\n[感知模块] 正在扫描 {len(self.services)} 条活跃业务的物理健康度...")

        risks = []
        healthy_count = 0

        for srv in self.services:
            # 1. 还原业务路径的物理长度 (Ground Truth)
            # 这里简单起见，我们假设业务跑在最短路径上
            try:
                # 使用 NetworkX 找一条路 (模拟 Step 6 的执行结果)
                path = nx.shortest_path(self.graph, srv['source'], srv['target'], weight='length_km')

                # 计算这条路的总长度
                total_len_km = 0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    total_len_km += self.graph[u][v]['length_km']
            except nx.NetworkXNoPath:
                print(f"警告: 节点 {srv['source']} -> {srv['target']} 不连通")
                continue

            # 2. 计算物理指标 (Simulation)
            true_osnr = self._calculate_theoretical_osnr(total_len_km)

            # 3. 引入不确定性 σ (师姐要求的重点！)
            # 逻辑：路越长，经过的放大器越多，不确定性越大
            sigma = 0.5 + (total_len_km / 1000.0)

            # 4. 模拟测量值 (加噪声)
            measured_osnr = true_osnr + np.random.normal(0, sigma)

            # 5. 风险判决
            # 如果 (测量值 - 3倍sigma) 跌破阈值，说明极其危险
            safety_margin = measured_osnr - 3 * sigma

            if safety_margin < self.osnr_threshold:
                # 生成风险报告
                risks.append({
                    "service_id": srv['id'],
                    "path_info": f"{srv['source']}->{srv['target']} ({int(total_len_km)}km)",
                    "risk_type": "Low_QoT",
                    "perception_data": {
                        "measured_osnr": round(measured_osnr, 2),
                        "uncertainty_sigma": round(sigma, 2),  # 这个必须有，为了解释性
                        "confidence_interval": f"[{round(measured_osnr - 3 * sigma, 1)}, {round(measured_osnr + 3 * sigma, 1)}]"
                    },
                    "severity": "Critical"
                })
            else:
                healthy_count += 1

        # 生成总结
        summary = {
            "total_scanned": len(self.services),
            "healthy": healthy_count,
            "risky": len(risks),
            "network_status": "Degraded" if len(risks) > 0 else "Healthy"
        }

        return summary, risks


# --- 运行主程序 ---
if __name__ == "__main__":
    # 1. 加载 TEFNET24 环境
    loader = TefnetLoader('tefnet_nodes.csv', 'tefnet_links.csv', 'tefnet_traffic.csv')
    G = loader.load_topology()
    demands = loader.load_traffic_demands()

    # 2. 挑选前 10 条业务进行测试
    # (在真实系统中，这些是正在运行的业务)
    active_services = demands[:10]

    # 3. 启动感知 Agent
    agent = PerceptionAgent(G, active_services)
    summary, risk_report = agent.sense()

    # 4. 打印给师姐看的报告
    print("\n" + "=" * 40)
    print("Step 1: 态势感知输出报告")
    print("=" * 40)
    print("【状态摘要】")
    print(json.dumps(summary, indent=2))

    print("\n【风险点详情】 (输入给 Step 2 的数据)")
    print(json.dumps(risk_report, indent=2))