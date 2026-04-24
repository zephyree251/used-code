# core/context.py
import networkx as nx  # <--- 别忘了在文件最上面加这个
import random
class SystemContext:
    """
    系统上下文 (System Context) - 相当于系统的“共享内存”。
    所有的 Agent 都从这里读取数据，处理完后把结果写回这里。
    """

    def __init__(self):
        # ==========================================
        # 1. 基础底座数据 (Static Data)
        # ==========================================
        self.graph = None  # NetworkX 图对象 (存放拓扑结构)
        self.all_demands = []  # CSV 里加载的所有潜在业务列表

        # ==========================================
        # 2. 运行时状态 (Runtime State)
        # ==========================================
        self.active_services = []  # 当前这一轮正在运行/被测试的业务

        # ==========================================
        # 3. OODA 流水线中间产物 (Pipeline Artifacts)
        # ==========================================
        # Step 1 (Perception) 产出的风险报告
        self.risk_report = []

        # Step 2 (Intent) 产出的意图 JSON
        self.intents = []

        # Step 3 (Evidence) 产出的检索证据
        self.evidence = []

        # Step 4 (Decision) 产出的决策指令
        self.decisions = []

        # ==========================================
        # [修正版] 动作 1: 功率调节 (Power Boost)
        # ==========================================
    def execute_power_boost(self, service_id):
            # 【修正1】这里要用 s['id']
            service = next((s for s in self.active_services if s['id'] == service_id), None)
            if not service:
                return False, "Service Not Found"

            old_osnr = service.get('osnr', 0)

            # 模拟物理效果
            new_osnr = old_osnr + 2.5

            service['osnr'] = new_osnr
            service['is_optimized'] = True

            return True, f"OSNR boosted from {old_osnr:.2f} to {new_osnr:.2f} dB"

        # ==========================================
        # [修正版] 动作 2: 业务重路由 (Reroute)
        # ==========================================
    def execute_reroute(self, service_id):
        # 1. 查找业务
        service = next((s for s in self.active_services if s['id'] == service_id), None)
        if not service:
            return False, "Service Not Found"

        source = service['source']
        target = service['target']  # 确保这里用的是 'target'

        try:
            # --- 【修改 1】寻路时使用 'length_km' 权重 ---
            # 这样 Dijkstra 算法就会基于真实公里数找最短路
            new_path = nx.shortest_path(self.graph, source=source, target=target, weight='length_km')

            # --- 【修改 2】计算长度时读取 'length_km' 属性 ---
            total_length = 0
            for i in range(len(new_path) - 1):
                u, v = new_path[i], new_path[i + 1]
                edge_data = self.graph.get_edge_data(u, v)

                # 优先读 'length_km'，读不到才用 50
                link_len = edge_data.get('length_km', edge_data.get('length', 50.0))
                total_length += link_len

            # 3. 重新估算 OSNR (公式不变)
            # 假设基准 32dB，每 100km 衰减 2.5dB
            new_osnr = 32.0 - (total_length / 100.0) * 2.5 + random.uniform(-0.5, 0.5)

            # 4. 更新业务状态
            service['path'] = new_path
            service['path_length'] = total_length
            service['osnr'] = new_osnr
            service['is_rerouted'] = True

            new_path_str = "->".join(new_path)
            log_msg = (f"Reroute Success. Path: {new_path_str}\n"
                       f"   Total Distance: {total_length:.1f} km (Read from 'length_km')\n"
                       f"   New OSNR: {new_osnr:.2f} dB")

            return True, log_msg

        except nx.NetworkXNoPath:
            return False, f"Reroute Failed: No path between {source} and {target}"
        except Exception as e:
            return False, f"Reroute Error: {str(e)}"

class BaseAgent:
    """
    智能体基类 (Base Agent)
    这是一个模板，规定了所有的 Agent (感知、意图、决策等) 必须长什么样。
    """

    def __init__(self, name):
        self.name = name

    def process(self, context: SystemContext):
        """
        核心处理函数。
        每个子类 (Subclass) 都必须重写这个方法，实现具体的逻辑。
        """
        # 如果子类忘了写这个方法，运行时就会报错提醒你
        raise NotImplementedError(f"错误: Agent '{self.name}' 没有实现 process 方法！")