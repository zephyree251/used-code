import networkx as nx
import numpy as np
from collections import defaultdict
from core.context import BaseAgent


class ActionAgent(BaseAgent):
    """
    Step 6: 执行层 (Action Agent)
    接收决策结果 (context.decisions)，执行对应的物理或网络层动作。
    """

    def __init__(self, name="Action"):
        super().__init__(name)

    @staticmethod
    def _parse_action(action_type: str):
        """
        将参数化动作统一解析为基础动作类型与参数。
        支持旧动作名，保证向后兼容。
        """
        if action_type.startswith("Action_Power_Boost_"):
            tail = action_type.replace("Action_Power_Boost_", "").replace("dB", "").replace("p", ".")
            try:
                return "Action_Power_Boost", {"boost_db": float(tail)}
            except ValueError:
                return "Action_Power_Boost", {"boost_db": 2.0}

        if action_type.startswith("Action_Reroute_K"):
            tail = action_type.replace("Action_Reroute_K", "")
            try:
                return "Action_Reroute", {"k_paths": max(1, int(tail))}
            except ValueError:
                return "Action_Reroute", {"k_paths": 3}

        if action_type == "Action_Power_Boost":
            return "Action_Power_Boost", {"boost_db": 2.0}
        if action_type == "Action_Reroute":
            return "Action_Reroute", {"k_paths": 5}
        if action_type in ("Action_Maintain", "Action_Do_Nothing"):
            return "Action_Maintain", {}

        return action_type, {}

    def process(self, context):
        # 1. 改为检查 context.decisions，因为动作是跟着大脑的决策走的
        if not getattr(context, 'decisions', []):
            print(f"      [执行层] 暂无决策指令，跳过执行。")
            return

        print(f"      🚑 接收到 {len(context.decisions)} 条决策指令，开始分类执行...")

        # --- 重路由需要的物理常数 ---
        LINK_CAPACITY = 8000.0
        ATTENUATION = 0.20
        EDFA_NF = 4.5
        P_LAUNCH_DBM = -4.0
        CONSTANT_C = 58.0

        if not hasattr(context, 'action_logs'):
            context.action_logs = []

        # 2. 遍历 RL 大脑下发的所有决策
        for decision in context.decisions:
            service_id = decision['service_id']
            action_type = decision['action']
            base_action, action_params = self._parse_action(action_type)
            print(f"      [执行层] 正在处理业务 {service_id} 的 {action_type} 指令...") # 建议增加这一行
            # 从上下文中抓取对应的真实业务对象
            def clean_id(raw_id):
                # 转成字符串 -> 去掉两端看不见的空格 -> 强行去掉可能干扰的前缀
                return str(raw_id).strip().replace('RISK-', '').replace('INT-', '')

            target_id_clean = clean_id(service_id)

            # 用清洗后的纯净 ID 进行匹配
            srv = next((s for s in context.active_services if clean_id(s['id']) == target_id_clean), None)

            if not srv:
                # 如果还是找不到，直接把底牌亮出来，看看它们到底长什么奇葩样！
                sample_ids = [s['id'] for s in context.active_services[:3]]  # 取前3个真实ID当样本
                print(f"      ⚠️ [致命对齐错误] 找不到业务！")
                print(f"         Qwen传来的原始ID: '{service_id}' (清洗后: '{target_id_clean}')")
                print(f"         内存库里的真实ID: {sample_ids} ...")
                continue
            # srv = next((s for s in context.active_services if s['id'] == service_id), None)
            # if not srv:
            #     continue

            old_path = srv.get('path', [])
            req_bw = srv.get('bandwidth', 0.0)
            # ==========================================
            # 分支 1：执行重路由 (拓扑剪枝避障)
            # ==========================================
            if base_action == 'Action_Reroute':
                print(f"      🔄 为 {service_id} 准备执行重路由手术...")

                # --- A. 释放当前业务占用 ---
                current_usage = defaultdict(float)
                for other_srv in context.active_services:
                    if other_srv['id'] != service_id and other_srv.get('path'):
                        p = other_srv['path']
                        bw = other_srv.get('bandwidth', 0.0)
                        for i in range(len(p) - 1):
                            current_usage[(p[i], p[i + 1])] += bw

                # --- B. 拓扑剪枝 (生成安全子图) ---
                def filter_edge(u, v):
                    used = current_usage.get((u, v), 0)
                    if LINK_CAPACITY - used < req_bw:
                        return False
                    return True

                safe_graph = nx.subgraph_view(context.graph, filter_edge=filter_edge)

                # --- C. 寻找新路径 (K-Shortest Paths) ---
                try:
                    import itertools
                    k_paths_gen = nx.shortest_simple_paths(safe_graph, source=srv['source'], target=srv['target'],
                                                           weight='length_km')
                    k_paths = list(itertools.islice(k_paths_gen, action_params.get("k_paths", 5)))

                    if not k_paths:
                        raise nx.NetworkXNoPath

                    new_path = next((p for p in k_paths if p != old_path), None)

                    if not new_path:
                        print(f"      ⚠️ 重路由无效: {service_id} 的原始路径已是当前负载下的唯一选择，无法绕行。")
                        continue

                except nx.NetworkXNoPath:
                    print(f"      ❌ 重路由失败: {service_id} 找不到满足 {req_bw}G 的备用路径 (硬阻塞)！")
                    continue
                except Exception as e:
                    print(f"      ❌ 重路由异常: 寻路过程发生未知错误 ({e})。")
                    continue

                # --- D. 物理指标重估 ---
                total_len_km = 0.0
                total_spans = 0
                for i in range(len(new_path) - 1):
                    u, v = new_path[i], new_path[i + 1]
                    edge_data = context.graph.get_edge_data(u, v, default={})
                    seg_len = float(edge_data.get('length_km', 50.0))
                    total_len_km += seg_len
                    total_spans += int(edge_data.get('num_spans', max(1, seg_len // 80)))

                loss = ATTENUATION * total_len_km
                avg_loss = loss / max(1, total_spans)
                linear_osnr = CONSTANT_C + P_LAUNCH_DBM - avg_loss - EDFA_NF - 10 * np.log10(max(1, total_spans))

                # --- E. 状态覆写 ---
                srv['path'] = new_path
                srv['osnr'] = round(linear_osnr, 2)
                srv['is_rerouted'] = True

                context.action_logs.append(
                    f"🔄 {service_id} 执行 Reroute(K={action_params.get('k_paths', 5)}) | 旧: {'->'.join(old_path)} | 新: {'->'.join(new_path)} | 新OSNR: {linear_osnr:.2f}dB")
                print(f"      🔄 {service_id} 执行 Reroute(K={action_params.get('k_paths', 5)}) | 旧: {'->'.join(old_path)} | 新: {'->'.join(new_path)} | 新OSNR: {linear_osnr:.2f}dB")
                print(f"      ✅ {service_id} 抢修成功！已切换至新路径。")
            # ==========================================
            # 分支 2：执行功率提升
            # ==========================================
            elif base_action == 'Action_Power_Boost':
                old_osnr = srv.get('osnr', 0)
                boost_db = action_params.get("boost_db", 2.0)
                # 模拟功率提升带来的 OSNR 收益，假设上限为 35dB
                new_osnr = min(old_osnr + boost_db, 35.0)

                srv['osnr'] = round(new_osnr, 2)
                context.action_logs.append(
                    f"🔋 {service_id} 执行 Power_Boost(+{boost_db:.1f}dB) | OSNR {old_osnr:.2f}dB -> {new_osnr:.2f}dB")
                print(f"      ✅ {service_id} 功率提升(+{boost_db:.1f}dB)成功！OSNR 增加至 {new_osnr:.2f}dB")

            # ==========================================
            # 分支 3：保持原样 (RL 输出 Action_Maintain，兼容旧名 Action_Do_Nothing)
            # ==========================================
            elif base_action == 'Action_Maintain':
                context.action_logs.append(f"⏸️ {service_id} 执行 Maintain | 保持原样")
                print(f"      ⏸️ {service_id} 保持原样。")