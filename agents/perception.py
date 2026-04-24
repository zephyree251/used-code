import networkx as nx
import numpy as np
from collections import defaultdict
from core.context import BaseAgent


class PerceptionAgent(BaseAgent):
    """
    Step 1: 态势感知 (高保真物理 + 带宽拥塞版)
    基于 TEFNET24 真实数据集计算 OSNR，同时引入流量矩阵监控全网带宽拥塞情况。
    """

    def process(self, context):
        if not context.active_services:
            print(f"   -> [感知层] 无活跃业务，跳过。")
            return

        print(f"\n   -> [感知层] 正在基于真实物理拓扑扫描 {len(context.active_services)} 条光路 (OSNR与带宽)...")
        context.risk_report = []

        # --- 物理常数 (基于 TEFNET24 论文 / 你的设定) ---
        P_LAUNCH_DBM = -4.0
        ATTENUATION = 0.2  # dB/km
        EDFA_NF = 4.5  # dB
        CONSTANT_C = 58.0  # 基准常数
        OSNR_LIMIT = 30  # OSNR门槛 (30dB)

        # --- 新增：网络层容量约束 ---
        LINK_CAPACITY = 8000.0  # 设定单链路物理容量为 8000G
        CONGESTION_LIMIT = 0.80  # 拥塞告警门限 (80%)

        # =======================================================
        # 预备阶段：确保所有业务都有寻路，并统计全网物理链路负载
        # =======================================================
        link_load = defaultdict(float)

        for srv in context.active_services:
            # 1. 如果业务还没有路径，就为它计算最短路，并存入字典
            if 'path' not in srv or not srv['path']:
                try:
                    srv['path'] = nx.shortest_path(context.graph, srv['source'], srv['target'], weight='length_km')
                except nx.NetworkXNoPath:
                    # 找不到路的先跳过，后面核心扫描会报 Warn
                    continue

            # 2. 统计带宽：把该业务的带宽累加到它路过的每一段光纤上
            path = srv['path']
            bw = srv.get('bandwidth', 0.0)  # 如果没有取到 bandwidth 则默认加 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_load[(u, v)] += bw

        # =======================================================
        # 核心扫描：双指标联合判定 (OSNR + 拥塞)
        # =======================================================
        for srv in context.active_services:
            if 'path' not in srv or not srv['path']:
                print(f"      [Warn] 业务 {srv['id']} 无物理通路！")
                continue

            path = srv['path']
            print(f"起点：{srv['source']} 终点：{srv['target']}通路：{path}")

            total_len_km = 0
            total_spans_count = 0
            max_utilization = 0.0
            bottleneck_link = None

            # 遍历路径上的每一段链路
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = context.graph[u][v]

                # 1. 物理参数累加 (取 loader 进来的真值)
                seg_len = float(edge_data.get('length_km', 50))  # 缺省50
                seg_spans = int(edge_data.get('num_spans', 1))  # 缺省1
                total_len_km += seg_len
                total_spans_count += seg_spans

                # 2. 检查带宽利用率 (查账本)
                used_bw = link_load.get((u, v), 0)
                util = used_bw / LINK_CAPACITY
                if util > max_utilization:
                    max_utilization = util
                    bottleneck_link = f"{u}->{v}"

            # --- 计算 OSNR (完全保留你原来的物理公式) ---
            total_loss_db = ATTENUATION * total_len_km
            avg_span_loss = total_loss_db / max(1, total_spans_count)
            linear_osnr_db = CONSTANT_C + P_LAUNCH_DBM - avg_span_loss - EDFA_NF - 10 * np.log10(
                max(1, total_spans_count))

            sigma = 0.3 + (total_len_km / 3000.0)
            measured_osnr = linear_osnr_db + np.random.normal(0, sigma)

            # 将关键观测量写回业务对象，供决策层与评估层使用
            srv['osnr'] = round(measured_osnr, 2)
            srv['path_length'] = round(total_len_km, 2)
            srv['max_utilization'] = round(max_utilization, 4)
            srv['bottleneck_link'] = bottleneck_link

            # --- 判决与报告 ---
            is_risk = False
            risk_type = ""
            current_val = 0.0
            desc = ""

            # 判决 1：拥塞判定优先 (路不通是最高优故障)
            if max_utilization > CONGESTION_LIMIT:
                is_risk = True
                risk_type = "High_Congestion"
                current_val = round(max_utilization * 100, 2)  # 存入百分比数值
                desc = f"拥塞告警! 瓶颈链路 {bottleneck_link} 负载率 {current_val}% > 限值 {CONGESTION_LIMIT * 100}%"
                print(f"      ⚠️ 发现异常: Service {srv['id']} ({desc})")

            # 判决 2：OSNR 劣化判定次之
            elif (measured_osnr - 3 * sigma) < OSNR_LIMIT:
                is_risk = True
                risk_type = "Low_QoT"
                current_val = round(measured_osnr, 2)
                desc = f"OSNR {measured_osnr:.2f}dB < Limit {OSNR_LIMIT}dB (Len:{int(total_len_km)}km, Spans:{total_spans_count})"
                print(f"      ⚠️ 发现异常: Service {srv['id']} (OSNR={measured_osnr:.2f}dB)")

            # 如果命中任何一种风险，装入风险报告
            if is_risk:
                context.risk_report.append({
                    "id": f"RISK-{srv['id']}",
                    "service_id": srv['id'],
                    "type": risk_type,  # 此时可能有 Low_QoT 或 High_Congestion 两种类型
                    "path": path,  # 【关键】依然保留了路径列表传给 Step 2
                    "current_val": current_val,
                    "description": desc,
                })

        print(f"   -> 扫描结束，共发现 {len(context.risk_report)} 个异常业务。")