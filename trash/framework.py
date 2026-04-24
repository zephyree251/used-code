
import json
import time
import random
import networkx as nx
import numpy as np
import dashscope # <--- 引入国内大模型 SDK
from http import HTTPStatus
import random
import numpy as np  # 记得在文件最开头加上这行

from tefnet_loader import TefnetLoader # <--- 关键！引入你之前写的加载器


# ==========================================
# 1. 定义数据包 (Context)
# 这是在 7 个 Agent 之间流转的“档案袋”
# ==========================================
class SystemContext:
    def __init__(self):
        self.graph = None  # 存放 NetworkX 拓扑图
        self.all_demands = []  # 存放所有 CSV 里的业务
        self.active_services = []  # 存放当前正在运行的业务 (随机抽取的)

        self.step_logs = []  # 记录每一步的日志
        self.raw_state = {}  # Step 1 读到的原始状态
        self.risk_report = []  # Step 1 产出的风险
        self.intents = []  # Step 2 产出的意图
        self.evidence = []  # Step 3 查到的证据
        self.candidate_actions = []  # Step 4 提出的方案
        self.final_plan = {}  # Step 6 确定的计划
        self.evaluation_result = {}  # Step 7 的评分


# ==========================================
# 2. 定义 Agent 基类 (标准接口)
# 所有的 Agent 都要继承这个模板
# ==========================================
class BaseAgent:
    def __init__(self, name):
        self.name = name

    def run(self, context: SystemContext):
        """
        每个 Agent 必须实现这个方法
        输入：系统上下文
        输出：修改后的上下文
        """
        print(f"[{self.name}] 正在启动...", end="")
        time.sleep(0.5)  # 假装在思考
        self.process(context)
        print(" ✅ 完成")

    def process(self, context):
        raise NotImplementedError


# ==========================================
# 3. 实现 7 个具体的 Agent (目前是空逻辑)
# ==========================================

# Step 1: 态势感知
# framework.py (修改 PerceptionAgent 类)

# 文件名: framework.py
# 替换 PerceptionAgent 类

class PerceptionAgent(BaseAgent):
    def process(self, context):
        if not context.active_services:
            print(f"   -> [感知层] 无活跃业务，跳过。")
            return

        print(f"\n   -> [TEFNET感知] 正在基于真实 #Spans 数据扫描 {len(context.active_services)} 条光路...")
        context.risk_report = []

        # --- 物理常数 (基于 TEFNET24 论文) ---
        P_LAUNCH_DBM = -4.0
        ATTENUATION = 0.2  # dB/km
        EDFA_NF = 4.5  # dB
        CONSTANT_C = 58.0
        OSNR_LIMIT = 30  # 设置门槛

        for srv in context.active_services:
            try:
                # 1. 寻找路径
                path = nx.shortest_path(context.graph, srv['source'], srv['target'], weight='length_km')
            except nx.NetworkXNoPath:
                continue

            # --- 【关键修改】累加真实数据 ---
            total_len_km = 0
            total_spans_count = 0
            detailed_path_info = []  # 用于日志展示

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = context.graph[u][v]

                # 直接取真值！
                seg_len = edge_data['length_km']
                seg_spans = edge_data['num_spans']

                total_len_km += seg_len
                total_spans_count += seg_spans

                detailed_path_info.append(f"{seg_len}km({seg_spans}段)")

            # --- 2. 计算 OSNR (不再使用估算公式) ---
            # 既然有了 num_spans 真值，公式中的 N 就非常准了
            # 损耗也用真值算: 总损耗 = 衰减系数 * 总长度
            total_loss_db = ATTENUATION * total_len_km

            # 每个放大器补偿一段损耗，引入一份 ASE 噪声
            # 简化公式: OSNR = P_in - Loss_span - NF - 10log(N) + 58
            # 这里我们用一种通用的线性累积估算:
            # 假设每个跨段长度大致均匀 (CSV里看 Span1,2...差异不大)，可以用平均跨段损耗
            avg_span_loss = total_loss_db / max(1, total_spans_count)

            linear_osnr_db = CONSTANT_C + P_LAUNCH_DBM - avg_span_loss - EDFA_NF - 10 * np.log10(total_spans_count)

            # --- 3. 结果输出 ---
            # 增加一点随机波动 σ，模拟老化/连接器损耗
            sigma = 0.3 + (total_len_km / 3000.0)
            measured_osnr = linear_osnr_db + np.random.normal(0, sigma)

            # 4. 判决
            if (measured_osnr - 3 * sigma) < OSNR_LIMIT:
                context.risk_report.append({
                    "id": f"RISK-{srv['id']}",
                    "service_id": srv['id'],
                    "path_str": f"{srv['source']}->{srv['target']}",
                    "type": "Low_QoT",
                    "current_val": round(measured_osnr, 2),
                    "detail": f"OSNR {measured_osnr:.2f}dB (Len:{int(total_len_km)}km, Spans:{total_spans_count})",
                    "path_debug": detailed_path_info  # 方便你 debug
                })

        print(f"   -> 扫描结束，共发现 {len(context.risk_report)} 个异常业务。")


# ==========================================
# Step 2: 意图解析 (LLM 驱动版)
# ==========================================


# ==========================================
# Step 2: 意图解析 (国内直连版 - 通义千问)
# ==========================================
import json
import dashscope  # <--- 引入国内大模型 SDK
from http import HTTPStatus


class IntentAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

        # =========== 【你的配置区】 ===========
        # 这里的 Key 是 sk- 开头的，去阿里云百炼后台复制
        self.api_key = "sk-491181945a824dcf9badc83c9c67a192"

        # 选用 "qwen-plus" (效果好) 或 "qwen-turbo" (速度快)
        self.model_name = "qwen-turbo"
        # ====================================

        # 配置 SDK
        if self.api_key:
            dashscope.api_key = self.api_key
            self.use_llm = True
        else:
            self.use_llm = False
            print(f"   -> [警告] 未填入 DashScope API Key")

    def _construct_prompt(self, risk_report, active_services):
        # 保持之前的 Prompt 逻辑不变，这个 Prompt 写得很好
        risk_desc = json.dumps(risk_report, indent=2, ensure_ascii=False)
        prompt = f"""
        你是一个光网络自治系统的智能运维专家。
        【任务】分析底层的“物理层风险报告”，将其翻译成结构化的“运维意图（Intent）”JSON。
        【网络知识库 (TEFNET24)】
        1. 核心层节点 (Core): 名字包含 N011, N021 等(HL1/HL2)。优先级最高 (90)。
        2. 汇聚/接入层节点: 名字包含 N3xx, N4xx 等。优先级一般 (50)。
        【输入：风险报告】
        {risk_desc}
        【输出要求】
        必须输出纯 JSON 格式。不要使用 Markdown 标记。针对每一个 Risk 生成一个 Intent。
        【目标 JSON 结构示例】
        [
            {{
                "intent_id": "INT-101",
                "target_service": 101,
                "user_level": "Core_VIP",
                "optimization_goal": "Optimize_QoT",
                "target_metric": "OSNR > 25dB",
                "constraints": ["Latency < 2ms"],
                "priority": 90
            }}
        ]
        """
        return prompt

    def _call_qwen(self, prompt):
        """
        调用通义千问 API
        """
        try:
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=prompt,
                result_format='message',  # 设置返回格式
            )

            if response.status_code == HTTPStatus.OK:
                # 获取内容
                return response.output.choices[0].message.content
            else:
                print(f"   -> [API 错误] Code:{response.code}, Msg:{response.message}")
                return None

        except Exception as e:
            print(f"   -> [调用报错] {e}")
            return None

    def process(self, context):
        if not context.risk_report:
            print(f"   -> [意图层] 无待处理风险。")
            return

        print(f"\n   -> [意图层 (Qwen)] 正在思考...")

        if not self.use_llm:
            context.intents = self._rule_based_fallback(context)
            return

        # 1. 构造 Prompt
        prompt = self._construct_prompt(context.risk_report, context.active_services)

        # 2. 发送请求
        llm_output = self._call_qwen(prompt)
        # 打印 Qwen 的完整原始回复 (用于截图展示)
        # =======================================================
        if llm_output:
            print("\n" + "=" * 20 + " [Qwen 原始回复 Start] " + "=" * 20)
            print(llm_output)  # <--- 这里会把大模型说的话全部打出来
            print("=" * 20 + " [Qwen 原始回复 End] " + "=" * 22 + "\n")
        # =======================================================
        # 3. 解析结果
        if llm_output:
            try:
                # 清洗 JSON (国产模型有时也会带 ```json)
                cleaned_json = llm_output.replace("```json", "").replace("```", "").strip()
                intents = json.loads(cleaned_json)

                context.intents = intents
                print(f"   -> [Qwen] 成功解析出 {len(intents)} 条意图！")
                if intents:
                    # 打印第一条看看
                    first = intents[0]
                    print(f"      └─ 首条: {first.get('user_level')} | 目标: {first.get('target_metric')}")
                return

            except (json.JSONDecodeError, AttributeError, IndexError) as e:
                print(f"   -> [解析失败] 返回内容非标准 JSON: {e}")
                # print(llm_output) # 调试时可以打开

        # 4. 兜底
        print("   -> [兜底] 启动规则引擎接管...")
        context.intents = self._rule_based_fallback(context)

    def _rule_based_fallback(self, context):
        # ... (这里还是保留原来的规则代码，作为双重保险) ...
        # ... (直接复制之前的逻辑即可) ...
        print("      (正在使用基于 TEFNET 层级的规则解析...)")
        intents = []
        for risk in context.risk_report:
            try:
                sid = int(risk['service_id'])
            except:
                sid = 100
            is_core = (sid % 2 == 0)
            if risk['type'] == 'Low_QoT':
                goal = "Optimize_Physical_Quality"
                metric = "OSNR > 25dB"
            else:
                goal = "Traffic_Offload"
                metric = "Utilization < 70%"
            intents.append({
                "intent_id": f"INT-{risk['service_id']}-Fallback",
                "target_service": risk['service_id'],
                "user_level": "Core_VIP" if is_core else "Metro_Std",
                "optimization_goal": goal,
                "target_metric": metric,
                "constraints": ["Latency < 2ms"] if is_core else ["Cost_Min"],
                "priority": 90 if is_core else 50
            })
        return intents


# ==========================================
# Step 3: 证据检索 (RAG 简易版 - 查阅 ONG/TEFNET 知识库)
# ==========================================
class EvidenceAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)
        self.kb_file = "knowledge_base.json"
        self.knowledge_base = []
        self._load_kb()

    def _load_kb(self):
        """加载知识库文件"""
        try:
            with open(self.kb_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"   -> [知识库] 成功加载 {len(self.knowledge_base)} 条专家规则 (源自 ONG/TEFNET)。")
        except FileNotFoundError:
            print(f"   -> [警告] 找不到 {self.kb_file}，将无法检索证据！")
        except json.JSONDecodeError:
            print(f"   -> [错误] 知识库 JSON 格式有误，请检查逗号和引号。")

    def process(self, context):
        if not context.intents:
            print("   -> [检索层] 无待处理意图，跳过。")
            return

        print(f"\n   -> [检索层] 正在为 {len(context.intents)} 条意图匹配解决方案...")
        context.evidence = []

        for intent in context.intents:
            # 1. 提取意图里的关键信息
            intent_id = intent.get('intent_id', 'Unknown')
            target_metric = intent.get('target_metric', '')  # 例如 "OSNR > 25dB"
            user_level = intent.get('user_level', '')  # 例如 "Core_VIP"

            # 2. 在知识库里进行匹配 (简单的关键词匹配)
            matched_rules = []

            for rule in self.knowledge_base:
                # 匹配逻辑 A: 针对物理指标 (OSNR)
                if "OSNR" in target_metric and "OSNR" in rule['keywords']:
                    matched_rules.append(rule['content'])

                # 匹配逻辑 B: 针对用户等级 (VIP)
                if user_level in rule['keywords']:
                    matched_rules.append(rule['content'])

            # 3. 生成证据链并存入 Context
            if matched_rules:
                # 去重并拼接
                unique_rules = list(set(matched_rules))
                evidence_text = " || ".join(unique_rules)

                context.evidence.append({
                    "intent_id": intent_id,
                    "related_rules": evidence_text
                })

                # 打印出来给你看，显得很智能
                print(f"      └─ 意图 {intent_id} ({user_level}) 匹配到 {len(unique_rules)} 条规则:")
                for i, r in enumerate(unique_rules):
                    # 只打印前 60 个字，防止刷屏
                    print(f"         {i + 1}. {r[:60]}...")
            else:
                print(f"      └─ 意图 {intent_id} 未检索到特定规则，使用默认策略。")
                context.evidence.append({
                    "intent_id": intent_id,
                    "related_rules": "默认策略: 尝试常规参数优化。"
                })

        print(f"   -> 检索完成，证据已注入 Context。")


# Step 4: 决策仲裁 (核心大脑)
class DecisionAgent(BaseAgent):
    def process(self, context):
        # 【模拟】生成决策
        context.candidate_actions = [
            {"action_id": "A1", "desc": "调大功率", "score": 0.9},
            {"action_id": "A2", "desc": "切换路由", "score": 0.7}
        ]
        # 选分最高的
        best_action = context.candidate_actions[0]
        print(f"\n   -> 决策完毕，推荐动作: {best_action['desc']} (置信度 {best_action['score']})")


# Step 5: 专家会诊
class ExpertAgent(BaseAgent):
    def process(self, context):
        # 【模拟】专家检查
        print(f"\n   -> 专家审核: 动作安全，无冲突。")


# Step 6: 执行编排
class ExecutionAgent(BaseAgent):
    def process(self, context):
        # 【模拟】下发指令
        context.final_plan = {"status": "Executed", "timestamp": time.time()}
        print(f"\n   -> 指令已下发到控制器。")


# Step 7: 评估回写
class EvaluationAgent(BaseAgent):
    def process(self, context):
        # 【模拟】计算奖励
        reward = random.uniform(0.8, 1.0)
        context.evaluation_result = {"reward": reward}
        print(f"\n   -> 效果评估完成，系统获得奖励: {reward:.2f}")


# ==========================================
# 4. 系统主控 (Orchestrator)
# ==========================================
class OODA_System:
    def __init__(self):
        self.context = SystemContext()

        # --- 接入 TEFNET24 数据集 (核心修改) ---
        print(">>> [System] 正在加载 TEFNET24 数据底座...")
        loader = TefnetLoader('tefnet_nodes.csv', 'tefnet_links.csv', 'tefnet_traffic.csv')

        # 1. 把地图加载进 Context
        self.context.graph = loader.load_topology()

        # 2. 把业务加载进 Context
        self.context.all_demands = loader.load_traffic_demands()

        # 3. 随机抽取 15 条业务作为“当前考题”
        self.context.active_services = random.sample(self.context.all_demands, 15)
        print(
            f">>> [System] 数据接入完成。节点数: {len(self.context.graph.nodes)}, 当前测试业务数: {len(self.context.active_services)}")

        # 注册所有 Agent (保持不变)
        self.agents = [
            PerceptionAgent("Step 1 态势感知"),
            IntentAgent("Step 2 意图解析"),
            EvidenceAgent("Step 3 证据检索"),
            DecisionAgent("Step 4 决策仲裁"),
            ExpertAgent("Step 5 专家会诊"),
            ExecutionAgent("Step 6 执行编排"),
            EvaluationAgent("Step 7 评估回写")
        ]

    def run_cycle(self):
        print("=" * 50)
        print("🚀 多智能体光网络自治系统启动 (Framework Mode)")
        print("=" * 50)

        for agent in self.agents:
            agent.run(self.context)

        print("=" * 50)
        print("✅ 闭环结束，等待下一轮。")


# ==========================================
# 5. 启动入口
# ==========================================
if __name__ == "__main__":
    system = OODA_System()
    system.run_cycle()