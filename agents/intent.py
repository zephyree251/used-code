import json
import os
from http import HTTPStatus
from core.context import BaseAgent

try:
    import dashscope
    _HAS_DASHSCOPE = True
except ModuleNotFoundError:
    _HAS_DASHSCOPE = False
    dashscope = None


class IntentAgent(BaseAgent):
    """
    Step 2: 意图解析智能体
    职责：
    1. 接收 Step 1 的风险报告。
    2. 如果配置了 API Key，调用 Qwen 大模型进行语义理解和业务定级。
    3. 如果没配置 Key，自动切换到“规则引擎”兜底，保证程序不挂。
    """

    def __init__(self, name):
        super().__init__(name)
        # =========== 【配置区】 ===========
        # API Key 优先从环境变量 DASHSCOPE_API_KEY 读取，避免硬编码泄露
        self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self.model_name = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
        # ================================

        if _HAS_DASHSCOPE and self.api_key and "sk-" in self.api_key:
            dashscope.api_key = self.api_key
            self.use_llm = True
        else:
            self.use_llm = False
            if not _HAS_DASHSCOPE:
                print("   -> [提示] 未安装 dashscope，将使用规则引擎兜底。")
            else:
                print("   -> [警告] IntentAgent 未检测到有效 API Key，将使用规则引擎兜底。")

    def _construct_prompt(self, risk_report):
        """构建提示词 (Prompt Engineering)"""
        # 将 Python 对象转为 JSON 字符串，让大模型读得懂
        risk_desc = json.dumps(risk_report, indent=2, ensure_ascii=False)

        prompt = f"""
        你是一个光网络自治系统的智能运维专家。
        【任务】分析“物理层风险报告”，将其翻译成结构化的“运维意图”JSON。

        【风险报告输入说明】
        报告中包含 "type" 字段：
        - "High_Congestion": 链路带宽超载。
        - "Low_QoT": 物理层信号质量(OSNR)不达标。

        【网络知识库 (TEFNET24) - 拓扑语义规则】
        1. 核心层节点 (Core_VIP): 
           - 判别规则: 路径(path)中包含 'N0' 开头的节点 (如 N011, N021...) 或 ID < 20 的节点。
           - 运维策略: 优先级设为 90 (最高)，定级为 "Core_VIP"。隐含约束: ["Path_Length < 500km"]。
        2. 汇聚/接入层节点 (Access): 
           - 判别规则: 路径中全为 'N3xx', 'N4xx' 等大号节点，无核心节点。
           - 运维策略: 优先级设为 50 (一般)，定级 for "Access_Aggregation"。约束为空 []。

        【输入：风险报告】
        {risk_desc}

        【输出要求】
        请只输出一个纯 JSON 列表，不要包含 Markdown 格式（如 ```json ... ```），不要包含任何解释性文字。
        JSON 格式模板：
        [
            {{
                "intent_id": "INT-248",
                "target_service": 248,
                "user_level": "Core_VIP", 
                "issue_type": "Congestion",  // 必须根据风险报告 type 填入 "Congestion" 或 "Low_OSNR"
                "optimization_goal": "Optimize_QoT",
                "target_metric": "OSNR > 25dB",
                "constraints": ["Path_Length < 500km"],
                "priority": 90
            }}
        ]
        """
        return prompt

    def _call_qwen(self, prompt):
        """调用阿里云 DashScope API"""
        try:
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=prompt,
                result_format='message',
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                print(f"   -> [API Error] Code: {response.code}, Msg: {response.message}")
                return None
        except Exception as e:
            print(f"   -> [Network Error] {e}")
            return None

    def process(self, context):
        if not context.risk_report:
            # 如果 Step 1 没发现问题，这一步就没事干
            return

        print(f"\n   -> [意图层 (Qwen)] 收到 {len(context.risk_report)} 条风险报告，正在思考...")

        # --- 分支 A: 使用大模型 (推荐) ---
        if self.use_llm:
            prompt = self._construct_prompt(context.risk_report)
            llm_output = self._call_qwen(prompt)

            if llm_output:
                try:
                    # 简单的数据清洗：去掉可能存在的 markdown 符号
                    cleaned_json = llm_output.replace("```json", "").replace("```", "").strip()
                    # 截取第一个 [ 和最后一个 ] 之间的内容
                    if "[" in cleaned_json:
                        start = cleaned_json.find("[")
                        end = cleaned_json.rfind("]") + 1
                        cleaned_json = cleaned_json[start:end]

                    context.intents = json.loads(cleaned_json)
                    print(f"   -> [Qwen] 解析成功！生成了 {len(context.intents)} 条标准意图。")

                    # 打印第一条看看效果
                    if context.intents:
                        print(
                            f"      示例: ID={context.intents[0]['intent_id']}, Level={context.intents[0]['user_level']}")
                    return
                except json.JSONDecodeError:
                    print(f"   -> [解析失败] 模型返回的不是合法 JSON，转为规则兜底。")
                    # print(llm_output) # 调试时可以打开这行看模型到底说了啥

        # --- 分支 B: 规则引擎兜底 (当没钱调API或断网时) ---
        print("   -> [兜底] 启动本地规则引擎...")
        context.intents = self._rule_based_fallback(context)

    def _rule_based_fallback(self, context):
        """
        一个简单的“模拟大模型”，根据死规则生成意图。
        用于断网测试或 API 欠费时的备份方案。
        """
        intents = []
        for risk in context.risk_report:
            # 这里的逻辑和 Prompt 里的逻辑保持一致
            path_str = str(risk.get('path', []))
            # 简单判断：路径里有 N0 或者 N1 开头的算核心
            is_core = "N0" in path_str or "N1" in path_str
            # 与 Perception 的 type 对齐：High_Congestion -> Congestion，Low_QoT -> Low_OSNR
            risk_type = risk.get("type", "Low_QoT")
            issue_type = "Congestion" if risk_type == "High_Congestion" else "Low_OSNR"
            # intent_id 避免双前缀：service_id 可能已是 "INT-118" 形式
            sid = risk["service_id"]
            intent_id = sid if str(sid).startswith("INT-") else f"INT-{sid}"

            intents.append({
                "intent_id": intent_id,
                "target_service": sid,
                "user_level": "Core_VIP" if is_core else "Access_Aggregation",
                "issue_type": issue_type,
                "optimization_goal": "Optimize_QoT",
                "target_metric": "OSNR > 25dB",
                "constraints": ["Path_Length < 500km"] if is_core else [],
                "priority": 90 if is_core else 50
            })
        return intents