import json
import os
from core.context import BaseAgent


class EvidenceAgent(BaseAgent):
    """
    Step 3: 证据检索智能体 (RAG Retriever)
    职责：
    1. 读取 Step 2 生成的意图 (Intents)。
    2. 加载本地专家知识库 (SOP / 论文规则)。
    3. 根据意图中的关键词 (如 'Core_VIP', 'OSNR') 检索相关规则。
    4. 生成 'Evidence Chain' (证据链) 注入 Context，约束 Step 4 的决策。
    """

    def __init__(self, name):
        super().__init__(name)
        # 假设知识库放在项目根目录的 data 文件夹下
        # 这样写是为了兼容不同的运行路径
        self.kb_path = os.path.join("data", "knowledge_base.json")
        self.knowledge_base = []
        self._load_kb()

    def _load_kb(self):
        """加载 JSON 格式的专家知识库"""
        try:
            # 尝试加载
            if os.path.exists(self.kb_path):
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                print(f"   -> [知识库] 成功加载 {len(self.knowledge_base)} 条专家规则 (Source: {self.kb_path})。")
            else:
                # 假如找不到 data/knowledge_base.json，尝试找当前目录的 (兼容性处理)
                fallback_path = "knowledge_base.json"
                if os.path.exists(fallback_path):
                    with open(fallback_path, 'r', encoding='utf-8') as f:
                        self.knowledge_base = json.load(f)
                    print(f"   -> [知识库] 成功加载 {len(self.knowledge_base)} 条规则 (Source: root)。")
                else:
                    print(f"   -> [警告] 未找到知识库文件，Step 3 将无法提供证据支持！")
        except json.JSONDecodeError:
            print(f"   -> [错误] 知识库 JSON 格式有误，请检查逗号和引号。")
        except Exception as e:
            print(f"   -> [错误] 加载知识库失败: {e}")

    def process(self, context):
        # 如果没有意图，这一步就没有存在的意义
        if not context.intents:
            return

        print(f"\n   -> [检索层 (RAG)] 正在根据意图匹配 SOP 规则...")
        context.evidence = []

        for intent in context.intents:
            # 1. 提取检索关键词
            intent_id = intent.get('intent_id')
            user_level = intent.get('user_level', '')  # 例如 "Core_VIP"
            metric = intent.get('target_metric', '')  # 例如 "OSNR > 25dB"

            # 2. 执行检索 (关键词匹配)
            matched_rules = []

            for rule in self.knowledge_base:
                # 匹配逻辑 A: 针对用户等级 (User Level)
                # 例如: 规则里提到了 "Core_VIP"，而当前意图也是 "Core_VIP"
                if user_level and user_level in rule['keywords']:
                    matched_rules.append(rule['content'])

                # 匹配逻辑 B: 针对故障指标 (Metric)
                # 例如: 规则里提到了 "OSNR"，而当前意图是修 "OSNR"
                if "OSNR" in metric and "OSNR" in rule['keywords']:
                    matched_rules.append(rule['content'])

            # 3. 生成证据链
            if matched_rules:
                # 去重并拼接
                unique_rules = list(set(matched_rules))
                evidence_text = " || ".join(unique_rules)

                # 存入 Context
                context.evidence.append({
                    "intent_id": intent_id,
                    "related_rules": evidence_text
                })

                print(f"      └─ 意图 {intent_id} ({user_level}) 命中 {len(unique_rules)} 条规则:")
                print(f"         证据: {unique_rules[0][:50]}...")  # 只打印前50个字避免刷屏
            else:
                print(f"      └─ 意图 {intent_id} 未检索到特定规则，将使用默认策略。")