# 项目优化建议与已做修改

## 已完成的修改

1. **安全：API Key 不再硬编码**
   - `agents/intent.py` 改为从环境变量 `DASHSCOPE_API_KEY`、`DASHSCOPE_MODEL` 读取。
   - 使用前请设置：`export DASHSCOPE_API_KEY=sk-xxx`

2. **Intent 规则兜底与 RL 状态对齐**
   - 规则兜底现在会设置 `issue_type`（Congestion / Low_OSNR），拥塞场景下 RL 能正确得到 `High_Congestion` 状态并选重路由。

3. **动作名称统一**
   - `agents/action.py` 已同时支持 `Action_Maintain` 与 `Action_Do_Nothing`，与 `decision_rl` 输出一致。

4. **Intent 双前缀修复**
   - 规则兜底生成的 `intent_id` 不再出现 `INT-INT-xxx`，与 `service_id` 一致。

5. **依赖管理**
   - 新增 `requirements.txt`，便于 `pip install -r requirements.txt`。
   - `utils/tefnet_loader.py` 去掉重复的 `import`。

---

## 建议后续优化

### 1. 配置与常量集中

- **问题**：`P_LAUNCH_DBM`、`ATTENUATION`、`EDFA_NF`、`LINK_CAPACITY`、`OSNR_LIMIT` 等分散在 `perception.py`、`action.py` 中，修改时要改多处。
- **建议**：在 `core/config.py` 或 `config.py` 中集中定义物理/网络常数，各模块从这里引用。

### 2. 统一日志

- **问题**：各处使用 `print()`，难以按级别过滤、重定向到文件。
- **建议**：使用 `logging`，按模块命名 logger，通过 level 控制输出（如开发用 DEBUG，生产用 INFO）。

### 3. 数据路径可配置

- **问题**：`data/tefnet_*.csv`、`data/knowledge_base.json`、`data/q_table_memory.json` 等路径写死在代码里。
- **建议**：用环境变量或单一配置文件（如 `config.yaml`）指定数据目录与关键文件路径。

### 4. Context 与 Action 的重复逻辑

- **问题**：`core/context.py` 中有 `execute_power_boost`、`execute_reroute`，但当前执行流程只用 `agents/action.py` 的实现，两处公式与行为可能不一致。
- **建议**：二选一：要么删除 context 中未使用的这两个方法，要么让 Action 只做“编排”，实际调用 context 的这两个方法，避免重复与漂移。

### 5. context.py 缩进与结构

- **问题**：`execute_power_boost` / `execute_reroute` 的方法体缩进不一致（多了一层空格），易影响可读性和后续修改。
- **建议**：统一为 4 空格缩进，并确认这两个方法是否仍需要保留（见上一条）。

### 6. 单元测试与回归

- **问题**：`test/` 下仅有少量测试，核心流水线（Perception → Intent → Evidence → Decision → Action → Evaluation）缺少自动化测试。
- **建议**：为各 Agent 的 `process()` 写基于 mock context 的单元测试；用固定随机种子或固定数据跑一轮主流程，做简单回归。

### 7. 异常与边界

- **问题**：如 `tefnet_loader` 中 `except:` 裸捕获；部分地方 `next(..., None)` 后未统一记录或上报。
- **建议**：改为 `except Exception as e` 并 log；对“找不到业务/路径”等关键分支打日志或返回明确错误码，便于排查。

### 8. 类型注解与文档

- **问题**：函数参数、返回值多为隐式类型，新人理解成本高。
- **建议**：为核心模块（context、各 agent、loader）补充类型注解（typing），并为 `SystemContext`、各 Agent 的输入输出写简短 docstring。

### 9. 可选依赖 dashscope

- **问题**：未安装 `dashscope` 时已能跑规则兜底，但文档未说明。
- **建议**：在 README 或 OPTIMIZATION 中注明：需要大模型意图解析时 `pip install dashscope` 并配置 `DASHSCOPE_API_KEY`。

### 10. .gitignore 与敏感文件

- **建议**：若尚未有 `.gitignore`，建议忽略 ` .venv/`、`__pycache__/`、`*.pyc`、`.idea/`、`data/q_table_memory.json`（若含本地训练状态），以及 `.env` 等可能含密钥的文件。

---

按优先级可先做：**配置/常量集中**、**统一 logging**、**数据路径可配置**、**删除或统一 context 与 action 的执行逻辑**，再逐步补测试与类型注解。
