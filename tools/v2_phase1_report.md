# FireGuardian V2 Phase 1 报告：DeepSeek Agent Core

> 日期：2026-08-12
> 分支：`v2-deepseek-agent`（基于 `main` @ `0eca3ad`，基线 tag：`v0.1.0-baseline`）
> 状态：**Phase 1 完成，等待人工验收；未提交 Git；未进入 Phase 2**

---

## 1. 修改了什么

### 新增文件（Phase 1 范围内，未修改任何 M1–M11 业务代码）

| 文件 | 内容 |
| --- | --- |
| `config/llm_agent.yaml` | V2 Agent 配置：模型、Base URL、超时、重试、最大工具轮数、上下文限制、日志路径 |
| `agent/schemas.py` | AgentResult / AgentAssessment / ToolCall / ToolResult 数据结构与 JSON Schema 校验 |
| `agent/event_context.py` | FireEvent + FireDecision → Agent 事件上下文（秒制时长、union area 面积特征、规则决策） |
| `agent/deepseek_client.py` | DeepSeekClient：OpenAI 兼容调用、JSON Output、Tool Calls、重试、时延/Token 统计、Key 脱敏 |
| `agent/tool_registry.py` | 5 个工具注册表 + 工具调用参数校验（event_id 白名单、数值范围、未知键拒绝） |
| `agent/tool_executor.py` | ToolExecutor + JsonlEventDataSource / InMemoryDataSource |
| `agent/prompts/` | system_prompt.md / event_analysis_prompt.md / tool_usage_prompt.md + 加载器 |
| `tests/test_schemas.py` | Schema 校验 Mock 测试（12 项） |
| `tests/test_event_context.py` | EventContext Mock 测试（5 项） |
| `tests/test_deepseek_client.py` | DeepSeekClient Mock 测试（9 项，全部 Mock，不消耗 API） |
| `tests/test_agent_tools.py` | ToolRegistry / ToolExecutor Mock 测试（18 项） |
| `tools/v2_phase1_api_smoke.py` | 真实 API 最小冒烟测试脚本 |
| `tools/v2_phase1_report.md` | 本报告 |
| `.env`（不入库） | DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL |
| `.gitignore` | 新增 `.env` / `.env.*` 忽略规则 |
| `requirements.txt` | 新增 `openai>=1.0.0`、`python-dotenv>=1.0.0` |

### 职责边界（按人工确认后的表述）

- **M6**：事件聚合，统一生成秒制时间（`duration_seconds`）与 union area 面积特征；
- **M7**：消费 M6 的特征进行规则风险决策（level/score/confidence/reasons）；
- **Phase 1 新增代码只做只读视图与工具层**，不修改 M6/M7/M10/M11/YOLO/现有阈值。

## 2. 为什么这样修改

- 遵循 V2 计划书 §3 原则（LLM 不替代 YOLO / M7，不降低 High）与 §38 Phase 1 范围；
- API Key 只从环境变量 / `.env` 读取，源码、yaml、日志均不包含；
- 工具参数来自模型输出，不可信，必须先经 ToolRegistry 校验，禁止任意文件访问；
- `get_similar_events` 属 Phase 3（Agent Memory），Phase 1 注册但执行返回明确不可用错误。

## 3. 测试了什么

### Mock 单元测试

运行：`python -m unittest discover -s tests -p "test*.py"`

| 测试 | 覆盖点 |
| --- | --- |
| `test_schemas` | 合法输出解析；非法 assessment / risk_level / uncertainty / confidence 拒绝；缺失字段默认值；to_dict 往返 |
| `test_event_context` | 事件上下文字段、秒制时长、fire/smoke 特征、规则决策、枚举转字符串、空 event_id 报错 |
| `test_deepseek_client` | JSON 解析成功/非法/空 content；429/5xx 重试；错误信息不泄露 API Key；缺 Key 报配置错误；工具调用解析 |
| `test_agent_tools` | 5 工具注册；event_id 白名单；参数类型/范围/必填/未知键校验；4 个可用工具执行；similar_events Phase 1 不可用；JSONL 数据源（duration/duration_seconds 兼容）；证据目录列举 |

### 真实 API 最小冒烟测试

运行：`python tools/v2_phase1_api_smoke.py`（使用 `.env` 中的 Key，不写日志、不执行工具）

结果：

```
model    : deepseek-v4-flash
base_url : https://api.deepseek.com
api_key  : sk-7***（掩码）

[1] JSON Output  : parsed=True latency=2613.8ms tokens={prompt:124, completion:104, total:228}
    content      : {'ok': True, 'value': 42}
[2] Tool Calling : tool_calls=['get_event_summary'] latency=1389.1ms tokens={prompt:830, completion:71, total:901}

SMOKE_RESULT: PASS
```

## 4. 测试结果

- **单元测试：56/56 通过**（44 项 Phase 1 新增 + 12 项 V1 回归 `test_m6_m7_fix`）；
- **真实 API 冒烟：PASS**（JSON Output 与 Tool Calling 均正常）；
- API Key 未出现在任何输出/日志/异常中（已由 `_sanitize` 与掩码验证覆盖）。

## 5. Agent Loop 冒烟测试

新增最小 Agent Loop（最多 2 轮 LLM 调用）：

```
DeepSeek(第1轮, 要求工具调用)
  → Tool Call (get_event_summary)
  → ToolExecutor 本地执行
  → Tool Result 按 tool message 格式回传
  → DeepSeek(第2轮, 输出最终 Agent JSON)
  → schemas.AgentResult Schema Validation
```

脚本：`tools/v2_phase1_agent_loop_smoke.py`（真实 API 手动执行）
Mock 测试：`tests/test_agent_loop.py`（8 项，全部 Mock，不消耗 API）

### 5.1 首轮失败（原始问题，2026-08-12）

第一次真实执行失败在第 2 轮：

| 环节 | 结果 |
| --- | --- |
| Tool Call 成功（第 1 轮返回 get_event_summary） | ✅ |
| Tool Executor 成功（本地执行 + 参数校验） | ✅ |
| Tool Result 回传成功（tool message 格式） | ✅ |
| 第二轮 LLM 成功 | ❌ 返回非法 JSON（未终止字符串，line 6 col 21） |
| 最终 Schema Validation | ⏭️ 未执行 |
| 总耗时 | 22257.4 ms |
| Token 消耗 | prompt=2406, completion=2527, total=4933 |

### 5.2 诊断与修复

第二次真实执行（仅将第二轮 max_tokens 降为 768，其余不变）暴露根因：

- `finish_reason=length`、`completion_tokens=768`（正好打满上限）、`raw_content_length=0`；
- 结论：**DeepSeek V4 默认开启 thinking 模式，推理 token 计入 completion_tokens**；
  推理阶段把 token 预算耗尽，content 为空或未完成，导致非法 JSON（与首轮同源）。

修复内容：

1. 第二轮 prompt 改为 `agent/prompts/final_output_prompt.md`：只输出 JSON、禁止
   Markdown/代码块/解释文字、禁止再次调用工具、限制文本字段长度、提供紧凑示例；
2. 第二轮调用不再携带 tools，并设置 `tool_choice="none"`；
3. 第二轮 `max_tokens`：**2048 → 768**（短结构化输出足够，避免超长输出）；
4. 两轮均通过 `extra_body={"thinking": {"type": "disabled"}}` 关闭 thinking 模式；
5. `DeepSeekClient.chat` 增加按调用覆盖 `max_tokens` 与 `extra_body` 的支持；
6. 新增失败诊断输出：`finish_reason` / `response_format` / `tool_choice` /
   `max_tokens` / `completion_tokens` / 原始 content 前 1000 字符（不含 API Key 与完整
   prompt），失败时保存 `logs/agent_loop_diagnostic.json`。

关于「为何 completion 达到 2527 tokens」：该值为两轮合计；第二轮在 768 上限下打满且
content 为空，说明主要消耗来自 thinking 推理 token，而非真实 JSON 内容过长。
修复后第二轮 `completion_tokens=152`、`finish_reason=stop`，确认非内容过长问题。

### 5.3 修复后真实执行结果（2026-08-13）

| 环节 | 结果 |
| --- | --- |
| Tool Call 成功（第 1 轮） | ✅ |
| Tool Executor 成功 | ✅ |
| Tool Result 回传成功（tool message 格式） | ✅ |
| 第二轮不再调用 Tool（tool_choice=none） | ✅ |
| 第二轮输出合法 JSON | ✅ |
| 最终 Schema Validation 通过 | ✅ |
| 总耗时 | 4063.6 ms |
| Token 消耗 | prompt=2507, completion=370, total=2877（第二轮 completion=152） |

第二轮参数：`response_format={"type":"json_object"}`、`tool_choice="none"`、
`max_tokens=768`、`thinking=disabled`；API Key 全程只显示掩码，无泄漏。

**结论：Agent Loop 冒烟测试全部通过（A~I 验收条件满足），Phase 1 可进入 Phase 2。**

## 6. 发现什么问题

1. `apply_patch` 创建隐藏文件 `.env` 时未实际落盘，已改用 PowerShell 直接写入并验证（文件存在、内容正确、`git check-ignore` 生效）。
2. OpenAI SDK 3.0 异常构造要求 `httpx2.Response` 附带 request，Mock 测试已适配；业务代码无需变更。
3. `events.jsonl` 历史记录存在 `duration` 与 `duration_seconds` 两种字段，JsonlEventDataSource 已做兼容解析并统一转秒制（供 Phase 3 使用）。
4. Agent Loop 真实冒烟失败根因：DeepSeek V4 默认 thinking 模式，推理 token 计入 completion_tokens，
   导致第二轮 content 为空/截断 → 非法 JSON；已通过关闭 thinking、严格输出 prompt、
   第二轮不带 tools + `tool_choice="none"`、`max_tokens=768` 修复并验证通过。

## 7. 是否可以进入下一阶段

Phase 1 代码与 Mock 测试全部达成（对照计划书 §35 第一阶段验收清单中属于 Phase 1 的条目）：

- [x] DeepSeek V4 Flash 成功接入（真实 API 冒烟通过）
- [x] API Key 不进入 Git（`.env` + `.gitignore` 验证通过）
- [x] Tool Calling 正常（真实 API + Mock）
- [x] Event Context 正常（Mock）
- [x] get_event_summary / get_rule_decision / get_detection_history / get_event_evidence 可用（Mock + JSONL 数据源）
- [x] JSON Schema 验证正常
- [x] 工具参数校验正常（event_id 白名单 / 类型 / 范围 / 未知键）
- [x] Mock 测试不消耗 API 额度
- [x] V1 测试不受影响（12/12 通过）
- [x] Agent Loop 冒烟测试通过（A~I 验收条件全部满足）

**结论：Phase 1 验收通过，可以进入 Phase 2（DeepSeekAgent + HybridCoordinator + Fallback），等待人工确认后启动。**

| 环节 | 结果 |
| --- | --- |
| Tool Call 成功（第 1 轮返回 get_event_summary） | ✅ |
| Tool Executor 成功（本地执行 + 参数校验） | ✅ |
| Tool Result 回传成功（tool message 格式） | ✅ |
| 第二轮 LLM 成功 | ❌ 返回非法 JSON |
| 最终 Schema Validation 成功 | ⏭️ 未执行（前置环节失败） |
| 总耗时 | 22257.4 ms |
| Token 消耗 | prompt=2406, completion=2527, total=4933 |

失败详情：第 2 轮模型输出非法 JSON（`Unterminated string starting at: line 6 column 21 (char 159)`）。
Agent Loop 按设计要求返回 `status=invalid_json` 的明确失败状态，未异常退出。

**结论：Agent Loop 冒烟测试未全部通过，按收尾要求停止，不给出“Phase 1 可进入 Phase 2”结论。**
失败环节为第 2 轮 LLM 的 JSON 输出（非链路接线问题）；
脚本已补充失败时原始内容前 300 字符预览，便于重试定位。

## 6. 发现什么问题

1. `apply_patch` 创建隐藏文件 `.env` 时未实际落盘，已改用 PowerShell 直接写入并验证（文件存在、内容正确、`git check-ignore` 生效）。
2. OpenAI SDK 3.0 异常构造要求 `httpx2.Response` 附带 request，Mock 测试已适配；业务代码无需变更。
3. `events.jsonl` 历史记录存在 `duration` 与 `duration_seconds` 两种字段，JsonlEventDataSource 已做兼容解析并统一转秒制（供 Phase 3 使用）。
4. 真实 Agent Loop 冒烟第 2 轮返回非法 JSON（未终止字符串），链路各环节正常，失败发生在模型输出质量/截断，等待重试确认。

## 7. 是否可以进入下一阶段

Phase 1 代码与 Mock 测试全部达成（对照计划书 §35 第一阶段验收清单中属于 Phase 1 的条目）：

- [x] DeepSeek V4 Flash 成功接入（真实 API 冒烟通过）
- [x] API Key 不进入 Git（`.env` + `.gitignore` 验证通过）
- [x] Tool Calling 正常（真实 API + Mock）
- [x] Event Context 正常（Mock）
- [x] get_event_summary / get_rule_decision / get_detection_history / get_event_evidence 可用（Mock + JSONL 数据源）
- [x] JSON Schema 验证正常
- [x] 工具参数校验正常（event_id 白名单 / 类型 / 范围 / 未知键）
- [x] Mock 测试不消耗 API 额度
- [x] V1 测试不受影响（12/12 通过）
- [ ] Agent Loop 冒烟测试（第 2 轮 LLM 输出非法 JSON，未通过）

**结论：Agent Loop 冒烟未通过（失败环节：第 2 轮 LLM 非法 JSON）。按人工确认的收尾要求，失败即停止，暂不进入 Phase 2；待重试 Agent Loop 冒烟并全部通过后，再确认进入 Phase 2。**
