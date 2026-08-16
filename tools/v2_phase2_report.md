# FireGuardian V2 Phase 2 报告：DeepSeekAgent + HybridCoordinator + Fallback

> 日期：2026-08-13
> 分支：`v2-deepseek-agent`（基线 tag：`v0.1.0-baseline`，HEAD 未变：`0eca3ad`）
> 状态：**Phase 2 完成，等待人工审核；未进入 Phase 3；未提交 Git**

---

## 1. 新增文件

| 文件 | 内容 |
| --- | --- |
| `agent/deepseek_agent.py` | `FireGuardianLLMAgent`（主链路 Agent）+ `AgentStatus` / `AgentOutcome` + `should_call_agent` 调用时机控制 |
| `agent/hybrid_decision.py` | `HybridDecisionCoordinator`（严格融合规则 1~7）+ `run_hybrid_pipeline` |
| `agent/agent_logger.py` | `AgentLogger`：`logs/agent_decisions.jsonl` / `logs/agent_tool_calls.jsonl` |
| `agent/deepseek_client.py` | 扩展：捕获 `reasoning_content`、按调用覆盖 `max_tokens`、透传 `extra_body`（thinking 开关） |
| `agent/prompts/*` | 系统/最终输出 prompt 补充枚举取值与「agent_assessment ≠ 事件状态」约束 |
| `tools/v2_phase2_agent_smoke.py` | Agent + Hybrid 真实 API 冒烟测试 |
| `tools/v2_phase2_thinking_eval.py` | Thinking Mode A/B 评估（支线实验） |
| `tests/test_deepseek_agent.py` | Agent 单元测试（12 项，Mock） |
| `tests/test_hybrid_decision.py` | Hybrid 单元测试（14 项，Mock） |
| `tests/test_agent_fallback.py` | Fallback 安全测试（5 项，Mock） |
| `tools/v2_phase2_report.md` | 本报告 |

未修改：YOLO / 数据集 / M6 / M7 / M8 / M9 / M10 / M11 / GUI。

## 2. Agent 架构

```
M7 Rule Decision (FireDecision)
        ↓
FireGuardianLLMAgent.analyze(event, decision, trigger)
        ↓
EventContext（M6 秒制/面积特征 + M7 规则结果）
        ↓
Tool Calling 循环（最多 3 轮：get_event_summary /
        get_rule_decision / get_detection_history / get_event_evidence）
        ↓
Agent Analysis（严格 Schema JSON）
        ↓
HybridDecisionCoordinator
        ↓
Final Decision（rule/agent/final/agreement/override/requires_review/reason）
```

- 输入：`FireEvent`（M6）+ `FireDecision`（M7）；输出：`AgentResult`（通过现有 Schema 校验）。
- 状态：`ok` / `unavailable`（API 失败）/ `invalid_output`（非法 JSON / Schema 失败）。
- 生产默认：`deepseek-v4-flash`，`thinking = disabled`（每轮调用均带 `extra_body={"thinking":{"type":"disabled"}}`）。
- 工具：Phase 2 仅开放 4 个工具；`get_similar_events` 保持 Phase 3 不可用（调用返回明确错误，不崩溃）。

## 3. Tool Calling 流程

1. 第 1~3 轮：`chat_with_tools`（携带 4 个工具 schema）；
2. 模型返回 `tool_calls` → 本地 `ToolExecutor` 执行（先经 ToolRegistry 参数校验）→ 按 `tool` message 格式回传；
3. 工具失败（含 Phase 3 工具被调用）不崩溃，结果照常回传，Agent 继续分析或最终输出；
4. 达到 3 轮后强制最终输出：不带 tools、`tool_choice="none"`、`response_format={"type":"json_object"}`、`max_tokens=768`；
5. thinking 模式下若有 `reasoning_content`，随 assistant 消息回传（A/B 实验已验证）。

## 4. Hybrid 规则（严格）

| Case | M7 | Agent | Final | requires_review | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | high | low | high | false | M7 High 不可被 Agent 降级 |
| 2 | high | high | high | false | 一致 |
| 3 | medium | high | high | **true** | Agent 升级 |
| 4 | medium | medium | medium | false | 一致 |
| 5 | low | high | high | **true** | Agent 升级 |
| 6 | any | unavailable | = M7 | false | 回退 |
| 7 | any | invalid_output | = M7 | false | 回退 |

扩展组合：low+low→low；low+medium→medium（override=true，不要求复核）；medium+low→medium（规则优先）。
每次融合记录：`rule_level / agent_level / final_level / agreement / override / requires_review / reason / agent_status`。

## 5. Fallback

- Agent API 失败 → `unavailable`；非法 JSON / Schema 失败 → `invalid_output`；两者 Final 均回退 M7；
- `FireGuardianLLMAgent.analyze` 绝不抛异常（任何意外异常也归为 `unavailable`）；
- 不阻塞 YOLO / M6 / M7 / 截图 / 日志 / 报告（Agent 只读，工具执行失败内部消化）。

## 6. 测试结果

运行：`python -m unittest discover -s tests -p "test*.py"`

- **总计：97/97 通过**（Phase 1 66 项 + Phase 2 新增 31 项）；
- V1 回归（M6/M7 等 12 项）全部通过；
- M7 High 安全测试：high+low→high、high+invalid→high、high+unavailable→high、medium+high→high、low+high→high+review、Agent 失败后 M7 继续运行——全部通过；
- Agent：工具循环（1 轮 / 3 轮 / 强制最终）、无工具直接输出、API 失败、非法 JSON、Schema 失败、工具失败不崩溃、thinking 回传——全部通过；
- 所有新测试均为 Mock，不消耗 API。

## 7. 真实 API 测试

`tools/v2_phase2_agent_smoke.py`（mock FireEvent + 真实 DeepSeek，thinking disabled）：

| 尝试 | 结果 | 说明 |
| --- | --- | --- |
| 第 1 次 | FAIL | 模型输出 `agent_assessment="CONFIRMED"`（与事件状态混淆），Schema 拒绝 → invalid_output → Final 回退 M7（Fallback 按设计生效） |
| 第 2 次 | FAIL | 同一问题；prompt 已补充枚举约束 |
| 第 3 次 | **PASS** | Agent=ok，3 次 LLM 调用、4 次工具调用，10556.2ms，total_tokens=6267；Schema 通过；Hybrid medium+medium→medium |

## 8. Thinking Mode A/B 实验结果

`tools/v2_phase2_thinking_eval.py`：6 个固定案例（real fire / smoke only / fireworks / normal / rule medium / rule high）× 2 模式，共 12 轮，**12/12 成功**，无 400 / 兼容问题。

| case | disabled level | enabled level | rule level |
| --- | --- | --- | --- |
| case_01_real_fire | high ✓ | medium ✗ | high |
| case_02_smoke_only | medium ✓ | medium ✓ | medium |
| case_03_fireworks | medium ✓ | medium ✓ | medium |
| case_04_normal_scene | low ✓ | medium ✗ | low |
| case_05_rule_medium | medium ✓ | medium ✓ | medium |
| case_06_rule_high | high ✓ | medium ✗ | high |

- 与 M7 对齐率：disabled 6/6，enabled 3/6（enabled 把 2 个 high 案例判成 medium）；
- `reasoning_content`：enabled 6/6 正确捕获并回传，工具调用契约满足；
- 结论（仅记录，不自动改变生产默认）：**thinking disabled 更快且等级更贴近 M7**，生产维持 disabled。

## 9. Token 与延迟

Agent + Hybrid 冒烟（PASS 轮）：10.6s，token=6267。

A/B 实验：

| 指标 | disabled（6 case 均值/合计） | enabled（6 case 均值/合计） |
| --- | --- | --- |
| 延迟 | 均值 8795.2ms（合计 52.8s） | 均值 15649.7ms（合计 93.9s），约 1.78x |
| Token | 合计 33879（均值 5646.5） | 合计 34911（均值 5818.5） |
| 成功率 | 6/6 | 6/6 |

## 10. 已知问题

1. 模型偶发把 `agent_assessment` 写成 `CONFIRMED`（与 status 混淆）；已通过 prompt 强化缓解，无硬性保证——Schema 校验 + Fallback 兜底，不回退则保持 FAIL 并记录。
2. thinking enabled 下 Agent 等级质量不稳定（2/6 的 high 案例判成 medium）；Hybrid 的 M7 保护保证 Final 不被降级，但 Agent 自身建议仍可能偏低。
3. Agent 输出质量存在波动（如空 evidence_summary / possible_cause 字段）；Phase 4/5 展示时需注意空字段处理。
4. 模型可能一次返回多个并行工具调用；日志按「实际执行」记录数量。

## 11. 是否可以进入 Phase 3

Phase 2 验收标准全部满足：

- [x] Agent Tool Loop 正常（≤3 轮 + 强制最终输出）
- [x] Hybrid 正常（Case 1~7 + 扩展组合全部通过）
- [x] M7 High 不可降级（单元测试 + 真实回退验证）
- [x] Fallback 正常（unavailable / invalid_output → Final=M7，异常不传播）
- [x] Schema 正常（AgentResult 严格校验，非法即 invalid_output）
- [x] V1 回归全部通过（97/97 含 V1 12 项）
- [x] Thinking A/B 实验完成（12/12，无兼容问题，结论已记录）

**结论：Phase 2 可验收。等待人工审核后进入 Phase 3（Agent Memory）。**
