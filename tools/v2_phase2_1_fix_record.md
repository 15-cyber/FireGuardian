# FireGuardian V2 Phase 2.1 修复记录

> 日期：2026-08-13
> 分支：`v2-deepseek-agent`（HEAD 未变：`0eca3ad`）
> 状态：**收尾完成，等待人工审核；未提交 Git**

---

## 1. 修改了什么

| 文件 | 修改 |
| --- | --- |
| `agent/prompts/system_prompt.md` | 强化 `agent_assessment` 枚举：四个合法值原样输出、禁止缩写/变体、与事件状态 status 无关 |
| `agent/prompts/final_output_prompt.md` | 同上强化 + 明确反例（CONFIRMED / CONFIRMED_RISK_RISK / confirmed_risk / Confirmed_Risk） |
| `agent/prompts/event_analysis_prompt.md` | 枚举取值说明补充到输出注意项 |
| `agent/display_text.py` | **新增**展示层空字段 fallback：`format_agent_result_for_display(AgentResult)` |
| `tests/test_display_text.py` | **新增**展示 fallback 单元测试（4 项） |

未修改：`HybridDecisionCoordinator`（Hybrid 规则零改动）、M6/M7/M8/M9/M10/M11、GUI、YOLO、数据集。

## 2. 为什么这样修改

- 真实 API 曾出现模型把 `agent_assessment` 输出为 `CONFIRMED`（与事件状态
  `status=confirmed` 混淆，或把 `CONFIRMED_RISK` 缩写），被 Schema 拒绝后触发
  Fallback。通过 Prompt 强化枚举集合与反例，从源头减少该混淆；
- 真实冒烟与 A/B 实验均观察到 AgentResult 存在空字段
  （possible_cause / evidence_summary / reasoning_summary / recommended_action /
  tools_used / agent_confidence=0.0），需要为未来 M10/GUI 展示准备明确 fallback 文案。

## 3. 展示空字段 fallback 策略（供 M10/GUI 接入）

接入方式：GUI/M10 展示时调用 `agent/display_text.py` 的
`format_agent_result_for_display(result)`，不要直接拼接 AgentResult 原始字段。

| 原始空字段 | 展示 fallback |
| --- | --- |
| possible_cause | 「暂无明确判断（证据不足或 Agent 未给出结论）」 |
| reasoning_summary | 「Agent 未给出推理摘要（可查看工具调用与证据记录）」 |
| recommended_action | 「未给出处置建议，请按当前风险等级执行标准流程」 |
| evidence_summary（空列表） | `["无引用证据（可查看截图/历史记录）"]` |
| tools_used（空列表） | `["未调用工具"]` |
| agent_confidence ≤ 0 | `agent_confidence_display = "未提供"`（不覆盖数值本身） |

原则：

- **只读转换**：不修改原始 AgentResult，原始数据仍完整可审计；
- 返回 dict 附带 `_display_fallbacks` 列表，标记哪些字段被替换，便于界面提示
  「Agent 信息不完整，仅供参考」；
- 非空字段保持原值；`agent_confidence_display` 为展示辅助字段，不进入 AgentResult Schema。

## 4. thinking 模式确认

- 生产默认保持 `thinking = disabled`（`extra_body={"thinking":{"type":"disabled"}}`）；
- 本次真实冒烟仍按 disabled 执行并通过；
- Thinking A/B 实验结论已记录在 Phase 2 报告，不自动改变默认。

## 5. Hybrid 规则确认

- `agent/hybrid_decision.py` 本次未做任何修改；
- Case 1~7 与 M7 High 不可降级逻辑保持不变（101/101 测试中相关用例全部通过）。

## 6. 测试结果

- **Mock 全量：101/101 通过**（Phase 2 基线 97 项 + 新增展示 fallback 4 项）；
- V1 回归（M6/M7 12 项）全部通过；
- 真实 API 冒烟（`tools/v2_phase2_agent_smoke.py`，thinking disabled）：**PASS**，
  Agent=ok，3 次 LLM 调用、4 次工具调用，9489.9ms，total_tokens=6274，
  `agent_assessment=SUSPICIOUS`（枚举强化生效，无 CONFIRMED 混淆），
  Hybrid medium+medium → medium。

## 7. 已知问题

1. Prompt 强化能显著降低枚举混淆，但模型输出仍无硬性保证；Schema 校验 + Fallback
   （invalid_output → Final=M7）继续兜底。
2. Agent 仍可能给出空字段结果（本次冒烟即出现），展示层 fallback 已覆盖，
   但 GUI/M10 实际接入时需使用 `format_agent_result_for_display`。

## 8. 结论

Phase 2.1 收尾完成，Phase 2 维持「可验收」状态；是否进入 Phase 3（Agent Memory）
等待人工审核决定。未提交 Git。
