# FireGuardian V2 Phase 4-A 报告：Agent Presentation Model + M10 Agent Analysis

> 日期：2026-08-15
> 分支：`v2-deepseek-agent`（HEAD 未变：`0eca3ad`）
> 状态：**Phase 4-A 完成，等待人工审核；未进入 Phase 4-B；未提交 Git**

---

## 1. M10 修改内容（reports/report_generator.py）

- `generate(data, agent_presentation=None)` 新增可选参数；`_write_markdown` / `_write_pdf` /
  `_write_event_json` 均接收并渲染 Agent Analysis；
- report.md / report.pdf 第 6 节「Agent 分析」：Rule Decision → Agent Assessment →
  Memory Evidence → Agent Explanation → Possible Cause → Recommended Action →
  Uncertainty → Final Decision → Tool Usage，数据全部来自 PresentationModel；
- **旧事件兼容**：无 Agent 数据时显示
  「Agent Analysis: Not available for this event.」并保留原 M7 决策块，报告照常生成；
- event.json 新增 `agent_analysis` 字段（无数据时为 null）；
- Agent Status = unavailable / invalid_output 时在 md/pdf 中展示对应回退说明；
- 空字段展示使用 `agent/display_text.py` 的 fallback 文案（如「暂无明确判断」「未给出处置建议…」）。

## 2. Agent 展示架构

```
M6 FireEvent → M7 FireDecision
        ↓
UiController（worker 线程，守卫 + 超时保护）
        ↓
FireGuardianLLMAgent（thinking=disabled，Memory/Tools）
        ↓
AgentOutcome → HybridDecision
        ↓
agent/presentation_models.py::AgentPresentationModel（一次性转换，不做重算）
        ↓
M10 ReportGenerator（md/pdf/event.json）  ← 仅消费 PresentationModel
```

- 新增 `agent/presentation_models.py`：`AgentPresentationModel`（字段按任务书 §26 +
  uncertainty / latency / token_usage / memory_results / display_fallbacks）、
  `AgentTriggerGuard`（调用去重/计数）、`run_agent_with_timeout`（整体超时保护）、
  `extract_memory_results`（紧凑相似事件摘要）。

## 3. Presentation Model 字段

`event_id / agent_status / rule_level / rule_score / rule_confidence / rule_reasons /
agent_assessment / agent_risk_level / agreement / memory_used / similar_event_count /
memory_summary / memory_results / possible_cause / reasoning_summary /
recommended_action / evidence_summary / uncertainty / tools_used / final_level /
agreement_hybrid / override / requires_review / reason / latency_ms / token_usage /
display_fallbacks / agent_available`

全部由 FireDecision + AgentOutcome + HybridDecision + Memory 结果转换，展示层不重新计算
final_level / similarity。

## 4. UiController 接入与线程策略（用户要求 1）

- DeepSeek 调用、Memory/Tool 调用、Hybrid 计算全部在 `FireGuardianAgentWorker`
  独立守护线程执行；GUI 只通过新增信号接收结果：
  `agent_started / agent_tool_called / agent_completed / agent_failed`；
- 帧处理线程、事件回调（M8/M9/M10）不被 Agent 阻塞；报告在事件结束时直接使用
  已完成的 PresentationModel，若 Agent 仍在运行则报告先展示已有数据，不等待；
- 冒烟已做 UiController 接线只读验证（实例化 + 信号存在性 + 守卫），不启动检测源。

## 5. 单事件调用保护与去重（用户要求 2）

- `AgentTriggerGuard`：仅允许 confirmed / risk_upgrade / disagreement / ended 四类触发；
- 去重：confirmed、ended、disagreement 每事件各最多一次；risk_upgrade 按等级去重
  （low→medium、medium→high 各自最多一次，同级不重复）；
- 计数：单事件累计调用 ≤ 4（`AGENT_MAX_CALLS_PER_EVENT`）；
- 记录：`logs/agent_trigger_calls.jsonl` 写入 `timestamp / event_id / trigger / call_count`；
- 单元测试覆盖：confirmed 一次、risk_upgrade 分级、ended/disagreement 一次、4 次上限、
  非法触发拒绝。

## 6. 超时保护（用户要求 3）

- `run_agent_with_timeout(agent, event, decision, trigger, timeout_seconds=60)`：
  Agent 整体运行超过 60s → 返回 `unavailable`（error=agent_timeout_exceeded），
  Hybrid 回退 M7；底层线程为 daemon，不无限等待、不阻塞 M8/M9/M10/GUI；
- 单元测试：慢 Agent（5s）在 0.3s 超时下返回 unavailable；快 Agent 正常返回。

## 7. Fallback 策略

- Agent unavailable / invalid_output → PresentationModel 保留 M7 规则等级，
  Final = M7，报告明确展示「DeepSeek 不可用/输出未通过 Schema 校验，回退 M7」；
- Memory 未使用 → 展示「Memory: Not Used」；Memory 不可用 → Agent 继续分析；
- 空字段全部走 `display_text.py` fallback（不出现 None/null/[]）。

## 8. 测试结果

- **总计 173/173 通过**（Phase 3-B 基线 157 + Phase 4-A 新增 16）；
- 新增 `tests/test_agent_presentation.py`（Presentation 构建 / 空字段 fallback /
  状态 / Hybrid high 保护 / Memory 提取 / 守卫去重计数 / 超时）；
- 新增 `tests/test_m10_agent_report.py`（Agent 报告生成 / Memory 摘要与相似事件 /
  旧事件兼容 / PDF 生成）；
- V1/V2 全部回归通过；全部 Mock，不消耗真实 API。

## 9. 真实 API 案例（Phase 4-A 冒烟）

`tools/v2_phase4a_agent_report_smoke.py`，`deepseek-v4-flash`、thinking=disabled：

| Case | Agent Status | Assessment | Agent Level | Final | Memory | Similar | Latency(ms) | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_01_real_fire | ok | UNCERTAIN | medium | **high** | Used | 2 | 14512.9 | 8218 |
| case_02_smoke_only | ok | SUSPICIOUS | medium | medium | Used | 3 | 8151.3 | 7635 |
| case_03_fire_smoke | ok | UNCERTAIN | medium | medium | Used | 4 | 13040.8 | 8858 |
| case_04_fireworks | ok | SUSPICIOUS | medium | medium | Used | 3 | 11329.8 | 8284 |
| case_05_normal | ok | UNCERTAIN | medium | medium | Used | 1 | 11794.9 | 8011 |
| case_06_ambiguous | invalid_output | - | - | medium | Not Used | 0 | 10933.6 | 5715 |

- 6 案例全部生成 md + pdf + event.json（`tools/v2_phase4a_output/reports/`）；
- **case_01（M7=high）：Agent=medium、Final=high，展示链路 M7 High 保护生效**；
- case_06 invalid_output → 回退 M7 且报告照常生成（Fallback + 兼容验证）；
- UiController 接线验证通过。

示例报告（case_01）关键区块：Rule HIGH / Agent UNCERTAIN(MEDIUM) / Memory Used (2)
（相似事件 FE-M01-001 0.93、FE-M01-002 0.91）/ Final HIGH / 空字段 fallback 正常。

## 10. 已知问题

1. Agent 结果质量仍依赖模型输出（本次 case_06 出现 schema 失败），已有 Fallback 兜底；
2. 报告「Agreement With Rule」取自 AgentResult.agreement_with_rule（模型自述），
   与 Hybrid 的 agreement（agreement_hybrid）分开保留，展示时需注意语义；
3. 事件结束立即生成报告时 Agent 可能仍在运行，报告将展示已有数据（不阻塞 M8/M9/M10）；
4. GUI（M11）展示尚未实现，属于 Phase 4-B。

## 11. 是否建议进入 Phase 4-B

Phase 4-A 验收项全部满足：PresentationModel ✓、M10 Agent Analysis（md+pdf+json）✓、
旧事件兼容 ✓、display_text fallback ✓、UiController Worker 接入（信号/守卫/超时）✓、
173/173 测试通过 ✓、真实 API 案例完成 ✓。

**结论：Phase 4-A 完成，可以进入 Phase 4-B（M11 GUI Agent 展示）。等待人工审核，不自动进入。**
