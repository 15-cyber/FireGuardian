# FireGuardian V2 Phase 3-B 报告：Agent Memory 集成与真实案例验证

> 日期：2026-08-14
> 分支：`v2-deepseek-agent`（HEAD 未变：`0eca3ad`）
> 状态：**Phase 3-B 完成，等待人工审核；未进入 Phase 4；未提交 Git**

---

## 1. Agent Memory 架构

```
当前 FireEvent + M7 FireDecision
        ↓
FireGuardianLLMAgent（thinking=disabled）
        ↓ 信息不足/模糊/矛盾时主动调用
get_similar_events(event_id, top_k, min_similarity)
        ↓
MemoryAgentDataSource → AgentMemory → StructuredMemoryRetriever
        ↓ 只读 fixture/events.jsonl（默认仅已结束事件）
Top-K 相似事件（紧凑结构化结果）
        ↓ tool message 回传
DeepSeek 继续分析
        ↓
AgentResult（memory_used / similar_event_count / memory_summary 可选字段）
        ↓
HybridDecisionCoordinator（安全规则不变）→ Final Decision
```

本阶段修改（均在允许清单内）：`agent/schemas.py`（可选 Memory 字段）、
`agent/deepseek_agent.py`（工具白名单 + Memory 字段确定性后填 + 幻觉工具拦截）、
`agent/prompts/*`（历史辅助证据规则）、`agent/agent_memory.py`（复合数据源）、
`agent/agent_logger.py`（memory 日志补充）、`agent/tool_registry.py`（get_similar_events
启用）。`agent/hybrid_decision.py` 未做任何修改（安全规则零改动）。

## 2. Memory Tool 工作流程

1. Agent 判断需要历史信息 → 调用 `get_similar_events`；
2. 参数经 ToolRegistry 校验（event_id 白名单 / top_k 1~10 / min_similarity 0~1）；
3. 查询记录由 `query_provider`（当前 FireEvent+FireDecision）构建，历史来自案例 fixture；
4. 只检索**已结束**历史事件，当前事件按 event_id 排除；
5. 加权相似度（阈值 0.65、top_k=5）→ 紧凑结果（query_event/results/summary）；
6. 结果按 tool message 回传，写 `logs/agent_memory.jsonl`（event_id/matched ids/相似度/latency/success，无 Key/prompt）；
7. Agent 最终输出经 Schema 校验；`memory_used/similar_event_count/memory_summary` 由
   **工具实际执行结果确定性回填**（不依赖模型自述），若模型输出 `null` 按默认值处理。

## 3. 六个固定案例

`tests/fixtures/memory/case_01..06/history.jsonl`（与真实 `logs/events.jsonl` 分离，仅用于可重复实验），
覆盖 real fire / smoke only / fire+smoke / fireworks / normal / ambiguous；案例定义共享于
`tests/memory_case_defs.py`（单元测试与真实 API 脚本共用，保证一致性）。

## 4. Without Memory vs With Memory（真实 API A/B，12 次实验）

`tools/v2_phase3_memory_smoke.py`，`deepseek-v4-flash`、thinking=disabled。

| Case | Arm | Memory | Similar | Tools | LLM | Latency(ms) | Tokens | Assessment | Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_01_real_fire | without | False | 0 | 4 | 2 | 7617 | 4364 | UNCERTAIN | **high** |
| case_01_real_fire | with | True | 2 | 5 | 3 | 10245 | 8282 | UNCERTAIN | **high** |
| case_02_smoke_only | without | False | 0 | 5 | 3 | 9059 | 6899 | UNCERTAIN | medium |
| case_02_smoke_only | with | True | 3 | 5 | 3 | 8754 | 7739 | SUSPICIOUS | medium |
| case_03_fire_smoke | without | False | 0 | 5 | 4 | 13907 | 10079 | SUSPICIOUS | medium |
| case_03_fire_smoke | with | True | 4 | 5 | 3 | 11927 | 8711 | UNCERTAIN | medium |
| case_04_fireworks | without | False | 0 | 4 | 2 | 8942 | 4766 | SUSPICIOUS | medium |
| case_04_fireworks | with | True | 3 | 5 | 3 | 11322 | 8115 | SUSPICIOUS | medium |
| case_05_normal | without | False | 0 | 0 | 2 | 11711 | 5339 | ❌非法JSON→回退 | low |
| case_05_normal | with | True | 1 | 5 | 3 | 11740 | 8271 | SUSPICIOUS | low |
| case_06_ambiguous | without | False | 0 | 5 | 3 | 9468 | 7435 | UNCERTAIN | medium |
| case_06_ambiguous | with | True | 2 | 5 | 2 | 9612 | 5713 | SUSPICIOUS | medium |

- Without Memory 臂全部 `memory_used=False`（模型幻觉调用被拦截，未执行）；
- With Memory 臂 6/6 `memory_used=True`，similar_event_count 1~4；
- **case_01（M7=high）两臂 Final 均 high，M7 High 未降级**。

## 5. Token / Latency

- Memory 增加 token：多数案例 +0.8k~3.9k，两例为负（模型轮数减少），平均约 **+1.3k tokens/事件**；
- Memory 延迟影响：平均约 **+0.5s/事件**（受模型轮数波动影响，非 Memory 检索本身，检索 <1ms）；
- 结论：Memory 成本中等可控，未出现「Memory 严重拖慢 Agent」的停止条件。

## 6. Agent Decision 差异

- case_02：UNCERTAIN → SUSPICIOUS（3 个相似烟雾事件，判断更明确）；
- case_03：SUSPICIOUS → UNCERTAIN（4 个相似，更保守）；
- case_04（烟花）：两臂均 SUSPICIOUS/medium，**Final 未错误升级**；With Memory 的
  memory_summary 明确引用历史烟花/烟雾模式（证据增强）；
- case_06：UNCERTAIN → SUSPICIOUS（2 个相似）；
- case_01/05：Final 由 Hybrid 保持 M7（high/low），Memory 未突破安全底线。

## 7. Memory 使用率

- With Memory 臂：6/6 主动调用（证据不足/模糊案例），符合「信息不足时查询」设计；
- Without Memory 臂：0/6（幻觉调用被拦截，统计口径干净）；
- 每事件 Memory 调用 = 1，工具调用/LLM 调用上限维持 max_tool_rounds=3、单事件 ≤4 次 LLM。

## 8. 历史相似事件数量

case_01: 2 / case_02: 3 / case_03: 4 / case_04: 3 / case_05: 1 / case_06: 2；
未出现「历史为空仍强行编造结论」的情况（case_05 有 1 条低相似命中，summary 如实说明）。

## 9. 烟花视频测试（两段新视频单独列出）

输入只读：`测试视频/烟花1.mp4`、`测试视频/烟花2.mp4`；`models/best.pt` 只读，未训练、未改 conf。
复用 M4 `VideoDetector.decode()`（frame_skip=2）+ M6 `EventAggregator` + M7 `FireDecisionAgent`。
输出：`tools/v2_phase3_fireworks/`（detection/video_summary/representative_frames（各 20 张）/agent 结果）。

| 视频 | FPS | 帧数 | 分辨率 | 时长 | fire正帧 | smoke正帧 | fire最大面积 | smoke最大面积 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 烟花1.mp4 | 30.0 | 301（处理151） | 4320×7680 | 10.0s | 9 | 116 | 0.027 | 0.543 |
| 烟花2.mp4 | 30.0 | 446（处理223） | 4320×7680 | 14.9s | 8 | 12 | 0.529 | 0.603 |

代表事件 Agent 对比（历史 fixture = case_04_fireworks）：

| 事件 | M7 | Without Memory | With Memory |
| --- | --- | --- | --- |
| 烟花1-e1（smoke 0.54, fire 0） | medium | SUSPICIOUS/medium | SUSPICIOUS/medium，1 相似（0.88），summary 引用历史烟雾事件 |
| 烟花1-e2（smoke 0.28, fire 0） | medium | SUSPICIOUS/medium | UNCERTAIN/medium，1 相似（0.82） |
| 烟花2-e1（smoke 0.43, fire 0.14） | **high** | SUSPICIOUS/medium → **Final high** | UNCERTAIN/medium，4 相似，summary 指出火焰数据自相矛盾 → **Final high** |
| 烟花2-e2（smoke 0.21, fire 0） | low | SUSPICIOUS/medium → Final medium | SUSPICIOUS/medium，1 相似（0.76） |

**重点验证**：烟花2-e1 的 M7=high 在两臂均未被降级（Hybrid 保护生效）；
With Memory 在 3/4 事件给出基于历史证据的 memory_summary，且烟花1 两事件均未因历史烟花模式而错误升级。

## 10. 是否改善异常场景理解

- **改善**：case_02/04/06 与烟花1/2 事件中，Memory 提供了「历史多为短时烟雾/烟花、未升级高风险」的证据，
  memory_summary 能引用具体相似事件与相似度（如 0.88/0.82/0.76），解释质量明显好于无 Memory 臂；
- **未突破安全底线**：没有任何案例因 Memory 把 M7=high 降级，也没有「历史烟花→直接判定非火灾」的硬编码结论；
- 按照任务书 §28 的 Memory Value 定义：决策一致性保持 + 证据/解释增强 − 约 +1.3k token 成本 → **Memory 有效**。

## 11. 已知问题

1. `event_id` 可能重复（历史运行残留）：检索仅按 event_id 排除当前事件，同名历史记录不会被排除，后续按时间窗口去重；
2. 置信度为日志均值（`avg_*_confidence`），检索用 `confidence_source="avg"`，语义略有偏差；
3. legacy duration（135 条真实记录）单位不能完全确认时标记 `duration_unit="unknown"`；
4. 真实 ended 历史仅 29 条，A/B 固定案例依赖 fixture，历史池偏小；
5. 无 Memory 臂中模型偶发「幻觉调用」被禁用的 `get_similar_events`，已被白名单拦截（安全兜底，非 Memory 使用）；
6. case_05 无 Memory 臂出现一次模型非法 JSON（Schema/Fallback 兜底，Final 回退 M7=low）；
7. 烟花2 中存在高面积「fire」检出（0.53，疑似烟花强光误检），Agent 与 Hybrid 均未据此错误升级，相关困难样本建议 Phase 6 模型优化时纳入。

## 12. 是否建议进入 Phase 4

验收项全部满足：Agent 自主调用 get_similar_events ✓、Tool Result 回传 ✓、Agent 利用历史继续分析 ✓、
无/有 Memory A/B 完成 ✓、6 固定案例完成 ✓、两段新烟花视频完成测试 ✓、M7 High 未降级 ✓、
Memory 失败不影响系统 ✓、V1/V2 测试 157/157 通过 ✓、Token/Latency 有统计 ✓、
memory_used/similar_event_count 正确 ✓、Phase 3B 报告已生成 ✓。

**结论：Phase 3-B 完成，可以进入 Phase 4（M10/M11 Agent 展示集成）。等待人工审核，不自动进入。**
