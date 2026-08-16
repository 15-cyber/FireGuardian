# FireGuardian V2 Phase 3-A 报告：Agent Memory（结构化检索）

> 日期：2026-08-14
> 分支：`v2-deepseek-agent`（HEAD 未变：`0eca3ad`）
> 状态：**Phase 3-A 完成，等待人工审核；未自动进入 Phase 3-B；未提交 Git**

---

## 1. 架构

```
logs/events.jsonl（只读）或 tests/fixtures/*.jsonl
        ↓ 加载 + mtime 缓存
AgentMemory.load()
        ↓ 事件记录 × M7 决策 join（取最新决策）
memory_normalizer.normalize_many()
        ↓ 标准化为 EventMemoryRecord（duration 兼容 / 枚举 / 默认值 / 坏记录跳过）
当前事件（FireEvent + FireDecision → build_query_record）
        ↓
memory_retriever.StructuredMemoryRetriever.search()
        ↓ 加权相似度（默认只检索已结束事件）
MemorySearchResult
        ↓
ToolExecutor.execute("get_similar_events", ...)
        ↓ 紧凑结构化结果 + logs/agent_memory.jsonl 审计
Agent（Phase 3-B 接入）
```

新增文件：

| 文件 | 内容 |
| --- | --- |
| `agent/memory_models.py` | `EventMemoryRecord` / `MemoryConfig` / `SimilarEvent` / `MemorySearchResult` |
| `agent/memory_normalizer.py` | `normalize_event_record` / `normalize_many` / `build_query_record` |
| `agent/memory_retriever.py` | `StructuredMemoryRetriever`（加权相似度 + top_k + 阈值 + 排序） |
| `agent/agent_memory.py` | `AgentMemory`（加载/join/缓存/检索/统计）+ `MemoryToolDataSource` |
| `tests/fixtures/history_events.jsonl` | 固定历史 fixture（可重复测试专用） |
| `tests/fixtures/bad_history_events.jsonl` | 含坏记录的历史 fixture |
| `tests/test_memory_normalizer.py` / `test_memory_retriever.py` / `test_agent_memory.py` | 42 项单元测试 |

修改：`agent/tool_registry.py`（get_similar_events 新 schema）、`agent/tool_executor.py`
（协议 + 分发 + 数据源）、`agent/agent_logger.py`（memory 日志）、
`config/llm_agent.yaml`（memory 配置段）、既有测试同步更新。

## 2. 数据

真实 `logs/events.jsonl`（只读核验）：

- 总行数 252；有效事件记录 **194**；无 details 记录 8（遗留 M3/M7 行）；坏 JSON 0；
- `duration_seconds` 存在 59 条，仅 `duration`（legacy）**135 条** → legacy 兼容必要；
- 已结束事件 **29** 条（默认正式检索池）；
- 标准化字段来源：面积比 ← `max_*_area_ratio`；置信度 ← `avg_*_confidence`
  （日志无 max 字段，`confidence_source="avg"`）；规则信息 ← `fire_decision_agent`
  记录按 event_id join（取最新）。

测试 fixture（`tests/fixtures/`）与真实日志完全分离：fixture 仅用于可重复测试，
任何代码/测试均不写入或修改 `logs/events.jsonl`（真实数据仅做只读 sanity）。

## 3. 检索

- 特征与权重（合计 1.00，来自 config/llm_agent.yaml `memory.weights`）：
  fire_presence 0.20 / smoke_presence 0.20 / fire_area 0.15 / smoke_area 0.15 /
  duration 0.10 / risk_level 0.10 / confidence 0.05 / growth_trend 0.05；
- `top_k` 默认 5、上限 10；`min_similarity` 默认 0.65（低于阈值不返回，
  `matched_count` 统计达标数）；
- 排序：相似度 DESC → 同分先同 rule_level → 再按 duration 差异小者优先；
- **默认只检索已结束事件**（`only_ended_events=true`），active/confirmed 不进入正式结果；
- 当前事件按 event_id 严格排除；
- 匹配特征：单特征相似度 ≥ 0.5 记为 `matched_features`；
- 评分公式：duration/面积用 `1 - |a-b|/max(a,b,eps)` 并限制 [0,1]；等级用
  `1 - rank差/2`；置信度用双通道数值相似度均值；growth 精确匹配（unknown 中性 0.5）。

## 4. Tool

`get_similar_events` 重新启用（保留 ToolRegistry 参数校验）：

```json
{"event_id": "string(必填)", "top_k": "int 1~10 默认5", "min_similarity": "number 0~1 默认0.65"}
```

- event_id 白名单校验、未知参数拒绝；
- 当前事件不存在 → 明确错误（success=false，不崩溃）；
- 历史文件不存在 → `memory_available=false, reason="history_not_found"`，Agent 继续分析；
- 坏历史记录跳过，不影响查询；
- 返回紧凑结构（query_event / results / summary / memory_available / reason），
  不返回整段历史 JSON；
- 每次调用写 `logs/agent_memory.jsonl`（timestamp/event_id/tool/top_k/min_similarity/
  candidate_count/matched_count/results[event_id,score]/latency_ms/success；
  禁止 Key / Authorization / 完整 prompt / 大型原始列表）。

## 5. Agent

Phase 3-A 未改动 `FireGuardianLLMAgent` 主流程（Agent 主动调用 Memory、AgentResult 可选字段
`memory_used / similar_event_count / memory_summary`、Prompt 增强均属 **Phase 3-B**）。

## 6. 测试

- **总计 143/143 通过**（Phase 2.1 基线 101 + Phase 3-A 新增 42）；
- V1 回归（12 项）与 Phase 1/2/2.1 全部回归通过；
- Normalizer：duration_seconds 优先、legacy duration 一致/不一致单位、缺字段默认、
  非法记录跳过、多条混合、Enum 值转换、final_level 无 Hybrid 保持 None；
- Retriever：无历史、单条/多条、top_k/上限、阈值、当前事件排除、
  **仅已结束默认**、同分排序、各特征权重、duration/等级/growth 相似度；
- AgentMemory/Tool：fixture 加载统计、坏记录跳过、history_not_found、
  事件不存在报错、Tool 返回紧凑结果、参数校验、memory 日志、缓存与统计计数；
- 全部 Mock，不消耗真实 API。

## 7. 真实案例（Phase 3-A：只读本地验证，无 API）

对真实 `logs/events.jsonl` 执行只读 sanity：

```
query=FE-LOG-0001（已结束）candidate=21 matched(>=0.65)=21 latency=0.25ms
  -> FE-20260801201434-66f76e score=0.7485 features=[fire_presence,smoke_presence,
     smoke_area,duration,risk_level,confidence,growth_trend]
  -> FE-20260803190628-2bd9ed score=0.5937 ...
```

检索正确排除当前事件与未结束事件；固定场景的真实 API A/B（无 Memory vs 有 Memory）
留待 Phase 3-B。

## 8. Token / Latency

- Phase 3-A 不调用 LLM：本地结构化检索实测 <1ms（0.25ms）；
- 无 Memory vs Memory 的 Token/延迟对比属 Phase 3-B 真实 API 测试内容。

## 9. 已知问题

1. 真实日志中同一 `event_id` 可能多次出现（历史测试/运行残留），检索仅按 event_id
   排除当前事件，同名历史记录不会被排除；后续可增加「按时间窗口去重」。
2. 日志置信度为均值（无 max），`max_*_confidence` 映射为 `avg_*_confidence`
   并记录 `confidence_source="avg"`，语义上存在轻微偏差，已在标准化中显式标注。
3. 29 条 ended 真实历史池偏小，Phase 3-B 固定案例需使用 fixture 保证可比性。
4. legacy duration 中有无法确认单位的情况（如 FE-H-009 型），`duration_unit="unknown"`，
   检索仍可使用但置信度说明见报告。

## 10. 是否可以进入 Phase 3-B

Phase 3-A 验收项全部满足：Normalizer ✓、legacy duration 兼容 ✓、Retriever ✓、
get_similar_events 启用 ✓、top_k/threshold ✓、当前事件排除 ✓、坏记录不崩溃 ✓、
仅 ended 正式检索 ✓、final_level 保持 null 语义 ✓、fixture 与真实日志分离 ✓、
memory 日志 ✓、V1/V2 回归 ✓、143/143 测试 ✓。

**结论：Phase 3-A 可验收。等待人工审核后进入 Phase 3-B（Agent 集成 + 真实 API 固定案例），
不自动进入。**
