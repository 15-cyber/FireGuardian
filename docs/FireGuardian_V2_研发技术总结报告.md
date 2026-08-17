# FireGuardian V2 研发技术总结报告

> 项目名称：FireGuardian V2 —— 基于 YOLO + 规则引擎 + DeepSeek Tool-Calling Agent 的智能火灾监控系统
> 文档版本：v2.0（最终封板版）
> 日期：2026-08-16
> 代码分支：`v2-deepseek-agent`（基于 `main` @ `0eca3ad`）
> 版本标签：`v0.1.0-baseline`（V1 稳定基线）；`v0.2.0-agent`（V2 封板，Stage B 待执行）

---

## 目录

1. 项目概述
2. 总体系统架构
3. V1 阶段
4. V1 数据集审计与模型实验
5. V1 问题发现与工程修复
6. V2 为什么需要 Agent
7. Phase 1：DeepSeek Agent Core
8. Phase 2：Hybrid Agent
9. Thinking A/B 实验
10. Phase 2.1：Agent 稳定性修复
11. Phase 3-A：Agent Memory
12. Phase 3-B：Agent + Memory
13. 真实烟花困难场景
14. Phase 4-A：Runtime + M10
15. Phase 4-B：M11 GUI
16. Phase 5：V2 模型优化
17. Phase 6：Agent + YOLO 联合测试
18. V2.7：最终闭环演示
19. 最终系统功能
20. 最终技术栈
21. 项目测试与验收体系
22. 项目目前存在的问题
23. 后续优化路线
24. 最终验收结论

附录 A：版本演进
附录 B：测试报告索引
附录 C：重要测试视频
附录 D：模型版本
附录 E：关键截图索引

---

## 第一章：项目概述

FireGuardian 是一个基于计算机视觉、规则决策与大语言模型 Agent 的智能火灾监控系统。

**项目目标**：以 YOLO 视觉检测为核心感知，以事件聚合（M6）与规则引擎（M7）为确定性安全底座，在此基础上引入 DeepSeek Agent 与历史记忆，形成「看得见、判断得清、解释得明、可审计」的智能火灾监控闭环。

**应用场景**：室内外火灾监控视频流 / 图片集；烟、火、烟花、雾、蒸汽、强光等易混淆场景的识别与复核；需要人工复核与过程留痕的安防场景。

**核心问题**：

- 单纯 YOLO 检测存在误报（烟花/强光/夜间灯光）与漏报（淡烟/小目标）；
- 规则引擎 M7 确定但缺少上下文解释，无法回答「为什么是这个等级」；
- 检测结果孤立，无法利用历史事件辅助当前判断；
- 引入 LLM 后必须保证安全底线（M7 High 不可被降级、失败可回退）。

**V1 → V2 演进路线**：

```text
V1：YOLO → M6 → M7 → M8 → M9 → M10 → M11
V2：YOLO → M6 → M7 → DeepSeek Agent → Tool Calling → Agent Memory
     → Hybrid Decision → Presentation Model → M10 → M11
```

---

## 第二章：总体系统架构

### 2.1 V1 架构

```text
YOLO11n（fire / smoke）
        ↓
M3 / M4 / M5  图片检测 / 视频检测 / 模拟验证流
        ↓
M6  EventAggregator（FireEvent：秒制时长、union area 面积特征）
        ↓
M7  FireDecisionAgent（规则评分 + 硬性覆盖）
        ↓
M8  Screenshot（confirmed / peak / upgrade / final）
        ↓
M9  Logs（events.jsonl / application.log）
        ↓
M10 Report（report.md / report.pdf）
        ↓
M11 GUI（PyQt6）
```

### 2.2 V2 架构

```text
YOLO11n（V2-15k，fire / smoke）
        ↓
M6  FireEvent
        ↓
M7  Rule Decision（确定性安全底座）
        ↓
DeepSeek Agent（deepseek-v4-flash，thinking=disabled）
        ↓
Tool Calling（get_event_summary / get_rule_decision /
            get_detection_history / get_event_evidence / get_similar_events）
        ↓
Agent Memory（结构化历史事件检索，仅已结束事件）
        ↓
Hybrid Decision Coordinator（M7 High 永不可降级）
        ↓
AgentPresentationModel（M10 / M11 共用展示数据）
        ↓
M10 Report（Agent Analysis 章节）→ M11 GUI（Agent Analysis Panel）
```

---

## 第三章：V1 阶段

V1 按 M1~M11 模块化实现，覆盖数据、训练、检测、聚合、决策、截图、日志、报告与 GUI 全链路。

| 模块 | 职责 |
| --- | --- |
| M1 | 项目基础设施、配置（config.yaml）、统一路径管理（PathManager） |
| M2 | YOLO 训练器（YOLOTrainer）、冒烟训练、Baseline V1 训练与评估 |
| M3 | 图片检测（fire/smoke 识别） |
| M4 | 视频检测（逐帧 decode） |
| M5 | 模拟验证流（事件输入） |
| M6 | FireEvent 事件聚合：状态机 CANDIDATE→CONFIRMED→ACTIVE→ENDED，面积与秒制时间特征 |
| M7 | 规则风险决策（评分 + 硬性覆盖，输出 level/score/confidence/reasons） |
| M8 | 证据截图（confirmed/peak/upgrade/final） |
| M9 | JSONL 日志（events.jsonl，含敏感字段脱敏） |
| M10 | 报告生成（md + pdf） |
| M11 | PyQt6 GUI（左侧数据源 / 中央画面 / 右侧信息 / 事件历史 / 训练 / 日志） |

M6 阶段实际发现并记录了两个关键问题（详见第五章）：bbox 重叠导致面积重复统计、frame_id 与 seconds 混用。

---

## 第四章：V1 数据集审计与模型实验

### 4.1 数据集审计

对本地数据集（train/valid/test）全量审计（105,257 张）：

| 指标 | 数值 |
| --- | --- |
| 总图片 / 标签 | 105,257 / 105,257 |
| 空标签（负样本） | 41,620 |
| fire 图 | 37,367 |
| smoke 图 | 48,784 |
| fire+smoke 图 | 22,514 |
| 越界框图片 | 319 |
| SHA-256 完全重复（跨集合） | 1,706 组 / 3,412 个文件 |
| pHash 近似重复对 | 636,434 对 |
| 抽样跨集合相邻帧 | 30 组抽样中 24 组疑似相邻视频帧 |

结论：数据量充足，但存在**跨集合重复/近似重复导致的数据泄漏风险**，训练/验证指标可能偏乐观；正式实验前需去重并重新划分。

### 4.2 冒烟模型（Smoke V1）

```text
train=3000  val=600  epochs=3  YOLO11n
```

主要作用：验证 M2 训练链路与 models/best.pt 产出，确认完整系统链路可运行。

### 4.3 Baseline V1

```text
train=15000  val=3000  test=3000  epochs=30  YOLO11n
```

独立 Test 结果（conf=0.25）：

| 指标 | 数值 |
| --- | --- |
| Precision | 0.7313 |
| Recall | 0.6533 |
| mAP50 | 0.7352 |
| mAP50-95 | 0.4557 |
| fire AP50 | 0.6942 |
| smoke AP50 | 0.7761 |

Baseline V1 替代冒烟模型成为 V1 正式默认模型（`models/best.pt`）。

---

## 第五章：V1 问题发现与工程修复

### 问题 1：bbox 面积重复统计

**现象**：同一类别多个 bbox 重叠时，简单求和导致 fire_area / smoke_area 被重复计算，事件特征失真。

**分析**：面积应基于并集而非框面积之和。

**方案与实现**：同类别 bbox 计算 union area，再归一化为 `fire_area_ratio` / `smoke_area_ratio`。

**验证**：M6 回归测试覆盖单框、分离框、完全重叠、包含、部分重叠等用例，全部通过。

### 问题 2：frame_id 与 seconds 混用

**现象**：部分时长判断把 frame_id 当作秒使用，事件时长失真。

**方案与实现**：统一秒制基准 `duration_seconds`，frame_id 通过 `frame_id / fps` 转换为秒。

**验证**：M7 决策统一使用秒制字段；`test_m6_m7_fix` 回归通过。

### 问题 3：GUI / Agent 链路失败可回退

**现象**：决策链路一旦异常，界面与下游模块可能中断。

**方案与实现**：Fallback = M7 规则决策；M8/M9/M10/M11 对失败容错。

**验证**：V1 全链路回归通过。

---

## 第六章：V2 为什么需要 Agent

V1 的 M7 是**规则引擎**，不是真正的 LLM Agent：它按固定权重与阈值输出等级，无法理解事件上下文、无法解释原因、无法利用历史证据。

V2 引入 DeepSeek Agent、Tool Calling、Memory 与 Hybrid，但**保留 M7 作为确定性安全底座**，而不是「LLM 替代 M7」：

```text
M7 负责安全（确定性规则，永不被降级）
LLM 负责分析（理解、解释、复核、历史查询、建议）
Hybrid 负责融合（安全单调性 + 审计记录）
```

关键原则：LLM 不替代 YOLO（不重新识别图片）、不替代 M7（不直接改等级）、不能降低 M7 High、失败必须回退 M7。

---

## 第七章：Phase 1 —— DeepSeek Agent Core

**技术**：DeepSeek V4 Flash（`deepseek-v4-flash`）、OpenAI 兼容 API、JSON Output、Tool Calling、Schema Validation、Retry / Timeout、Fallback、`.env` 密钥隔离。

**实际工作（真实研发过程）**：

1. 实现 DeepSeekClient、EventContext、ToolRegistry、ToolExecutor、Schemas、Prompts；
2. 首次 Agent Loop 真实测试时发现：**V4 Flash 默认 thinking 模式消耗大量 completion 预算，导致第二轮 JSON 输出失败**（非法 JSON：未终止字符串）；
3. 进一步诊断（第二轮 max_tokens 降为 768）发现 `finish_reason=length`、`completion_tokens=768` 打满、`raw_content_length=0`——模型把预算耗在推理阶段；
4. 修复：Agent Loop 改为 non-thinking（`thinking=disabled`），最终输出轮不带 tools 并设置 `tool_choice=none`，`max_tokens=768`，保留 Schema Validation 与 Fallback；
5. 重新测试通过。

**测试**：Phase 1 最终 66/66（含 V1 回归 12 项）；真实 API 冒烟 PASS。

---

## 第八章：Phase 2 —— Hybrid Agent

**实现**：FireGuardianLLMAgent（≤3 轮 Tool Calling + 最终严格 JSON）、HybridDecisionCoordinator、AgentStatus（ok/unavailable/invalid_output）、`should_call_agent` 调用时机控制。

**Hybrid 安全规则（Case 1~7）**：

| Case | M7 | Agent | Final | requires_review |
| --- | --- | --- | --- | --- |
| 1 | high | low | high | false |
| 2 | high | high | high | false |
| 3 | medium | high | high | true |
| 4 | medium | medium | medium | false |
| 5 | low | high | high | true |
| 6 | any | unavailable | = M7 | false |
| 7 | any | invalid_output | = M7 | false |

**真实测试记录**：

```text
M7 high + Agent low   → high（M7 High 不可降级）
M7 medium + Agent high → high + requires_review
Agent unavailable      → M7
Agent invalid_output   → M7
```

**测试**：Phase 2 总计 97/97（Phase 1 66 + 新增 31）。

---

## 第九章：Thinking A/B 实验

6 个固定案例 × 2 模式（thinking disabled / enabled）：

| 指标 | disabled | enabled |
| --- | --- | --- |
| 与 M7 等级对齐率 | 6/6 | 3/6 |
| 平均延迟 | 约 8.8s | 约 15.6s |
| Token 总量 | 33,879 | 34,911 |
| 成功率 | 6/6 | 6/6 |

**结论**：当前 FireGuardian 采用 thinking disabled。这是基于当前固定测试集得到的**工程决策**，不声称 thinking 模式普遍更差。

---

## 第十章：Phase 2.1 —— Agent 稳定性修复

**问题**：真实 API 中模型偶发把 `agent_assessment` 输出为 `CONFIRMED`（与事件状态 status 混淆、或缩写 CONFIRMED_RISK），触发 Schema 失败。

**修复**：

- 三个 Prompt（system / event_analysis / final_output）强化枚举取值与反例；
- 新增 `agent/display_text.py`：空字段展示 fallback（暂无明确判断 / 无引用证据 / 未给出处置建议…），不修改原始 AgentResult；
- AgentResult 可选字段向后兼容。

**测试**：Phase 2.1 总计 101/101。

---

## 第十一章：Phase 3-A —— Agent Memory

**实现**：

- `memory_models.py`：EventMemoryRecord / MemoryConfig / SimilarEvent / MemorySearchResult；
- `memory_normalizer.py`：字段兼容（duration/duration_seconds、max_*_area_ratio、avg_*_confidence）、枚举标准化、坏记录跳过；
- `memory_retriever.py`：结构化加权相似度（8 特征，权重合计 1.00）、top_k、阈值、排序；
- `agent_memory.py`：只读加载 + mtime 缓存 + M7 决策 join + MemoryToolDataSource；
- `get_similar_events` 重新启用（event_id / top_k 1~10 / min_similarity 0~1）。

**关键约束**：默认只检索**已结束**历史事件；当前事件按 event_id 排除；无 Hybrid 日志时 `final_level` 保持 null；fixture 与真实日志分离。

**说明**：第一版**没有引入 Embedding / Vector DB**——当前结构化事件数量有限，结构化检索透明且足够。

**测试**：Phase 3-A 总计 143/143。

---

## 第十二章：Phase 3-B —— Agent + Memory

**A/B 实验**：6 固定案例 ×（Without Memory / With Memory）。

**重点记录**：

```text
case_02 smoke_only：Without Memory → UNCERTAIN
                    With Memory    → SUSPICIOUS（3 个相似烟雾事件，判断更明确）

case_03 fire_smoke：Without Memory → SUSPICIOUS
                    With Memory    → UNCERTAIN（4 个相似事件，更保守）
```

**说明**：Memory 不是单纯让模型风险升级，而是提供历史证据让 Agent 重新评估；M7 High 始终不被降级；Without Memory 臂中模型偶发「幻觉调用」被白名单拦截。

**测试**：Phase 3-B 总计 157/157。

---

## 第十三章：真实烟花困难场景

独立重要章节，素材：

| 视频 | 特点 |
| --- | --- |
| me.mp4 | 烟花/短时亮光+烟雾 |
| 烟花1.mp4 | smoke 主导（smoke 最大面积约 0.54，fire 约 0.03） |
| 烟花2.mp4 | fire 最大面积约 0.529（强光误检风险），M7=high |

**重点工程案例**：烟花2 中 YOLO 曾把强光误检为 fire（最大面积约 0.529、置信度 0.819），但：

- Agent 结合历史烟花模式判 SUSPICIOUS/UNCERTAIN，而非 CONFIRMED_RISK；
- Memory 提供「历史相似多为短时烟雾/烟花」证据；
- **Hybrid Final 始终保持 HIGH——烟花2 最终没有因为 Memory/Agent 而错误降低 M7 High**。

---

## 第十四章：Phase 4-A —— Runtime + M10

**实现**：

- `presentation_models.py`：AgentPresentationModel（M10/M11 共用展示数据，不做重算）；
- UiController Worker 接入：DeepSeek / Memory / Tool / Hybrid 全部在独立守护线程；
- `AgentTriggerGuard`：confirmed / risk_upgrade / disagreement / ended 去重，单事件 ≤4 次，risk_upgrade 按等级去重；
- `run_agent_with_timeout`：整体超时 60s → unavailable 回退 M7，不阻塞 M8/M9/M10/GUI；
- M10 `report_generator.py`：新增 Agent Analysis 章节（Rule / Assessment / Memory / Explanation / Cause / Action / Final / Tools），旧事件无 Agent 数据时显示 Not available，md/pdf/event.json 三端一致。

**测试**：Phase 4-A 总计 173/173。

---

## 第十五章：Phase 4-B —— M11 GUI

**实现**：

- `ui/agent_panel.py`：AgentAnalysisPanel（Hybrid Final → Rule/Agent/Memory → Explanation/Cause/Action → Details 四级信息层级）；
- 四个 Qt Signal（agent_started / agent_tool_called / agent_completed / agent_failed）驱动状态切换（Analyzing / Tool / Completed / Fallback / Timeout）；
- 事件历史表新增 Rule / Agent / Final / Memory 列；
- 关闭安全：closeEvent 增加「正在停止Agent...」，Agent Worker 为 daemon，不阻塞退出；
- 历史详情只使用该事件自身 PresentationModel，无字段显示「未提供」。

**验证**：offscreen（QT_QPA_PLATFORM=offscreen）自动化测试覆盖全状态；真实 API + offscreen GUI 冒烟通过；烟花2 展示 Rule HIGH / Agent UNCERTAIN / Final HIGH。

**测试**：Phase 4-B 总计 186/186。

---

## 第十六章：Phase 5 —— V2 模型优化

### 16.1 数据重划分（消除泄漏）

- 全量扫描 105,257 张；
- 基于审计 exact_duplicates + near_duplicates（distance ≤ 8）做 union-find 片段分组（99,730 片段，最大 212），**同一片段不跨集合**；
- 排除越界框 288 张；
- 类别配额分配：train 15,066 / val 3,000 / test 3,000（negative/smoke_only/fire_only/both 四类齐全且比例近似全量）；
- 输出 `datasets/v2_15k_*.txt` + `fire_smoke_v2_15k.yaml`，扫描缓存可复现。

### 16.2 困难负样本

从烟花1/烟花2/me.mp4 抽取 66 张「被误检为 fire/smoke」帧（空标签）并入训练集。

### 16.3 训练

```text
yolo11n、40 epochs、batch=16、imgsz=640、patience=15、seed=42
训练时长约 4.8h；best epoch=40；val mAP50 0.294 → 0.694
产物：models/v2_15k_best.pt（未覆盖 models/best.pt）
```

### 16.4 评估（同一 v2 test 集公平对比）

| 模型 | mAP50 | mAP50-95 | P@0.25 | R@0.25 | 负样本图级误报率（0.25/0.4/0.5） |
| --- | --- | --- | --- | --- | --- |
| V1 baseline | 0.7325 | 0.4491 | 0.7456 | 0.6419 | 5.63% / 3.70% / 2.61% |
| **V2-15k** | **0.7161** | **0.4334** | **0.7397** | **0.6356** | **2.52% / 1.34% / 0.84%** |

### 16.5 conf 阈值结论

**建议生产 conf 保持 0.50**：V2 在该阈值图级误报率 0.84%（V1 同测试集 2.61%），符合误报优先策略；如需更高召回可降 0.25（误报 2.52%）。

---

## 第十七章：Phase 6 —— Agent + YOLO 联合测试

### 17.1 检测对比（V1 vs V2，同一视频，conf≥0.25）

| 视频 | V1 fire 最大面积 | V2 fire 最大面积 | V1 fire 最大置信度 | V2 fire 最大置信度 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 烟火.mp4（真实火灾） | 0.519 | 0.521 | 0.607 | 0.759 | 真实火灾保留 |
| me.mp4（烟花） | 0.140 | 0.009 | 0.671 | 0.525 | fire 误检下降约 15 倍 |
| 烟花1.mp4 | 0.104 | 0.022 | 0.626 | 0.334 | fire 误检下降约 4.6 倍 |
| 烟花2.mp4 | 0.781 | 0.349 | 0.819 | 0.482 | 误检下降约 2.2 倍，生产 conf=0.5 下不再触发 fire |

### 17.2 Agent + Hybrid 联合评估（真实 API）

| 场景 | V2 fire 面积 | Agent | Agent 等级 | Final | 说明 |
| --- | --- | --- | --- | --- | --- |
| real_fire（烟火.mp4） | 0.521 | CONFIRMED_RISK | medium | high | M7 high 保持 |
| fireworks_me（me.mp4） | 0.009 | invalid_output | - | medium | Schema 失败回退 M7 |
| fireworks1（烟花1.mp4） | 0.022 | UNCERTAIN | medium | medium | 历史未达阈值，如实报告 |
| fireworks2（烟花2.mp4） | 0.349 | SUSPICIOUS | medium | high | M7 high 不被降级 |

**结论**：V2 模型显著抑制烟花/强光 fire 误检（2.2~15 倍），真实火灾检出保留；联合链路 YOLO→M6→M7→Agent→Hybrid 全部运行成功，Fallback 与 M7 High 保护正常。

---

## 第十八章：V2.7 —— 最终闭环演示

使用 `tools/v2_final_demo.py` 以 V2 模型 + 真实视频跑完整闭环（烟火.mp4，conf=0.5）：

```text
YOLO(V2)         OK  576 帧（skip=2）
M6 事件聚合       OK  8 个事件（代表事件 smoke=0.630）
M7 Rule Decision OK  level=medium score=0.394
M9 事件日志       OK  events.jsonl
DeepSeek Agent   OK  SUSPICIOUS / medium
Hybrid Final     OK  final=medium
M8 证据截图       OK  3 张代表帧
M10 报告          OK  event.json / report.md / report.pdf
M11 GUI 展示      OK  gui_agent_panel.png
FINAL_DEMO_RESULT: PASS
```

**烟花2 闭环验证**：生产 conf=0.5 + V2 模型下不再产生误报事件（fire 置信度 0.482 < 0.5）；即使强制形成事件，Agent 判 SUSPICIOUS、Hybrid Final 保持 HIGH。

---

## 第十九章：最终系统功能

- **图片检测**：加载图片输出 fire/smoke 检测结果；
- **视频检测**：实时/离线视频逐帧检测；
- **火灾事件识别**：M6 聚合为 FireEvent（秒制时长、面积特征、增长趋势）；
- **规则风险判断**：M7 输出 low/medium/high + score/confidence/reasons；
- **LLM Agent 分析**：DeepSeek 主动查询工具、解释规则、给出可能原因与建议；
- **历史 Memory**：相似已结束事件检索，辅助证据；
- **Hybrid 安全决策**：融合规则与 Agent，M7 High 不可降级，可升级但进入人工复核；
- **截图**：confirmed/peak/upgrade/final 证据归档；
- **日志**：事件、决策、Agent、Memory、工具调用 JSONL 全程留痕（无 API Key）；
- **自动报告**：M10 生成含 Agent Analysis 的 md/pdf；
- **GUI**：M11 展示 Rule / Agent / Memory / Hybrid Final 与事件历史。

---

## 第二十章：最终技术栈

| 类别 | 技术 |
| --- | --- |
| 语言 | Python 3.11 |
| 视觉 | OpenCV |
| 检测 | Ultralytics YOLO |
| 模型（V1） | YOLO11n Baseline V1 |
| 模型（V2） | YOLO11n V2-15k |
| Agent | DeepSeek V4 Flash |
| Agent 协议 | Tool Calling / JSON Output |
| 决策 | Rule Engine + Hybrid |
| Memory | Structured Event Retrieval（无 Embedding/Vector DB） |
| GUI | PyQt6 / Qt |
| 报告 | Markdown + PDF（reportlab） |
| 日志 | JSONL |
| 测试 | Python unittest |
| 版本控制 | Git（tag：v0.1.0-baseline；v0.2.0-agent 待封板） |
| GPU | NVIDIA RTX 4060 Laptop（CUDA） |

---

## 第二十一章：项目测试与验收体系

```text
数据审计 → Smoke Training → Baseline Training → 独立 Test
→ M6/M7 回归 → Agent Loop → Hybrid → Memory → M10 → M11 → 人工 GUI 验收
```

各阶段最终测试数量（全部来自实际运行记录）：

| 阶段 | 测试数 |
| --- | --- |
| Phase 1 | 66 |
| Phase 2 | 97 |
| Phase 2.1 | 101 |
| Phase 3-A | 143 |
| Phase 3-B | 157 |
| Phase 4-A | 173 |
| Phase 4-B | 186 |
| 当前总计（Phase 5/6 后） | **186** |

全部测试为 Mock / 离线，不消耗真实 API 额度；真实 API 通过独立冒烟脚本验证。

---

## 第二十二章：项目目前存在的问题

如实列出（不以「已知问题」包装为「全部解决」）：

1. fire 困难负样本不足（当前仅 66 张本地烟花负样本）；
2. 烟花强光仍存在部分误检（V2 已大幅降低，但 conf=0.25 下仍可能出现 fire 检出）；
3. 真实 ended 历史事件数量有限（约 29 条），Memory 实际召回有限；
4. 历史 `event_id` 存在重复，检索仅按 event_id 排除当前事件；
5. 日志置信度为 avg 而非 max，语义存在差异；
6. legacy duration 单位不确定时只能标记 unknown；
7. Agent 输出存在偶发 Schema 失败（已由 Fallback 兜底）；
8. Agent 实时延迟为秒级（4.5k~8.8k tokens / 7~18s），不适用于每帧；
9. 当前 Memory 为结构化检索，非向量语义检索。

---

## 第二十三章：后续优化路线

### V2.1（GUI / 报告细节）

- Agent 完成后补刷最终报告（当前事件结束时若 Agent 未完成，报告先展示已有数据）；
- Memory 历史表区分 Not Used / Unavailable；
- 历史详情按事件自身数据完善展示。

### V2 Model（模型优化）

- 扩充困难负样本：烟花/强光/车灯/路灯/云雾/蒸汽/夜间灯光；
- 视频分组数据划分（已落地 V2-15k，可继续扩大）；
- 更长训练（50~60 轮）或更大数据集（50k）；
- 必要时 YOLO11n → YOLO11s 以提升召回。

### V3（未来方向，未实施）

- Embedding / Vector Memory；
- 更强 Agent Memory（跨事件推理）；
- 多模态视觉 Agent。

---

## 第二十四章：最终验收结论

```text
FireGuardian V2 已完成：
V1 核心视觉检测
+ 规则安全底座
+ DeepSeek Agent
+ Tool Calling
+ Agent Memory
+ Hybrid Decision
+ M10 Agent Report
+ M11 GUI
```

- 自动化测试 186/186 通过；
- 用户已完成最终人工 GUI 验收（按 Phase 4-B 人工验收清单），图片、视频、Agent、Memory、M10、M11 人工检查通过；
- 真实闭环演示（烟火.mp4）PASS；烟花2 强光误检显著下降且 M7 High 未被降级；
- **当前版本可以作为稳定 Demo 版本封存**；Git 封板（Stage B：commit + `v0.2.0-agent` tag）按任务书后续执行。

---

## 附录 A：版本演进

| 版本 | 说明 | 状态 |
| --- | --- | --- |
| v0.1.0-baseline | FireGuardian V1 稳定基线（tag 已创建） | 已完成 |
| v0.2.0-agent | FireGuardian V2 Agent Stable Demo | Stage B 待封板 |

---

## 附录 B：测试报告索引（仅列真实存在文件）

```text
tools/
    dataset_audit.md
    smoke_training_report.md
    baseline_v1_training_report.md
    baseline_v1_visual_audit.md
    m6_m7_fix_report.md
    v2_phase1_report.md
    v2_phase2_report.md
    v2_phase2_1_fix_record.md
    v2_phase3_report.md
    v2_phase3b_report.md
    v2_phase4a_report.md
    v2_phase4b_report.md
    v2_phase5_report.md
    v2_phase6_report.md
    v2_final_demo_report.md
    v2_conf_analysis.md
```

---

## 附录 C：重要测试视频

- me.mp4（烟花/短时亮光）
- 测试视频/烟花1.mp4（smoke 主导）
- 测试视频/烟花2.mp4（强光误检风险，M7=high）
- 烟火.mp4（真实火灾）

**说明：测试视频不进入 Git 源码。**

---

## 附录 D：模型版本

| 模型 | 用途 | 默认 | SHA-256（前 16 位） |
| --- | --- | --- | --- |
| smoke_v1_best.pt | V1 冒烟模型 | 否 | - |
| baseline_v1_best.pt | V1 Baseline 最佳 | 否 | 5b84350ebdfaf489 |
| models/best.pt | V1 生产默认（config 默认加载） | 是 | - |
| models/v2_15k_best.pt | V2 模型（Phase 5） | 否（可选切换） | 9e674c156d8d638a |

---

## 附录 E：关键截图索引

- **GUI Agent Analysis 面板（Phase 4-B，fireworks2 重点案例）**：
  `tools/v2_phase4b_screenshots/fireworks2_agent_panel.png`

  ![GUI Agent Analysis 面板（Phase 4-B：fireworks2）](tools/v2_phase4b_screenshots/fireworks2_agent_panel.png)

- **V2 闭环演示 GUI 面板（烟火.mp4 主闭环）**：
  `tools/v2_final_demo/event_FE-20260816205941-79e2a5/gui_agent_panel.png`

  ![V2 闭环演示 GUI 面板（烟火.mp4）](tools/v2_final_demo/event_FE-20260816205941-79e2a5/gui_agent_panel.png)

- **V1 评估 F1 曲线**：`runs/baseline_train/baseline_v1/eval_curves/f1_curve.png`

  ![V1 Baseline 评估 F1 曲线](runs/baseline_train/baseline_v1/eval_curves/f1_curve.png)

- **V2 评估 F1 曲线**：`runs/v2_model/v2_15k/eval_curves/f1_curve.png`

  ![V2-15k 评估 F1 曲线](runs/v2_model/v2_15k/eval_curves/f1_curve.png)

---

*本报告由 FireGuardian V2 各阶段真实测试与运行记录整理而成；所有指标均来自实际执行结果。*
