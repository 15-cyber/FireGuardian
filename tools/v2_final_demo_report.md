# FireGuardian V2.7 最终闭环演示报告

> 日期：2026-08-16
> 状态：**V2.7 演示完成；未提交 Git；最终人工 GUI 验收清单见 Phase 4-B 报告**

---

## 1. 演示方式

tools/v2_final_demo.py：真实视频 -> YOLO(V2) -> M6 -> M7 -> DeepSeek Agent(Memory) ->
Hybrid -> M8 截图 -> M9 日志 -> M10 报告(md/pdf/json) -> M11 GUI(offscreen 截图)。

## 2. 主闭环演示：烟火.mp4（真实火灾，V2 模型，conf=0.5）

```
YOLO(V2)         OK  576 帧（skip=2）
M6 事件聚合       OK  8 个事件，代表事件 FE-20260816205941-79e2a5
                  duration=0.6s smoke=0.630 fire=0.000
M7 Rule Decision OK  level=medium score=0.394
M9 事件日志       OK  events.jsonl
DeepSeek Agent   OK  SUSPICIOUS / medium（真实 API，thinking=disabled）
Hybrid Final     OK  rule=medium agent=medium final=medium
M8 证据截图       OK  3 张代表帧
M10 报告          OK  event.json / report.md / report.pdf
M11 GUI 展示      OK  gui_agent_panel.png
FINAL_DEMO_RESULT: PASS
```

产物：tools/v2_final_demo/event_FE-20260816205941-79e2a5/
（screenshots、reports、gui_agent_panel.png、final_demo_trace.json）。

## 3. 烟花2 闭环验证（重点）

- 生产 conf=0.5 + V2 模型：不再产生误报事件（fire 置信度 0.482 < 0.5，
  M6 聚合阈值 0.5 未达）——「烟花2 不再触发 fire 误报」目标达成；
- 即使以 conf=0.25 强制形成事件，Phase 6 联合测试已证明：
  Agent 判 SUSPICIOUS、Hybrid Final 保持 HIGH（M7 High 不可降级）；
- Phase 4-B GUI 冒烟中该案例的 Rule HIGH / Agent UNCERTAIN / Final HIGH 展示已通过。

## 4. V2 总体结论

- 闭环完整：视频 -> 检测 -> 聚合 -> 规则 -> Agent -> Hybrid -> 证据 -> 日志 ->
  报告 -> GUI，全链路真实运行；
- 安全底线：M7 High 在 Agent/Memory 联合下不被降级（真实火灾与烟花2 均验证）；
- 误报改善：V2 模型将烟花 fire 误检降低 2.2~15 倍，生产 conf=0.5 下
  负样本图级误报率 0.84%（V1 同测试集 2.61%）；
- 遗留项：V2 mAP50 略低于 V1（0.716 vs 0.733，无泄漏集上），
  如需更高召回可续训/升 YOLO11s；困难负样本池可扩充；conf 保持 0.5。

## 5. 待办（需人工决策）

- 人工 GUI 验收（Phase 4-B 报告第 14 节清单）；
- Git 提交整理（V2 分支至今未 commit，HEAD 仍 0eca3ad，等你指令）。
