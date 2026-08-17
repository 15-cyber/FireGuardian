# FireGuardian 架构说明

完整研发过程见 [FireGuardian_V2_研发技术总结报告.md](FireGuardian_V2_研发技术总结报告.md)。

## V1 架构

```text
YOLO11n（fire / smoke）
  → M3/M4/M5 图片 / 视频 / 模拟检测
  → M6 EventAggregator（FireEvent：秒制时长、union area 面积特征）
  → M7 FireDecisionAgent（规则评分 + 硬性覆盖）
  → M8 Screenshot → M9 Logs → M10 Report → M11 GUI
```

## V2 架构

```text
YOLO11n（V2-15k）
  → M6 FireEvent
  → M7 Rule Decision（确定性安全底座）
  → DeepSeek Agent（deepseek-v4-flash，thinking=disabled）
  → Tool Calling（事件摘要 / 规则决策 / 检测历史 / 证据 / 相似事件）
  → Agent Memory（结构化历史事件检索，仅已结束事件）
  → Hybrid Decision Coordinator（M7 High 永不可降级）
  → AgentPresentationModel → M10 Report / M11 GUI
```

## 核心模块职责

| 模块 | 职责 |
| --- | --- |
| agent/ | DeepSeek Client、Agent 主循环、Tool Registry/Executor、Hybrid、Memory、Presentation |
| aggregator/ | M6 事件聚合（状态机、秒制时长、面积特征） |
| detect/ | M3 图片检测 |
| video_detect/ | M4 视频检测 |
| reports/ | M10 报告生成（md/pdf/json，含 Agent Analysis） |
| ui/ | M11 GUI（监控 / 训练 / 日志 / 事件历史 / Agent 面板） |
| train/ | M2 YOLO 训练器 |
| tools/ | 数据集审计、训练 / 评估入口、真实 API 冒烟、闭环演示 |
| tests/ | 单元 / 集成测试（Mock + 离线） |

## 安全设计

- M7 是确定性规则底座，LLM 不直接替代它；
- Hybrid 安全融合：M7 High 不可被 Agent / Memory 降级；
- Agent unavailable / invalid JSON / timeout 时回退 M7；
- DeepSeek / Memory / Hybrid 全部在 Worker 线程执行，GUI 只接收 Qt Signal。
