# FireGuardian V2 Phase 6 报告：Agent + YOLO 联合测试

> 日期：2026-08-16
> 状态：**Phase 6 完成，未提交 Git**

---

## 1. 检测对比（V1 vs V2，同一视频素材，conf>=0.25）

| 视频 | V1 fire最大面积 | V2 fire最大面积 | V1 fire最大置信度 | V2 fire最大置信度 | 结论 |
|---|---|---|---|---|---|
| 烟火.mp4（真实火灾） | 0.519 | 0.521 | 0.607 | 0.759 | 真实火灾保留 |
| me.mp4（烟花） | 0.140 | 0.009 | 0.671 | 0.525 | fire 误检下降约 15 倍 |
| 烟花1.mp4 | 0.104 | 0.022 | 0.626 | 0.334 | fire 误检下降约 4.6 倍 |
| 烟花2.mp4 | 0.781 | 0.349 | 0.819 | 0.482 | fire 误检下降约 2.2 倍，生产 conf=0.5 下不再触发 fire |

核心结果：V2 模型显著抑制烟花/强光的 fire 误检，
真实火灾（烟火.mp4）检出不受影响（置信度反而更高）。

## 2. Agent + Hybrid 联合评估（真实 API，deepseek-v4-flash，thinking=disabled）

| 场景 | V2 fire面积 | Agent | Agent 等级 | Final | Memory | 说明 |
|---|---|---|---|---|---|---|
| real_fire（烟火.mp4） | 0.521 | CONFIRMED_RISK | medium | high | 未用 | M7 high 保持 |
| fireworks_me（me.mp4） | 0.009 | invalid_output | - | medium | 未用 | Schema 失败回退 M7（Fallback 正常） |
| fireworks1（烟花1.mp4） | 0.022 | UNCERTAIN | medium | medium | Used(0) | 历史未达阈值，如实报告 |
| fireworks2（烟花2.mp4） | 0.349 | SUSPICIOUS | medium | high | Used(0) | M7 high 不被降级 |

- fireworks2 在 V2 下 fire 置信度 0.482（低于生产 0.5），即使触发事件，
  Agent 判 SUSPICIOUS（非 CONFIRMED_RISK）、Hybrid Final 保持 HIGH；
- 联合链路（YOLO -> M6 -> M7 -> Agent -> Hybrid）在 4 个场景全部运行成功，
  Fallback 与 M7 High 保护均正常。

## 3. 结论

- 误报率改善量化确认：烟花场景 fire 误检面积下降 2.2~15 倍；
- 安全底线保持：M7 High 在 Agent/Memory 介入下始终不被降级；
- 联合闭环可用：V2 模型 + Agent + Hybrid 端到端正常，Token/延迟与 Phase 4 持平
  （单事件 4.5k~8.8k tokens，7~18s）。

详细数据：tools/v2_phase6_joint_test/（detection_comparison.json、agent_joint_results.json）。
