# 研发历史

完整研发过程（发现问题 → 分析 → 方案 → 实现 → 测试 → 修复 → 验证）见
[FireGuardian_V2_研发技术总结报告.md](FireGuardian_V2_研发技术总结报告.md)。

## 阶段时间线

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| V1 | YOLO 检测 + M6~M11 全链路（Baseline V1） | 完成 |
| Phase 1 | DeepSeek Agent Core（Client/Tools/Schema/Prompts） | 完成 |
| Phase 2 | Hybrid Agent + Fallback + Thinking A/B | 完成 |
| Phase 2.1 | Agent 枚举稳定性修复 + 展示层 fallback | 完成 |
| Phase 3-A | 结构化 Agent Memory | 完成 |
| Phase 3-B | Agent + Memory 真实 API 验证 + 烟花视频 | 完成 |
| Phase 4-A | Agent Runtime + M10 Agent Report | 完成 |
| Phase 4-B | M11 GUI Agent 展示 | 完成 |
| Phase 5 | V2 模型优化（无泄漏数据划分 / 困难负样本 / 训练评估） | 完成 |
| Phase 6 | Agent + YOLO 联合测试 | 完成 |
| V2.7 | 最终闭环演示 | 完成 |

## 关键问题与修复（摘要）

- bbox 重叠导致面积重复统计 → union area；
- frame_id 与 seconds 混用 → 统一秒制基准；
- V4 Flash thinking 消耗 completion 预算导致 JSON 截断 → 关闭 thinking + 最终输出轮不带 tools；
- 烟花 / 强光 fire 误检 → V2 模型优化 + 困难负样本 + Agent 复核；
- 历史数据跨集合泄漏 → 片段级重划分。
