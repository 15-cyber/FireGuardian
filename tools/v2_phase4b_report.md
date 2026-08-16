# FireGuardian V2 Phase 4-B 报告：M11 GUI Agent 展示集成

> 日期：2026-08-15
> 分支：`v2-deepseek-agent`（HEAD 未变：`0eca3ad`）
> 状态：**Phase 4-B 完成，等待人工审核；未进入下一阶段；未提交 Git**

---

## 1. M11 修改内容

- 新增 `ui/agent_panel.py`：`AgentAnalysisPanel`（只读展示控件）；
- 修改 `ui/main_window.py`：Agent 决策页内嵌 Agent Panel；连接
  `agent_started / agent_tool_called / agent_completed / agent_failed` 四个信号；
  事件历史表由 5 列扩展为 8 列（新增 Rule / Agent / Final / Memory）；
  `closeEvent` 增加「正在停止Agent...」状态提示（沿用原 `controller.shutdown()`）；
- 修改 `ui/app_controller.py`：`_emit_history` 为每条历史事件附带其**自身**的
  Agent 摘要（rule/agent/final/memory，来自该事件 PresentationModel）；新增
  `get_agent_presentation(event_id)`；
- 未重构 V1 左侧/播放控制/中央画面/当前帧/当前事件/状态栏/训练页/日志页。

## 2. Agent Panel 设计

信息层级（§18）：

- 一级：`Hybrid Final`（大字加粗，沿用危险等级配色；Agent 非 ok 时后缀 `(M7)`）；
- 二级：`Rule` / `Agent (Assessment + Risk)` / `Memory` / `Reason`；
- 三级：`Explanation` / `Possible Cause` / `Recommended Action`；
- 四级（小字）：`Similar Events`（event_id + score，≤5 条）与 `Details`
  （Status / Tools / Memory / Latency / Tokens；**Model / Thinking 显示「未提供」**，
  不读取当前全局配置冒充历史信息）；
- 空字段走 `display_text` fallback；存在 fallback 时显示「Agent 信息不完整，仅供参考」。

## 3. Agent 状态信号

| 信号 | GUI 展示 |
| --- | --- |
| agent_started | Agent: Analyzing... + Trigger / Agent calls: n/4 |
| agent_tool_called | Agent: Analyzing... + Tool: 名称（不显示参数详情） |
| agent_completed | Completed；状态非 ok 时按 unavailable/invalid_output → Fallback，error 含 timeout → Timeout；刷新全部字段 |
| agent_failed | Fallback / Timeout + Reason；Final 使用 M7 |

## 4. 线程策略

- DeepSeek / Memory / Tool / Hybrid 全部在 `FireGuardianAgentWorker`（daemon）执行；
- GUI 线程只收 Qt Signal 更新 widgets，无任何 API/Memory 调用；
- Agent 运行中不影响视频播放/暂停/停止/历史/报告；事件结束早于 Agent 完成时
  报告与 GUI 显示已有数据（Report: Preliminary / Agent analysis pending 语义），不阻塞。

## 5. Trigger Guard 真实验证

- `AgentTriggerGuard`（Phase 4-A）在 GUI 链路中生效：confirmed / risk_upgrade /
  disagreement / ended 去重，risk_upgrade 按等级去重，单事件 ≤ 4 次；
- 记录 `logs/agent_trigger_calls.jsonl`（event_id + trigger + call_count）；
- offscreen 单测验证非法触发拒绝与计数上限；真实 GUI 冒烟中事件生命周期未出现每帧调用。

## 6. Timeout 验证

- `run_agent_with_timeout`（60s）在 GUI 链路中生效；超时 → unavailable(timeout) →
  GUI 显示 `Agent: Timeout`、`Fallback: M7`、`Final: M7 Level`；
- 单测（慢 Agent 0.3s 超时）与 GUI 状态映射均已覆盖。

## 7. M11 测试结果

- 新增 `tests/test_m11_agent_display.py` 13 项（offscreen）：
  Agent OK / unavailable / invalid_output / timeout / Memory Used / Not Used /
  Unavailable / Rule high+Agent medium / Rule high+Agent low / 空字段 fallback /
  旧事件无 Agent / 四信号流转 / 关闭时 Agent 运行中 / 历史显示；
- **全量回归 186/186 通过**（Phase 4-A 基线 173 + 新增 13），V1/V2/M6/M7 全部无回归。

## 8. Offscreen 测试结果

- `QT_QPA_PLATFORM=offscreen` 下完成 MainWindow 实例化、Agent Panel 渲染、
  信号更新、历史表更新、状态切换与关闭窗口，无异常；
- 说明：offscreen 截图仅作为自动化产物（文本内容以断言为准），
  **最终视觉验收以人工 GUI 验收清单为准**。

## 9. 真实 API 测试（tools/v2_phase4b_agent_gui_smoke.py）

| Case | Agent 状态 | Rule | Agent | Hybrid Final | Memory |
| --- | --- | --- | --- | --- | --- |
| case_01_real_fire | Completed | HIGH | UNCERTAIN (MEDIUM) | **HIGH** | Used (2) |
| case_02_smoke_only | Completed | MEDIUM | SUSPICIOUS (MEDIUM) | MEDIUM | Used (3) |
| case_04_fireworks | Completed | MEDIUM | SUSPICIOUS (MEDIUM) | MEDIUM | Used (3) |
| case_05_normal | Completed | LOW | SUSPICIOUS (MEDIUM) | MEDIUM | Used (1) |

- case_01 的 M7=HIGH 在 GUI 中保持 Final=HIGH（Rule/Agent/Final 清晰区分）；
- case_05 中 Agent 建议 medium 被 Hybrid 采纳为 medium（无 review 升级逻辑之外的安全问题）。

## 10. 烟花1 GUI 测试（Phase 3-B 结果注入）

- Rule: MEDIUM | Agent: SUSPICIOUS (MEDIUM) | Final: MEDIUM | Memory: Used (1)；
- 无错误升级；面板正确呈现历史证据摘要（agent_with_memory_01_e1.json）。

## 11. 烟花2 GUI 测试（重点案例）

- **Rule: HIGH | Agent: UNCERTAIN (MEDIUM) | Hybrid Final: HIGH | Memory: Used (4)**；
- GUI 明确区分 Rule / Agent / Final，Agent 判断未被展示为 Final；
- Final HIGH 标注来自 Hybrid（Agent 非 ok 语义之外，Reason 展示
  「M7 High 不可被 Agent 降低（Hybrid 安全规则）」）；
- 截图：`tools/v2_phase4b_screenshots/fireworks2_agent_panel.png` / `_window.png`。

## 12. M10/M11 一致性

- 两者消费同一 `AgentPresentationModel`（UiController `_agent_results` 存储）；
- GUI Final = PDF Final = Hybrid Final（同一数据源，无重复计算）；
- 历史事件详情遵循用户要求：优先使用该事件自身 PresentationModel，
  Model / Thinking 等无对应字段时显示「未提供」，不用当前全局配置冒充。

## 13. 已知问题

1. Agent 结果质量依赖模型输出（偶发 schema 失败/非法 JSON），已有 Fallback 兜底，
   GUI 与报告正确展示回退状态；
2. offscreen 渲染截图信息量有限（无字体合成），视觉验收需人工完成；
3. 事件结束早于 Agent 完成时，报告/GUI 先展示已有数据并标记 pending，
   本版本未实现完成后自动补刷报告（可作后续优化）；
4. 历史表 Memory 列未区分「Not Used」与「Unavailable」（GUI 面板已区分）。

## 14. 人工 GUI 验收清单（需用户人工执行，不声称自动完成）

- [ ] `python main.py` 正常启动 GUI
- [ ] 加载 fire 图片 / smoke 图片 / 正常图片并开始
- [ ] 播放 `me.mp4`、`测试视频/烟花1.mp4`、`测试视频/烟花2.mp4`
- [ ] 观察 Agent 状态：Analyzing → Tool → Completed / Fallback / Timeout
- [ ] 观察 Memory：Used (N) / Not Used / Unavailable
- [ ] 观察 Rule / Agent / Hybrid Final 三级清晰可区分
- [ ] 烟花2 事件：Rule HIGH、Agent UNCERTAIN/SUSPICIOUS、Final HIGH
- [ ] 事件结束生成 M10 PDF，内容与 GUI 一致
- [ ] 事件历史表显示 Rule / Agent / Final / Memory 摘要
- [ ] 测试暂停/继续/停止，Agent 分析期间 UI 保持响应
- [ ] 关闭程序：显示「正在停止Agent...」后安全退出

## 15. 是否建议进入后续阶段

验收项全部满足（Agent Panel 显示、Rule/Agent/Final 区分、Memory 三态、四信号、
Timeout/Fallback、空字段 fallback、历史增强、GUI 线程不阻塞、关闭安全、旧事件兼容、
M10/M11 一致、烟花1/2 GUI 测试、真实 API 案例、offscreen 测试、186/186 回归、
人工验收清单、本报告）。

**结论：Phase 4-B 完成，可进入人工 GUI 验收；后续阶段（M10/M11 展示已完成，
建议下一阶段为 V2 模型数据优化或最终联调）由人工决定，本阶段不自动进入。**
