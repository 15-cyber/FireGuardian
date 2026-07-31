# FireGuardian Module 6～Module 11 改进要点（源自改进说明文档）

> 开发每个模块时，对照此清单逐条检查，确保全部落地。

---

## 一、整体数据流

```
M5 Simulator ──→ DetectionResult ──→ M6 Aggregator ──→ FireEvent ──→ M7 Agent ──→ FireDecision
                                                                           │
                          ┌────────────────────────────────────────────────┤
                          ▼                   ▼                   ▼
                       M8 截图            M9 日志             M10 报告
                          ▼                   ▼                   ▼
                          └──────────────── M11 GUI ────────────────┘
```

模块间不得直接传递 Ultralytics 原始 Results，不得互相访问内部状态。

---

## 二、M6 Event Aggregator

- [ ] **FireEvent + EventStatus 状态机**
  - EventStatus: CANDIDATE → CONFIRMED → ACTIVE → ENDED → DISCARDED
  - FireEvent: event_id / status / start_timestamp / last_timestamp / end_timestamp
  - total_frames / positive_frames / consecutive_positive/negative_frames
  - max_fire_area_ratio / max_smoke_area_ratio
  - avg_fire_confidence / avg_smoke_confidence
  - growth_trend / source_type / source_name
  - representative_frame_id / representative_image_path

- [ ] **可配置参数**（写入 config.yaml）
  - min_positive_frames: 3
  - confirmation_window: 5
  - max_negative_frames: 3
  - fire_confidence_threshold: 0.50
  - smoke_confidence_threshold: 0.50
  - min_fire_area_ratio: 0.002
  - min_smoke_area_ratio: 0.003
  - growth_threshold: 0.01

- [ ] **火焰面积统计**
  - 同类框面积之和 / 图像面积，上限 1.0
  - 第一版直接求和，不做并集计算

- [ ] **增长趋势计算**
  - 首尾差值法: delta = latest_area - earliest_area
  - 趋势: increasing / stable / decreasing / unknown

- [ ] **事件回调**
  - on_event_confirmed(event) → 触发 M8 截图
  - on_event_updated(event) → 通知 GUI 更新
  - on_event_ended(event) → 触发 M10 报告
  - on_event_discarded(event) → 清理

- [ ] **Agent 调用控制**
  - metadata["decision_required"] = True
  - 触发条件: 事件首次确认 / 增长趋势变化 / 持续时间达到阈值
  - 不每帧调用 Agent

---

## 三、M7 Fire Decision Agent

- [ ] **FireDecision 数据类**
  - danger_level: low / medium / high
  - reason: 原因描述
  - suggestion: 建议措施
  - source: "rule" / "llm"
  - confidence: 决策置信度
  - decision_time: 决策时间
  - debug_info: dict（各因素贡献值）

- [ ] **规则引擎（第一版）**
  - 基于 fire_area / smoke_area / duration / growth_trend / confidence
  - 输出 danger_level + reason + suggestion
  - 确定性的规则，每次运行结果一致

- [ ] **LLM 增强（可选，后续扩展）**
  - 通过 source 字段区分 "rule" 和 "llm"
  - source="rule" 时有完整的规则逻辑
  - LLM 调用失败退回到规则结果

- [ ] **Debug 日志**
  - 记录每项因素的数值和权重贡献
  - 方便调参和排查

---

## 四、M8 自动截图（已完成，已按核查文档复核）

- [x] **回调驱动**
  - M6 on_event_confirmed / on_event_updated / on_event_ended → M8 保存截图
  - M7 on_decision → 危险等级升级截图（可选挂钩）
  - 不轮询、不主动检测；不重复推理（只复用检测框绘制）

- [x] **四个截图时机（数量受控）**
  - confirmed: 事件首次确认（每事件去重，只保存一组）
  - peak: 峰值指标配置化（peak_metric: fire_area/smoke_area/combined_area），更高峰值替换旧图，始终对应最终最高峰值
  - danger_level_upgraded: 按转换去重（upgrade_low_to_medium / upgrade_medium_to_high），互不覆盖
  - final: 默认使用最后阳性帧（use_last_positive_frame_for_final）

- [x] **保存内容**
  - 原图（_raw）+ 检测标注图（_annotated）+ 元数据（_meta.json，原子写入）
  - 事件汇总 event_info.json（原子写入，供 M10 报告使用）

- [x] **每事件独立目录**：screenshots/event_<event_id>/

- [x] **元数据字段**
  - event_id / frame_id / timestamp / reason / fire_area_ratio / smoke_area_ratio
  - 升级图附带 previous/current_danger_level、decision_score/confidence/source
  - write_status 记录每个文件写入成功/失败

- [x] **统一截图记录**
  - utils/common.py 新增 ScreenshotRecord 数据类（M8 → M10 契约）

- [x] **统一路径管理**：utils/path_manager.py（PathManager）

- [x] **失败隔离**：写入失败记录 write_status，不影响主检测流程

- [x] **状态隔离与清理**：按 event_id 隔离，事件结束后 cleanup_event_cache

- [x] **自测覆盖 11 项**：confirmed 去重 / 峰值替换 / 升级去重 / 多级升级 / 降级不截图 / 最后阳性帧 / 写入失败 / 原子写入 / 多事件隔离 / 不重复推理 / M6 联动

---

## 五、M9 日志模块（已完成，按改进说明重构）

- [x] **系统日志与事件日志分离**
  - logs/application.log：程序运行与模块状态
  - logs/error.log：异常堆栈（ERROR 及以上）
  - logs/events.jsonl：结构化事件日志（JSON Lines）

- [x] **log_event(event, event_type)** 结构化日志函数
  - 统一字段：time / level / module / event_type / event_id / frame_id / details
  - 记录事件全部字段（to_dict + metadata）
  - 提供 log_event_confirmed/updated/ended/discarded、log_decision

- [x] **日志轮转**：rotation / retention / compression 配置化（loguru）

- [x] **敏感信息脱敏**：sensitive_keys 递归掩码（api_key / token / secret / password 等）

- [x] **写入失败不影响主流程**（EventLogWriter try/except）

- [x] **程序关闭统计摘要**（print_summary，按 event_type 计数）

- [x] **桥接 M6**（attach_event_logger 链式挂接，不覆盖已有回调）

- [x] **Session ID**（run_YYYYMMDD_HHMMSS，支持环境变量覆盖，区分多次运行）

- [x] **source_module 字段**（如 M6/M7，日志可快速追溯来源）

- [x] **模块耗时统计**（log_elapsed / timed 上下文计时器，event_type=elapsed）

- [x] **Debug 开关**（config log.debug: false，true 时提升为 DEBUG 级详细日志）

- [x] **run_summary.json**（save_run_summary：session/时长/事件计数/危险等级分布/runtime_stats）

- [x] **时间统一 ISO8601**（毫秒精度，可直接被 Pandas/Elastic 解析）

---

## 六、M10 报告生成

- [ ] **事件驱动**
  - M7 on_analysis_complete → M10 生成报告

- [ ] **三种输出格式**
  - report.md - 人类可读
  - report.pdf - 正式文档
  - event.json - 机器可读

- [ ] **报告内容**
  - 事件编号 / 检测时间 / 检测来源
  - 检测类别 / 置信度 / 持续时间
  - 火焰面积 / 烟雾面积 / 危险等级
  - Agent 分析 / 应急建议 / 检测截图

---

## 七、M11 GUI

- [ ] **Worker 线程**
  - YOLO 推理在独立线程运行，不阻塞 UI
  - 通过 Signal 更新界面

- [ ] **左侧菜单**
  - 图片检测 / 视频检测 / Validation Simulator / 模型训练

- [ ] **M5 播放控制集成到 GUI**
  - interval / mode(sequential/random) / loop / max_frames
  - 提供 UI 控件，不要求用户改 YAML

- [ ] **右侧信息区分三区**
  - 当前帧检测: 类别 / 置信度 / 火焰面积 / 烟雾面积 / 推理耗时
  - 当前事件: 事件编号 / 状态 / 持续时间 / 增长趋势 / 危险等级
  - Agent 决策: 原因 / 建议 / 决策来源 / 更新时间

- [ ] **事件历史列表**
  - 表格: 事件编号 / 开始时间 / 持续时间 / 最高危险等级 / 报告状态 / 查看报告

- [ ] **日志限长**
  - max_log_lines: 1000 条
  - 完整日志仍写入文件

- [ ] **训练页面（第一版简化）**
  - 选择数据集 YAML / 选择权重 / epochs / imgsz / batch / device
  - 开始/停止按钮
  - 显示关键训练日志
  - 打开训练结果目录
  - 不做实时曲线

- [ ] **安全释放资源**
  - 关闭时: 停止工作线程 → 等待推理完成 → 释放视频/文件 → 退出
  - 防止输出视频损坏 / 报告写一半 / GPU 未释放

- [ ] **GUI 不含业务逻辑**
  - 只负责: 接收用户操作 → 调用控制器 → 显示结果
  - 业务逻辑在: detector / simulator / aggregator / agent / report

---

## 八、接口契约总表

| 方向 | 数据结构 | 说明 |
|------|----------|------|
| M5 → M6 | DetectionResult | 包含 FrameData + bboxes + inference_time_ms |
| M6 → M7 | FireEvent | 包含事件状态 + 统计 + 增长趋势 |
| M7 → M10/M11 | FireDecision | 包含危险等级 + 原因 + 建议 + 来源 |
| M8 → M10 | list[Path] | 截图路径列表 |
| M10 输出 | report.md / report.pdf / event.json | 三格式报告 |

---

## 九、当前阶段不做

- Kafka / RabbitMQ
- 分布式 EventBus
- 多摄像头并发
- WebSocket
- 企业微信真实告警
- FastAPI 服务化
- LangGraph 多 Agent
- 消防知识库 RAG
- 复杂目标跟踪算法
- 实时语义分割

这些写进 README Roadmap，第一版专注桌面端闭环。
