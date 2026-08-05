# FireGuardian M6 / M7 修复报告

日期：2026-08-03

## 1. 修复背景

来源：`tools/baseline_v1_visual_audit.md`（Baseline V1 可视化综合核查报告）第八章与第十二章。

审计发现的两个必要问题：

1. **M6 面积比例算法**：`fire_area_ratio` / `smoke_area_ratio` 使用 bbox 面积简单求和，重叠框会被重复计数。
   审计实例：最大 fire 帧（`fire_Fire_CV7563.jpg`，300×168）两框求和 `0.97549`，矩形并集 `0.93663`，被放大约 3.9 个百分点。
2. **M7 时间单位混用**：链路/视频测试驱动把 `frame_id` 直接写入 `timestamp`，被当作秒使用，
   导致 `duration_warning=3.0`、`duration_alert=10.0` 等秒制阈值被按"帧"解释，事件时长被严重放大
   （审计结论：真实 20 FPS 下 3.0/10.0 帧仅等价 0.15s / 0.49s）。

本次修复**只改 M6、M7 相关代码与测试驱动**，不重新训练模型、不修改数据集、不调整 M7 危险等级阈值、不修改 M1–M5 检测逻辑。

## 2. 修改内容

### 2.1 M6 面积比例改为同类别框并集（union）

- 新增 `utils/common.py::union_area_of_boxes(boxes)`：扫描线算法（按 x 排序 + y 区间合并），
  对同类别轴对齐矩形框计算并集面积，重叠区域不重复计数；输入顺序无关，无效框自动忽略。
- `Detection.fire_area` / `smoke_area` 由"面积求和"改为"同类别框并集面积"；
  `fire_area_ratio` / `smoke_area_ratio` 保持 `min(并集面积 / 图像面积, 1.0)`，输出恒在 `[0, 1]`。
- 未改动 `Detection.bboxes` 原始框信息，截图与报告继续使用原始框；未改变 `FireEvent` 数据结构与回调接口。

### 2.2 M6 输出秒制事件时间

- `Detection` 新增可选字段 `fps`（来源帧率）。
- `FireEvent` 新增规范时间字段：`fps`、`start_time_seconds`、`last_time_seconds`、`end_time_seconds`、`duration_seconds`；
  `FireEvent.duration` 优先返回 `duration_seconds`，缺失时回退旧 timestamp 差值（旧日志/旧构造方式兼容）。
- `EventAggregator` 新增 `_to_seconds()` 统一时间基准：
  1) 已知 `fps` 且 `frame_id >= 0` → `seconds = frame_id / fps`；
  2) 否则使用 `timestamp`（视为秒）；
  3) 都没有 → 0。
- 事件创建/更新/结束时维护 `start_time_seconds` / `end_time_seconds` / `duration_seconds`；
  `to_dict()` 输出新增字段，旧字段 `start_timestamp` 等保留。

### 2.3 M7 决策逻辑统一使用秒

- 新增 `_event_duration_seconds()`：优先取 `event.duration_seconds`，缺失回退 `event.duration`。
- 所有时长判定（`duration_warning` / `duration_alert` / `overrides.min_duration_seconds`）统一走该方法，明确基于秒。
- `debug_info` 增加 `duration_seconds`，便于核查每次决策所用的秒制时长。

### 2.4 数据源与测试驱动补 fps

- `video_detect/video_detector.py`：`Detection` 附带视频真实 `fps`（`cv2.CAP_PROP_FPS`）。
- `simulator/validator_simulator.py`：`Detection` 附带 `fps = 1 / interval_seconds`。
- `tools/baseline_v1_me_video.py`：读取 me.mp4 真实 fps（cv2 实测 30.0）并传给 `Detection`。
- `tools/baseline_v1_chain_verify.py`：16 帧链路声明 `fps=1.0`（`interval_seconds=1.0`，1 帧 = 1 秒名义时间）。

## 3. 修改前后区别

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| fire/smoke 面积 | 同类框面积直接求和 | 同类框矩形并集（重叠不重复计数） |
| 面积比输出 | `min(求和/图像面积, 1.0)`，重叠框被放大 | `min(并集/图像面积, 1.0)`，恒在 `[0,1]` |
| 事件时间单位 | `timestamp` 直接当秒（链路/视频驱动曾写帧号） | 统一秒制：`frame_id / fps`（fps 已知时），否则回退 timestamp |
| M7 时长阈值 | 依赖 `event.duration`（可能被帧号污染） | 统一使用 `duration_seconds`（秒） |
| 事件输出 | 无秒制字段 | `fps / start_time_seconds / end_time_seconds / duration_seconds` |

典型变化：

- 面积：审计帧 6 由 `0.97549`（求和）变为并集 `0.93663`，不再重复计数。
- 时间：me.mp4 事件时长由"帧数"变为真实秒（如事件 1 起止帧 1→83，秒制时长 `2.733s`；修复前该数值会被当作 82 秒）。

## 4. 测试结果

### 4.1 单元测试（新增 `tests/test_m6_m7_fix.py`，12 项全部通过）

面积并集用例（任务要求 4 例 + 补充）：

| 用例 | 期望 | 结果 |
| --- | --- | --- |
| case1 单个 bbox | 面积 = 10000 | 通过 |
| case2 两个不重叠 bbox | 面积 = 两框之和 | 通过 |
| case3 两个完全重叠 bbox | 面积 = 单框面积（不翻倍） | 通过 |
| case4 大框包含小框 | 面积 = 大框面积（**不重复增加**） | 通过 |
| 部分重叠 | 面积 = 15000 | 通过 |
| 混合类别 | fire 只计 fire 框、smoke 只计 smoke 框 | 通过 |
| 面积比范围 | 所有 `fire/smoke_area_ratio` 均 ∈ `[0,1]` | 通过 |

时间单位用例：frame_id/fps 换算、timestamp 回退、`duration` 优先秒制、`to_dict` 新字段、M7 秒制阈值（2s→low、12s→medium）均通过。

### 4.2 模块自带回归

- `python aggregator/event_aggregator.py`：状态机流转正常，M6 测试完成 [OK]。
- `python agent/fire_decision_agent.py`：8 组业务用例 + 3 组边界 + 4 组非法输入，全部测试通过 [OK]。

### 4.3 M6–M10 链路重新验证（`tools/baseline_v1_chain_verify.py`，模型 `models/baseline_v1_best.pt`）

- 状态 ok；处理 16 帧；结束事件 **2**、确认 0、丢弃 0（与审计基线一致）。
- 决策 8 次；危险等级分布 `{medium: 3, high: 3, low: 2}`（与审计基线一致）。
- 截图正常：每事件 `screenshots/event_*/` 16 个文件（confirmed/peak/upgrade/final 原图+标注+元数据）。
- 报告正常：每事件 `reports/event_*/` 4 项（event.json / report.md / report.pdf / images）。
- `logs/events.jsonl` 增长 24 条；新事件带 `start_time_seconds / duration_seconds / fps`，旧行仍可正常解析。
- 事件时长已为秒制：`event.json` 中 `duration_seconds = 10.0 / 6.0`，`fps = 1.0`（链路名义 1 帧 = 1 秒）。

### 4.4 me.mp4 真实视频重新验证（`tools/baseline_v1_me_video.py` + 秒制核查）

- 总帧 182；fire 阳性帧 **7**、smoke 阳性帧 **110**、连续阳性片段 **7**（与审计基线完全一致）。
- M6 事件：ended=4、active=1、confirmed=0、discarded=0（与审计基线一致）。
- 视频源 fps=30.0（cv2 实测）；事件秒制时长：`2.733s / 0.167s / 1.000s / 0.200s`（active `0.167s`），为真实视频时间。

### 4.5 M11 GUI

未进行无头运行；`ui/app_controller.py` 相关调用（`evt.duration` / `evt.to_dict()`）为增量兼容，全部改动文件 `py_compile` 通过。

## 5. 是否需要重新训练模型

**不需要。**

- 本次仅修复 M6 面积统计与 M7 时间单位，不涉及模型权重、数据或超参。
- Baseline V1 审计结论为"有条件晋升（开发测试）"；本次修复属于业务链路工程修复，模型侧无需重训。

## 6. 是否可以执行 promote_model

**可以执行 dry-run（默认模式，不修改任何文件）**：

```powershell
python tools/promote_model.py --candidate models/baseline_v1_best.pt
```

说明：

- `promote_model.py` 默认仅做预检（校验 nc=2、names[0]=fire、names[1]=smoke、SHA-256 等），加 `--execute` 才会替换 `models/best.pt`。
- 依据用户约束，本次**未执行正式替换**；正式晋升需人工确认后再执行。
- 建议在 dry-run 通过后，将"正式晋升"作为独立步骤由用户确认。

## 修改文件清单

| 文件 | 修改原因 |
| --- | --- |
| `utils/common.py` | M6 面积改并集；新增秒制时间字段与 `union_area_of_boxes`；`duration` 优先秒制 |
| `aggregator/event_aggregator.py` | M6 时间统一：`frame_id/fps` 换算、维护秒制事件时间字段 |
| `agent/fire_decision_agent.py` | M7 所有时长阈值统一基于秒（`_event_duration_seconds`） |
| `video_detect/video_detector.py` | 检测结果附带视频真实 fps |
| `simulator/validator_simulator.py` | 模拟器结果附带 `fps = 1/interval_seconds` |
| `tools/baseline_v1_chain_verify.py` | 链路驱动声明 fps=1.0，避免帧号被当秒 |
| `tools/baseline_v1_me_video.py` | 读取 me.mp4 真实 fps 并传给检测结果 |
| `tests/test_m6_m7_fix.py` | 新增面积并集 4 用例 + 时间秒制回归测试（12 项） |

## 验证命令

```powershell
python -m unittest tests.test_m6_m7_fix -v
python aggregator/event_aggregator.py
python agent/fire_decision_agent.py
python tools/baseline_v1_chain_verify.py
python tools/baseline_v1_me_video.py
python tools/promote_model.py --candidate models/baseline_v1_best.pt   # dry-run（可选，待确认）
```

未提交 Git；未执行正式 promote_model。
