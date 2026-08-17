# FireGuardian M1-M11 系统实现审计文档

- 文档生成时间：2026-08-01
- 审计基准：当前工作区真实代码（仅阅读，未修改任何 M1-M11 文件）
- 代码版本：HEAD = `784e0b5`（2026-07-31 16:34，feat: M11 按核查文档完善）
- 运行环境：Python 3.11.9 / PyTorch 2.13.0+cu126 / CUDA 可用（NVIDIA GeForce RTX 4060 Laptop GPU）/ OpenCV 5.0.0 / ultralytics 8.4.108
- 审计方式：逐个读取 M1-M11 源码、`config/config.yaml`、`datasets/fire.yaml`、`main.py`、git 历史，并在本会话中运行各模块内置自测与离屏 GUI 自测后记录结论。

---

## ⚠ 特别记录：当前模型现状（审计结论的前置事实）

1. **`models/best.pt` 不存在**：`models/` 目录存在但为空；项目根目录仅有两个预训练权重 `yolo11n.pt`、`yolo26n.pt`，`weights/` 目录为空。
2. **当前实际加载的模型名称**：`yolo11n.pt`（M3/M4/M5 的模型查找顺序均为 `models/best.pt` → 项目根 `yolo11n.pt`，因 `best.pt` 不存在，实际加载 `yolo11n.pt`）。
3. **当前模型的 class names**：COCO 预训练 80 类（0=person，1=bicycle，2=car，…，79=toothbrush），**不包含 fire，不包含 smoke**（本会话已通过 `YOLO("yolo11n.pt").names` 实测确认）。
4. **结论**：当前所有"检测 → 聚合 → 决策 → 截图 → 报告"链路运行在 COCO 通用检测模型之上，**仅供链路测试，不能完成火灾检测**。`Detection.fire_boxes` / `smoke_boxes` 在真实图片/视频上恒为空，M6 的阳性帧判定、M7 的决策、M8 的截图、M10 的报告均无法产生有业务意义的火灾结果。
5. **历史数据提醒**：`reports/event_FE-20260731144839-b97610/event.json` 中的 `max_fire_area_ratio≈0.13`、`avg_fire_confidence=0.85`、`source_type=...\bus.jpg` 是开发期自测链路（用 `bus.jpg` 作为画面 + 合成 fire 检测框驱动 M6-M10）产生的数据，并非当前 COCO 模型的真实火焰输出；**该数字不能作为火焰检测成功的证据**。
6. **前置条件**：只有完成 M2 训练并产出 `models/best.pt`（含 fire/smoke 类别）后，M3-M11 的火灾业务语义才成立。

---

## 模块总览

| 模块 | 真实文件路径 | 一句话职责 | 工作线程 | 接入 UI |
| --- | --- | --- | --- | --- |
| M1 | `config/config.yaml`、`datasets/fire.yaml`、`utils/common.py`、`utils/path_manager.py`、`train/data_manager.py`、`main.py` | 项目骨架、配置、共享数据类、路径管理、数据管理 | 否 | 间接（被全部模块依赖） |
| M2 | `train/yolo_trainer.py` | YOLO 训练（产出 `models/best.pt`） | GUI 中在线程 | 是（训练 Tab） |
| M3 | `detect/image_detector.py` | 图片 YOLO 检测 | GUI 中在线程 | 是 |
| M4 | `video_detect/video_detector.py` | 视频逐帧 YOLO 检测 | GUI 中在线程 | 是 |
| M5 | `simulator/validator_simulator.py` | 验证集图片流模拟监控输入 | GUI 中在线程 | 是 |
| M6 | `aggregator/event_aggregator.py` | 逐帧 Detection 聚合为 FireEvent 状态机 | 否（工作线程内被调用） | 是 |
| M7 | `agent/fire_decision_agent.py` | FireEvent → FireDecision 规则决策 | 否（工作线程内被调用） | 是 |
| M8 | `utils/screenshot.py` | 事件生命周期自动截图 | 否（回调链） | 是 |
| M9 | `logs/logger.py` | 系统/事件日志与运行摘要 | 否（线程安全写） | 是（日志 Tab 读取） |
| M10 | `reports/report_generator.py` | 事件报告（json/md/pdf） | 否（事件结束回调链） | 是（查看/打开报告） |
| M11 | `ui/app_controller.py` + `ui/main_window.py` | GUI 控制器与主窗口（整合层） | 是（检测/训练工作线程） | —（本身即 UI） |

---

## M1 项目骨架、配置、共享数据与数据管理

1. **模块名称与真实文件路径**
   - `config/config.yaml`（全局配置）
   - `datasets/fire.yaml`（YOLO 训练数据集配置）
   - `utils/common.py`（枚举、数据类、配置加载、公共函数）
   - `utils/path_manager.py`（PathManager 统一输出路径）
   - `train/data_manager.py`（头注释标注为 "M1 - 数据管理模块"）
   - `main.py`（程序入口）
   - 附带项目目录骨架：`agent/ aggregator/ detect/ logs/ reports/ simulator/ train/ ui/ utils/ video_detect/`（各含 `__init__.py`）

2. **当前实际职责**
   - 提供全局配置加载（带缓存）、共享数据类与枚举、统一输出路径、数据集目录健康检查与统计、GUI 程序入口。

3. **已实现功能**
   - `load_config(reload=False)`：YAML 配置加载与缓存。
   - 枚举：`EventStatus`（candidate/confirmed/active/ended/discarded）、`GrowthTrend`（increasing/stable/decreasing/unknown）、`DangerLevel`（low/medium/high）。
   - 数据类：`BoundingBox`、`Detection`、`FrameData`、`SimulatorStats`、`FireEvent`、`FireDecision`、`EventReportData`、`ScreenshotRecord`、`AgentDecision`（字段详见 B 章）。
   - 公共函数：`project_root()`、`ensure_dir()`。
   - `PathManager`：`screenshots_dir()`、`screenshot_event_dir(event_id)`、`reports_dir()`、`report_event_dir(event_id)`、`logs_dir()`、`runs_dir()`、`sanitize_dir_name()`。
   - `DataManager`：`scan_split()`、`scan_dataset()`、`verify_dataset_health()`、`DatasetReport`、`SplitStats`；支持 images/labels 配对检查、缺失/孤立标签、类别分布采样统计。
   - `main.py`：`QApplication` + `MainWindow` 启动。

4. **输入对象及字段**
   - 配置输入：`config/config.yaml`（YAML 字典）。
   - `DataManager` 输入：数据集根目录（配置为 `DATASET_ROOT`）与 `datasets/fire.yaml` 的类别名（`0: fire, 1: smoke`）。

5. **输出对象及字段**
   - `load_config()` → `dict`（全部配置段）。
   - `PathManager` 方法 → `Path`。
   - `DataManager.scan_dataset()` → `DatasetReport`（`dataset_path` / `splits` / `total_images` / `total_labels` / `total_matched` / `total_missing` / `total_orphan` / `total_instances` / `class_names` / `scan_time` / `label_scan_mode`）。

6. **依赖的其他模块**
   - 无（是所有模块的公共基础）。

7. **使用的 config.yaml 配置项**
   - 全部顶层段：`project`、`model`、`dataset`、`train`、`detect`、`video_detect`、`simulator`、`aggregator`、`agent`、`screenshot`、`log`、`report`、`gui`。

8. **对外接口、回调或 Qt Signal**
   - 无回调/信号。对外为函数与类接口：`load_config`、`PathManager` 方法、`DataManager` 扫描函数。

9. **是否运行在工作线程**
   - 否。`DataManager` 目前仅命令行使用，GUI 未集成数据集扫描。

10. **异常与降级处理**
    - `load_config`：配置文件缺失抛 `FileNotFoundError`。
    - `PathManager`：相对路径自动以项目根为基准，绝对路径原样保留。
    - `DataManager`：目录缺失/空目录/无标签时输出统计报告而非崩溃。

11. **已有测试及测试结果**
    - 内置自测：`python utils/common.py`、`python utils/path_manager.py`（本会话运行通过）。
    - `tests/` 仅为占位（`tests/test_all.py` 打印占位信息），无 pytest 用例。

12. **当前已知限制**
    - `config.yaml` 中 `model.path: models/best.pt` 指向不存在的文件（实际加载靠 M3-M5 的候选回退逻辑）。
    - `dataset.path` / `simulator.dataset_path` 指向 F 盘，F 盘未挂载时无法扫描/模拟。
    - `DataManager` 未接入 GUI。
    - `logs/logger.py` 当前内容为 M9 重构后的版本（M1 时期为基础封装），M1/M9 共享同一文件。

13. **是否已接入 UiController 和主运行链路**
    - 是。`config/common/path_manager` 被 M2-M11 全链路使用；`DataManager` 未接入 GUI（仅独立命令行）。

14. **与最初项目计划的差异**
    - 数据管理在代码中归属于 M1（`data_manager.py` 头注释 "M1 - 数据管理模块"）。
    - 首个提交 `931c5c9`（项目初始化）同时包含 M1（骨架/配置/数据类）与 M2（trainer + data_manager）的初版，之后模块按 M3→M11 顺序独立演进。
    - 原计划的"目录结构"最终以扁平包结构（每模块一个顶层包）落地，未使用 `src/` 布局；数据类集中在 `utils/common.py` 而非 `src/domain/`（当前数据类约 300 行内，单文件管理可行，后续数据类膨胀时可再拆分）。

## M2 YOLO 训练模块

1. **模块名称与真实文件路径**
   - `train/yolo_trainer.py`（头注释 "M2 - YOLO 训练模块"）

2. **当前实际职责**
   - 封装 YOLO 训练全流程：加载预训练权重（默认 `yolo11n.pt`）在火灾数据集（`datasets/fire.yaml`）上微调，实时回调训练指标，训练完成后将最佳权重保存到 `models/best.pt`。

3. **已实现功能**
   - `YOLOTrainer`：`train()`（返回 best 模型路径）、`request_stop()`、`plot_metrics()`。
   - `TrainingStopped` 异常：协作式停止，在 epoch 回调边界安全退出，由 GUI 捕获。
   - 每 epoch 回调收集 loss / precision / recall / mAP50 / mAP50-95，并维护 `_metrics_history`。
   - 默认数据集 `datasets/fire.yaml`：`path=DATASET_ROOT`，`nc=2`，`names: {0: fire, 1: smoke}`。
   - 常量：`_MODEL_SOURCE = 根目录/yolo11n.pt`，`_MODEL_TARGET_DIR = models/`。

4. **输入对象及字段**
   - `weights`（默认 `yolo11n.pt`）、`dataset_yaml`（默认 `datasets/fire.yaml`）、`epochs` / `batch` / `imgsz` / `device` / `project` / `name`（默认来自 `config.yaml` 的 `train` 段）。

5. **输出对象及字段**
   - 训练产物：`project/name` 目录下的权重与指标文件；`models/best.pt`（目标位置）。
   - 每 epoch 回调 `callback(dict)`：epoch / loss / precision / recall / mAP50 / mAP50-95 / elapsed / eta 等。
   - `train()` 返回 best 模型路径（`Path`）。

6. **依赖的其他模块**
   - `utils/common.load_config`、`datasets/fire.yaml`、F 盘数据集、`ultralytics`。

7. **使用的 config.yaml 配置项**
   - `train.*`：`epochs=100`、`batch=16`、`imgsz=640`、`lr=0.01`、`patience=20`、`project=train`、`name=exp`、`exist_ok=true`、`pretrained=true`、`workers=4`。
   - `model.device=0`。

8. **对外接口、回调或 Qt Signal**
   - `train(...) -> Path`；`callback(dict)` 每 epoch/结束回调；`request_stop()`。
   - GUI 侧：`UiController.start_training(...)` / `stop_training()` / `training_running()` / `training_results_dir()`。

9. **是否运行在工作线程**
   - GUI 中由 `UiController.start_training` 在 `threading.Thread`（daemon，名 `FireGuardianTrain`）中运行；独立运行则在主线程。

10. **异常与降级处理**
    - `_validate_paths()` 校验模型/数据集路径，缺失时报错。
    - epoch 回调抛出 `TrainingStopped`，由 GUI 捕获并结束训练。
    - `plot_metrics` 在 matplotlib 缺失时跳过绘图。

11. **已有测试及测试结果**
    - 提供 `python train/yolo_trainer.py` 独立入口；本会话**未实际执行训练**（F 盘数据集未连接，且避免长时间占用）。
    - 当前无任何训练产物（`models/` 为空，`runs/detect/train` 无结果），即训练流程尚未在真实数据上跑通。

12. **当前已知限制**
    - 未在真实数据集上验证；无收敛数据，无法回答"半小时跑多少 epoch""全量数据耗时"等性能问题。
    - GUI 训练端到端（启动→epoch 回调→best.pt→被 M3-M5 重新加载）未验证。
    - `train` 段的 `project`/`name` 与配置默认值在 GUI 覆盖逻辑下行为待实际训练确认。

13. **是否已接入 UiController 和主运行链路**
    - 是。M11 训练 Tab → `start_training` → 线程 → `YOLOTrainer.train` → `train_epoch_ready` / `train_log_ready` / `train_finished` 信号。

14. **与最初项目计划的差异**
    - M11 阶段补充了训练参数化（weights/project/name/dataset_yaml 可由 GUI 覆盖）与 `TrainingStopped` 协作停止。
    - "训练后模型自动放入 `models/best.pt` 并被 M3-M5 使用"的闭环尚未完成（`best.pt` 不存在）。

## M3 图片检测模块

1. **模块名称与真实文件路径**
   - `detect/image_detector.py`（头注释 "M3 - 图片检测模块"）

2. **当前实际职责**
   - 单张/批量图片 YOLO 推理，将模型输出解析为统一 `Detection` 对象，可选保存标注图。

3. **已实现功能**
   - `ImageDetector`：`_find_model()`（按 `models/best.pt` → 项目根 `yolo11n.pt` 优先级查找）、`detect_single()`、`detect_batch()`、`_save_annotated()`、`print_stats()`。
   - 支持 `jpg/jpeg/png/bmp`。
   - 统计：`total_detections`、`total_inference_time`。

4. **输入对象及字段**
   - `image_path`（`str|Path`，单张或目录）；`conf_threshold` / `iou_threshold`（可覆盖，默认取 `config.yaml`）。
   - 模型：`YOLO(str(model_path))`。

5. **输出对象及字段**
   - `Detection`：`bboxes`（`List[BoundingBox]`，字段 x1/y1/x2/y2/confidence/class_id/class_name）、`image_path`、`timestamp`、`wall_time`、`inference_time_ms`、`image_width`、`image_height`、`frame_id=-1`（单图未提供帧号）。
   - 标注图（`save_result=True` 时）输出到 `detect.output_dir`（默认 `runs/detect`）。

6. **依赖的其他模块**
   - `utils/common`（`load_config`、`BoundingBox`、`Detection`、`ensure_dir`）、`ultralytics`。

7. **使用的 config.yaml 配置项**
   - `model.path`、`model.confidence=0.5`、`model.iou=0.45`、`model.device=0`、`detect.output_dir=runs/detect`、`detect.image_extensions`。

8. **对外接口、回调或 Qt Signal**
   - `detect_single(image_path, save_result=True, output_dir=None) -> Detection`；`detect_batch(...)`；命令行入口（`python detect/image_detector.py [图片路径]`）。无回调/信号。

9. **是否运行在工作线程**
   - GUI 中由 `UiController._run_image`（`FireGuardianWorker` 线程）调用；独立运行在主线程。

10. **异常与降级处理**
    - 图片不存在 → `FileNotFoundError`；模型不存在 → `FileNotFoundError`（提示放置位置）。
    - 读取失败由上层（UiController）捕获并记日志，不中断。

11. **已有测试及测试结果**
    - 本会话用 `bus.jpg` 运行 `detect_single` 实测：输出 `bus/person` 等 COCO 类别——**证明当前加载的是 COCO `yolo11n.pt`（80 类，无 fire/smoke）**。

12. **当前已知限制**
    - 当前模型无火灾类别，`detection.fire_boxes/smoke_boxes` 恒为空，业务输出无意义。
    - `detect_batch` 未在 GUI 中使用。

13. **是否已接入 UiController 和主运行链路**
    - 是。`UiController._ensure_image_detector()` 惰性创建（首次图片源才加载 YOLO），`_run_image` 调用。

14. **与最初项目计划的差异**
    - 计划中的"模型自动查找/回退"落地为 `_MODEL_CANDIDATES = [models/best.pt, yolo11n.pt]` 硬编码顺序；与计划一致度较高，主要差异是模型未训练导致业务输出为空。

## M4 视频检测模块

1. **模块名称与真实文件路径**
   - `video_detect/video_detector.py`（头注释 "M4 - 视频检测模块"）

2. **当前实际职责**
   - 视频逐帧 YOLO 检测。两种模式：`detect()`（写标注视频并输出统计）与 `decode()`（生成器逐帧产出，不写文件，供 GUI 实时显示）。

3. **已实现功能**
   - `VideoDetector`：`detect()`、`decode()`、`detect_fire_optimized()`、`_get_color()`、`_find_model()`（与 M3 相同优先级）。
   - `VideoDetectionResult.print_summary()`。
   - 跳帧 `frame_skip`；`decode()` 的 `try/finally` 保证 `cap.release()`。
   - 支持 `mp4/avi/mov`。

4. **输入对象及字段**
   - `video_path`（`str|Path`）；`conf_threshold` / `iou_threshold` / `frame_skip`。

5. **输出对象及字段**
   - `decode()` 逐帧产出元组 `(frame[BGR ndarray], detection[Detection], frame_idx, total_frames)`；`Detection.frame_id` 为 1-based 帧号。
   - `detect()` 产出 `VideoDetectionResult`（`video_path/total_frames/output_path` 等）并写 `<stem>_detect.mp4` 到 `runs/video_detect`。

6. **依赖的其他模块**
   - `utils/common`、`ultralytics`、`cv2`。

7. **使用的 config.yaml 配置项**
   - `model.confidence`、`model.iou`、`model.device`、`video_detect.extensions`、`video_detect.output_dir=runs/video_detect`、`video_detect.show_fps=true`。

8. **对外接口、回调或 Qt Signal**
   - `detect(video_path) -> VideoDetectionResult`；`decode(video_path) -> generator`；命令行入口。无回调/信号。

9. **是否运行在工作线程**
   - GUI 中由 `UiController._run_video` 在工作线程内逐帧消费 `decode()`；独立运行在主线程。

10. **异常与降级处理**
    - 文件不存在 → `FileNotFoundError`；格式不支持 → `ValueError`；无法打开 → `RuntimeError`；`decode` 使用 `finally` 释放视频捕获。

11. **已有测试及测试结果**
    - 本会话用 `me.mp4` 实测 `decode()`：182 帧正常遍历，无异常；GUI 视频源可正常播放检测。

12. **当前已知限制**
    - 当前模型为 COCO，无法检出火焰/烟雾。
    - `detect_fire_optimized()` 名义上"更低阈值关注 fire/smoke"，实现仅直接调用 `detect()`，是占位优化（已实现但未生效）。
    - GUI 只用 `decode()`，`detect()` 的写文件模式未在 GUI 使用；无帧率目标控制（GUI 按推理速度播放）。

13. **是否已接入 UiController 和主运行链路**
    - 是。`UiController._ensure_video_detector()` 惰性创建，`_run_video` 消费 `decode()`。

14. **与最初项目计划的差异**
    - GUI 采用 `decode` 生成器（不写文件），命令行仍保留 `detect` 写视频模式；`detect_fire_optimized` 为计划中"火灾优化模式"的未完成占位。

## M5 验证集监控流模拟器

1. **模块名称与真实文件路径**
   - `simulator/validator_simulator.py`（头注释 "M5 - Validation Stream Simulator"）

2. **当前实际职责**
   - 将验证集图片（`valid/images`）按间隔封装为连续帧流，模拟监控摄像头输入；输出统一 `FrameData + Detection`，供 M6 与 GUI 消费。

3. **已实现功能**
   - `ValidationStreamSimulator`：状态机 `IDLE/RUNNING/PAUSED/STOPPED/COMPLETED`；`start()`（阻塞式）/ `pause()` / `resume()` / `stop()` / `next_frame()`（步进式，GUI 用）/ `reset()` / `get_statistics()`。
   - 播放模式 `sequential/random`；`loop` / `max_frames` 控制；`interval_seconds` 节奏（补偿推理耗时）。
   - 损坏图片自动跳过（`_cv_read_image` 用 `np.fromfile + cv2.imdecode` 兼容中文路径），计入 `failed_frames`。
   - 回调：`on_frame` / `on_progress` / `on_error` / `on_complete`。
   - CLI 入口 `_create_default_detector()`（模型顺序 `models/best.pt` → `yolo11n.pt`；找不到则返回空检测器 lambda，不崩溃）。

4. **输入对象及字段**
   - `image_dir`（默认 `config.yaml` 的 `simulator.dataset_path`，缺失时尝试 `dataset.path/valid/images`）。
   - `detector`：依赖注入的 `callable(image) -> List[BoundingBox]`（默认 CLI 场景使用 `_create_default_detector`；GUI 场景由 UiController 注入）。

5. **输出对象及字段**
   - 每帧 `(FrameData, Detection)`：`FrameData(frame_id, timestamp, image, image_path, source_type, source_name)`；`Detection` 同 M3/M4。
   - 运行结束输出 `SimulatorStats`（`total_frames/processed_frames/failed_frames/fire_frames/smoke_frames/normal_frames/avg_inference_time_ms/total_simulated_duration`）。

6. **依赖的其他模块**
   - `utils/common`、M3 检测器（依赖注入）、`cv2` / `numpy`。

7. **使用的 config.yaml 配置项**
   - `simulator.dataset_path=DATASET_ROOT/valid/images`、`interval_seconds=0.5`、`mode=sequential`、`loop=false`、`max_frames=null`、`supported_extensions`、`output_dir=runs/simulator`；`dataset.path`（回退）。

8. **对外接口、回调或 Qt Signal**
   - 方法：`start/pause/resume/stop/next_frame/reset/get_statistics`；回调：`on_frame(frame_data, detection)`、`on_progress(current, total, percent)`、`on_error(frame_id, image_path, error)`、`on_complete(stats)`。

9. **是否运行在工作线程**
   - `start()` 为阻塞式（代码注释注明"在独立线程中调用"）；GUI 中由 `UiController._run_simulator` 在工作线程内以 `next_frame()` 步进驱动，不使用 `start()`。

10. **异常与降级处理**
    - 图片目录为空：打印 WARN 并返回，不崩溃。
    - 单帧处理异常：计入 `failed_frames`，触发 `on_error`，不中断流程。
    - 中文路径图片读取失败：跳过该帧。

11. **已有测试及测试结果**
    - 本会话用中文路径图片目录验证 `next_frame()` 正常；M6 自测中复用其数据结构通过。

12. **当前已知限制**
    - 依赖 F 盘验证集存在；F 盘未连接时无数据可播。
    - `random` 模式仅打乱顺序，不重复采样。
    - 仅图片流模拟，无真实摄像头/RTSP 输入。

13. **是否已接入 UiController 和主运行链路**
    - 是。`UiController._run_simulator` 创建模拟器并逐帧驱动。

14. **与最初项目计划的差异**
    - 已按《FireGuardian Module 5 改进说明.docx》实现（状态机、依赖注入复用 YOLO Detector、统一 FrameData+Detection、损坏帧跳过、中文路径）。
    - CLI（`start` 阻塞）与 GUI（`next_frame` 步进）双驱动，与计划一致。

## M6 事件聚合器

1. **模块名称与真实文件路径**
   - `aggregator/event_aggregator.py`（头注释 "M6 - Event Aggregator"）

2. **当前实际职责**
   - 将逐帧 `Detection` 按阈值规则聚合成火灾事件 `FireEvent`；维护状态机 `CANDIDATE → CONFIRMED → ACTIVE → ENDED / DISCARDED`；计算增长趋势；通过 `metadata["decision_required"]` 控制 M7 调用频率。

3. **已实现功能**
   - `EventAggregator`：`process()`、`_is_positive_frame()`、`_create_candidate()`、`_update_event()`、`_check_confirmation()`、`_end_event()`、`_discard_event()`、`_calc_growth_trend()`、`_check_decision_required()`、`get_stats()`、`reset()`。
   - 回调：`on_event_confirmed` / `on_event_updated` / `on_event_ended` / `on_event_discarded`。
   - 事件 ID：`FE-YYYYMMDDHHMMSS-<6位hex>`。

4. **输入对象及字段**
   - `Detection`：关键字段 `bboxes`（含 `class_name`、`confidence`）、`timestamp`、`image_width`、`image_height`、`image_path`、`frame_id`。
   - 阳性帧判定依赖 `Detection.has_fire/has_smoke`、`avg_fire_confidence/avg_smoke_confidence`、`fire_area_ratio/smoke_area_ratio`。

5. **输出对象及字段**
   - `FireEvent`（`event_id/status/start_timestamp/last_timestamp/end_timestamp/total_frames/positive_frames/consecutive_positive_frames/consecutive_negative_frames/max_fire_area_ratio/max_smoke_area_ratio/avg_fire_confidence/avg_smoke_confidence/growth_trend/source_type/source_name/representative_frame_id/representative_image_path/metadata/detections`）。
   - `process()` 返回当前活跃事件（或 None）；`get_stats()` 返回 `total_frames_processed/current_event/current_event_status/total_events_confirmed/total_events_ended/total_events_discarded`。

6. **依赖的其他模块**
   - `utils/common`（`Detection`、`FireEvent`、`EventStatus`、`GrowthTrend`）。

7. **使用的 config.yaml 配置项**
   - `aggregator.min_positive_frames=3`、`confirmation_window=5`、`max_negative_frames=3`、`fire_confidence_threshold=0.50`、`smoke_confidence_threshold=0.50`、`min_fire_area_ratio=0.002`、`min_smoke_area_ratio=0.003`、`growth_threshold=0.01`。

8. **对外接口、回调或 Qt Signal**
   - `process(detection) -> Optional[FireEvent]`；属性 `current_event` / `event_history` / `total_processed`；`get_stats()`、`reset()`；4 个回调属性。

9. **是否运行在工作线程**
   - 否。由 UI 工作线程逐帧调用（`UiController._process_frame`），回调也执行在调用线程（工作线程）中，因此 M7/M8/M9/M10 实际运行于工作线程上下文。

10. **异常与降级处理**
    - 聚合器本身无内置 try；上层 UiController 对 `aggregator.process` 异常打日志不中断（`_process_frame` 内 try/except）。

11. **已有测试及测试结果**
    - 内置自测（`python aggregator/event_aggregator.py`）：10 帧火焰 → 3 帧空白 → 5 帧火焰 → 结束，状态机流转与统计正确；本会话运行通过。

12. **当前已知限制**
    - 同一时刻只跟踪一个当前事件（单一事件模型），不支持并发多事件。
    - 不区分多目标；区域比值按整帧面积计算。
    - 依赖 `Detection.timestamp` 单调递增（模拟器/视频场景满足；图片单帧场景只有 1 帧，无法确认事件）。

13. **是否已接入 UiController 和主运行链路**
    - 是。`UiController.__init__` 创建实例并在 `_wire_event_callbacks` 手工接线。

14. **与最初项目计划的差异**
    - 与计划状态机一致；`metadata["decision_required"]` 节流机制与"趋势变化/持续时长触发再决策"为后续改进说明补充。

## M7 火灾决策智能体

1. **模块名称与真实文件路径**
   - `agent/fire_decision_agent.py`（头注释 "M7 - Fire Decision Agent v2"）

2. **当前实际职责**
   - 接收 `FireEvent`，输出结构化 `FireDecision`：归一化加权评分 + 关键规则覆盖（override）混合决策；独立计算决策可信度；生成结构化 reasons / suggestions / summary；输入校验。

3. **已实现功能**
   - `FireDecisionAgent`：`analyze()`、`_validate_event()`、`_extract_features()`、`_normalize_area()`、`_normalize_duration()`、`_normalize_growth()`、`_calculate_score()`、`_map_score_to_level()`、`_apply_override_rules()`、`_calculate_confidence()`、`_build_reasons()`、`_build_suggestions()`、`_build_summary()`、`_try_llm_enhance()`、`_log_decision()`。
   - 规则覆盖（`config.agent.overrides`）：
     - 规则 1：火焰面积 ≥ `min_fire_area_ratio`（0.15）且持续 ≥ `min_duration_seconds`（10）且趋势扩大 → 强制 HIGH。
     - 规则 2：持续 ≥ `duration_alert`（10s）且基础等级 LOW → 提升为 MEDIUM。
     - 规则 3：浓烟无明火 + 长时间 → 至少 MEDIUM（`smoke_only_medium`）。
   - LLM 增强（可选，默认关闭）：`_try_llm_enhance` 当前直接 `return None`（TODO 占位），失败不影响规则结果。

4. **输入对象及字段**
   - `FireEvent`：`event_id`、`max_fire_area_ratio`、`max_smoke_area_ratio`、`duration`（属性）、`growth_trend`、`total_frames`、`positive_frames` 等。
   - 输入校验（`ValueError`）：缺少 `event_id`；面积 < 0 或 > 1；时长为负；帧数为负。

5. **输出对象及字段**
   - `FireDecision`：`event_id`、`danger_level`（low/medium/high）、`score`（0~1）、`confidence`（独立可信度）、`reasons`、`suggestions`、`summary`、`decision_source="rule_engine"`、`generated_at`、`debug_info`（`base_level` / `override_applied` / `features` / `weights`）。

6. **依赖的其他模块**
   - `utils/common`（`DangerLevel`、`EventStatus`、`FireEvent`、`FireDecision`、`GrowthTrend`）。

7. **使用的 config.yaml 配置项**
   - `agent.weights`：`fire_area=0.4`、`smoke_area=0.3`、`duration=0.2`、`growth=0.1`。
   - `agent.thresholds`：`low_max_score=0.3`、`medium_max_score=0.6`、`fire_area_warning=0.05`、`smoke_area_warning=0.08`、`duration_warning=3.0`、`duration_alert=10.0`。
   - `agent.overrides.high`、`agent.overrides.smoke_only_medium`。
   - `agent.llm`：`enabled=false`、`api_key=''`、`model='gpt-4o-mini'`、`prompt_template`。

8. **对外接口、回调或 Qt Signal**
   - `analyze(event) -> FireDecision`；属性 `total_analyses`。无 Qt 信号（结果由 UiController 通过 `decision_ready` 信号转发）。

9. **是否运行在工作线程**
   - 否。由 UI 工作线程回调链调用（`UiController._analyze_event`）。

10. **异常与降级处理**
    - 非法输入明确抛 `ValueError`（上层捕获并记日志，不中断流程）。
    - LLM 增强整体 try/except，失败/未实现均回退规则结果。
    - override 规则不满足时保持基础等级。

11. **已有测试及测试结果**
    - 内置 `main()`：8 组业务用例 + 3 组阈值边界用例 + 4 组非法输入用例 + 配置读取校验；本会话运行**全部通过**。

12. **当前已知限制**
    - LLM 增强未实现（占位 `return None`）。
    - 决策完全依赖输入 `FireEvent` 质量；当前 COCO 模型下无真实 fire/smoke 事件，决策逻辑仅被合成数据验证。
    - `confidence` 为启发式计算，未经真实数据校准。

13. **是否已接入 UiController 和主运行链路**
    - 是。`UiController._analyze_event` 调用，结果经 `decision_ready` 信号发 UI，并联动 M8（`on_decision`）与 M9（`log_decision`）。

14. **与最初项目计划的差异**
    - v2 按《FireGuardian M7 Fire Decision Agent 核查与必要修改说明.docx》修正（输入校验、独立可信度、结构化原因/建议、debug_info）。
    - LLM 能力保留但默认关闭，属于"预留未落地"能力。

## M8 自动截图模块

1. **模块名称与真实文件路径**
   - `utils/screenshot.py`（头注释 "M8 - 自动截图模块"）

2. **当前实际职责**
   - 在事件生命周期内自动截图：四个时机 `confirmed / peak / danger_level_upgraded / final`；每个事件独立目录；保存原图 + 标注图 + 元数据 JSON（原子写入）+ `event_info.json`；输出结构化 `ScreenshotRecord`。

3. **已实现功能**
   - `ScreenshotManager`：`attach()`（覆盖 M6 三个回调）、`on_event_confirmed()`、`on_event_updated()`、`on_decision()`（记录危险等级、检查升级截图）、`on_event_ended()`。
   - peak 替换：`peak_metric` 配置化（`fire_area/smoke_area/combined_area`），更高峰值替换旧截图。
   - danger upgrade 去重：按 `low→medium→high` 等级排序，同一升级只截一次。
   - final 截图默认使用最后阳性帧（`use_last_positive_frame_for_final=true`）。
   - `_atomic_write_json`、`_save_event_info`；查询接口 `get_saved_paths/get_latest/get_event_shot_records/get_stats`。
   - `_cv_read_image` / `_cv_write_image` 兼容中文路径。

4. **输入对象及字段**
   - `FireEvent`（来自 M6 回调，含 `detections` 列表与 `metadata.danger_level`）；`FireDecision`（来自 M7 的 `on_decision`）。

5. **输出对象及字段**
   - 文件：`screenshots/event_<event_id>/` 下 `*_raw.*`（原图）、`*_annotated.*`（标注图）、`*_meta.json`、`event_info.json`。
   - `ScreenshotRecord`：`event_id/reason/frame_id/simulated_timestamp/raw_image_path/annotated_image_path/metadata_path/fire_area_ratio/smoke_area_ratio/danger_level/decision_score/decision_confidence/decision_source/saved_at/write_status`。

6. **依赖的其他模块**
   - `utils/common`（`FireEvent`、`ScreenshotRecord` 等）、`utils/path_manager`（`PathManager`）、`cv2`。

7. **使用的 config.yaml 配置项**
   - `screenshot.enabled=true`、`output_dir=screenshots`、`per_event_dir=true`、`peak_metric=combined_area`、`save_original=true`、`save_annotated=true`、`save_metadata=true`、`write_event_info=true`、`use_last_positive_frame_for_final=true`。
   - `screenshot.triggers`：`confirmed=true`、`peak=true`、`danger_upgrade=true`、`final=true`。

8. **对外接口、回调或 Qt Signal**
   - `attach(aggregator)`：直接覆盖 M6 的 `on_event_confirmed/on_event_updated/on_event_ended`（注意：UiController 未调用 `attach`，而是手工接线，时序由 `_on_event_*` 控制）。
   - `on_decision(event, decision, danger_level)`：由 M7 结果驱动。
   - 查询接口：`get_event_shot_records(event_id)`、`get_saved_paths()`、`get_latest()`、`get_stats()`。

9. **是否运行在工作线程**
   - 否。回调链在 UI 工作线程内同步执行（文件写入为同步 I/O）。

10. **异常与降级处理**
    - 单次截图异常由 UiController 各回调 try/except 包裹（仅告警不中断）。
    - `_atomic_write_json` 失败不崩溃；未知 `peak_metric` 回退 `combined_area`。

11. **已有测试及测试结果**
    - 内置 `main()` 覆盖 11 项（四时机/peak 替换/升级去重/ScreenshotRecord 字段等）；本会话运行通过。
    - 已生成示例截图（`screenshots/event_*` 目录，含 raw/annotated/meta）。

12. **当前已知限制**
    - 截图标注内容依赖检测框；当前 COCO 模型在火灾场景无框，截图只有原图、无有效火灾标注内容。
    - final 截图依赖最后阳性帧存在（纯阴性事件无 final 内容）。

13. **是否已接入 UiController 和主运行链路**
    - 是。`UiController.__init__` 创建 `ScreenshotManager`，`_wire_event_callbacks` 中在 M6 回调内依次调用 M8/M7/M9 保证时序（确认/更新 → 截图 → 决策；结束 → 截图 → 报告）。

14. **与最初项目计划的差异**
    - M8 经历"完成 → 按改进说明重构 → 按核查文档复核"三阶段（git：`ddaefee` → `998adc3` → `5a84c5f`）。
    - `peak_metric` 配置化与 `ScreenshotRecord` 结构化记录、原子写入为核查文档补充项。

## M9 日志模块

1. **模块名称与真实文件路径**
   - `logs/logger.py`（头注释 "M9 - 日志模块，按核查文档完善"）

2. **当前实际职责**
   - 系统日志与事件日志分离：`logs/application.log`（程序运行与模块状态）、`logs/error.log`（异常堆栈）、`logs/events.jsonl`（结构化火灾事件记录，JSON Lines）、`logs/run_summary.json`（退出摘要）。含敏感信息脱敏、日志轮转、`session_id`、模块耗时统计。

3. **已实现功能**
   - `get_logger(name)`（loguru 绑定 name）。
   - `mask_sensitive(data, key)`：递归脱敏（api_key/token 等字段 → `***`）。
   - `EventLogWriter`：线程安全（`threading.Lock`）JSONL 写入，写失败返回 False 不影响主流程。
   - `log_event()` 及封装 `log_event_confirmed/log_event_updated/log_event_ended/log_event_discarded`。
   - `log_decision()`（event_type=`decision_made`）；`log_elapsed()` / `timed()` 模块耗时；`add_stat()` / `get_event_stats()` / `get_danger_stats()`。
   - `attach_event_logger(aggregator)`：链式挂接 M6 回调（不覆盖已有回调）。
   - `flush()`、`save_run_summary()`、`print_summary()`。
   - `SESSION_ID`：`run_YYYYMMDD_HHMMSS`，可用环境变量 `FIREGUARDIAN_SESSION_ID` 覆盖。

4. **输入对象及字段**
   - `FireEvent` / `FireDecision` + 模块名；事件日志统一字段：`time/session_id/level/module/source_module/event_type/event_id/frame_id/details`。

5. **输出对象及字段**
   - `logs/application.log`、`logs/error.log`（loguru 轮转：10MB / 30 天 / zip）。
   - `logs/events.jsonl`（结构化事件记录）。
   - `logs/run_summary.json`（`session_id/started_at/ended_at/duration_seconds/debug/total_events/event_types/danger_levels/runtime_stats`）。

6. **依赖的其他模块**
   - `loguru`、`utils/common.load_config`。

7. **使用的 config.yaml 配置项**
   - `log.level=INFO`、`debug=false`、`application_file=logs/application.log`、`error_file=logs/error.log`、`events_file=logs/events.jsonl`、`rotation="10 MB"`、`retention="30 days"`、`compression=zip`、`format`、`colorize=true`、`sensitive_keys`。

8. **对外接口、回调或 Qt Signal**
   - 函数级接口（见第 3 条）；`attach_event_logger(aggregator)` 用于 M6 桥接；GUI 通过 `UiController.log_file()` / `read_log_tail()` / `log_line_count()` 读取。

9. **是否运行在工作线程**
   - 否。`EventLogWriter` 用锁保证多线程写安全；由 UI 工作线程调用。

10. **异常与降级处理**
    - 事件日志写入失败返回 False 不影响主流程（写失败不崩溃）。
    - 敏感字段自动脱敏，防止 API Key 落盘。

11. **已有测试及测试结果**
    - 内置 `main()`：事件统一字段/session_id/脱敏/模块耗时/写失败降级/M6 桥接/run_summary；本会话运行通过。
    - 现存日志产物：`logs/application.log`（23KB）、`error.log`（2.8KB）、`events.jsonl`（109KB，含开发期测试记录）、`run_summary.json`。

12. **当前已知限制**
    - `events.jsonl` 未按日分片（轮转配置仅作用于 application/error）。
    - 逐帧 DEBUG 日志需 `log.debug=true` 开启（默认关闭）。
    - 历史 `events.jsonl` 混有开发期自测记录，正式使用前建议清空。

13. **是否已接入 UiController 和主运行链路**
    - 是。UiController 在各事件回调调用 `log_event` / `log_decision`；M11 日志 Tab 实时读取 `application.log` 尾部。

14. **与最初项目计划的差异**
    - M9 经历"按改进说明重构（`e0f5d49`）→ 按核查文档完善（`805fa51`）"两轮；新增 `session_id`、`source_module`、`run_summary.json`、模块耗时等核查项。

## M10 报告生成模块

1. **模块名称与真实文件路径**
   - `reports/report_generator.py`（头注释 "M10 - 火灾事件分析报告生成模块"）

2. **当前实际职责**
   - 事件结束生成报告：`event.json`（机器可读，原子写入）+ `report.md` + `report.pdf`（需中文字体，缺字体时降级跳过）；复制截图到 `images/`、生成趋势图、构建时间线、配置快照与版本信息、完整性检查。

3. **已实现功能**
   - `ReportGenerator`：`generate()`、`pdf_available`、`_check_completeness()`、`_build_summary()`、`_config_snapshot()`、`_version_info()`（含 `_git_commit`）、`_collect_images()`、`_make_trend_chart()`、`_build_timeline()`、`_write_event_json()`、`_write_markdown()`、`_write_pdf()`、`get_generated_paths()`、`get_stats()`。
   - `attach_event_report(aggregator, generator, screenshot_provider, decision_provider)`：M6 `on_event_ended` 链式挂接，不覆盖已有回调。
   - 输出目录：`reports/event_<event_id>/`。

4. **输入对象及字段**
   - `EventReportData`：`event`（FireEvent）、`decisions`（List[FireDecision]）、`screenshots`（List[ScreenshotRecord]）、`system_info`（dict）、`model_info`（dict）。

5. **输出对象及字段**
   - `reports/event_<id>/event.json`、`report.md`、`report.pdf`（可选）、`images/`（截图副本 + `trend.png`）。
   - `generate()` 返回输出文件路径列表；`_check_completeness()` 返回警告列表（缺数据只警告不崩溃）。

6. **依赖的其他模块**
   - `utils/common`（`EventReportData`、`load_config`）、`utils/path_manager`（`PathManager`）、`reportlab`（PDF）、`matplotlib`（趋势图）。

7. **使用的 config.yaml 配置项**
   - `report.output_dir=reports`、`formats=[md, pdf]`、`pdf_font=""`（自动探测中文字体）、`copy_images=true`、`disclaimer`（报告免责声明）。

8. **对外接口、回调或 Qt Signal**
   - `generate(data) -> List[Path]`；`pdf_available` 属性；`attach_event_report` 工具函数；`get_generated_paths/get_stats`。

9. **是否运行在工作线程**
   - 否。在事件结束回调链（UI 工作线程）内同步生成；GUI 中"查看报告"由主线程打开。

10. **异常与降级处理**
    - `_check_completeness`：缺 Decision/Screenshot/Timeline 等只警告，不崩溃。
    - PDF 缺中文字体：打印提示并仅输出 md + json（不崩溃）。
    - `_atomic_write_json`：原子写，失败不产生半截文件。

11. **已有测试及测试结果**
    - 内置 `main()` 本会话运行通过。
    - 已生成示例报告（如 `reports/event_FE-RPT-0001`）：`event.json` + `report.md` + `report.pdf`（约 1.8MB，中文正常）+ `images/`（confirmed/peak/upgrade/final 截图 + trend.png）。
    - 早期报告（如 `FE-20260731144839`）的 `model_info` / `decisions` / `screenshots` / `system_info` 为空——对应链路早期未接入 M7/M8 数据，属开发期产物。

12. **当前已知限制**
    - 报告中的 `model_info` 依赖 UiController 传入；当前未训练时记录的是 COCO 模型（无火灾语义）。
    - 趋势图基于事件内 `detections` 的面积历史；COCO 模型下无 fire/smoke，趋势图无业务意义。

13. **是否已接入 UiController 和主运行链路**
    - 是。`UiController._generate_report` 在事件结束时组装 `EventReportData` 并调用 `report_gen.generate(data)`；M11 事件历史双击可打开对应报告。

14. **与最初项目计划的差异**
    - M10 经历"完成（`ca788d2`）→ 按核查文档完善（`2bd7d19`）"；新增封面、事件摘要、趋势图、完整性检查、配置快照、版本信息。
    - 相比计划增加 `event.json` 机器可读报告与 `images/` 截图副本。

## M11 GUI 控制器与主窗口

1. **模块名称与真实文件路径**
   - `ui/app_controller.py`（头注释 "M11 - GUI 控制器 (UiController)"）
   - `ui/main_window.py`（头注释 "M11 - GUI 主窗口 (MainWindow, PyQt6)"）
   - 入口：`main.py` → `MainWindow`

2. **当前实际职责**
   - M11 是整合层：装配检测源（M3/M4/M5）→ M6 聚合 → M7 决策 → M8 截图 → M9 日志 → M10 报告，并以 Qt 信号驱动界面；提供训练（M2）异步执行与逐 epoch 日志回调；MainWindow 只负责用户操作与信号显示。

3. **已实现功能**
   - `UiController`：`start/pause/resume/stop/reset/running/paused`；`_run_worker` 分发 `image/video/simulator`；`_process_frame` 统一帧处理（绘制 → `frame_ready`/`frame_info_ready` → `aggregator.process`）；`_analyze_event`（M7 + M8.on_decision + M9.log_decision + `decision_ready`）；`_generate_report`（M10）；`start_training/_train_worker/stop_training/training_running/training_results_dir`；`system_info()`；`shutdown()`（幂等：停线程 → flush → `save_run_summary`）。
   - `MainWindow`：监控页（左数据源/播放控制、中实时画面、右三区信息、底部事件历史与状态栏）、训练页、日志页（实时轮询 `application.log` 尾部，限长 1000 行）；状态灯、FPS、危险等级配色、日志过滤、错误弹窗、系统信息、最近路径（`QSettings`）、查看/打开报告目录。
   - 检测器惰性创建（`_ensure_image_detector/_ensure_video_detector/_simulator_detector`），避免 GUI 启动即加载 YOLO。

4. **输入对象及字段**
   - 检测源：`start(source_type, path, params)`；`params` 含 `conf/iou/interval/mode/loop/max_frames`。
   - 训练：`start_training(weights, dataset_yaml, epochs, batch, imgsz, device, project, name)`。
   - 帧输入：`Detection`（来自 M3/M4/M5）。

5. **输出对象及字段（Qt 信号 payload）**
   - `frame_ready`：`{image: QImage, frame_id, total_frames, source_name, inference_ms, timestamp}`。
   - `frame_info_ready`：`{frame_id, timestamp, classes, fire_area_ratio, smoke_area_ratio, inference_time_ms, detection_count}`。
   - `event_ready` / `history_ready`：事件状态与历史列表。
   - `decision_ready`：`{event, decision, ...}`。
   - `status_ready`：`{running, paused, source, fps, infer_ms, ...}`。
   - `source_finished`：`{source, processed_frames, events_confirmed, events_ended, events_discarded}`。
   - `train_log_ready(str)`、`train_epoch_ready(object)`、`train_finished(object)`、`error_ready(str)`。

6. **依赖的其他模块**
   - M3-M10 全部模块 + `PyQt6` + `QSettings` + `psutil`（系统信息）。

7. **使用的 config.yaml 配置项**
   - `gui.window_title`、`gui.window_size{width=1280,height=800}`、`gui.fps_display=true`、`gui.gpu_monitor=true`、`gui.stylesheet`；并间接使用全部业务配置段。

8. **对外接口、回调或 Qt Signal**
   - Qt 信号：`frame_ready / frame_info_ready / event_ready / decision_ready / history_ready / status_ready / source_finished / train_log_ready / train_finished / train_epoch_ready / error_ready`（均声明为 `pyqtSignal(object)` 或 `pyqtSignal(str)`）。
   - 方法接口：`start/pause/resume/stop/reset`、`start_training/stop_training`、`shutdown`、`log_file/read_log_tail/log_line_count/reports_dir/system_info`。

9. **是否运行在工作线程**
   - 是。检测源运行在 `threading.Thread`（daemon，名 `FireGuardianWorker`），训练运行在 `threading.Thread`（daemon，名 `FireGuardianTrain`）；暂停用 `threading.Event`；通过 Qt 信号跨线程通知 UI。

10. **异常与降级处理**
    - 工作线程内异常：捕获后 `error_ready.emit(...)`，`finally` 中复位状态并发出 `source_finished`。
    - 各业务模块回调 try/except（仅告警不中断主流程）。
    - `shutdown()` 幂等，退出前停止线程、flush、保存运行摘要。
    - 检测器加载失败会在首次使用图片/视频时经 `error_ready` 提示。

11. **已有测试及测试结果**
    - `UiController` 离屏自测（`QT_QPA_PLATFORM=offscreen`，合成 6 帧火焰 + 4 帧空白驱动 M6→M7→M8→M9→M10 全链路）本会话通过。
    - `MainWindow` 离屏实例化/信号接线本会话通过。
    - 用户已确认 GUI 可正常打开运行（此前 PyCharm "SDK is not defined" 为运行配置问题，与代码无关）。

12. **当前已知限制**
    - 检测器首次加载耗时数秒（惰性加载）；无帧率目标控制（视频按推理速度播放）。
    - GPU 监控依赖 psutil/pynvml 可用性。
    - 训练端到端（训练 Tab 启动真实训练）未验证；训练 Tab 第一版不做曲线（计划注明）。
    - 离屏自测使用合成 fire 检测框，不代表真实火灾检测。

13. **是否已接入 UiController 和主运行链路**
    - M11 本身即主运行链路（`main.py` 启动 MainWindow → 创建 UiController）。M3-M10 均已在其中接线。

14. **与最初项目计划的差异**
    - 计划中的"结果展示窗口"被扩展为完整监控/训练/日志/历史多 Tab GUI。
    - 训练控制、日志查看、最近路径、报告打开等为 M11 阶段新增能力（git：`fb793c5` → `784e0b5`）。

---

# A. M1-M11 完整调用链和数据流

```mermaid
flowchart LR
    subgraph SRC[检测源 M3/M4/M5]
        M3["M3 ImageDetector"]
        M4["M4 VideoDetector"]
        M5["M5 ValidationStreamSimulator"]
    end
    subgraph CORE[业务核心]
        M6["M6 EventAggregator"]
        M7["M7 FireDecisionAgent"]
        M8["M8 ScreenshotManager"]
        M9["M9 日志 logger"]
        M10["M10 ReportGenerator"]
    end
    M3 -->|Detection| M6
    M4 -->|Detection| M6
    M5 -->|FrameData + Detection| M6
    M6 -->|FireEvent 回调| M8
    M6 -->|FireEvent| M7
    M7 -->|FireDecision| M8
    M8 -->|ScreenshotRecord| M10
    M7 -->|FireDecision| M9
    M6 -->|FireEvent| M9
    M6 -->|FireEvent 结束| M10
    M6 -->|Qt 信号| UI["M11 UiController + MainWindow"]
    M7 -->|Qt 信号| UI
    M8 -->|Qt 信号| UI
    M10 -->|打开报告| UI
    UI -->|start/params| M3
    UI -->|start/params| M4
    UI -->|start/params| M5
    M2["M2 YOLOTrainer"] -.训练.->|models/best.pt| M3
    M2 -.->|models/best.pt| M4
    M2 -.->|models/best.pt| M5
```

**运行期主链路（检测）**
1. 用户在 GUI 选择检测源（图片/视频/验证集目录）→ `UiController.start(source_type, path, params)`。
2. 工作线程 `_run_worker` 分发：
   - `image` → `_run_image` → `ImageDetector.detect_single` → `Detection`（补齐 `frame_id=1`）。
   - `video` → `_run_video` → `VideoDetector.decode` 逐帧 → `(frame, Detection, frame_idx, total)`。
   - `simulator` → `_run_simulator` → `ValidationStreamSimulator.next_frame` → `(FrameData, Detection)`。
3. `_process_frame`：绘制标注 → 发 `frame_ready` / `frame_info_ready` → `aggregator.process(detection)`。
4. M6 状态机：
   - 阳性帧数 ≥ `min_positive_frames(3)` → `CONFIRMED` → `on_event_confirmed`：M8 确认截图 → M7 `analyze` → `decision_ready` → M9 `log_event_confirmed`。
   - 事件更新（阳性帧/趋势变化）→ `on_event_updated`：M8 峰值/升级截图 → 仅当 `metadata["decision_required"]` 时调用 M7 → M9 `log_event_updated`。
   - 连续阴性 ≥ `max_negative_frames(3)` → `ENDED` → `on_event_ended`：M7 最终决策 → M8 final 截图 → M10 生成报告 → M9 `log_event_ended` → `history_ready`。
   - 确认窗口内未达标 → `DISCARDED` → M9 `log_event_discarded` → `history_ready`。
5. M10 报告：`EventReportData(event, decisions, screenshots)` → `reports/event_<id>/{event.json, report.md, report.pdf, images/}`。
6. 结束：`source_finished` → 用户可停止/重置。

**运行期主链路（训练，M2）**
1. 训练 Tab 填写参数 → `UiController.start_training(...)` → `YOLOTrainer.train` 在独立线程运行。
2. 每 epoch → `train_epoch_ready` / `train_log_ready`；结束/停止 → `train_finished`。
3. 训练完成后期望 `models/best.pt` 就位，供 M3/M4/M5 下一次加载（当前未完成）。

---

# B. 共享对象定义位置与使用关系

全部共享对象定义在 `utils/common.py`：

| 对象 | 定义位置 | 生产者 | 消费者 |
| --- | --- | --- | --- |
| `BoundingBox` | `utils/common.py` | M3/M4/M5（解析 YOLO 输出） | M6（聚合判定）、M8（绘制/面积）、M11（绘制） |
| `Detection` | `utils/common.py` | M3/M4/M5 | M6（`process` 输入）、M8（`_pick_detection`）、M11（`_process_frame`） |
| `FrameData` | `utils/common.py` | M5 | M11（`_run_simulator`） |
| `SimulatorStats` | `utils/common.py` | M5（`_finalize`） | M5 自身、`on_complete` 回调 |
| `FireEvent` | `utils/common.py` | M6 | M7（`analyze`）、M8（截图）、M9（`log_event*`）、M10（`EventReportData.event`）、M11（回调/展示） |
| `FireDecision` | `utils/common.py` | M7 | M8（`on_decision`）、M9（`log_decision`）、M10（`EventReportData.decisions`）、M11（`decision_ready`） |
| `ScreenshotRecord` | `utils/common.py` | M8（`_build_record`） | M10（报告/图片说明）、M11（查询展示） |
| `EventReportData` | `utils/common.py` | M11（`_generate_report` 组装） | M10（`generate` 输入） |
| `AgentDecision` | `utils/common.py` | 无（预留） | 无（当前未被 M7/M11 使用） |

关键依赖链：`BoundingBox ⊂ Detection`；`FireEvent.detections: List[Detection]`；`EventReportData = FireEvent + List[FireDecision] + List[ScreenshotRecord] + system_info + model_info`。对象均可 `to_dict()`（`FireEvent/FrameData/SimulatorStats/FireDecision/ScreenshotRecord/AgentDecision`），供 M9/M10 序列化。

---

# C. 模型加载优先级及回退规则

| 位置 | 候选顺序 | 未找到时行为 |
| --- | --- | --- |
| M3 `detect/image_detector.py`（`_MODEL_CANDIDATES`） | `models/best.pt` → 项目根 `yolo11n.pt` | 抛 `FileNotFoundError` 并提示放置位置 |
| M4 `video_detect/video_detector.py`（`_find_model`） | `models/best.pt` → 项目根 `yolo11n.pt` | 同上 |
| M5 CLI `_create_default_detector()` | `models/best.pt` → 项目根 `yolo11n.pt` | 返回"空检测器"lambda（每帧无框，不崩溃） |
| M2 训练 `_MODEL_SOURCE` / `_MODEL_TARGET_DIR` | 源权重：项目根 `yolo11n.pt`；目标：`models/best.pt` | 训练前置校验报错 |

- `config.yaml` 中 `model.path=models/best.pt` 仅作为界面默认显示（`UiController._current_model_name` 在检测器未加载时显示该值）；实际加载以各模块 `_find_model` 为准。
- **当前回退结果**：`models/best.pt` 不存在 → 实际加载 `yolo11n.pt`（COCO 80 类，无 fire/smoke）→ **仅供链路测试，不能完成火灾检测**。

---

# D. 当前项目启动、训练、检测、报告生成流程

1. **启动**：`python main.py` → `MainWindow` → 创建 `UiController`（加载 config，实例化 M6/M7/M8/M10，检测器惰性不加载）→ 显示监控/训练/日志 Tab。
2. **训练**：训练 Tab → 选择权重/数据集 yaml/超参 → `start_training` → `FireGuardianTrain` 线程 → `YOLOTrainer.train` → 逐 epoch 回调 → `models/best.pt`（目标）→ `train_finished`。
3. **检测**：选择源（图片/视频/验证集）→ `start` → `FireGuardianWorker` 线程 → M3/M4/M5 → M6 → M7 → M8 → M9 → M10 → Qt 信号刷新界面。
4. **报告**：事件结束 → `_generate_report` → `ReportGenerator.generate` → `reports/event_<id>/`（event.json + report.md + report.pdf + images/）；GUI 事件历史双击可查看/打开。
5. **退出**：`closeEvent` → `controller.shutdown()`（停止检测/训练线程 → `flush` → `save_run_summary`）。

---

# E. 当前仍未完成真实业务验证的部分

1. **未训练火灾模型**：`models/best.pt` 不存在，M2 训练未在真实数据集上跑通。
2. **未做真实火灾端到端验证**：`me.mp4` 为烟花视频，COCO 模型无火灾语义；未用真实火焰/烟雾视频或图片验证全链路。
3. **M5 未在真实验证集批量运行**：F 盘数据集未连接时模拟器无数据。
4. **M7 决策未基于真实检出**：仅合成数据测试通过。
5. **M8 截图内容未验证**：当前真实场景无检测框，截图无火灾标注内容。
6. **M10 报告的 model_info/system_info**：早期报告为空；正式报告依赖 UiController 传入完整数据，未经真实事件验证。
7. **M11 训练端到端**：GUI 启动训练 → epoch 回调 → best.pt → 被 M3-M5 重新加载的闭环未验证。
8. **性能指标缺失**：无实测 FPS/训练速度/全量数据训练耗时的数据。

---

# F. 当前不建议继续修改的稳定模块

以下模块已按"改进说明 + 核查文档"完成多轮迭代，内置自测全部通过，结构稳定：

- **M7 FireDecisionAgent**：v2 核查后稳定，15 项用例全过。
- **M8 ScreenshotManager**：三阶段迭代后稳定，11 项自测全过。
- **M9 日志模块**：两轮迭代后稳定。
- **M10 ReportGenerator**：核查文档完善后稳定。
- **M11 UiController/MainWindow**：GUI 完成并核查完善。

建议：以上模块在模型训练完成前**不要再做结构重构**；模型训练完成后如需调参（M7 权重/阈值、M6 阈值、M8 峰值指标），应基于真实火灾数据评估后再改，且改动应走"核查文档 + 自测回归"流程。

---

# G. 需要在模型训练后才能验证的模块或逻辑

| 模块 | 需训练后验证的内容 |
| --- | --- |
| M2 | 训练可跑通、收敛曲线、`models/best.pt` 产出（前置条件） |
| M3/M4/M5 | fire/smoke 检出能力、置信度与面积比分布（`class_names` 含 fire/smoke 后才有业务输出） |
| M6 | 阳性帧判定阈值（`min_positive_frames`、`fire/smoke_confidence_threshold`、`min_area_ratio`）在真实检出分布下是否合理 |
| M7 | 权重（fire_area/smoke_area/duration/growth）与阈值（low/medium_max_score、override 规则）需基于真实事件校准 |
| M8 | peak 替换、danger upgrade 截图在真实火情下的有效性 |
| M9 | 事件日志中 fire/smoke 统计、危险等级分布的真实性 |
| M10 | 趋势图、截图证据、决策摘要、PDF 在真实事件上的可读性与完整性 |
| M11 | 端到端：真实视频/图片 → 事件 → 决策 → 截图 → 报告全链路 |

**结论**：当前代码链路完整、模块自测全部通过，但全部"火灾业务"结论均依赖 `models/best.pt`（训练产物）。在完成 M2 训练前，系统只能作为链路演示（COCO 模型），不能完成火灾检测。
