# FireGuardian M2 首次真实冒烟训练报告

> 执行时间：2026-08-01
> 任务来源：《8.FireGuardian M2 首次真实冒烟训练任务.docx》
> 执行方式：通过 M2 `YOLOTrainer` 完成真实数据训练，未使用 CLI `yolo train` 绕过
> 本次目的：验证 M2 训练闭环 + M3/M4/M5 模型切换 + M6-M10 业务链路，不追求精度，不校准业务参数

---

## 1. 训练环境

| 项目 | 值 |
|------|-----|
| Python | 3.11.9 |
| PyTorch | 2.13.0+cu126 |
| CUDA | 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（8GB） |
| Ultralytics | 8.4.108 |
| OpenCV | 5.0.0 |

## 2. 数据集

- 数据配置：`datasets/fire_smoke_smoke.yaml`（`nc: 2`，`0: fire`，`1: smoke`）
- 训练集（train）：**3000 张**
- 验证集（valid）：**600 张**

| 类别 | train | valid |
|------|------:|------:|
| fire-only | 900 | 180 |
| smoke-only | 900 | 180 |
| fire + smoke | 600 | 120 |
| 无目标负样本 | 600 | 120 |

标注框数量：

| 集合 | fire 框 | smoke 框 | 合计 |
|------|--------:|---------:|-----:|
| train | 3288 | 1762 | 5050 |
| valid | 725 | 359 | 1084 |

抽样说明：固定随机种子 42；已排除审计报告中的越界标注样本（319 个越界框 / 288 个文件）与损坏/异常样本；同一 SHA-256 重复组只进入一个集合；valid 与 train 无 SHA-256 完全重复。

## 3. 训练参数

| 参数 | 值 |
|------|-----|
| weights | `yolo11n.pt` |
| epochs | 3 |
| batch | 16 |
| imgsz | 640 |
| device | 0（GPU） |
| workers | 4 |
| seed | 42（入口脚本固定 random/numpy/torch 种子；Ultralytics 内部 args.seed=0，deterministic=True） |
| project / name | `runs/smoke_train` / `smoke_v1` |

## 4. 训练结果

来源：`runs/smoke_train/smoke_v1/results.csv`（真实验证指标）

| epoch | precision | recall | mAP50 | mAP50-95 |
|------:|----------:|-------:|------:|---------:|
| 1 | 0.2599 | 0.2879 | 0.1830 | 0.0662 |
| 2 | 0.4224 | 0.3087 | 0.3061 | 0.1421 |
| 3 | 0.5085 | 0.4229 | 0.4306 | 0.2200 |

- 最佳 epoch：3（最后一次验证）
- 最佳 precision / recall / mAP50 / mAP50-95：0.5085 / 0.4229 / 0.4306 / 0.2200
- 训练总耗时：约 310.5 秒（5.2 分钟）

## 5. 模型产物

| 项目 | 值 |
|------|-----|
| 训练原始 best.pt | `runs/smoke_train/smoke_v1/weights/best.pt`（5,469,329 字节） |
| `models/best.pt` 是否存在 | **存在**（5,469,329 字节） |
| `model.names`（新进程加载） | `{0: 'fire', 1: 'smoke'}`，nc=2 |

## 6. 模型加载验证（M2 → M3/M4/M5 切换闭环）

| 阶段 | 加载模型 | class names |
|------|----------|-------------|
| 训练前（best.pt 不存在） | 回退 `yolo11n.pt` | COCO 80 类（仅供链路测试，不能完成火灾检测，见 `docs/SYSTEM_AUDIT.md`） |
| 训练后（新进程） | `models/best.pt` | `{0: 'fire', 1: 'smoke'}` |

- **M3 ImageDetector**：自动加载 `models/best.pt`
  - Fire 图片 → 检出 `fire`，conf 0.698
  - Smoke 图片 → 检出 `smoke`，conf 0.517
  - 正常负样本图片 → 0 个目标
  - Fire+Smoke 图片 → 检出 `fire`，conf 0.873
- **M4 VideoDetector**：自动加载 `models/best.pt`，`me.mp4` 共 182 帧全部处理
  - fire 帧 12 帧（最高 conf 0.753），smoke 帧 31 帧（最高 conf 0.786）
  - `Detection.class_name` 可输出 `fire` / `smoke`
- **M5 ValidationStreamSimulator**：自动加载 `models/best.pt`，12 张代表性验证集图片模拟流
  - `fire_area_ratio` 峰值 0.66114（非零帧 3 帧）
  - `smoke_area_ratio` 峰值 0.08474（非零帧 3 帧）
  - 5 帧符合 M6 阳性条件

## 7. 业务链路验证（M6 → M7 → M8/M9/M10）

使用 best.pt 的真实检测结果（4 帧阳性 + 3 帧阴性）驱动完整链路，产生事件 `FE-20260801201434-66f76e`：

| 模块 | 验证项 | 结果 |
|------|--------|------|
| M6 | 收到真实 Detection → 生成 FireEvent（confirmed → ended） | ✅ |
| M7 | 基于真实 FireEvent 输出 FireDecision（3 次：medium/medium/high） | ✅ |
| M8 | `screenshots/event_FE-20260801201434-66f76e/`（confirmed/peak/升级/final 共 16 个文件） | ✅ |
| M9 | `logs/events.jsonl` 新增 9 行（事件确认/更新/结束 + 决策） | ✅ |
| M10 | `reports/event_FE-20260801201434-66f76e/`（event.json + report.md + report.pdf） | ✅ |

最终决策：危险等级 **high**（score 0.615，confidence 0.614），事件摘要包含火焰峰值 66.1%、烟雾峰值 8.5%、时长 6.0s、增长趋势 increasing。

结论：**真实 fire/smoke 检测 → FireEvent → FireDecision → Screenshot → EventLog → Report 全链路打通。**

## 8. 代码修改记录（最小必要修复）

本次任务只做了“训练/验证无法完成”的最小修复，未重构任何模块、未调整任何业务阈值：

1. **M2 回调注册 bug**（`train/yolo_trainer.py`）
   - 问题：`YOLO.add_callback(...)` 以类方式调用，在 Ultralytics 8.4.108 报 `TypeError: Model.add_callback() missing 1 required positional argument: 'func'`
   - 修复：改为实例方法 `self.model.add_callback("on_train_epoch_end", ...)` / `("on_train_end", ...)`
   - 影响：训练回调无法注册 → 训练失败；修复后训练正常

2. **M2 输出目录 / best.pt 复制 bug**（`train/yolo_trainer.py`）
   - 问题：Ultralytics 8.4 将相对 project 嵌套成 `runs/detect/runs/smoke_train/...`，且复制源路径找不到文件，`models/best.pt` 无法生成
   - 修复：`train()` 内将相对 project 转为基于项目根的绝对路径；复制源改为 `Path(self.model.trainer.save_dir) / "weights" / "best.pt"`
   - 影响：`models/best.pt` 无法生成 → 修复后正常生成并复制

3. **M5 threading 未导入 bug**（`simulator/validator_simulator.py`）
   - 问题：`__init__` 中引用 `threading` 但模块未 `import threading`，在其他库先加载 `threading` 时触发 `NameError`
   - 修复：补充 `import threading`
   - 影响：M5 模拟器在完整环境中无法构建 → 修复后验证通过

## 9. 发现的问题

| # | 问题 | 原因 | 是否影响主流程 | 建议下一步 |
|---|------|------|----------------|------------|
| 1 | M2 回调中 `total_epochs` 显示 100、loss 显示 0 | 回调使用实例默认 epochs 值、`trainer.metrics` 的 loss 键取不到 | 不影响（`results.csv` 指标正确） | 后续让回调展示实际 epochs 与正确 loss 键 |
| 2 | M3 `detect_single` 返回的 Detection 未填 `image_width/height` | 单图接口未读取图像尺寸 | 生产 GUI 会补全（`ui/app_controller.py`）；直接喂 M6 需补尺寸 | 可选增强 |
| 3 | `me.mp4` 中 fire/smoke 检测帧数偏少且有噪声（fire 12 / smoke 31 帧） | 3 epoch 小模型精度有限 | 不影响链路验证 | 正式训练后复测 |
| 4 | M5 模拟器 `next_frame` 模式统计“总帧数/含火焰帧”显示 0 | 单步模式未更新 `stats` 计数 | 不影响 `next_frame` 输出 | 可选修复统计字段 |
| 5 | M10 报告“检测来源”引用 `runs/e2e_m5_images`（临时目录已清理） | 验证使用临时拷贝路径 | 不影响报告主体（图片已复制进报告） | 正式运行时使用真实数据目录 |

## 10. 任务书验收对照

| 验收项 | 结果 |
|--------|------|
| M2 YOLOTrainer 成功完成 3 epoch 真实训练 | ✅ |
| `models/best.pt` 成功生成 | ✅ |
| best.pt class names = fire/smoke | ✅ |
| M3 / M4 / M5 重启后自动加载 best.pt | ✅ |
| M3 能实际检测 fire / smoke | ✅ |
| M5 输出有效 `fire_area_ratio` / `smoke_area_ratio` | ✅ |
| M6 能收到真实 Detection 并产生 FireEvent | ✅ |
| M7 能基于真实 FireEvent 输出 FireDecision | ✅ |
| M8 / M9 / M10 能正常处理真实事件 | ✅ |
| 生成 `smoke_training_report.md` | ✅ |

## 11. 结论与后续建议

- 本次冒烟训练验证了 FireGuardian 从“真实数据训练 → 模型落盘 → 自动加载 → 真实检测 → 事件/决策/截图/日志/报告”的完整闭环。
- **不根据本次 3 epoch 结果调整** M6 阈值、M7 权重与危险等级、M8 peak_metric（按要求）。
- 后续正式训练建议使用更大 epoch 与更充分的训练集，再重新评估 M3-M10 的检测精度与业务参数。
