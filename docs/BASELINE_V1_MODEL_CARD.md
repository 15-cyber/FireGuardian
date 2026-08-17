# FireGuardian Baseline V1 模型说明卡

## 基本信息

- **模型名称**：Baseline V1（`baseline_v1_best.pt`，YOLO11n）
- **类别**：`{0: "fire", 1: "smoke"}`，`nc = 2`
- **当前默认模型**：`models/best.pt`（已于 2026-08-05 晋升为 Baseline V1）
- **训练配置**：`weights=yolo11n.pt`，30 epochs（未早停），batch=16，imgsz=640，patience=10，seed=42

## 数据规模

- 训练集：15,000 张
- 验证集：3,000 张
- 测试集：3,000 张
- 负样本误报评估：600 张正常图

## 测试指标（test 集，conf=0.25，IoU=0.5）

| 指标 | 总体 | fire | smoke |
| --- | --- | --- | --- |
| Precision | 0.7313 | 0.6987 | 0.7902 |
| Recall | 0.6533 | 0.6284 | 0.6973 |
| mAP50 | 0.7352 | 0.6942 | 0.7761 |
| mAP50-95 | 0.4557 | 0.3903 | 0.5211 |
| TP / FP / FN | 3264 / 1199 / 1732 | 2006 / 865 / 1186 | 1258 / 334 / 546 |

负样本（600 张，conf=0.50）：图级误报率 2.5%（fire 5 张、smoke 10 张）。

## 推荐使用范围

- 仅用于**开发测试与端到端链路联调**（M3～M11 演示、GUI 演示、事件链路验证）。
- 不作为最终生产模型；正式对外/生产使用需在数据修正、完整训练集与更多轮次后重新评估。

## 已知局限

- fire 类 recall 偏低（conf=0.50 时约 0.44），夜间、远距离、小目标与强光/亮光误判集中。
- 大量正确检测集中在 0.25～0.45 低置信带，模型校准与训练轮次不足。
- 跨集合存在约 34,483 对疑似近似重复（pHash 距离 1～8），Test 指标可能被轻微抬高。
- 图片级混淆矩阵的串类数字被双类图片主导（占 89.9% / 96.0%），建议修正标注口径后重评。

## 配置与工程约定

- **默认 conf 仍为 0.50**（`config.yaml → model.confidence`），本模型卡不修改该值。
- **M6 面积**：`fire_area_ratio` / `smoke_area_ratio` 使用**同类别 bbox 矩形并集**面积（重叠不重复计数），输出恒在 `[0, 1]`。
- **M7 时长**：事件时间为**秒制**（`start_time_seconds` / `end_time_seconds` / `duration_seconds`），阈值基于真实秒。
- **模型能力不夸大**：本模型为 Baseline 精度基线，不等同于生产部署认证。

## 权重获取与生成方式

- 训练：`python tools/baseline_v1_train_entry.py`（使用 `DATASET_ROOT` 构建的子集，yaml：`datasets/fire_smoke_baseline_v1.yaml`）。
- 晋升：`python tools/promote_model.py --candidate models/baseline_v1_best.pt --execute`（旧模型备份在 `models/best.backup_*.pt`）。
- 版本记录：`models/model_versions.jsonl`（本地运行记录，不入库）。
- **模型权重 `.pt` 不进入普通 Git**（已在 `.gitignore` 忽略）。
