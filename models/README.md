# Models

Model weight files are **not included** in this repository.

## V1

- `baseline_v1_best.pt` — V1 Baseline 最佳权重
  - SHA-256 前缀：`5b84350ebdfaf489`

## V2

- `v2_15k_best.pt` — V2 模型（Phase 5，无泄漏数据集 + 困难负样本训练）
  - SHA-256 前缀：`9e674c156d8d638a`

## 默认模型

`config/config.yaml` 默认加载 `models/best.pt`。请将你训练或获得的权重放入 `models/`
目录并命名为 `best.pt`（或将 `config.yaml` 的模型路径指向对应文件）。

## 没有模型时会发生什么

- M3/M4/M5 检测模块在加载时找不到模型文件会明确报错；
- GUI 在启动检测源前不会加载模型（惰性加载），放置权重后即可运行；
- Agent / Memory / Hybrid 依赖事件特征，模型缺失时无法产生检测事件。

## 如何获得权重

模型权重不随仓库分发；你可以使用自己训练的结果，或训练自己的模型
（参见 `tools/v2_model_train_entry.py` / `train/`）。请不要虚构下载地址。
