# FireGuardian Baseline V1 数据子集构建报告

- 工具: `tools/build_baseline_v1_subset.py`
- 数据源: `DATASET_ROOT`
- 审计报告: `tools\dataset_audit.json`
- 生成时间: 2026-08-02T17:08:48 | 耗时: 541.5s
- 随机种子: 42 | 并行 worker: 8

## 一、规模与类别分布

| 集合 | fire-only | smoke-only | fire+smoke | 负样本 | 合计(目标) | 合计(实际) |
|---|---|---|---|---|---|---|
| train | 4500 | 4500 | 3000 | 3000 | 15000 | 15000 |
| val | 900 | 900 | 600 | 600 | 3000 | 3000 |
| test | 900 | 900 | 600 | 600 | 3000 | 3000 |

## 二、过滤与防泄漏

- 审计越界框图片排除: 288
- 哈希失败: 0
- SHA-256 重复移除: 共 201（跨集合 88 / 集合内 113）
- pHash 完全一致(距离0)替换: 717 | 无替补删除: 0
- pHash 统计总数: 21000
- 处理后跨集合 pHash 完全一致配对: 0
- 疑似近似重复(汉明距离 1..8): 34483
- 近似重复样例: 30 对（仅报告，不删除）

## 三、构建后验证

| 检查项 | 结果 | 详情 |
|---|---|---|
| all_image_paths_exist | PASS | missing=0 |
| all_label_paths_exist | PASS | missing=0 |
| yolo_label_format | PASS | bad=0 |
| check_det_dataset | PASS | train=54 val=52 test=53 nc=2 names={0: 'fire', 1: 'smoke'} |
| ultralytics_random_load_32 | PASS | loaded={'train': 64, 'val': 64, 'test': 64} |

## 四、结论

最终结论: **SUITABLE**

## 五、输出文件

- `train`: `datasets\baseline_v1_train.txt`
- `val`: `datasets\baseline_v1_val.txt`
- `test`: `datasets\baseline_v1_test.txt`
- `yaml`: `datasets\fire_smoke_baseline_v1.yaml`
