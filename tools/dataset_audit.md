# FireGuardian 数据集审计报告

- 数据集根目录：`F:\火焰数据集\Dataset`
- 生成时间：2026-08-01T17:13:08
- 工具：`tools/dataset_audit.py` v1.0.0 | 扫描耗时 1988.5s | 进程数 8
- 感知哈希：启用（阈值 16）

## 1. 最终结论

**可以开始 YOLO 冒烟训练（存在警告）**

> 可以开始 YOLO 冒烟训练，但存在以下警告：319 个标注框坐标/宽高越界；跨集合完全重复图片 1706 组 / 3412 个文件；近似重复图片对 636434 对。

**警告**
- 319 个标注框坐标/宽高越界
- 跨集合完全重复图片 1706 组 / 3412 个文件
- 近似重复图片对 636434 对

**建议**：建议先运行 1-3 个 epoch 的冒烟训练（如 epochs=3, batch=16, imgsz=640）验证 M2 训练链路与 models/best.pt 产出，确认无误后再进行完整训练；test 集合保留用于最终评估。注意：train/valid/test 之间存在大量近似重复（含部分完全重复）图片，训练/验证指标可能偏乐观，正式实验前建议先做去重并重新划分集合。

## 2. 总体统计

| 指标 | 数值 |
| --- | --- |
| 图片总数 | 105257 |
| 标签文件总数 | 105257 |
| 匹配的图片-标签对 | 105257 |
| 缺失标签（图片无标签） | 0 |
| 孤立标签（标签无图片） | 0 |
| 损坏/不可解码图片 | 0 (0.00%) |
| 标签文件含解析错误 | 0 (0.00%) |
| 空标签文件 | 41620 |
| 空标签对应图片 | 41620 |
| 正常负样本（可解码+空标签） | 41620 |
| fire 图片数 / 框数 | 37367 / 87358 |
| smoke 图片数 / 框数 | 48784 / 58920 |
| 同时含 fire 与 smoke 的图片 | 22514 |
| class_id 越界框数（非 0/1） | 0 |
| 坐标/宽高越界框数 | 319 |
| 跨集合完全重复组 / 文件 | 1706 / 3412 |
| 集合内部重复组（参考） | 1921 |
| 近似重复图片对（参考） | 636434 对（列表已截断） |
| 数据总量 | 14256.3 MB |

## 3. 各集合统计

### train

| 指标 | 数值 |
| --- | --- |
| 图片数 | 73698 |
| 标签数 | 73698 |
| 匹配数 | 73698 |
| 缺失标签 | 0 |
| 孤立标签 | 0 |
| 损坏图片 | 0 (0.00%) |
| 标签解析错误文件 | 0 (0.00%) |
| 空标签文件 / 对应图片 | 29138 / 29138 |
| 正常负样本 | 29138 |
| fire 图片 / 框 | 26166 / 61391 |
| smoke 图片 / 框 | 34159 / 41289 |
| 双类图片 | 15765 |
| class_id 越界框 | 0 |
| 坐标/宽高越界框 | 228 |
| 图片总大小 | 10026.8 MB |

**异常清单（每类最多展示 10 条）**
- **坐标/宽高越界标签**：共 205 个
  - `train/labels/Fire_CV280.txt`
  - `train/labels/FireAndSmoke_CV19302.txt`
  - `train/labels/FireAndSmoke_CV1926.txt`
  - `train/labels/FireAndSmoke_CV864.txt`
  - `train/labels/FireAndSmoke_CV900.txt`
  - `train/labels/FireAndSmoke_CV1697.txt`
  - `train/labels/FireAndSmoke_CV19504.txt`
  - `train/labels/FireAndSmoke_CV19206.txt`
  - `train/labels/Fire_CV1507.txt`
  - `train/labels/Smoke_CV8269.txt`
  - … 其余 195 个见 dataset_audit.json

### valid

| 指标 | 数值 |
| --- | --- |
| 图片数 | 10517 |
| 标签数 | 10517 |
| 匹配数 | 10517 |
| 缺失标签 | 0 |
| 孤立标签 | 0 |
| 损坏图片 | 0 (0.00%) |
| 标签解析错误文件 | 0 (0.00%) |
| 空标签文件 / 对应图片 | 4160 / 4160 |
| 正常负样本 | 4160 |
| fire 图片 / 框 | 3732 / 8899 |
| smoke 图片 / 框 | 4874 / 5923 |
| 双类图片 | 2249 |
| class_id 越界框 | 0 |
| 坐标/宽高越界框 | 30 |
| 图片总大小 | 1450.4 MB |

**异常清单（每类最多展示 10 条）**
- **坐标/宽高越界标签**：共 29 个
  - `valid/labels/Smoke_CV1331.txt`
  - `valid/labels/Smoke_CV8303.txt`
  - `valid/labels/Fire_CV1702.txt`
  - `valid/labels/FireAndSmoke_CV19299.txt`
  - `valid/labels/FireAndSmoke_CV19495.txt`
  - `valid/labels/FireAndSmoke_CV19220.txt`
  - `valid/labels/Fire_CV1706.txt`
  - `valid/labels/Fire_CV1668.txt`
  - `valid/labels/Fire_CV1284.txt`
  - `valid/labels/Fire_CV1526.txt`
  - … 其余 19 个见 dataset_audit.json

### test

| 指标 | 数值 |
| --- | --- |
| 图片数 | 21042 |
| 标签数 | 21042 |
| 匹配数 | 21042 |
| 缺失标签 | 0 |
| 孤立标签 | 0 |
| 损坏图片 | 0 (0.00%) |
| 标签解析错误文件 | 0 (0.00%) |
| 空标签文件 / 对应图片 | 8322 / 8322 |
| 正常负样本 | 8322 |
| fire 图片 / 框 | 7469 / 17068 |
| smoke 图片 / 框 | 9751 / 11708 |
| 双类图片 | 4500 |
| class_id 越界框 | 0 |
| 坐标/宽高越界框 | 61 |
| 图片总大小 | 2779.1 MB |

**异常清单（每类最多展示 10 条）**
- **坐标/宽高越界标签**：共 54 个
  - `test/labels/Smoke_CV8398.txt`
  - `test/labels/FireAndSmoke_CV1603.txt`
  - `test/labels/Fire_CV1707.txt`
  - `test/labels/Smoke_CV8472.txt`
  - `test/labels/Smoke_CV8244.txt`
  - `test/labels/FireAndSmoke_CV19484.txt`
  - `test/labels/FireAndSmoke_CV1204.txt`
  - `test/labels/FireAndSmoke_CV19371.txt`
  - `test/labels/FireAndSmoke_CV1192.txt`
  - `test/labels/FireAndSmoke_CV19458.txt`
  - … 其余 44 个见 dataset_audit.json

## 4. 类别分布（图片级 / 框级）

| 集合 | fire 图片 | fire 框 | smoke 图片 | smoke 框 | 双类图片 | 负样本 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 26166 | 61391 | 34159 | 41289 | 15765 | 29138 |
| valid | 3732 | 8899 | 4874 | 5923 | 2249 | 4160 |
| test | 7469 | 17068 | 9751 | 11708 | 4500 | 8322 |

## 5. 跨集合完全重复图片（SHA-256）

- 共 1706 组，涉及 3412 个文件（重复副本 1706 张）。
- `89778a41472610a20145dcacc08ebdcb61d4c9e2d67fd5d48f6533dfc5a420c5`（2 个文件）
  - `test/images/FireAndSmoke_CV17024.jpg`
  - `train/images/FireAndSmoke_CV1001.jpg`
- `f860ea6601a4895d386a611fb201321eed57ed9dc399cbc44ac2514ab33bd7f9`（2 个文件）
  - `test/images/FireAndSmoke_CV18341.jpg`
  - `train/images/FireAndSmoke_CV1006.jpg`
- `176953e9997923d1b42bfedd05887a6246212e5017b67165726f61614b32675e`（2 个文件）
  - `test/images/FireAndSmoke_CV17656.jpg`
  - `train/images/FireAndSmoke_CV1009.jpg`
- `6cdc9c15db38480a8b6ee99f70703c9a4ff67cdca45e0f098bbe072d11135fe3`（2 个文件）
  - `test/images/FireAndSmoke_CV17904.jpg`
  - `train/images/FireAndSmoke_CV1014.jpg`
- `eaca6195f9c65350bcdca8e26402cc46e1af62467bfe3ff3506d764f71f03f5f`（2 个文件）
  - `train/images/FireAndSmoke_CV1016.jpg`
  - `valid/images/FireAndSmoke_CV19879.jpg`
- `9a836885e3cc438c9dc0d40e5d7fa82ea8ceef6d793c4532b91fea17497bb2c9`（2 个文件）
  - `test/images/FireAndSmoke_CV19620.jpg`
  - `train/images/FireAndSmoke_CV1029.jpg`
- `5c6fdd660e0f1650a4b9eaba3bd8d0c8632eb4030f5204cdf2ce678b315218fc`（2 个文件）
  - `test/images/FireAndSmoke_CV17457.jpg`
  - `train/images/FireAndSmoke_CV1035.jpg`
- `a53e886bd1bf552192c34dad3b5b37012dcd1b334bb2a88f26c68b552c61f69b`（2 个文件）
  - `test/images/FireAndSmoke_CV18750.jpg`
  - `train/images/FireAndSmoke_CV1064.jpg`
- `5512d897833750ab0e7546556c28dc46c8718613ffd9c70c5d80aac025470d49`（2 个文件）
  - `train/images/FireAndSmoke_CV1074.jpg`
  - `valid/images/FireAndSmoke_CV18641.jpg`
- `f4ae36f58526cc0fc4d0641b4e04abeb83a67248b7ad2e07671c755d02693f8c`（2 个文件）
  - `test/images/FireAndSmoke_CV20089.jpg`
  - `train/images/FireAndSmoke_CV1080.jpg`
- … 其余 1696 组见 dataset_audit.json

## 6. 近似重复图片（感知哈希）

- 共 636434 对（16x16 aHash 汉明距离 ≤ 16；距离 0 的完全相同图由第 5 节覆盖）。
- 有 355 个哈希桶因过大被跳过，实际近似重复对数可能略多。
- 距离 1：`test/images/FireAndSmoke_CV10794.jpg` ↔ `test/images/Fire_CV8253.jpg`
- 距离 1：`test/images/FireAndSmoke_CV10794.jpg` ↔ `test/images/Fire_CV8694.jpg`
- 距离 1：`test/images/FireAndSmoke_CV10955.jpg` ↔ `test/images/FireAndSmoke_CV3782.jpg`
- 距离 1：`test/images/FireAndSmoke_CV11075.jpg` ↔ `test/images/FireAndSmoke_CV12246.jpg`
- 距离 1：`test/images/FireAndSmoke_CV11097.jpg` ↔ `test/images/FireAndSmoke_CV14642.jpg`
- 距离 1：`test/images/FireAndSmoke_CV11447.jpg` ↔ `test/images/FireAndSmoke_CV12878.jpg`
- 距离 1：`test/images/FireAndSmoke_CV11633.jpg` ↔ `test/images/FireAndSmoke_CV6457.jpg`
- 距离 1：`test/images/FireAndSmoke_CV11664.jpg` ↔ `test/images/FireAndSmoke_CV9133.jpg`
- 距离 1：`test/images/FireAndSmoke_CV11964.jpg` ↔ `test/images/FireAndSmoke_CV13227.jpg`
- 距离 1：`test/images/FireAndSmoke_CV12180.jpg` ↔ `test/images/FireAndSmoke_CV7976.jpg`
- 距离 1：`test/images/FireAndSmoke_CV12295.jpg` ↔ `test/images/FireAndSmoke_CV4416.jpg`
- 距离 1：`test/images/FireAndSmoke_CV12396.jpg` ↔ `test/images/FireAndSmoke_CV6258.jpg`
- 距离 1：`test/images/FireAndSmoke_CV12644.jpg` ↔ `test/images/FireAndSmoke_CV12645.jpg`
- 距离 1：`test/images/FireAndSmoke_CV12645.jpg` ↔ `test/images/Fire_CV10505.jpg`
- 距离 1：`test/images/FireAndSmoke_CV12751.jpg` ↔ `test/images/FireAndSmoke_CV9411.jpg`
- 距离 1：`test/images/FireAndSmoke_CV12800.jpg` ↔ `test/images/Fire_CV3406.jpg`
- 距离 1：`test/images/FireAndSmoke_CV13025.jpg` ↔ `test/images/FireAndSmoke_CV3341.jpg`
- 距离 1：`test/images/FireAndSmoke_CV13025.jpg` ↔ `test/images/FireAndSmoke_CV7972.jpg`
- 距离 1：`test/images/FireAndSmoke_CV13120.jpg` ↔ `test/images/FireAndSmoke_CV4244.jpg`
- 距离 1：`test/images/FireAndSmoke_CV13145.jpg` ↔ `test/images/FireAndSmoke_CV7237.jpg`
- 列表已截断，仅保留前 10000 对示例；总数见上方。

## 7. 文件名前缀与标签一致性（参考信息）

- 文件名前缀（Fire/Smoke/FireAndSmoke/FireNorSmoke）与标签内容完全一致。

## 8. 审计要求核对清单

| # | 检查项 | 结果 |
| --- | --- | --- |
| 1 | 各集合图片数/标签数/匹配数 | 见第 2、3 节 |
| 2 | 缺失标签与孤立标签 | 0 缺失 / 0 孤立 |
| 3 | 损坏/不可解码图片（OpenCV+Pillow） | 0 张 |
| 4 | YOLO txt 严格格式（5 列） | 0 个文件含错误行 |
| 5 | class_id 仅 0/1 | 0 个越界框 |
| 6 | 坐标/宽高合法范围 | 319 个越界框 |
| 7 | 空标签文件及对应图片 | 41620 / 41620 |
| 8 | fire/smoke 图片级与框级数量 | fire 37367/87358，smoke 48784/58920 |
| 9 | 同时含 fire 和 smoke 的图片 | 22514 张 |
| 10 | 不含目标的正常负样本 | 41620 张 |
| 11 | 跨集合完全重复（SHA-256） | 1706 组 / 3412 文件 |
| 12 | 近似重复（感知哈希，可选） | 已启用：636434 对（列表已截断） |
| 13 | 异常文件清单（只读，不删除/修改） | 见第 3 节与 dataset_audit.json |
| 14 | 输出 dataset_audit.json / dataset_audit.md | 已生成 |
| 15 | 是否适合开始 YOLO 冒烟训练 | SUITABLE_WITH_WARNINGS |

---
*本报告由 `tools/dataset_audit.py` 自动生成；审计过程为只读操作，未复制、移动或删除任何数据。*
