# FireGuardian V2 Phase 5 报告：V2 模型优化

> 日期：2026-08-16
> 分支：`v2-deepseek-agent`（HEAD 未变：`0eca3ad`）
> 状态：**Phase 5 完成，未提交 Git**

---

## 1. 数据准备（5-1：无泄漏重划分）

- 全量扫描 DATASET_ROOT：105,257 张（train 73,698 / valid 10,517 / test 21,042）；
- 片段级 union-find 分组：基于审计 JSON 的 exact_duplicates（1,706 组）+
  near_duplicates（distance ≤ 8）合并为 99,730 个片段（最大片段 212），
  确保同一片段（含跨集合重复/近重复）整体归属单一集合；
- 排除越界框图片 288 张；
- 类别配额分配：train 15,066 / val 3,000 / test 3,000，
  四类别（negative/fire_only/smoke_only/both）在三个集合中齐全且比例近似全量
  （negative ~40%、smoke ~25%、fire ~14%、both ~21%）；
- 校验：片段不跨集合（clip_split_disjoint=true）；清单见
  datasets/v2_15k_*.txt + datasets/fire_smoke_v2_15k.yaml；
- 扫描结果缓存 datasets/.v2_scan_cache.json（重跑秒级）。

## 2. 困难负样本（5-2）

- 用当前模型从 烟花1/烟花2/me.mp4 抽取 66 张「被误检为 fire/smoke」帧
  （烟花1:41、烟花2:5、me:20），空标签负样本，存入 datasets/v2_negatives/
  并并入训练集（train 60 张），只读原视频/模型。

## 3. 训练（5-3）

- tools/v2_model_train_entry.py：yolo11n.pt、40 epochs、batch=16、imgsz=640、
  device=0、patience=15、seed=42，runs/v2_model/v2_15k；
- 训练时长 286 分钟（约 4.8h），40 轮全部完成（未早停），best epoch=40；
- best 权重：models/v2_15k_best.pt（SHA-256 已校验），未覆盖 models/best.pt；
- 训练曲线：val mAP50 0.294 -> 0.694，val loss 1.54 -> 1.02。

## 4. 评估（5-4，同一 v2 test 集公平对比）

| 模型 | mAP50 | mAP50-95 | P@0.25 | R@0.25 | 负样本图级误报率(0.25/0.4/0.5) |
|---|---|---|---|---|---|
| V1 baseline | 0.7325 | 0.4491 | 0.7456 | 0.6419 | 5.63% / 3.70% / 2.61% |
| V2-15k | 0.7161 | 0.4334 | 0.7397 | 0.6356 | 2.52% / 1.34% / 0.84% |

- V2 在无泄漏测试集上 mAP50 略低于 V1（0.716 vs 0.733，约 -2.2%），
  但负样本误报率下降 2~3 倍（conf=0.5 时 2.61% -> 0.84%）；
- 说明：V1 基线原报告（mAP50 0.7352）基于含跨集合近重复的旧测试集，
  指标偏乐观；本表为同一 v2 test 集上的真实对比。

## 5. conf 阈值结论（计划书 §31）

- 建议生产 conf 保持 0.50：V2 在该阈值图级误报率 0.84%，误报优先策略下最优；
- 如需更高召回可降 0.25（误报 2.52%）；F1@0.25 两者接近（0.684 vs 0.690）。
- 分析输出：tools/v2_conf_analysis.json / tools/v2_conf_analysis.md。

## 6. 已知问题

1. V2 在保持误报极低的同时召回略降（fire R 0.628->0.636 区间波动），
   若需提升召回，计划书 Step 4/5 可选：继续训练至 50~60 轮或升级 YOLO11s；
2. 困难负样本池规模小（66 张），后续可扩充；
3. 训练受 F 盘读取速度限制（约 6MB/s），40 轮耗时 4.8h。

## 7. 结论

Phase 5 完成：无泄漏数据集、困难负样本、V2 模型训练与 conf 分析全部落地；
模型以「误报大幅下降、召回基本持平」为代价换取了更好的误报控制，
与 Agent/Memory/Hybrid 组合在 Phase 6 联合测试中验证。
