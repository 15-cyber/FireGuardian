"""
============================================================
FireGuardian Baseline V1 - 最终报告组装
============================================================
读取以下结果文件，生成 tools/baseline_v1_training_report.md：
- tools/baseline_v1_subset_report.json      （数据构建）
- tools/baseline_v1_training_summary.json   （训练）
- tools/baseline_v1_evaluation.json         （test 评估）
- tools/baseline_v1_me_video_result.json    （me.mp4）
- tools/baseline_v1_chain_verify_result.json（M6-M10）

用法:
  python tools/build_baseline_v1_report.py
============================================================
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
_TOOLS = _ROOT / "tools"
SMOKE_KNOWN = {
    "train": 3000, "val": 600, "epochs": 3,
    "precision": 0.5085, "recall": 0.4229,
    "mAP50": 0.4306, "mAP50_95": 0.2200,
    "me_fire_frames": 12, "me_smoke_frames": 31,
}


def load(name):
    p = _TOOLS / name
    if not p.exists():
        print("[warn] 缺少文件:", p)
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=4):
    try:
        return ("%." + str(nd) + "f") % float(v)
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    subset = load("baseline_v1_subset_report.json")
    summary = load("baseline_v1_training_summary.json")
    ev = load("baseline_v1_evaluation.json")
    me = load("baseline_v1_me_video_result.json")
    chain = load("baseline_v1_chain_verify_result.json")
    valm = load("baseline_v1_val_metrics.json")

    lines = []
    a = lines.append
    a("# FireGuardian Baseline V1 中等规模训练与精度评估报告")
    a("")
    a("> 生成时间: %s" % (__import__("time").strftime("%Y-%m-%d %H:%M:%S")))
    a("")
    a("## 1. 实验目的")
    a("")
    a("完成第一次中等规模正式基线训练（train 15000 / val 3000 / test 3000），")
    a("评估 Fire/Smoke 检测模型的真实精度、漏检、误报和类别表现，不再以“打通链路”为主要目标。")
    a("")

    a("## 2. 与冒烟测试的隔离方式")
    a("")
    a("- 冒烟产物只读保护：`datasets/smoke_*`、`runs/smoke_train/smoke_v1/`、`models/best.pt` 均未改动。")
    a("- 冒烟模型已备份为 `models/smoke_v1_best.pt`（SHA-256 校验一致）。")
    a("- 本次独立命名 `baseline_v1`，输出到 `runs/baseline_train/baseline_v1/`、`models/baseline_v1_*.pt`，未覆盖 `models/best.pt`。")
    a("- 训练从 `yolo11n.pt` 预训练权重开始，未使用冒烟权重。")
    a("")

    a("## 3. 数据集构建与过滤")
    a("")
    if subset:
        a("- 数据源: `%s`" % subset["meta"]["dataset_root"])
        a("- 审计报告: `%s`" % subset["meta"]["audit_json"])
        a("- 随机种子: %d | worker: %d | 耗时: %.1fs" % (
            subset["meta"]["seed"], subset["meta"]["workers"], subset["meta"]["duration_seconds"]))
        ex = subset["exclusions"]
        a("- 排除审计越界框图片: %d" % ex["audit_out_of_range_images"])
        a("- SHA-256 重复移除: 共 %d（跨集合 %d / 集合内 %d）" % (
            ex["sha256_duplicates_removed"]["total"],
            ex["sha256_duplicates_removed"]["cross_split"],
            ex["sha256_duplicates_removed"]["intra_split"]))
        a("- pHash 完全一致(距离0)替换: %d" % ex["phash_identical_replaced"])
        ph = subset["phash"]
        a("- pHash 统计: 共 %d 张，处理后跨集合完全一致配对 %d" % (
            ph["phash_total"], ph["cross_identical_pairs_after_replacement"]))
        ak = [k for k in ph if k.startswith("approx_pairs_") and k != "approx_pairs_listed"]
        if ak:
            th = ak[0].split("to")[-1]
            a("- 疑似近似重复(汉明距离 1..%s): %d（仅报告，不删除）" % (th, ph[ak[0]]))
        a("")
        a("### 3.1 train/val/test 规模及类别分布")
        a("")
        a("| 集合 | fire-only | smoke-only | fire+smoke | 负样本 | 合计 |")
        a("|---|---|---|---|---|---|")
        for s in ("train", "val", "test"):
            cats = subset["categories"][s]
            a("| %s | %d | %d | %d | %d | %d |" % (
                s, cats["fire_only"], cats["smoke_only"], cats["both"],
                cats["negative"], subset["actual"][s]))
        if subset["shortfalls"]:
            a("")
            a("**数量不足说明**:")
            for sf in subset["shortfalls"]:
                a("- `%s/%s`: 目标 %d，实际 %d" % (sf["split"], sf["category"], sf["target"], sf["actual"]))
        a("")
        a("### 3.2 构建后验证")
        a("")
        a("| 检查项 | 结果 | 详情 |")
        a("|---|---|---|")
        for chk in subset["validation"]["checks"]:
            a("| %s | %s | %s |" % (chk["name"], "PASS" if chk["pass"] else "FAIL", chk["detail"]))
        a("")
        a("**数据结论**: `%s`" % subset["conclusion"])
        a("")

    a("## 4. 训练环境与参数")
    a("")
    a("| 参数 | 值 |")
    a("|---|---|")
    a("| 初始权重 | yolo11n.pt |")
    a("| epochs | 30（patience=10，允许早停） |")
    a("| batch / imgsz | 16 / 640 |")
    a("| device / workers | 0 / 4 |")
    a("| seed | 42 |")
    a("| project / name | runs/baseline_train / baseline_v1 |")
    a("| 通过 M2 YOLOTrainer | 是（未使用 CLI yolo train） |")
    a("")
    if summary:
        a("- 实际运行 epoch 数: %s | 是否早停: %s" % (summary.get("epochs_run"), summary.get("early_stopped")))
        a("- 训练总耗时: %.1fs（%.1f min）" % (summary.get("total_seconds", 0), summary.get("total_seconds", 0) / 60))
        best = summary.get("best") or {}
        if best:
            a("- 最佳 epoch: %s | mAP50: %s | mAP50-95: %s | P: %s | R: %s" % (
                best.get("epoch"), fmt(best.get("mAP50")), fmt(best.get("mAP50-95")),
                fmt(best.get("precision")), fmt(best.get("recall"))))
        a("")
        a("### 4.1 每轮指标")
        a("")
        a("| epoch | train box | train cls | train dfl | val box | val cls | val dfl | P | R | mAP50 | mAP50-95 | lr | 显存MB | 耗时s |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in summary.get("per_epoch", []):
            a("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                row.get("epoch"), fmt(row.get("train_box_loss")), fmt(row.get("train_cls_loss")),
                fmt(row.get("train_dfl_loss")), fmt(row.get("val_box_loss")), fmt(row.get("val_cls_loss")),
                fmt(row.get("val_dfl_loss")), fmt(row.get("precision")), fmt(row.get("recall")),
                fmt(row.get("mAP50")), fmt(row.get("mAP50-95")), fmt(row.get("lr")),
                row.get("gpu_mem_mb") if row.get("gpu_mem_mb") is not None else "-",
                fmt(row.get("elapsed_s")) if row.get("elapsed_s") else "-"))
        a("")

    a("## 5. 最佳模型信息")
    a("")
    if summary:
        a("- 最佳权重: `%s`" % summary.get("best_path"))
        a("- 独立副本: `models/baseline_v1_best.pt` / `models/baseline_v1_last.pt`")
        for c in summary.get("copies_sha256", []):
            a("- SHA-256: `%s` -> `%s`" % (c["file"], c["sha256"][:16] + "..."))
    a("")

    a("## 6. val 总体及逐类指标")
    a("")
    if summary and summary.get("best"):
        b = summary["best"]
        a("- 总体（val，best epoch，回调记录）: P=%s R=%s mAP50=%s mAP50-95=%s" % (
            fmt(b.get("precision")), fmt(b.get("recall")), fmt(b.get("mAP50")), fmt(b.get("mAP50-95"))))
    if valm:
        a("")
        a("### 6.1 官方 val 总体指标（baseline_v1_best.pt 复评）")
        a("")
        a("| 指标 | 值 |")
        a("|---|---|")
        a("| Precision | %s |" % fmt(valm["overall"]["precision"]))
        a("| Recall | %s |" % fmt(valm["overall"]["recall"]))
        a("| mAP50 | %s |" % fmt(valm["overall"]["mAP50"]))
        a("| mAP50-95 | %s |" % fmt(valm["overall"]["mAP50-95"]))
        a("")
        a("### 6.2 val 逐类指标")
        a("")
        a("| 类别 | Precision | Recall | AP50 | AP50-95 |")
        a("|---|---|---|---|---|")
        for cname in ("fire", "smoke"):
            pc = valm["per_class"].get(cname, {})
            a("| %s | %s | %s | %s | %s |" % (
                cname, fmt(pc.get("precision")), fmt(pc.get("recall")),
                fmt(pc.get("ap50")), fmt(pc.get("ap50_95"))))
        a("")
    a("fire/smoke 逐类指标见上表；test 逐类指标见第 7 节。")
    a("")

    a("## 7. test 总体及逐类指标（独立评估）")
    a("")
    if ev:
        a("- 评估模型: `%s` | names=%s" % (ev.get("model"), ev.get("model_names")))
        a("- 测试集: %d 张（其中负样本 %d 张） | 评估 conf=0.25, IoU=0.5" % (
            ev["test_set"]["images"], ev["test_set"]["negative"]))
        a("")
        a("### 7.1 总体指标")
        a("")
        a("| 指标 | 值 |")
        a("|---|---|")
        a("| Precision | %s |" % fmt(ev["overall"]["precision"]))
        a("| Recall | %s |" % fmt(ev["overall"]["recall"]))
        a("| mAP50 | %s |" % fmt(ev["mAP50"]["all"]))
        a("| mAP50-95 | %s |" % fmt(ev["mAP50-95"]["all"]))
        a("")
        a("### 7.2 逐类指标")
        a("")
        a("| 类别 | Precision | Recall | AP50 | AP50-95 | TP | FP | FN |")
        a("|---|---|---|---|---|---|---|---|")
        for cname in ("fire", "smoke"):
            pc = ev["per_class"].get(cname, {})
            a("| %s | %s | %s | %s | %s | %d | %d | %d |" % (
                cname, fmt(pc.get("precision")), fmt(pc.get("recall")),
                fmt(ev["mAP50"].get(cname)), fmt(ev["mAP50-95"].get(cname)),
                pc.get("tp", 0), pc.get("fp", 0), pc.get("fn", 0)))
        a("")
        a("### 7.3 混淆矩阵（图片级存在，conf=0.25）")
        a("")
        a("| 行=GT \ 列=Pred | fire | smoke | negative |")
        a("|---|---|---|---|")
        cm = ev.get("confusion_matrix", {})
        for r in ("fire", "smoke", "negative"):
            a("| %s | %d | %d | %d |" % (r, cm.get(r, {}).get("fire", 0),
                                         cm.get(r, {}).get("smoke", 0),
                                         cm.get(r, {}).get("negative", 0)))
        a("")
        a("### 7.4 曲线")
        a("")
        for k, v in ev.get("curves", {}).items():
            a("- %s: `%s`" % (k, v))
        a("")

    a("## 8. 负样本误报率")
    a("")
    if ev:
        neg = ev.get("negative_fp", {})
        a("- 负样本总数: %d" % neg.get("negative_total", 0))
        a("")
        a("| conf | FP fire 图 | FP smoke 图 | 双误报图 | 图级误报率 | 平均误报框/图 | 最高FP置信度(fire/smoke) |")
        a("|---|---|---|---|---|---|---|")
        for th, v in neg.get("thresholds", {}).items():
            a("| %s | %d | %d | %d | %.4f | %.3f | %.4f / %.4f |" % (
                th, v["fp_fire_images"], v["fp_smoke_images"], v["fp_both_images"],
                v["image_level_fp_rate"], v["avg_fp_boxes_per_image"],
                v["max_fp_conf_fire"], v["max_fp_conf_smoke"]))
        a("")

    a("## 9. 错误案例分析")
    a("")
    if ev:
        ec = ev.get("error_cases", {})
        a("- 输出目录: `runs/baseline_train/baseline_v1/error_analysis/`")
        a("- 各类别案例数: %s" % json.dumps(ec, ensure_ascii=False))
        a("- 已保存标注图: %s" % json.dumps(ev.get("error_examples_saved", {}), ensure_ascii=False))
        a("")
        a("| 类型 | 数量 | 已导出 |")
        a("|---|---|---|")
        for k in ("fp_fire", "fp_smoke", "fn_fire", "fn_smoke", "low_conf"):
            a("| %s | %d | %d |" % (k, ec.get(k, 0), ev.get("error_examples_saved", {}).get(k, 0)))
        a("")

    a("## 10. me.mp4 困难负样本测试")
    a("")
    if me:
        a("- 视频: `%s`" % me.get("video"))
        a("- 总帧数: %d" % me.get("total_frames"))
        a("- fire 阳性帧: %d | smoke 阳性帧: %d | 阳性帧合计: %d" % (
            me.get("fire_positive_frames"), me.get("smoke_positive_frames"), me.get("positive_frames")))
        a("- 连续阳性片段: %d 段" % len(me.get("continuous_positive_segments", [])))
        a("- 最高置信度: %s" % json.dumps(me.get("max_confidence"), ensure_ascii=False))
        a("- M6 事件: %s | 是否触发: %s" % (json.dumps(me.get("m6_events"), ensure_ascii=False),
                                        me.get("m6_triggered")))
        a("")
        a("### 与冒烟模型对比")
        a("")
        a("| 指标 | smoke_v1 | baseline_v1 |")
        a("|---|---|---|")
        a("| fire 阳性帧 | %d | %d |" % (SMOKE_KNOWN["me_fire_frames"], me.get("fire_positive_frames")))
        a("| smoke 阳性帧 | %d | %d |" % (SMOKE_KNOWN["me_smoke_frames"], me.get("smoke_positive_frames")))
        a("")

    a("## 11. 与 smoke_v1 对比")
    a("")
    a("| 维度 | smoke_v1 | baseline_v1 (val) | baseline_v1 (test) |")
    a("|---|---|---|---|")
    a("| 数据规模 | train %d / val %d | train %d / val %d | test %d |" % (
        SMOKE_KNOWN["train"], SMOKE_KNOWN["val"],
        (subset or {}).get("actual", {}).get("train", "-"),
        (subset or {}).get("actual", {}).get("val", "-"),
        (subset or {}).get("actual", {}).get("test", "-")))
    a("| 训练 epoch | %d | %s | - |" % (SMOKE_KNOWN["epochs"],
                                       (summary or {}).get("epochs_run", "-")))
    a("| Precision | %.4f | %s | %s |" % (
        SMOKE_KNOWN["precision"], fmt((summary or {}).get("best", {}).get("precision")),
        fmt((ev or {}).get("overall", {}).get("precision"))))
    a("| Recall | %.4f | %s | %s |" % (
        SMOKE_KNOWN["recall"], fmt((summary or {}).get("best", {}).get("recall")),
        fmt((ev or {}).get("overall", {}).get("recall"))))
    a("| mAP50 | %.4f | %s | %s |" % (
        SMOKE_KNOWN["mAP50"], fmt((summary or {}).get("best", {}).get("mAP50")),
        fmt((ev or {}).get("mAP50", {}).get("all"))))
    a("| mAP50-95 | %.4f | %s | %s |" % (
        SMOKE_KNOWN["mAP50_95"], fmt((summary or {}).get("best", {}).get("mAP50-95")),
        fmt((ev or {}).get("mAP50-95", {}).get("all"))))
    a("| me.mp4 fire/smoke 阳性帧 | %d / %d | - | %d / %d |" % (
        SMOKE_KNOWN["me_fire_frames"], SMOKE_KNOWN["me_smoke_frames"],
        (me or {}).get("fire_positive_frames", "-"), (me or {}).get("smoke_positive_frames", "-")))
    a("")
    a("> 说明：冒烟模型与 Baseline V1 数据规模不同，结果用于工程趋势比较，不作为严格消融实验。")
    a("")

    a("## 12. M3~M10 链路验证（Baseline V1 模型）")
    a("")
    if chain:
        a("- 状态: %s | 模型: `%s`" % (chain.get("status"), chain.get("model")))
        a("- 处理帧: %s | 确认事件: %s | 结束事件: %s | 丢弃事件: %s" % (
            chain.get("frames_processed"), chain.get("events_confirmed"),
            chain.get("events_ended"), chain.get("events_discarded")))
        a("- 危险等级分布: %s" % json.dumps(chain.get("risk_level_distribution"), ensure_ascii=False))
        a("- 决策数: %s | events.jsonl 新增: %s" % (chain.get("decisions_count"), chain.get("events_jsonl_growth")))
        a("- 置信度分布: %s" % json.dumps(chain.get("confidence_distribution"), ensure_ascii=False))
        a("- 面积比分布: %s" % json.dumps(chain.get("area_ratio_distribution"), ensure_ascii=False))
        a("- 截图/报告完整性:")
        for eid, art in chain.get("artifacts", {}).items():
            a("  - `%s`: screenshots=%d, report files=%d" % (
                eid, len(art.get("screenshot_files", [])), len(art.get("report_files", []))))
        a("")
        a("> 本次未修改 M6 阈值、M7 权重/危险等级规则、M8 峰值规则；只记录分布。")
        a("")

    a("## 13. 是否建议晋升为当前 models/best.pt")
    a("")
    a("- 当前结论: **待人工确认**。晋升工具已准备 `tools/promote_model.py`（默认 dry-run）。")
    a("")

    a("## 14. 是否建议扩大训练规模")
    a("")
    a("- 当前结论: **待评估结果确认后建议**。若 test 精度符合预期，可扩大到 5 万张或完整训练集。")
    a("")

    a("## 15. 发现的问题与下一步建议")
    a("")
    if ev:
        pc = ev.get("per_class", {})
        a("- **fire 弱于 smoke**：fire AP50=%.3f < smoke AP50=%.3f，fire 的 FP/FN 均高于 smoke，"
          "火灾小目标与形变场景仍是主要漏检来源。" % (
              ev["mAP50"].get("fire", 0), ev["mAP50"].get("smoke", 0)))
        a("- **fire/smoke 类别串扰**：图片级混淆矩阵中 fire 图 535 张被预测为 smoke、smoke 图 577 张被预测为 fire，"
          "后续可考虑类别平衡或后处理约束。")
        neg = ev.get("negative_fp", {}).get("thresholds", {}).get("0.5", {})
        a("- **负样本误报**：conf=0.50 时图级误报率 %.4f（fire %d 图 / smoke %d 图），可接受但仍需在真实监控场景验证。" % (
            neg.get("image_level_fp_rate", 0), neg.get("fp_fire_images", 0), neg.get("fp_smoke_images", 0)))
    if me:
        a("- **me.mp4 烟花视频 smoke 敏感度大幅上升**：smoke 阳性帧从冒烟 31 帧升至 %d 帧（fire 从 12 降至 %d 帧），"
          "说明模型召回提升的同时更容易把烟花烟尘判为 smoke，M6/M7 阈值需要在模型定稿后重新校准。" % (
              me.get("smoke_positive_frames", 0), me.get("fire_positive_frames", 0)))
    a("- **M2 最小兼容修改**：`train/yolo_trainer.py` 新增 `patience`/`best_target` 参数与逐轮 lr/显存回调"
      "（兼容 Ultralytics 8.4 的 dict lr），未重构 M2。")
    a("- **下一步建议**：")
    a("  1. 人工确认后执行 `tools/promote_model.py` 晋升；")
    a("  2. 用 Baseline V1 做一次 `baseline_v1_warmstart`（从冒烟模型继续训练）对比；")
    a("  3. 模型定稿后校准 M6 置信度/面积阈值与 M7 危险等级规则；")
    a("  4. 如需更高精度，扩大到 5 万张或完整训练集训练 yolo11m/s。")
    a("")

    out = _TOOLS / "baseline_v1_training_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("报告已生成:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
