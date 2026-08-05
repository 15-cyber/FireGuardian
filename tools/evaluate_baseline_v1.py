"""
============================================================
FireGuardian Baseline V1 - 独立 Test 集精度评估
============================================================
按《9.FireGuardian Baseline V1 中等规模训练与精度评估任务.docx》第十~十二节执行：
- 显式加载 models/baseline_v1_best.pt（不替换 models/best.pt）
- 对 baseline_v1_test.txt 做总体 + fire/smoke 逐类评估
- 混淆矩阵 / PR / F1 / Precision / Recall 曲线
- 负样本误报专项（conf=0.25 / 0.40 / 0.50）
- 错误案例导出 error_analysis/（FP/FP/FN/low_conf 各最多 50 张）

输出:
- tools/baseline_v1_evaluation.json
- runs/baseline_train/baseline_v1/eval_curves/*.png
- runs/baseline_train/baseline_v1/error_analysis/{fp_fire,fp_smoke,fn_fire,fn_smoke,low_confidence_correct}/ + json/md

用法:
  python tools/evaluate_baseline_v1.py [--model models/baseline_v1_best.pt] [--imgsz 640]
============================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

CLASS_NAMES = {0: "fire", 1: "smoke"}
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.5..0.95
EVAL_CONF = 0.25
NEGATIVE_CONFS = [0.25, 0.40, 0.50]
MAX_ERROR_EXAMPLES = 50
LOW_CONF_BAND = (0.25, 0.45)


def read_cv(path):
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_cv(path, img):
    import cv2
    ext = os.path.splitext(str(path))[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
        return True
    return False


def parse_label(lbl_path):
    """返回 [(class_id, cx, cy, w, h)] 归一化坐标"""
    boxes = []
    if not lbl_path.exists():
        return boxes
    try:
        lines = lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return boxes
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = [float(x) for x in parts[1:5]]
        except ValueError:
            continue
        if cid not in CLASS_NAMES:
            continue
        boxes.append((cid, cx, cy, w, h))
    return boxes


def load_test_set(test_txt):
    images = []
    with open(test_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img = Path(line)
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            images.append({"image": img, "label": lbl, "gt": parse_label(lbl)})
    return images


def run_predictions(model, images, conf=0.001, iou=0.5, imgsz=640, device=0, batch=16):
    """分块 + stream 批量预测，避免显存/内存峰值；返回每图 [(class_id,x1,y1,x2,y2,conf)]"""
    import torch
    preds = []
    CHUNK = 500
    for start in range(0, len(images), CHUNK):
        chunk = images[start:start + CHUNK]
        paths = [str(r["image"]) for r in chunk]
        results = model.predict(
            source=paths, conf=conf, iou=iou, imgsz=imgsz, device=device,
            verbose=False, batch=batch, agnostic_nms=False, stream=True,
        )
        for res in results:
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    cid = int(cls[i])
                    if cid not in CLASS_NAMES:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    boxes.append((cid, x1, y1, x2, y2, float(confs[i])))
            preds.append(boxes)
        torch.cuda.empty_cache()
    return preds


def box_iou(a, b):
    """a: (x1,y1,x2,y2), b: (x1,y1,x2,y2) -> IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def xywh_to_xyxy(boxes, img_w, img_h):
    out = []
    for (cid, cx, cy, w, h) in boxes:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append((cid, x1, y1, x2, y2))
    return out


def match_predictions(gt_boxes, pred_boxes, iou_thr):
    """贪心匹配。返回 (tp_flags, fp_flags, fn_count)
    gt_boxes: list of (cid, x1,y1,x2,y2) 像素
    pred_boxes: list of (cid, x1,y1,x2,y2, conf) 像素
    """
    gt = list(gt_boxes)
    matched_gt = set()
    tp = [False] * len(pred_boxes)
    for i, pb in enumerate(pred_boxes):
        cid = pb[0]
        best_iou, best_gi = iou_thr, -1
        for gi, gb in enumerate(gt):
            if gi in matched_gt or gb[0] != cid:
                continue
            iou = box_iou(pb[1:5], gb[1:5])
            if iou >= best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            tp[i] = True
    fn_count = len(gt) - len(matched_gt)
    fp_flags = [not t for t in tp]
    return tp, fp_flags, fn_count


def compute_ap(confidences, tp_flags, n_gt):
    """101-point 插值 AP"""
    if n_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(confidences, dtype=np.float64), kind="stable")
    tp = np.asarray(tp_flags, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / (tp_c + fp_c + 1e-16)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(np.diff(mrec) != 0)[0]
    ap = 0.0
    for i in idx:
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(ap)


def per_class_metrics(images, preds, conf, iou_thr, img_dims):
    """返回 {class_id: {ap...}}, 以及总体 mAP"""
    class_aps = {c: [] for c in CLASS_NAMES}
    class_pr = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    class_curves = {c: {"confs": [], "precisions": [], "recalls": [], "f1s": []} for c in CLASS_NAMES}
    curve_thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2).tolist()

    for img, pboxes, dims in zip(images, preds, img_dims):
        w, h = dims
        gt = xywh_to_xyxy(img["gt"], w, h)
        # 全量预测（conf>=conf 已经过滤）；这里用传入的低置信预测
        for cid in CLASS_NAMES:
            gt_c = [g for g in gt if g[0] == cid]
            pred_c = [p for p in pboxes if p[0] == cid and p[5] >= conf]
            tp, fp_flags, fn = match_predictions(gt_c, pred_c, iou_thr)
            class_pr[cid]["tp"] += sum(tp)
            class_pr[cid]["fp"] += sum(fp_flags)
            class_pr[cid]["fn"] += fn
        # 曲线数据（所有预测，不按阈值过滤，保留 conf 排序）
        for cid in CLASS_NAMES:
            gt_c = [g for g in gt if g[0] == cid]
            pred_c = [p for p in pboxes if p[0] == cid]
            tp, fp_flags, _ = match_predictions(gt_c, pred_c, iou_thr)
            confs = [p[5] for p in pred_c]
            n_gt = len(gt_c)
            # 在各阈值下求 P/R
            for th in curve_thresholds:
                sel = [i for i, c in enumerate(confs) if c >= th]
                if not sel:
                    continue
                tps = sum(1 for i in sel if tp[i])
                fps = len(sel) - tps
                prec = tps / (tps + fps) if (tps + fps) else 0.0
                rec = tps / n_gt if n_gt else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                class_curves[cid]["confs"].append(th)
                class_curves[cid]["precisions"].append(prec)
                class_curves[cid]["recalls"].append(rec)
                class_curves[cid]["f1s"].append(f1)

    # AP 计算（用 conf>=0.001 的全部预测）
    all_scores = {c: {"conf": [], "tp": [], "n_gt": 0} for c in CLASS_NAMES}
    for img, pboxes, dims in zip(images, preds, img_dims):
        w, h = dims
        gt = xywh_to_xyxy(img["gt"], w, h)
        for cid in CLASS_NAMES:
            gt_c = [g for g in gt if g[0] == cid]
            pred_c = [p for p in pboxes if p[0] == cid]
            tp, _, _ = match_predictions(gt_c, pred_c, iou_thr)
            all_scores[cid]["conf"].extend([p[5] for p in pred_c])
            all_scores[cid]["tp"].extend(tp)
            all_scores[cid]["n_gt"] += len(gt_c)
    for cid in CLASS_NAMES:
        ap = compute_ap(all_scores[cid]["conf"], all_scores[cid]["tp"], all_scores[cid]["n_gt"])
        class_aps[cid].append(ap)

    return class_aps, class_pr, class_curves


def compute_all_metrics(images, preds, img_dims):
    """整体评估：mAP50、mAP50-95、总体/逐类 P/R(conf=0.25, iou=0.5)"""
    result = {}
    # 逐 IoU 阈值 AP
    map50_list = {c: [] for c in CLASS_NAMES}
    map50_95_list = {c: [] for c in CLASS_NAMES}
    for iou_thr in IOU_THRESHOLDS:
        class_aps, _, _ = per_class_metrics(images, preds, EVAL_CONF, iou_thr, img_dims)
        for cid, aps in class_aps.items():
            if iou_thr == 0.5:
                map50_list[cid] = aps
            map50_95_list[cid].extend(aps)
    result["mAP50"] = {CLASS_NAMES[c]: float(np.mean(map50_list[c])) for c in CLASS_NAMES}
    result["mAP50"] ["all"] = float(np.mean(list(result["mAP50"].values())))
    result["mAP50-95"] = {CLASS_NAMES[c]: float(np.mean(map50_95_list[c])) for c in CLASS_NAMES}
    result["mAP50-95"]["all"] = float(np.mean(list(result["mAP50-95"].values())))

    # 总体/逐类 P/R 在 conf=0.25, iou=0.5
    _, class_pr, class_curves = per_class_metrics(images, preds, EVAL_CONF, 0.5, img_dims)
    result["per_class"] = {}
    for cid, name in CLASS_NAMES.items():
        pr = class_pr[cid]
        p = pr["tp"] / (pr["tp"] + pr["fp"]) if (pr["tp"] + pr["fp"]) else 0.0
        r = pr["tp"] / (pr["tp"] + pr["fn"]) if (pr["tp"] + pr["fn"]) else 0.0
        result["per_class"][name] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "tp": pr["tp"], "fp": pr["fp"], "fn": pr["fn"],
        }
    tot_tp = sum(class_pr[c]["tp"] for c in CLASS_NAMES)
    tot_fp = sum(class_pr[c]["fp"] for c in CLASS_NAMES)
    tot_fn = sum(class_pr[c]["fn"] for c in CLASS_NAMES)
    result["overall"] = {
        "precision": round(tot_tp / (tot_tp + tot_fp), 4) if (tot_tp + tot_fp) else 0.0,
        "recall": round(tot_tp / (tot_tp + tot_fn), 4) if (tot_tp + tot_fn) else 0.0,
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
    }
    result["curves"] = {
        name: {"conf": class_curves[c]["confs"],
               "precision": [round(v, 4) for v in class_curves[c]["precisions"]],
               "recall": [round(v, 4) for v in class_curves[c]["recalls"]],
               "f1": [round(v, 4) for v in class_curves[c]["f1s"]]}
        for c, name in CLASS_NAMES.items()
    }
    return result


def image_presence_confusion(images, preds, conf):
    """图片级存在混淆矩阵：行 GT(fire/smoke/negative)，列 Pred(fire/smoke/negative)"""
    mat = {g: {p: 0 for p in ("fire", "smoke", "negative")} for g in ("fire", "smoke", "negative")}
    for img, pboxes in zip(images, preds):
        gt_set = {CLASS_NAMES[c] for c, *_ in img["gt"]}
        pred_set = {CLASS_NAMES[p[0]] for p in pboxes if p[5] >= conf}
        gt_row = "negative" if not gt_set else ("fire" if gt_set == {"fire"} else "smoke" if gt_set == {"smoke"} else "both")
        pred_col = "negative" if not pred_set else ("fire" if pred_set == {"fire"} else "smoke" if pred_set == {"smoke"} else "both")
        # both 行/列展开到对应类
        row_keys = ["fire", "smoke"] if gt_row == "both" else [gt_row]
        col_keys = ["fire", "smoke"] if pred_col == "both" else [pred_col]
        for rk in row_keys:
            for ck in col_keys:
                mat[rk][ck] += 1
    return mat


def negative_fp_evaluation(images, preds, img_dims):
    """负样本误报专项（十一节）"""
    neg_images = [i for i in images if not i["gt"]]
    neg_preds = [p for p, i in zip(preds, images) if not i["gt"]]
    result = {"negative_total": len(neg_images), "thresholds": {}}
    for conf in NEGATIVE_CONFS:
        fp_fire_imgs = 0
        fp_smoke_imgs = 0
        both_imgs = 0
        total_fp_boxes = 0
        max_conf_fire = 0.0
        max_conf_smoke = 0.0
        for pboxes in neg_preds:
            fire_boxes = [p for p in pboxes if p[0] == 0 and p[5] >= conf]
            smoke_boxes = [p for p in pboxes if p[0] == 1 and p[5] >= conf]
            if fire_boxes:
                fp_fire_imgs += 1
                max_conf_fire = max(max_conf_fire, max(b[5] for b in fire_boxes))
            if smoke_boxes:
                fp_smoke_imgs += 1
                max_conf_smoke = max(max_conf_smoke, max(b[5] for b in smoke_boxes))
            if fire_boxes and smoke_boxes:
                both_imgs += 1
            total_fp_boxes += len(fire_boxes) + len(smoke_boxes)
        n = len(neg_images) if neg_images else 0
        result["thresholds"][str(conf)] = {
            "fp_fire_images": fp_fire_imgs,
            "fp_smoke_images": fp_smoke_imgs,
            "fp_both_images": both_imgs,
            "fp_fire_rate": round(fp_fire_imgs / n, 4) if n else 0.0,
            "fp_smoke_rate": round(fp_smoke_imgs / n, 4) if n else 0.0,
            "image_level_fp_rate": round((fp_fire_imgs + fp_smoke_imgs - both_imgs) / n, 4) if n else 0.0,
            "avg_fp_boxes_per_image": round(total_fp_boxes / n, 3) if n else 0.0,
            "max_fp_conf_fire": round(max_conf_fire, 4),
            "max_fp_conf_smoke": round(max_conf_smoke, 4),
        }
    return result


def collect_error_cases(images, preds, img_dims):
    """收集错误案例（十二节）"""
    cases = {
        "fp_fire": [], "fp_smoke": [], "fn_fire": [], "fn_smoke": [], "low_conf": [],
    }
    for img, pboxes, dims in zip(images, preds, img_dims):
        w, h = dims
        gt = xywh_to_xyxy(img["gt"], w, h)
        gt_set = {CLASS_NAMES[c] for c, *_ in img["gt"]}
        pred_conf = [p for p in pboxes if p[5] >= EVAL_CONF]
        pred_set = {CLASS_NAMES[p[0]] for p in pred_conf}
        # FP fire: 预测 fire 但 GT 无 fire
        if "fire" in pred_set and "fire" not in gt_set:
            fp_fire = [p for p in pred_conf if p[0] == 0]
            cases["fp_fire"].append({
                "image": str(img["image"]), "gt_classes": sorted(gt_set),
                "pred_classes": sorted(pred_set),
                "max_conf": round(max(p[5] for p in fp_fire), 4),
                "boxes": [{"class": CLASS_NAMES[p[0]], "conf": round(p[5], 4),
                           "xyxy": [round(v, 1) for v in p[1:5]]} for p in fp_fire],
                "gt_boxes": [{"class": CLASS_NAMES[g[0]], "xyxy": [round(v, 1) for v in g[1:5]]} for g in gt],
            })
        if "smoke" in pred_set and "smoke" not in gt_set:
            fp_smoke = [p for p in pred_conf if p[0] == 1]
            cases["fp_smoke"].append({
                "image": str(img["image"]), "gt_classes": sorted(gt_set),
                "pred_classes": sorted(pred_set),
                "max_conf": round(max(p[5] for p in fp_smoke), 4),
                "boxes": [{"class": CLASS_NAMES[p[0]], "conf": round(p[5], 4),
                           "xyxy": [round(v, 1) for v in p[1:5]]} for p in fp_smoke],
                "gt_boxes": [{"class": CLASS_NAMES[g[0]], "xyxy": [round(v, 1) for v in g[1:5]]} for g in gt],
            })
        # FN fire: GT 有 fire 但预测无 fire
        if "fire" in gt_set and "fire" not in pred_set:
            cases["fn_fire"].append({
                "image": str(img["image"]), "gt_classes": sorted(gt_set),
                "pred_classes": sorted(pred_set), "max_conf": 0.0,
                "boxes": [], "gt_boxes": [{"class": CLASS_NAMES[g[0]],
                                           "xyxy": [round(v, 1) for v in g[1:5]]} for g in gt],
            })
        if "smoke" in gt_set and "smoke" not in pred_set:
            cases["fn_smoke"].append({
                "image": str(img["image"]), "gt_classes": sorted(gt_set),
                "pred_classes": sorted(pred_set), "max_conf": 0.0,
                "boxes": [], "gt_boxes": [{"class": CLASS_NAMES[g[0]],
                                           "xyxy": [round(v, 1) for v in g[1:5]]} for g in gt],
            })
        # low-confidence correct: 有正确匹配框且置信度在低带内
        for cid in CLASS_NAMES:
            gt_c = [g for g in gt if g[0] == cid]
            pred_c = [p for p in pboxes if p[0] == cid]
            tp, _, _ = match_predictions(gt_c, pred_c, 0.5)
            low = [p for i, p in enumerate(pred_c) if tp[i] and LOW_CONF_BAND[0] <= p[5] < LOW_CONF_BAND[1]]
            if low:
                cases["low_conf"].append({
                    "image": str(img["image"]), "gt_classes": sorted(gt_set),
                    "pred_classes": sorted(pred_set),
                    "max_conf": round(max(p[5] for p in low), 4),
                    "boxes": [{"class": CLASS_NAMES[p[0]], "conf": round(p[5], 4),
                               "xyxy": [round(v, 1) for v in p[1:5]]} for p in low],
                    "gt_boxes": [{"class": CLASS_NAMES[g[0]], "xyxy": [round(v, 1) for v in g[1:5]]} for g in gt],
                })
                break
    for key in cases:
        if key.startswith("fp"):
            cases[key].sort(key=lambda r: -r["max_conf"])
        elif key == "low_conf":
            cases[key].sort(key=lambda r: r["max_conf"])
    return cases


def export_error_images(cases, out_dir):
    """保存标注图（不修改原图）"""
    import cv2
    saved = {}
    for key, records in cases.items():
        folder = out_dir / key
        folder.mkdir(parents=True, exist_ok=True)
        count = 0
        for rec in records[:MAX_ERROR_EXAMPLES]:
            img = read_cv(Path(rec["image"]))
            if img is None:
                continue
            for gb in rec.get("gt_boxes", []):
                x1, y1, x2, y2 = [int(v) for v in gb["xyxy"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "GT " + gb["class"], (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for pb in rec.get("boxes", []):
                x1, y1, x2, y2 = [int(v) for v in pb["xyxy"]]
                color = (0, 0, 255) if pb["class"] == "fire" else (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, "{} {:.2f}".format(pb["class"], pb["conf"]),
                            (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            dst = folder / "{:03d}_{}.jpg".format(count, Path(rec["image"]).stem)
            if write_cv(dst, img):
                count += 1
        saved[key] = count
    return saved


def plot_curves(metrics, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = metrics["curves"]
    colors = {"fire": "#d62728", "smoke": "#1f77b4"}
    # PR 曲线
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, name_cn in (("fire", "fire"), ("smoke", "smoke")):
        ax.plot(curves[name]["recall"], curves[name]["precision"],
                color=colors[name_cn], label=name_cn)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Curve (IoU=0.5, baseline_v1 test)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150); plt.close(fig)
    # F1 曲线
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, name_cn in (("fire", "fire"), ("smoke", "smoke")):
        ax.plot(curves[name]["conf"], curves[name]["f1"], color=colors[name_cn], label=name_cn)
    ax.set_xlabel("Confidence"); ax.set_ylabel("F1 Score"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("F1 vs Confidence (IoU=0.5, baseline_v1 test)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "f1_curve.png", dpi=150); plt.close(fig)
    # Precision 曲线
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, name_cn in (("fire", "fire"), ("smoke", "smoke")):
        ax.plot(curves[name]["conf"], curves[name]["precision"], color=colors[name_cn], label=name_cn)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("Precision vs Confidence (baseline_v1 test)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "precision_curve.png", dpi=150); plt.close(fig)
    # Recall 曲线
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, name_cn in (("fire", "fire"), ("smoke", "smoke")):
        ax.plot(curves[name]["conf"], curves[name]["recall"], color=colors[name_cn], label=name_cn)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("Recall vs Confidence (baseline_v1 test)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "recall_curve.png", dpi=150); plt.close(fig)
    return {f.name: str(f) for f in out_dir.glob("*.png")}


def write_error_md(cases, path):
    lines = ["# Baseline V1 错误案例分析", ""]
    for key, label in (("fp_fire", "False Positive - fire"),
                       ("fp_smoke", "False Positive - smoke"),
                       ("fn_fire", "False Negative - fire"),
                       ("fn_smoke", "False Negative - smoke"),
                       ("low_conf", "Low Confidence Correct")):
        lines.append("## %s (%d)" % (label, len(cases[key])))
        lines.append("")
        for rec in cases[key][:20]:
            lines.append("- %s | GT=%s | Pred=%s | max_conf=%.3f" % (
                rec["image"], ",".join(rec["gt_classes"]) or "-",
                ",".join(rec["pred_classes"]) or "-", rec["max_conf"]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(_ROOT / "models" / "baseline_v1_best.pt"))
    ap.add_argument("--test-txt", default=str(_ROOT / "datasets" / "baseline_v1_test.txt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    model_path = Path(args.model)
    if not model_path.exists():
        print("[错误] 模型不存在:", model_path)
        return 1
    from ultralytics import YOLO
    print("[加载] 模型:", model_path)
    model = YOLO(str(model_path))
    print("[加载] names:", model.names)

    images = load_test_set(Path(args.test_txt))
    print("[数据] test 图片:", len(images))
    n_neg = sum(1 for i in images if not i["gt"])
    print("[数据] 其中负样本:", n_neg)

    print("[预测] 全量推理 conf=0.001 ...")
    preds = run_predictions(model, images, conf=0.001, iou=0.5, imgsz=args.imgsz,
                            device=args.device, batch=args.batch)
    img_dims = []
    for i, r in enumerate(images):
        img = read_cv(r["image"])
        if img is None:
            print("[warn] 无法读取图片:", r["image"])
            img_dims.append((640, 640))
        else:
            h, w = img.shape[:2]
            img_dims.append((w, h))
    print("[预测] 完成，总框数:", sum(len(p) for p in preds))

    print("[指标] 计算 mAP / P / R ...")
    metrics = compute_all_metrics(images, preds, img_dims)
    print("  总体 mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f" % (
        metrics["mAP50"]["all"], metrics["mAP50-95"]["all"],
        metrics["overall"]["precision"], metrics["overall"]["recall"]))

    print("[混淆矩阵] 图片级存在混淆矩阵 (conf=0.25)")
    cm = image_presence_confusion(images, preds, EVAL_CONF)

    print("[负样本] 误报专项")
    neg_eval = negative_fp_evaluation(images, preds, img_dims)
    for th, v in neg_eval["thresholds"].items():
        print("  conf=%s: 误报图 fire=%d smoke=%d 图级误报率=%.4f" % (
            th, v["fp_fire_images"], v["fp_smoke_images"], v["image_level_fp_rate"]))

    print("[错误案例] 收集与导出")
    cases = collect_error_cases(images, preds, img_dims)
    for k, v in cases.items():
        print("  %s: %d" % (k, len(v)))
    base_dir = _ROOT / "runs" / "baseline_train" / "baseline_v1"
    err_dir = base_dir / "error_analysis"
    saved = export_error_images(cases, err_dir)
    print("[错误案例] 已保存标注图:", saved)

    print("[曲线] 绘制 PR/F1/Precision/Recall")
    curve_dir = base_dir / "eval_curves"
    curve_files = plot_curves(metrics, curve_dir)

    write_error_md(cases, err_dir / "error_analysis.md")
    eval_result = {
        "model": str(model_path),
        "model_names": {int(k): v for k, v in model.names.items()},
        "test_set": {"images": len(images), "negative": n_neg},
        "eval_conf": EVAL_CONF,
        "overall": metrics["overall"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "per_class": metrics["per_class"],
        "confusion_matrix": cm,
        "negative_fp": neg_eval,
        "error_cases": {k: len(v) for k, v in cases.items()},
        "error_examples_saved": saved,
        "curves": curve_files,
        "duration_seconds": round(time.time() - t0, 1),
    }
    out_json = _ROOT / "tools" / "baseline_v1_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print("[保存] 评估结果:", out_json)
    print("[完成] 总耗时: %.1fs" % eval_result["duration_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
