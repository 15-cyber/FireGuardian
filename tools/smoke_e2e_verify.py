"""
============================================================
FireGuardian - 冒烟训练端到端链路验证 (Smoke Train E2E Verify)
============================================================
按照《8.FireGuardian M2 首次真实冒烟训练任务.docx》第八~十一步执行：

  - M3 ImageDetector 加载 models/best.pt，验证 fire/smoke/正常图片
  - M4 VideoDetector 加载 models/best.pt，用 me.mp4 验证逐帧检测
  - M5 ValidationStreamSimulator 验证 has_fire/has_smoke/面积占比非零
  - M6 -> M7 -> M8/M9/M10 真实 Detection 端到端链路：
      FireEvent -> FireDecision -> screenshots/event_* -> events.jsonl -> reports/event_*

不修改 M1-M11 业务逻辑，不调阈值，不改原始数据。
输出: tools/smoke_e2e_verify_result.json

用法:
  python tools/smoke_e2e_verify.py
============================================================
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

# 数据集（与 datasets/fire_smoke_smoke.yaml 一致）
_DATASET_ROOT = Path(r"DATASET_ROOT")
_VAL_IMAGES = _DATASET_ROOT / "valid" / "images"
_VAL_LABELS = _DATASET_ROOT / "valid" / "labels"
_SMOKE_VAL_TXT = _ROOT / "datasets" / "smoke_val.txt"
_VIDEO = _ROOT / "me.mp4"
_M5_TMP_DIR = _ROOT / "runs" / "e2e_m5_images"
_RESULT_PATH = _ROOT / "tools" / "smoke_e2e_verify_result.json"
_EVENTS_JSONL = _ROOT / "logs" / "events.jsonl"

AGG_MIN_FIRE_AREA = 0.002
AGG_MIN_SMOKE_AREA = 0.003
AGG_CONF = 0.50


def read_cv(path):
    """兼容中文路径读取图片"""
    import cv2
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def classify_val_images():
    """按标签内容把 smoke_val.txt 中的图片分为 fire-only/smoke-only/both/negative/other"""
    cats = {"fire_only": [], "smoke_only": [], "both": [], "negative": [], "other": []}
    paths = []
    with open(_SMOKE_VAL_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(Path(line))
    for img in paths:
        lbl = _VAL_LABELS / (img.stem + ".txt")
        cls_ids = set()
        if lbl.exists():
            try:
                for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = ln.split()
                    if len(parts) >= 1:
                        try:
                            cls_ids.add(int(parts[0]))
                        except ValueError:
                            pass
            except Exception:
                pass
        if not cls_ids:
            cats["negative"].append(img)
        elif cls_ids == {0}:
            cats["fire_only"].append(img)
        elif cls_ids == {1}:
            cats["smoke_only"].append(img)
        elif cls_ids == {0, 1}:
            cats["both"].append(img)
        else:
            cats["other"].append(img)
    cats["_total"] = len(paths)
    return cats


def fill_dims(detection):
    """若 Detection 缺少图像尺寸，用 cv2 补全（模拟 GUI 生产链路的行为）"""
    if detection.image_width > 0 and detection.image_height > 0:
        return
    if detection.image_path:
        img = read_cv(Path(detection.image_path))
        if img is not None:
            h, w = img.shape[:2]
            detection.image_width = w
            detection.image_height = h


def m3_verify():
    """M3 图片检测验证"""
    print("\n" + "=" * 60)
    print("[M3] 图片检测验证")
    print("=" * 60)
    from detect.image_detector import ImageDetector

    det = ImageDetector()
    result = {
        "model_path": str(det.model_path),
        "model_name": det.model_path.name,
        "class_names": dict(det.class_names),
    }
    cats = classify_val_images()
    picks = {
        "fire": cats["fire_only"][0] if cats["fire_only"] else None,
        "smoke": cats["smoke_only"][0] if cats["smoke_only"] else None,
        "negative": cats["negative"][0] if cats["negative"] else None,
        "both": cats["both"][0] if cats["both"] else None,
    }
    result["test_images"] = {k: (str(v) if v else None) for k, v in picks.items()}
    detections = {}
    for key, img in picks.items():
        if img is None:
            detections[key] = {"status": "no_image_selected"}
            continue
        d = det.detect_single(img, save_result=False)
        detections[key] = {
            "image": str(img),
            "boxes": [
                {"class_name": b.class_name, "class_id": b.class_id, "confidence": round(float(b.confidence), 4)}
                for b in d.bboxes
            ],
            "has_fire": d.has_fire,
            "has_smoke": d.has_smoke,
            "_obj": d,
        }
    result["detections"] = detections
    return result


def m4_verify(max_frames=300):
    """M4 视频逐帧检测验证"""
    print("\n" + "=" * 60)
    print("[M4] 视频检测验证 (me.mp4, 最多 {} 帧)".format(max_frames))
    print("=" * 60)
    from video_detect.video_detector import VideoDetector

    det = VideoDetector()
    result = {
        "model_path": str(det.model_path),
        "model_name": det.model_path.name,
        "class_names": dict(det.class_names),
        "video": str(_VIDEO),
    }
    if not _VIDEO.exists():
        result["status"] = "video_missing"
        return result

    seen_classes = set()
    fire_frames = 0
    smoke_frames = 0
    no_target_frames = 0
    processed = 0
    max_conf = {"fire": 0.0, "smoke": 0.0}
    samples = []
    for frame, detection, frame_idx, total in det.decode(_VIDEO):
        processed += 1
        for b in detection.bboxes:
            cn = b.class_name
            seen_classes.add(cn)
            max_conf[cn] = max(max_conf.get(cn, 0.0), float(b.confidence))
        if detection.has_fire:
            fire_frames += 1
        if detection.has_smoke:
            smoke_frames += 1
        if not detection.bboxes:
            no_target_frames += 1
        if len(samples) < 5 and detection.bboxes:
            samples.append({
                "frame": frame_idx,
                "classes": sorted({b.class_name for b in detection.bboxes}),
                "conf": round(float(max(b.confidence for b in detection.bboxes)), 4),
            })
        if processed >= max_frames:
            break
    result.update({
        "status": "ok",
        "total_frames": processed,
        "seen_classes": sorted(seen_classes),
        "fire_frames": fire_frames,
        "smoke_frames": smoke_frames,
        "no_target_frames": no_target_frames,
        "max_conf": {k: round(v, 4) for k, v in max_conf.items()},
        "samples": samples,
    })
    return result


def _build_detect_fn(image_detector):
    """与 ui/app_controller._simulator_detector 相同的 M5 detect_fn"""
    model = image_detector.model
    conf = image_detector.conf_threshold
    iou = image_detector.iou_threshold
    device = image_detector.device

    from utils.common import BoundingBox

    def detect_fn(image):
        results = model(image, conf=conf, iou=iou, device=device, verbose=False)
        r = results[0]
        bboxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cid = int(cls_ids[i])
                bboxes.append(BoundingBox(
                    x1=float(xyxy[i][0]), y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]), y2=float(xyxy[i][3]),
                    confidence=float(confs[i]),
                    class_id=cid,
                    class_name=r.names.get(cid, "class_{}".format(cid)),
                ))
        return bboxes

    return detect_fn


def m5_verify(image_detector, max_images=12):
    """M5 验证集模拟器验证（含 has_fire/has_smoke/面积占比）"""
    print("\n" + "=" * 60)
    print("[M5] 验证集模拟器验证")
    print("=" * 60)
    from simulator.validator_simulator import ValidationStreamSimulator

    cats = classify_val_images()
    # 复制少量代表性图片到临时目录（只读原数据，不改原数据）
    if _M5_TMP_DIR.exists():
        shutil.rmtree(_M5_TMP_DIR)
    _M5_TMP_DIR.mkdir(parents=True, exist_ok=True)
    chosen = []
    per = max(1, max_images // 4)
    for prefix, cat in (("fire", "fire_only"), ("smoke", "smoke_only"),
                        ("both", "both"), ("neg", "negative")):
        for img in cats[cat][:per]:
            dst = _M5_TMP_DIR / "{}_{}".format(prefix, img.name)
            shutil.copy2(str(img), str(dst))
            chosen.append(dst)

    detect_fn = _build_detect_fn(image_detector)
    sim = ValidationStreamSimulator(
        image_dir=_M5_TMP_DIR,
        detector=detect_fn,
        interval_seconds=1.0,
        mode="sequential",
        max_frames=len(chosen),
    )
    frames = []
    nonzero_ratio = {"fire_area_ratio": 0.0, "smoke_area_ratio": 0.0}
    n_nonzero = {"fire_area_ratio": 0, "smoke_area_ratio": 0}
    positive = []
    while True:
        out = sim.next_frame()
        if out is None:
            break
        fd, det = out
        rec = {
            "frame_id": fd.frame_id,
            "image": fd.image_path,
            "has_fire": det.has_fire,
            "has_smoke": det.has_smoke,
            "fire_area_ratio": round(det.fire_area_ratio, 5),
            "smoke_area_ratio": round(det.smoke_area_ratio, 5),
            "avg_fire_conf": round(det.avg_fire_confidence, 4),
            "avg_smoke_conf": round(det.avg_smoke_confidence, 4),
            "bbox_count": len(det.bboxes),
        }
        frames.append(rec)
        if det.fire_area_ratio > 0:
            nonzero_ratio["fire_area_ratio"] = max(nonzero_ratio["fire_area_ratio"], det.fire_area_ratio)
            n_nonzero["fire_area_ratio"] += 1
        if det.smoke_area_ratio > 0:
            nonzero_ratio["smoke_area_ratio"] = max(nonzero_ratio["smoke_area_ratio"], det.smoke_area_ratio)
            n_nonzero["smoke_area_ratio"] += 1
        # 记录符合 M6 阳性条件的帧（供 M6-M10 链路使用）
        is_pos = (
            (det.has_fire and det.avg_fire_confidence >= AGG_CONF and det.fire_area_ratio >= AGG_MIN_FIRE_AREA)
            or (det.has_smoke and det.avg_smoke_confidence >= AGG_CONF and det.smoke_area_ratio >= AGG_MIN_SMOKE_AREA)
        )
        if is_pos:
            positive.append(det)

    result = {
        "model_name": image_detector.model_path.name,
        "images_copied": len(chosen),
        "frames_processed": len(frames),
        "nonzero_ratio_max": {k: round(v, 5) for k, v in nonzero_ratio.items()},
        "nonzero_ratio_count": n_nonzero,
        "positive_frames_for_m6": len(positive),
        "frames": frames,
        "positive_detections": positive,
    }
    return result


def m6_to_m10_verify(m5_result, m3_result):
    """M6 -> M7 -> M8/M9/M10 端到端链路"""
    print("\n" + "=" * 60)
    print("[M6-M10] 真实检测端到端链路验证")
    print("=" * 60)
    from aggregator.event_aggregator import EventAggregator
    from agent.fire_decision_agent import FireDecisionAgent
    from utils.screenshot import ScreenshotManager
    from reports.report_generator import ReportGenerator
    from utils.common import Detection, EventReportData
    from logs.logger import log_event, log_decision, flush as log_flush

    agg = EventAggregator()
    shot = ScreenshotManager()
    agent = FireDecisionAgent()
    report_gen = ReportGenerator()
    decisions = {}

    def analyze(evt):
        decision = agent.analyze(evt)
        decisions.setdefault(evt.event_id, []).append(decision)
        try:
            shot.on_decision(evt, decision)
        except Exception as e:
            print("  [warn] M8 on_decision: {}".format(e))
        try:
            log_decision(evt, decision, module="fire_decision_agent", source_module="M7")
        except Exception as e:
            print("  [warn] M9 log_decision: {}".format(e))
        return decision

    def on_confirmed(evt):
        try:
            shot.on_event_confirmed(evt)
        except Exception as e:
            print("  [warn] M8 confirmed: {}".format(e))
        analyze(evt)
        try:
            log_event(evt, "event_confirmed", module="event_aggregator", source_module="M6")
        except Exception:
            pass

    def on_updated(evt):
        try:
            shot.on_event_updated(evt)
        except Exception as e:
            print("  [warn] M8 updated: {}".format(e))
        if evt.metadata.get("decision_required"):
            evt.metadata["decision_required"] = False
            analyze(evt)
        try:
            log_event(evt, "event_updated", module="event_aggregator", source_module="M6")
        except Exception:
            pass

    def on_ended(evt):
        analyze(evt)  # 最终决策
        try:
            shot.on_event_ended(evt)
        except Exception as e:
            print("  [warn] M8 ended: {}".format(e))
        try:
            shots = shot.get_event_shot_records(evt.event_id)
            data = EventReportData(event=evt, decisions=decisions.get(evt.event_id, []), screenshots=shots)
            report_gen.generate(data)
        except Exception as e:
            print("  [warn] M10 report: {}".format(e))
        try:
            log_event(evt, "event_ended", module="event_aggregator", source_module="M6")
        except Exception:
            pass

    agg.on_event_confirmed = on_confirmed
    agg.on_event_updated = on_updated
    agg.on_event_ended = on_ended
    agg.on_event_discarded = lambda evt: log_event(evt, "event_discarded",
                                                   module="event_aggregator", source_module="M6")

    # 收集真实阳性 Detection（优先 M5 已带尺寸的帧，不足时用 M3 结果并补尺寸）
    positives = list(m5_result.get("positive_detections", []))
    if len(positives) < 3:
        for key in ("fire", "smoke", "both"):
            d = m3_result.get(key)
            if d and d.get("boxes"):
                det = d["_obj"]
                fill_dims(det)
                det.timestamp = 0.0
                positives.append(det)
    if not positives:
        print("  [fail] 没有真实阳性 Detection 可用，跳过 M6-M10 链路验证")
        return {"status": "skipped_no_positive"}

    # 阴性帧（真实负样本图片，空检测框）
    negative_img = None
    cats = classify_val_images()
    if cats["negative"]:
        negative_img = cats["negative"][0]
    if negative_img is None and m5_result.get("frames"):
        negative_img = Path(m5_result["frames"][0]["image"])

    feed = []
    for i, det in enumerate(positives[:4]):
        det.timestamp = float(i)
        det.frame_id = i + 1
        feed.append(det)
    # 连续 3 帧阴性结束事件（max_negative_frames=3）
    base = len(positives[:4])
    for j in range(3):
        neg = Detection(
            bboxes=[],
            image_path=str(negative_img) if negative_img else "",
            timestamp=float(base + j),
            frame_id=base + j + 1,
        )
        if negative_img:
            img = read_cv(negative_img)
            if img is not None:
                h, w = img.shape[:2]
                neg.image_width = w
                neg.image_height = h
        feed.append(neg)

    before_lines = 0
    if _EVENTS_JSONL.exists():
        before_lines = sum(1 for _ in open(_EVENTS_JSONL, encoding="utf-8"))

    for det in feed:
        try:
            agg.process(det)
        except Exception as e:
            print("  [warn] M6 process: {}".format(e))

    if _EVENTS_JSONL.exists():
        after_lines = sum(1 for _ in open(_EVENTS_JSONL, encoding="utf-8"))
    else:
        after_lines = 0

    result = {
        "status": "ok",
        "positive_frames_fed": len(positives[:4]),
        "negative_frames_fed": 3,
        "events_confirmed": sum(1 for e in agg.event_history if e.status.value == "confirmed")
        + (1 if agg.current_event and agg.current_event.status.value == "confirmed" else 0),
        "events_ended": sum(1 for e in agg.event_history if e.status.value == "ended"),
        "events_discarded": sum(1 for e in agg.event_history if e.status.value == "discarded"),
        "decisions": {k: [d.to_dict() for d in v] for k, v in decisions.items()},
        "events_jsonl_lines_before": before_lines,
        "events_jsonl_lines_after": after_lines,
        "events_jsonl_growth": max(0, after_lines - before_lines),
    }

    # 验证产物
    for evt in agg.event_history:
        if evt.status.value != "ended":
            continue
        shot_dir = shot.path_mgr.screenshot_event_dir(evt.event_id)
        rpt_dir = report_gen.path_mgr.report_event_dir(evt.event_id)
        result.setdefault("artifacts", {})[evt.event_id] = {
            "screenshot_dir": str(shot_dir),
            "screenshot_files": sorted(p.name for p in shot_dir.glob("*")) if shot_dir.exists() else [],
            "report_dir": str(rpt_dir),
            "report_files": sorted(p.name for p in rpt_dir.glob("*")) if rpt_dir.exists() else [],
        }
    log_flush()
    return result


def main():
    t0 = time.time()
    print("FireGuardian 冒烟训练端到端链路验证")
    print("数据集目录: {}".format(_DATASET_ROOT))
    print("验证集子集: {}".format(_SMOKE_VAL_TXT))
    print("视频素材: {}".format(_VIDEO))

    result = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 数据分类
    try:
        cats = classify_val_images()
        result["val_subset"] = {
            "total": cats.pop("_total"),
            "fire_only": len(cats["fire_only"]),
            "smoke_only": len(cats["smoke_only"]),
            "both": len(cats["both"]),
            "negative": len(cats["negative"]),
            "other": len(cats["other"]),
        }
    except Exception as e:
        result["val_subset"] = {"error": str(e)}

    # M3
    try:
        m3 = m3_verify()
        result["m3"] = {k: v for k, v in m3.items() if k != "detections"}
        result["m3"]["detections"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "_obj"}
            for k, v in m3["detections"].items()
        }
        result["_m3_objs"] = m3["detections"]
        print("  [M3] 模型: {} | names: {}".format(m3["model_name"], m3["class_names"]))
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["m3"] = {"error": str(e)}

    # M4
    try:
        m4 = m4_verify()
        result["m4"] = m4
        print("  [M4] 模型: {} | 检测类别: {} | fire帧 {} | smoke帧 {}".format(
            m4.get("model_name"), m4.get("seen_classes"),
            m4.get("fire_frames"), m4.get("smoke_frames")))
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["m4"] = {"error": str(e)}

    # M5
    m5 = None
    try:
        if "error" not in result.get("m3", {}):
            from detect.image_detector import ImageDetector
            det = ImageDetector()
            m5 = m5_verify(det)
            result["m5"] = {k: v for k, v in m5.items() if k != "frames" and k != "positive_detections"}
            result["m5"]["frames"] = m5["frames"]
            print("  [M5] 处理帧 {} | 非零面积占比: {} | M6阳性帧 {}".format(
                m5["frames_processed"], m5["nonzero_ratio_max"], m5["positive_frames_for_m6"]))
        else:
            result["m5"] = {"error": "m3 失败，跳过"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["m5"] = {"error": str(e)}

    # M6-M10
    try:
        if m5 is not None:
            m6 = m6_to_m10_verify(m5, result["_m3_objs"])
            result["m6_m10"] = {k: v for k, v in m6.items()}
            print("  [M6-M10] 状态: {} | 已结束事件: {} | 决策数: {} | events.jsonl 新增: {}".format(
                m6.get("status"), m6.get("events_ended"),
                sum(len(v) for v in m6.get("decisions", {}).values()),
                m6.get("events_jsonl_growth")))
            for eid, art in m6.get("artifacts", {}).items():
                print("    - {}: screenshots={} 报告={}".format(
                    eid, len(art["screenshot_files"]), art["report_files"]))
        else:
            result["m6_m10"] = {"error": "m5 失败，跳过"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["m6_m10"] = {"error": str(e)}

    result["_m3_objs"] = None  # 不序列化对象
    result["elapsed_seconds"] = round(time.time() - t0, 1)

    with open(_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n[保存] 验证结果: {}".format(_RESULT_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
