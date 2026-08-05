"""
============================================================
FireGuardian Baseline V1 - M6~M10 链路验证（任务书十六节）
============================================================
- 显式加载 models/baseline_v1_best.pt（不替换 models/best.pt）
- 从 baseline_v1_val.txt 抽取 fire/smoke/both/negative 代表图
- M5 模拟器 -> M6 EventAggregator -> M7 FireDecisionAgent
  -> M8 ScreenshotManager -> M9 日志 -> M10 ReportGenerator
- 只记录分布，不修改任何业务阈值/权重/规则

输出: tools/baseline_v1_chain_verify_result.json
============================================================
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

_MODEL = _ROOT / "models" / "baseline_v1_best.pt"
_VAL_TXT = _ROOT / "datasets" / "baseline_v1_val.txt"
_TMP_DIR = _ROOT / "runs" / "baseline_chain_m5_images"
_RESULT = _ROOT / "tools" / "baseline_v1_chain_verify_result.json"
_EVENTS_JSONL = _ROOT / "logs" / "events.jsonl"


def read_cv(path):
    import cv2
    import numpy as np
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def classify_images():
    cats = {"fire_only": [], "smoke_only": [], "both": [], "negative": []}
    paths = []
    with open(_VAL_TXT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(Path(line))
    for img in paths:
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
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
    return cats, len(paths)


def build_detect_fn(image_detector):
    from utils.common import BoundingBox
    model = image_detector.model
    conf = image_detector.conf_threshold
    iou = image_detector.iou_threshold
    device = image_detector.device

    def detect_fn(image):
        results = model(image, conf=conf, iou=iou, device=device, verbose=False)
        r = results[0]
        bboxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cid = int(cls[i])
                bboxes.append(BoundingBox(
                    x1=float(xyxy[i][0]), y1=float(xyxy[i][1]),
                    x2=float(xyxy[i][2]), y2=float(xyxy[i][3]),
                    confidence=float(confs[i]), class_id=cid,
                    class_name=r.names.get(cid, "class_%d" % cid),
                ))
        return bboxes

    return detect_fn


def run_chain(image_detector, max_images=16):
    """M5 模拟器 + M6-M10 链路"""
    from simulator.validator_simulator import ValidationStreamSimulator
    from aggregator.event_aggregator import EventAggregator
    from agent.fire_decision_agent import FireDecisionAgent
    from utils.screenshot import ScreenshotManager
    from reports.report_generator import ReportGenerator
    from utils.common import Detection, EventReportData
    from logs.logger import log_event, log_decision, flush as log_flush

    cats, total = classify_images()
    if _TMP_DIR.exists():
        shutil.rmtree(_TMP_DIR)
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    chosen = []
    per = max(1, max_images // 4)
    for prefix, cat in (("fire", "fire_only"), ("smoke", "smoke_only"),
                        ("both", "both"), ("neg", "negative")):
        for img in cats[cat][:per]:
            dst = _TMP_DIR / ("%s_%s" % (prefix, img.name))
            shutil.copy2(str(img), str(dst))
            chosen.append(dst)
    if len(chosen) < 4:
        return {"status": "skipped_not_enough_images"}

    detect_fn = build_detect_fn(image_detector)
    sim = ValidationStreamSimulator(image_dir=_TMP_DIR, detector=detect_fn,
                                    interval_seconds=1.0, mode="sequential",
                                    max_frames=len(chosen))
    frames = []
    conf_dist = {"fire": [], "smoke": []}
    area_dist = {"fire": [], "smoke": []}
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
        if det.has_fire:
            conf_dist["fire"].append(round(det.avg_fire_confidence, 4))
            area_dist["fire"].append(round(det.fire_area_ratio, 5))
        if det.has_smoke:
            conf_dist["smoke"].append(round(det.avg_smoke_confidence, 4))
            area_dist["smoke"].append(round(det.smoke_area_ratio, 5))
        positive.append(det)

    # M6-M10
    agg = EventAggregator()
    shot = ScreenshotManager()
    agent = FireDecisionAgent()
    report_gen = ReportGenerator()
    decisions = {}
    risk_dist = {}

    def analyze(evt):
        decision = agent.analyze(evt)
        decisions.setdefault(evt.event_id, []).append(decision)
        risk_dist[decision.danger_level.value if hasattr(decision.danger_level, "value") else str(decision.danger_level)] = \
            risk_dist.get(decision.danger_level.value if hasattr(decision.danger_level, "value") else str(decision.danger_level), 0) + 1
        try:
            shot.on_decision(evt, decision)
        except Exception as e:
            print("  [warn] M8 on_decision: %s" % e)
        try:
            log_decision(evt, decision, module="fire_decision_agent", source_module="M7")
        except Exception:
            pass
        return decision

    def on_confirmed(evt):
        try:
            shot.on_event_confirmed(evt)
        except Exception as e:
            print("  [warn] M8 confirmed: %s" % e)
        analyze(evt)
        try:
            log_event(evt, "event_confirmed", module="event_aggregator", source_module="M6")
        except Exception:
            pass

    def on_updated(evt):
        try:
            shot.on_event_updated(evt)
        except Exception as e:
            print("  [warn] M8 updated: %s" % e)
        if evt.metadata.get("decision_required"):
            evt.metadata["decision_required"] = False
            analyze(evt)
        try:
            log_event(evt, "event_updated", module="event_aggregator", source_module="M6")
        except Exception:
            pass

    def on_ended(evt):
        analyze(evt)
        try:
            shot.on_event_ended(evt)
        except Exception as e:
            print("  [warn] M8 ended: %s" % e)
        try:
            shots = shot.get_event_shot_records(evt.event_id)
            data = EventReportData(event=evt, decisions=decisions.get(evt.event_id, []), screenshots=shots)
            report_gen.generate(data)
        except Exception as e:
            print("  [warn] M10 report: %s" % e)
        try:
            log_event(evt, "event_ended", module="event_aggregator", source_module="M6")
        except Exception:
            pass

    agg.on_event_confirmed = on_confirmed
    agg.on_event_updated = on_updated
    agg.on_event_ended = on_ended
    agg.on_event_discarded = lambda evt: log_event(evt, "event_discarded",
                                                   module="event_aggregator", source_module="M6")

    before_lines = 0
    if _EVENTS_JSONL.exists():
        before_lines = sum(1 for _ in open(_EVENTS_JSONL, encoding="utf-8"))

    # 按原帧顺序喂给 M6
    frame_index = 0
    for det in positive:
        det.timestamp = float(frame_index)
        det.frame_id = frame_index + 1
        det.fps = 1.0  # 本链路 interval_seconds=1.0 → 1 帧 = 1 秒（名义时间）
        frame_index += 1
        try:
            agg.process(det)
        except Exception as e:
            print("  [warn] M6 process: %s" % e)
    # 连续阴性帧结束事件
    neg_img = None
    if cats["negative"]:
        neg_img = cats["negative"][0]
    if neg_img is None and frames:
        neg_img = Path(frames[0]["image"])
    base = frame_index
    for j in range(3):
        from utils.common import Detection as D
        neg = D(bboxes=[], image_path=str(neg_img) if neg_img else "",
                timestamp=float(base + j), frame_id=base + j + 1, fps=1.0)
        if neg_img:
            img = read_cv(neg_img)
            if img is not None:
                h, w = img.shape[:2]
                neg.image_width = w
                neg.image_height = h
        try:
            agg.process(neg)
        except Exception as e:
            print("  [warn] M6 process negative: %s" % e)

    if _EVENTS_JSONL.exists():
        after_lines = sum(1 for _ in open(_EVENTS_JSONL, encoding="utf-8"))
    else:
        after_lines = 0

    result = {
        "status": "ok",
        "model": str(image_detector.model_path),
        "model_name": image_detector.model_path.name,
        "class_names": dict(image_detector.class_names),
        "val_subset_total": total,
        "frames_processed": len(frames),
        "frames": frames,
        "confidence_distribution": {k: {"min": min(v) if v else 0, "max": max(v) if v else 0,
                                        "mean": round(sum(v) / len(v), 4) if v else 0,
                                        "count": len(v)} for k, v in conf_dist.items()},
        "area_ratio_distribution": {k: {"min": min(v) if v else 0, "max": max(v) if v else 0,
                                        "mean": round(sum(v) / len(v), 5) if v else 0,
                                        "count": len(v)} for k, v in area_dist.items()},
        "events_confirmed": sum(1 for e in agg.event_history if e.status.value == "confirmed")
        + (1 if agg.current_event and agg.current_event.status.value == "confirmed" else 0),
        "events_ended": sum(1 for e in agg.event_history if e.status.value == "ended"),
        "events_discarded": sum(1 for e in agg.event_history if e.status.value == "discarded"),
        "risk_level_distribution": risk_dist,
        "decisions_count": sum(len(v) for v in decisions.values()),
        "events_jsonl_growth": max(0, after_lines - before_lines),
        "artifacts": {},
    }
    for evt in agg.event_history:
        if evt.status.value != "ended":
            continue
        shot_dir = shot.path_mgr.screenshot_event_dir(evt.event_id)
        rpt_dir = report_gen.path_mgr.report_event_dir(evt.event_id)
        result["artifacts"][evt.event_id] = {
            "screenshot_dir": str(shot_dir),
            "screenshot_files": sorted(p.name for p in shot_dir.glob("*")) if shot_dir.exists() else [],
            "report_dir": str(rpt_dir),
            "report_files": sorted(p.name for p in rpt_dir.glob("*")) if rpt_dir.exists() else [],
        }
    log_flush()
    return result


def main() -> int:
    t0 = time.time()
    if not _MODEL.exists():
        print("[错误] 模型不存在:", _MODEL)
        return 1
    from detect.image_detector import ImageDetector
    print("[加载] 模型:", _MODEL)
    det = ImageDetector(model_path=_MODEL)
    print("[加载] names:", dict(det.class_names))
    result = run_chain(det)
    result["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(_RESULT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n[M6-M10] 状态: %s | 确认事件: %s | 结束事件: %s | 丢弃事件: %s | 决策数: %s" % (
        result.get("status"), result.get("events_confirmed"), result.get("events_ended"),
        result.get("events_discarded"), result.get("decisions_count")))
    print("  危险等级分布:", result.get("risk_level_distribution"))
    for eid, art in result.get("artifacts", {}).items():
        print("  - %s: screenshots=%d 报告=%d" % (eid, len(art["screenshot_files"]), len(art["report_files"])))
    print("[保存] %s" % _RESULT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
