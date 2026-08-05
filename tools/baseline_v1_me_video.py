"""
============================================================
FireGuardian Baseline V1 - me.mp4 困难负样本补充测试
============================================================
按《9.FireGuardian Baseline V1 中等规模训练与精度评估任务.docx》第十三节执行：
- 显式加载 models/baseline_v1_best.pt（不替换 models/best.pt）
- 对 me.mp4 逐帧检测：总帧数 / fire 阳性帧 / smoke 阳性帧
- 连续阳性片段数量 / 最高置信度 / 是否触发 M6 事件
- 与冒烟模型（smoke_v1: fire 12 帧 / smoke 31 帧）对比

不修改任何阈值，不修改 M1-M11。
输出: tools/baseline_v1_me_video_result.json
============================================================
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

_VIDEO = _ROOT / "me.mp4"
_MODEL = _ROOT / "models" / "baseline_v1_best.pt"
_OUT = _ROOT / "tools" / "baseline_v1_me_video_result.json"

SMOKE_REF = {"fire_frames": 12, "smoke_frames": 31, "note": "冒烟模型 smoke_v1 (train 3000/val 600/epochs 3)"}


def main() -> int:
    t0 = time.time()
    if not _VIDEO.exists():
        print("[错误] 视频不存在:", _VIDEO)
        return 1
    if not _MODEL.exists():
        print("[错误] 模型不存在:", _MODEL)
        return 1

    # 读取真实视频帧率（用于 frame_id → seconds 转换）
    import cv2
    _cap = cv2.VideoCapture(str(_VIDEO))
    _FPS = float(_cap.get(cv2.CAP_PROP_FPS) or 0)
    _cap.release()
    if _FPS <= 0:
        _FPS = 20.4

    from video_detect.video_detector import VideoDetector
    from aggregator.event_aggregator import EventAggregator
    from utils.common import Detection

    print("[加载] 模型:", _MODEL)
    det = VideoDetector(model_path=_MODEL)

    agg = EventAggregator()
    fire_frames = 0
    smoke_frames = 0
    positive_frames = 0
    max_conf = {"fire": 0.0, "smoke": 0.0}
    segments = []  # 连续阳性片段
    cur_seg_start = None
    cur_seg_len = 0
    processed = 0
    total = None
    conf_dist = []
    fire_area_dist = []
    smoke_area_dist = []

    for frame, detection, frame_idx, total in det.decode(_VIDEO):
        processed += 1
        has_fire = detection.has_fire
        has_smoke = detection.has_smoke
        if has_fire:
            fire_frames += 1
        if has_smoke:
            smoke_frames += 1
        is_pos = has_fire or has_smoke
        if is_pos:
            positive_frames += 1
            if cur_seg_start is None:
                cur_seg_start = frame_idx
                cur_seg_len = 1
            else:
                cur_seg_len += 1
            conf_dist.append({
                "frame": frame_idx,
                "fire_conf": round(detection.avg_fire_confidence, 4),
                "smoke_conf": round(detection.avg_smoke_confidence, 4),
                "fire_area_ratio": round(detection.fire_area_ratio, 5),
                "smoke_area_ratio": round(detection.smoke_area_ratio, 5),
            })
            fire_area_dist.append(round(detection.fire_area_ratio, 5))
            smoke_area_dist.append(round(detection.smoke_area_ratio, 5))
        else:
            if cur_seg_start is not None:
                segments.append({"start": cur_seg_start, "length": cur_seg_len,
                                 "end": frame_idx - 1})
                cur_seg_start = None
                cur_seg_len = 0
        for b in detection.bboxes:
            cn = b.class_name
            if cn in max_conf:
                max_conf[cn] = max(max_conf[cn], float(b.confidence))
        # M6 链路：喂给 aggregator
        det_obj = Detection(
            bboxes=detection.bboxes,
            image_path=detection.image_path or str(_VIDEO),
            timestamp=float(frame_idx),
            frame_id=frame_idx,
            image_width=detection.image_width,
            image_height=detection.image_height,
            fps=_FPS,
        )
        try:
            agg.process(det_obj)
        except Exception as e:
            print("  [warn] M6 process frame %d: %s" % (frame_idx, e))

    if cur_seg_start is not None:
        segments.append({"start": cur_seg_start, "length": cur_seg_len,
                         "end": processed - 1})

    # M6 事件统计
    events = {
        "confirmed": sum(1 for e in agg.event_history if e.status.value == "confirmed")
        + (1 if agg.current_event and agg.current_event.status.value == "confirmed" else 0),
        "ended": sum(1 for e in agg.event_history if e.status.value == "ended"),
        "discarded": sum(1 for e in agg.event_history if e.status.value == "discarded"),
        "active": 1 if agg.current_event else 0,
    }

    result = {
        "model": str(_MODEL),
        "video": str(_VIDEO),
        "total_frames": processed,
        "fire_positive_frames": fire_frames,
        "smoke_positive_frames": smoke_frames,
        "positive_frames": positive_frames,
        "continuous_positive_segments": segments,
        "max_confidence": {k: round(v, 4) for k, v in max_conf.items()},
        "m6_events": events,
        "m6_triggered": events["confirmed"] > 0 or events["ended"] > 0,
        "confidence_distribution": conf_dist,
        "fire_area_ratio_dist": fire_area_dist,
        "smoke_area_ratio_dist": smoke_area_dist,
        "smoke_model_reference": SMOKE_REF,
        "comparison": {
            "fire_frames_delta": fire_frames - SMOKE_REF["fire_frames"],
            "smoke_frames_delta": smoke_frames - SMOKE_REF["smoke_frames"],
        },
        "duration_seconds": round(time.time() - t0, 1),
    }
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n[结果] 总帧数: %d" % processed)
    print("  fire 阳性帧: %d (冒烟 %d) | smoke 阳性帧: %d (冒烟 %d)" % (
        fire_frames, SMOKE_REF["fire_frames"], smoke_frames, SMOKE_REF["smoke_frames"]))
    print("  连续阳性片段: %d 段" % len(segments))
    print("  最高置信度: %s" % result["max_confidence"])
    print("  M6 事件: %s | 触发: %s" % (events, result["m6_triggered"]))
    print("[保存] %s" % _OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
