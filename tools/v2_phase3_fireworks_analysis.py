"""
============================================================
FireGuardian V2 Phase 3-B - 烟花视频离线检测 + Agent Memory 对比

只读输入：
  - 测试视频/烟花1.mp4、烟花2.mp4（不修改原视频）
  - models/best.pt（只读，不训练、不改 conf）

复用现有实现（不新写 YOLO 推理）：
  - M4 VideoDetector.decode() 逐帧检测
  - M6 EventAggregator 事件聚合
  - M7 FireDecisionAgent 规则决策

输出 tools/v2_phase3_fireworks/：
  fireworks_XX_detection.json / video_summary_XX.json
  representative_frames_XX/（≤20 张/视频）
  agent_without_memory_XX.json / agent_with_memory_XX.json

用法：
    python tools/v2_phase3_fireworks_analysis.py [01|02|all]
============================================================
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2

from aggregator.event_aggregator import EventAggregator
from agent.fire_decision_agent import FireDecisionAgent
from agent.memory_models import EventMemoryRecord
from tests.memory_case_defs import CASES
from tools.v2_phase3_memory_smoke import run_agent_arm
from utils.common import Detection
from video_detect.video_detector import VideoDetector


OUTPUT_DIR = _ROOT / "tools" / "v2_phase3_fireworks"
VIDEO_DIR = _ROOT / "测试视频"
FRAME_SKIP = 2
MAX_REP_FRAMES = 20


def _find_videos() -> List[Path]:
    if not VIDEO_DIR.exists():
        return []
    return sorted(
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov")
    )


def _probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    info = {
        "filename": path.name,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration_seconds"] = (
        round(info["total_frames"] / info["fps"], 2) if info["fps"] > 0 else 0.0
    )
    return info


def _segments(positive: List[int]) -> List[dict]:
    segments: List[dict] = []
    for idx in positive:
        if segments and idx == segments[-1]["end"] + 1:
            segments[-1]["end"] = idx
            segments[-1]["length"] += 1
        else:
            segments.append({"start": idx, "end": idx, "length": 1})
    return segments


def _pick_rep_frames(stats: List[dict], processed_ids: List[int]) -> List[int]:
    """挑选代表帧（≤20）：高置信度/双类同现/事件起止/峰值 + 均匀补充。"""
    priority: List[int] = []

    def _add(idx: int) -> None:
        if idx not in priority:
            priority.append(idx)

    if stats:
        best_fire = max(stats, key=lambda s: s["fire_max_conf"])
        best_smoke = max(stats, key=lambda s: s["smoke_max_conf"])
        both = [s for s in stats if s["has_fire"] and s["has_smoke"]]
        best_both = max(both, key=lambda s: s["fire_area"] + s["smoke_area"]) if both else None
        peak = max(stats, key=lambda s: s["fire_area"] + s["smoke_area"])
        positives = [s for s in stats if s["has_fire"] or s["has_smoke"]]
        if positives:
            _add(positives[0]["frame_idx"])
            _add(positives[-1]["frame_idx"])
        if best_fire["fire_max_conf"] > 0:
            _add(best_fire["frame_idx"])
        if best_smoke["smoke_max_conf"] > 0:
            _add(best_smoke["frame_idx"])
        if best_both is not None:
            _add(best_both["frame_idx"])
        _add(peak["frame_idx"])

    picked = list(priority)
    for idx in processed_ids:
        if len(picked) >= MAX_REP_FRAMES:
            break
        if idx not in picked:
            picked.append(idx)
    return picked[:MAX_REP_FRAMES]


def _extract_frames(path: Path, frame_indices: List[int], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    saved: List[str] = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx - 1, 0))
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame)
        saved.append(name)
    cap.release()
    return saved


def analyze_video(path: Path, detector: VideoDetector, seq: str) -> dict:
    print(f"分析视频: {path.name}（frame_skip={FRAME_SKIP}）")
    probe = _probe(path)
    aggregator = EventAggregator()
    events: List = []
    aggregator.on_event_confirmed = lambda e: events.append(e)
    aggregator.on_event_ended = lambda e: events.append(e)

    stats: List[dict] = []
    t0 = time.time()
    for frame, detection, frame_idx, total in detector.decode(str(path)):
        h, w = frame.shape[:2]
        det = Detection(
            bboxes=detection.bboxes,
            image_width=w,
            image_height=h,
            frame_id=frame_idx,
            fps=probe["fps"],
            timestamp=frame_idx / probe["fps"] if probe["fps"] > 0 else 0.0,
        )
        fire_conf = max((b.confidence for b in det.fire_boxes), default=0.0)
        smoke_conf = max((b.confidence for b in det.smoke_boxes), default=0.0)
        stats.append(
            {
                "frame_idx": frame_idx,
                "has_fire": bool(det.fire_boxes),
                "has_smoke": bool(det.smoke_boxes),
                "fire_max_conf": round(float(fire_conf), 4),
                "smoke_max_conf": round(float(smoke_conf), 4),
                "fire_area": round(float(det.fire_area_ratio), 6),
                "smoke_area": round(float(det.smoke_area_ratio), 6),
            }
        )
        aggregator.process(det)
        if len(stats) % 500 == 0:
            print(f"  已处理 {len(stats)} 帧，耗时 {time.time() - t0:.0f}s", flush=True)

    fire_pos = [s["frame_idx"] for s in stats if s["has_fire"]]
    smoke_pos = [s["frame_idx"] for s in stats if s["has_smoke"]]
    detection_summary = {
        "filename": path.name,
        "processed_frames": len(stats),
        "total_frames": probe["total_frames"],
        "frame_skip": FRAME_SKIP,
        "fire_positive_frames": len(fire_pos),
        "smoke_positive_frames": len(smoke_pos),
        "fire_max_confidence": round(max((s["fire_max_conf"] for s in stats), default=0.0), 4),
        "smoke_max_confidence": round(max((s["smoke_max_conf"] for s in stats), default=0.0), 4),
        "fire_max_area_ratio": round(max((s["fire_area"] for s in stats), default=0.0), 6),
        "smoke_max_area_ratio": round(max((s["smoke_area"] for s in stats), default=0.0), 6),
        "continuous_fire_segments": _segments(fire_pos),
        "continuous_smoke_segments": _segments(smoke_pos),
    }

    rep_dir = OUTPUT_DIR / f"representative_frames_{seq}"
    rep_indices = _pick_rep_frames(stats, fire_pos + smoke_pos)
    rep_files = _extract_frames(path, rep_indices, rep_dir)

    result = {
        "video_summary": probe,
        "detection": detection_summary,
        "representative_frames": rep_files,
        "m6_events": events,
    }
    return result


def _choose_representative_events(analysis: dict, max_events: int = 2) -> List:
    """优先选择 fire/smoke 冲突明显（smoke 高 fire 低或双类并存）的事件。"""
    events = [
        ev
        for ev in analysis["m6_events"]
        if ev.status.value in ("confirmed", "ended")
        and (ev.max_fire_area_ratio > 0 or ev.max_smoke_area_ratio > 0)
    ]
    unique: Dict[str, object] = {}
    for ev in events:
        unique.setdefault(ev.event_id, ev)
    events = list(unique.values())
    events.sort(
        key=lambda e: (
            e.max_smoke_area_ratio
            + (0.5 if e.max_fire_area_ratio > 0 else 0.0)
            + (
                0.2
                if getattr(e.growth_trend, "value", e.growth_trend) == "decreasing"
                else 0.0
            )
        ),
        reverse=True,
    )
    return events[:max_events]


def _run_agent_arms(seq: str, event, decision, history_path: Path, out: Path) -> None:
    case = {
        "case_id": f"fireworks_{seq}_event",
        "event": event,
        "decision": decision,
        "history": history_path,
    }
    without = run_agent_arm(case, with_memory=False)
    with_mem = run_agent_arm(case, with_memory=True)
    (out / f"agent_without_memory_{seq}.json").write_text(
        json.dumps(without, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / f"agent_with_memory_{seq}.json").write_text(
        json.dumps(with_mem, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    videos = _find_videos()
    if not videos:
        print("未找到测试视频目录或视频文件")
        return 1
    args = sys.argv[1] if len(sys.argv) > 1 else "all"
    selected = [v for i, v in enumerate(videos, 1) if args == "all" or f"{i:02d}" == args]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch

    detector = VideoDetector(frame_skip=FRAME_SKIP)
    if not torch.cuda.is_available():
        detector.device = "cpu"
        print("CUDA 不可用，推理使用 CPU")

    fireworks_history = next(c for c in CASES if c["case_id"] == "case_04_fireworks")["history"]
    m7 = FireDecisionAgent()

    for idx, video in enumerate(selected, 1):
        seq = f"{idx:02d}"
        print(f"\n=== 视频 {seq}: {video.name} ===")
        analysis = analyze_video(video, detector, seq)
        (OUTPUT_DIR / f"fireworks_{seq}_detection.json").write_text(
            json.dumps(analysis["detection"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUTPUT_DIR / f"video_summary_{seq}.json").write_text(
            json.dumps(analysis["video_summary"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        events = _choose_representative_events(analysis)
        print(f"  M6 事件数: {len(analysis['m6_events'])}，选择代表事件: {[e.event_id for e in events]}")
        for k, event in enumerate(events, 1):
            decision = m7.analyze(event)
            print(
                f"  事件 {event.event_id}: fire={event.max_fire_area_ratio:.3f} "
                f"smoke={event.max_smoke_area_ratio:.3f} "
                f"M7={getattr(decision.danger_level, 'value', decision.danger_level)}"
            )
            _run_agent_arms(f"{seq}_e{k}", event, decision, fireworks_history, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
