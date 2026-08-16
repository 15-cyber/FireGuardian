"""
============================================================
FireGuardian V2 Phase 6 - Agent + YOLO 联合测试

对比 V1 模型（models/best.pt）与 V2 模型（models/v2_15k_best.pt）在
真实素材上的检测质量，并用各自检测结果驱动 Agent + Hybrid（真实 API）
做端到端联合评估：
  - 视频：烟火.mp4（真实火灾）、me.mp4（烟花）、烟花1/2.mp4（烟花）
  - 指标：fire/smoke 正帧、最大置信度/面积、连续段、误检对比
  - 联合：代表性事件 → M7 → Agent(Memory) → Hybrid → Final

用法：python tools/v2_phase6_joint_test.py [--v2-model models/v2_15k_best.pt]
输出：tools/v2_phase6_joint_test/
============================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO

from agent.agent_memory import AgentMemory, MemoryAgentDataSource
from agent.deepseek_agent import FireGuardianLLMAgent
from agent.deepseek_client import DeepSeekClient
from agent.hybrid_decision import run_hybrid_pipeline
from agent.memory_normalizer import build_query_record
from agent.presentation_models import (
    build_agent_presentation_model,
    extract_memory_results,
    run_agent_with_timeout,
)
from agent.tool_executor import InMemoryDataSource
from tests.memory_case_defs import CASES
from utils.common import Detection, EventStatus, FireDecision, FireEvent, GrowthTrend


OUT_DIR = _ROOT / "tools" / "v2_phase6_joint_test"
FRAME_SKIP = 4
CONFS = (0.25, 0.5)

VIDEOS = [
    _ROOT / "烟火.mp4",
    _ROOT / "me.mp4",
    _ROOT / "测试视频" / "烟花1.mp4",
    _ROOT / "测试视频" / "烟花2.mp4",
]


def _segments(positive: List[int]) -> List[dict]:
    segs = []
    for idx in positive:
        if segs and idx == segs[-1]["end"] + 1:
            segs[-1]["end"] = idx
            segs[-1]["length"] += 1
        else:
            segs.append({"start": idx, "end": idx, "length": 1})
    return segs


def analyze_video(model: YOLO, path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stats = []
    frame_idx = 0
    cap = cv2.VideoCapture(str(path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % FRAME_SKIP != 0:
            continue
        results = model(frame, conf=0.001, iou=0.45, device=0, verbose=False)
        boxes = results[0].boxes
        fire_confs, smoke_confs, fire_area, smoke_area = 0.0, 0.0, 0.0, 0.0
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                if int(cls[i]) == 0:
                    fire_confs = max(fire_confs, float(confs[i]))
                elif int(cls[i]) == 1:
                    smoke_confs = max(smoke_confs, float(confs[i]))
        # 面积用模型框面积（conf>=0.25 计入）
        fire_area, smoke_area = 0.0, 0.0
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                area = max(0.0, (x2 - x1) * (y2 - y1)) / (width * height)
                if int(cls[i]) == 0 and confs[i] >= 0.25:
                    fire_area = max(fire_area, area)
                elif int(cls[i]) == 1 and confs[i] >= 0.25:
                    smoke_area = max(smoke_area, area)
        stats.append(
            {
                "frame_idx": frame_idx,
                "fire_conf": round(float(fire_confs), 4),
                "smoke_conf": round(float(smoke_confs), 4),
                "fire_area": round(float(fire_area), 6),
                "smoke_area": round(float(smoke_area), 6),
            }
        )
    cap.release()

    fire_pos = [s["frame_idx"] for s in stats if s["fire_conf"] >= 0.25]
    smoke_pos = [s["frame_idx"] for s in stats if s["smoke_conf"] >= 0.25]
    fire_pos_50 = [s["frame_idx"] for s in stats if s["fire_conf"] >= 0.5]
    smoke_pos_50 = [s["frame_idx"] for s in stats if s["smoke_conf"] >= 0.5]
    return {
        "video": path.name,
        "fps": round(fps, 2),
        "total_frames": total,
        "processed_frames": len(stats),
        "frame_skip": FRAME_SKIP,
        "fire_positive_frames@0.25": len(fire_pos),
        "smoke_positive_frames@0.25": len(smoke_pos),
        "fire_positive_frames@0.5": len(fire_pos_50),
        "smoke_positive_frames@0.5": len(smoke_pos_50),
        "fire_max_confidence": round(max((s["fire_conf"] for s in stats), default=0.0), 4),
        "smoke_max_confidence": round(max((s["smoke_conf"] for s in stats), default=0.0), 4),
        "fire_max_area_ratio": round(max((s["fire_area"] for s in stats), default=0.0), 6),
        "smoke_max_area_ratio": round(max((s["smoke_area"] for s in stats), default=0.0), 6),
        "continuous_fire_segments": _segments(fire_pos_50),
        "continuous_smoke_segments": _segments(smoke_pos_50),
    }


def _event_from_stats(video_stats: dict, label: str) -> FireEvent:
    fps = video_stats["fps"]
    processed = video_stats["processed_frames"]
    return FireEvent(
        event_id=f"P6-{label}-{video_stats['video'][:8]}",
        status=EventStatus.ENDED,
        start_time_seconds=0.0,
        last_time_seconds=processed / fps if fps else 0.0,
        duration_seconds=processed / fps if fps else 0.0,
        fps=fps,
        total_frames=processed,
        positive_frames=video_stats["fire_positive_frames@0.25"]
        + video_stats["smoke_positive_frames@0.25"],
        max_fire_area_ratio=video_stats["fire_max_area_ratio"],
        max_smoke_area_ratio=video_stats["smoke_max_area_ratio"],
        avg_fire_confidence=video_stats["fire_max_confidence"],
        avg_smoke_confidence=video_stats["smoke_max_confidence"],
        growth_trend=GrowthTrend.UNKNOWN,
        source_type="video",
        source_name=video_stats["video"],
    )


def _run_agent_arm(event: FireEvent, rule_level: str, history_case: str, client: DeepSeekClient) -> dict:
    case = next(c for c in CASES if c["case_id"] == history_case)
    decision = FireDecision(
        event_id=event.event_id,
        danger_level=rule_level,
        score=0.5,
        confidence=0.7,
        reasons=["Phase 6 联合测试"],
    )
    memory = AgentMemory(events_path=case["history"])
    data_source = MemoryAgentDataSource(
        base=InMemoryDataSource(
            summaries={
                event.event_id: {"found": True, "event_id": event.event_id,
                                 "summary": {"status": "ended", "duration_seconds": event.duration}}
            },
            decisions={
                event.event_id: {"found": True, "event_id": event.event_id,
                                 "decision": {"level": rule_level}}
            },
        ),
        memory=memory,
        query_provider=lambda eid: (
            build_query_record(event, decision) if eid == event.event_id else None
        ),
    )
    agent = FireGuardianLLMAgent(
        client=client,
        data_source=data_source,
        max_tool_rounds=3,
        thinking_disabled=True,
    )
    outcome = run_agent_with_timeout(agent, event, decision, trigger="event_confirmed")
    hybrid = run_hybrid_pipeline(
        rule_level=rule_level,
        outcome=outcome,
        event_id=event.event_id,
        trigger="event_confirmed",
        model=client.model,
        logger=agent.logger,
    )
    presentation = build_agent_presentation_model(
        event.event_id, decision, outcome, hybrid,
        memory_results=extract_memory_results(outcome),
    )
    return {
        "event_id": event.event_id,
        "agent_status": outcome.status.value,
        "assessment": outcome.agent_assessment,
        "agent_level": outcome.agent_level,
        "final_level": hybrid.final_level,
        "memory_used": presentation.memory_used,
        "similar_count": presentation.similar_event_count,
        "latency_ms": round(outcome.latency_ms, 1),
        "tokens": int(outcome.token_usage.get("total_tokens", 0)),
        "llm_calls": outcome.rounds,
        "tool_calls": len(outcome.tool_calls),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-model", default=str(_ROOT / "models" / "best.pt"))
    ap.add_argument("--v2-model", default=str(_ROOT / "models" / "v2_15k_best.pt"))
    args = ap.parse_args()
    v1_path = Path(args.v1_model)
    v2_path = Path(args.v2_model)
    if not v1_path.exists():
        print("V1 模型不存在:", v1_path)
        return 1
    if not v2_path.exists():
        print("V2 模型不存在（Phase 5 训练尚未完成）:", v2_path)
        return 1
    if not torch.cuda.is_available():
        print("CUDA 不可用，联合测试需要 GPU")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1 = YOLO(str(v1_path))
    v2 = YOLO(str(v2_path))

    detections = {"v1": {}, "v2": {}}
    for name, model in (("v1", v1), ("v2", v2)):
        for video in VIDEOS:
            if not video.exists():
                print("跳过（不存在）:", video)
                continue
            print(f"[{name}] 分析 {video.name} ...", flush=True)
            detections[name][video.name] = analyze_video(model, video)

    (OUT_DIR / "detection_comparison.json").write_text(
        json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n检测对比摘要（fire 最大面积 / fire 最大置信度 @conf>=0.25）：")
    for video_name in detections["v1"]:
        a = detections["v1"][video_name]
        b = detections["v2"].get(video_name, {})
        print(
            f"  {video_name:<14} V1 fire_area={a['fire_max_area_ratio']:.4f} conf={a['fire_max_confidence']:.3f} "
            f"| V2 fire_area={b.get('fire_max_area_ratio', 0):.4f} conf={b.get('fire_max_confidence', 0):.3f}"
        )

    # 联合 Agent 测试（真实 API）
    client = DeepSeekClient()
    print(f"\nAgent 联合测试（model={client.model} thinking=disabled）...")
    agent_results = []
    for label, video_name, rule_level, history_case in (
        ("real_fire", "烟火.mp4", "high", "case_01_real_fire"),
        ("fireworks_me", "me.mp4", "medium", "case_04_fireworks"),
        ("fireworks1", "烟花1.mp4", "medium", "case_04_fireworks"),
        ("fireworks2", "烟花2.mp4", "high", "case_04_fireworks"),
    ):
        stats = detections["v2"].get(video_name)
        if not stats:
            continue
        event = _event_from_stats(stats, label)
        rec = {"label": label, "video": video_name, **stats}
        print(f"  运行 {label}（{video_name}）...", flush=True)
        rec["agent"] = _run_agent_arm(event, rule_level, history_case, client)
        agent_results.append(rec)

    (OUT_DIR / "agent_joint_results.json").write_text(
        json.dumps(agent_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nAgent 联合结果：")
    for r in agent_results:
        a = r["agent"]
        print(
            f"  {r['label']:<14} V2 fire_area={r['fire_max_area_ratio']:.4f} | "
            f"agent={a['agent_status']} assess={a['assessment']} level={a['agent_level']} "
            f"final={a['final_level']} mem={a['memory_used']}({a['similar_count']}) "
            f"lat={a['latency_ms']}ms tok={a['tokens']}"
        )
    print(f"\n结果已保存: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
