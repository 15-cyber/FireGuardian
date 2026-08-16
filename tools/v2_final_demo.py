"""
============================================================
FireGuardian V2.7 - 最终闭环演示

完整链路：视频 → YOLO(V2 模型) → M6 → M7 → DeepSeek Agent
          → Hybrid → M8 截图 → M9 日志 → M10 报告 → M11 GUI

用法：
  python tools/v2_final_demo.py --model models/v2_15k_best.pt \
      --video "测试视频/烟花2.mp4" [--conf 0.5]

输出：tools/v2_final_demo/
============================================================
"""
from __future__ import annotations

import argparse
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import torch
from ultralytics import YOLO

from aggregator.event_aggregator import EventAggregator
from agent.deepseek_agent import FireGuardianLLMAgent
from agent.deepseek_client import DeepSeekClient
from agent.fire_decision_agent import FireDecisionAgent
from agent.hybrid_decision import run_hybrid_pipeline
from agent.presentation_models import (
    build_agent_presentation_model,
    extract_memory_results,
    run_agent_with_timeout,
)
from agent.tool_executor import InMemoryDataSource
from logs.logger import log_event, log_decision
from reports.report_generator import ReportGenerator
from tests.memory_case_defs import CASES
from utils.common import Detection, EventReportData


OUT_DIR = _ROOT / "tools" / "v2_final_demo"


def _pick_history_case(event) -> str:
    """根据事件特征选择 Memory 历史案例。"""
    if event.max_fire_area_ratio > 0.12 and event.max_smoke_area_ratio > 0.2:
        return "case_01_real_fire"
    if event.max_smoke_area_ratio > 0.2 and event.max_fire_area_ratio < 0.05:
        return "case_04_fireworks"
    return "case_02_smoke_only"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(_ROOT / "models" / "v2_15k_best.pt"))
    ap.add_argument("--video", default=str(_ROOT / "测试视频" / "烟花2.mp4"))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--frame-skip", type=int, default=2)
    args = ap.parse_args()

    model_path = Path(args.model)
    video_path = Path(args.video)
    if not model_path.exists():
        print("模型不存在:", model_path)
        return 1
    if not video_path.exists():
        print("视频不存在:", video_path)
        return 1
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trace = {"started_at": time.time(), "steps": []}

    def step(name, ok=True, detail=""):
        trace["steps"].append({"step": name, "ok": bool(ok), "detail": detail,
                               "t": round(time.time() - trace["started_at"], 1)})
        print(f"[{name}] {'OK' if ok else 'FAIL'} {detail}")

    # ---- YOLO(V2) 逐帧检测 ----
    print("加载 V2 模型:", model_path.name)
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    aggregator = EventAggregator()
    events = []
    aggregator.on_event_confirmed = lambda e: events.append(e)
    aggregator.on_event_ended = lambda e: events.append(e)

    frame_idx = 0
    rep_frames = {}
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if (frame_idx - 1) % args.frame_skip != 0:
            continue
        results = model(frame, conf=args.conf, iou=0.45, device=0, verbose=False)
        boxes = results[0].boxes
        bboxes = []
        fire_conf = smoke_conf = 0.0
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cid = int(cls[i])
                bboxes.append(
                    __import__("utils.common", fromlist=["BoundingBox"]).BoundingBox(
                        x1=float(xyxy[i][0]), y1=float(xyxy[i][1]),
                        x2=float(xyxy[i][2]), y2=float(xyxy[i][3]),
                        confidence=float(confs[i]), class_id=cid,
                        class_name=model.names.get(cid, str(cid)),
                    )
                )
                if cid == 0:
                    fire_conf = max(fire_conf, float(confs[i]))
                elif cid == 1:
                    smoke_conf = max(smoke_conf, float(confs[i]))
        detection = Detection(
            bboxes=bboxes, image_width=width, image_height=height,
            frame_id=frame_idx, fps=fps, timestamp=frame_idx / fps if fps else 0.0,
        )
        if not rep_frames:
            rep_frames["first"] = (frame_idx, frame.copy())
        if fire_conf > 0.5 and "fire" not in rep_frames:
            rep_frames["fire"] = (frame_idx, frame.copy())
        if smoke_conf > 0.5 and "smoke" not in rep_frames:
            rep_frames["smoke"] = (frame_idx, frame.copy())
        aggregator.process(detection)
        if frame_idx % 1000 == 0:
            print(f"  已处理 {frame_idx} 帧（{time.time() - t0:.0f}s）", flush=True)
    cap.release()
    step("YOLO(M3/M4) 检测", True, f"{total} 帧，处理 {frame_idx}（skip={args.frame_skip}）")

    # ---- M6 事件 ----
    step("M6 事件聚合", bool(events), f"{len(events)} 个事件")
    if not events:
        print("未产生事件，演示终止")
        return 1
    event = events[-1]
    step("M6 FireEvent", True, f"{event.event_id} duration={event.duration:.1f}s "
                               f"fire={event.max_fire_area_ratio:.3f} smoke={event.max_smoke_area_ratio:.3f}")

    # ---- M7 规则决策 ----
    m7 = FireDecisionAgent()
    decision = m7.analyze(event)
    log_decision(event, decision, module="fire_decision_agent", source_module="M7")
    step("M7 Rule Decision", True, f"level={getattr(decision.danger_level, 'value', decision.danger_level)} "
                                   f"score={decision.score:.3f}")

    # ---- M9 事件日志 ----
    log_event(event, "event_ended", module="event_aggregator", source_module="M6")
    step("M9 事件日志", True, "events.jsonl 已记录")

    # ---- DeepSeek Agent + Hybrid ----
    client = DeepSeekClient()
    history_case = _pick_history_case(event)
    case = next(c for c in CASES if c["case_id"] == history_case)
    from agent.agent_memory import AgentMemory, MemoryAgentDataSource
    from agent.memory_normalizer import build_query_record

    memory = AgentMemory(events_path=case["history"])
    data_source = MemoryAgentDataSource(
        base=InMemoryDataSource(
            summaries={event.event_id: {"found": True, "event_id": event.event_id,
                                        "summary": {"status": "ended", "duration_seconds": event.duration}}},
            decisions={event.event_id: {"found": True, "event_id": event.event_id,
                                        "decision": {"level": getattr(decision.danger_level, "value", decision.danger_level)}}},
        ),
        memory=memory,
        query_provider=lambda eid: (
            build_query_record(event, decision) if eid == event.event_id else None
        ),
    )
    agent = FireGuardianLLMAgent(client=client, data_source=data_source,
                                 max_tool_rounds=3, thinking_disabled=True)
    outcome = run_agent_with_timeout(agent, event, decision, trigger="event_confirmed")
    hybrid = run_hybrid_pipeline(
        rule_level=getattr(decision.danger_level, "value", decision.danger_level),
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
    step("DeepSeek Agent", outcome.status == "ok", f"{outcome.status.value} "
         f"assess={outcome.agent_assessment} level={outcome.agent_level}")
    step("Hybrid Final", True, f"rule={hybrid.rule_level} agent={hybrid.agent_level} "
                               f"final={hybrid.final_level}")

    # ---- M8 截图（代表帧） ----
    shot_dir = OUT_DIR / f"event_{event.event_id}" / "screenshots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    for key, (idx, frame) in rep_frames.items():
        cv2.imwrite(str(shot_dir / f"{key}_{idx:06d}.jpg"), frame)
    step("M8 证据截图", bool(rep_frames), f"{len(rep_frames)} 张代表帧")

    # ---- M10 报告 ----
    from utils.common import load_config
    cfg = load_config()
    cfg["report"] = {"output_dir": str(OUT_DIR / "reports"), "formats": ["md", "pdf"]}
    report_gen = ReportGenerator(config=cfg)
    report_data = EventReportData(event=event, decisions=[decision], screenshots=[])
    paths = report_gen.generate(report_data, agent_presentation=presentation)
    step("M10 报告", bool(paths), "、".join(str(p.relative_to(_ROOT)) for p in paths))

    # ---- M11 GUI（offscreen 注入展示） ----
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from ui.app_controller import UiController
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    controller = UiController()
    window = MainWindow(controller)
    window.resize(1200, 800)
    window.show()
    app.processEvents()
    controller.agent_completed.emit(presentation.to_dict())
    app.processEvents()
    gui_shot = OUT_DIR / f"event_{event.event_id}" / "gui_agent_panel.png"
    window.agent_panel.grab().save(str(gui_shot))
    window.close()
    controller.shutdown()
    app.processEvents()
    step("M11 GUI 展示", gui_shot.exists(), str(gui_shot.relative_to(_ROOT)))

    trace["completed_at"] = time.time()
    trace["event"] = event.to_dict()
    trace["decision"] = {k: getattr(decision, k) for k in
                         ("danger_level", "score", "confidence", "reasons")}
    trace["agent"] = outcome.to_dict()
    trace["hybrid"] = hybrid.to_dict()
    trace["presentation"] = presentation.to_dict()
    trace_path = OUT_DIR / f"event_{event.event_id}" / "final_demo_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n演示完成，trace 已保存: {trace_path}")
    print(f"FINAL_DEMO_RESULT: {'PASS' if all(s['ok'] for s in trace['steps']) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
