"""
============================================================
FireGuardian V2 Phase 4-B - Agent → GUI 端到端冒烟（offscreen）

真实 API 案例：case_01_real_fire / case_02_smoke_only /
              case_04_fireworks / case_05_normal
烟花2 重点案例：复用 Phase 3-B 已生成结果（不重新跑 YOLO）
                Rule=HIGH / Agent=UNCERTAIN|SUSPICIOUS / Final=HIGH

输出截图：tools/v2_phase4b_screenshots/
用法：python tools/v2_phase4b_agent_gui_smoke.py
============================================================
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ["QT_QPA_PLATFORM"] = "offscreen"

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt6.QtWidgets import QApplication

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
from ui.app_controller import UiController
from ui.main_window import MainWindow


SCREENSHOT_DIR = _ROOT / "tools" / "v2_phase4b_screenshots"


def _base_data_source(event_id: str) -> InMemoryDataSource:
    return InMemoryDataSource(
        summaries={
            event_id: {
                "found": True,
                "event_id": event_id,
                "summary": {
                    "status": "confirmed",
                    "duration_seconds": 3.0,
                    "total_frames": 60,
                    "positive_frames": 40,
                    "max_fire_area_ratio": 0.0,
                    "max_smoke_area_ratio": 0.4,
                    "avg_fire_confidence": 0.0,
                    "avg_smoke_confidence": 0.8,
                    "growth_trend": "stable",
                },
            }
        },
        decisions={
            event_id: {
                "found": True,
                "event_id": event_id,
                "decision": {"level": "medium", "score": 0.5, "confidence": 0.72},
            }
        },
    )


def _run_case_presentation(case: dict, client: DeepSeekClient) -> dict:
    event = case["event"]
    decision = case["decision"]
    memory = AgentMemory(events_path=case["history"])
    data_source = MemoryAgentDataSource(
        base=_base_data_source(event.event_id),
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
        rule_level=getattr(decision.danger_level, "value", decision.danger_level),
        outcome=outcome,
        event_id=event.event_id,
        trigger="event_confirmed",
        model=client.model,
        logger=agent.logger,
    )
    return build_agent_presentation_model(
        event.event_id,
        decision,
        outcome,
        hybrid,
        memory_results=extract_memory_results(outcome),
    ).to_dict()


def _fireworks_presentation(seq: str, rule_level: str, final_level: str) -> dict:
    """用 Phase 3-B 已生成结果构造展示模型（不重新跑 YOLO）。"""
    src = _ROOT / "tools" / "v2_phase3_fireworks" / f"agent_with_memory_{seq}.json"
    if not src.exists():
        raise FileNotFoundError(f"缺少 Phase 3-B 结果: {src}")
    fw = json.loads(src.read_text(encoding="utf-8"))
    return {
        "event_id": f"FE-FIREWORKS-{seq}-E1",
        "agent_status": fw.get("agent_status", "ok"),
        "rule_level": rule_level,
        "agent_assessment": fw.get("agent_assessment"),
        "agent_risk_level": fw.get("agent_risk_level"),
        "memory_used": bool(fw.get("memory_used", False)),
        "similar_event_count": int(fw.get("similar_event_count", 0) or 0),
        "memory_summary": fw.get("memory_summary", ""),
        "memory_results": [],
        "possible_cause": fw.get("possible_cause", ""),
        "reasoning_summary": fw.get("reasoning_summary", ""),
        "recommended_action": fw.get("recommended_action", ""),
        "uncertainty": "medium",
        "tools_used": [],
        "final_level": final_level,
        "reason": (
            "M7 High 不可被 Agent 降低（Hybrid 安全规则）"
            if final_level == "high"
            else "Hybrid 融合结果"
        ),
        "latency_ms": float(fw.get("latency_ms", 0.0)),
        "token_usage": {"total_tokens": int(fw.get("total_tokens", 0))},
        "display_fallbacks": [],
        "error": None,
    }


def _verify_and_screenshot(window: MainWindow, name: str) -> Dict[str, str]:
    panel = window.agent_panel
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = SCREENSHOT_DIR / f"{name}_agent_panel.png"
    win_path = SCREENSHOT_DIR / f"{name}_window.png"
    panel.grab().save(str(panel_path))
    window.grab().save(str(win_path))
    return {
        "status": panel.status_label.text(),
        "rule": panel.rule_label.text(),
        "agent": panel.agent_label.text(),
        "final": panel.final_label.text(),
        "memory": panel.memory_label.text(),
        "panel_png": str(panel_path.relative_to(_ROOT)),
        "window_png": str(win_path.relative_to(_ROOT)),
    }


def main() -> int:
    app = QApplication.instance() or QApplication([])
    controller = UiController()
    window = MainWindow(controller)
    window.resize(1200, 800)
    window.show()
    app.processEvents()

    client = DeepSeekClient()
    print(f"model: {client.model} | thinking: disabled | api_key: {client._masked_key}")

    results: List[dict] = []
    case_ids = ["case_01_real_fire", "case_02_smoke_only", "case_04_fireworks", "case_05_normal"]
    for case_id in case_ids:
        case = next(c for c in CASES if c["case_id"] == case_id)
        print(f"运行 {case_id} ...", flush=True)
        presentation = _run_case_presentation(case, client)
        controller.agent_completed.emit(presentation)
        app.processEvents()
        shot = _verify_and_screenshot(window, case_id)
        results.append({"case_id": case_id, **shot})
        print(
            f"  {shot['status']} | {shot['rule']} | {shot['agent']} | {shot['final']} | {shot['memory']}"
        )

    for seq, rule_level, final_level in (("01_e1", "medium", "medium"), ("02_e1", "high", "high")):
        num = int(seq[:2])
        print(f"运行 烟花{num} 重点案例（Phase 3-B 结果）...", flush=True)
        fw_presentation = _fireworks_presentation(seq, rule_level, final_level)
        controller.agent_completed.emit(fw_presentation)
        app.processEvents()
        fw_shot = _verify_and_screenshot(window, f"fireworks{num}")
        results.append({"case_id": f"fireworks{num}_phase3b", **fw_shot})
        print(
            f"  {fw_shot['status']} | {fw_shot['rule']} | {fw_shot['agent']} | {fw_shot['final']} | {fw_shot['memory']}"
        )

    out = SCREENSHOT_DIR / "v2_phase4b_gui_smoke_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out}")

    # 关键断言：状态为合法展示状态（Completed/Fallback/Timeout）；
    # case_01 与 fireworks2 的 Rule/Final=HIGH 且层级可区分
    ok = all(
        any(k in r["status"] for k in ("Completed", "Fallback", "Timeout"))
        for r in results
    )
    case01 = next(r for r in results if r["case_id"] == "case_01_real_fire")
    ok = ok and "Rule: HIGH" in case01["rule"]
    ok = ok and "Hybrid Final: HIGH" in case01["final"]
    fw = next(r for r in results if r["case_id"] == "fireworks2_phase3b")
    ok = ok and "Rule: HIGH" in fw["rule"]
    ok = ok and "Hybrid Final: HIGH" in fw["final"]
    agent_text = fw["agent"]
    ok = ok and ("UNCERTAIN" in agent_text or "SUSPICIOUS" in agent_text)
    ok = ok and agent_text != fw["final"]
    fw1 = next(r for r in results if r["case_id"] == "fireworks1_phase3b")
    ok = ok and "Rule: MEDIUM" in fw1["rule"] and "Hybrid Final: MEDIUM" in fw1["final"]

    window.close()
    controller.shutdown()
    app.processEvents()
    print(f"PHASE4B_GUI_SMOKE_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
