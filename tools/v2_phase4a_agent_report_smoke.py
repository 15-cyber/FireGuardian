"""
============================================================
FireGuardian V2 Phase 4-A - Agent → Hybrid → Presentation → M10 报告 冒烟

真实 API 案例：case_01_real_fire / case_02_smoke_only /
              case_04_fireworks / case_05_normal

输出：tools/v2_phase4a_output/reports/event_*/（md + pdf + event.json）
另做 UiController 接线只读验证（实例化 + 信号 + 守卫），不启动检测源。

用法：python tools/v2_phase4a_agent_report_smoke.py [case_id,...]
============================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_memory import AgentMemory, MemoryAgentDataSource
from agent.deepseek_agent import AgentStatus, FireGuardianLLMAgent
from agent.deepseek_client import DeepSeekClient
from agent.hybrid_decision import run_hybrid_pipeline
from agent.memory_normalizer import build_query_record
from agent.presentation_models import (
    build_agent_presentation_model,
    extract_memory_results,
    run_agent_with_timeout,
)
from agent.tool_executor import InMemoryDataSource
from reports.report_generator import ReportGenerator
from tests.memory_case_defs import CASES
from utils.common import EventReportData, FireDecision, load_config


OUTPUT_DIR = _ROOT / "tools" / "v2_phase4a_output" / "reports"


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
                    "positive_ratio": 0.67,
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


def run_case(case: dict, client: DeepSeekClient, gen: ReportGenerator) -> dict:
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
    presentation = build_agent_presentation_model(
        event.event_id,
        decision,
        outcome,
        hybrid,
        memory_results=extract_memory_results(outcome),
    )
    report_data = EventReportData(
        event=event,
        decisions=[decision],
        screenshots=[],
    )
    paths = gen.generate(report_data, agent_presentation=presentation)
    return {
        "case_id": case["case_id"],
        "agent_status": outcome.status.value,
        "assessment": outcome.agent_assessment,
        "agent_level": outcome.agent_level,
        "final_level": hybrid.final_level,
        "memory_used": presentation.memory_used,
        "similar_count": presentation.similar_event_count,
        "latency_ms": round(outcome.latency_ms, 1),
        "total_tokens": int(outcome.token_usage.get("total_tokens", 0)),
        "reports": [str(p.relative_to(_ROOT)) for p in paths],
    }


def _verify_ui_controller_wiring() -> str:
    from ui.app_controller import UiController

    ctrl = UiController()
    for name in ("agent_started", "agent_tool_called", "agent_completed", "agent_failed"):
        assert hasattr(ctrl, name), f"缺少信号 {name}"
    assert not ctrl._agent_guard.allowed("E-TEST", "frame_update")
    assert ctrl._agent_guard.count("E-TEST") == 0
    del ctrl
    return "UiController 接线验证通过（信号 + 守卫）"


def main() -> int:
    filters: List[str] = []
    if len(sys.argv) > 1:
        filters = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    cases = [c for c in CASES if not filters or c["case_id"] in filters]
    print(f"Phase 4-A 冒烟：{len(cases)} 案例 → Agent → Hybrid → Presentation → M10 报告")

    cfg = load_config()
    cfg["report"] = {"output_dir": str(OUTPUT_DIR), "formats": ["md", "pdf"]}
    gen = ReportGenerator(config=cfg)
    client = DeepSeekClient()
    print(f"model: {client.model} | thinking: disabled | api_key: {client._masked_key}")

    results = []
    for case in cases:
        print(f"运行 {case['case_id']} ...", flush=True)
        results.append(run_case(case, client, gen))

    print("-" * 100)
    for r in results:
        print(
            f"{r['case_id']:<20} status={r['agent_status']:<14} "
            f"assess={str(r['assessment']):<14} level={r['agent_level']} "
            f"final={r['final_level']} mem={r['memory_used']} sim={r['similar_count']} "
            f"lat={r['latency_ms']}ms tokens={r['total_tokens']}"
        )
    print("-" * 100)
    print(_verify_ui_controller_wiring())

    out = OUTPUT_DIR.parent / "v2_phase4a_agent_report_smoke_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out}")
    ok = all(any(p.endswith(".md") for p in r["reports"]) for r in results)
    print(f"PHASE4A_SMOKE_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
