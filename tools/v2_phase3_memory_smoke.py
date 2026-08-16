"""
============================================================
FireGuardian V2 Phase 3-B - Memory A/B 真实 API 冒烟

6 个固定案例 × (Without Memory / With Memory) = 12 次 Agent 实验
  model: deepseek-v4-flash
  thinking: disabled（本阶段只评估 Memory 变量，不做 thinking A/B）

用法：
    python tools/v2_phase3_memory_smoke.py [case_id,...]
结果保存：logs/v2_phase3_memory_ab_results.json
============================================================
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.agent_memory import AgentMemory, MemoryAgentDataSource
from agent.deepseek_agent import AgentStatus, FireGuardianLLMAgent
from agent.deepseek_client import DeepSeekClient
from agent.hybrid_decision import run_hybrid_pipeline
from agent.memory_normalizer import build_query_record
from agent.tool_executor import InMemoryDataSource
from tests.memory_case_defs import CASES


BASE_TOOLS = [
    "get_event_summary",
    "get_rule_decision",
    "get_detection_history",
    "get_event_evidence",
]


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


def run_agent_arm(
    case: dict,
    with_memory: bool,
    client: Optional[DeepSeekClient] = None,
) -> dict:
    """执行单个案例的一个实验臂，返回结构化记录。"""
    event = case["event"]
    decision = case["decision"]
    client = client or DeepSeekClient()

    if with_memory:
        memory = AgentMemory(events_path=case["history"])
        data_source = MemoryAgentDataSource(
            base=_base_data_source(event.event_id),
            memory=memory,
            query_provider=lambda eid: (
                build_query_record(event, decision) if eid == event.event_id else None
            ),
        )
    else:
        data_source = _base_data_source(event.event_id)

    agent = FireGuardianLLMAgent(
        client=client,
        data_source=data_source,
        max_tool_rounds=3,
        thinking_disabled=True,
        allowed_tool_names=None if with_memory else BASE_TOOLS,
        logger=AgentLogger(),
    )
    outcome = agent.analyze(event, decision, trigger="event_confirmed")
    hybrid = run_hybrid_pipeline(
        rule_level=getattr(decision.danger_level, "value", decision.danger_level),
        outcome=outcome,
        event_id=event.event_id,
        trigger="event_confirmed",
        model=client.model,
        logger=agent.logger,
    )
    return {
        "case_id": case["case_id"],
        "arm": "with_memory" if with_memory else "without_memory",
        "agent_status": outcome.status.value,
        "memory_used": bool(outcome.agent_result and outcome.agent_result.memory_used),
        "similar_event_count": int(
            outcome.agent_result.similar_event_count
            if outcome.agent_result
            else 0
        ),
        "agent_assessment": outcome.agent_assessment,
        "agent_risk_level": outcome.agent_level,
        "final_level": hybrid.final_level,
        "reasoning_summary": (
            outcome.agent_result.reasoning_summary if outcome.agent_result else ""
        ),
        "recommended_action": (
            outcome.agent_result.recommended_action if outcome.agent_result else ""
        ),
        "memory_summary": (
            outcome.agent_result.memory_summary if outcome.agent_result else ""
        ),
        "tool_calls": len(outcome.tool_calls),
        "llm_calls": outcome.rounds,
        "latency_ms": round(outcome.latency_ms, 1),
        "prompt_tokens": int(outcome.token_usage.get("prompt_tokens", 0)),
        "completion_tokens": int(outcome.token_usage.get("completion_tokens", 0)),
        "total_tokens": int(outcome.token_usage.get("total_tokens", 0)),
        "error": outcome.error,
    }


def _print_table(results: List[dict]) -> None:
    header = (
        f"{'case':<16} {'arm':<15} {'mem':<5} {'sim':<4} {'tools':<5} "
        f"{'llm':<4} {'lat_ms':<9} {'tokens':<7} {'assess':<14} {'final':<7}"
    )
    print(header)
    for r in results:
        print(
            f"{r['case_id'][:14]:<16} {r['arm']:<15} {str(r['memory_used']):<5} "
            f"{r['similar_event_count']:<4} {r['tool_calls']:<5} {r['llm_calls']:<4} "
            f"{r['latency_ms']:<9} {r['total_tokens']:<7} "
            f"{str(r['agent_assessment']):<14} {str(r['final_level']):<7}"
        )
        if r["error"]:
            print(f"    error: {r['error'][:200]}")
    print("-" * 100)


def main() -> int:
    filters: List[str] = []
    if len(sys.argv) > 1:
        filters = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    cases = [c for c in CASES if not filters or c["case_id"] in filters]
    print(f"Phase 3-B Memory A/B 真实 API：{len(cases)} 案例 × 2 臂 = {len(cases) * 2} 次实验")

    results: List[dict] = []
    for case in cases:
        for with_memory in (False, True):
            label = "With Memory" if with_memory else "Without Memory"
            print(f"运行 {case['case_id']} / {label} ...", flush=True)
            results.append(run_agent_arm(case, with_memory))

    _print_table(results)
    out_path = _ROOT / "logs" / "v2_phase3_memory_ab_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "deepseek-v4-flash",
        "thinking": "disabled",
        "note": "Phase 3-B Memory A/B；fixture 历史仅用于可重复实验",
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
