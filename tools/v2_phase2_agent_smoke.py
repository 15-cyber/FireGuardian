"""
============================================================
FireGuardian V2 Phase 2 - Agent + Hybrid 真实 API 冒烟测试

链路：
  M7 Rule Decision → DeepSeek Agent（thinking disabled）→ Tool Calling
  → Agent Analysis → HybridCoordinator → Final Decision

用法：
    python tools/v2_phase2_agent_smoke.py

安全约束：
  - 真实 API 只在手动执行时调用；不打印 API Key；不写敏感 prompt；
  - Agent 决策日志写入 logs/agent_decisions.jsonl（无 Key / prompt）。
============================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.deepseek_agent import AgentStatus, FireGuardianLLMAgent
from agent.deepseek_client import DeepSeekClient
from agent.hybrid_decision import run_hybrid_pipeline
from agent.tool_executor import InMemoryDataSource
from utils.common import DangerLevel, EventStatus, FireDecision, FireEvent, GrowthTrend


def _mock_event() -> FireEvent:
    return FireEvent(
        event_id="FE-P2-SMOKE-0001",
        status=EventStatus.CONFIRMED,
        start_time_seconds=0.0,
        last_time_seconds=8.2,
        duration_seconds=8.2,
        total_frames=40,
        positive_frames=28,
        max_fire_area_ratio=0.09,
        max_smoke_area_ratio=0.46,
        avg_fire_confidence=0.66,
        avg_smoke_confidence=0.83,
        growth_trend=GrowthTrend.STABLE,
        source_type="smoke",
        source_name="phase2_smoke",
    )


def _mock_decision() -> FireDecision:
    return FireDecision(
        event_id="FE-P2-SMOKE-0001",
        danger_level=DangerLevel.MEDIUM,
        score=0.52,
        confidence=0.74,
        reasons=["持续烟雾，火焰证据不足"],
        decision_source="rule_engine",
    )


def _data_source(event_id: str) -> InMemoryDataSource:
    return InMemoryDataSource(
        summaries={
            event_id: {
                "found": True,
                "event_id": event_id,
                "summary": {
                    "status": "confirmed",
                    "duration_seconds": 8.2,
                    "total_frames": 40,
                    "positive_frames": 28,
                    "positive_ratio": 0.7,
                    "max_fire_area_ratio": 0.09,
                    "max_smoke_area_ratio": 0.46,
                    "avg_fire_confidence": 0.66,
                    "avg_smoke_confidence": 0.83,
                    "growth_trend": "stable",
                },
            }
        },
        decisions={
            event_id: {
                "found": True,
                "event_id": event_id,
                "decision": {"level": "medium", "score": 0.52, "confidence": 0.74},
            }
        },
        histories={
            event_id: {
                "event_id": event_id,
                "window_seconds": 30.0,
                "snapshots": [
                    {"timestamp_seconds": 2.0, "max_smoke_area_ratio": 0.3},
                    {"timestamp_seconds": 5.0, "max_smoke_area_ratio": 0.4},
                    {"timestamp_seconds": 8.2, "max_smoke_area_ratio": 0.46},
                ],
            }
        },
        evidence={event_id: {"found": False, "event_id": event_id}},
    )


def main() -> int:
    print("FireGuardian V2 Phase 2 - Agent + Hybrid 真实 API 冒烟测试")
    print("=" * 64)

    client = DeepSeekClient()
    print(f"model    : {client.model}")
    print(f"base_url : {client.base_url}")
    print(f"api_key  : {client._masked_key}")
    print("-" * 64)

    event = _mock_event()
    decision = _mock_decision()
    agent = FireGuardianLLMAgent(
        client=client,
        data_source=_data_source(event.event_id),
        max_tool_rounds=3,
        thinking_disabled=True,
        logger=AgentLogger(),
    )

    outcome = agent.analyze(event, decision, trigger="event_confirmed")
    hybrid = run_hybrid_pipeline(
        rule_level=decision.danger_level,
        outcome=outcome,
        event_id=event.event_id,
        trigger="event_confirmed",
        model=client.model,
        logger=agent.logger,
    )

    print(f"Agent 状态   : {outcome.status.value}")
    print(f"LLM 调用数   : {outcome.rounds}")
    print(f"工具调用数   : {len(outcome.tool_calls)}")
    print(f"总耗时       : {outcome.latency_ms:.1f}ms")
    print(f"Token        : {outcome.token_usage}")
    if outcome.status == AgentStatus.OK:
        print("Agent 结果（已通过 Schema Validation）:")
        print(json.dumps(outcome.agent_result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Agent 失败   : {outcome.error}")
    print("-" * 64)
    print(f"Hybrid       : {json.dumps(hybrid.to_dict(), ensure_ascii=False)}")
    print(f"Final Level  : {hybrid.final_level}")
    print("-" * 64)
    ok = outcome.status == AgentStatus.OK
    print(f"PHASE2_AGENT_SMOKE_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
