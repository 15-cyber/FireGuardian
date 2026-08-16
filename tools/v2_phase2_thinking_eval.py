"""
============================================================
FireGuardian V2 Phase 2 - Thinking Mode A/B Evaluation（支线实验）

对固定事件案例分别执行：
  A: thinking disabled（生产默认）
  B: thinking enabled

记录成功率 / 延迟 / Token 成本 / Agent 结果质量，以及
reasoning_content 是否保存与回传、thinking enabled + tool calling
是否出现 400 / 兼容问题（保留原始错误，不修改主流程规避）。

实验结论不自动改变生产默认（生产仍为 thinking disabled）。

用法：
    python tools/v2_phase2_thinking_eval.py [case_id,...]   # 可选过滤
结果保存：logs/v2_phase2_thinking_eval_results.json
============================================================
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.deepseek_agent import FireGuardianLLMAgent
from agent.deepseek_client import DeepSeekClient
from agent.tool_executor import InMemoryDataSource
from utils.common import DangerLevel, EventStatus, FireDecision, FireEvent, GrowthTrend


# ---- 固定事件案例（至少覆盖 real fire / smoke only / fireworks / normal / rule medium / rule high）----
CASES: List[dict] = [
    {
        "case_id": "case_01_real_fire",
        "event": dict(
            status=EventStatus.CONFIRMED,
            duration_seconds=30.0,
            total_frames=120,
            positive_frames=100,
            max_fire_area_ratio=0.35,
            max_smoke_area_ratio=0.42,
            avg_fire_confidence=0.92,
            avg_smoke_confidence=0.88,
            growth_trend=GrowthTrend.INCREASING,
        ),
        "decision": dict(level=DangerLevel.HIGH, score=0.82, confidence=0.9,
                         reasons=["火焰面积大且持续增长"]),
    },
    {
        "case_id": "case_02_smoke_only",
        "event": dict(
            status=EventStatus.CONFIRMED,
            duration_seconds=12.0,
            total_frames=60,
            positive_frames=45,
            max_fire_area_ratio=0.0,
            max_smoke_area_ratio=0.55,
            avg_fire_confidence=0.0,
            avg_smoke_confidence=0.85,
            growth_trend=GrowthTrend.STABLE,
        ),
        "decision": dict(level=DangerLevel.MEDIUM, score=0.5, confidence=0.72,
                         reasons=["持续烟雾，未见火焰"]),
    },
    {
        "case_id": "case_03_fireworks",
        "event": dict(
            status=EventStatus.CONFIRMED,
            duration_seconds=5.0,
            total_frames=30,
            positive_frames=20,
            max_fire_area_ratio=0.02,
            max_smoke_area_ratio=0.50,
            avg_fire_confidence=0.35,
            avg_smoke_confidence=0.80,
            growth_trend=GrowthTrend.DECREASING,
        ),
        "decision": dict(level=DangerLevel.MEDIUM, score=0.45, confidence=0.7,
                         reasons=["短暂亮光+烟雾，需人工确认"]),
    },
    {
        "case_id": "case_04_normal_scene",
        "event": dict(
            status=EventStatus.CONFIRMED,
            duration_seconds=1.5,
            total_frames=12,
            positive_frames=4,
            max_fire_area_ratio=0.0,
            max_smoke_area_ratio=0.0,
            avg_fire_confidence=0.0,
            avg_smoke_confidence=0.0,
            growth_trend=GrowthTrend.UNKNOWN,
        ),
        "decision": dict(level=DangerLevel.LOW, score=0.1, confidence=0.9,
                         reasons=["无有效目标"]),
    },
    {
        "case_id": "case_05_rule_medium",
        "event": dict(
            status=EventStatus.CONFIRMED,
            duration_seconds=8.0,
            total_frames=40,
            positive_frames=26,
            max_fire_area_ratio=0.08,
            max_smoke_area_ratio=0.22,
            avg_fire_confidence=0.68,
            avg_smoke_confidence=0.76,
            growth_trend=GrowthTrend.STABLE,
        ),
        "decision": dict(level=DangerLevel.MEDIUM, score=0.52, confidence=0.74,
                         reasons=["中等面积烟雾"]),
    },
    {
        "case_id": "case_06_rule_high",
        "event": dict(
            status=EventStatus.CONFIRMED,
            duration_seconds=25.0,
            total_frames=100,
            positive_frames=85,
            max_fire_area_ratio=0.22,
            max_smoke_area_ratio=0.34,
            avg_fire_confidence=0.87,
            avg_smoke_confidence=0.82,
            growth_trend=GrowthTrend.INCREASING,
        ),
        "decision": dict(level=DangerLevel.HIGH, score=0.75, confidence=0.86,
                         reasons=["火焰+烟雾持续增长"]),
    },
]


def _data_source(event_id: str, case: dict) -> InMemoryDataSource:
    ev = case["event"]
    return InMemoryDataSource(
        summaries={
            event_id: {
                "found": True,
                "event_id": event_id,
                "summary": {
                    "status": "confirmed",
                    "duration_seconds": ev["duration_seconds"],
                    "total_frames": ev["total_frames"],
                    "positive_frames": ev["positive_frames"],
                    "positive_ratio": round(ev["positive_frames"] / ev["total_frames"], 4),
                    "max_fire_area_ratio": ev["max_fire_area_ratio"],
                    "max_smoke_area_ratio": ev["max_smoke_area_ratio"],
                    "avg_fire_confidence": ev["avg_fire_confidence"],
                    "avg_smoke_confidence": ev["avg_smoke_confidence"],
                    "growth_trend": ev["growth_trend"].value,
                },
            }
        },
        decisions={
            event_id: {
                "found": True,
                "event_id": event_id,
                "decision": {"level": case["decision"]["level"].value},
            }
        },
        histories={
            event_id: {
                "event_id": event_id,
                "window_seconds": 30.0,
                "snapshots": [
                    {"timestamp_seconds": 1.0, "growth_trend": ev["growth_trend"].value},
                    {"timestamp_seconds": 5.0, "growth_trend": ev["growth_trend"].value},
                ],
            }
        },
        evidence={event_id: {"found": False, "event_id": event_id}},
    )


def _run_case(case: dict, mode: str) -> dict:
    case_id = case["case_id"]
    event_id = f"EV-{case_id}"
    ev = case["event"]
    decision = case["decision"]
    event = FireEvent(event_id=event_id, **ev)
    fire_decision = FireDecision(
        event_id=event_id,
        danger_level=decision["level"],
        score=decision["score"],
        confidence=decision["confidence"],
        reasons=decision["reasons"],
    )
    logger = AgentLogger(
        decisions_path=_ROOT / "logs" / "thinking_eval" / "agent_decisions.jsonl",
        tool_calls_path=_ROOT / "logs" / "thinking_eval" / "agent_tool_calls.jsonl",
    )
    record: Dict = {
        "case_id": case_id,
        "thinking_mode": mode,
        "status": None,
        "tool_call_success": None,
        "tool_call_count": 0,
        "final_json_success": False,
        "schema_success": False,
        "latency_ms": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "final_level": None,
        "agent_assessment": None,
        "reasoning_available": False,
        "reasoning_echoed": False,
        "error": None,
    }
    try:
        agent = FireGuardianLLMAgent(
            client=DeepSeekClient(),
            data_source=_data_source(event_id, case),
            max_tool_rounds=3,
            thinking_disabled=(mode == "disabled"),
            logger=logger,
        )
        outcome = agent.analyze(event, fire_decision, trigger="event_confirmed")
        record.update(
            {
                "status": outcome.status.value,
                "tool_call_success": bool(outcome.tool_calls)
                and all(tc["success"] for tc in outcome.tool_calls),
                "tool_call_count": len(outcome.tool_calls),
                "final_json_success": outcome.final_json_success,
                "schema_success": outcome.schema_success,
                "latency_ms": round(outcome.latency_ms, 1),
                "prompt_tokens": outcome.token_usage.get("prompt_tokens", 0),
                "completion_tokens": outcome.token_usage.get("completion_tokens", 0),
                "total_tokens": outcome.token_usage.get("total_tokens", 0),
                "final_level": outcome.agent_level,
                "agent_assessment": outcome.agent_assessment,
                "reasoning_available": outcome.reasoning_seen,
                "reasoning_echoed": outcome.reasoning_echoed,
                "error": outcome.error,
            }
        )
    except Exception as exc:  # noqa: BLE001 - 实验不能因单 case 崩溃
        record["status"] = "unexpected_error"
        record["error"] = str(exc)
    return record


def _print_summary(results: List[dict]) -> None:
    print("\nThinking Mode A/B 实验结果汇总")
    print("-" * 120)
    header = (
        f"{'case_id':<20} {'mode':<9} {'status':<16} {'tools':<5} "
        f"{'json':<5} {'schema':<6} {'latency_ms':<11} {'tokens':<7} "
        f"{'level':<7} {'assessment':<14} {'reasoning':<10}"
    )
    print(header)
    for r in results:
        print(
            f"{r['case_id']:<20} {r['thinking_mode']:<9} {str(r['status']):<16} "
            f"{r['tool_call_count']:<5} {str(r['final_json_success']):<5} "
            f"{str(r['schema_success']):<6} {str(r['latency_ms']):<11} "
            f"{r['total_tokens']:<7} {str(r['final_level']):<7} "
            f"{str(r['agent_assessment']):<14} {str(r['reasoning_available']):<10}"
        )
        if r["error"]:
            print(f"    error: {r['error'][:300]}")
    print("-" * 120)


def main() -> int:
    filters: List[str] = []
    if len(sys.argv) > 1:
        filters = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    cases = [c for c in CASES if not filters or c["case_id"] in filters]
    print(f"Thinking Mode A/B Evaluation：{len(cases)} 个案例 × 2 种模式")
    print(f"过滤: {filters or '全部'}")

    results: List[dict] = []
    for case in cases:
        for mode in ("disabled", "enabled"):
            print(f"运行 {case['case_id']} / thinking={mode} ...", flush=True)
            results.append(_run_case(case, mode))

    _print_summary(results)
    out_path = _ROOT / "logs" / "v2_phase2_thinking_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "deepseek-v4-flash",
        "note": "Thinking A/B 实验；结论不自动改变生产默认（生产 thinking=disabled）",
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
