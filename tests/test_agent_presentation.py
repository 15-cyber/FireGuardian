# -*- coding: utf-8 -*-
"""V2 Agent - Presentation Model / Trigger Guard / Timeout 单元测试（Phase 4-A，Mock）。"""
import sys
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.deepseek_agent import AgentOutcome, AgentStatus
from agent.hybrid_decision import HybridDecision
from agent.presentation_models import (
    AgentTriggerGuard,
    build_agent_presentation_model,
    extract_memory_results,
    run_agent_with_timeout,
)
from agent.schemas import AgentResult
from utils.common import DangerLevel, FireDecision


def _decision(level="medium"):
    return FireDecision(
        event_id="FE-P4-001",
        danger_level=getattr(DangerLevel, level.upper(), level),
        score=0.5,
        confidence=0.72,
        reasons=["持续烟雾"],
    )


def _outcome(agent_result=None, status=AgentStatus.OK, tool_calls=None):
    return AgentOutcome(
        status=status,
        agent_result=agent_result,
        error=None if status == AgentStatus.OK else "error",
        tool_calls=tool_calls or [],
        rounds=2,
        latency_ms=1234.5,
        token_usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        final_json_success=True,
        schema_success=True,
    )


def _memory_tool_call():
    return {
        "name": "get_similar_events",
        "arguments": {"event_id": "FE-P4-001", "top_k": 3},
        "success": True,
        "result_summary": {
            "tool_name": "get_similar_events",
            "success": True,
            "data": {
                "results": [
                    {"event_id": "H-1", "similarity_score": 0.88},
                    {"event_id": "H-2", "similarity_score": 0.82},
                ]
            },
        },
    }


class TestBuildPresentationModel(unittest.TestCase):
    def test_full_fields(self):
        agent_result = AgentResult(
            event_id="FE-P4-001",
            agent_assessment="SUSPICIOUS",
            risk_level="medium",
            agreement_with_rule=True,
            possible_cause="疑似非火灾烟雾",
            evidence_summary=["持续烟雾无火焰"],
            reasoning_summary="烟雾特征明显但火焰证据不足",
            recommended_action="继续观察并人工核查",
            uncertainty="medium",
            tools_used=["get_event_summary", "get_similar_events"],
            agent_confidence=0.78,
            memory_used=True,
            similar_event_count=2,
            memory_summary="检索到2个相似历史事件",
        )
        outcome = _outcome(agent_result, tool_calls=[_memory_tool_call()])
        hybrid = HybridDecision(
            rule_level="medium", agent_level="medium", final_level="medium",
            agreement=True, override=False, requires_review=False,
            reason="一致", agent_status="ok",
        )
        model = build_agent_presentation_model(
            "FE-P4-001", _decision(), outcome, hybrid
        )
        self.assertEqual(model.agent_status, "ok")
        self.assertEqual(model.agent_assessment, "SUSPICIOUS")
        self.assertEqual(model.rule_level, "medium")
        self.assertEqual(model.final_level, "medium")
        self.assertTrue(model.memory_used)
        self.assertEqual(model.similar_event_count, 2)
        self.assertEqual(model.possible_cause, "疑似非火灾烟雾")
        self.assertEqual(model.latency_ms, 1234.5)
        self.assertTrue(model.agent_available)

    def test_empty_fields_use_display_fallback(self):
        outcome = _outcome(AgentResult(event_id="FE-P4-001", risk_level="low"))
        hybrid = HybridDecision(rule_level="low", agent_level="low", final_level="low")
        model = build_agent_presentation_model("FE-P4-001", _decision("low"), outcome, hybrid)
        self.assertIn("暂无明确判断", model.possible_cause)
        self.assertIn("未给出处置建议", model.recommended_action)
        self.assertEqual(model.evidence_summary, ["无引用证据（可查看截图/历史记录）"])
        self.assertEqual(model.tools_used, ["未调用工具"])

    def test_unavailable_status(self):
        outcome = _outcome(None, status=AgentStatus.UNAVAILABLE)
        hybrid = HybridDecision(
            rule_level="high", final_level="high", agent_status="unavailable"
        )
        model = build_agent_presentation_model("FE-P4-001", _decision("high"), outcome, hybrid)
        self.assertEqual(model.status_label, "UNAVAILABLE")
        self.assertFalse(model.agent_available)
        self.assertEqual(model.final_level, "high")  # M7 兜底

    def test_invalid_output_status(self):
        outcome = _outcome(None, status=AgentStatus.INVALID_OUTPUT)
        hybrid = HybridDecision(
            rule_level="medium", final_level="medium", agent_status="invalid_output"
        )
        model = build_agent_presentation_model("FE-P4-001", _decision(), outcome, hybrid)
        self.assertEqual(model.status_label, "INVALID_OUTPUT")

    def test_high_never_downgraded_in_presentation(self):
        agent_result = AgentResult(event_id="FE-P4-001", agent_assessment="NORMAL", risk_level="low")
        outcome = _outcome(agent_result)
        hybrid = HybridDecision(
            rule_level="high", agent_level="low", final_level="high",
            agreement=False, override=False, reason="M7 High 不可被 Agent 降级",
        )
        model = build_agent_presentation_model("FE-P4-001", _decision("high"), outcome, hybrid)
        self.assertEqual(model.final_level, "high")
        self.assertEqual(model.agent_risk_level, "low")

    def test_extract_memory_results(self):
        outcome = _outcome(AgentResult(event_id="FE-P4-001"), tool_calls=[_memory_tool_call()])
        results = extract_memory_results(outcome)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["event_id"], "H-1")
        self.assertAlmostEqual(results[0]["similarity_score"], 0.88)


class TestAgentTriggerGuard(unittest.TestCase):
    def setUp(self):
        self.guard = AgentTriggerGuard(max_calls_per_event=4)

    def test_confirmed_once(self):
        self.assertTrue(self.guard.allowed("E1", "event_confirmed", current_level="medium"))
        self.guard.record("E1", "event_confirmed", current_level="medium")
        self.assertFalse(self.guard.allowed("E1", "event_confirmed", current_level="medium"))
        self.assertEqual(self.guard.count("E1"), 1)

    def test_risk_upgrade_level_based(self):
        self.assertTrue(self.guard.allowed("E1", "risk_upgrade", current_level="medium"))
        self.guard.record("E1", "risk_upgrade", current_level="medium")
        self.assertFalse(self.guard.allowed("E1", "risk_upgrade", current_level="medium"))
        self.assertTrue(self.guard.allowed("E1", "risk_upgrade", current_level="high"))
        self.guard.record("E1", "risk_upgrade", current_level="high")
        self.assertFalse(self.guard.allowed("E1", "risk_upgrade", current_level="high"))

    def test_ended_once_and_disagreement_once(self):
        self.guard.record("E1", "event_confirmed")
        self.assertTrue(self.guard.allowed("E1", "event_ended"))
        self.guard.record("E1", "event_ended")
        self.assertFalse(self.guard.allowed("E1", "event_ended"))
        self.assertTrue(self.guard.allowed("E1", "disagreement"))
        self.guard.record("E1", "disagreement")
        self.assertFalse(self.guard.allowed("E1", "disagreement"))

    def test_max_calls_cap(self):
        self.guard.record("E1", "event_confirmed")
        self.guard.record("E1", "risk_upgrade", current_level="medium")
        self.guard.record("E1", "disagreement")
        self.guard.record("E1", "event_ended")
        self.assertEqual(self.guard.count("E1"), 4)
        self.assertFalse(self.guard.allowed("E1", "event_confirmed"))

    def test_unknown_trigger_rejected(self):
        self.assertFalse(self.guard.allowed("E1", "frame_update"))


class _SlowAgent:
    def analyze(self, event, decision, trigger="event_confirmed"):
        time.sleep(5)
        return _outcome()


class TestRunAgentWithTimeout(unittest.TestCase):
    def test_timeout_returns_unavailable(self):
        outcome = run_agent_with_timeout(
            _SlowAgent(), None, None, timeout_seconds=0.3
        )
        self.assertEqual(outcome.status, AgentStatus.UNAVAILABLE)
        self.assertIn("timeout", outcome.error)

    def test_fast_agent_returns_result(self):
        class FastAgent:
            def analyze(self, event, decision, trigger="event_confirmed"):
                return _outcome(AgentResult(event_id="FE-X", risk_level="medium"))

        outcome = run_agent_with_timeout(FastAgent(), None, None, timeout_seconds=5)
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertEqual(outcome.agent_level, "medium")


if __name__ == "__main__":
    unittest.main()
