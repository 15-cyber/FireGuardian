# -*- coding: utf-8 -*-
"""V2 Agent - Fallback 安全测试（Phase 2）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.deepseek_agent import AgentOutcome, AgentStatus, FireGuardianLLMAgent
from agent.deepseek_client import ChatResult, DeepSeekAPIError
from agent.hybrid_decision import run_hybrid_pipeline
from agent.tool_executor import InMemoryDataSource
from utils.common import EventStatus, FireDecision, FireEvent

from tests.test_deepseek_agent import FakeClient, _final_response, _make_data_source, _make_decision, _make_event


class TestFallback(unittest.TestCase):
    def test_agent_unavailable_final_is_m7(self):
        outcome = AgentOutcome(status=AgentStatus.UNAVAILABLE, error="timeout")
        decision = run_hybrid_pipeline("high", outcome, event_id="FE-001")
        self.assertEqual(decision.final_level, "high")
        self.assertEqual(decision.agent_status, "unavailable")

    def test_agent_invalid_output_final_is_m7(self):
        outcome = AgentOutcome(status=AgentStatus.INVALID_OUTPUT, error="bad json")
        decision = run_hybrid_pipeline("medium", outcome, event_id="FE-001")
        self.assertEqual(decision.final_level, "medium")
        self.assertEqual(decision.agent_status, "invalid_output")

    def test_agent_exception_does_not_propagate(self):
        client = FakeClient([DeepSeekAPIError("boom")])
        agent = FireGuardianLLMAgent(
            client=client,
            data_source=_make_data_source(),
            max_tool_rounds=3,
            thinking_disabled=True,
        )
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.UNAVAILABLE)
        decision = run_hybrid_pipeline("medium", outcome, event_id="FE-001")
        self.assertEqual(decision.final_level, "medium")

    def test_m7_rule_engine_keeps_running_after_agent_failure(self):
        # Agent 失败后，M7 规则引擎独立运行不受影响
        from agent.fire_decision_agent import FireDecisionAgent

        client = FakeClient([DeepSeekAPIError("boom")])
        agent = FireGuardianLLMAgent(client=client, data_source=_make_data_source())
        event = _make_event()
        outcome = agent.analyze(event, _make_decision())
        self.assertEqual(outcome.status, AgentStatus.UNAVAILABLE)

        m7 = FireDecisionAgent()
        decision = m7.analyze(event)
        self.assertEqual(decision.event_id, "FE-001")
        self.assertIn(decision.danger_level, ("low", "medium", "high"))
        self.assertGreaterEqual(decision.score, 0.0)

    def test_full_pipeline_ok(self):
        client = FakeClient([_final_response()])
        agent = FireGuardianLLMAgent(
            client=client,
            data_source=_make_data_source(),
            max_tool_rounds=3,
            thinking_disabled=True,
        )
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.OK)
        decision = run_hybrid_pipeline("medium", outcome, event_id="FE-001")
        self.assertEqual(decision.final_level, "medium")
        self.assertTrue(decision.agreement)


if __name__ == "__main__":
    unittest.main()
