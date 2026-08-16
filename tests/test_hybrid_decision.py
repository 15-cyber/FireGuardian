# -*- coding: utf-8 -*-
"""V2 Agent - HybridDecisionCoordinator 单元测试（Phase 2）。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.deepseek_agent import AgentOutcome, AgentStatus
from agent.hybrid_decision import (
    HybridDecisionCoordinator,
    HybridValidationError,
    run_hybrid_pipeline,
)
from agent.schemas import AgentResult


class TestHybridDecisionCoordinator(unittest.TestCase):
    def setUp(self):
        self.coordinator = HybridDecisionCoordinator()

    def test_case1_high_agent_low(self):
        d = self.coordinator.decide("high", "low", "ok")
        self.assertEqual(d.final_level, "high")
        self.assertFalse(d.agreement)
        self.assertFalse(d.override)
        self.assertFalse(d.requires_review)
        self.assertIn("不可被 Agent 降级", d.reason)

    def test_case2_high_agent_high(self):
        d = self.coordinator.decide("high", "high", "ok")
        self.assertEqual(d.final_level, "high")
        self.assertTrue(d.agreement)
        self.assertFalse(d.override)

    def test_case3_medium_agent_high(self):
        d = self.coordinator.decide("medium", "high", "ok")
        self.assertEqual(d.final_level, "high")
        self.assertTrue(d.override)
        self.assertTrue(d.requires_review)
        self.assertFalse(d.agreement)

    def test_case4_medium_agent_medium(self):
        d = self.coordinator.decide("medium", "medium", "ok")
        self.assertEqual(d.final_level, "medium")
        self.assertTrue(d.agreement)
        self.assertFalse(d.override)
        self.assertFalse(d.requires_review)

    def test_case5_low_agent_high(self):
        d = self.coordinator.decide("low", "high", "ok")
        self.assertEqual(d.final_level, "high")
        self.assertTrue(d.override)
        self.assertTrue(d.requires_review)

    def test_case6_agent_unavailable(self):
        d = self.coordinator.decide("medium", None, "unavailable")
        self.assertEqual(d.final_level, "medium")
        self.assertEqual(d.agent_status, "unavailable")
        self.assertFalse(d.override)
        self.assertFalse(d.requires_review)
        self.assertIn("回退到 M7", d.reason)

    def test_case7_agent_invalid_json(self):
        d = self.coordinator.decide("high", None, "invalid_output")
        self.assertEqual(d.final_level, "high")
        self.assertEqual(d.agent_status, "invalid_output")

    def test_high_never_downgraded(self):
        for agent_level in ("low", "medium"):
            d = self.coordinator.decide("high", agent_level, "ok")
            self.assertEqual(d.final_level, "high")

    def test_high_not_downgraded_by_invalid_or_unavailable(self):
        for status in ("unavailable", "invalid_output"):
            d = self.coordinator.decide("high", None, status)
            self.assertEqual(d.final_level, "high")

    def test_low_low_and_low_medium(self):
        low_low = self.coordinator.decide("low", "low", "ok")
        self.assertEqual(low_low.final_level, "low")
        self.assertTrue(low_low.agreement)
        low_medium = self.coordinator.decide("low", "medium", "ok")
        self.assertEqual(low_medium.final_level, "medium")
        self.assertTrue(low_medium.override)
        self.assertFalse(low_medium.requires_review)

    def test_medium_low_rule_wins(self):
        d = self.coordinator.decide("medium", "low", "ok")
        self.assertEqual(d.final_level, "medium")
        self.assertFalse(d.agreement)
        self.assertFalse(d.override)

    def test_validation_errors(self):
        with self.assertRaises(HybridValidationError):
            self.coordinator.decide("extreme", "low", "ok")
        with self.assertRaises(HybridValidationError):
            self.coordinator.decide("low", "critical", "ok")
        with self.assertRaises(HybridValidationError):
            self.coordinator.decide("low", "low", "maybe")

    def test_to_dict_fields(self):
        d = self.coordinator.decide("medium", "high", "ok")
        data = d.to_dict()
        for key in (
            "rule_level",
            "agent_level",
            "final_level",
            "agreement",
            "override",
            "requires_review",
            "reason",
            "agent_status",
        ):
            self.assertIn(key, data)


class TestRunHybridPipeline(unittest.TestCase):
    def _ok_outcome(self, level="medium"):
        result = AgentResult(
            event_id="FE-001",
            agent_assessment="SUSPICIOUS",
            risk_level=level,
            tools_used=["get_event_summary"],
            agent_confidence=0.8,
        )
        return AgentOutcome(
            status=AgentStatus.OK,
            agent_result=result,
            latency_ms=1200.5,
            token_usage={"total_tokens": 400},
        )

    def test_pipeline_writes_decision_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = AgentLogger(
                decisions_path=Path(tmp) / "agent_decisions.jsonl",
                tool_calls_path=Path(tmp) / "agent_tool_calls.jsonl",
            )
            decision = run_hybrid_pipeline(
                "medium",
                self._ok_outcome("high"),
                event_id="FE-001",
                trigger="risk_upgrade",
                model="deepseek-v4-flash",
                logger=logger,
            )
            self.assertEqual(decision.final_level, "high")
            self.assertTrue(decision.requires_review)
            lines = (Path(tmp) / "agent_decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["event_id"], "FE-001")
            self.assertEqual(record["final_level"], "high")
            self.assertTrue(record["requires_review"])
            self.assertEqual(record["agent_status"], "ok")
            self.assertEqual(record["token_usage"]["total_tokens"], 400)
            self.assertNotIn("api_key", json.dumps(record).lower())
            self.assertNotIn("Authorization", json.dumps(record))


if __name__ == "__main__":
    unittest.main()
