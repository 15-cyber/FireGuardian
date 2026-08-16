# -*- coding: utf-8 -*-
"""V2 Agent - Schemas 单元测试（Phase 1）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.schemas import (
    AgentAssessment,
    AgentResult,
    SchemaValidationError,
    ToolCall,
    ToolResult,
)


class TestAgentResultValidation(unittest.TestCase):
    def test_valid_from_dict(self):
        result = AgentResult.from_dict(
            {
                "event_id": "FE-001",
                "agent_assessment": "SUSPICIOUS",
                "risk_level": "medium",
                "agreement_with_rule": True,
                "possible_cause": "疑似烟花",
                "evidence_summary": ["持续烟雾"],
                "reasoning_summary": "烟雾明显但火焰证据不足",
                "recommended_action": "人工核查",
                "uncertainty": "medium",
                "tools_used": ["get_event_summary"],
                "agent_confidence": 0.78,
            }
        )
        self.assertEqual(result.event_id, "FE-001")
        self.assertEqual(result.agent_assessment, AgentAssessment.SUSPICIOUS.value)
        self.assertEqual(result.risk_level, "medium")
        self.assertAlmostEqual(result.agent_confidence, 0.78)
        result.validate()

    def test_invalid_assessment_rejected(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict(
                {"event_id": "FE-001", "agent_assessment": "TOTALLY_FINE", "risk_level": "low"}
            )

    def test_invalid_risk_level_rejected(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict({"event_id": "FE-001", "risk_level": "extreme"})

    def test_invalid_uncertainty_rejected(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict({"event_id": "FE-001", "uncertainty": "unknown"})

    def test_confidence_out_of_range_rejected(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict({"event_id": "FE-001", "agent_confidence": 1.5})

    def test_confidence_non_numeric_rejected(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict({"event_id": "FE-001", "agent_confidence": "high"})

    def test_empty_event_id_rejected(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict({"event_id": "  "})

    def test_missing_fields_use_defaults(self):
        result = AgentResult.from_dict({"event_id": "FE-001"})
        self.assertEqual(result.agent_assessment, AgentAssessment.UNCERTAIN.value)
        self.assertTrue(result.agreement_with_rule)
        self.assertEqual(result.tools_used, [])
        self.assertFalse(result.memory_used)
        self.assertEqual(result.similar_event_count, 0)
        self.assertEqual(result.memory_summary, "")

    def test_non_dict_rejected(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict(["not", "a", "dict"])

    def test_to_dict_roundtrip(self):
        data = {
            "event_id": "FE-002",
            "agent_assessment": "CONFIRMED_RISK",
            "risk_level": "high",
            "agreement_with_rule": True,
            "possible_cause": "真实火灾",
            "evidence_summary": ["火焰持续增长"],
            "reasoning_summary": "火焰面积与时长均超过阈值",
            "recommended_action": "立即处置",
            "uncertainty": "low",
            "tools_used": ["get_rule_decision"],
            "agent_confidence": 0.93,
            "memory_used": True,
            "similar_event_count": 3,
            "memory_summary": "检索到3个相似历史事件",
        }
        self.assertEqual(AgentResult.from_dict(data).to_dict(), data)

    def test_memory_fields_valid(self):
        result = AgentResult.from_dict(
            {
                "event_id": "FE-001",
                "memory_used": True,
                "similar_event_count": 5,
                "memory_summary": "检索到5个相似历史事件",
            }
        )
        self.assertTrue(result.memory_used)
        self.assertEqual(result.similar_event_count, 5)

    def test_memory_used_must_be_bool(self):
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict({"event_id": "FE-001", "memory_used": "yes"})

    def test_similar_event_count_must_be_non_negative_int(self):
        for bad in (-1, 2.5, "3"):
            with self.assertRaises(SchemaValidationError):
                AgentResult.from_dict(
                    {"event_id": "FE-001", "similar_event_count": bad}
                )

    def test_memory_summary_must_be_string(self):
        # JSON null → 默认空字符串；其他错误类型 → 报错
        result = AgentResult.from_dict({"event_id": "FE-001", "memory_summary": None})
        self.assertEqual(result.memory_summary, "")
        with self.assertRaises(SchemaValidationError):
            AgentResult.from_dict({"event_id": "FE-001", "memory_summary": 123})

    def test_memory_fields_null_treated_as_defaults(self):
        result = AgentResult.from_dict(
            {
                "event_id": "FE-001",
                "memory_used": None,
                "similar_event_count": None,
                "memory_summary": None,
            }
        )
        self.assertFalse(result.memory_used)
        self.assertEqual(result.similar_event_count, 0)
        self.assertEqual(result.memory_summary, "")


class TestToolDataclasses(unittest.TestCase):
    def test_tool_call(self):
        call = ToolCall(name="get_event_summary", arguments={"event_id": "FE-001"}, call_id="c1")
        self.assertEqual(call.to_dict()["name"], "get_event_summary")
        self.assertEqual(call.to_dict()["arguments"], {"event_id": "FE-001"})

    def test_tool_result(self):
        ok = ToolResult(tool_name="get_event_summary", success=True, data={"found": True})
        self.assertTrue(ok.success)
        self.assertIsNone(ok.error)
        failed = ToolResult(tool_name="x", success=False, error="boom")
        self.assertFalse(failed.success)
        self.assertEqual(failed.to_dict()["error"], "boom")


if __name__ == "__main__":
    unittest.main()
