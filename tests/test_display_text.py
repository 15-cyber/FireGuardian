# -*- coding: utf-8 -*-
"""V2 Agent - 展示层空字段 fallback 单元测试（Phase 2.1）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.display_text import format_agent_result_for_display
from agent.schemas import AgentResult


class TestDisplayTextFallback(unittest.TestCase):
    def test_empty_fields_get_fallback_text(self):
        result = AgentResult(
            event_id="FE-001",
            agent_assessment="SUSPICIOUS",
            risk_level="medium",
            agent_confidence=0.0,
        )
        display = format_agent_result_for_display(result)
        self.assertEqual(display["possible_cause"], "暂无明确判断（证据不足或 Agent 未给出结论）")
        self.assertEqual(display["reasoning_summary"], "Agent 未给出推理摘要（可查看工具调用与证据记录）")
        self.assertEqual(display["recommended_action"], "未给出处置建议，请按当前风险等级执行标准流程")
        self.assertEqual(display["evidence_summary"], ["无引用证据（可查看截图/历史记录）"])
        self.assertEqual(display["tools_used"], ["未调用工具"])
        self.assertEqual(display["agent_confidence_display"], "未提供")
        self.assertEqual(
            set(display["_display_fallbacks"]),
            {"possible_cause", "reasoning_summary", "recommended_action",
             "evidence_summary", "tools_used", "agent_confidence"},
        )

    def test_original_result_unchanged(self):
        result = AgentResult(event_id="FE-001", risk_level="low", agent_confidence=0.0)
        before = result.to_dict()
        format_agent_result_for_display(result)
        self.assertEqual(result.to_dict(), before)

    def test_filled_fields_keep_original_values(self):
        result = AgentResult(
            event_id="FE-001",
            agent_assessment="CONFIRMED_RISK",
            risk_level="high",
            possible_cause="真实火灾",
            evidence_summary=["火焰持续增长"],
            reasoning_summary="火焰面积与时长均超阈值",
            recommended_action="立即处置",
            tools_used=["get_event_summary", "get_rule_decision"],
            agent_confidence=0.93,
        )
        display = format_agent_result_for_display(result)
        self.assertEqual(display["possible_cause"], "真实火灾")
        self.assertEqual(display["evidence_summary"], ["火焰持续增长"])
        self.assertEqual(display["reasoning_summary"], "火焰面积与时长均超阈值")
        self.assertEqual(display["recommended_action"], "立即处置")
        self.assertEqual(display["tools_used"], ["get_event_summary", "get_rule_decision"])
        self.assertEqual(display["agent_confidence_display"], "0.93")
        self.assertEqual(display["_display_fallbacks"], [])

    def test_schema_roundtrip_still_valid(self):
        result = AgentResult(event_id="FE-001", risk_level="medium", agent_confidence=0.5)
        display = format_agent_result_for_display(result)
        # 展示 dict 只应包含 AgentResult 的合法字段 + 展示辅助字段
        legal = set(AgentResult().__dict__.keys())
        extra = set(display.keys()) - legal - {"_display_fallbacks", "agent_confidence_display"}
        self.assertEqual(extra, set())


if __name__ == "__main__":
    unittest.main()
