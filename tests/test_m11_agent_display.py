# -*- coding: utf-8 -*-
"""M11 GUI - Agent 展示 offscreen 单元测试（Phase 4-B，Mock，不调用真实 API）。"""
import os
import sys
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt6.QtWidgets import QApplication

from agent.deepseek_agent import AgentOutcome, AgentStatus
from agent.hybrid_decision import HybridDecision
from agent.presentation_models import build_agent_presentation_model
from agent.schemas import AgentResult
from ui.app_controller import UiController
from ui.main_window import MainWindow
from utils.common import DangerLevel, FireDecision


def _presentation_payload(**overrides) -> dict:
    payload = {
        "event_id": "FE-GUI-001",
        "agent_status": "ok",
        "rule_level": "medium",
        "agent_assessment": "SUSPICIOUS",
        "agent_risk_level": "medium",
        "memory_used": True,
        "similar_event_count": 3,
        "memory_summary": "检索到3个相似历史事件",
        "memory_results": [
            {"event_id": "H-1", "similarity_score": 0.93},
            {"event_id": "H-2", "similarity_score": 0.91},
        ],
        "possible_cause": "疑似非火灾烟雾",
        "reasoning_summary": "烟雾特征明显但火焰证据不足",
        "recommended_action": "继续观察并人工核查",
        "uncertainty": "medium",
        "tools_used": ["get_event_summary", "get_similar_events"],
        "final_level": "medium",
        "reason": "Agent=medium 与 M7=medium 一致",
        "latency_ms": 9500.0,
        "token_usage": {"total_tokens": 700},
        "display_fallbacks": [],
        "error": None,
    }
    payload.update(overrides)
    return payload


class TestM11AgentDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.controller = UiController()
        cls.window = MainWindow(cls.controller)

    @classmethod
    def tearDownClass(cls):
        cls.window.close()
        cls.controller.shutdown()
        cls.window.deleteLater()
        cls.app.processEvents()

    def setUp(self):
        self.window.agent_panel.reset()

    def _emit(self, signal_name, payload):
        getattr(self.controller, signal_name).emit(payload)
        self.app.processEvents()

    def test_agent_ok_full_display(self):
        self._emit("agent_completed", _presentation_payload())
        panel = self.window.agent_panel
        self.assertIn("Completed", panel.status_label.text())
        self.assertIn("Rule: MEDIUM", panel.rule_label.text())
        self.assertIn("SUSPICIOUS", panel.agent_label.text())
        self.assertIn("Hybrid Final: MEDIUM", panel.final_label.text())
        self.assertIn("Used (3)", panel.memory_label.text())
        self.assertIn("H-1", panel.similar_label.text())
        self.assertIn("Model: 未提供", panel.detail_label.text())
        self.assertIn("Tokens: 700", panel.detail_label.text())
        self.assertTrue(panel.fallback_note.isHidden())

    def test_agent_unavailable_fallback(self):
        self._emit(
            "agent_completed",
            _presentation_payload(
                agent_status="unavailable",
                final_level="high",
                rule_level="high",
                memory_used=False,
                error="DeepSeek 不可用",
            ),
        )
        panel = self.window.agent_panel
        self.assertIn("Fallback", panel.status_label.text())
        self.assertIn("Hybrid Final: HIGH (M7)", panel.final_label.text())
        self.assertIn("Not Used", panel.memory_label.text())

    def test_agent_invalid_output_fallback(self):
        self._emit(
            "agent_completed",
            _presentation_payload(
                agent_status="invalid_output",
                final_level="medium",
                error="Schema 校验失败",
            ),
        )
        self.assertIn("Fallback", self.window.agent_panel.status_label.text())

    def test_agent_timeout(self):
        self._emit(
            "agent_completed",
            _presentation_payload(
                agent_status="unavailable",
                error="agent_timeout_exceeded (60s)",
            ),
        )
        self.assertIn("Timeout", self.window.agent_panel.status_label.text())

    def test_memory_not_used(self):
        self._emit(
            "agent_completed",
            _presentation_payload(memory_used=False, similar_event_count=0),
        )
        self.assertIn("Not Used", self.window.agent_panel.memory_label.text())

    def test_memory_unavailable(self):
        self._emit(
            "agent_completed",
            _presentation_payload(
                memory_used=True,
                similar_event_count=0,
                memory_summary="历史记忆不可用（history_not_found）",
            ),
        )
        self.assertIn("Unavailable", self.window.agent_panel.memory_label.text())

    def test_rule_high_agent_medium_final_high_distinct(self):
        self._emit(
            "agent_completed",
            _presentation_payload(
                rule_level="high",
                agent_assessment="UNCERTAIN",
                agent_risk_level="medium",
                final_level="high",
                reason="M7 High 不可被 Agent 降低",
            ),
        )
        panel = self.window.agent_panel
        self.assertIn("Rule: HIGH", panel.rule_label.text())
        self.assertIn("Agent: UNCERTAIN (MEDIUM)", panel.agent_label.text())
        self.assertIn("Hybrid Final: HIGH", panel.final_label.text())
        # 三个层级必须可区分
        self.assertNotEqual(panel.rule_label.text(), panel.agent_label.text())
        self.assertNotEqual(panel.agent_label.text(), panel.final_label.text())

    def test_rule_high_agent_low_final_high(self):
        self._emit(
            "agent_completed",
            _presentation_payload(
                rule_level="high",
                agent_assessment="NORMAL",
                agent_risk_level="low",
                final_level="high",
            ),
        )
        panel = self.window.agent_panel
        self.assertIn("Rule: HIGH", panel.rule_label.text())
        self.assertIn("Agent: NORMAL (LOW)", panel.agent_label.text())
        self.assertIn("Hybrid Final: HIGH", panel.final_label.text())

    def test_empty_fields_display_fallback(self):
        result = AgentResult(event_id="FE-GUI-001", risk_level="medium")
        outcome = AgentOutcome(
            status=AgentStatus.OK, agent_result=result, latency_ms=100.0
        )
        hybrid = HybridDecision(
            rule_level="medium", agent_level="medium", final_level="medium"
        )
        model = build_agent_presentation_model(
            "FE-GUI-001",
            FireDecision(event_id="FE-GUI-001", danger_level=DangerLevel.MEDIUM),
            outcome,
            hybrid,
        )
        self._emit("agent_completed", model.to_dict())
        panel = self.window.agent_panel
        self.assertIn("暂无明确判断", panel.cause_label.text())
        self.assertIn("未给出处置建议", panel.action_label.text())
        self.assertFalse(panel.fallback_note.isHidden())

    def test_old_event_history_without_agent(self):
        self.window._on_history_ready(
            [
                {
                    "event_id": "FE-OLD-001",
                    "start_time": "10:00:00",
                    "duration": 3.0,
                    "max_danger": "medium",
                    "report_status": "已生成",
                    "report_path": "",
                }
            ]
        )
        table = self.window.history_table
        self.assertEqual(table.item(0, 4).text(), "-")   # Agent
        self.assertEqual(table.item(0, 5).text(), "-")   # Final
        self.assertEqual(table.item(0, 6).text(), "-")   # Memory

    def test_history_with_agent_summary(self):
        self.window._on_history_ready(
            [
                {
                    "event_id": "FE-GUI-001",
                    "start_time": "10:00:00",
                    "duration": 3.0,
                    "max_danger": "high",
                    "rule_level": "high",
                    "agent_level": "medium",
                    "agent_status": "ok",
                    "final_level": "high",
                    "memory_used": True,
                    "similar_event_count": 2,
                    "report_status": "已生成",
                    "report_path": "",
                }
            ]
        )
        table = self.window.history_table
        self.assertEqual(table.item(0, 3).text(), "high")
        self.assertEqual(table.item(0, 4).text(), "MEDIUM")
        self.assertEqual(table.item(0, 5).text(), "HIGH")
        self.assertEqual(table.item(0, 6).text(), "Used (2)")

    def test_signals_started_tool_completed_failed(self):
        self._emit("agent_started", {"event_id": "FE-GUI-001", "trigger": "event_confirmed", "call_count": 1})
        self.assertIn("Analyzing", self.window.agent_panel.status_label.text())
        self._emit("agent_tool_called", {"event_id": "FE-GUI-001", "tool_name": "get_similar_events"})
        self.assertIn("get_similar_events", self.window.agent_panel.tool_label.text())
        self._emit("agent_completed", _presentation_payload())
        self.assertIn("Completed", self.window.agent_panel.status_label.text())
        self._emit("agent_failed", {"event_id": "FE-GUI-001", "status": "unavailable", "error": "timeout"})
        self.assertIn("Timeout", self.window.agent_panel.status_label.text())

    def test_close_with_agent_running(self):
        self._emit("agent_started", {"event_id": "FE-GUI-001", "trigger": "event_ended", "call_count": 2})
        self.window.close()
        self.controller.shutdown()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
