# -*- coding: utf-8 -*-
"""V2 Agent - M10 报告 Agent Analysis 单元测试（Phase 4-A，Mock，不调用 API）。"""
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.deepseek_agent import AgentOutcome, AgentStatus
from agent.hybrid_decision import HybridDecision
from agent.presentation_models import (
    build_agent_presentation_model,
    extract_memory_results,
)
from agent.schemas import AgentResult
from reports.report_generator import ReportGenerator
from utils.common import (
    DangerLevel,
    EventReportData,
    EventStatus,
    FireDecision,
    FireEvent,
    GrowthTrend,
    load_config,
)


def _event(event_id="FE-RPT-AGENT-0001") -> FireEvent:
    return FireEvent(
        event_id=event_id,
        status=EventStatus.ENDED,
        duration_seconds=8.0,
        total_frames=40,
        positive_frames=28,
        max_fire_area_ratio=0.0,
        max_smoke_area_ratio=0.4,
        avg_fire_confidence=0.0,
        avg_smoke_confidence=0.8,
        growth_trend=GrowthTrend.STABLE,
        source_type="video",
        source_name="test.mp4",
    )


def _decision() -> FireDecision:
    return FireDecision(
        event_id="FE-RPT-AGENT-0001",
        danger_level=DangerLevel.MEDIUM,
        score=0.48,
        confidence=0.72,
        reasons=["持续烟雾，未见持续火焰"],
        decision_source="rule_engine",
    )


def _presentation():
    agent_result = AgentResult(
        event_id="FE-RPT-AGENT-0001",
        agent_assessment="SUSPICIOUS",
        risk_level="medium",
        agreement_with_rule=True,
        possible_cause="疑似非火灾烟雾",
        reasoning_summary="烟雾特征明显但火焰证据不足",
        recommended_action="继续观察并人工核查",
        tools_used=["get_event_summary", "get_similar_events"],
        agent_confidence=0.78,
        memory_used=True,
        similar_event_count=3,
        memory_summary="检索到3个相似历史事件，其中2个为短时烟雾场景",
    )
    outcome = AgentOutcome(
        status=AgentStatus.OK,
        agent_result=agent_result,
        rounds=3,
        latency_ms=9500.0,
        token_usage={"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
        tool_calls=[
            {
                "name": "get_similar_events",
                "success": True,
                "result_summary": {
                    "data": {
                        "results": [
                            {"event_id": "H-1", "similarity_score": 0.88},
                            {"event_id": "H-2", "similarity_score": 0.82},
                            {"event_id": "H-3", "similarity_score": 0.76},
                        ]
                    }
                },
            }
        ],
    )
    hybrid = HybridDecision(
        rule_level="medium", agent_level="medium", final_level="medium",
        agreement=True, override=False, requires_review=False,
        reason="Agent=medium 与 M7=medium 一致", agent_status="ok",
    )
    return build_agent_presentation_model(
        "FE-RPT-AGENT-0001",
        _decision(),
        outcome,
        hybrid,
        memory_results=extract_memory_results(outcome),
    )


class TestM10AgentReport(unittest.TestCase):
    def _generator(self, tmp: Path) -> ReportGenerator:
        cfg = load_config()
        cfg["report"] = {
            "output_dir": str(tmp / "reports"),
            "formats": ["md", "pdf"],
        }
        return ReportGenerator(config=cfg)

    def test_report_with_agent_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = self._generator(Path(tmp))
            data = EventReportData(
                event=_event(),
                decisions=[_decision()],
                screenshots=[],
            )
            paths = gen.generate(data, agent_presentation=_presentation())
            self.assertTrue(paths)
            md = next(p for p in paths if p.suffix == ".md")
            text = md.read_text(encoding="utf-8")
            self.assertIn("## 6. Agent 分析", text)
            self.assertIn("Agent Status", text)
            self.assertIn("SUSPICIOUS", text)
            self.assertIn("Used (3)", text)
            self.assertIn("检索到3个相似历史事件", text)
            self.assertIn("H-1", text)
            self.assertIn("Final Decision", text)
            self.assertIn("Tools Used", text)
            event_json = next(p for p in paths if p.suffix == ".json")
            import json

            payload = json.loads(event_json.read_text(encoding="utf-8"))
            self.assertIsNotNone(payload["agent_analysis"])
            self.assertEqual(payload["agent_analysis"]["final_level"], "medium")

    def test_old_event_without_agent_still_generates(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = self._generator(Path(tmp))
            data = EventReportData(
                event=_event("FE-RPT-OLD-0001"),
                decisions=[_decision()],
                screenshots=[],
            )
            paths = gen.generate(data, agent_presentation=None)
            md = next(p for p in paths if p.suffix == ".md")
            text = md.read_text(encoding="utf-8")
            self.assertIn("Agent Analysis: Not available for this event.", text)
            self.assertIn("Rule Decision (M7)", text)
            event_json = next(p for p in paths if p.suffix == ".json")
            import json

            payload = json.loads(event_json.read_text(encoding="utf-8"))
            self.assertIsNone(payload["agent_analysis"])

    def test_pdf_generated_if_font_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = self._generator(Path(tmp))
            data = EventReportData(event=_event(), decisions=[_decision()], screenshots=[])
            paths = gen.generate(data, agent_presentation=_presentation())
            pdf = next((p for p in paths if p.suffix == ".pdf"), None)
            if gen.pdf_available:
                self.assertIsNotNone(pdf)
                self.assertGreater(pdf.stat().st_size, 0)
            else:
                self.assertIsNone(pdf)


if __name__ == "__main__":
    unittest.main()
