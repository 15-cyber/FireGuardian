# -*- coding: utf-8 -*-
"""V2 Agent - EventContext 单元测试（Phase 1）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.event_context import build_event_context, summarize_event_context
from utils.common import DangerLevel, EventStatus, FireDecision, FireEvent, GrowthTrend


def _sample_event() -> FireEvent:
    return FireEvent(
        event_id="FE-20260812-0001",
        status=EventStatus.CONFIRMED,
        start_time_seconds=0.0,
        last_time_seconds=2.73,
        duration_seconds=2.73,
        total_frames=20,
        positive_frames=15,
        max_fire_area_ratio=0.13,
        max_smoke_area_ratio=0.54,
        avg_fire_confidence=0.73,
        avg_smoke_confidence=0.86,
        growth_trend=GrowthTrend.STABLE,
        source_type="video",
        source_name="test.mp4",
    )


def _sample_decision() -> FireDecision:
    return FireDecision(
        event_id="FE-20260812-0001",
        danger_level=DangerLevel.MEDIUM,
        score=0.48,
        confidence=0.72,
        reasons=["持续烟雾，未见持续火焰"],
        decision_source="rule_engine",
    )


class TestEventContext(unittest.TestCase):
    def test_build_with_decision(self):
        ctx = build_event_context(_sample_event(), _sample_decision())
        self.assertEqual(ctx["event_id"], "FE-20260812-0001")
        self.assertEqual(ctx["status"], "confirmed")
        self.assertAlmostEqual(ctx["duration_seconds"], 2.73)
        self.assertTrue(ctx["fire"]["detected"])
        self.assertAlmostEqual(ctx["fire"]["max_confidence"], 0.73)
        self.assertAlmostEqual(ctx["fire"]["max_area_ratio"], 0.13)
        self.assertTrue(ctx["smoke"]["detected"])
        self.assertAlmostEqual(ctx["smoke"]["max_confidence"], 0.86)
        self.assertAlmostEqual(ctx["smoke"]["max_area_ratio"], 0.54)
        self.assertEqual(ctx["growth_trend"], "stable")
        self.assertEqual(ctx["rule_decision"]["level"], "medium")
        self.assertAlmostEqual(ctx["rule_decision"]["score"], 0.48)
        self.assertAlmostEqual(ctx["rule_decision"]["confidence"], 0.72)
        self.assertEqual(ctx["rule_decision"]["reasons"], ["持续烟雾，未见持续火焰"])

    def test_build_without_decision(self):
        ctx = build_event_context(_sample_event())
        self.assertIsNone(ctx["rule_decision"])

    def test_empty_event_id_raises(self):
        with self.assertRaises(ValueError):
            build_event_context(FireEvent(event_id=""))

    def test_enum_and_str_growth_trend(self):
        event = _sample_event()
        event.growth_trend = "increasing"
        ctx = build_event_context(event)
        self.assertEqual(ctx["growth_trend"], "increasing")

    def test_summarize_contains_key_facts(self):
        text = summarize_event_context(build_event_context(_sample_event(), _sample_decision()))
        self.assertIn("FE-20260812-0001", text)
        self.assertIn("2.73", text)
        self.assertIn("medium", text)


if __name__ == "__main__":
    unittest.main()
