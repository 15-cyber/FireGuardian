# -*- coding: utf-8 -*-
"""V2 Agent - StructuredMemoryRetriever 单元测试（Phase 3-A）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.memory_models import EventMemoryRecord, MemoryConfig
from agent.memory_retriever import StructuredMemoryRetriever


def _record(
    event_id,
    *,
    duration=10.0,
    fire=False,
    smoke=False,
    fire_area=0.0,
    smoke_area=0.0,
    fire_conf=0.0,
    smoke_conf=0.0,
    growth="unknown",
    rule="medium",
    status="ended",
):
    return EventMemoryRecord(
        event_id=event_id,
        duration_seconds=duration,
        fire_detected=fire,
        smoke_detected=smoke,
        fire_area_ratio=fire_area,
        smoke_area_ratio=smoke_area,
        max_fire_confidence=fire_conf,
        max_smoke_confidence=smoke_conf,
        growth_trend=growth,
        rule_level=rule,
        status=status,
        is_ended=status == "ended",
    )


class TestMemoryRetriever(unittest.TestCase):
    def setUp(self):
        self.retriever = StructuredMemoryRetriever()
        self.query = _record(
            "Q-1", duration=5.0, smoke=True, smoke_area=0.5, smoke_conf=0.8,
            growth="decreasing", rule="medium",
        )

    def test_no_history(self):
        result = self.retriever.search(self.query, [])
        self.assertEqual(result.results, [])
        self.assertEqual(result.candidate_count, 0)
        self.assertEqual(result.matched_count, 0)

    def test_single_history_match(self):
        history = [_record("H-1", duration=4.5, smoke=True, smoke_area=0.48,
                           smoke_conf=0.78, growth="decreasing", rule="medium")]
        result = self.retriever.search(self.query, history)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].event_id, "H-1")
        self.assertGreaterEqual(result.results[0].similarity_score, 0.65)

    def test_current_event_excluded(self):
        history = [_record("Q-1", duration=5.0, smoke=True, smoke_area=0.5,
                           smoke_conf=0.8, growth="decreasing", rule="medium")]
        result = self.retriever.search(self.query, history)
        self.assertEqual(result.results, [])

    def test_only_ended_by_default(self):
        active = _record("A-1", duration=5.0, smoke=True, smoke_area=0.5,
                         smoke_conf=0.8, growth="decreasing", rule="medium", status="active")
        ended = _record("E-1", duration=4.0, smoke=True, smoke_area=0.5,
                        smoke_conf=0.8, growth="decreasing", rule="medium")
        result = self.retriever.search(self.query, [active, ended])
        ids = [r.event_id for r in result.results]
        self.assertIn("E-1", ids)
        self.assertNotIn("A-1", ids)

    def test_active_included_when_flag_disabled(self):
        retriever = StructuredMemoryRetriever(MemoryConfig(only_ended_events=False))
        active = _record("A-1", duration=5.0, smoke=True, smoke_area=0.5,
                         smoke_conf=0.8, growth="decreasing", rule="medium", status="active")
        result = retriever.search(self.query, [active])
        self.assertEqual([r.event_id for r in result.results], ["A-1"])

    def test_top_k_limited(self):
        history = [
            _record(f"H-{i}", duration=5.0 + i * 0.1, smoke=True, smoke_area=0.5,
                    smoke_conf=0.8, growth="decreasing", rule="medium")
            for i in range(8)
        ]
        result = self.retriever.search(self.query, history, top_k=3)
        self.assertEqual(len(result.results), 3)

    def test_top_k_capped_by_max(self):
        retriever = StructuredMemoryRetriever(MemoryConfig(max_top_k=10))
        history = [
            _record(f"H-{i}", duration=5.0, smoke=True, smoke_area=0.5,
                    smoke_conf=0.8, growth="decreasing", rule="medium")
            for i in range(20)
        ]
        result = retriever.search(self.query, history, top_k=999)
        self.assertEqual(len(result.results), 10)

    def test_threshold_filters(self):
        retriever = StructuredMemoryRetriever(MemoryConfig(min_similarity=0.99))
        history = [_record("H-1", duration=4.5, smoke=True, smoke_area=0.48,
                           smoke_conf=0.78, growth="decreasing", rule="medium")]
        result = retriever.search(self.query, history)
        self.assertEqual(result.results, [])
        self.assertEqual(result.matched_count, 0)

    def test_matched_count_vs_top_k(self):
        history = [
            _record(f"H-{i}", duration=5.0 + i, smoke=True, smoke_area=0.5,
                    smoke_conf=0.8, growth="decreasing", rule="medium")
            for i in range(6)
        ]
        result = self.retriever.search(self.query, history, top_k=2, min_similarity=0.5)
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.matched_count, 6)

    def test_fire_presence_weight(self):
        same = _record("S-1", duration=5.0, smoke=True, smoke_area=0.5,
                       smoke_conf=0.8, growth="decreasing", rule="medium")
        diff = _record("D-1", duration=5.0, fire=True, smoke=True, fire_area=0.3,
                       smoke_area=0.5, fire_conf=0.8, smoke_conf=0.8,
                       growth="decreasing", rule="medium")
        r_same = self.retriever.search(
            self.query, [same], min_similarity=0.5
        ).results[0].similarity_score
        r_diff = self.retriever.search(
            self.query, [diff], min_similarity=0.5
        ).results[0].similarity_score
        self.assertGreater(r_same, r_diff)

    def test_duration_similarity_formula(self):
        far = _record("F-1", duration=100.0, smoke=True, smoke_area=0.5,
                      smoke_conf=0.8, growth="decreasing", rule="medium")
        near = _record("N-1", duration=5.1, smoke=True, smoke_area=0.5,
                       smoke_conf=0.8, growth="decreasing", rule="medium")
        r_far = self.retriever.search(self.query, [far]).results[0].similarity_score
        r_near = self.retriever.search(self.query, [near]).results[0].similarity_score
        self.assertGreater(r_near, r_far)

    def test_risk_level_similarity(self):
        same = _record("S-1", duration=5.0, smoke=True, smoke_area=0.5,
                       smoke_conf=0.8, growth="decreasing", rule="medium")
        far_level = _record("L-1", duration=5.0, smoke=True, smoke_area=0.5,
                            smoke_conf=0.8, growth="decreasing", rule="low")
        r_same = self.retriever.search(self.query, [same]).results[0].similarity_score
        r_far = self.retriever.search(self.query, [far_level]).results[0].similarity_score
        self.assertGreater(r_same, r_far)

    def test_growth_trend_match(self):
        match = _record("M-1", duration=5.0, smoke=True, smoke_area=0.5,
                        smoke_conf=0.8, growth="decreasing", rule="medium")
        mismatch = _record("X-1", duration=5.0, smoke=True, smoke_area=0.5,
                           smoke_conf=0.8, growth="increasing", rule="medium")
        r_match = self.retriever.search(self.query, [match]).results[0].similarity_score
        r_mismatch = self.retriever.search(self.query, [mismatch]).results[0].similarity_score
        self.assertGreater(r_match, r_mismatch)

    def test_tie_break_same_level_then_duration(self):
        history = [
            _record("H-FAR", duration=50.0, fire=True, smoke=True, fire_area=0.35,
                    smoke_area=0.42, fire_conf=0.92, smoke_conf=0.88,
                    growth="increasing", rule="high", status="ended"),
            _record("H-NEAR", duration=30.1, fire=True, smoke=True, fire_area=0.35,
                    smoke_area=0.42, fire_conf=0.92, smoke_conf=0.88,
                    growth="increasing", rule="high", status="ended"),
        ]
        q2 = _record("Q-2", duration=30.0, fire=True, smoke=True, fire_area=0.35,
                     smoke_area=0.42, fire_conf=0.92, smoke_conf=0.88,
                     growth="increasing", rule="high")
        result = self.retriever.search(q2, history)
        self.assertEqual([r.event_id for r in result.results], ["H-NEAR", "H-FAR"])

    def test_query_event_compact(self):
        result = self.retriever.search(self.query, [])
        self.assertEqual(
            set(result.query_event.keys()),
            {"event_id", "duration_seconds", "fire_detected", "smoke_detected", "rule_level"},
        )

    def test_matched_features_present(self):
        history = [_record("H-1", duration=4.5, smoke=True, smoke_area=0.48,
                           smoke_conf=0.78, growth="decreasing", rule="medium")]
        result = self.retriever.search(self.query, history)
        self.assertIn("smoke_presence", result.results[0].matched_features)


if __name__ == "__main__":
    unittest.main()
