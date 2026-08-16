# -*- coding: utf-8 -*-
"""V2 Agent - Memory Normalizer 单元测试（Phase 3-A）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.memory_normalizer import (
    build_query_record,
    normalize_event_record,
    normalize_many,
)
from utils.common import DangerLevel, EventStatus, FireDecision, FireEvent, GrowthTrend


def _raw(**overrides):
    data = {
        "event_id": "FE-T-001",
        "status": "ended",
        "start_timestamp": 0.0,
        "end_timestamp": 5.0,
        "duration_seconds": 5.0,
        "total_frames": 20,
        "positive_frames": 15,
        "max_fire_area_ratio": 0.1,
        "max_smoke_area_ratio": 0.3,
        "avg_fire_confidence": 0.7,
        "avg_smoke_confidence": 0.8,
        "growth_trend": "stable",
        "source_type": "video",
        "metadata": {},
    }
    data.update(overrides)
    return data


class TestNormalizeEventRecord(unittest.TestCase):
    def test_duration_seconds_preferred(self):
        rec = normalize_event_record(_raw(duration_seconds=6.0, duration=999.0))
        self.assertAlmostEqual(rec.duration_seconds, 6.0)
        self.assertEqual(rec.duration_source, "normalized")
        self.assertEqual(rec.duration_unit, "seconds")

    def test_legacy_duration_with_consistent_timestamps(self):
        rec = normalize_event_record(
            _raw(duration_seconds=None, duration=5.0, end_timestamp=5.0)
        )
        self.assertAlmostEqual(rec.duration_seconds, 5.0)
        self.assertEqual(rec.duration_source, "legacy")
        self.assertEqual(rec.duration_unit, "seconds")

    def test_legacy_duration_unit_unknown_when_inconsistent(self):
        rec = normalize_event_record(
            _raw(
                duration_seconds=None,
                duration=3.0,
                start_timestamp=100.0,
                end_timestamp=130.0,
            )
        )
        self.assertEqual(rec.duration_source, "legacy")
        self.assertEqual(rec.duration_unit, "unknown")

    def test_missing_fields_defaults(self):
        rec = normalize_event_record({"event_id": "FE-X"})
        self.assertEqual(rec.event_id, "FE-X")
        self.assertEqual(rec.duration_seconds, 0.0)
        self.assertFalse(rec.fire_detected)
        self.assertFalse(rec.smoke_detected)
        self.assertEqual(rec.growth_trend, "unknown")
        self.assertEqual(rec.rule_level, "low")
        self.assertEqual(rec.rule_level_source, "none")

    def test_invalid_records_return_none(self):
        self.assertIsNone(normalize_event_record(None))
        self.assertIsNone(normalize_event_record("text"))
        self.assertIsNone(normalize_event_record({"no_event_id": 1}))
        self.assertIsNone(normalize_event_record({"event_id": "  "}))

    def test_growth_aliases_normalized(self):
        self.assertEqual(normalize_event_record(_raw(growth_trend="increase")).growth_trend, "increasing")
        self.assertEqual(normalize_event_record(_raw(growth_trend="rising")).growth_trend, "increasing")
        self.assertEqual(normalize_event_record(_raw(growth_trend="steady")).growth_trend, "stable")
        self.assertEqual(normalize_event_record(_raw(growth_trend="declining")).growth_trend, "decreasing")
        self.assertEqual(normalize_event_record(_raw(growth_trend=None)).growth_trend, "unknown")

    def test_rule_level_from_joined_decision(self):
        raw = _raw(rule_decision={"danger_level": "high", "score": 0.8, "confidence": 0.9})
        rec = normalize_event_record(raw)
        self.assertEqual(rec.rule_level, "high")
        self.assertEqual(rec.rule_level_source, "decision")
        self.assertAlmostEqual(rec.rule_score, 0.8)

    def test_rule_level_from_metadata_fallback(self):
        rec = normalize_event_record(_raw(metadata={"danger_level": "medium"}))
        self.assertEqual(rec.rule_level, "medium")
        self.assertEqual(rec.rule_level_source, "metadata")

    def test_final_level_none_without_hybrid(self):
        rec = normalize_event_record(_raw(rule_decision={"danger_level": "high"}))
        self.assertIsNone(rec.final_level)  # 不得用 rule_level 冒充 final_level

    def test_final_level_set_only_from_hybrid(self):
        rec = normalize_event_record(_raw(hybrid_decision={"final_level": "high"}))
        self.assertEqual(rec.final_level, "high")

    def test_is_ended_from_status(self):
        self.assertTrue(normalize_event_record(_raw(status="ended")).is_ended)
        self.assertFalse(normalize_event_record(_raw(status="confirmed")).is_ended)

    def test_confidence_source_avg_when_no_max(self):
        rec = normalize_event_record(_raw())
        self.assertEqual(rec.confidence_source, "avg")
        self.assertAlmostEqual(rec.max_fire_confidence, 0.7)

    def test_build_query_record_from_fire_event(self):
        event = FireEvent(
            event_id="FE-Q-001",
            status=EventStatus.CONFIRMED,
            duration_seconds=2.7,
            total_frames=20,
            positive_frames=14,
            max_fire_area_ratio=0.0,
            max_smoke_area_ratio=0.4,
            avg_fire_confidence=0.0,
            avg_smoke_confidence=0.8,
            growth_trend=GrowthTrend.STABLE,
        )
        decision = FireDecision(
            event_id="FE-Q-001",
            danger_level=DangerLevel.MEDIUM,
            score=0.5,
            confidence=0.7,
        )
        query = build_query_record(event, decision)
        self.assertEqual(query.event_id, "FE-Q-001")
        self.assertAlmostEqual(query.duration_seconds, 2.7)
        self.assertTrue(query.smoke_detected)
        self.assertFalse(query.fire_detected)
        self.assertEqual(query.rule_level, "medium")
        self.assertEqual(query.rule_level_source, "decision")


class TestNormalizeMany(unittest.TestCase):
    def test_mixed_records_counts(self):
        result = normalize_many([_raw(), None, {"event_id": ""}, _raw(event_id="FE-2")])
        self.assertEqual(len(result.valid_records), 2)
        self.assertEqual(result.invalid_count, 2)


if __name__ == "__main__":
    unittest.main()
