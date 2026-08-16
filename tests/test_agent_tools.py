# -*- coding: utf-8 -*-
"""V2 Agent - ToolRegistry / ToolExecutor 单元测试（Phase 1）。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.tool_executor import (
    InMemoryDataSource,
    JsonlEventDataSource,
    ToolExecutor,
)
from agent.tool_registry import (
    TOOL_DEFINITIONS,
    ToolRegistry,
    ToolValidationError,
)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    def test_five_tools_registered(self):
        self.assertEqual(
            sorted(self.registry.list()),
            sorted(
                [
                    "get_event_summary",
                    "get_rule_decision",
                    "get_detection_history",
                    "get_event_evidence",
                    "get_similar_events",
                ]
            ),
        )
        self.assertEqual(len(self.registry.schemas()), len(TOOL_DEFINITIONS))

    def test_valid_calls_accepted(self):
        self.registry.validate_call("get_event_summary", {"event_id": "FE-001"})
        self.registry.validate_call(
            "get_detection_history", {"event_id": "FE-001", "window_seconds": 30}
        )
        self.registry.validate_call(
            "get_similar_events",
            {"event_id": "FE-001", "top_k": 5, "min_similarity": 0.65},
        )

    def test_unknown_tool_rejected(self):
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call("rm_rf", {})

    def test_bad_event_id_rejected(self):
        for bad in ("../etc/passwd", "a" * 65, "EVENT ID", "事件001"):
            with self.assertRaises(ToolValidationError):
                self.registry.validate_call("get_event_summary", {"event_id": bad})

    def test_missing_required_rejected(self):
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call("get_event_summary", {})

    def test_unknown_argument_rejected(self):
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call(
                "get_event_summary", {"event_id": "FE-001", "path": "/etc/passwd"}
            )

    def test_window_seconds_out_of_range_rejected(self):
        for bad in (0, 3601, -5, "30"):
            with self.assertRaises(ToolValidationError):
                self.registry.validate_call(
                    "get_detection_history", {"event_id": "FE-001", "window_seconds": bad}
                )

    def test_window_seconds_float_accepted(self):
        self.registry.validate_call(
            "get_detection_history", {"event_id": "FE-001", "window_seconds": 3.5}
        )

    def test_similar_events_params_validated(self):
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call(
                "get_similar_events", {"event_id": "FE-001", "top_k": 0}
            )
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call(
                "get_similar_events", {"event_id": "FE-001", "top_k": 11}
            )
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call(
                "get_similar_events", {"event_id": "FE-001", "top_k": 2.5}
            )
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call(
                "get_similar_events", {"event_id": "FE-001", "min_similarity": 1.1}
            )
        with self.assertRaises(ToolValidationError):
            self.registry.validate_call(
                "get_similar_events", {"event_id": "FE-001", "limit": 5}
            )


class TestToolExecutor(unittest.TestCase):
    def setUp(self):
        self.data_source = InMemoryDataSource(
            summaries={
                "FE-001": {
                    "found": True,
                    "event_id": "FE-001",
                    "summary": {"status": "confirmed", "duration_seconds": 3.0},
                }
            },
            decisions={
                "FE-001": {
                    "found": True,
                    "event_id": "FE-001",
                    "decision": {"level": "medium", "score": 0.5},
                }
            },
            histories={
                "FE-001": {
                    "event_id": "FE-001",
                    "window_seconds": 30.0,
                    "snapshots": [{"timestamp_seconds": 1.0}],
                }
            },
            evidence={
                "FE-001": {"found": True, "event_id": "FE-001", "files": ["a.jpg"]},
            },
        )
        self.executor = ToolExecutor(self.data_source)

    def test_execute_event_summary(self):
        result = self.executor.execute("get_event_summary", {"event_id": "FE-001"})
        self.assertTrue(result.success)
        self.assertTrue(result.data["found"])

    def test_execute_rule_decision(self):
        result = self.executor.execute("get_rule_decision", {"event_id": "FE-001"})
        self.assertTrue(result.success)
        self.assertEqual(result.data["decision"]["level"], "medium")

    def test_execute_detection_history(self):
        result = self.executor.execute(
            "get_detection_history", {"event_id": "FE-001", "window_seconds": 30}
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["snapshots"]), 1)

    def test_execute_evidence(self):
        result = self.executor.execute("get_event_evidence", {"event_id": "FE-001"})
        self.assertTrue(result.success)
        self.assertEqual(result.data["files"], ["a.jpg"])

    def test_similar_events_returns_results(self):
        executor = ToolExecutor(
            InMemoryDataSource(
                similar_events={
                    "FE-001": {
                        "query_event": {"event_id": "FE-001"},
                        "results": [{"event_id": "H-1", "similarity_score": 0.88}],
                        "summary": {"candidate_count": 3, "matched_count": 1},
                        "memory_available": True,
                        "reason": None,
                    }
                }
            )
        )
        result = executor.execute(
            "get_similar_events",
            {"event_id": "FE-001", "top_k": 3, "min_similarity": 0.6},
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["event_id"], "H-1")

    def test_unknown_tool_returns_structured_error(self):
        result = self.executor.execute("nope", {})
        self.assertFalse(result.success)
        self.assertIn("未知工具", result.error)

    def test_invalid_args_return_structured_error_not_exception(self):
        result = self.executor.execute("get_event_summary", {"event_id": "../../etc"})
        self.assertFalse(result.success)
        self.assertIn("event_id", result.error)


class TestJsonlEventDataSource(unittest.TestCase):
    def _write_jsonl(self, path: Path):
        lines = [
            {
                "time": "2026-08-12T10:00:00",
                "module": "event_aggregator",
                "event_type": "event_confirmed",
                "event_id": "FE-REAL-001",
                "details": {
                    "event_id": "FE-REAL-001",
                    "status": "confirmed",
                    "duration": 1.0,
                    "total_frames": 4,
                    "positive_frames": 3,
                    "positive_ratio": 0.75,
                    "max_fire_area_ratio": 0.1,
                    "max_smoke_area_ratio": 0.4,
                    "avg_fire_confidence": 0.7,
                    "avg_smoke_confidence": 0.8,
                    "growth_trend": "increasing",
                },
            },
            {
                "time": "2026-08-12T10:00:03",
                "module": "event_aggregator",
                "event_type": "event_updated",
                "event_id": "FE-REAL-001",
                "details": {
                    "event_id": "FE-REAL-001",
                    "status": "active",
                    "duration_seconds": 3.0,
                    "total_frames": 12,
                    "positive_frames": 10,
                    "max_fire_area_ratio": 0.12,
                    "max_smoke_area_ratio": 0.5,
                },
            },
            {
                "time": "2026-08-12T10:00:04",
                "module": "fire_decision_agent",
                "event_type": "decision_made",
                "event_id": "FE-REAL-001",
                "details": {
                    "event_id": "FE-REAL-001",
                    "danger_level": "high",
                    "score": 0.62,
                    "confidence": 0.85,
                    "reasons": ["火焰面积较大"],
                },
            },
        ]
        path.write_text(
            "\n".join(json.dumps(line, ensure_ascii=False) for line in lines),
            encoding="utf-8",
        )

    def test_summary_decision_history_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jsonl_path = tmp_path / "events.jsonl"
            self._write_jsonl(jsonl_path)
            source = JsonlEventDataSource(events_path=jsonl_path)

            summary = source.get_event_summary("FE-REAL-001")
            self.assertTrue(summary["found"])
            self.assertEqual(summary["summary"]["duration_seconds"], 3.0)  # 兼容 duration/duration_seconds
            self.assertEqual(summary["summary"]["status"], "active")

            decision = source.get_rule_decision("FE-REAL-001")
            self.assertTrue(decision["found"])
            self.assertEqual(decision["decision"]["level"], "high")
            self.assertEqual(decision["decision"]["reasons"], ["火焰面积较大"])

            history = source.get_detection_history("FE-REAL-001", 30.0)
            self.assertEqual(len(history["snapshots"]), 2)

    def test_evidence_lists_screenshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            event_dir = tmp_path / "event_FE-EV-001"
            event_dir.mkdir()
            (event_dir / "confirmed.jpg").write_bytes(b"jpg")
            (event_dir / "metadata.json").write_text('{"peak": true}', encoding="utf-8")
            source = JsonlEventDataSource(screenshots_root=tmp_path)
            result = source.get_event_evidence("FE-EV-001")
            self.assertTrue(result["found"])
            self.assertEqual(len(result["files"]), 2)
            self.assertTrue(result["metadata"]["peak"])


if __name__ == "__main__":
    unittest.main()
