# -*- coding: utf-8 -*-
"""V2 Agent - AgentMemory / MemoryTool 单元测试（Phase 3-A，全部 Mock，不消耗 API）。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.agent_memory import AgentMemory, MemoryToolDataSource
from agent.memory_normalizer import build_query_record
from agent.tool_executor import ToolExecutor
from agent.tool_registry import ToolRegistry, ToolValidationError
from utils.common import DangerLevel, EventStatus, FireDecision, FireEvent, GrowthTrend

FIXTURE = _ROOT / "tests" / "fixtures" / "history_events.jsonl"
BAD_FIXTURE = _ROOT / "tests" / "fixtures" / "bad_history_events.jsonl"


def _query_event():
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
    return build_query_record(event, decision)


class TestAgentMemory(unittest.TestCase):
    def test_load_fixture_stats(self):
        memory = AgentMemory(events_path=FIXTURE)
        result = memory.load()
        self.assertEqual(len(result.valid_records), 9)  # 9 条 event_aggregator
        self.assertEqual(result.invalid_count, 0)
        self.assertEqual(memory._load_stats["ended_count"], 7)  # FE-H-001..006,009

    def test_load_bad_fixture_skips_corrupt(self):
        memory = AgentMemory(events_path=BAD_FIXTURE)
        result = memory.load()
        self.assertEqual(len(result.valid_records), 2)
        self.assertEqual(result.invalid_count, 0)
        stats = memory._load_stats
        self.assertEqual(stats["invalid_json"], 2)   # 非 JSON 字符串行 + 非对象 JSON
        self.assertEqual(stats["no_details"], 1)     # 无 details 记录

    def test_history_missing_returns_available_false(self):
        memory = AgentMemory(events_path=Path(tempfile.gettempdir()) / "no_such_events.jsonl")
        result = memory.search(_query_event())
        self.assertFalse(result.memory_available)
        self.assertEqual(result.reason, "history_not_found")
        self.assertEqual(result.results, [])

    def test_search_fixture_excludes_active_and_self(self):
        memory = AgentMemory(events_path=FIXTURE)
        result = memory.search(_query_event())
        ids = [r.event_id for r in result.results]
        self.assertNotIn("FE-Q-001", ids)
        self.assertNotIn("FE-H-007", ids)  # active，不允许作为正式结果
        self.assertNotIn("FE-H-008", ids)  # confirmed，不允许作为正式结果
        self.assertGreaterEqual(result.candidate_count, 5)
        self.assertTrue(all(r.similarity_score >= 0.65 for r in result.results))

    def test_get_record(self):
        memory = AgentMemory(events_path=FIXTURE)
        rec = memory.get_record("FE-H-001")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.rule_level, "high")
        self.assertEqual(rec.rule_level_source, "decision")
        self.assertIsNone(rec.final_level)  # 无 Hybrid 记录 → None
        self.assertEqual(memory.get_record("NOPE"), None)

    def test_search_for_event_missing_raises(self):
        memory = AgentMemory(events_path=FIXTURE)
        with self.assertRaises(ValueError):
            memory.search_for_event("NO-SUCH-EVENT")

    def test_mtime_cache_load_once(self):
        memory = AgentMemory(events_path=FIXTURE)
        first = memory.load()
        second = memory.load()
        self.assertIs(first, second)


class TestMemoryTool(unittest.TestCase):
    def setUp(self):
        self.memory = AgentMemory(events_path=FIXTURE)

    def test_tool_executor_returns_results(self):
        source = MemoryToolDataSource(
            memory=self.memory,
            query_provider=lambda eid: _query_event() if eid == "FE-Q-001" else None,
        )
        executor = ToolExecutor(source)
        result = executor.execute(
            "get_similar_events",
            {"event_id": "FE-Q-001", "top_k": 3, "min_similarity": 0.65},
        )
        self.assertTrue(result.success, result.error)
        data = result.data
        self.assertEqual(data["query_event"]["event_id"], "FE-Q-001")
        self.assertIn("summary", data)
        self.assertIn("results", data)
        self.assertLessEqual(len(data["results"]), 3)

    def test_tool_current_event_not_found(self):
        source = MemoryToolDataSource(memory=self.memory)
        executor = ToolExecutor(source)
        result = executor.execute(
            "get_similar_events", {"event_id": "NO-SUCH", "top_k": 5}
        )
        self.assertFalse(result.success)
        self.assertIn("当前事件不存在", result.error)

    def test_tool_validation(self):
        registry = ToolRegistry()
        for bad in (0, 11, 2.5):
            with self.assertRaises(ToolValidationError):
                registry.validate_call(
                    "get_similar_events", {"event_id": "FE-Q-001", "top_k": bad}
                )
        for bad in (-0.1, 1.1):
            with self.assertRaises(ToolValidationError):
                registry.validate_call(
                    "get_similar_events", {"event_id": "FE-Q-001", "min_similarity": bad}
                )
        with self.assertRaises(ToolValidationError):
            registry.validate_call(
                "get_similar_events", {"event_id": "FE-Q-001", "limit": 5}
            )
        with self.assertRaises(ToolValidationError):
            registry.validate_call("get_similar_events", {})

    def test_memory_log_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = AgentLogger(memory_path=Path(tmp) / "agent_memory.jsonl")
            source = MemoryToolDataSource(
                memory=self.memory,
                query_provider=lambda eid: _query_event() if eid == "FE-Q-001" else None,
                logger=logger,
            )
            executor = ToolExecutor(source)
            executor.execute(
                "get_similar_events",
                {"event_id": "FE-Q-001", "top_k": 5, "min_similarity": 0.65},
            )
            lines = (Path(tmp) / "agent_memory.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["tool"], "get_similar_events")
            self.assertEqual(record["event_id"], "FE-Q-001")
            self.assertIn("results", record)
            self.assertNotIn("api_key", json.dumps(record).lower())

    def test_memory_stats_counters(self):
        source = MemoryToolDataSource(
            memory=self.memory,
            query_provider=lambda eid: _query_event() if eid == "FE-Q-001" else None,
        )
        executor = ToolExecutor(source)
        executor.execute("get_similar_events", {"event_id": "FE-Q-001"})
        executor.execute("get_similar_events", {"event_id": "FE-Q-001"})
        stats = self.memory.stats()
        self.assertEqual(stats["memory_calls"], 2)
        self.assertGreater(stats["total_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
