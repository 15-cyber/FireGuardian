"""
============================================================
V2 Agent - ToolExecutor（Phase 1）

安全执行 Agent 发起的工具调用：
  校验 → 类型/范围/权限检查 → 执行 → 返回结构化结果。

数据源：
  - JsonlEventDataSource：读取 M9 logs/events.jsonl 与 M8 screenshots/，
    只读，不修改任何 V1 模块。
  - InMemoryDataSource：单元测试用。

Phase 1 中 get_similar_events 属于 Phase 3（Agent Memory），
调用时返回明确的不可用错误。
============================================================
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.schemas import ToolResult
from agent.tool_registry import ToolRegistry, ToolValidationError


class ToolUnavailableError(RuntimeError):
    """工具在当前阶段不可用。"""


class ToolDataSource:
    """工具数据源接口。"""

    def get_event_summary(self, event_id: str) -> dict:
        raise NotImplementedError

    def get_rule_decision(self, event_id: str) -> dict:
        raise NotImplementedError

    def get_detection_history(self, event_id: str, window_seconds: float) -> List[dict]:
        raise NotImplementedError

    def get_event_evidence(self, event_id: str) -> dict:
        raise NotImplementedError

    def get_similar_events(self, event_id: str, top_k: int, min_similarity: float) -> dict:
        raise NotImplementedError


def _sanitize_dir_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9\-_]", "_", name)
    return safe.strip("_") or "event"


class JsonlEventDataSource(ToolDataSource):
    """基于 M9 logs/events.jsonl 与 M8 screenshots/ 的只读数据源。"""

    def __init__(
        self,
        events_path: Optional[Path] = None,
        screenshots_root: Optional[Path] = None,
    ):
        self.events_path = Path(events_path) if events_path else _ROOT / "logs" / "events.jsonl"
        self.screenshots_root = (
            Path(screenshots_root) if screenshots_root else _ROOT / "screenshots"
        )

    def _iter_records(self):
        if not self.events_path.exists():
            return
        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _aggregator_records(self, event_id: str):
        for record in self._iter_records():
            details = record.get("details") or {}
            if details.get("event_id") == event_id and record.get("module") == "event_aggregator":
                yield record

    def get_event_summary(self, event_id: str) -> dict:
        records = list(self._aggregator_records(event_id))
        if not records:
            return {"found": False, "event_id": event_id}
        details = records[-1]["details"]
        return {
            "found": True,
            "event_id": event_id,
            "summary": {
                "status": details.get("status"),
                "duration_seconds": details.get("duration_seconds", details.get("duration", 0.0)),
                "total_frames": details.get("total_frames"),
                "positive_frames": details.get("positive_frames"),
                "positive_ratio": details.get("positive_ratio"),
                "max_fire_area_ratio": details.get("max_fire_area_ratio"),
                "max_smoke_area_ratio": details.get("max_smoke_area_ratio"),
                "avg_fire_confidence": details.get("avg_fire_confidence"),
                "avg_smoke_confidence": details.get("avg_smoke_confidence"),
                "growth_trend": details.get("growth_trend"),
                "source_type": details.get("source_type"),
                "source_name": details.get("source_name"),
                "last_update": records[-1].get("time"),
            },
        }

    def get_rule_decision(self, event_id: str) -> dict:
        for record in self._iter_records():
            details = record.get("details") or {}
            if (
                details.get("event_id") == event_id
                and record.get("module") == "fire_decision_agent"
            ):
                return {
                    "found": True,
                    "event_id": event_id,
                    "decision": {
                        "level": details.get("danger_level"),
                        "score": details.get("score"),
                        "confidence": details.get("confidence"),
                        "reasons": details.get("reasons", []),
                        "decision_source": details.get("decision_source"),
                    },
                }
        return {"found": False, "event_id": event_id}

    def get_detection_history(self, event_id: str, window_seconds: float) -> dict:
        snapshots: List[dict] = []
        for record in self._aggregator_records(event_id):
            details = record["details"]
            snapshots.append(
                {
                    "timestamp_seconds": details.get(
                        "last_time_seconds",
                        details.get("last_timestamp", details.get("duration", 0.0)),
                    ),
                    "total_frames": details.get("total_frames"),
                    "positive_frames": details.get("positive_frames"),
                    "max_fire_area_ratio": details.get("max_fire_area_ratio"),
                    "max_smoke_area_ratio": details.get("max_smoke_area_ratio"),
                    "avg_fire_confidence": details.get("avg_fire_confidence"),
                    "avg_smoke_confidence": details.get("avg_smoke_confidence"),
                    "growth_trend": details.get("growth_trend"),
                }
            )
        if snapshots:
            latest = max(float(s["timestamp_seconds"] or 0.0) for s in snapshots)
            snapshots = [
                s for s in snapshots
                if (float(s["timestamp_seconds"] or 0.0)) >= latest - float(window_seconds)
            ]
        return {
            "event_id": event_id,
            "window_seconds": float(window_seconds),
            "snapshots": snapshots,
        }

    def get_event_evidence(self, event_id: str) -> dict:
        event_dir = self.screenshots_root / f"event_{_sanitize_dir_name(event_id)}"
        if not event_dir.exists():
            return {
                "found": False,
                "event_id": event_id,
                "screenshot_dir": str(event_dir),
            }
        files: List[dict] = []
        for path in sorted(event_dir.iterdir()):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    }
                )
        metadata: dict = {}
        metadata_path = event_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                metadata = {}
        return {
            "found": True,
            "event_id": event_id,
            "screenshot_dir": str(event_dir),
            "files": files,
            "metadata": metadata,
        }

    def get_similar_events(
        self, event_id: str, top_k: int, min_similarity: float
    ) -> dict:
        """基于同一 events.jsonl 的只读 Memory 检索（懒加载，避免循环导入）。"""
        from agent.agent_memory import AgentMemory

        memory = AgentMemory(events_path=self.events_path)
        return memory.search_for_event(
            event_id, top_k=top_k, min_similarity=min_similarity
        ).to_dict()


class InMemoryDataSource(ToolDataSource):
    """单元测试用内存数据源。"""

    def __init__(
        self,
        summaries: Optional[Dict[str, dict]] = None,
        decisions: Optional[Dict[str, dict]] = None,
        histories: Optional[Dict[str, dict]] = None,
        evidence: Optional[Dict[str, dict]] = None,
        similar_events: Optional[Dict[str, dict]] = None,
    ):
        self.summaries = summaries or {}
        self.decisions = decisions or {}
        self.histories = histories or {}
        self.evidence = evidence or {}
        self.similar_events = similar_events or {}

    def get_event_summary(self, event_id: str) -> dict:
        return self.summaries.get(
            event_id, {"found": False, "event_id": event_id}
        )

    def get_rule_decision(self, event_id: str) -> dict:
        return self.decisions.get(
            event_id, {"found": False, "event_id": event_id}
        )

    def get_detection_history(self, event_id: str, window_seconds: float) -> dict:
        return self.histories.get(
            event_id,
            {"event_id": event_id, "window_seconds": float(window_seconds), "snapshots": []},
        )

    def get_event_evidence(self, event_id: str) -> dict:
        return self.evidence.get(
            event_id, {"found": False, "event_id": event_id}
        )

    def get_similar_events(
        self, event_id: str, top_k: int, min_similarity: float
    ) -> dict:
        return self.similar_events.get(
            event_id,
            {
                "query_event": {"event_id": event_id},
                "results": [],
                "summary": {"candidate_count": 0, "matched_count": 0},
                "memory_available": True,
                "reason": None,
            },
        )


class ToolExecutor:
    """工具执行器：校验参数 → 调用数据源 → 返回结构化 ToolResult。"""

    def __init__(
        self,
        data_source: ToolDataSource,
        registry: Optional[ToolRegistry] = None,
    ):
        self.data_source = data_source
        self.registry = registry or ToolRegistry()

    def execute(self, name: str, arguments: dict) -> ToolResult:
        t0 = time.perf_counter()
        try:
            self.registry.validate_call(name, arguments)
            if name == "get_event_summary":
                data = self.data_source.get_event_summary(arguments["event_id"])
            elif name == "get_rule_decision":
                data = self.data_source.get_rule_decision(arguments["event_id"])
            elif name == "get_detection_history":
                data = self.data_source.get_detection_history(
                    arguments["event_id"], float(arguments["window_seconds"])
                )
            elif name == "get_event_evidence":
                data = self.data_source.get_event_evidence(arguments["event_id"])
            elif name == "get_similar_events":
                data = self.data_source.get_similar_events(
                    arguments["event_id"],
                    int(arguments.get("top_k", 5)),
                    float(arguments.get("min_similarity", 0.65)),
                )
            else:
                raise ToolValidationError(f"未知工具: {name!r}")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                tool_name=name, success=True, data=data, latency_ms=round(latency_ms, 2)
            )
        except Exception as exc:  # noqa: BLE001 - 工具执行失败必须返回结构化错误
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(exc),
                latency_ms=round(latency_ms, 2),
            )
