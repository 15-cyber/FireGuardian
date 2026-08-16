"""
============================================================
V2 Agent - AgentMemory（Phase 3-A）

职责：
  - 只读加载 logs/events.jsonl（或指定历史文件）+ mtime 缓存；
  - 事件记录与 M7 决策按 event_id join（取最新决策）；
  - 结构化相似事件检索（默认只查已结束事件）；
  - MemoryToolDataSource：接入 ToolExecutor 的 get_similar_events；
  - 效率统计（calls / candidates / matched / latency）。

约束：
  - 不修改 / 不重写历史日志；
  - 不引入 Embedding / 向量库；
  - Memory 结果仅作辅助证据，不参与 Hybrid 安全规则。
============================================================
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.deepseek_client import load_llm_agent_config
from agent.memory_models import (
    EventMemoryRecord,
    MemoryConfig,
    MemorySearchResult,
)
from agent.memory_normalizer import NormalizeResult, normalize_many
from agent.memory_retriever import StructuredMemoryRetriever
from agent.tool_executor import ToolDataSource


def load_memory_config(config_path: Optional[Path] = None) -> MemoryConfig:
    """从 config/llm_agent.yaml 的 memory 段加载配置（缺失时用任务书默认值）。"""
    cfg = load_llm_agent_config(config_path)
    mem = cfg.get("memory") or {}
    weights = MemoryConfig().weights
    weights.update({k: float(v) for k, v in (mem.get("weights") or {}).items()})
    return MemoryConfig(
        top_k=int(mem.get("top_k", 5)),
        max_top_k=int(mem.get("max_top_k", 10)),
        min_similarity=float(mem.get("min_similarity", 0.65)),
        only_ended_events=bool(mem.get("only_ended_events", True)),
        weights=weights,
    )


class AgentMemory:
    """历史事件记忆：只读加载 + join + 结构化检索。"""

    def __init__(
        self,
        events_path: Optional[Path] = None,
        config: Optional[MemoryConfig] = None,
    ):
        self.events_path = (
            Path(events_path) if events_path else _ROOT / "logs" / "events.jsonl"
        )
        self.config = config or load_memory_config()
        self.retriever = StructuredMemoryRetriever(self.config)
        self._records: Optional[List[EventMemoryRecord]] = None
        self._mtime: Optional[float] = None
        self._load_result: Optional[NormalizeResult] = None
        self._load_stats: Optional[dict] = None

        # 效率统计
        self.total_calls = 0
        self.total_candidates = 0
        self.total_matched = 0
        self.total_latency_ms = 0.0

    # ---------- 加载 ----------

    def load(self, force: bool = False) -> NormalizeResult:
        """加载并标准化历史事件（mtime 缓存：文件未变化不重复解析）。"""
        mtime = self.events_path.stat().st_mtime if self.events_path.exists() else None
        if (
            not force
            and self._records is not None
            and self._mtime == mtime
            and self._load_result is not None
        ):
            return self._load_result

        decisions: dict = {}
        raw_events: List[dict] = []
        raw_line_count = 0
        invalid_json_count = 0
        no_details_count = 0
        if self.events_path.exists():
            with open(self.events_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            raw_line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_count += 1
                continue
            if not isinstance(record, dict):
                invalid_json_count += 1
                continue
            details = record.get("details")
            if not isinstance(details, dict):
                no_details_count += 1
                continue
            event_id = details.get("event_id")
            if not event_id:
                no_details_count += 1
                continue
            if record.get("module") == "fire_decision_agent":
                decisions[event_id] = details  # 后写覆盖 → 最新决策
            elif record.get("module") == "event_aggregator":
                raw_events.append(details)

        joined = []
        for details in raw_events:
            row = dict(details)
            decision = decisions.get(row.get("event_id"))
            if decision is not None:
                row["rule_decision"] = decision
            joined.append(row)

        result = normalize_many(joined)
        self._records = result.valid_records
        self._mtime = mtime
        self._load_result = result
        self._load_stats = {
            "path": str(self.events_path),
            "raw_lines": raw_line_count,
            "invalid_json": invalid_json_count,
            "no_details": no_details_count,
            "valid_records": len(result.valid_records),
            "invalid_records": result.invalid_count,
            "duration_legacy_count": sum(
                1 for r in result.valid_records if r.duration_source == "legacy"
            ),
            "ended_count": sum(1 for r in result.valid_records if r.is_ended),
        }
        return result

    def records(self) -> List[EventMemoryRecord]:
        self.load()
        return list(self._records or [])

    def get_record(self, event_id: str) -> Optional[EventMemoryRecord]:
        for record in self.records():
            if record.event_id == event_id:
                return record
        return None

    # ---------- 检索 ----------

    def search(
        self,
        query: EventMemoryRecord,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> MemorySearchResult:
        if not self.events_path.exists():
            result = MemorySearchResult(
                memory_available=False,
                reason="history_not_found",
                query_event={
                    "event_id": query.event_id,
                    "duration_seconds": round(query.duration_seconds, 4),
                    "fire_detected": query.fire_detected,
                    "smoke_detected": query.smoke_detected,
                    "rule_level": query.rule_level,
                },
            )
            return result

        self.load()
        t0 = time.perf_counter()
        result = self.retriever.search(query, self.records(), top_k=top_k, min_similarity=min_similarity)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.total_calls += 1
        self.total_candidates += result.candidate_count
        self.total_matched += result.matched_count
        self.total_latency_ms += latency_ms
        return result

    def search_for_event(
        self,
        event_id: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> MemorySearchResult:
        """按 event_id 检索（查询记录从历史/缓存中解析）。"""
        query = self.get_record(event_id)
        if query is None:
            raise ValueError(f"当前事件不存在: {event_id}")
        return self.search(query, top_k=top_k, min_similarity=min_similarity)

    def stats(self) -> dict:
        return {
            "memory_calls": self.total_calls,
            "total_candidates": self.total_candidates,
            "total_matched": self.total_matched,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "load_stats": self._load_stats,
        }


class MemoryToolDataSource(ToolDataSource):
    """get_similar_events 的 ToolExecutor 数据源。"""

    def __init__(
        self,
        memory: AgentMemory,
        query_provider: Optional[Callable[[str], Optional[EventMemoryRecord]]] = None,
        logger: Optional[AgentLogger] = None,
    ):
        self.memory = memory
        self.query_provider = query_provider
        self.logger = logger or AgentLogger()

    def get_similar_events(
        self, event_id: str, top_k: int, min_similarity: float
    ) -> dict:
        t0 = time.perf_counter()
        query = None
        if self.query_provider is not None:
            try:
                query = self.query_provider(event_id)
            except Exception:  # noqa: BLE001 - provider 失败回退到历史记录解析
                query = None
        if query is None:
            query = self.memory.get_record(event_id)
        if query is None:
            raise ValueError(f"当前事件不存在: {event_id}")

        result = self.memory.search(
            query, top_k=top_k, min_similarity=min_similarity
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.logger.log_memory(
            event_id=event_id,
            top_k=top_k,
            min_similarity=min_similarity,
            candidate_count=result.candidate_count,
            matched_count=result.matched_count,
            results=[
                {"event_id": s.event_id, "score": s.similarity_score}
                for s in result.results
            ],
            latency_ms=latency_ms,
            success=result.memory_available,
            reason=result.reason,
        )
        return result.to_dict()


class MemoryAgentDataSource(ToolDataSource):
    """Phase 3-B 复合数据源：基础 4 工具 + get_similar_events（历史 fixture）。"""

    def __init__(
        self,
        base: ToolDataSource,
        memory: AgentMemory,
        query_provider: Optional[Callable[[str], Optional[EventMemoryRecord]]] = None,
        logger: Optional[AgentLogger] = None,
    ):
        self.base = base
        self.memory_source = MemoryToolDataSource(
            memory=memory, query_provider=query_provider, logger=logger
        )

    def get_event_summary(self, event_id: str) -> dict:
        return self.base.get_event_summary(event_id)

    def get_rule_decision(self, event_id: str) -> dict:
        return self.base.get_rule_decision(event_id)

    def get_detection_history(self, event_id: str, window_seconds: float) -> dict:
        return self.base.get_detection_history(event_id, window_seconds)

    def get_event_evidence(self, event_id: str) -> dict:
        return self.base.get_event_evidence(event_id)

    def get_similar_events(
        self, event_id: str, top_k: int, min_similarity: float
    ) -> dict:
        return self.memory_source.get_similar_events(event_id, top_k, min_similarity)
