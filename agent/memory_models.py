"""
============================================================
V2 Agent - Memory 数据模型（Phase 3-A）

定义：
  - EventMemoryRecord：统一历史事件结构（字段名按 logs/events.jsonl
    真实结构适配，不假设不存在的字段）
  - MemoryConfig：检索配置（权重 / top_k / 阈值 / 仅已结束事件）
  - SimilarEvent / MemorySearchResult：检索结果结构

语义约束：
  - final_level 仅在存在真实 Hybrid 记录时设置；日志没有时保持 None，
    禁止用 rule_level 冒充 final_level；
  - 默认只检索 is_ended=True 的历史事件。
============================================================
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_MEMORY_WEIGHTS: Dict[str, float] = {
    "fire_presence": 0.20,
    "smoke_presence": 0.20,
    "fire_area": 0.15,
    "smoke_area": 0.15,
    "duration": 0.10,
    "risk_level": 0.10,
    "confidence": 0.05,
    "growth_trend": 0.05,
}


@dataclass
class EventMemoryRecord:
    """标准化后的历史事件记忆记录。"""

    event_id: str = ""
    start_time_seconds: Optional[float] = None
    end_time_seconds: Optional[float] = None
    duration_seconds: float = 0.0
    fps: Optional[float] = None

    fire_detected: bool = False
    smoke_detected: bool = False

    fire_area_ratio: float = 0.0
    smoke_area_ratio: float = 0.0

    max_fire_confidence: float = 0.0
    max_smoke_confidence: float = 0.0

    growth_trend: str = "unknown"

    rule_level: str = "low"
    rule_score: float = 0.0
    rule_confidence: float = 0.0
    rule_level_source: str = "none"  # decision / metadata / default

    source: str = ""
    final_level: Optional[str] = None  # 无 Hybrid 日志时为 None
    status: str = ""

    # ---- 审计 / 兼容字段 ----
    duration_source: str = "normalized"  # normalized / legacy / computed / none
    duration_unit: str = "seconds"       # seconds / unknown
    confidence_source: str = "none"      # max / avg / none
    is_ended: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryConfig:
    """Memory 检索配置（默认值来自任务书 §9/§11/§12）。"""

    top_k: int = 5
    max_top_k: int = 10
    min_similarity: float = 0.65
    only_ended_events: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MEMORY_WEIGHTS))


@dataclass
class SimilarEvent:
    """单条相似历史事件（结构化摘要，供 Agent 使用）。"""

    event_id: str = ""
    similarity_score: float = 0.0
    matched_features: List[str] = field(default_factory=list)
    rule_level: str = "low"
    duration_seconds: float = 0.0
    fire_detected: bool = False
    smoke_detected: bool = False
    fire_area_ratio: float = 0.0
    smoke_area_ratio: float = 0.0
    final_level: Optional[str] = None
    growth_trend: str = "unknown"
    status: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemorySearchResult:
    """Memory 检索结果（Tool 返回格式，保持紧凑）。"""

    memory_available: bool = True
    reason: Optional[str] = None
    query_event: Optional[dict] = None
    results: List[SimilarEvent] = field(default_factory=list)
    candidate_count: int = 0
    matched_count: int = 0

    def to_dict(self) -> dict:
        return {
            "query_event": self.query_event,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "candidate_count": self.candidate_count,
                "matched_count": self.matched_count,
            },
            "memory_available": self.memory_available,
            "reason": self.reason,
        }
