"""
============================================================
V2 Agent - StructuredMemoryRetriever（Phase 3-A）

透明、可解释的加权相似度检索（不使用 Embedding / 向量库）。

特征与初始权重（合计 1.00，可配置）：
  fire_presence 0.20 / smoke_presence 0.20 / fire_area 0.15 /
  smoke_area 0.15 / duration 0.10 / risk_level 0.10 /
  confidence 0.05 / growth_trend 0.05

约束：
  - 默认只检索已结束历史事件（only_ended_events=True）；
  - 当前事件严格排除；
  - 低于 min_similarity 不返回；排序 score DESC，同分优先同 rule_level，
    再按 duration 差异小者优先；最多 top_k 条。
============================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.memory_models import (
    EventMemoryRecord,
    MemoryConfig,
    MemorySearchResult,
    SimilarEvent,
)


_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}
_EPS = 1e-9
_MATCH_THRESHOLD = 0.5

_WEIGHT_KEYS = (
    "fire_presence",
    "smoke_presence",
    "fire_area",
    "smoke_area",
    "duration",
    "risk_level",
    "confidence",
    "growth_trend",
)


def _ratio_similarity(a: float, b: float) -> float:
    """数值相对相似度：1 - |a-b| / max(a,b,eps)，限制 [0,1]；双零视为 1.0。"""
    if a <= 0 and b <= 0:
        return 1.0
    denom = max(a, b, _EPS)
    return max(0.0, min(1.0, 1.0 - abs(a - b) / denom))


def _level_similarity(a: str, b: str) -> float:
    rank_a = _LEVEL_RANK.get(a, 1)
    rank_b = _LEVEL_RANK.get(b, 1)
    return 1.0 - abs(rank_a - rank_b) / 2.0


def _confidence_similarity(
    q_fire: float, q_smoke: float, h_fire: float, h_smoke: float
) -> float:
    fire_sim = max(0.0, min(1.0, 1.0 - abs(q_fire - h_fire)))
    smoke_sim = max(0.0, min(1.0, 1.0 - abs(q_smoke - h_smoke)))
    return (fire_sim + smoke_sim) / 2.0


def _growth_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a == "unknown" or b == "unknown":
        return 0.5
    return 0.0


class StructuredMemoryRetriever:
    """结构化相似事件检索器。"""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        weights = self.config.weights or {}
        total = sum(float(weights.get(k, 0.0)) for k in _WEIGHT_KEYS)
        self._weights = {
            key: (float(weights.get(key, 0.0)) / total if total > 0 else 0.0)
            for key in _WEIGHT_KEYS
        }

    def _feature_similarities(
        self, query: EventMemoryRecord, history: EventMemoryRecord
    ) -> Dict[str, float]:
        return {
            "fire_presence": 1.0 if query.fire_detected == history.fire_detected else 0.0,
            "smoke_presence": 1.0 if query.smoke_detected == history.smoke_detected else 0.0,
            "fire_area": _ratio_similarity(query.fire_area_ratio, history.fire_area_ratio),
            "smoke_area": _ratio_similarity(query.smoke_area_ratio, history.smoke_area_ratio),
            "duration": _ratio_similarity(query.duration_seconds, history.duration_seconds),
            "risk_level": _level_similarity(query.rule_level, history.rule_level),
            "confidence": _confidence_similarity(
                query.max_fire_confidence,
                query.max_smoke_confidence,
                history.max_fire_confidence,
                history.max_smoke_confidence,
            ),
            "growth_trend": _growth_similarity(query.growth_trend, history.growth_trend),
        }

    def search(
        self,
        query: EventMemoryRecord,
        history: List[EventMemoryRecord],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> MemorySearchResult:
        """检索相似历史事件。history 中当前事件会自动排除。"""
        top_k = int(top_k if top_k is not None else self.config.top_k)
        min_sim = float(
            min_similarity if min_similarity is not None else self.config.min_similarity
        )
        top_k = max(1, min(top_k, int(self.config.max_top_k)))

        candidates = [h for h in history if h.event_id != query.event_id]
        if self.config.only_ended_events:
            candidates = [h for h in candidates if h.is_ended]
        candidate_count = len(candidates)

        scored: List[tuple] = []
        for h in candidates:
            sims = self._feature_similarities(query, h)
            score = sum(self._weights[key] * sims[key] for key in _WEIGHT_KEYS)
            matched = [key for key, value in sims.items() if value >= _MATCH_THRESHOLD]
            scored.append((score, h, matched))

        filtered = [item for item in scored if item[0] >= min_sim]
        filtered.sort(
            key=lambda item: (
                -item[0],
                0 if item[1].rule_level == query.rule_level else 1,
                abs(item[1].duration_seconds - query.duration_seconds),
            )
        )
        top = filtered[:top_k]

        results = [
            SimilarEvent(
                event_id=h.event_id,
                similarity_score=round(score, 4),
                matched_features=matched,
                rule_level=h.rule_level,
                duration_seconds=round(h.duration_seconds, 4),
                fire_detected=h.fire_detected,
                smoke_detected=h.smoke_detected,
                fire_area_ratio=round(h.fire_area_ratio, 6),
                smoke_area_ratio=round(h.smoke_area_ratio, 6),
                final_level=h.final_level,
                growth_trend=h.growth_trend,
                status=h.status,
            )
            for score, h, matched in top
        ]
        return MemorySearchResult(
            memory_available=True,
            reason=None,
            query_event={
                "event_id": query.event_id,
                "duration_seconds": round(query.duration_seconds, 4),
                "fire_detected": query.fire_detected,
                "smoke_detected": query.smoke_detected,
                "rule_level": query.rule_level,
            },
            results=results,
            candidate_count=candidate_count,
            matched_count=len(filtered),
        )
