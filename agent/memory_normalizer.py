"""
============================================================
V2 Agent - Memory Normalizer（Phase 3-A）

将 logs/events.jsonl 的真实记录标准化为 EventMemoryRecord：
  - 字段名兼容（max_*_area_ratio / avg_*_confidence / *_timestamp / *_time_seconds）；
  - duration 与 duration_seconds 兼容（duration_source / duration_unit 审计）；
  - 枚举标准化、缺失字段默认值、非法记录跳过（单条坏记录不影响整体）；
  - rule_decision 由加载层按 event_id join 后注入到 "rule_decision" 键；
  - final_level 仅在存在 hybrid_decision 记录时设置，否则保持 None。
============================================================
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.memory_models import EventMemoryRecord
from utils.common import FireDecision, FireEvent


_GROWTH_ALIASES = {
    "increasing": "increasing", "increase": "increasing", "rising": "increasing",
    "stable": "stable", "steady": "stable", "flat": "stable",
    "decreasing": "decreasing", "decrease": "decreasing", "declining": "decreasing",
    "unknown": "unknown",
}

_LEVELS = ("low", "medium", "high")


def _plain(value: Any) -> Any:
    """Enum 取 .value，避免 str(Enum) 产生 'DangerLevel.MEDIUM' 这类字符串。"""
    return value.value if isinstance(value, Enum) else value


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(data: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _normalize_growth(value: Any) -> str:
    if value is None:
        return "unknown"
    return _GROWTH_ALIASES.get(str(_plain(value)).strip().lower(), "unknown")


def _normalize_level(value: Any) -> str:
    level = str(_plain(value) or "").strip().lower()
    return level if level in _LEVELS else "low"


def normalize_event_record(raw_event: Any) -> Optional[EventMemoryRecord]:
    """标准化单条历史事件；非法记录（非 dict / 无 event_id）返回 None。"""
    if not isinstance(raw_event, dict):
        return None
    data = (
        raw_event.get("details")
        if isinstance(raw_event.get("details"), dict)
        else raw_event
    )
    event_id = str(data.get("event_id", "")).strip()
    if not event_id:
        return None

    start = _to_float(_first(data, "start_time_seconds", "start_timestamp"), 0.0)
    end_raw = _first(data, "end_time_seconds", "end_timestamp")
    end = _to_float(end_raw, 0.0) if end_raw is not None else None

    duration_seconds_raw = _first(data, "duration_seconds")
    duration_legacy = _first(data, "duration")
    if duration_seconds_raw is not None:
        duration_seconds = _to_float(duration_seconds_raw)
        duration_source = "normalized"
        duration_unit = "seconds"
    elif duration_legacy is not None:
        duration_seconds = _to_float(duration_legacy)
        duration_source = "legacy"
        duration_unit = "seconds"
        if end is not None and start >= 0 and end >= start and end > start:
            diff = end - start
            if abs(diff - duration_seconds) > max(0.5, 0.1 * max(diff, duration_seconds, 1.0)):
                duration_unit = "unknown"
        elif duration_seconds <= 0:
            duration_unit = "unknown"
    elif end is not None and start >= 0 and end > start:
        duration_seconds = round(end - start, 4)
        duration_source = "computed"
        duration_unit = "seconds"
    else:
        duration_seconds = 0.0
        duration_source = "none"
        duration_unit = "unknown"

    fire_area = _to_float(_first(data, "max_fire_area_ratio", "fire_area_ratio"))
    smoke_area = _to_float(_first(data, "max_smoke_area_ratio", "smoke_area_ratio"))
    max_fire_conf = _first(data, "max_fire_confidence")
    max_smoke_conf = _first(data, "max_smoke_confidence")
    avg_fire_conf = _first(data, "avg_fire_confidence")
    avg_smoke_conf = _first(data, "avg_smoke_confidence")
    if max_fire_conf is not None and max_smoke_conf is not None:
        confidence_source = "max"
    elif avg_fire_conf is not None or avg_smoke_conf is not None:
        confidence_source = "avg"
    else:
        confidence_source = "none"
    fire_conf = _to_float(max_fire_conf if max_fire_conf is not None else avg_fire_conf)
    smoke_conf = _to_float(max_smoke_conf if max_smoke_conf is not None else avg_smoke_conf)

    # 规则决策（加载层 join 后注入 rule_decision；兼容 metadata.danger_level）
    decision = data.get("rule_decision") if isinstance(data.get("rule_decision"), dict) else {}
    rule_level = decision.get("danger_level") or decision.get("level")
    rule_level_source = "decision" if rule_level else "none"
    if not rule_level:
        meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        rule_level = meta.get("danger_level")
        if rule_level:
            rule_level_source = "metadata"

    # Hybrid final_level：仅当存在真实 hybrid 记录时设置，否则保持 None
    final_level: Optional[str] = None
    hybrid = data.get("hybrid_decision")
    if isinstance(hybrid, dict):
        fl = str(hybrid.get("final_level") or "").strip().lower()
        if fl in _LEVELS:
            final_level = fl

    status = str(_plain(data.get("status")) or "").strip()
    source = str(_plain(_first(data, "source_type", "source_name", default="")) or "")

    return EventMemoryRecord(
        event_id=event_id,
        start_time_seconds=start if start > 0 else None,
        end_time_seconds=end if end and end > 0 else None,
        duration_seconds=round(duration_seconds, 4),
        fps=_to_float(_first(data, "fps"), 0.0) or None,
        fire_detected=fire_area > 0 or fire_conf > 0,
        smoke_detected=smoke_area > 0 or smoke_conf > 0,
        fire_area_ratio=round(fire_area, 6),
        smoke_area_ratio=round(smoke_area, 6),
        max_fire_confidence=round(fire_conf, 4),
        max_smoke_confidence=round(smoke_conf, 4),
        growth_trend=_normalize_growth(data.get("growth_trend")),
        rule_level=_normalize_level(rule_level),
        rule_score=round(_to_float(decision.get("score")), 4),
        rule_confidence=round(_to_float(decision.get("confidence")), 4),
        rule_level_source=rule_level_source,
        source=source,
        final_level=final_level,
        status=status,
        duration_source=duration_source,
        duration_unit=duration_unit,
        confidence_source=confidence_source,
        is_ended=status == "ended",
    )


def build_query_record(
    event: FireEvent,
    decision: Optional[FireDecision] = None,
) -> EventMemoryRecord:
    """从当前 FireEvent / FireDecision 构建查询记录（复用同一套标准化逻辑）。"""
    raw: dict = event.to_dict()
    if decision is not None:
        raw["rule_decision"] = decision.to_dict()
    record = normalize_event_record(raw)
    if record is None:
        raise ValueError("无法从 FireEvent 构建查询记录（event_id 为空）")
    return record


@dataclass
class NormalizeResult:
    """批量标准化结果。"""

    valid_records: List[EventMemoryRecord] = field(default_factory=list)
    invalid_count: int = 0
    invalid_reasons: List[str] = field(default_factory=list)


def normalize_many(records: List[Any]) -> NormalizeResult:
    result = NormalizeResult()
    for raw in records:
        record = normalize_event_record(raw)
        if record is None:
            result.invalid_count += 1
            event_id = raw.get("event_id") if isinstance(raw, dict) else None
            result.invalid_reasons.append(f"invalid record: {event_id!r}")
        else:
            result.valid_records.append(record)
    return result
