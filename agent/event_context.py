"""
============================================================
V2 Agent - EventContext（Phase 1）

职责：把 M6 聚合出的 FireEvent（秒制时间 / union area 面积特征）
与 M7 的规则决策 FireDecision，转换成 Agent 可理解的结构化上下文。

职责边界（按 V2 计划书修正后的表述）：
  M6 负责事件聚合与秒制时间、面积等事件特征；
  M7 负责消费这些特征进行规则风险决策；
  EventContext 只做视图转换，不修改 M6/M7 的任何行为。
============================================================
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import FireDecision, FireEvent


def _enum_value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _max_class_confidence(event: FireEvent, cls: str) -> float:
    """从事件携带的逐帧检测中取该类最大置信度；无检测数据时回退到均值。"""
    best = 0.0
    for det in event.detections:
        boxes = det.fire_boxes if cls == "fire" else det.smoke_boxes
        best = max(best, max((b.confidence for b in boxes), default=0.0))
    if best > 0.0:
        return round(float(best), 4)
    if cls == "fire":
        return round(float(event.avg_fire_confidence), 4)
    return round(float(event.avg_smoke_confidence), 4)


def build_event_context(
    event: FireEvent,
    decision: Optional[FireDecision] = None,
) -> dict:
    """构造 Agent 事件上下文（V2 计划书 §8 结构）。"""
    if event is None or not getattr(event, "event_id", ""):
        raise ValueError("event_id 不能为空")

    duration = float(event.duration_seconds or event.duration or 0.0)
    fire_area = float(event.max_fire_area_ratio or 0.0)
    smoke_area = float(event.max_smoke_area_ratio or 0.0)
    fire_conf = _max_class_confidence(event, "fire")
    smoke_conf = _max_class_confidence(event, "smoke")

    ctx: Dict[str, Any] = {
        "event_id": event.event_id,
        "status": _enum_value(event.status),
        "duration_seconds": round(duration, 3),
        "fire": {
            "detected": fire_area > 0.0 or fire_conf > 0.0,
            "max_confidence": fire_conf,
            "max_area_ratio": round(fire_area, 6),
        },
        "smoke": {
            "detected": smoke_area > 0.0 or smoke_conf > 0.0,
            "max_confidence": smoke_conf,
            "max_area_ratio": round(smoke_area, 6),
        },
        "growth_trend": _enum_value(event.growth_trend),
        "total_frames": int(event.total_frames),
        "positive_ratio": round(float(event.positive_ratio), 4),
        "rule_decision": None,
    }

    if decision is not None:
        ctx["rule_decision"] = {
            "level": _enum_value(decision.danger_level),
            "score": round(float(decision.score or 0.0), 4),
            "confidence": round(float(decision.confidence or 0.0), 4),
            "reasons": list(decision.reasons or []),
        }
    return ctx


def summarize_event_context(ctx: dict) -> str:
    """生成紧凑的文本摘要，用于 Prompt 上下文（不发送完整日志）。"""
    fire = ctx.get("fire", {})
    smoke = ctx.get("smoke", {})
    rule = ctx.get("rule_decision")
    rule_text = (
        f"规则决策 level={rule['level']} score={rule['score']} "
        f"confidence={rule['confidence']} reasons={rule['reasons']}"
        if rule
        else "规则决策：暂无"
    )
    return (
        f"事件 {ctx.get('event_id')}：状态={ctx.get('status')}，"
        f"持续时长={ctx.get('duration_seconds')}s，"
        f"增长趋势={ctx.get('growth_trend')}，正样本占比={ctx.get('positive_ratio')}；"
        f"火焰：检测到={fire.get('detected')} 最大置信度={fire.get('max_confidence')} "
        f"最大面积占比={fire.get('max_area_ratio')}；"
        f"烟雾：检测到={smoke.get('detected')} 最大置信度={smoke.get('max_confidence')} "
        f"最大面积占比={smoke.get('max_area_ratio')}；{rule_text}"
    )
