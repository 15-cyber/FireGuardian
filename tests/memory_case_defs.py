# -*- coding: utf-8 -*-
"""Phase 3-B 固定 Agent Memory 案例定义（共享给单元测试与真实 API A/B 脚本）。"""
from pathlib import Path

from utils.common import DangerLevel, EventStatus, FireDecision, FireEvent, GrowthTrend

_ROOT = Path(__file__).resolve().parent.parent


def _event(event_id, *, duration, fire_area, smoke_area, fire_conf, smoke_conf,
           growth, status=EventStatus.CONFIRMED):
    return FireEvent(
        event_id=event_id,
        status=status,
        duration_seconds=duration,
        total_frames=60,
        positive_frames=40,
        max_fire_area_ratio=fire_area,
        max_smoke_area_ratio=smoke_area,
        avg_fire_confidence=fire_conf,
        avg_smoke_confidence=smoke_conf,
        growth_trend=growth,
        source_type="video",
        source_name="case_fixture",
    )


def _decision(event_id, level, score, confidence, reasons):
    return FireDecision(
        event_id=event_id,
        danger_level=level,
        score=score,
        confidence=confidence,
        reasons=reasons,
        decision_source="rule_engine",
    )


CASES = [
    {
        "case_id": "case_01_real_fire",
        "event": _event(
            "EV-CASE-01", duration=32.0, fire_area=0.38, smoke_area=0.4,
            fire_conf=0.93, smoke_conf=0.87, growth=GrowthTrend.INCREASING,
        ),
        "decision": _decision(
            "EV-CASE-01", DangerLevel.HIGH, 0.82, 0.9,
            ["火焰面积大且持续增长"],
        ),
        "history": _ROOT / "tests" / "fixtures" / "memory" / "case_01_real_fire" / "history.jsonl",
    },
    {
        "case_id": "case_02_smoke_only",
        "event": _event(
            "EV-CASE-02", duration=2.7, fire_area=0.0, smoke_area=0.4,
            fire_conf=0.0, smoke_conf=0.8, growth=GrowthTrend.STABLE,
        ),
        "decision": _decision(
            "EV-CASE-02", DangerLevel.MEDIUM, 0.48, 0.72,
            ["持续烟雾，未见火焰"],
        ),
        "history": _ROOT / "tests" / "fixtures" / "memory" / "case_02_smoke_only" / "history.jsonl",
    },
    {
        "case_id": "case_03_fire_smoke",
        "event": _event(
            "EV-CASE-03", duration=14.0, fire_area=0.18, smoke_area=0.32,
            fire_conf=0.86, smoke_conf=0.81, growth=GrowthTrend.INCREASING,
        ),
        "decision": _decision(
            "EV-CASE-03", DangerLevel.MEDIUM, 0.56, 0.74,
            ["火焰与烟雾并存"],
        ),
        "history": _ROOT / "tests" / "fixtures" / "memory" / "case_03_fire_smoke" / "history.jsonl",
    },
    {
        "case_id": "case_04_fireworks",
        "event": _event(
            "EV-CASE-04", duration=3.2, fire_area=0.02, smoke_area=0.5,
            fire_conf=0.33, smoke_conf=0.79, growth=GrowthTrend.DECREASING,
        ),
        "decision": _decision(
            "EV-CASE-04", DangerLevel.MEDIUM, 0.45, 0.7,
            ["短暂亮光+高烟雾"],
        ),
        "history": _ROOT / "tests" / "fixtures" / "memory" / "case_04_fireworks" / "history.jsonl",
    },
    {
        "case_id": "case_05_normal",
        "event": _event(
            "EV-CASE-05", duration=1.2, fire_area=0.0, smoke_area=0.0,
            fire_conf=0.0, smoke_conf=0.0, growth=GrowthTrend.UNKNOWN,
        ),
        "decision": _decision(
            "EV-CASE-05", DangerLevel.LOW, 0.1, 0.9, ["无有效目标"],
        ),
        "history": _ROOT / "tests" / "fixtures" / "memory" / "case_05_normal" / "history.jsonl",
    },
    {
        "case_id": "case_06_ambiguous",
        "event": _event(
            "EV-CASE-06", duration=4.0, fire_area=0.03, smoke_area=0.45,
            fire_conf=0.4, smoke_conf=0.8, growth=GrowthTrend.DECREASING,
        ),
        "decision": _decision(
            "EV-CASE-06", DangerLevel.MEDIUM, 0.47, 0.7,
            ["烟雾明显但火焰证据不足"],
        ),
        "history": _ROOT / "tests" / "fixtures" / "memory" / "case_06_ambiguous" / "history.jsonl",
    },
]
