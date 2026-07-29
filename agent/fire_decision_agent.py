"""
============================================================
M7 - Fire Decision Agent（火灾决策智能体）
功能：
  - 接收 FireEvent，输出 FireDecision
  - 规则引擎：基于多因素加权评分判断危险等级
  - LLM 增强（可选，默认关闭）
  - 各因素贡献值 Debug 日志

输入: FireEvent（来自 M6 Event Aggregator）
输出: FireDecision（发送给 GUI / 报告 / 截图模块）

可独立运行: python agent/fire_decision_agent.py
============================================================
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    DangerLevel,
    FireEvent,
    FireDecision,
    GrowthTrend,
)


# ======================== 决策智能体 ========================

class FireDecisionAgent:
    """
    火灾决策智能体

    基于稳定规则评估火灾危险等级，可选使用 LLM 增强分析。

    使用方式:
        agent = FireDecisionAgent()
        decision = agent.analyze(fire_event)
        print(decision.danger_level, decision.reason)
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        cfg = load_config()
        agent_cfg = cfg.get("agent", {})

        # 权重
        weights = agent_cfg.get("weights", {})
        self.weight_fire_area = weights.get("fire_area", 0.4)
        self.weight_smoke_area = weights.get("smoke_area", 0.3)
        self.weight_duration = weights.get("duration", 0.2)
        self.weight_growth = weights.get("growth", 0.1)

        # 阈值
        thresholds = agent_cfg.get("thresholds", {})
        self.low_max_score = thresholds.get("low_max_score", 0.3)
        self.medium_max_score = thresholds.get("medium_max_score", 0.6)
        self.fire_area_warning = thresholds.get("fire_area_warning", 0.05)
        self.smoke_area_warning = thresholds.get("smoke_area_warning", 0.08)
        self.duration_warning = thresholds.get("duration_warning", 3.0)
        self.duration_alert = thresholds.get("duration_alert", 10.0)

        # LLM 配置
        llm_cfg = agent_cfg.get("llm", {})
        self.llm_enabled = llm_cfg.get("enabled", False)

        # 统计
        self.total_analyses = 0

    def analyze(self, event: FireEvent) -> FireDecision:
        """
        分析火灾事件，返回决策结果

        参数:
            event: 来自 M6 的 FireEvent 对象

        返回:
            FireDecision 对象
        """
        self.total_analyses += 1
        t0 = time.time()

        # 1. 计算各因素得分 (0~1)
        fire_score = self._score_fire_area(event.max_fire_area_ratio)
        smoke_score = self._score_smoke_area(event.max_smoke_area_ratio)
        duration_score = self._score_duration(event.duration)
        growth_score = self._score_growth(event.growth_trend)

        # 2. 加权综合得分
        total_score = (
            fire_score * self.weight_fire_area
            + smoke_score * self.weight_smoke_area
            + duration_score * self.weight_duration
            + growth_score * self.weight_growth
        )

        # 3. 确定危险等级
        danger_level, level_reason = self._determine_danger_level(
            total_score, event
        )

        # 4. 生成建议
        suggestion = self._generate_suggestion(danger_level, event)

        # 5. Debug 信息
        debug_info = {
            "total_score": round(total_score, 4),
            "factors": {
                "fire_area": {"value": round(event.max_fire_area_ratio, 4), "score": round(fire_score, 4), "weight": self.weight_fire_area},
                "smoke_area": {"value": round(event.max_smoke_area_ratio, 4), "score": round(smoke_score, 4), "weight": self.weight_smoke_area},
                "duration": {"value": round(event.duration, 2), "score": round(duration_score, 4), "weight": self.weight_duration},
                "growth": {"value": event.growth_trend, "score": round(growth_score, 4), "weight": self.weight_growth},
            },
            "config": {
                "low_max_score": self.low_max_score,
                "medium_max_score": self.medium_max_score,
                "fire_area_warning": self.fire_area_warning,
                "smoke_area_warning": self.smoke_area_warning,
                "duration_warning": self.duration_warning,
                "duration_alert": self.duration_alert,
            },
        }

        # 6. 构建决策（规则模式）
        decision = FireDecision(
            danger_level=danger_level,
            reason=level_reason,
            suggestion=suggestion,
            source="rule",
            confidence=round(total_score, 4),
            decision_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            debug_info=debug_info,
        )

        # 7. 可选 LLM 增强
        if self.llm_enabled:
            llm_result = self._try_llm_enhance(event, decision)
            if llm_result is not None:
                decision = llm_result

        elapsed = (time.time() - t0) * 1000
        # 内部调试日志
        self._log_decision(event, decision, elapsed)

        return decision

    # ======================== 评分函数 (0~1) ========================

    def _score_fire_area(self, area_ratio: float) -> float:
        """火焰面积评分：越大越危险"""
        if area_ratio <= 0:
            return 0.0
        # 0.2 以上算满分
        return min(area_ratio / 0.2, 1.0)

    def _score_smoke_area(self, area_ratio: float) -> float:
        """烟雾面积评分"""
        if area_ratio <= 0:
            return 0.0
        return min(area_ratio / 0.3, 1.0)

    def _score_duration(self, duration: float) -> float:
        """持续时间评分：越长越危险"""
        if duration <= 0:
            return 0.0
        # 30 秒以上算满分
        return min(duration / 30.0, 1.0)

    def _score_growth(self, trend: str) -> float:
        """增长趋势评分"""
        mapping = {
            GrowthTrend.INCREASING: 0.9,
            GrowthTrend.INCREASING.value: 0.9,
            GrowthTrend.STABLE: 0.5,
            GrowthTrend.STABLE.value: 0.5,
            GrowthTrend.DECREASING: 0.2,
            GrowthTrend.DECREASING.value: 0.2,
        }
        return mapping.get(trend, 0.3)

    # ======================== 危险等级判定 ========================

    def _determine_danger_level(self, score: float, event: FireEvent) -> tuple:
        """
        根据综合得分和事件信息判定危险等级
        返回 (danger_level, reason)
        """
        reasons = []

        # 硬性条件：超过阈值直接升级
        if event.max_fire_area_ratio >= self.fire_area_warning:
            reasons.append(f"火焰面积占比 {event.max_fire_area_ratio:.2%} 超过预警阈值")
        if event.max_smoke_area_ratio >= self.smoke_area_warning:
            reasons.append(f"烟雾面积占比 {event.max_smoke_area_ratio:.2%} 超过预警阈值")
        if event.duration >= self.duration_alert:
            reasons.append(f"持续 {event.duration:.0f} 秒超过严重告警阈值")
        elif event.duration >= self.duration_warning:
            reasons.append(f"持续 {event.duration:.0f} 秒超过预警阈值")

        # 增长趋势
        if event.growth_trend == GrowthTrend.INCREASING:
            reasons.append("火势呈扩大趋势")
        elif event.growth_trend == GrowthTrend.DECREASING:
            reasons.append("火势呈减弱趋势")

        # 综合评分判定
        if score >= self.medium_max_score:
            level = DangerLevel.HIGH
            if not reasons:
                reasons.append(f"综合风险评分 {score:.2f}，等级为高")
        elif score >= self.low_max_score:
            level = DangerLevel.MEDIUM
            if not reasons:
                reasons.append(f"综合风险评分 {score:.2f}，等级为中")
        else:
            level = DangerLevel.LOW
            reasons.append(f"综合风险评分 {score:.2f}，等级为低")

        reason = "；".join(reasons)
        return level.value, reason

    # ======================== 建议生成 ========================

    def _generate_suggestion(self, danger_level: str, event: FireEvent) -> str:
        """根据危险等级和事件信息生成建议"""
        suggestions = []

        if danger_level == DangerLevel.HIGH:
            suggestions.append("立即启动消防应急预案")
            suggestions.append("通知值班负责人和安保人员")
            suggestions.append("确认现场人员安全，准备疏散")
            if event.growth_trend == GrowthTrend.INCREASING:
                suggestions.append("火势正在扩大，建议立即联系119消防部门")
        elif danger_level == DangerLevel.MEDIUM:
            suggestions.append("安排人员前往现场确认情况")
            suggestions.append("持续监控火势变化")
            suggestions.append("准备应急设备")
        else:
            suggestions.append("继续保持监控")
            suggestions.append("关注火势变化趋势")

        return "；".join(suggestions)

    # ======================== LLM 增强（预留） ========================

    def _try_llm_enhance(self, event: FireEvent, rule_decision: FireDecision) -> Optional[FireDecision]:
        """
        尝试 LLM 增强分析
        失败时返回 None，使用规则结果
        """
        try:
            # TODO: 后续接入 DeepSeek / OpenAI API
            # prompt = self._build_llm_prompt(event)
            # response = call_llm_api(prompt)
            # return self._parse_llm_response(response, rule_decision)
            return None
        except Exception:
            return None

    # ======================== Debug 日志 ========================

    def _log_decision(self, event: FireEvent, decision: FireDecision, elapsed_ms: float):
        """记录决策详情到内部日志"""
        dbg = decision.debug_info
        factors = dbg.get("factors", {})
        factor_str = " | ".join(
            f"{k}={v.get('score', 0):.3f}" for k, v in sorted(factors.items())
        )
        print(
            f"  [Agent] 事件 {event.event_id[:20]:20s} | "
            f"等级 {decision.danger_level:6s} | "
            f"评分 {decision.confidence:.3f} | "
            f"耗时 {elapsed_ms:.0f}ms"
        )
        print(f"          {factor_str}")
        if decision.debug_info.get("factors"):
            f = decision.debug_info["factors"]
            print(f"          火焰={f.get('fire_area',{}).get('value',0):.4f} "
                  f"烟雾={f.get('smoke_area',{}).get('value',0):.4f} "
                  f"持续={f.get('duration',{}).get('value',0):.1f}s "
                  f"趋势={f.get('growth',{}).get('value','?')}")


# ======================== 命令行测试 ========================

def main():
    """独立运行测试"""
    print("\n  FireGuardian M7 - Fire Decision Agent 测试")
    print("  ===============================================\n")

    agent = FireDecisionAgent()

    # 构造测试 FireEvent
    test_cases = [
        # (event_id, fire_area, smoke_area, duration, growth)
        ("低风险-小面积短时", 0.01, 0.01, 2.0, GrowthTrend.STABLE),
        ("中风险-中等面积",  0.08, 0.05, 5.0, GrowthTrend.STABLE),
        ("高风险-大面积长时", 0.15, 0.10, 12.0, GrowthTrend.INCREASING),
        ("火势减弱-下降趋势", 0.10, 0.08, 8.0, GrowthTrend.DECREASING),
    ]

    for name, fa, sa, dur, trend in test_cases:
        trend_val = trend.value if hasattr(trend, "value") else trend
        from utils.common import EventStatus
        event = FireEvent(
            event_id=f"TEST-{name}",
            status=EventStatus.ACTIVE,
            start_timestamp=0.0,
            last_timestamp=dur,
            total_frames=int(dur / 0.5),
            positive_frames=int(dur / 0.5),
            max_fire_area_ratio=fa,
            max_smoke_area_ratio=sa,
            avg_fire_confidence=0.85,
            avg_smoke_confidence=0.75,
            growth_trend=trend_val,
            source_type="test",
            source_name="unit_test",
        )
        print(f"\n  ── [{name}] ──")
        decision = agent.analyze(event)
        print(f"  → 危险等级: {decision.danger_level}")
        print(f"  → 原因: {decision.reason}")
        print(f"  → 建议: {decision.suggestion}")

    print("\n  M7 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
