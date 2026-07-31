"""
============================================================
M7 - Fire Decision Agent（火灾决策智能体）v2
功能：
  - 接收 FireEvent，输出结构化 FireDecision
  - 归一化评分 + 关键规则覆盖（override）混合决策
  - 独立计算决策可信度（confidence）
  - 结构化 reasons / suggestions / summary
  - 输入校验：非法数据明确报错
  - LLM 增强（可选，默认关闭，失败不影响核心结果）

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
from typing import List, Optional, Tuple

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    DangerLevel,
    EventStatus,
    FireEvent,
    FireDecision,
    GrowthTrend,
)


# ======================== 决策智能体 ========================

class FireDecisionAgent:
    """
    火灾决策智能体

    决策流程:
        validate_input(event)
        → features = normalize(event)           # 各因素归一化到 0~1
        → score = weighted_sum(features)        # 加权评分
        → base_level = map_score_to_level(score)
        → final_level, override_reasons = apply_override_rules(event, base_level)
        → confidence = calculate_confidence(event)
        → build reasons / suggestions / summary
        → FireDecision
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

        # 基础阈值
        thresholds = agent_cfg.get("thresholds", {})
        self.low_max_score = thresholds.get("low_max_score", 0.3)
        self.medium_max_score = thresholds.get("medium_max_score", 0.6)
        self.fire_area_warning = thresholds.get("fire_area_warning", 0.05)
        self.smoke_area_warning = thresholds.get("smoke_area_warning", 0.08)
        self.duration_warning = thresholds.get("duration_warning", 3.0)
        self.duration_alert = thresholds.get("duration_alert", 10.0)

        # 硬性覆盖规则（影响最终等级）
        overrides = agent_cfg.get("overrides", {})
        self.high_override = overrides.get("high", {})
        self.smoke_medium_override = overrides.get("smoke_only_medium", {})

        # LLM 配置
        llm_cfg = agent_cfg.get("llm", {})
        self.llm_enabled = llm_cfg.get("enabled", False)

        # 统计
        self.total_analyses = 0

    # ======================== 主入口 ========================

    def analyze(self, event: FireEvent) -> FireDecision:
        """
        分析火灾事件，返回结构化决策

        参数:
            event: 来自 M6 的 FireEvent

        返回:
            FireDecision 对象

        异常:
            ValueError: 输入不合法
        """
        self.total_analyses += 1
        t0 = time.time()

        # 1. 输入校验
        self._validate_event(event)

        # 2. 归一化评分
        features = self._extract_features(event)
        score = self._calculate_score(features)

        # 3. 基础等级
        base_level = self._map_score_to_level(score)

        # 4. 硬性覆盖规则（可提升/降级）
        final_level, override_reasons = self._apply_override_rules(event, base_level)

        # 5. 独立可信度
        confidence = self._calculate_confidence(event)

        # 6. 构建原因和建议
        reasons = self._build_reasons(event, features, override_reasons)
        suggestions = self._build_suggestions(final_level, confidence, event)

        # 7. 汇总
        summary = self._build_summary(final_level, reasons)

        # 8. Debug 信息
        debug_info = {
            "base_level": base_level,
            "override_applied": final_level != base_level,
            "features": {k: round(v, 4) for k, v in features.items()},
            "weights": {
                "fire_area": self.weight_fire_area,
                "smoke_area": self.weight_smoke_area,
                "duration": self.weight_duration,
                "growth": self.weight_growth,
            },
        }

        decision = FireDecision(
            event_id=event.event_id,
            danger_level=final_level.value if hasattr(final_level, "value") else final_level,
            score=round(score, 4),
            confidence=round(confidence, 4),
            reasons=reasons,
            suggestions=suggestions,
            summary=summary,
            decision_source="rule_engine",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            debug_info=debug_info,
        )

        # 9. 可选 LLM 增强
        if self.llm_enabled:
            llm_result = self._try_llm_enhance(event, decision)
            if llm_result is not None:
                decision = llm_result

        elapsed = (time.time() - t0) * 1000
        self._log_decision(event, decision, elapsed)

        return decision

    # ======================== 输入校验 ========================

    def _validate_event(self, event: FireEvent):
        """校验输入事件是否合法"""
        if not event.event_id:
            raise ValueError("FireEvent 缺少 event_id")

        if event.max_fire_area_ratio < 0 or event.max_fire_area_ratio > 1:
            raise ValueError(
                f"max_fire_area_ratio 超出范围 [0, 1]: {event.max_fire_area_ratio}"
            )

        if event.max_smoke_area_ratio < 0 or event.max_smoke_area_ratio > 1:
            raise ValueError(
                f"max_smoke_area_ratio 超出范围 [0, 1]: {event.max_smoke_area_ratio}"
            )

        if event.duration < 0:
            raise ValueError(f"事件持续时长不能为负数: {event.duration}s")

        if event.total_frames < 0 or event.positive_frames < 0:
            raise ValueError("帧数不能为负数")

    # ======================== 归一化评分 ========================

    def _extract_features(self, event: FireEvent) -> dict:
        """提取并归一化各因素到 0~1 范围"""
        fire_score = self._normalize_area(event.max_fire_area_ratio, norm_max=0.2)
        smoke_score = self._normalize_area(event.max_smoke_area_ratio, norm_max=0.3)
        duration_score = self._normalize_duration(event.duration, norm_max=30.0)
        growth_score = self._normalize_growth(event.growth_trend)

        return {
            "fire_area": fire_score,
            "smoke_area": smoke_score,
            "duration": duration_score,
            "growth": growth_score,
        }

    @staticmethod
    def _normalize_area(value: float, norm_max: float) -> float:
        """面积归一化：value / norm_max，截断到 [0, 1]"""
        if value <= 0:
            return 0.0
        return min(value / norm_max, 1.0)

    @staticmethod
    def _normalize_duration(seconds: float, norm_max: float) -> float:
        """时长归一化：seconds / norm_max，截断到 [0, 1]"""
        if seconds <= 0:
            return 0.0
        return min(seconds / norm_max, 1.0)

    @staticmethod
    def _normalize_growth(trend) -> float:
        """趋势归一化到 [0, 1]"""
        mapping = {
            GrowthTrend.INCREASING: 0.9,
            GrowthTrend.INCREASING.value: 0.9,
            GrowthTrend.STABLE: 0.5,
            GrowthTrend.STABLE.value: 0.5,
            GrowthTrend.DECREASING: 0.2,
            GrowthTrend.DECREASING.value: 0.2,
            GrowthTrend.UNKNOWN: 0.3,
            GrowthTrend.UNKNOWN.value: 0.3,
        }
        return mapping.get(trend, 0.3)

    def _calculate_score(self, features: dict) -> float:
        """加权综合评分"""
        return (
            features["fire_area"] * self.weight_fire_area
            + features["smoke_area"] * self.weight_smoke_area
            + features["duration"] * self.weight_duration
            + features["growth"] * self.weight_growth
        )

    def _map_score_to_level(self, score: float) -> str:
        """评分 → 基础等级"""
        if score >= self.medium_max_score:
            return DangerLevel.HIGH
        elif score >= self.low_max_score:
            return DangerLevel.MEDIUM
        else:
            return DangerLevel.LOW

    # ======================== 硬性覆盖规则 ========================

    def _apply_override_rules(self, event: FireEvent, base_level: str) -> Tuple[str, List[str]]:
        """
        应用硬性覆盖规则，修正评分无法表达的关键组合风险
        返回 (最终等级, 覆盖原因列表)
        """
        override_reasons: List[str] = []

        # 规则 1: 大面积 + 长时间 + 扩大趋势 → 强制 high
        high_cfg = self.high_override
        if high_cfg:
            min_fire = high_cfg.get("min_fire_area_ratio", 0.15)
            min_dur = high_cfg.get("min_duration_seconds", 10)
            require_growth = high_cfg.get("require_growth", True)

            fire_ok = event.max_fire_area_ratio >= min_fire
            dur_ok = event.duration >= min_dur
            growth_ok = (not require_growth) or event.growth_trend in (
                GrowthTrend.INCREASING, GrowthTrend.INCREASING.value
            )

            if fire_ok and dur_ok and growth_ok:
                if base_level != DangerLevel.HIGH:
                    override_reasons.append(
                        f"火焰面积 {event.max_fire_area_ratio:.1%} ≥ {min_fire:.0%}"
                        f" 且持续 {event.duration:.0f}s ≥ {min_dur:.0f}s"
                        f" 且趋势扩大 → 强制升级为高风险"
                    )
                return DangerLevel.HIGH, override_reasons

        # 规则 2: 持续超长时间 → 至少 medium（防止面积小但持续久被低估）
        if event.duration >= self.duration_alert and base_level == DangerLevel.LOW:
            override_reasons.append(
                f"持续 {event.duration:.0f} 秒超过严重告警阈值"
                f" → 即使面积较小也提升为中等风险"
            )
            base_level = DangerLevel.MEDIUM

        # 规则 3: 浓烟无明火 + 长时间 → 至少 medium
        smoke_cfg = self.smoke_medium_override
        if smoke_cfg:
            min_smoke = smoke_cfg.get("min_smoke_area_ratio", 0.15)
            min_dur = smoke_cfg.get("min_duration_seconds", 10)

            smoke_ok = event.max_smoke_area_ratio >= min_smoke
            dur_ok = event.duration >= min_dur

            if smoke_ok and dur_ok and event.max_fire_area_ratio < 0.05:
                if base_level == DangerLevel.LOW:
                    override_reasons.append(
                        f"烟雾面积 {event.max_smoke_area_ratio:.1%} ≥ {min_smoke:.0%}"
                        f" 且持续 {event.duration:.0f}s ≥ {min_dur:.0f}s"
                        f" 无明火 → 提升为中等风险"
                    )
                return DangerLevel.MEDIUM, override_reasons

        return base_level, override_reasons

    # ======================== 独立可信度 ========================

    def _calculate_confidence(self, event: FireEvent) -> float:
        """
        计算决策可信度（与危险评分相互独立）
        考虑: 阳性帧比例 / 平均检测置信度 / 事件稳定性
        """
        # 因素 1: 阳性帧比例（证据一致性）
        positive_ratio = event.positive_ratio
        ratio_score = min(positive_ratio / 0.8, 1.0)  # 80% 阳性算满分

        # 因素 2: 平均检测置信度
        avg_conf = max(event.avg_fire_confidence, event.avg_smoke_confidence)
        conf_score = min(avg_conf / 0.9, 1.0)  # 0.9 置信度算满分

        # 因素 3: 事件稳定性（连续阳性占比）
        stability = 0.5
        if event.total_frames > 0:
            consecutive = event.consecutive_positive_frames
            stability = min(consecutive / max(event.total_frames, 1), 1.0)
            stability = max(stability, 0.3)

        # 加权: 证据 50% + 置信度 30% + 稳定性 20%
        confidence = (
            ratio_score * 0.5
            + conf_score * 0.3
            + stability * 0.2
        )
        return max(0.0, min(confidence, 1.0))

    # ======================== 原因 / 建议 / 汇总 ========================

    def _build_reasons(self, event: FireEvent, features: dict, override_reasons: List[str]) -> List[str]:
        """构建结构化原因列表"""
        reasons: List[str] = []

        # 面积原因
        if event.max_fire_area_ratio > 0:
            reasons.append(f"火焰面积占比 {event.max_fire_area_ratio:.1%}")
        if event.max_smoke_area_ratio > 0:
            reasons.append(f"烟雾面积占比 {event.max_smoke_area_ratio:.1%}")

        # 时长原因
        if event.duration > 0:
            if event.duration >= self.duration_alert:
                reasons.append(f"持续 {event.duration:.0f} 秒（超过严重告警阈值）")
            elif event.duration >= self.duration_warning:
                reasons.append(f"持续 {event.duration:.0f} 秒（超过预警阈值）")
            else:
                reasons.append(f"持续 {event.duration:.0f} 秒")

        # 趋势原因
        trend_str = event.growth_trend.value if isinstance(event.growth_trend, GrowthTrend) else str(event.growth_trend)
        trend_desc = {
            "increasing": "火势呈扩大趋势",
            "stable": "火势保持稳定",
            "decreasing": "火势呈减弱趋势",
        }
        if trend_str in trend_desc:
            reasons.append(trend_desc[trend_str])

        # 覆盖规则原因
        reasons.extend(override_reasons)

        # 兜底
        if not reasons:
            reasons.append("未检测到明显风险因素")

        return reasons

    def _build_suggestions(self, danger_level: str, confidence: float, event: FireEvent) -> List[str]:
        """按等级生成分级建议"""
        suggestions: List[str] = []

        if danger_level == DangerLevel.HIGH:
            suggestions.append("立即启动消防应急预案")
            suggestions.append("通知值班负责人和安保人员")
            suggestions.append("确认现场人员安全，准备疏散")
            if event.growth_trend in (GrowthTrend.INCREASING, GrowthTrend.INCREASING.value):
                suggestions.append("火势正在扩大，建议立即联系119消防部门")
        elif danger_level == DangerLevel.MEDIUM:
            suggestions.append("安排人员前往现场确认情况")
            suggestions.append("持续监控火势变化")
            suggestions.append("准备应急设备")
            if event.max_fire_area_ratio == 0:
                suggestions.append("存在浓烟但未检测到明火，注意排查阴燃风险")
        else:
            suggestions.append("继续保持监控")
            suggestions.append("关注火势变化趋势")
            if confidence < 0.6:
                suggestions.append("当前证据较弱，建议人工复核检测结果")

        return suggestions

    def _build_summary(self, danger_level: str, reasons: List[str]) -> str:
        """生成一句话总结"""
        level_names = {DangerLevel.HIGH: "高风险", DangerLevel.MEDIUM: "中风险", DangerLevel.LOW: "低风险"}
        level_name = level_names.get(danger_level, danger_level)
        key_reason = reasons[0] if reasons else ""
        return f"判定为{level_name}：{key_reason}"

    # ======================== LLM 增强（预留） ========================

    def _try_llm_enhance(self, event: FireEvent, rule_decision: FireDecision) -> Optional[FireDecision]:
        """LLM 增强，失败返回 None 使用规则结果"""
        try:
            # TODO: 后续接入 DeepSeek / OpenAI API
            return None
        except Exception:
            return None

    # ======================== Debug 日志 ========================

    def _log_decision(self, event: FireEvent, decision: FireDecision, elapsed_ms: float):
        """记录决策详情"""
        dbg = decision.debug_info or {}
        features = dbg.get("features", {})
        feature_str = " | ".join(
            f"{k}={v:.3f}" for k, v in sorted(features.items())
        )
        override_mark = " [OVERRIDE]" if dbg.get("override_applied") else ""
        print(
            f"  [Agent] {event.event_id[:16]:16s} | "
            f"等级 {decision.danger_level:6s}{override_mark} | "
            f"评分 {decision.score:.3f} | "
            f"可信度 {decision.confidence:.3f} | "
            f"耗时 {elapsed_ms:.0f}ms"
        )
        print(f"          {feature_str}")


# ======================== 命令行测试 ========================

def main():
    """独立运行测试（含边界/业务/非法输入用例）"""
    print("\n  FireGuardian M7 - Fire Decision Agent v2 测试")
    print("  ===============================================\n")

    agent = FireDecisionAgent()

    def make_event(event_id, fire_area, smoke_area, dur, trend,
                   total=None, positive=None, avg_conf=0.85):
        """构造测试事件"""
        total = total or int(dur / 0.5) + 1
        positive = positive or total
        trend_val = trend.value if hasattr(trend, "value") else trend
        return FireEvent(
            event_id=event_id,
            status=EventStatus.ACTIVE,
            start_timestamp=0.0,
            last_timestamp=dur,
            total_frames=total,
            positive_frames=positive,
            consecutive_positive_frames=positive,
            consecutive_negative_frames=total - positive,
            max_fire_area_ratio=fire_area,
            max_smoke_area_ratio=smoke_area,
            avg_fire_confidence=avg_conf,
            avg_smoke_confidence=avg_conf,
            growth_trend=trend_val,
            source_type="test",
            source_name="unit_test",
        )

    test_cases = [
        # (name, fire, smoke, dur, trend, expected_level)
        ("用例1: 小面积短时",       0.01, 0.01, 2.0,  GrowthTrend.STABLE,     DangerLevel.LOW),
        ("用例2: 中等面积(边界)",   0.08, 0.05, 5.0,  GrowthTrend.STABLE,     DangerLevel.LOW),
        ("用例3: 大面积长时扩大",   0.15, 0.10, 12.0, GrowthTrend.INCREASING, DangerLevel.HIGH),
        ("用例4: 火势减弱",         0.10, 0.08, 8.0,  GrowthTrend.DECREASING, DangerLevel.MEDIUM),
        ("用例5: 浓烟无明火",       0.00, 0.20, 15.0, GrowthTrend.STABLE,     DangerLevel.MEDIUM),
        ("用例6: 小火持续久",       0.02, 0.01, 25.0, GrowthTrend.STABLE,     DangerLevel.MEDIUM),
        ("用例7: 双因素快速扩大",   0.18, 0.15, 12.0, GrowthTrend.INCREASING, DangerLevel.HIGH),
        ("用例8: 疑似误检(低置信)", 0.01, 0.00, 1.0,  GrowthTrend.UNKNOWN,    DangerLevel.LOW),
    ]

    all_pass = True
    for name, fa, sa, dur, trend, expected in test_cases:
        event = make_event(f"T-{name}", fa, sa, dur, trend)
        decision = agent.analyze(event)
        passed = decision.danger_level == expected.value if hasattr(expected, 'value') else decision.danger_level == expected
        status = "[OK]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        print(f"\n  ── {name} → 期望 {expected} {status} ──")
        print(f"  → 实际: {decision.danger_level} | 评分 {decision.score:.3f} | 可信度 {decision.confidence:.3f}")
        print(f"  → 原因: {decision.reasons[0] if decision.reasons else '无'}")

    # 边界测试
    print("\n  === 阈值边界测试 ===")
    boundary_cases = [
        # fire_area/0.2*0.4 + 0.5*0.1(growth stable) = score
        # score=0.2999 → fire_area=0.12495;  score=0.3001 → fire_area=0.12505
        ("边界A: score≈0.2999→low",  0.12495, 0.0, 0.0, GrowthTrend.STABLE, DangerLevel.LOW),
        ("边界B: score≈0.3001→med",  0.12505, 0.0, 0.0, GrowthTrend.STABLE, DangerLevel.MEDIUM),
        # 直接构造 0.60 边界: fire=0.15/0.2*0.4=0.30, smoke=0.15/0.3*0.3=0.15,
        # duration=30/30*0.2=0.20, growth=0.5*0.1=0.05 → 0.70, 但需要 0.60
        # fire=0.15/0.2*0.4=0.30, smoke=0.10/0.3*0.3=0.10, duration=15/30*0.2=0.10,
        # growth=0.5*0.1=0.05 → 0.55
        # fire=0.15/0.2*0.4=0.30, smoke=0.15/0.3*0.3=0.15, duration=15/30*0.2=0.10,
        # growth=0.5*0.1=0.05 → 0.60 ✓
        ("边界C: score=0.6000→high", 0.15, 0.15, 15.0, GrowthTrend.STABLE, DangerLevel.HIGH),
    ]
    for name, fa, sa, dur, trend, expected in boundary_cases:
        event = make_event(f"B-{name}", fa, sa, dur, trend)
        decision = agent.analyze(event)
        passed = decision.danger_level == expected.value if hasattr(expected, 'value') else decision.danger_level == expected
        status = "[OK]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        print(f"  {name} → {decision.danger_level} ({decision.score:.4f}) {status}")

    # 非法输入测试
    print("\n  === 非法输入测试 ===")
    invalid_cases = [
        ("缺少event_id", lambda: make_event("", 0.01, 0.01, 1.0, GrowthTrend.STABLE)),
        ("负面积",       lambda: make_event("neg", -0.1, 0.01, 1.0, GrowthTrend.STABLE)),
        ("面积>1",       lambda: make_event("big", 1.5, 0.01, 1.0, GrowthTrend.STABLE)),
        ("负时长",       lambda: make_event("dur", 0.01, 0.01, -1.0, GrowthTrend.STABLE)),
    ]
    for name, fn in invalid_cases:
        try:
            event = fn()
            agent.analyze(event)
            print(f"  {name} → [FAIL] 未抛出异常")
            all_pass = False
        except ValueError as e:
            print(f"  {name} → [OK] ValueError: {str(e)[:50]}")
        except Exception as e:
            print(f"  {name} → [OK] {type(e).__name__}: {str(e)[:50]}")

    # 配置读取验证
    print("\n  === 配置读取验证 ===")
    cfg = load_config()
    weights = cfg["agent"]["weights"]
    print(f"  config.yaml weights: {weights}")
    print(f"  agent 使用 weights: fire={agent.weight_fire_area} smoke={agent.weight_smoke_area} duration={agent.weight_duration} growth={agent.weight_growth}")
    print(f"  [OK] 权重从配置读取" if weights["fire_area"] == agent.weight_fire_area else "  [FAIL] 权重不一致")

    print(f"\n  {'=' * 45}")
    print(f"  {'全部测试通过 [OK]' if all_pass else '存在失败的测试用例 [FAIL]'}")
    print(f"  {'=' * 45}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
