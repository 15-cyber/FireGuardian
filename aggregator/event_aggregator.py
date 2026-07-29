"""
============================================================
M6 - Event Aggregator（事件聚合器）
功能：
  - 将逐帧 Detection 聚合为火灾事件（FireEvent）
  - 状态机: CANDIDATE → CONFIRMED → ACTIVE → ENDED / DISCARDED
  - 增长趋势计算
  - 回调通知（confirmed / updated / ended / discarded）
  - 通过 metadata["decision_required"] 控制 Agent 调用频率

输入: Detection（来自 M5 Simulator 或 M3/M4 Detector）
输出: FireEvent（通过回调传递给 M7 Agent 和其他模块）

可独立运行: python aggregator/event_aggregator.py
============================================================
"""
from __future__ import annotations

import os
import sys
import uuid
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, List, Optional

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    Detection,
    FireEvent,
    EventStatus,
    GrowthTrend,
)


# ======================== 回调类型 ========================

EventCallback = Callable[["FireEvent"], None]


# ======================== 聚合器类 ========================

class EventAggregator:
    """
    事件聚合器

    将逐帧的 Detection 检测结果按照可配置规则聚合成 FireEvent。

    使用方式:
        agg = EventAggregator()
        agg.on_event_confirmed = lambda e: print(f"事件确认: {e.event_id}")
        agg.on_event_ended = lambda e: print(f"事件结束: {e.event_id}")

        for detection in detection_stream:
            agg.process(detection)
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        cfg = load_config()
        agg_cfg = cfg.get("aggregator", {})

        # 配置参数
        self.min_positive_frames = agg_cfg.get("min_positive_frames", 3)
        self.confirmation_window = agg_cfg.get("confirmation_window", 5)
        self.max_negative_frames = agg_cfg.get("max_negative_frames", 3)
        self.fire_conf_threshold = agg_cfg.get("fire_confidence_threshold", 0.50)
        self.smoke_conf_threshold = agg_cfg.get("smoke_confidence_threshold", 0.50)
        self.min_fire_area_ratio = agg_cfg.get("min_fire_area_ratio", 0.002)
        self.min_smoke_area_ratio = agg_cfg.get("min_smoke_area_ratio", 0.003)
        self.growth_threshold = agg_cfg.get("growth_threshold", 0.01)

        # 当前事件
        self._current_event: Optional[FireEvent] = None
        self._frame_buffer: Deque[Detection] = deque(maxlen=self.confirmation_window)
        self._area_history: Deque[float] = deque(maxlen=10)  # 用于增长趋势计算
        self._total_frames_processed = 0

        # 事件历史
        self._event_history: List[FireEvent] = []

        # 回调
        self.on_event_confirmed: Optional[EventCallback] = None
        self.on_event_updated: Optional[EventCallback] = None
        self.on_event_ended: Optional[EventCallback] = None
        self.on_event_discarded: Optional[EventCallback] = None

    # ======================== 核心处理 ========================

    def process(self, detection: Detection) -> Optional[FireEvent]:
        """
        处理一帧检测结果

        参数:
            detection: 单帧检测结果

        返回:
            当前活跃的 FireEvent（如有），否则 None
        """
        self._total_frames_processed += 1
        self._frame_buffer.append(detection)

        # 判断当前帧是否包含有效目标
        is_positive = self._is_positive_frame(detection)

        if self._current_event is None:
            # 无事件 → 检测到目标 → 创建候选事件
            if is_positive:
                self._create_candidate(detection)
        else:
            # 有事件 → 更新状态
            self._update_event(detection, is_positive)

        return self._current_event

    def _is_positive_frame(self, detection: Detection) -> bool:
        """
        判断一帧是否包含有效目标（超过阈值的火焰/烟雾）
        """
        has_fire = (
            detection.has_fire
            and detection.avg_fire_confidence >= self.fire_conf_threshold
            and detection.fire_area_ratio >= self.min_fire_area_ratio
        )
        has_smoke = (
            detection.has_smoke
            and detection.avg_smoke_confidence >= self.smoke_conf_threshold
            and detection.smoke_area_ratio >= self.min_smoke_area_ratio
        )
        return has_fire or has_smoke

    def _create_candidate(self, detection: Detection):
        """创建候选事件"""
        self._current_event = FireEvent(
            event_id=self._generate_event_id(),
            status=EventStatus.CANDIDATE,
            start_timestamp=detection.timestamp,
            last_timestamp=detection.timestamp,
            total_frames=1,
            positive_frames=1,
            consecutive_positive_frames=1,
            consecutive_negative_frames=0,
            max_fire_area_ratio=detection.fire_area_ratio,
            max_smoke_area_ratio=detection.smoke_area_ratio,
            avg_fire_confidence=detection.avg_fire_confidence,
            avg_smoke_confidence=detection.avg_smoke_confidence,
            growth_trend=GrowthTrend.UNKNOWN,
            source_type=detection.image_path or "unknown",
            source_name=Path(detection.image_path).parent.name if detection.image_path else "",
            representative_frame_id=self._total_frames_processed,
            representative_image_path=detection.image_path,
            detections=[detection],
        )
        self._area_history.append(detection.fire_area_ratio + detection.smoke_area_ratio)
        self._check_confirmation()

    def _update_event(self, detection: Detection, is_positive: bool):
        """更新当前事件状态"""
        evt = self._current_event
        if evt is None:
            return

        evt.total_frames += 1
        evt.last_timestamp = detection.timestamp

        if is_positive:
            evt.positive_frames += 1
            evt.consecutive_positive_frames += 1
            evt.consecutive_negative_frames = 0

            # 更新面积和置信度
            evt.max_fire_area_ratio = max(evt.max_fire_area_ratio, detection.fire_area_ratio)
            evt.max_smoke_area_ratio = max(evt.max_smoke_area_ratio, detection.smoke_area_ratio)

            # 更新平均置信度（移动平均）
            n = evt.positive_frames
            evt.avg_fire_confidence = (evt.avg_fire_confidence * (n - 1) + detection.avg_fire_confidence) / n
            evt.avg_smoke_confidence = (evt.avg_smoke_confidence * (n - 1) + detection.avg_smoke_confidence) / n

            # 更新代表帧（选置信度最高的）
            total_conf = detection.avg_fire_confidence + detection.avg_smoke_confidence
            if total_conf > (evt.avg_fire_confidence + evt.avg_smoke_confidence):
                evt.representative_frame_id = self._total_frames_processed
                evt.representative_image_path = detection.image_path

            evt.detections.append(detection)
            self._area_history.append(detection.fire_area_ratio + detection.smoke_area_ratio)

            # 计算增长趋势
            evt.growth_trend = self._calc_growth_trend()

        else:
            evt.consecutive_positive_frames = 0
            evt.consecutive_negative_frames += 1

        # 检查状态流转
        if evt.status == EventStatus.CANDIDATE:
            self._check_confirmation()
        elif evt.status in (EventStatus.CONFIRMED, EventStatus.ACTIVE):
            if evt.consecutive_negative_frames >= self.max_negative_frames:
                self._end_event()
            else:
                # 更新活跃状态
                self._current_event.status = EventStatus.ACTIVE
                # 判断是否需要重新分析
                self._check_decision_required()
                if self.on_event_updated:
                    self.on_event_updated(evt)

    def _check_confirmation(self):
        """检查候选事件是否可以确认"""
        evt = self._current_event
        if evt is None or evt.status != EventStatus.CANDIDATE:
            return

        # 检查确认条件
        if evt.positive_frames >= self.min_positive_frames:
            evt.status = EventStatus.CONFIRMED
            evt.metadata["decision_required"] = True
            if self.on_event_confirmed:
                self.on_event_confirmed(evt)
            if self.on_event_updated:
                self.on_event_updated(evt)
        elif evt.total_frames >= self.confirmation_window:
            # 窗口内未达到确认条件 → 废弃
            self._discard_event()

    def _end_event(self):
        """结束事件"""
        evt = self._current_event
        if evt is None:
            return

        evt.status = EventStatus.ENDED
        evt.end_timestamp = evt.last_timestamp
        evt.metadata["decision_required"] = True

        # 保存到历史
        self._event_history.append(evt)

        if self.on_event_ended:
            self.on_event_ended(evt)

        self._current_event = None
        self._area_history.clear()

    def _discard_event(self):
        """废弃候选事件"""
        evt = self._current_event
        if evt is None:
            return

        evt.status = EventStatus.DISCARDED
        self._event_history.append(evt)

        if self.on_event_discarded:
            self.on_event_discarded(evt)

        self._current_event = None
        self._area_history.clear()

    def _calc_growth_trend(self) -> str:
        """基于面积历史计算增长趋势"""
        if len(self._area_history) < 2:
            return GrowthTrend.UNKNOWN

        earliest = self._area_history[0]
        latest = self._area_history[-1]
        delta = latest - earliest

        if delta > self.growth_threshold:
            return GrowthTrend.INCREASING
        elif delta < -self.growth_threshold:
            return GrowthTrend.DECREASING
        else:
            return GrowthTrend.STABLE

    def _check_decision_required(self):
        """判断是否需要 Agent 重新分析"""
        evt = self._current_event
        if evt is None:
            return

        # 触发条件：增长趋势变化、持续时间达到阈值
        old_trend = evt.metadata.get("last_trend", "")
        if old_trend and old_trend != evt.growth_trend:
            evt.metadata["decision_required"] = True
        evt.metadata["last_trend"] = evt.growth_trend

    def _generate_event_id(self) -> str:
        """生成唯一事件 ID"""
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"FE-{ts}-{suffix}"

    # ======================== 查询接口 ========================

    @property
    def current_event(self) -> Optional[FireEvent]:
        """获取当前活跃事件"""
        return self._current_event

    @property
    def event_history(self) -> List[FireEvent]:
        """获取已结束的事件历史"""
        return list(self._event_history)

    @property
    def total_processed(self) -> int:
        return self._total_frames_processed

    def reset(self):
        """重置聚合器状态"""
        self._current_event = None
        self._frame_buffer.clear()
        self._area_history.clear()
        self._event_history.clear()
        self._total_frames_processed = 0

    def get_stats(self) -> dict:
        """获取聚合器统计"""
        return {
            "total_frames_processed": self._total_frames_processed,
            "current_event": self._current_event.event_id if self._current_event else None,
            "current_event_status": self._current_event.status.value if self._current_event else None,
            "total_events_confirmed": sum(
                1 for e in self._event_history if e.status == EventStatus.CONFIRMED
            ),
            "total_events_ended": sum(
                1 for e in self._event_history if e.status == EventStatus.ENDED
            ),
            "total_events_discarded": sum(
                1 for e in self._event_history if e.status == EventStatus.DISCARDED
            ),
        }


# ======================== 命令行测试 ========================

def main():
    """
    独立运行测试：模拟一组检测数据，验证状态机流转
    """
    print("\n  FireGuardian M6 - Event Aggregator 测试")
    print("  ===============================================\n")

    agg = EventAggregator()

    # 注册回调
    def on_confirmed(e):
        print(f"  [回调] 事件确认: {e.event_id} | 阳性帧: {e.positive_frames}")

    def on_updated(e):
        print(f"  [回调] 事件更新: {e.status.value} | 趋势: {e.growth_trend}")

    def on_ended(e):
        print(f"  [回调] 事件结束: {e.event_id} | 持续 {e.duration:.1f}s")

    def on_discarded(e):
        print(f"  [回调] 事件废弃: {e.event_id}")

    agg.on_event_confirmed = on_confirmed
    agg.on_event_updated = on_updated
    agg.on_event_ended = on_ended
    agg.on_event_discarded = on_discarded

    # 模拟检测数据：10 帧包含火焰，3 帧空白，再 5 帧火焰
    from utils.common import BoundingBox

    print("  模拟场景: 10帧持续火焰 → 3帧空白 → 5帧火焰 → 结束\n")

    fire_box = BoundingBox(x1=100, y1=100, x2=300, y2=300, confidence=0.85,
                           class_id=0, class_name="fire")

    # 阶段 1: 10 帧火焰
    for i in range(10):
        det = Detection(
            bboxes=[fire_box],
            timestamp=float(i) * 0.5,
            image_width=640,
            image_height=480,
            image_path="sim_test",
        )
        result = agg.process(det)
        status = result.status.value if result else "none"
        print(f"  帧 {i+1:>2}: fire [OK]  | 事件: {status}")

    # 阶段 2: 3 帧无目标
    for i in range(10, 13):
        det = Detection(
            bboxes=[],
            timestamp=float(i) * 0.5,
            image_width=640,
            image_height=480,
            image_path="sim_test",
        )
        result = agg.process(det)
        status = result.status.value if result else "none"
        print(f"  帧 {i+1:>2}: 空白  | 事件: {status}")

    # 阶段 3: 5 帧火焰（新事件）
    for i in range(13, 18):
        det = Detection(
            bboxes=[fire_box],
            timestamp=float(i) * 0.5,
            image_width=640,
            image_height=480,
            image_path="sim_test",
        )
        result = agg.process(det)
        status = result.status.value if result else "none"
        print(f"  帧 {i+1:>2}: fire [OK]  | 事件: {status}")

    print(f"\n  统计: {agg.get_stats()}")
    print(f"\n  事件历史: {len(agg.event_history)} 个")
    for evt in agg.event_history:
        print(f"    {evt.event_id} | {evt.status.value} | {evt.duration:.1f}s")

    print("\n  M6 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
