# -*- coding: utf-8 -*-
"""
M6 面积并集修复 / M7 时间单位（秒）统一修复的回归测试。

运行方式（项目根目录）:
    python -m unittest tests.test_m6_m7_fix -v
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    BoundingBox,
    Detection,
    FireEvent,
    EventStatus,
    union_area_of_boxes,
)
from aggregator.event_aggregator import EventAggregator
from agent.fire_decision_agent import FireDecisionAgent


class TestUnionArea(unittest.TestCase):
    """任务1：M6 面积比例改为同类别框 union 计算"""

    @staticmethod
    def _box(x1, y1, x2, y2, cls="fire", conf=0.9):
        return BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf,
            class_id=0 if cls == "fire" else 1, class_name=cls,
        )

    def test_case1_single_box(self):
        b = self._box(0, 0, 100, 100)
        self.assertAlmostEqual(union_area_of_boxes([b]), 10000.0)

    def test_case2_two_non_overlapping_boxes(self):
        b1 = self._box(0, 0, 100, 100)
        b2 = self._box(200, 200, 300, 300)
        self.assertAlmostEqual(union_area_of_boxes([b1, b2]), 20000.0)

    def test_case3_two_fully_overlapping_boxes(self):
        b1 = self._box(0, 0, 100, 100)
        b2 = self._box(0, 0, 100, 100)
        self.assertAlmostEqual(union_area_of_boxes([b1, b2]), 10000.0)

    def test_case4_big_box_contains_small_no_double_count(self):
        big = self._box(0, 0, 200, 200)
        small = self._box(50, 50, 100, 100)
        self.assertAlmostEqual(union_area_of_boxes([big, small]), 40000.0)
        # 关键验证：case4 面积不能重复增加（并集 == 大框面积）
        self.assertAlmostEqual(
            union_area_of_boxes([big, small]), union_area_of_boxes([big])
        )

    def test_partial_overlap(self):
        b1 = self._box(0, 0, 100, 100)
        b2 = self._box(50, 0, 150, 100)
        self.assertAlmostEqual(union_area_of_boxes([b1, b2]), 15000.0)

    def test_detection_union_and_mixed_classes(self):
        big = self._box(0, 0, 100, 100)
        small = self._box(10, 10, 40, 40)  # 完全位于 big 内
        det = Detection(bboxes=[big, small], image_width=100, image_height=100)
        self.assertAlmostEqual(det.fire_area, 10000.0)
        self.assertAlmostEqual(det.fire_area_ratio, 1.0)
        # 混合类别：smoke 框不参与 fire 面积
        smoke = self._box(0, 0, 50, 50, cls="smoke")
        det2 = Detection(bboxes=[big, small, smoke], image_width=100, image_height=100)
        self.assertAlmostEqual(det2.fire_area, 10000.0)
        self.assertAlmostEqual(det2.smoke_area, 2500.0)

    def test_ratio_always_in_01(self):
        cases = [
            ([self._box(0, 0, 50, 50)], 100, 100),
            ([self._box(0, 0, 100, 100), self._box(0, 0, 100, 100)], 100, 100),
            ([self._box(0, 0, 200, 200)], 100, 100),  # 超出图像尺寸也截断到 1.0
        ]
        for boxes, w, h in cases:
            det = Detection(bboxes=boxes, image_width=w, image_height=h)
            for r in (det.fire_area_ratio, det.smoke_area_ratio):
                self.assertTrue(0.0 <= r <= 1.0, f"ratio 越界: {r}")


class TestTimeUnits(unittest.TestCase):
    """任务2：M6/M7 时间单位统一为秒"""

    @staticmethod
    def _fire_det(frame_id, timestamp=None, fps=None):
        b = BoundingBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9,
                        class_id=0, class_name="fire")
        return Detection(bboxes=[b], frame_id=frame_id, timestamp=timestamp,
                         fps=fps, image_width=100, image_height=100)

    def test_frame_id_fps_conversion_in_aggregator(self):
        agg = EventAggregator()
        fps = 20.4
        for i in range(1, 6):  # 5 个阳性帧
            agg.process(self._fire_det(frame_id=i, timestamp=float(i), fps=fps))
        for i in range(6, 9):  # 3 个阴性帧结束事件
            neg = Detection(bboxes=[], frame_id=i, timestamp=float(i), fps=fps,
                            image_width=100, image_height=100)
            agg.process(neg)
        ended = [e for e in agg.event_history if e.status == EventStatus.ENDED]
        self.assertEqual(len(ended), 1)
        evt = ended[0]
        # 秒制：frame 1 → frame 8，时长 = 7 / 20.4
        self.assertAlmostEqual(evt.start_time_seconds, 1.0 / fps, places=5)
        self.assertAlmostEqual(evt.end_time_seconds, 8.0 / fps, places=5)
        self.assertAlmostEqual(evt.duration_seconds, 7.0 / fps, places=5)
        self.assertAlmostEqual(evt.duration, 7.0 / fps, places=5)

    def test_timestamp_fallback_without_fps(self):
        agg = EventAggregator()
        for i in range(5):
            agg.process(self._fire_det(frame_id=-1, timestamp=float(i) * 0.5))
        for i in range(5, 8):
            neg = Detection(bboxes=[], frame_id=-1, timestamp=float(i) * 0.5,
                            image_width=100, image_height=100)
            agg.process(neg)
        ended = [e for e in agg.event_history if e.status == EventStatus.ENDED]
        self.assertEqual(len(ended), 1)
        evt = ended[0]
        self.assertAlmostEqual(evt.duration_seconds, 3.5, places=5)
        self.assertAlmostEqual(evt.duration, 3.5, places=5)

    def test_fire_event_duration_prefers_seconds(self):
        evt = FireEvent(event_id="E1", status=EventStatus.ACTIVE,
                        start_timestamp=0.0, last_timestamp=100.0,
                        start_time_seconds=1.0, last_time_seconds=11.0,
                        duration_seconds=10.0)
        self.assertAlmostEqual(evt.duration, 10.0)

    def test_to_dict_contains_new_fields(self):
        evt = FireEvent(event_id="E2", fps=20.4, start_time_seconds=1.0,
                        end_time_seconds=3.0, duration_seconds=2.0)
        d = evt.to_dict()
        for k in ("fps", "start_time_seconds", "end_time_seconds",
                  "duration_seconds"):
            self.assertIn(k, d)

    def test_m7_uses_seconds_based_duration(self):
        agent = FireDecisionAgent()

        def make(dur_sec):
            return FireEvent(
                event_id=f"T-{dur_sec}", status=EventStatus.ACTIVE,
                start_timestamp=0.0, last_timestamp=25.0,  # 旧口径（帧）会误导
                start_time_seconds=0.0, last_time_seconds=float(dur_sec),
                duration_seconds=float(dur_sec),
                total_frames=10, positive_frames=10,
                consecutive_positive_frames=10,
                max_fire_area_ratio=0.01, max_smoke_area_ratio=0.01,
                avg_fire_confidence=0.85, avg_smoke_confidence=0.85,
                growth_trend="stable",
            )

        low_dec = agent.analyze(make(2.0))    # 真实 2 秒 → low
        med_dec = agent.analyze(make(12.0))   # 真实 12 秒 ≥ duration_alert → medium
        self.assertEqual(low_dec.danger_level, "low")
        self.assertEqual(med_dec.danger_level, "medium")


if __name__ == "__main__":
    unittest.main()
