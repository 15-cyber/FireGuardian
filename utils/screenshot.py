"""
============================================================
M8 - 自动截图模块 (Screenshot Manager)
功能：
  - 回调驱动：M6 on_event_confirmed 时自动保存截图
  - 保存原图 + 检测标注图 + event_info.txt
  - 文件名格式: 20260727153022.jpg（按事件时间戳）
  - 供 M10 报告生成使用（list[Path] 输出）

与其他模块的协作:
  M6 Aggregator ──on_event_confirmed──→ M8 保存截图
                                            │
                                            ▼
                                       screenshots/ (原图+标注图+info)
                                            │
                                            ▼
                                      M10 报告引用截图

可独立运行: python utils/screenshot.py
============================================================
"""
from __future__ import annotations

import os
import sys
import json
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
    FireEvent,
    Detection,
    BoundingBox,
    EventStatus,
    ensure_dir,
)


# ======================== 中文路径兼容 ========================

def _cv_read_image(path: str | Path) -> Optional[object]:
    """使用 np.fromfile + imdecode 读取图片（兼容中文路径）"""
    import cv2
    import numpy as np
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _cv_write_image(path: str | Path, img) -> bool:
    """使用 imencode + tofile 写入图片（兼容中文路径）"""
    import cv2
    suffix = Path(path).suffix or ".jpg"
    ext = f".{suffix.lstrip('.').lower()}"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


# ======================== 常量 ========================
# 类别 → 标注颜色 (BGR, OpenCV 格式)
_CLASS_COLORS = {
    "fire": (0, 0, 255),       # 红色
    "smoke": (0, 165, 255),    # 橙色
}


# ======================== 截图管理器 ========================

class ScreenshotManager:
    """
    自动截图管理器

    监听 M6 的事件确认回调，自动保存：
      - 原图 (original)
      - 检测标注图 (annotated)
      - 事件信息文件 (event_info.txt)

    使用方式:
        screenshot_mgr = ScreenshotManager()
        aggregator.on_event_confirmed = screenshot_mgr.on_event_confirmed
    """

    def __init__(self, output_dir: Optional[str | Path] = None):
        cfg = load_config()
        sc_cfg = cfg.get("screenshot", {})

        self.output_dir = ensure_dir(str(output_dir) if output_dir else str(_ROOT / sc_cfg.get("output_dir", "screenshots")))
        self.save_original = sc_cfg.get("save_original", True)
        self.save_annotated = sc_cfg.get("save_annotated", True)

        # 统计
        self.total_saved = 0
        self.saved_paths: List[Path] = []

    # ======================== 回调入口 ========================

    def on_event_confirmed(self, event: FireEvent) -> List[Path]:
        """
        事件确认回调（由 M6 触发）

        参数:
            event: 已确认的 FireEvent

        返回:
            保存的截图路径列表
        """
        if event.status not in (EventStatus.CONFIRMED, EventStatus.ACTIVE, EventStatus.ENDED):
            # 只有确认后的事件才截图
            return []

        return self.save_event_screenshots(event)

    # ======================== 核心保存逻辑 ========================

    def save_event_screenshots(self, event: FireEvent) -> List[Path]:
        """
        保存事件截图（原图 + 标注图 + info）

        参数:
            event: FireEvent 对象

        返回:
            保存的文件路径列表
        """
        # 找到代表帧的检测结果
        representative_detection = self._find_representative_detection(event)

        if representative_detection is None:
            print(f"  [Screenshot] 事件 {event.event_id} 无代表帧图片，跳过截图")
            return []

        # 生成时间戳文件名（与项目计划一致: YYYYMMDDHHMMSS）
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        event_ts = datetime.fromtimestamp(event.start_timestamp).strftime("%Y%m%d%H%M%S") \
            if event.start_timestamp > 0 else ts
        base_name = event_ts

        saved: List[Path] = []

        # 1. 保存原图
        original_path = self._save_original(event, representative_detection, base_name)
        if original_path:
            saved.append(original_path)

        # 2. 保存标注图
        annotated_path = self._save_annotated(event, representative_detection, base_name)
        if annotated_path:
            saved.append(annotated_path)

        # 3. 保存事件信息
        info_path = self._save_event_info(event, base_name)
        if info_path:
            saved.append(info_path)

        self.total_saved += len(saved)
        self.saved_paths.extend(saved)

        print(f"  [Screenshot] 事件 {event.event_id} 截图已保存 ({len(saved)} 个文件)")

        return saved

    # ======================== 内部方法 ========================

    def _find_representative_detection(self, event: FireEvent) -> Optional[Detection]:
        """
        查找事件的代表帧检测结果
        优先用代表帧，否则用最后一个有检测结果的帧
        """
        if not event.detections:
            return None

        # 优先返回有检测结果的最后一帧
        for detection in reversed(event.detections):
            if detection.bboxes and detection.image_path:
                return detection

        return None

    def _save_original(self, event: FireEvent, detection: Detection, base_name: str) -> Optional[Path]:
        """保存原始图片"""
        if not self.save_original:
            return None

        image_path = Path(detection.image_path)
        if not image_path.exists():
            # 尝试从 FrameData 中恢复
            return None

        import shutil
        suffix = image_path.suffix or ".jpg"
        target = self.output_dir / f"{base_name}_original{suffix}"
        shutil.copy2(str(image_path), str(target))
        return target

    def _save_annotated(self, event: FireEvent, detection: Detection, base_name: str) -> Optional[Path]:
        """绘制检测框并保存标注图"""
        if not self.save_annotated:
            return None

        import cv2

        image_path = Path(detection.image_path)
        if not image_path.exists():
            return None

        img = _cv_read_image(image_path)
        if img is None:
            return None

        # 绘制所有检测框
        for box in detection.bboxes:
            self._draw_box(img, box)

        # 添加事件信息水印
        cv2.putText(img, f"Event: {event.event_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Level: {event.metadata.get('danger_level', 'N/A')}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, f"Duration: {event.duration:.1f}s",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        target = self.output_dir / f"{base_name}_annotated.jpg"
        if _cv_write_image(target, img):
            return target
        return None

    def _draw_box(self, img, box: BoundingBox):
        """在图像上绘制单个检测框"""
        import cv2

        x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
        color = _CLASS_COLORS.get(box.class_name, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{box.class_name} {box.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _save_event_info(self, event: FireEvent, base_name: str) -> Optional[Path]:
        """保存事件信息文件"""
        info = {
            "event_id": event.event_id,
            "status": event.status.value if hasattr(event.status, 'value') else str(event.status),
            "start_timestamp": event.start_timestamp,
            "last_timestamp": event.last_timestamp,
            "end_timestamp": event.end_timestamp,
            "duration": round(event.duration, 2),
            "total_frames": event.total_frames,
            "positive_frames": event.positive_frames,
            "max_fire_area_ratio": round(event.max_fire_area_ratio, 4),
            "max_smoke_area_ratio": round(event.max_smoke_area_ratio, 4),
            "avg_fire_confidence": round(event.avg_fire_confidence, 4),
            "avg_smoke_confidence": round(event.avg_smoke_confidence, 4),
            "growth_trend": event.growth_trend.value if hasattr(event.growth_trend, 'value') else str(event.growth_trend),
            "source_type": event.source_type,
            "source_name": event.source_name,
            "danger_level": event.metadata.get("danger_level", "N/A"),
            "agent_decision": event.metadata.get("agent_decision", "N/A"),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        target = self.output_dir / f"{base_name}_event_info.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return target

    # ======================== 查询接口 ========================

    def get_saved_paths(self) -> List[Path]:
        """获取所有已保存的文件路径"""
        return list(self.saved_paths)

    def get_latest(self) -> Optional[Path]:
        """获取最近保存的文件"""
        if not self.saved_paths:
            return None
        return self.saved_paths[-1]

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_saved": self.total_saved,
            "output_dir": str(self.output_dir),
        }


# ======================== 命令行测试 ========================

def main():
    """独立运行测试：模拟一个确认事件，验证截图保存"""
    print("\n  FireGuardian M8 - 自动截图模块测试")
    print("  ===============================================\n")

    # 构造模拟事件
    from utils.common import BoundingBox, Detection, EventStatus, GrowthTrend

    # 使用项目中的 bus.jpg 作为测试图片
    test_img = _ROOT / "bus.jpg"
    if not test_img.exists():
        print("  未找到测试图片 bus.jpg")
        return 1

    box1 = BoundingBox(x1=420, y1=390, x2=800, y2=730, confidence=0.94,
                       class_id=5, class_name="bus")
    box2 = BoundingBox(x1=670, y1=395, x2=810, y2=880, confidence=0.88,
                       class_id=0, class_name="person")

    detection = Detection(
        bboxes=[box1, box2],
        image_path=str(test_img),
        timestamp=2.5,
        inference_time_ms=18.5,
        image_width=810,
        image_height=1080,
    )

    event = FireEvent(
        event_id="FE-TEST-0001",
        status=EventStatus.CONFIRMED,
        start_timestamp=0.0,
        last_timestamp=5.0,
        total_frames=10,
        positive_frames=8,
        consecutive_positive_frames=8,
        max_fire_area_ratio=0.12,
        max_smoke_area_ratio=0.05,
        avg_fire_confidence=0.85,
        avg_smoke_confidence=0.75,
        growth_trend=GrowthTrend.INCREASING,
        source_type="validation_simulator",
        source_name="valid",
        detections=[detection],
        metadata={"danger_level": "high"},
    )

    # 保存截图
    mgr = ScreenshotManager()
    saved = mgr.save_event_screenshots(event)

    print(f"\n  保存文件数: {len(saved)}")
    for p in saved:
        print(f"  {p.name} ({p.stat().st_size/1024:.0f} KB)")

    print(f"\n  M8 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
