"""
============================================================
M8 - 自动截图模块 (Screenshot Manager)
功能（按改进说明实现）：
  - 回调驱动：由 M6 事件回调触发，不参与检测与决策
  - 四个截图时机：事件首次确认 / 面积达到新峰值 / 危险等级升级 / 事件结束
  - 每个事件建立独立目录: screenshots/event_<event_id>/
  - 每张截图保存：原图 + 标注图 + 元数据 JSON
  - 路径统一由 PathManager 生成，不在模块中自行拼接
  - 写入失败不影响主检测流程

目录结构（改进说明 M8-3）:
  screenshots/
  └── event_FE-20260729-xxxxxx/
      ├── confirmed_raw.jpg / confirmed_annotated.jpg / confirmed_meta.json
      ├── peak_raw.jpg / peak_annotated.jpg / peak_meta.json
      ├── danger_level_upgraded_raw.jpg / ..._annotated.jpg / ..._meta.json
      ├── final_raw.jpg / final_annotated.jpg / final_meta.json
      └── event_info.json              # 事件汇总（供 M10 报告使用）

与其他模块的协作:
  M6 Aggregator ──on_event_confirmed/updated/ended──→ M8 保存截图
  M7 Agent ──on_decision──→ M8（可选，检测危险等级升级）
  M8 ──list[Path] / shots──→ M10 报告生成

可独立运行: python utils/screenshot.py
============================================================
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    FireEvent,
    Detection,
)
from utils.path_manager import PathManager


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

_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}


def _level_rank(level) -> int:
    """危险等级 → 数值（用于判断升级）"""
    if level is None:
        return -1
    return _LEVEL_RANK.get(str(level).strip().lower(), -1)


# ======================== 截图管理器 ========================

class ScreenshotManager:
    """
    自动截图管理器（M8）

    监听 M6 的事件回调，按四个时机自动保存事件证据：
      - confirmed:            事件首次确认
      - peak:                 事件面积达到新峰值（每个事件最多一张）
      - danger_level_upgraded:危险等级升级（low→medium→high 最多两张）
      - final:                事件结束的最后一帧

    使用方式:
        screenshot_mgr = ScreenshotManager()
        aggregator.on_event_confirmed = screenshot_mgr.on_event_confirmed
        aggregator.on_event_updated   = screenshot_mgr.on_event_updated
        aggregator.on_event_ended     = screenshot_mgr.on_event_ended
        # 或一行接线:
        screenshot_mgr.attach(aggregator)
    """

    def __init__(self, output_dir: Optional[str | Path] = None, config: Optional[dict] = None):
        cfg = config or load_config()
        sc_cfg = cfg.get("screenshot", {})

        self.path_mgr = PathManager(cfg)
        self.output_dir = Path(output_dir) if output_dir else self.path_mgr.screenshots_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_original = sc_cfg.get("save_original", True)
        self.save_annotated = sc_cfg.get("save_annotated", True)
        self.save_metadata = sc_cfg.get("save_metadata", True)

        triggers = sc_cfg.get("triggers", {})
        self.enabled_triggers = {
            "confirmed": triggers.get("confirmed", True),
            "peak": triggers.get("peak", True),
            "danger_level_upgraded": triggers.get("danger_upgrade", True),
            "final": triggers.get("final", True),
        }

        # 每个事件的状态: max_area / last_level / peak_saved
        self._event_state: dict = {}

        # 统计与查询
        self.total_saved = 0
        self.saved_paths: List[Path] = []
        self.shots: List[dict] = []  # 每张截图的记录（供 M10 报告使用）

    # ======================== 接线 ========================

    def attach(self, aggregator) -> "ScreenshotManager":
        """挂接到 M6 EventAggregator 的全部事件回调"""
        aggregator.on_event_confirmed = self.on_event_confirmed
        aggregator.on_event_updated = self.on_event_updated
        aggregator.on_event_ended = self.on_event_ended
        return self

    # ======================== 回调入口 ========================

    def on_event_confirmed(self, event: FireEvent) -> List[Path]:
        """事件首次确认 → 保存确认截图"""
        if not self.enabled_triggers["confirmed"]:
            return []
        state = self._ensure_state(event)
        state["last_level"] = event.metadata.get("danger_level", "")
        detection = self._pick_detection(event, first=True)
        if detection is not None:
            # 以确认帧为峰值基准，后续超过它才触发峰值截图
            state["max_area"] = detection.fire_area_ratio + detection.smoke_area_ratio
        return self._save_shot(event, detection, reason="confirmed")

    def on_event_updated(self, event: FireEvent) -> List[Path]:
        """事件更新 → 检查峰值与危险等级升级"""
        saved: List[Path] = []
        state = self._ensure_state(event)

        if self.enabled_triggers["peak"]:
            saved.extend(self._check_peak(event, state))

        if self.enabled_triggers["danger_level_upgraded"]:
            saved.extend(self._check_danger_upgrade(event, state))

        return saved

    def on_decision(self, event: FireEvent, decision=None,
                    danger_level: Optional[str] = None) -> List[Path]:
        """M7 决策完成后的挂钩（可选）：写入危险等级并检查升级截图"""
        level = danger_level
        if level is None and decision is not None:
            level = getattr(decision, "danger_level", None)
        if level is not None:
            event.metadata["danger_level"] = str(level)
            if hasattr(decision, "to_dict"):
                event.metadata["agent_decision"] = decision.to_dict()
        if not self.enabled_triggers["danger_level_upgraded"]:
            return []
        return self._check_danger_upgrade(event, self._ensure_state(event))

    def on_event_ended(self, event: FireEvent) -> List[Path]:
        """事件结束 → 保存最后一帧 + 汇总 event_info.json"""
        saved: List[Path] = []
        if self.enabled_triggers["final"]:
            detection = self._pick_detection(event, first=False)
            saved.extend(self._save_shot(event, detection, reason="final"))

        info_path = self._save_event_info(event)
        if info_path:
            saved.append(info_path)
            self.total_saved += 1
            self.saved_paths.append(info_path)

        self._event_state.pop(event.event_id, None)
        return saved

    # ======================== 触发判断 ========================

    def _ensure_state(self, event: FireEvent) -> dict:
        if event.event_id not in self._event_state:
            self._event_state[event.event_id] = {
                "max_area": 0.0,
                "last_level": event.metadata.get("danger_level", ""),
                "peak_saved": False,
            }
        return self._event_state[event.event_id]

    def _check_peak(self, event: FireEvent, state: dict) -> List[Path]:
        """面积达到新峰值时保存截图（每个事件最多一张，控制截图数量）"""
        if not event.detections:
            return []
        detection = event.detections[-1]
        total_area = detection.fire_area_ratio + detection.smoke_area_ratio
        if total_area > state["max_area"] + 1e-9:
            state["max_area"] = total_area
            if not state["peak_saved"]:
                state["peak_saved"] = True
                return self._save_shot(event, detection, reason="peak")
        return []

    def _check_danger_upgrade(self, event: FireEvent, state: dict) -> List[Path]:
        """危险等级升级（低→中→高）时保存截图"""
        level = event.metadata.get("danger_level")
        if level is None:
            return []
        current = _level_rank(level)
        last = _level_rank(state.get("last_level"))
        if current > last:
            state["last_level"] = str(level)
            detection = self._pick_detection(event, first=False)
            return self._save_shot(event, detection, reason="danger_level_upgraded")
        return []

    # ======================== 保存逻辑 ========================

    def _pick_detection(self, event: FireEvent, first: bool) -> Optional[Detection]:
        """选择事件内可用帧：first=True 取最早有图片的帧，否则取最后一帧"""
        if not event.detections:
            return None
        if first:
            for d in event.detections:
                if d.image_path:
                    return d
            return event.detections[0]
        for d in reversed(event.detections):
            if d.image_path:
                return d
        return event.detections[-1]

    def _event_dir(self, event: FireEvent) -> Path:
        """事件独立目录（由统一 PathManager 生成）"""
        return self.path_mgr.screenshot_event_dir(event.event_id)

    def _save_shot(self, event: FireEvent, detection: Optional[Detection],
                   reason: str) -> List[Path]:
        """保存一次截图（原图 + 标注图 + 元数据），任何失败都不影响主流程"""
        saved: List[Path] = []
        try:
            if detection is None or not detection.image_path:
                return []
            image_path = Path(detection.image_path)
            if not image_path.exists():
                return []

            event_dir = self._event_dir(event)
            event_dir.mkdir(parents=True, exist_ok=True)

            if self.save_original:
                p = self._save_original(event_dir, detection, reason)
                if p:
                    saved.append(p)

            if self.save_annotated:
                p = self._save_annotated(event_dir, event, detection, reason)
                if p:
                    saved.append(p)

            if self.save_metadata:
                p = self._save_meta(event_dir, event, detection, reason)
                if p:
                    saved.append(p)

            if saved:
                self.total_saved += len(saved)
                self.saved_paths.extend(saved)
                self.shots.append({
                    "event_id": event.event_id,
                    "reason": reason,
                    "frame_id": detection.frame_id,
                    "timestamp": round(detection.timestamp, 3),
                    "fire_area_ratio": round(detection.fire_area_ratio, 4),
                    "smoke_area_ratio": round(detection.smoke_area_ratio, 4),
                    "paths": [str(p) for p in saved],
                })
                print(f"  [Screenshot] {event.event_id} | 时机={reason} | 保存 {len(saved)} 个文件")
        except Exception as exc:
            print(f"  [Screenshot] 保存失败（不影响主检测流程）: {exc}")
        return saved

    def _save_original(self, event_dir: Path, detection: Detection, reason: str) -> Optional[Path]:
        """保存原图（证据保留）"""
        import shutil
        suffix = Path(detection.image_path).suffix or ".jpg"
        target = event_dir / f"{reason}_raw{suffix}"
        shutil.copy2(str(detection.image_path), str(target))
        return target

    def _save_annotated(self, event_dir: Path, event: FireEvent,
                        detection: Detection, reason: str) -> Optional[Path]:
        """绘制检测框并保存标注图（报告展示）"""
        img = _cv_read_image(detection.image_path)
        if img is None:
            return None

        import cv2
        for box in detection.bboxes:
            self._draw_box(img, box)

        # 添加事件信息水印（仅 ASCII，避免中文渲染问题）
        cv2.putText(img, f"Event: {event.event_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Reason: {reason}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        level = event.metadata.get("danger_level", "N/A")
        cv2.putText(img, f"Level: {level}", (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        target = event_dir / f"{reason}_annotated.jpg"
        return target if _cv_write_image(target, img) else None

    def _draw_box(self, img, box):
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

    def _save_meta(self, event_dir: Path, event: FireEvent,
                   detection: Detection, reason: str) -> Optional[Path]:
        """保存截图元数据（改进说明 M8-4 字段）"""
        meta = {
            "event_id": event.event_id,
            "frame_id": detection.frame_id,
            "timestamp": round(detection.timestamp, 3),
            "reason": reason,
            "fire_area_ratio": round(detection.fire_area_ratio, 4),
            "smoke_area_ratio": round(detection.smoke_area_ratio, 4),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "raw_file": f"{reason}_raw{Path(detection.image_path).suffix or '.jpg'}",
            "annotated_file": f"{reason}_annotated.jpg",
        }
        target = event_dir / f"{reason}_meta.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return target

    def _save_event_info(self, event: FireEvent) -> Optional[Path]:
        """事件结束时汇总事件信息 + 截图清单，供 M10 报告使用"""
        try:
            info = {
                "event_id": event.event_id,
                "status": event.status.value if hasattr(event.status, "value") else str(event.status),
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
                "growth_trend": event.growth_trend.value if hasattr(event.growth_trend, "value") else str(event.growth_trend),
                "source_type": event.source_type,
                "source_name": event.source_name,
                "danger_level": event.metadata.get("danger_level", "N/A"),
                "agent_decision": event.metadata.get("agent_decision", "N/A"),
                "screenshots": [s for s in self.shots if s["event_id"] == event.event_id],
            }
            target = self._event_dir(event) / "event_info.json"
            with open(target, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            return target
        except Exception as exc:
            print(f"  [Screenshot] event_info 保存失败（不影响主流程）: {exc}")
            return None

    # ======================== 查询接口 ========================

    def get_saved_paths(self) -> List[Path]:
        """获取所有已保存的文件路径（M8 → M10 契约）"""
        return list(self.saved_paths)

    def get_latest(self) -> Optional[Path]:
        """获取最近保存的文件"""
        return self.saved_paths[-1] if self.saved_paths else None

    def get_event_shot_records(self, event_id: str) -> List[dict]:
        """获取指定事件的截图记录（供 M10 报告引用）"""
        return [s for s in self.shots if s["event_id"] == event_id]

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_saved": self.total_saved,
            "output_dir": str(self.output_dir),
            "active_events": len(self._event_state),
        }


# ======================== 命令行测试 ========================

def main():
    """独立运行测试：端到端模拟 M6 事件流，验证四个截图时机"""
    print("\n  FireGuardian M8 - 自动截图模块测试")
    print("  ===============================================\n")

    from utils.common import BoundingBox, EventStatus, GrowthTrend, FireEvent

    test_img = _ROOT / "bus.jpg"
    if not test_img.exists():
        print("  未找到测试图片 bus.jpg")
        return 1

    img_w, img_h = 810, 1080

    def make_detection(area_ratio, ts, frame_id, conf=0.85):
        """按面积占比构造火焰检测框"""
        side = int((area_ratio * img_w * img_h) ** 0.5)
        box = BoundingBox(x1=100, y1=100, x2=100 + side, y2=100 + side,
                          confidence=conf, class_id=0, class_name="fire")
        return Detection(bboxes=[box], image_path=str(test_img), timestamp=ts,
                         image_width=img_w, image_height=img_h, frame_id=frame_id)

    # ---------- 场景 1: 完整事件流（confirmed → peak → upgrade → final） ----------
    print("  场景 1: 完整事件流\n")
    mgr = ScreenshotManager()
    event = FireEvent(
        event_id="FE-TEST-0001",
        status=EventStatus.CONFIRMED,
        start_timestamp=0.0,
        last_timestamp=6.0,
        total_frames=8,
        positive_frames=6,
        consecutive_positive_frames=6,
        max_fire_area_ratio=0.30,
        avg_fire_confidence=0.87,
        growth_trend=GrowthTrend.INCREASING,
        source_type="test",
        source_name="bus",
        detections=[
            make_detection(0.05, 1.0, 1),
            make_detection(0.10, 2.0, 2),
            make_detection(0.30, 3.0, 3),
        ],
        metadata={"danger_level": "low"},
    )

    n1 = len(mgr.on_event_confirmed(event))          # confirmed
    n2 = len(mgr.on_event_updated(event))            # peak (0.30 > 0.05)
    n3 = len(mgr.on_decision(event, danger_level="medium"))  # upgrade
    n4 = len(mgr.on_decision(event, danger_level="medium"))  # 不重复
    event.status = EventStatus.ENDED
    n5 = len(mgr.on_event_ended(event))              # final + event_info

    print(f"    confirmed={n1} peak={n2} upgrade={n3} dup_upgrade={n4} final+info={n5}")

    # 校验目录与文件
    event_dir = mgr.path_mgr.screenshot_event_dir("FE-TEST-0001")
    files = sorted(p.name for p in event_dir.iterdir()) if event_dir.exists() else []
    expect = [
        "confirmed_raw.jpg", "confirmed_annotated.jpg", "confirmed_meta.json",
        "peak_raw.jpg", "peak_annotated.jpg", "peak_meta.json",
        "danger_level_upgraded_raw.jpg",
        "danger_level_upgraded_annotated.jpg",
        "danger_level_upgraded_meta.json",
        "final_raw.jpg", "final_annotated.jpg", "final_meta.json",
        "event_info.json",
    ]
    missing = [e for e in expect if e not in files]
    assert not missing, f"缺少文件: {missing}"
    print(f"    事件目录: {event_dir.name}")
    print(f"    文件数: {len(files)} | 预期 13 个全在 [OK]")

    # 元数据字段校验
    import json as _json
    meta = _json.loads((event_dir / "confirmed_meta.json").read_text(encoding="utf-8"))
    for key in ("event_id", "frame_id", "timestamp", "reason",
                "fire_area_ratio", "smoke_area_ratio"):
        assert key in meta, f"元数据缺少字段: {key}"
    assert meta["reason"] == "confirmed"
    assert meta["frame_id"] == 1
    print(f"    元数据字段完整 [OK] | reason={meta['reason']} frame_id={meta['frame_id']}")

    # 文件名无中文
    non_ascii = [f for f in files if any(ord(ch) > 127 for ch in f)]
    assert not non_ascii, f"文件名包含非 ASCII 字符: {non_ascii}"
    print("    文件名全部为 ASCII [OK]")

    # event_info 汇总
    info = _json.loads((event_dir / "event_info.json").read_text(encoding="utf-8"))
    assert info["event_id"] == "FE-TEST-0001"
    assert len(info["screenshots"]) == 4  # confirmed + peak + upgrade + final
    print(f"    event_info 汇总 [OK] | 截图记录 {len(info['screenshots'])} 条")

    # ---------- 场景 2: 失败隔离（图片不存在时不崩溃） ----------
    print("\n  场景 2: 写入失败不影响主流程")
    bad_event = FireEvent(
        event_id="FE-TEST-0002",
        status=EventStatus.CONFIRMED,
        start_timestamp=0.0,
        last_timestamp=1.0,
        total_frames=1,
        positive_frames=1,
        max_fire_area_ratio=0.05,
        source_type="test",
        source_name="bad",
        detections=[Detection(bboxes=[], image_path="nonexistent_frame.jpg",
                              timestamp=0.5, image_width=810, image_height=1080)],
    )
    n_bad = len(mgr.on_event_confirmed(bad_event))
    assert n_bad == 0
    print(f"    异常帧跳过且不抛错 [OK] (保存数={n_bad})")

    # ---------- 场景 3: 与 M6 聚合器端到端联动 ----------
    print("\n  场景 3: 挂接 M6 EventAggregator 端到端联动")
    from aggregator.event_aggregator import EventAggregator

    agg = EventAggregator()
    mgr2 = ScreenshotManager()
    mgr2.attach(agg)

    # 3 帧确认 + 2 帧增长 + 3 帧空白结束
    for i in range(1, 6):
        det = make_detection(0.05 + i * 0.03, float(i) * 0.5, i)
        agg.process(det)
    for i in range(6, 9):
        agg.process(Detection(bboxes=[], image_path=str(test_img), timestamp=float(i) * 0.5,
                              image_width=img_w, image_height=img_h, frame_id=i))
    ended = agg.event_history
    assert len(ended) == 1, f"预期 1 个事件结束，实际 {len(ended)}"
    evt = ended[0]
    assert evt.status == EventStatus.ENDED
    evt_dir = mgr2.path_mgr.screenshot_event_dir(evt.event_id)
    e_files = sorted(p.name for p in evt_dir.iterdir()) if evt_dir.exists() else []
    e_expect = ["confirmed_raw.jpg", "final_raw.jpg", "event_info.json"]
    e_missing = [e for e in e_expect if e not in e_files]
    assert not e_missing, f"联动缺少文件: {e_missing}"
    print(f"    事件 {evt.event_id} 自动完成 confirmed/peak/final 截图 [OK]")
    print(f"    目录: {evt_dir.name} | 文件: {len(e_files)}")

    print("\n  M8 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())