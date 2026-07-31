"""
============================================================
M8 - 自动截图模块 (Screenshot Manager)
功能（按核查文档要求实现）：
  - 回调驱动：由 M6/M7 事件回调触发，不参与检测与决策
  - 四个截图时机：confirmed / peak / danger_level_upgraded / final
  - 每个事件建立独立目录: screenshots/event_<event_id>/
  - 每张截图保存：原图 + 标注图 + 元数据 JSON（原子写入）
  - peak 随更高峰值替换，始终对应最终最高峰值
  - 危险等级升级按转换去重（upgrade_low_to_medium / upgrade_medium_to_high）
  - final 默认使用最后阳性帧
  - 截图记录使用统一 ScreenshotRecord 对象（供 M10 报告）
  - 路径统一由 PathManager 生成
  - 写入失败记录 write_status，不影响主检测流程

目录结构:
  screenshots/event_FE-xxx/
    confirmed_raw.jpg / confirmed_annotated.jpg / confirmed_meta.json
    peak_raw.jpg / peak_annotated.jpg / peak_meta.json          # 更高峰值时替换
    upgrade_low_to_medium_raw.jpg / ..._annotated.jpg / ..._meta.json
    upgrade_medium_to_high_raw.jpg / ..._annotated.jpg / ..._meta.json
    final_raw.jpg / final_annotated.jpg / final_meta.json
    event_info.json

与其他模块的协作:
  M6 Aggregator ──on_event_confirmed/updated/ended──→ M8 保存截图
  M7 Agent ──on_decision──→ M8（写入危险等级并检查升级）
  M8 ──list[Path] / ScreenshotRecord──→ M10 报告生成

可独立运行: python utils/screenshot.py
============================================================
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    FireEvent,
    Detection,
    ScreenshotRecord,
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

_PEAK_METRICS = ("fire_area", "smoke_area", "combined_area")


def _level_rank(level) -> int:
    """危险等级 → 数值（用于判断升级）"""
    if level is None:
        return -1
    return _LEVEL_RANK.get(str(level).strip().lower(), -1)


# ======================== 事件截图状态 ========================

@dataclass
class EventScreenshotState:
    """M8 内部维护的“截图管理状态”（按事件隔离，事件结束后清理）"""
    event_id: str
    peak_metric_value: float = 0.0          # 当前历史峰值（按 peak_metric）
    peak_record_index: Optional[int] = None  # peak 记录在 self.shots 中的下标
    last_positive_frame: Optional[Detection] = None  # 最后阳性帧（final 用）
    last_danger_level: str = ""             # 上一次危险等级
    confirmed_saved: bool = False           # confirmed 只保存一次
    peak_saved: bool = False                # peak 首次保存标记（之后改为替换）
    saved_upgrade_transitions: set = field(default_factory=set)  # 已保存的升级转换


# ======================== 截图管理器 ========================

class ScreenshotManager:
    """
    自动截图管理器（M8）

    监听 M6/M7 的事件回调，按四个时机自动保存事件证据：
      - confirmed:             事件首次确认（每事件最多一组）
      - peak:                  峰值截图（更高峰值时替换，始终对应最终最高峰值）
      - danger_level_upgraded: 危险等级升级（按 low_to_medium 等转换去重，互不覆盖）
      - final:                 事件结束，默认使用最后阳性帧

    使用方式:
        screenshot_mgr = ScreenshotManager()
        screenshot_mgr.attach(aggregator)            # 挂接 M6 回调
        screenshot_mgr.on_decision(event, decision)  # M7 决策后可调（可选）
    """

    def __init__(self, output_dir: Optional[str | Path] = None, config: Optional[dict] = None):
        cfg = config or load_config()
        sc_cfg = cfg.get("screenshot", {})

        self.enabled = sc_cfg.get("enabled", True)
        self.path_mgr = PathManager(cfg)
        self.output_dir = Path(output_dir) if output_dir else self.path_mgr.screenshots_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 峰值指标配置化（核查文档情况 1）
        self.peak_metric = str(sc_cfg.get("peak_metric", "combined_area")).strip().lower()
        if self.peak_metric not in _PEAK_METRICS:
            print(f"  [Screenshot] 未知 peak_metric={self.peak_metric}，回退 combined_area")
            self.peak_metric = "combined_area"

        self.save_original = sc_cfg.get("save_original", True)
        self.save_annotated = sc_cfg.get("save_annotated", True)
        self.save_metadata = sc_cfg.get("save_metadata", True)
        self.write_event_info = sc_cfg.get("write_event_info", True)
        self.use_last_positive_frame_for_final = sc_cfg.get(
            "use_last_positive_frame_for_final", True)

        triggers = sc_cfg.get("triggers", {})
        self.enabled_triggers = {
            "confirmed": triggers.get("confirmed", True),
            "peak": triggers.get("peak", True),
            "danger_level_upgraded": triggers.get("danger_upgrade", True),
            "final": triggers.get("final", True),
        }

        # 每个事件独立维护截图状态（核查文档情况 9）
        self._event_state: Dict[str, EventScreenshotState] = {}

        # 统计与查询
        self.total_saved = 0
        self.saved_paths: List[Path] = []
        self.shots: List[ScreenshotRecord] = []  # 统一结构化记录（核查文档情况 5）

    # ======================== 接线 ========================

    def attach(self, aggregator) -> "ScreenshotManager":
        """挂接到 M6 EventAggregator 的全部事件回调"""
        aggregator.on_event_confirmed = self.on_event_confirmed
        aggregator.on_event_updated = self.on_event_updated
        aggregator.on_event_ended = self.on_event_ended
        return self

    # ======================== 回调入口 ========================

    def on_event_confirmed(self, event: FireEvent) -> List[Path]:
        """事件首次确认 → 保存确认截图（每事件最多一组）"""
        if not self.enabled or not self.enabled_triggers["confirmed"]:
            return []
        state = self._ensure_state(event)
        if state.confirmed_saved:
            return []

        state.last_danger_level = str(event.metadata.get("danger_level", ""))
        detection = self._pick_detection(event, first=True)
        if detection is not None:
            state.last_positive_frame = detection
            state.peak_metric_value = self._peak_value(detection)

        files, _ = self._save_shot(event, detection, reason="confirmed")
        if files:
            state.confirmed_saved = True
        return files

    def on_event_updated(self, event: FireEvent) -> List[Path]:
        """事件更新 → 更新最后阳性帧，检查峰值替换与危险等级升级"""
        if not self.enabled:
            return []
        saved: List[Path] = []
        state = self._ensure_state(event)

        # 更新最后阳性帧（只认含检测框的帧）
        if event.detections:
            last = event.detections[-1]
            if last.bboxes and last.image_path:
                state.last_positive_frame = last

        if self.enabled_triggers["peak"]:
            saved.extend(self._check_peak(event, state))
        if self.enabled_triggers["danger_level_upgraded"]:
            saved.extend(self._check_danger_upgrade(event, state))
        return saved

    def on_decision(self, event: FireEvent, decision=None,
                    danger_level: Optional[str] = None) -> List[Path]:
        """M7 决策完成后的挂钩（可选）：写入危险等级并检查升级截图"""
        if not self.enabled:
            return []
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
        """事件结束 → 保存 final（最后阳性帧）+ 汇总 event_info.json + 清理缓存"""
        if not self.enabled:
            return []
        saved: List[Path] = []
        state = self._ensure_state(event)

        if self.enabled_triggers["final"]:
            detection = None
            if self.use_last_positive_frame_for_final and state.last_positive_frame is not None:
                detection = state.last_positive_frame
            else:
                detection = self._pick_detection(event, first=False)
            saved.extend(self._save_shot(event, detection, reason="final")[0])

        if self.write_event_info:
            info_path = self._save_event_info(event)
            if info_path:
                saved.append(info_path)
                self.total_saved += 1
                self.saved_paths.append(info_path)

        self.cleanup_event_cache(event.event_id)
        return saved
    # ======================== 触发判断 ========================

    def _ensure_state(self, event: FireEvent) -> EventScreenshotState:
        """按 event_id 获取（或创建）事件截图状态（核查文档情况 9）"""
        if event.event_id not in self._event_state:
            self._event_state[event.event_id] = EventScreenshotState(
                event_id=event.event_id,
                last_danger_level=str(event.metadata.get("danger_level", "")),
            )
        return self._event_state[event.event_id]

    def cleanup_event_cache(self, event_id: str):
        """事件结束后清理内存缓存（核查文档情况 8）"""
        self._event_state.pop(event_id, None)

    def _peak_value(self, detection: Detection) -> float:
        """按配置的 peak_metric 计算峰值指标（核查文档情况 1）"""
        if self.peak_metric == "fire_area":
            return detection.fire_area_ratio
        if self.peak_metric == "smoke_area":
            return detection.smoke_area_ratio
        return detection.fire_area_ratio + detection.smoke_area_ratio

    def _check_peak(self, event: FireEvent, state: EventScreenshotState) -> List[Path]:
        """
        峰值判断：超过历史峰值时保存；已保存过则替换旧图与记录，
        保证 peak 始终对应最终最高峰值（核查文档情况 2）。
        """
        detection = state.last_positive_frame or self._pick_detection(event, first=False)
        if detection is None:
            return []
        value = self._peak_value(detection)
        if value <= state.peak_metric_value + 1e-9:
            return []

        state.peak_metric_value = value
        if state.peak_saved:
            # 更高峰值 → 替换同一组 peak 文件并更新记录
            files, _ = self._save_shot(
                event, detection, reason="peak",
                replace_record_index=state.peak_record_index,
            )
            print(f"  [Screenshot] {event.event_id} | peak 更新为更高峰值 ({value:.4f})")
            return files

        files, idx = self._save_shot(event, detection, reason="peak")
        if idx is not None:
            state.peak_saved = True
            state.peak_record_index = idx
        return files

    def _check_danger_upgrade(self, event: FireEvent,
                              state: EventScreenshotState) -> List[Path]:
        """
        危险等级升级：仅当等级提升且该转换未保存过时截图，
        文件名按转换命名（upgrade_low_to_medium），互不覆盖（核查文档情况 3）。
        """
        level = event.metadata.get("danger_level")
        if level is None:
            return []
        current = _level_rank(level)
        last = _level_rank(state.last_danger_level)
        if current <= last:
            return []

        old = (state.last_danger_level or "unknown").strip().lower()
        new = str(level).strip().lower()
        transition = f"{old}_to_{new}"
        if transition in state.saved_upgrade_transitions:
            return []

        state.saved_upgrade_transitions.add(transition)
        state.last_danger_level = new
        detection = state.last_positive_frame or self._pick_detection(event, first=False)

        extra_meta = {
            "previous_danger_level": old,
            "current_danger_level": new,
            "transition": transition,
        }
        decision = event.metadata.get("agent_decision")
        if isinstance(decision, dict):
            extra_meta["decision_score"] = decision.get("score")
            extra_meta["decision_confidence"] = decision.get("confidence")
            extra_meta["decision_source"] = decision.get("decision_source")

        files, _ = self._save_shot(
            event, detection, reason="danger_level_upgraded",
            file_stem=f"upgrade_{transition}", extra_meta=extra_meta,
        )
        return files

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
                   reason: str, file_stem: Optional[str] = None,
                   extra_meta: Optional[dict] = None,
                   replace_record_index: Optional[int] = None) -> Tuple[List[Path], Optional[int]]:
        """
        保存一次截图（原图 + 标注图 + 元数据）。
        返回 (保存的文件列表, 新记录下标或 None)；replace 时更新已有记录。
        任何失败都不影响主流程，write_status 记录各文件写入结果。
        """
        saved_files: List[Path] = []
        write_status: dict = {}
        raw_path = ann_path = meta_path = None
        try:
            if detection is None or not detection.image_path:
                return [], None
            image_path = Path(detection.image_path)
            if not image_path.exists():
                return [], None

            event_dir = self._event_dir(event)
            event_dir.mkdir(parents=True, exist_ok=True)
            stem = file_stem or reason

            if self.save_original:
                raw_path = self._save_original(event_dir, detection, stem)
                write_status["raw"] = raw_path is not None
            if self.save_annotated:
                ann_path = self._save_annotated(event_dir, event, detection, stem)
                write_status["annotated"] = ann_path is not None
            if self.save_metadata:
                meta_path = self._save_meta(
                    event_dir, event, detection, reason, stem, write_status, extra_meta)
                write_status["metadata"] = meta_path is not None

            saved_files = [p for p in (raw_path, ann_path, meta_path) if p is not None]

            if replace_record_index is not None and 0 <= replace_record_index < len(self.shots):
                # 峰值替换：覆盖旧记录，不追加新记录
                rec = self.shots[replace_record_index]
                rec.frame_id = detection.frame_id
                rec.simulated_timestamp = round(detection.timestamp, 3)
                rec.raw_image_path = str(raw_path) if raw_path else rec.raw_image_path
                rec.annotated_image_path = str(ann_path) if ann_path else rec.annotated_image_path
                rec.metadata_path = str(meta_path) if meta_path else rec.metadata_path
                rec.fire_area_ratio = round(detection.fire_area_ratio, 4)
                rec.smoke_area_ratio = round(detection.smoke_area_ratio, 4)
                rec.saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rec.write_status = write_status
                return saved_files, replace_record_index

            if saved_files:
                self.total_saved += len(saved_files)
                self.saved_paths.extend(saved_files)
                self.shots.append(self._build_record(
                    event, detection, reason, raw_path, ann_path, meta_path, write_status))
                print(f"  [Screenshot] {event.event_id} | 时机={reason} | 保存 {len(saved_files)} 个文件")
                return saved_files, len(self.shots) - 1
        except Exception as exc:
            print(f"  [Screenshot] 保存失败（不影响主检测流程）: {exc}")
        return saved_files, None

    def _build_record(self, event: FireEvent, detection: Detection,
                      reason: str, raw_path, ann_path, meta_path,
                      write_status: dict) -> ScreenshotRecord:
        """构造统一截图记录（核查文档情况 5）"""
        decision = event.metadata.get("agent_decision")
        score = decision.get("score") if isinstance(decision, dict) else None
        confidence = decision.get("confidence") if isinstance(decision, dict) else None
        source = decision.get("decision_source") if isinstance(decision, dict) else None
        return ScreenshotRecord(
            event_id=event.event_id,
            reason=reason,
            frame_id=detection.frame_id,
            simulated_timestamp=round(detection.timestamp, 3),
            raw_image_path=str(raw_path) if raw_path else "",
            annotated_image_path=str(ann_path) if ann_path else "",
            metadata_path=str(meta_path) if meta_path else "",
            fire_area_ratio=round(detection.fire_area_ratio, 4),
            smoke_area_ratio=round(detection.smoke_area_ratio, 4),
            danger_level=event.metadata.get("danger_level"),
            decision_score=score,
            decision_confidence=confidence,
            decision_source=source,
            saved_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            write_status=write_status,
        )
    def _save_original(self, event_dir: Path, detection: Detection, stem: str) -> Optional[Path]:
        """保存原图（证据保留）"""
        import shutil
        suffix = Path(detection.image_path).suffix or ".jpg"
        target = event_dir / f"{stem}_raw{suffix}"
        shutil.copy2(str(detection.image_path), str(target))
        return target

    def _save_annotated(self, event_dir: Path, event: FireEvent,
                        detection: Detection, stem: str) -> Optional[Path]:
        """直接复用检测框绘制标注图（不重新推理，核查文档情况 5）"""
        img = _cv_read_image(detection.image_path)
        if img is None:
            return None

        import cv2
        for box in detection.bboxes:
            self._draw_box(img, box)

        # 水印仅 ASCII，避免中文渲染问题
        cv2.putText(img, f"Event: {event.event_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Reason: {stem}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        level = event.metadata.get("danger_level", "N/A")
        cv2.putText(img, f"Level: {level}", (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        target = event_dir / f"{stem}_annotated.jpg"
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

    def _save_meta(self, event_dir: Path, event: FireEvent, detection: Detection,
                   reason: str, stem: str, write_status: dict,
                   extra_meta: Optional[dict] = None) -> Optional[Path]:
        """保存截图元数据（原子写入；升级图附带 M7 决策字段，核查文档情况 6/7）"""
        meta = {
            "event_id": event.event_id,
            "frame_id": detection.frame_id,
            "timestamp": round(detection.timestamp, 3),
            "reason": reason,
            "fire_area_ratio": round(detection.fire_area_ratio, 4),
            "smoke_area_ratio": round(detection.smoke_area_ratio, 4),
            "event_status": event.status.value if hasattr(event.status, "value") else str(event.status),
            "avg_fire_confidence": round(event.avg_fire_confidence, 4),
            "avg_smoke_confidence": round(event.avg_smoke_confidence, 4),
            "growth_trend": event.growth_trend.value if hasattr(event.growth_trend, "value") else str(event.growth_trend),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "write_status": write_status,
            "raw_file": f"{stem}_raw{Path(detection.image_path).suffix or '.jpg'}",
            "annotated_file": f"{stem}_annotated.jpg",
        }
        if extra_meta:
            meta.update(extra_meta)

        target = event_dir / f"{stem}_meta.json"
        return target if self._atomic_write_json(target, meta) else None

    def _save_event_info(self, event: FireEvent) -> Optional[Path]:
        """事件结束时汇总事件信息 + 截图记录（原子写入，供 M10 报告）"""
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
                "screenshots": [r.to_dict() for r in self.shots if r.event_id == event.event_id],
            }
            target = self._event_dir(event) / "event_info.json"
            return target if self._atomic_write_json(target, info) else None
        except Exception as exc:
            print(f"  [Screenshot] event_info 保存失败（不影响主流程）: {exc}")
            return None

    @staticmethod
    def _atomic_write_json(target: Path, data: dict) -> bool:
        """先写 .tmp 再替换为正式文件，避免留下损坏的 JSON（核查文档情况 7）"""
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(target)
            return True
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    # ======================== 查询接口 ========================

    def get_saved_paths(self) -> List[Path]:
        """获取所有已保存的文件路径（M8 → M10 契约）"""
        return list(self.saved_paths)

    def get_latest(self) -> Optional[Path]:
        """获取最近保存的文件"""
        return self.saved_paths[-1] if self.saved_paths else None

    def get_event_shot_records(self, event_id: str) -> List[ScreenshotRecord]:
        """获取指定事件的截图记录（供 M10 报告引用）"""
        return [r for r in self.shots if r.event_id == event_id]

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "enabled": self.enabled,
            "peak_metric": self.peak_metric,
            "total_saved": self.total_saved,
            "output_dir": str(self.output_dir),
            "active_events": len(self._event_state),
        }


# ======================== 命令行测试 ========================

def main():
    """独立运行测试：覆盖核查文档要求的 10 项测试 + M6 端到端联动"""
    print("\n  FireGuardian M8 - 自动截图模块测试（按核查文档）")
    print("  =====================================================\n")

    import json as _json
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

    def make_event(event_id, dets, danger="low"):
        return FireEvent(
            event_id=event_id,
            status=EventStatus.CONFIRMED,
            start_timestamp=0.0,
            last_timestamp=float(len(dets)) * 0.5,
            total_frames=len(dets) + 3,
            positive_frames=len(dets),
            consecutive_positive_frames=len(dets),
            max_fire_area_ratio=max(d.fire_area_ratio for d in dets),
            avg_fire_confidence=0.85,
            growth_trend=GrowthTrend.INCREASING,
            source_type="test",
            source_name="bus",
            detections=dets,
            metadata={"danger_level": danger},
        )

    # ---------- 测试 1+2: confirmed 只保存一次 + peak 替换 ----------
    print("  测试 1+2: confirmed 去重 + peak 随更高峰值替换")
    mgr = ScreenshotManager()
    d0, d1, d2, d3 = (make_detection(r, t, i)
                      for r, t, i in ((0.05, 1.0, 1), (0.10, 2.0, 2),
                                      (0.08, 3.0, 3), (0.15, 4.0, 4)))
    evt1 = make_event("FE-CHK-0001", [d0])
    n_c = len(mgr.on_event_confirmed(evt1))
    n_c2 = len(mgr.on_event_confirmed(evt1))          # 重复触发
    assert n_c == 3 and n_c2 == 0, f"confirmed 应只保存一次: {n_c}/{n_c2}"
    assert len(mgr.get_event_shot_records("FE-CHK-0001")) == 1

    evt1.detections.append(d1); mgr.on_event_updated(evt1)  # 0.10 → 首次 peak
    evt1.detections.append(d2); mgr.on_event_updated(evt1)  # 0.08 → 不触发
    evt1.detections.append(d3); mgr.on_event_updated(evt1)  # 0.15 → 替换 peak

    evt1_dir = mgr.path_mgr.screenshot_event_dir("FE-CHK-0001")
    peak_meta = _json.loads((evt1_dir / "peak_meta.json").read_text(encoding="utf-8"))
    assert peak_meta["frame_id"] == 4, f"peak 应指向最高峰帧, 实际 {peak_meta['frame_id']}"
    assert peak_meta["fire_area_ratio"] >= 0.14
    peak_recs = [r for r in mgr.get_event_shot_records("FE-CHK-0001") if r.reason == "peak"]
    assert len(peak_recs) == 1, "peak 记录应只有一条"
    assert peak_recs[0].frame_id == 4
    print("    confirmed 去重 [OK] | peak 替换至 0.15 帧 [OK] | 记录唯一 [OK]")

    # ---------- 测试 3+4+5: 升级去重 / 多级升级 / 降级不截图 ----------
    print("  测试 3+4+5: 升级去重 + 多级升级 + 降级不截图")
    evt2 = make_event("FE-CHK-0002", [d0], danger="low")
    mgr.on_event_confirmed(evt2)
    r1 = mgr.on_decision(evt2, danger_level="medium")   # low→medium
    r2 = mgr.on_decision(evt2, danger_level="medium")   # 重复
    r3 = mgr.on_decision(evt2, danger_level="medium")   # 重复
    assert len(r1) == 3 and len(r2) == 0 and len(r3) == 0, f"{len(r1)}/{len(r2)}/{len(r3)}"
    r4 = mgr.on_decision(evt2, danger_level="high")     # medium→high
    assert len(r4) == 3, f"多级升级应保存: {len(r4)}"
    r5 = mgr.on_decision(evt2, danger_level="medium")   # 降级
    assert len(r5) == 0, "降级不应保存升级截图"

    evt2_dir = mgr.path_mgr.screenshot_event_dir("FE-CHK-0002")
    f2 = sorted(p.name for p in evt2_dir.iterdir())
    assert "upgrade_low_to_medium_raw.jpg" in f2, f2
    assert "upgrade_medium_to_high_raw.jpg" in f2, f2
    assert not any(n.startswith("danger_level_upgraded_") for n in f2), f2
    up_meta = _json.loads((evt2_dir / "upgrade_medium_to_high_meta.json").read_text(encoding="utf-8"))
    assert up_meta["previous_danger_level"] == "medium"
    assert up_meta["current_danger_level"] == "high"
    print("    low→medium / medium→high 分开保存 [OK] | 重复与降级去重 [OK] | 决策字段 [OK]")
    # ---------- 测试 6: final 使用最后阳性帧 ----------
    print("  测试 6: final 使用最后阳性帧")
    evt3 = make_event("FE-CHK-0003", [make_detection(0.05, 1.0, 1), make_detection(0.12, 2.0, 2)])
    mgr.on_event_confirmed(evt3)
    evt3.detections.append(make_detection(0.12, 3.0, 3))
    mgr.on_event_updated(evt3)                          # 更新最后阳性帧
    evt3.status = EventStatus.ENDED                      # 之后连续阴性帧结束
    evt3.last_timestamp = 6.0
    mgr.on_event_ended(evt3)
    final_meta = _json.loads(
        (mgr.path_mgr.screenshot_event_dir("FE-CHK-0003") / "final_meta.json").read_text(encoding="utf-8"))
    assert final_meta["frame_id"] == 3, f"final 应取最后阳性帧, 实际 {final_meta['frame_id']}"
    print(f"    final 帧 = 最后阳性帧(frame_id={final_meta['frame_id']}) [OK]")

    # ---------- 测试 7: 单张图片写入失败 ----------
    print("  测试 7: 标注图写入失败不影响主流程")
    import sys as _sys
    main_mod = _sys.modules["__main__"]
    orig_write = main_mod._cv_write_image
    main_mod._cv_write_image = lambda *a, **k: False     # 模拟标注图写失败
    evt4 = make_event("FE-CHK-0004", [make_detection(0.05, 1.0, 1)])
    mgr4 = ScreenshotManager()
    files4 = mgr4.on_event_confirmed(evt4)
    main_mod._cv_write_image = orig_write
    assert len(files4) >= 2, "raw+meta 应仍保存"
    m4 = _json.loads(
        (mgr4.path_mgr.screenshot_event_dir("FE-CHK-0004") / "confirmed_meta.json").read_text(encoding="utf-8"))
    assert m4["write_status"]["annotated"] is False
    assert m4["write_status"]["raw"] is True
    print("    标注图失败已记录 write_status，主流程未中断 [OK]")

    # ---------- 测试 8: JSON 原子写入（无残留 .tmp） ----------
    print("  测试 8: JSON 原子写入")
    for d in (evt1_dir, evt2_dir, mgr.path_mgr.screenshot_event_dir("FE-CHK-0003")):
        leftovers = list(d.glob("*.tmp"))
        assert not leftovers, f"存在残留 .tmp: {leftovers}"
    print("    所有 JSON 写入无 .tmp 残留 [OK]")

    # ---------- 测试 9: 多事件隔离 ----------
    print("  测试 9: 多事件状态隔离")
    mgr5 = ScreenshotManager()
    evtA = make_event("FE-CHK-0009A", [make_detection(0.05, 1.0, 1)], danger="low")
    evtB = make_event("FE-CHK-0009B", [make_detection(0.30, 1.0, 1)], danger="medium")
    mgr5.on_event_confirmed(evtA)
    mgr5.on_event_confirmed(evtB)
    evtA.detections.append(make_detection(0.12, 2.0, 2))
    mgr5.on_event_updated(evtA)                          # 只影响 A
    stA = mgr5._event_state["FE-CHK-0009A"]
    stB = mgr5._event_state["FE-CHK-0009B"]
    assert stA.peak_saved and stB.peak_saved is False, "B 的峰值不应被 A 影响"
    assert stA.last_danger_level == "low" and stB.last_danger_level == "medium"
    assert all(r.event_id == "FE-CHK-0009A" for r in mgr5.get_event_shot_records("FE-CHK-0009A"))
    print("    事件 A/B 状态与记录互不串档 [OK]")

    # ---------- 测试 10: 不重复推理 ----------
    print("  测试 10: M8 不重新调用 YOLO")
    module_code = Path(main_mod.__file__).read_text(encoding="utf-8").split("def main()")[0].lower()
    assert "ultralytics" not in module_code
    assert "yolo(" not in module_code
    print("    M8 仅复用检测框绘制，无模型推理 [OK]")

    # ---------- 测试 11: 与 M6 聚合器端到端联动 ----------
    print("  测试 11: 挂接 M6 EventAggregator 端到端联动")
    from aggregator.event_aggregator import EventAggregator

    agg = EventAggregator()
    mgr6 = ScreenshotManager()
    mgr6.attach(agg)
    for i in range(1, 6):
        agg.process(make_detection(0.05 + i * 0.03, float(i) * 0.5, i))
    for i in range(6, 9):
        agg.process(Detection(bboxes=[], image_path=str(test_img), timestamp=float(i) * 0.5,
                              image_width=img_w, image_height=img_h, frame_id=i))
    ended = agg.event_history
    assert len(ended) == 1
    evt6 = ended[0]
    assert evt6.status == EventStatus.ENDED
    evt6_dir = mgr6.path_mgr.screenshot_event_dir(evt6.event_id)
    e_files = sorted(p.name for p in evt6_dir.iterdir())
    for must in ("confirmed_raw.jpg", "peak_raw.jpg", "final_raw.jpg", "event_info.json"):
        assert must in e_files, f"联动缺少 {must}: {e_files}"
    assert not list(evt6_dir.glob("*.tmp"))
    assert mgr6._event_state.get(evt6.event_id) is None, "事件结束后缓存应清理"
    print(f"    事件 {evt6.event_id} 自动完成 confirmed/peak/final + event_info [OK]")
    print("    事件结束后缓存已清理 [OK]")

    print("\n  M8 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())