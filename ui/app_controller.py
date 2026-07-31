"""
============================================================
M11 - GUI 控制器 (UiController)
功能：
  - 装配检测源(M3/M4/M5) → M6 聚合 → M7 决策 → M8 截图 → M9 日志 → M10 报告
  - 后台工作线程 + Qt 信号，界面不接触业务逻辑
  - 支持图片 / 视频 / 验证集模拟器三种输入源
  - 训练（M2）异步执行，逐 epoch 日志回调

界面（MainWindow）只做两件事：
  1. 接收用户操作 → 调用 UiController 方法
  2. 接收 UiController 信号 → 更新控件

可独立运行自测: python ui/app_controller.py
============================================================
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 项目路径与 ultralytics 配置（须在导入依赖模块前设置）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

from utils.common import (
    load_config,
    BoundingBox,
    Detection,
    FireEvent,
    EventReportData,
    EventStatus,
)
from utils.screenshot import ScreenshotManager, _cv_read_image
from reports.report_generator import ReportGenerator
from agent.fire_decision_agent import FireDecisionAgent
from aggregator.event_aggregator import EventAggregator
from logs.logger import get_logger, log_event, log_decision, flush as _log_flush


class UiController(QObject):
    """M11 控制器：业务装配 + 线程管理 + 信号输出"""

    # ---- 监控页信号（object 承载，避免 PyQt 对 Python 对象/枚举的签名限制）----
    frame_ready = pyqtSignal(object)         # 画面帧: image(QImage) + 元信息
    frame_info_ready = pyqtSignal(object)    # 当前帧检测信息（右栏）
    event_ready = pyqtSignal(object)         # 当前事件信息（右栏）
    decision_ready = pyqtSignal(object)      # Agent 决策（右栏）
    history_ready = pyqtSignal(object)       # 事件历史列表
    status_ready = pyqtSignal(object)        # 状态栏：运行/暂停/进度/FPS
    source_finished = pyqtSignal(object)     # 检测源运行结束摘要

    # ---- 训练页信号 ----
    train_log_ready = pyqtSignal(str)
    train_finished = pyqtSignal(object)
    train_epoch_ready = pyqtSignal(object)   # 训练逐 epoch 指标（Epoch/Loss/mAP/ETA）

    # ---- 全局 ----
    error_ready = pyqtSignal(str)            # 错误提示（GUI 弹窗）

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.cfg = load_config()
        self.log = get_logger("UI")

        # 业务模块（M6/M7/M8/M10）
        self.aggregator = EventAggregator()
        self.screenshot_mgr = ScreenshotManager()
        self.report_gen = ReportGenerator()
        self.agent = FireDecisionAgent()

        # 检测器惰性创建（避免 GUI 启动时加载 YOLO）
        self._image_detector = None
        self._video_detector = None

        # 决策与事件跟踪
        self.decisions_by_event: Dict[str, list] = {}
        self.event_max_level: Dict[str, str] = {}
        self._reported: set = set()

        # 运行状态
        self._running = False
        self._paused = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._worker: Optional[threading.Thread] = None
        self._current_source = ""
        self._last_frame_id = 0
        self._total_frames = 0
        self._fps = 0.0
        self._last_infer_ms = 0.0
        self._params: dict = {}

        # 训练状态
        self._trainer = None
        self._train_thread: Optional[threading.Thread] = None
        self._shutdown_done = False

        # 接线事件回调（M6 → M7/M8/M9/M10，保证截图→报告时序）
        self._wire_event_callbacks()

        self.log.info("M11 UiController 初始化完成")

    # ======================== M6 事件回调 ========================

    def _wire_event_callbacks(self):
        """手工接线（保证时序：确认/更新 → 截图→决策，结束 → 截图→报告）"""
        agg = self.aggregator
        agg.on_event_confirmed = self._on_event_confirmed
        agg.on_event_updated = self._on_event_updated
        agg.on_event_ended = self._on_event_ended
        agg.on_event_discarded = self._on_event_discarded

    def _on_event_confirmed(self, evt: FireEvent):
        """事件确认：M8 截图 → M7 决策 → M8 升级截图 → M9 日志 → 通知界面"""
        try:
            self.screenshot_mgr.on_event_confirmed(evt)
        except Exception as e:
            self.log.warning(f"M8 确认截图失败: {e}")
        try:
            self._analyze_event(evt)
        except Exception as e:
            self.log.warning(f"M7 确认决策失败: {e}")
        try:
            log_event(evt, "event_confirmed", module="event_aggregator", source_module="M6")
        except Exception:
            pass
        self._emit_event_state()

    def _on_event_updated(self, evt: FireEvent):
        """事件更新：M8 峰值/升级截图 → 按需决策（趋势变化等）→ M9 日志"""
        try:
            self.screenshot_mgr.on_event_updated(evt)
        except Exception as e:
            self.log.warning(f"M8 更新截图失败: {e}")
        # 仅当聚合器标记需要决策时调用 Agent，避免每帧调用
        if evt.metadata.get("decision_required"):
            evt.metadata["decision_required"] = False
            try:
                self._analyze_event(evt)
            except Exception as e:
                self.log.warning(f"M7 更新决策失败: {e}")
        try:
            log_event(evt, "event_updated", module="event_aggregator", source_module="M6")
        except Exception:
            pass
        self._emit_event_state()

    def _on_event_ended(self, evt: FireEvent):
        """事件结束：最终决策 → M8 final 截图 → M10 报告 → M9 日志 → 通知界面"""
        try:
            self._analyze_event(evt)
        except Exception as e:
            self.log.warning(f"M7 结束决策失败: {e}")
        try:
            self.screenshot_mgr.on_event_ended(evt)
        except Exception as e:
            self.log.warning(f"M8 结束截图失败: {e}")
        try:
            self._generate_report(evt)
        except Exception as e:
            self.log.warning(f"M10 报告生成失败: {e}")
        try:
            log_event(evt, "event_ended", module="event_aggregator", source_module="M6")
        except Exception:
            pass
        self._emit_event_state()
        self._emit_history()

    def _on_event_discarded(self, evt: FireEvent):
        """候选事件废弃：记录日志并刷新历史"""
        try:
            log_event(evt, "event_discarded", module="event_aggregator", source_module="M6")
        except Exception:
            pass
        self._emit_history()

    def _analyze_event(self, evt: FireEvent):
        """调用 M7 生成决策，记录最高危险等级并通知界面"""
        decision = self.agent.analyze(evt)
        self.decisions_by_event.setdefault(evt.event_id, []).append(decision)

        level = decision.danger_level
        if hasattr(level, "value"):
            level = level.value
        rank = {"low": 0, "medium": 1, "high": 2}
        cur = self.event_max_level.get(evt.event_id)
        if cur is None or rank.get(str(level), 0) > rank.get(str(cur), 0):
            self.event_max_level[evt.event_id] = str(level)

        # M8 记录危险等级 / 检查升级截图
        try:
            self.screenshot_mgr.on_decision(evt, decision)
        except Exception as e:
            self.log.warning(f"M8 决策截图挂钩失败: {e}")

        # M9 决策日志
        try:
            log_decision(evt, decision, module="fire_decision_agent", source_module="M7")
        except Exception:
            pass

        evt.metadata["decision_required"] = False
        self.decision_ready.emit(self._decision_payload(evt, decision))

    def _generate_report(self, evt: FireEvent):
        """M10 事件驱动报告（截图已就绪后调用）"""
        if evt.event_id in self._reported:
            return
        self._reported.add(evt.event_id)
        shots = self.screenshot_mgr.get_event_shot_records(evt.event_id)
        decisions = self.decisions_by_event.get(evt.event_id, [])
        data = EventReportData(event=evt, decisions=decisions, screenshots=shots)
        self.report_gen.generate(data)
        self.log.info(f"M10 报告已生成: {evt.event_id}")

    # ======================== 数据源控制 ========================

    def start(self, source_type: str, path: str | Path, params: Optional[dict] = None):
        """启动检测源（工作线程）；返回是否成功"""
        if self._running:
            self.log.warning("已有检测源在运行，请先停止")
            return False
        path = Path(path)
        if not path.exists():
            self.log.error(f"路径不存在: {path}")
            return False
        # 清理上一次运行状态
        self.aggregator.reset()
        self.decisions_by_event.clear()
        self.event_max_level.clear()
        self._reported.clear()
        self.event_ready.emit(None)
        self.history_ready.emit([])
        self.frame_info_ready.emit(None)

        self._running = True
        self._paused = False
        self._pause_event.set()
        self._current_source = source_type
        self._last_frame_id = 0
        self._total_frames = 0
        self._fps = 0.0
        self._params = params or {}
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(source_type, str(path), self._params),
            daemon=True,
            name="FireGuardianWorker",
        )
        self._worker.start()
        self._emit_status()
        self.log.info(f"启动检测源: {source_type} -> {path.name}")
        return True

    def pause(self):
        """暂停播放"""
        if self._running and not self._paused:
            self._paused = True
            self._pause_event.clear()
            self._emit_status()
            self.log.info("检测已暂停")

    def resume(self):
        """继续播放"""
        if self._running and self._paused:
            self._paused = False
            self._pause_event.set()
            self._emit_status()
            self.log.info("检测已继续")

    def stop(self):
        """停止当前检测源（等待当前帧推理完成）"""
        if not self._running:
            return
        self.log.info("停止检测源...")
        self._running = False
        self._pause_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=10)
        self._emit_status()

    def reset(self):
        """重置聚合/决策状态（切换数据源前调用）"""
        self.stop()
        self.aggregator.reset()
        self.decisions_by_event.clear()
        self.event_max_level.clear()
        self._reported.clear()
        self.event_ready.emit(None)
        self.history_ready.emit([])
        self.frame_info_ready.emit(None)
        self.log.info("已重置检测状态")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return self._paused

    # ======================== 工作线程 ========================

    def _run_worker(self, source_type: str, path: str, params: dict):
        """检测源工作线程主入口"""
        try:
            if source_type == "image":
                self._run_image(path)
            elif source_type == "video":
                self._run_video(path)
            elif source_type == "simulator":
                self._run_simulator(path, params)
            else:
                self.log.error(f"未知检测源: {source_type}")
        except Exception as e:
            self.log.error(f"检测线程异常: {e}")
            import traceback
            traceback.print_exc()
            self.error_ready.emit(f"检测线程异常: {e}")
        finally:
            self._running = False
            self._paused = False
            self._emit_status()
            stats = self.aggregator.get_stats()
            self.source_finished.emit({
                "source": source_type,
                "processed_frames": self.aggregator.total_processed,
                "events_confirmed": stats.get("total_events_confirmed", 0),
                "events_ended": stats.get("total_events_ended", 0),
                "events_discarded": stats.get("total_events_discarded", 0),
            })
            self.log.info(f"检测源结束: {source_type}")

    def _run_image(self, path: str):
        """M3 单图检测（一次推理）"""
        det = self._ensure_image_detector(self._params.get("conf"), self._params.get("iou"))
        img = _cv_read_image(path)
        if img is None:
            self.log.error(f"无法读取图片: {path}")
            return
        detection = det.detect_single(path, save_result=False)
        h, w = img.shape[:2]
        if detection.image_width <= 0:
            detection.image_width = w
            detection.image_height = h
        if detection.frame_id < 0:
            detection.frame_id = 1
        if detection.timestamp <= 0:
            detection.timestamp = 0.0
        self._total_frames = 1
        self._last_frame_id = 1
        self._process_frame(img, detection, f"图片/{Path(path).name}")
        self._emit_status()

    def _run_video(self, path: str):
        """M4 视频逐帧检测（decode 生成器，不写输出视频）"""
        det = self._ensure_video_detector(self._params.get("conf"), self._params.get("iou"))
        t0 = time.time()
        processed = 0
        for frame, detection, frame_idx, total in det.decode(path):
            if not self._running:
                break
            self._pause_event.wait()
            if not self._running:
                break
            self._total_frames = total
            self._last_frame_id = frame_idx
            self._process_frame(frame, detection, f"视频/{Path(path).name}")
            processed += 1
            elapsed = time.time() - t0
            self._fps = processed / elapsed if elapsed > 0 else 0.0
            # 按视频原始帧率节奏播放（补偿推理耗时）
            if detection.timestamp > 0:
                fps = frame_idx / detection.timestamp
                wait = max(0.0, 1.0 / fps - detection.inference_time_ms / 1000.0)
                time.sleep(wait)
            self._emit_status()

    def _run_simulator(self, path: str, params: dict):
        """M5 验证集模拟器（next_frame 驱动，支持 interval/mode/loop/max_frames）"""
        from simulator.validator_simulator import ValidationStreamSimulator

        detector = self._simulator_detector(self._params.get("conf"), self._params.get("iou"))
        interval = float(params.get("interval", 0.5))
        mode = str(params.get("mode", "sequential"))
        loop = bool(params.get("loop", False))
        max_frames = params.get("max_frames") or None

        sim = ValidationStreamSimulator(
            image_dir=path,
            detector=detector,
            interval_seconds=interval,
            mode=mode,
            loop=loop,
            max_frames=max_frames,
        )
        t0 = time.time()
        processed = 0
        while self._running:
            self._pause_event.wait()
            if not self._running:
                break
            try:
                out = sim.next_frame()
            except Exception as e:
                self.log.warning(f"模拟器单帧失败: {e}")
                break
            if out is None:
                break
            frame_data, detection = out
            img = frame_data.image
            if img is None and frame_data.image_path:
                img = _cv_read_image(frame_data.image_path)
            if img is None:
                self.log.warning(f"跳过无法读取的帧: {frame_data.image_path}")
                continue
            if detection.frame_id < 0:
                detection.frame_id = frame_data.frame_id
            self._last_frame_id = frame_data.frame_id
            self._process_frame(img, detection, f"验证集/{Path(frame_data.image_path).parent.name}")
            processed += 1
            elapsed = time.time() - t0
            self._fps = processed / elapsed if elapsed > 0 else 0.0
            # 按 interval 节奏（补偿推理耗时）
            wait = max(0.0, interval - detection.inference_time_ms / 1000.0)
            time.sleep(wait)
            self._emit_status()

    def _process_frame(self, img, detection: Detection, source_name: str):
        """单帧通用处理：绘制 → 发送画面/当前帧信息 → 送入 M6 聚合"""
        annotated = self._draw_detections(img, detection)
        qimg = self._bgr_to_qimage(annotated)
        self._last_infer_ms = detection.inference_time_ms

        self.frame_ready.emit({
            "image": qimg,
            "frame_id": detection.frame_id,
            "total_frames": self._total_frames,
            "source_name": source_name,
            "inference_ms": round(detection.inference_time_ms, 1),
            "timestamp": round(detection.timestamp, 2),
        })

        self.frame_info_ready.emit({
            "frame_id": detection.frame_id,
            "timestamp": round(detection.timestamp, 2),
            "classes": self._class_summary(detection),
            "fire_area_ratio": round(detection.fire_area_ratio * 100, 2),
            "smoke_area_ratio": round(detection.smoke_area_ratio * 100, 2),
            "inference_time_ms": round(detection.inference_time_ms, 1),
            "detection_count": len(detection.bboxes),
        })

        try:
            self.aggregator.process(detection)
        except Exception as e:
            self.log.error(f"M6 聚合失败: {e}")

    # ======================== 检测器（惰性）====================

    def _ensure_image_detector(self, conf=None, iou=None):
        if self._image_detector is None:
            from detect.image_detector import ImageDetector
            self.log.info("加载 YOLO 检测模型（首次较慢）...")
            self._image_detector = ImageDetector(conf_threshold=conf, iou_threshold=iou)
        return self._image_detector

    def _ensure_video_detector(self, conf=None, iou=None):
        if self._video_detector is None:
            from video_detect.video_detector import VideoDetector
            self.log.info("加载 YOLO 检测模型（首次较慢）...")
            self._video_detector = VideoDetector(conf_threshold=conf, iou_threshold=iou)
        return self._video_detector

    def _simulator_detector(self, conf=None, iou=None):
        """构造 M5 需要的 detector callable(image) -> List[BoundingBox]"""
        det = self._ensure_image_detector(conf, iou)
        model = det.model
        conf = det.conf_threshold
        iou = det.iou_threshold
        device = det.device

        def detect_fn(image):
            results = model(image, conf=conf, iou=iou, device=device, verbose=False)
            r = results[0]
            bboxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    cid = int(cls_ids[i])
                    bboxes.append(BoundingBox(
                        x1=float(xyxy[i][0]), y1=float(xyxy[i][1]),
                        x2=float(xyxy[i][2]), y2=float(xyxy[i][3]),
                        confidence=float(confs[i]),
                        class_id=cid,
                        class_name=r.names.get(cid, f"class_{cid}"),
                    ))
            return bboxes

        return detect_fn

    def _current_model_name(self) -> str:
        """当前使用的模型文件名（供界面展示）"""
        det = self._image_detector or self._video_detector
        if det is not None and getattr(det, "model_path", None):
            return Path(det.model_path).name
        model_cfg = self.cfg.get("model", {})
        return Path(str(model_cfg.get("path", "models/best.pt"))).name

    # ======================== 画面绘制与转换 ========================

    @staticmethod
    def _draw_detections(img, detection: Detection):
        """按 M8 样式绘制检测框（fire=红, smoke=橙），不改原图"""
        import cv2
        out = img.copy()
        colors = {"fire": (0, 0, 255), "smoke": (0, 165, 255)}
        for box in detection.bboxes:
            x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
            color = colors.get(box.class_name, (0, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{box.class_name} {box.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    @staticmethod
    def _bgr_to_qimage(bgr):
        """BGR ndarray → QImage(RGB888)，深拷贝数据"""
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

    @staticmethod
    def _class_summary(detection: Detection) -> list:
        """按类别汇总：类别/数量/最高置信度"""
        agg: Dict[str, dict] = {}
        for box in detection.bboxes:
            entry = agg.setdefault(box.class_name, {
                "class_name": box.class_name, "count": 0, "max_confidence": 0.0,
            })
            entry["count"] += 1
            entry["max_confidence"] = max(entry["max_confidence"], box.confidence)
        return list(agg.values())

    # ======================== 界面数据载荷 ========================

    def _decision_payload(self, evt: FireEvent, decision) -> dict:
        return {
            "event_id": evt.event_id,
            "danger_level": decision.danger_level.value if hasattr(decision.danger_level, "value") else str(decision.danger_level),
            "score": round(float(getattr(decision, "score", 0.0)), 3),
            "confidence": round(float(getattr(decision, "confidence", 0.0)), 3),
            "reasons": list(getattr(decision, "reasons", [])),
            "suggestions": list(getattr(decision, "suggestions", [])),
            "summary": getattr(decision, "summary", ""),
            "decision_source": getattr(decision, "decision_source", ""),
            "generated_at": getattr(decision, "generated_at", ""),
        }

    def _emit_event_state(self):
        evt = self.aggregator.current_event
        if evt is None:
            self.event_ready.emit(None)
            return
        self.event_ready.emit({
            "event_id": evt.event_id,
            "status": evt.status.value if hasattr(evt.status, "value") else str(evt.status),
            "duration": round(evt.duration, 2),
            "growth_trend": evt.growth_trend.value if hasattr(evt.growth_trend, "value") else str(evt.growth_trend),
            "danger_level": evt.metadata.get("danger_level", "-"),
            "fire_area": round(evt.max_fire_area_ratio * 100, 2),
            "smoke_area": round(evt.max_smoke_area_ratio * 100, 2),
            "positive_ratio": round(evt.positive_ratio, 2),
            "total_frames": evt.total_frames,
        })

    def _emit_history(self):
        items = []
        for evt in self.aggregator.event_history:
            evt_dir = self.report_gen.path_mgr.report_event_dir(evt.event_id)
            report_path = evt_dir / "report.pdf"
            if not report_path.exists():
                report_path = evt_dir / "report.md"
            exists = report_path.exists()
            if exists:
                status = "已生成"
            elif evt.event_id in self._reported:
                status = "生成失败"
            else:
                status = "生成中"
            start_time = (datetime.fromtimestamp(evt.start_timestamp).strftime("%H:%M:%S")
                          if evt.start_timestamp > 0 else "")
            items.append({
                "event_id": evt.event_id,
                "start_time": start_time,
                "duration": round(evt.duration, 1),
                "max_danger": self.event_max_level.get(evt.event_id,
                                                       evt.metadata.get("danger_level", "-")),
                "report_status": status,
                "report_path": str(report_path) if exists else "",
            })
        self.history_ready.emit(items)

    def _emit_status(self, **kw):
        payload = {
            "running": self._running,
            "paused": self._paused,
            "source": self._current_source,
            "frame_id": self._last_frame_id,
            "total_frames": self._total_frames,
            "fps": round(self._fps, 1),
            "inference_ms": round(self._last_infer_ms, 1),
            "model": self._current_model_name(),
            "events_confirmed": self.aggregator.get_stats().get("total_events_confirmed", 0),
            "events_ended": self.aggregator.get_stats().get("total_events_ended", 0),
        }
        payload.update(kw)
        self.status_ready.emit(payload)
    # ======================== 日志读取（GUI 轮询）====================

    @staticmethod
    def log_file() -> Path:
        return _ROOT / "logs" / "application.log"

    @classmethod
    def read_log_tail(cls, max_lines: int = 100) -> List[str]:
        """读取 application.log 尾部若干行（GUI 日志页轮询）"""
        try:
            path = cls.log_file()
            if not path.exists():
                return []
            text = path.read_text(encoding="utf-8", errors="replace")
            return text.splitlines()[-max_lines:]
        except Exception:
            return []

    @classmethod
    def log_line_count(cls) -> int:
        try:
            path = cls.log_file()
            if not path.exists():
                return 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def reports_dir(self) -> str:
        """报告根目录（GUI「打开报告目录」按钮）"""
        return str(self.report_gen.path_mgr.reports_dir())

    @staticmethod
    def system_info() -> dict:
        """采集系统/环境信息（GUI 右下角展示）"""
        info = {
            "python": sys.version.split()[0],
            "torch": "-",
            "cuda": "不可用",
            "gpu": "-",
            "cv2": "-",
            "ultralytics": "-",
        }
        try:
            import torch
            info["torch"] = torch.__version__
            if torch.cuda.is_available():
                info["cuda"] = "可用"
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        try:
            import cv2
            info["cv2"] = cv2.__version__
        except Exception:
            pass
        try:
            import ultralytics
            info["ultralytics"] = ultralytics.__version__
        except Exception:
            pass
        return info

    # ======================== 训练（M2）====================

    def start_training(self, weights: str, dataset_yaml: str, epochs: int,
                       batch: int, imgsz: int, device: str,
                       project: str, name: str) -> bool:
        """异步启动训练；返回是否成功"""
        if self._train_thread and self._train_thread.is_alive():
            self.log.warning("训练已在进行中")
            return False
        from train.yolo_trainer import YOLOTrainer

        self._trainer = YOLOTrainer(weights=weights, project=project,
                                    name=name, dataset_yaml=dataset_yaml)
        self._train_thread = threading.Thread(
            target=self._train_worker,
            args=(self._trainer, epochs, batch, imgsz, device),
            daemon=True,
            name="FireGuardianTrain",
        )
        self._train_thread.start()
        self.train_log_ready.emit(f"[开始] 训练启动: weights={Path(weights).name}, "
                                  f"epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}")
        return True

    def _train_worker(self, trainer, epochs: int, batch: int, imgsz: int, device: str):
        def callback(info: dict):
            if info.get("kind") == "epoch":
                self.train_epoch_ready.emit(info)
                line = (f"[Epoch {info['epoch']:>3}/{info['epochs']}] "
                        f"loss={info['loss']:.4f} P={info['precision']:.4f} "
                        f"R={info['recall']:.4f} mAP50={info['mAP50']:.4f} "
                        f"mAP50-95={info['mAP50-95']:.4f} ETA={info['eta_s']:.0f}s")
                self.train_log_ready.emit(line)
            elif info.get("kind") == "done":
                self.train_log_ready.emit(
                    f"[完成] 总耗时 {info['total_seconds']:.1f}s | 最佳 mAP50 {info['best_map']:.4f}\n"
                    f"最佳模型: {info['best']}\n最后模型: {info['last']}")

        from train.yolo_trainer import TrainingStopped
        try:
            best = trainer.train(epochs=epochs, batch=batch, imgsz=imgsz,
                                 device=device, callback=callback)
            self.train_finished.emit({"success": True, "best": str(best)})
        except TrainingStopped:
            self.train_finished.emit({"success": False, "stopped": True,
                                      "error": "训练已由用户停止"})
        except Exception as e:
            self.train_finished.emit({"success": False, "error": str(e)})

    def stop_training(self):
        """请求停止训练（下一个 epoch 边界生效）"""
        if self._trainer is not None:
            self._trainer.request_stop()
        if self._train_thread and self._train_thread.is_alive():
            self._train_thread.join(timeout=5)
        self.train_log_ready.emit("[停止] 训练停止请求已发送")

    def training_running(self) -> bool:
        return bool(self._train_thread and self._train_thread.is_alive())

    def training_results_dir(self) -> str:
        """训练结果目录（供 GUI 打开）"""
        return str(_ROOT / getattr(self._trainer, "project", "train")
                   / getattr(self._trainer, "exp_name", "exp"))

    # ======================== 关闭 ========================

    def shutdown(self):
        """退出前释放资源：停止线程 → 等待 → 落盘日志摘要（幂等）"""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.stop()
        self.stop_training()
        try:
            _log_flush()
        except Exception:
            pass
        try:
            from logs.logger import save_run_summary
            save_run_summary()
        except Exception:
            pass
        self.log.info("M11 控制器已关闭")


# ======================== 离屏自测 ========================

def main():
    """合成帧驱动 M6→M7→M8→M9→M10 全链路，不依赖 F 盘数据"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    print("\n  FireGuardian M11 - UiController 自测（离屏）")
    print("  ===============================================\n")

    ctrl = UiController()
    counts = {"frames": 0, "decisions": 0, "history": 0}
    ctrl.frame_ready.connect(lambda _p: counts.__setitem__("frames", counts["frames"] + 1))
    ctrl.decision_ready.connect(lambda _p: counts.__setitem__("decisions", counts["decisions"] + 1))
    ctrl.history_ready.connect(lambda _p: counts.__setitem__("history", counts["history"] + 1))

    test_img = _ROOT / "bus.jpg"
    if not test_img.exists():
        print("  未找到测试图片 bus.jpg")
        return 1
    img = _cv_read_image(test_img)
    assert img is not None, "测试图片读取失败"
    h, w = img.shape[:2]

    # 6 帧火焰（≥3 帧确认事件）+ 4 帧空白（≥3 帧连续阴性结束事件）
    for i in range(1, 7):
        box = BoundingBox(x1=100, y1=100, x2=300, y2=300, confidence=0.85,
                          class_id=0, class_name="fire")
        det = Detection(bboxes=[box], image_path=str(test_img), timestamp=float(i) * 0.5,
                        inference_time_ms=5.0, image_width=w, image_height=h, frame_id=i)
        ctrl._process_frame(img, det, "自测")
    for i in range(7, 11):
        det = Detection(bboxes=[], image_path=str(test_img), timestamp=float(i) * 0.5,
                        inference_time_ms=5.0, image_width=w, image_height=h, frame_id=i)
        ctrl._process_frame(img, det, "自测")

    ended = ctrl.aggregator.event_history
    assert len(ended) == 1, f"应结束 1 个事件, 实际 {len(ended)}"
    evt = ended[0]
    assert evt.status == EventStatus.ENDED, f"事件状态错误: {evt.status}"
    report = ctrl.report_gen.path_mgr.report_event_dir(evt.event_id) / "report.md"
    assert report.exists(), f"报告未生成: {report}"
    print(f"  事件 {evt.event_id} 已结束并生成报告 [OK]")
    assert counts["frames"] >= 10, f"帧数不足: {counts['frames']}"
    assert counts["decisions"] >= 2, f"决策次数不足: {counts['decisions']}"
    assert counts["history"] >= 1, "未发送事件历史"
    print(f"  帧信号 {counts['frames']} | 决策信号 {counts['decisions']} | 历史信号 {counts['history']} [OK]")
    assert ctrl.log_line_count() > 0, "application.log 无内容"
    print(f"  系统日志行数: {ctrl.log_line_count()} [OK]")

    # 训练接口冒烟（不实际训练）
    from train.yolo_trainer import YOLOTrainer
    t = YOLOTrainer()
    assert t.request_stop() is None
    print(f"  M2 训练接口冒烟: weights={t.weights.name}, dataset={Path(t.dataset_yaml).name} [OK]")

    print("\n  M11 UiController 自测完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())