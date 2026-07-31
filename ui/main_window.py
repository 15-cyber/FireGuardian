"""
============================================================
M11 - GUI 主窗口 (MainWindow, PyQt6)
功能：
  - 监控页：左栏数据源/播放控制、中央实时画面、右栏三区信息、
            底部事件历史与状态栏
  - 训练页：权重/数据集/epochs/batch/imgsz/device + 实时训练日志（第一版不做曲线）
  - 日志页：实时轮询 application.log 尾部（限长 1000 行）

界面只负责：接收用户操作 → 调用 UiController；接收信号 → 显示结果。
业务逻辑全部在 UiController / 各业务模块中。

可独立运行: python ui/main_window.py
============================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 项目路径与 ultralytics 配置
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

from PyQt6.QtCore import Qt, QTimer, QUrl, QSettings
from PyQt6.QtGui import QDesktopServices, QPixmap, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPlainTextEdit,
    QScrollArea,
    QProgressBar,
    QFileDialog,
    QMessageBox,
)

from ui.app_controller import UiController
from utils.common import load_config

# ======================== 常量 ========================
_SOURCES = {
    "image": "图片检测",
    "video": "视频检测",
    "simulator": "验证集模拟器",
}
_MAX_LOG_LINES = 1000  # 日志页限长（完整日志仍写入文件）


class MainWindow(QMainWindow):
    """FireGuardian 主窗口"""

    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller or UiController(self)
        self.cfg = load_config()
        self._settings = QSettings("FireGuardian", "FireGuardian")
        self._last_qimage = None
        self._last_log_tail: list = []
        self._history_report_paths: dict = {}
        self._error_state = False

        self._build_ui()
        self._connect_signals()
        self._apply_defaults()
        self._apply_recent_paths()

        gui_cfg = self.cfg.get("gui", {})
        self.setWindowTitle(gui_cfg.get("window_title", "FireGuardian - 智能火灾监测系统"))
        size = gui_cfg.get("window_size", {})
        self.resize(int(size.get("width", 1280)), int(size.get("height", 800)))
        self._refresh_buttons()
        self._refresh_train_buttons()

    # ======================== UI 构建 ========================

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_monitor_tab(), "监控")
        self.tabs.addTab(self._build_train_tab(), "训练")
        self.tabs.addTab(self._build_log_tab(), "日志")
        self.setCentralWidget(self.tabs)

    def _build_monitor_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 720, 320])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_history_panel())
        root.addWidget(self._build_status_bar())
        return page

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        # ---- 数据源 ----
        gb = QGroupBox("数据源")
        form = QFormLayout(gb)
        self.source_combo = QComboBox()
        for key, label in _SOURCES.items():
            self.source_combo.addItem(label, key)
        form.addRow("类型:", self.source_combo)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("选择图片 / 视频 / 验证集目录")
        form.addRow("路径:", self.path_edit)
        self.open_btn = QPushButton("打开...")
        form.addRow(self.open_btn)
        lay.addWidget(gb)

        # ---- 播放控制 ----
        gb2 = QGroupBox("播放控制")
        grid = QGridLayout(gb2)
        self.start_btn = QPushButton("开始")
        self.pause_btn = QPushButton("暂停")
        self.resume_btn = QPushButton("继续")
        self.stop_btn = QPushButton("停止")
        grid.addWidget(self.start_btn, 0, 0)
        grid.addWidget(self.pause_btn, 0, 1)
        grid.addWidget(self.resume_btn, 1, 0)
        grid.addWidget(self.stop_btn, 1, 1)
        lay.addWidget(gb2)

        # ---- 模拟器参数 ----
        gb3 = QGroupBox("模拟器参数")
        form3 = QFormLayout(gb3)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.05, 10.0)
        self.interval_spin.setValue(0.5)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setSuffix(" s")
        form3.addRow("帧间隔:", self.interval_spin)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("顺序播放", "sequential")
        self.mode_combo.addItem("随机播放", "random")
        form3.addRow("模式:", self.mode_combo)
        self.loop_check = QCheckBox("循环播放")
        form3.addRow(self.loop_check)
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 100000)
        self.max_frames_spin.setValue(0)
        self.max_frames_spin.setSpecialValueText("不限")
        form3.addRow("最大帧数:", self.max_frames_spin)
        self.sim_group = gb3
        lay.addWidget(gb3)

        # ---- 检测参数 ----
        gb4 = QGroupBox("检测参数")
        form4 = QFormLayout(gb4)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.05)
        form4.addRow("置信度:", self.conf_spin)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        form4.addRow("IoU:", self.iou_spin)
        lay.addWidget(gb4)

        lay.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        self.video_label = QLabel("请在左侧选择数据源并点击「开始」")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #1e1e1e; color: #8a8a8a; font-size: 14px;")
        self.video_label.setMinimumSize(480, 360)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.video_label)
        return scroll

    def _build_right_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_frame_tab(), "当前帧")
        tabs.addTab(self._build_event_tab(), "当前事件")
        tabs.addTab(self._build_decision_tab(), "Agent 决策")
        return tabs

    @staticmethod
    def _make_kv_table(rows: int = 0) -> QTableWidget:
        table = QTableWidget(rows, 2)
        table.setHorizontalHeaderLabels(["项目", "数值"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        return table

    @staticmethod
    def _level_color(level) -> str:
        """危险等级 → 统一颜色（High 红 / Medium 橙 / Low 绿）"""
        lv = str(level or "").strip().lower()
        if lv in ("high", "高"):
            return "#e74c3c"
        if lv in ("medium", "中"):
            return "#f39c12"
        if lv in ("low", "低"):
            return "#2ecc71"
        return "#666666"

    def _build_frame_tab(self) -> QWidget:
        self.frame_table = self._make_kv_table(0)
        return self.frame_table

    def _build_event_tab(self) -> QWidget:
        self.event_table = self._make_kv_table(0)
        return self.event_table

    def _build_decision_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        self.decision_level_label = QLabel("危险等级: -")
        self.decision_level_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #666666;")
        self.decision_text = QPlainTextEdit()
        self.decision_text.setReadOnly(True)
        self.decision_meta = QLabel("来源: - | 更新: -")
        self.decision_meta.setStyleSheet("color: #666;")
        lay.addWidget(self.decision_level_label)
        lay.addWidget(self.decision_text, 1)
        lay.addWidget(self.decision_meta)
        return page

    def _build_history_panel(self) -> QWidget:
        gb = QGroupBox("事件历史")
        lay = QVBoxLayout(gb)
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(
            ["事件编号", "开始时间", "持续时间(s)", "最高危险等级", "报告状态"])
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.history_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.doubleClicked.connect(self._on_history_double_click)
        self.view_report_btn = QPushButton("查看报告")
        self.view_report_btn.clicked.connect(self._on_view_report)
        self.open_reports_btn = QPushButton("打开报告目录")
        self.open_reports_btn.clicked.connect(self._on_open_reports_dir)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.open_reports_btn)
        btn_row.addWidget(self.view_report_btn)
        lay.addWidget(self.history_table)
        lay.addLayout(btn_row)
        return gb

    def _build_status_bar(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        self.status_light = QLabel("●")
        self._set_status_light("ready")
        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("FPS: -")
        self.infer_label = QLabel("推理: -")
        self.model_label = QLabel("模型: -")
        self.sysinfo_label = QLabel("GPU: - | CUDA: - | Torch: - | CV: -")
        self.sysinfo_label.setStyleSheet("color: #666;")
        self.sysinfo_label.setToolTip("系统环境信息")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(200)
        lay.addWidget(self.status_light)
        lay.addWidget(self.status_label, 1)
        lay.addWidget(self.fps_label)
        lay.addWidget(self.infer_label)
        lay.addWidget(self.model_label)
        lay.addWidget(self.sysinfo_label)
        lay.addWidget(self.progress)
        return bar

    def _set_status_light(self, state: str):
        """状态指示灯：Ready 绿 / Running 蓝 / Warning 橙 / Error 红"""
        colors = {
            "ready": "#2ecc71",
            "running": "#3498db",
            "warning": "#f39c12",
            "error": "#e74c3c",
        }
        tips = {
            "ready": "就绪",
            "running": "运行中",
            "warning": "警告/暂停",
            "error": "错误",
        }
        if not hasattr(self, "status_light"):
            return
        self.status_light.setStyleSheet(
            f"color: {colors.get(state, '#2ecc71')}; font-size: 20px;")
        self.status_light.setToolTip(tips.get(state, "就绪"))

    def _build_train_tab(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        gb = QGroupBox("训练参数")
        form = QFormLayout(gb)
        w_row = QHBoxLayout()
        self.weights_combo = QComboBox()
        self.weights_combo.setEditable(True)
        self.weights_combo.addItems(self._find_weights())
        self.weights_browse_btn = QPushButton("浏览...")
        self.weights_browse_btn.clicked.connect(self._on_browse_weights)
        w_row.addWidget(self.weights_combo, 1)
        w_row.addWidget(self.weights_browse_btn)
        form.addRow("权重:", w_row)

        d_row = QHBoxLayout()
        self.dataset_edit = QLineEdit(str(_ROOT / "datasets" / "fire.yaml"))
        self.dataset_browse_btn = QPushButton("浏览...")
        self.dataset_browse_btn.clicked.connect(self._on_browse_dataset)
        d_row.addWidget(self.dataset_edit, 1)
        d_row.addWidget(self.dataset_browse_btn)
        form.addRow("数据集YAML:", d_row)

        train_cfg = self.cfg.get("train", {})
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(int(train_cfg.get("epochs", 100)))
        form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(int(train_cfg.get("batch", 16)))
        form.addRow("Batch:", self.batch_spin)

        self.imgsz_combo = QComboBox()
        for s in ("320", "416", "512", "640", "768", "1024"):
            self.imgsz_combo.addItem(s, int(s))
        idx = self.imgsz_combo.findText(str(train_cfg.get("imgsz", 640)))
        self.imgsz_combo.setCurrentIndex(idx if idx >= 0 else 3)
        form.addRow("Imgsz:", self.imgsz_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["0 (GPU)", "-1 (CPU)", "cpu", "0", "1"])
        default_device = str(self.cfg.get("model", {}).get("device", 0))
        di = self.device_combo.findText(default_device)
        self.device_combo.setCurrentIndex(di if di >= 0 else 0)
        form.addRow("Device:", self.device_combo)

        self.project_edit = QLineEdit(str(train_cfg.get("project", "train")))
        form.addRow("Project:", self.project_edit)
        self.name_edit = QLineEdit(str(train_cfg.get("name", "exp")))
        form.addRow("Name:", self.name_edit)
        root.addWidget(gb)

        btn_row = QHBoxLayout()
        self.train_start_btn = QPushButton("开始训练")
        self.train_stop_btn = QPushButton("停止")
        self.train_open_btn = QPushButton("打开结果目录")
        btn_row.addWidget(self.train_start_btn)
        btn_row.addWidget(self.train_stop_btn)
        btn_row.addWidget(self.train_open_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        status_row = QHBoxLayout()
        self.train_epoch_label = QLabel("Epoch: -")
        self.train_loss_label = QLabel("Loss: -")
        self.train_map_label = QLabel("mAP50: -")
        self.train_eta_label = QLabel("ETA: -")
        self.train_device_label = QLabel("Device: -")
        for lbl in (self.train_epoch_label, self.train_loss_label,
                    self.train_map_label, self.train_eta_label, self.train_device_label):
            lbl.setStyleSheet("font-weight: bold;")
            status_row.addWidget(lbl)
        status_row.addStretch(1)
        root.addLayout(status_row)

        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        root.addWidget(self.train_log, 1)
        return page

    def _build_log_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("级别过滤:"))
        self.log_filter_info = QCheckBox("INFO")
        self.log_filter_warning = QCheckBox("WARNING")
        self.log_filter_error = QCheckBox("ERROR")
        for cb in (self.log_filter_info, self.log_filter_warning, self.log_filter_error):
            cb.setChecked(True)
            cb.toggled.connect(self._apply_log_filter)
            filter_row.addWidget(cb)
        filter_row.addStretch(1)
        lay.addLayout(filter_row)
        lay.addWidget(self.log_view, 1)
        return page

    @staticmethod
    def _find_weights() -> list:
        """收集可用的 .pt 权重文件"""
        names: list = []
        seen: set = set()
        roots = [_ROOT / "weights", _ROOT / "models", _ROOT]
        for root in roots:
            if not root.exists():
                continue
            for p in sorted(root.glob("*.pt")):
                if str(p) not in seen:
                    seen.add(str(p))
                    names.append(str(p))
        return names or ["yolo11n.pt"]

    # ======================== 信号连接 ========================

    def _connect_signals(self):
        c = self.controller
        c.frame_ready.connect(self._on_frame_ready)
        c.frame_info_ready.connect(self._on_frame_info)
        c.event_ready.connect(self._on_event_ready)
        c.decision_ready.connect(self._on_decision_ready)
        c.history_ready.connect(self._on_history_ready)
        c.status_ready.connect(self._on_status_ready)
        c.source_finished.connect(self._on_source_finished)
        c.train_log_ready.connect(self._on_train_log)
        c.train_finished.connect(self._on_train_finished)
        c.train_epoch_ready.connect(self._on_train_epoch)
        c.error_ready.connect(self._on_error)

        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.open_btn.clicked.connect(self._on_open)
        self.start_btn.clicked.connect(self._on_start)
        self.pause_btn.clicked.connect(self.controller.pause)
        self.resume_btn.clicked.connect(self.controller.resume)
        self.stop_btn.clicked.connect(self.controller.stop)
        self.train_start_btn.clicked.connect(self._on_train_start)
        self.train_stop_btn.clicked.connect(self.controller.stop_training)
        self.train_open_btn.clicked.connect(self._on_open_train_results)

        # 日志页轮询
        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._poll_log)
        self._log_timer.start(1000)

    def _apply_defaults(self):
        model_cfg = self.cfg.get("model", {})
        self.conf_spin.setValue(float(model_cfg.get("confidence", 0.5)))
        self.iou_spin.setValue(float(model_cfg.get("iou", 0.45)))
        # 默认数据源路径（便于快速演示）
        if (self.source_combo.currentData() == "image") and (_ROOT / "bus.jpg").exists():
            self.path_edit.setText(str(_ROOT / "bus.jpg"))
        elif (_ROOT / "me.mp4").exists():
            self.path_edit.setText(str(_ROOT / "me.mp4"))
        # 系统环境信息（右下角）
        info = self.controller.system_info()
        self.sysinfo_label.setText(
            f"GPU: {info['gpu']} | CUDA: {info['cuda']} | Torch: {info['torch']} | CV: {info['cv2']}")
        self.sysinfo_label.setToolTip(
            f"Python {info['python']} | Ultralytics {info['ultralytics']} | OpenCV {info['cv2']}")

    def _apply_recent_paths(self):
        """最近使用的路径（QSettings），启动时自动回填"""
        image = self._settings.value("recent/image", "")
        video = self._settings.value("recent/video", "")
        weights = self._settings.value("recent/weights", "")
        dataset = self._settings.value("recent/dataset", "")
        kind = self.source_combo.currentData()
        if kind == "image" and image and Path(image).exists():
            self.path_edit.setText(image)
        elif kind == "video" and video and Path(video).exists():
            self.path_edit.setText(video)
        if weights:
            self.weights_combo.setCurrentText(weights)
        if dataset and Path(dataset).exists():
            self.dataset_edit.setText(dataset)

    # ======================== 监控页信号处理 ========================

    def _on_frame_ready(self, payload):
        if not payload or not payload.get("image"):
            return
        self._last_qimage = payload["image"]
        self._update_video_label()

    def _update_video_label(self):
        if self._last_qimage is None:
            return
        pix = QPixmap.fromImage(self._last_qimage)
        size = self.video_label.size()
        if size.width() > 0 and size.height() > 0:
            pix = pix.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_video_label()

    def _on_frame_info(self, payload):
        table = self.frame_table
        if not payload:
            table.setRowCount(0)
            return
        classes = payload.get("classes") or []
        cls_text = "、".join(
            f"{c['class_name']}x{c['count']}({c['max_confidence']:.2f})" for c in classes) or "无"
        rows = [
            ("帧号", str(payload.get("frame_id", "-"))),
            ("时间(s)", str(payload.get("timestamp", "-"))),
            ("检测目标", str(payload.get("detection_count", 0))),
            ("火焰面积%", f"{payload.get('fire_area_ratio', 0)}%"),
            ("烟雾面积%", f"{payload.get('smoke_area_ratio', 0)}%"),
            ("推理耗时(ms)", str(payload.get("inference_time_ms", "-"))),
            ("类别", cls_text),
        ]
        table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(k))
            table.setItem(i, 1, QTableWidgetItem(v))

    def _on_event_ready(self, payload):
        table = self.event_table
        if not payload:
            table.setRowCount(0)
            return
        rows = [
            ("事件编号", str(payload.get("event_id", "-"))),
            ("状态", str(payload.get("status", "-"))),
            ("持续时间(s)", str(payload.get("duration", "-"))),
            ("增长趋势", str(payload.get("growth_trend", "-"))),
            ("危险等级", str(payload.get("danger_level", "-"))),
            ("火焰峰值%", f"{payload.get('fire_area', 0)}%"),
            ("烟雾峰值%", f"{payload.get('smoke_area', 0)}%"),
            ("阳性比例", str(payload.get("positive_ratio", "-"))),
        ]
        table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(k))
            item = QTableWidgetItem(v)
            if k == "危险等级":
                item.setForeground(QBrush(QColor(self._level_color(v))))
            table.setItem(i, 1, item)

    def _on_decision_ready(self, payload):
        if not payload:
            self.decision_text.setPlainText("")
            self.decision_meta.setText("来源: - | 更新: -")
            self.decision_level_label.setText("危险等级: -")
            self.decision_level_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #666666;")
            return
        reasons = "\n".join(f"- {r}" for r in payload.get("reasons", [])) or "无"
        suggestions = "\n".join(f"- {s}" for s in payload.get("suggestions", [])) or "无"
        text = (f"事件: {payload.get('event_id', '-')}\n"
                f"危险等级: {payload.get('danger_level', '-')}  |  "
                f"评分: {payload.get('score', 0)}  |  可信度: {payload.get('confidence', 0)}\n\n"
                f"原因:\n{reasons}\n\n建议:\n{suggestions}")
        if payload.get("summary"):
            text += f"\n\n摘要: {payload['summary']}"
        self.decision_text.setPlainText(text)
        level = payload.get("danger_level", "-")
        self.decision_level_label.setText(f"危险等级: {level}")
        self.decision_level_label.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {self._level_color(level)};")
        self.decision_meta.setText(
            f"来源: {payload.get('decision_source', '-')} | 更新: {payload.get('generated_at', '-')}")

    def _on_history_ready(self, items):
        table = self.history_table
        table.setRowCount(len(items))
        self._history_report_paths = {}
        for row, item in enumerate(items):
            table.setItem(row, 0, QTableWidgetItem(item.get("event_id", "")))
            table.setItem(row, 1, QTableWidgetItem(item.get("start_time", "")))
            table.setItem(row, 2, QTableWidgetItem(str(item.get("duration", ""))))
            danger_item = QTableWidgetItem(str(item.get("max_danger", "-")))
            danger_item.setForeground(QBrush(QColor(self._level_color(item.get("max_danger")))))
            table.setItem(row, 3, danger_item)
            table.setItem(row, 4, QTableWidgetItem(item.get("report_status", "")))
            self._history_report_paths[row] = item.get("report_path", "")

    def _on_status_ready(self, payload):
        total = payload.get("total_frames") or 0
        frame = payload.get("frame_id") or 0
        if total > 0 and frame > 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(min(frame / total * 100, 100)))
        else:
            self.progress.setValue(0)
        if self._error_state:
            state = "错误"
            self._set_status_light("error")
        elif payload.get("running"):
            state = "运行中"
            self._set_status_light("running")
        elif payload.get("paused"):
            state = "已暂停"
            self._set_status_light("warning")
        else:
            state = "就绪"
            self._set_status_light("ready")
        total_txt = total if total else "-"
        self.status_label.setText(
            f"{state} | 帧 {frame}/{total_txt} | "
            f"事件 确认{payload.get('events_confirmed', 0)}/结束{payload.get('events_ended', 0)}")
        self.fps_label.setText(f"FPS: {payload.get('fps', 0)}")
        infer = payload.get("inference_ms", "-")
        self.infer_label.setText(f"推理: {infer}ms" if infer != "-" else "推理: -")
        model = payload.get("model", "-")
        if model and model != "-":
            self.model_label.setText(f"模型: {model}")
        self._refresh_buttons()

    def _on_source_finished(self, payload):
        msg = (f"检测结束: {payload.get('source', '')}\n"
               f"处理帧数: {payload.get('processed_frames', 0)}\n"
               f"确认事件: {payload.get('events_confirmed', 0)} | "
               f"结束事件: {payload.get('events_ended', 0)}")
        self.status_label.setText(msg.replace("\n", " | "))
        QMessageBox.information(self, "检测结束", msg)

    def _refresh_buttons(self):
        running = self.controller.running
        paused = self.controller.paused
        is_sim = self.source_combo.currentData() == "simulator"
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running and not paused)
        self.resume_btn.setEnabled(running and paused)
        self.stop_btn.setEnabled(running)
        self.open_btn.setEnabled(not running)
        self.source_combo.setEnabled(not running)
        self.sim_group.setEnabled(is_sim and not running)

    def _on_error(self, message: str):
        """程序异常 → 状态灯 + 弹窗（GUI 异常提示）"""
        self._error_state = True
        self._set_status_light("error")
        self.status_label.setText(f"错误: {message}")
        QMessageBox.critical(self, "FireGuardian 错误", str(message))

    # ======================== 监控页操作 ========================

    def _on_source_changed(self):
        is_sim = self.source_combo.currentData() == "simulator"
        self.sim_group.setEnabled(is_sim and not self.controller.running)
        # 切换类型时给出默认路径
        if self.source_combo.currentData() == "image" and (_ROOT / "bus.jpg").exists():
            self.path_edit.setText(str(_ROOT / "bus.jpg"))
        elif self.source_combo.currentData() == "video" and (_ROOT / "me.mp4").exists():
            self.path_edit.setText(str(_ROOT / "me.mp4"))
        else:
            self.path_edit.clear()

    def _on_open(self):
        kind = self.source_combo.currentData()
        if kind == "simulator":
            d = QFileDialog.getExistingDirectory(self, "选择验证集图片目录")
            if d:
                self.path_edit.setText(d)
                self._settings.setValue("recent/simulator", d)
            return
        if kind == "image":
            f, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
        else:
            f, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "视频 (*.mp4 *.avi *.mov)")
        if f:
            self.path_edit.setText(f)
            self._settings.setValue(f"recent/{kind}", f)

    def _on_start(self):
        self._error_state = False
        self._set_status_light("ready")
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择数据源文件/目录")
            return
        params = {
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
        }
        if self.source_combo.currentData() == "simulator":
            params.update({
                "interval": self.interval_spin.value(),
                "mode": self.mode_combo.currentData(),
                "loop": self.loop_check.isChecked(),
                "max_frames": self.max_frames_spin.value() or None,
            })
        ok = self.controller.start(self.source_combo.currentData(), path, params)
        if not ok:
            QMessageBox.warning(self, "提示", "启动失败，请查看日志页")

    def _on_history_double_click(self, index):
        self._open_report_for_row(index.row())

    def _on_view_report(self):
        row = self.history_table.currentRow()
        if row >= 0:
            self._open_report_for_row(row)

    def _open_report_for_row(self, row: int):
        path = self._history_report_paths.get(row, "")
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.information(self, "提示", "该事件暂无报告")

    def _on_open_reports_dir(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.controller.reports_dir()))

    # ======================== 训练页操作 ========================

    def _on_train_start(self):
        weights = self.weights_combo.currentText().strip()
        dataset = self.dataset_edit.text().strip()
        device = self.device_combo.currentText().split(" ")[0]
        self.train_device_label.setText(f"Device: {device}")
        ok = self.controller.start_training(
            weights=weights,
            dataset_yaml=dataset,
            epochs=self.epochs_spin.value(),
            batch=self.batch_spin.value(),
            imgsz=int(self.imgsz_combo.currentData()),
            device=device,
            project=self.project_edit.text().strip() or "train",
            name=self.name_edit.text().strip() or "exp",
        )
        if not ok:
            QMessageBox.warning(self, "提示", "训练已在运行中")
        self._refresh_train_buttons()

    def _on_train_epoch(self, info):
        self.train_epoch_label.setText(f"Epoch: {info.get('epoch', '-')}/{info.get('epochs', '-')}")
        self.train_loss_label.setText(f"Loss: {info.get('loss', '-')}")
        self.train_map_label.setText(f"mAP50: {info.get('mAP50', '-')}")
        self.train_eta_label.setText(f"ETA: {info.get('eta_s', '-')}s")

    def _on_train_log(self, line: str):
        self.train_log.appendPlainText(line)
        # 限长：超过 1000 行时裁剪
        block_count = self.train_log.document().blockCount()
        if block_count > _MAX_LOG_LINES:
            cursor = self.train_log.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.Down,
                                cursor.MoveMode.KeepAnchor,
                                block_count - _MAX_LOG_LINES)
            cursor.removeSelectedText()

    def _on_train_finished(self, payload):
        if payload.get("stopped"):
            self.train_log.appendPlainText(f"[停止] {payload.get('error', '训练已停止')}")
        elif payload.get("success"):
            self.train_log.appendPlainText(
                f"[结果] 训练完成，最佳模型: {payload.get('best', '')}")
        else:
            self.train_log.appendPlainText(f"[失败] {payload.get('error', '未知错误')}")
        self._refresh_train_buttons()

    def _refresh_train_buttons(self):
        running = self.controller.training_running()
        self.train_start_btn.setEnabled(not running)
        self.train_stop_btn.setEnabled(running)

    def _on_open_train_results(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.controller.training_results_dir()))

    def _on_browse_weights(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "YOLO 权重 (*.pt)")
        if f:
            self.weights_combo.setCurrentText(f)
            self._settings.setValue("recent/weights", f)

    def _on_browse_dataset(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择数据集配置", "", "YAML (*.yaml *.yml)")
        if f:
            self.dataset_edit.setText(f)
            self._settings.setValue("recent/dataset", f)

    # ======================== 日志页 ========================

    def _poll_log(self):
        lines = self.controller.read_log_tail(_MAX_LOG_LINES)
        if lines == self._last_log_tail:
            return
        self._last_log_tail = lines
        self._apply_log_filter()

    def _apply_log_filter(self):
        if not hasattr(self, "log_view"):
            return
        levels = []
        if self.log_filter_info.isChecked():
            levels.append("INFO")
        if self.log_filter_warning.isChecked():
            levels.append("WARNING")
        if self.log_filter_error.isChecked():
            levels.append("ERROR")
        lines = [ln for ln in self._last_log_tail if any(lv in ln for lv in levels)]
        self.log_view.setPlainText("\n".join(lines))
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ======================== 关闭 ========================

    def closeEvent(self, event):
        self.controller.shutdown()
        super().closeEvent(event)


# ======================== 程序入口 ========================

def main():
    """启动 GUI"""
    from logs.logger import get_logger
    log = get_logger("Main")
    log.info("=" * 50)
    log.info("FireGuardian GUI 启动...")
    log.info("=" * 50)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())