"""
============================================================
M5 - Validation Stream Simulator（验证集监控流模拟器）
功能：
  - 将验证集离散图片封装成连续帧流，模拟监控输入
  - 状态机控制：IDLE → RUNNING → PAUSED → ... → COMPLETED
  - 通过依赖注入复用已有 YOLO Detector
  - 每帧输出统一 FrameData + Detection 对象
  - 回调机制供 GUI 和 M6 消费
  - 损坏图片自动跳过，不中断流程

改进说明文档: FireGuardian Module 5 改进说明.docx
可独立运行: python simulator/validator_simulator.py
============================================================
"""
from __future__ import annotations

import os
import sys
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

# ----- 环境修复 -----
_ROOT = Path(__file__).resolve().parent.parent
_ULTRA_CONFIG = _ROOT / ".venv" / "ultralytics_config"
_ULTRA_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

# ----- 项目导入 -----
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    BoundingBox,
    Detection,
    FrameData,
    SimulatorStats,
    ensure_dir,
)


# ======================== 状态枚举 ========================

class SimulatorState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


# ======================== 回调类型别名 ========================

# on_frame(frame_data: FrameData, detection: Detection) -> None
FrameCallback = Callable[["FrameData", "Detection"], None]

# on_progress(current: int, total: int, percent: float) -> None
ProgressCallback = Callable[[int, int, float], None]

# on_error(frame_id: int, image_path: str, error: str) -> None
ErrorCallback = Callable[[int, str, str], None]

# on_complete(stats: SimulatorStats) -> None
CompleteCallback = Callable[["SimulatorStats"], None]


def _cv_read_image(path) -> object:
    """兼容中文路径的图片读取（np.fromfile + imdecode），失败返回 None"""
    import cv2
    import numpy as np
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ======================== 模拟器类 ========================

class ValidationStreamSimulator:
    """
    验证集监控流模拟器

    将 valid/images 中的图片按照可配置的时间间隔逐张输出，
    模拟真实监控摄像头输入。

    使用方式:
        simulator = ValidationStreamSimulator(detector=my_detector)
        simulator.on_frame = lambda fd, det: print(f"Frame {fd.frame_id}")
        simulator.start()
    """

    def __init__(
        self,
        image_dir: Optional[str | Path] = None,
        detector: Optional[Callable] = None,
        interval_seconds: Optional[float] = None,
        mode: Optional[str] = None,
        loop: Optional[bool] = None,
        max_frames: Optional[int] = None,
        config_path: Optional[str | Path] = None,
    ):
        """
        初始化模拟器

        参数:
            image_dir: 图片目录路径（默认从 config.yaml 读取）
            detector: 检测函数 callable(image) -> List[BoundingBox]
                      如果为 None，使用 M3 ImageDetector
            interval_seconds: 帧间隔（秒）
            mode: 播放模式 "sequential" / "random"
            loop: 是否循环
            max_frames: 最大处理帧数
            config_path: 配置文件路径
        """
        cfg = load_config()
        sim_cfg = cfg.get("simulator", {})

        # 图片目录
        self.image_dir = Path(image_dir) if image_dir else Path(sim_cfg.get("dataset_path", ""))
        if not self.image_dir.exists():
            # 尝试从数据集路径推导
            fallback = Path(cfg["dataset"]["path"]) / "valid" / "images"
            if fallback.exists():
                self.image_dir = fallback

        # 播放参数
        self.interval_seconds = interval_seconds if interval_seconds is not None else sim_cfg.get("interval_seconds", 0.5)
        self.mode = mode if mode is not None else sim_cfg.get("mode", "sequential")
        self.loop = loop if loop is not None else sim_cfg.get("loop", False)
        self.max_frames = max_frames if max_frames is not None else sim_cfg.get("max_frames", None)
        self.extensions = set(sim_cfg.get("supported_extensions", [".jpg", ".jpeg", ".png", ".bmp"]))
        self.output_dir = ensure_dir(str(_ROOT / sim_cfg.get("output_dir", "runs/simulator")))

        # 检测器（依赖注入）
        self.detector = detector

        # 状态
        self.state = SimulatorState.IDLE
        self._frame_id = 0
        self._current_index = 0
        self._image_list: List[Path] = []
        self._pause_event = threading.Event() if "threading" in sys.modules else None

        # 统计
        self.stats = SimulatorStats()

        # 回调（外部注册）
        self.on_frame: Optional[FrameCallback] = None
        self.on_progress: Optional[ProgressCallback] = None
        self.on_error: Optional[ErrorCallback] = None
        self.on_complete: Optional[CompleteCallback] = None

        # 内部状态
        self._wall_start_time: float = 0.0
        self._inference_times: List[float] = []

    def _build_image_list(self) -> List[Path]:
        """扫描图片目录，返回排序后的图片路径列表"""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.image_dir}")

        images = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in self.extensions and f.is_file()
        ])
        return images

    # ======================== 播放控制接口 ========================

    def start(self):
        """
        开始模拟播放（阻塞式）
        在独立线程中调用，或直接作为生成器使用
        """
        if self.state == SimulatorState.RUNNING:
            return

        # 构建图片列表
        self._image_list = self._build_image_list()
        if not self._image_list:
            print("  [WARN] 图片目录为空，无图可播")
            return

        total = len(self._image_list)
        if self.max_frames:
            total = min(total, self.max_frames)

        self.stats = SimulatorStats(total_frames=total)
        self.state = SimulatorState.RUNNING
        self._frame_id = 0
        self._current_index = 0
        self._wall_start_time = time.time()
        self._inference_times = []

        print(f"\n  Validation Simulator 启动")
        print(f"  图片目录: {self.image_dir}")
        print(f"  图片总数: {len(self._image_list)}")
        print(f"  播放模式: {self.mode}  |  间隔: {self.interval_seconds}s  |  循环: {self.loop}")
        print(f"  {'=' * 45}")

        # 主循环
        self._run_loop()

    def pause(self):
        """暂停播放"""
        if self.state == SimulatorState.RUNNING:
            self.state = SimulatorState.PAUSED
            print("  [Simulator] PAUSED")

    def resume(self):
        """继续播放"""
        if self.state == SimulatorState.PAUSED:
            self.state = SimulatorState.RUNNING
            print("  [Simulator] RESUMED")

    def stop(self):
        """停止播放"""
        if self.state in (SimulatorState.RUNNING, SimulatorState.PAUSED):
            self.state = SimulatorState.STOPPED
            print("  [Simulator] STOPPED")

    def next_frame(self):
        """
        单步执行一帧（用于调试 / GUI 单步模式）
        返回 (FrameData, Detection) 或 None（无更多帧）
        """
        if self.state == SimulatorState.COMPLETED:
            return None

        if not self._image_list:
            self._image_list = self._build_image_list()

        # 选择图片
        img_path = self._select_next_image()
        if img_path is None:
            self.state = SimulatorState.COMPLETED
            self._finalize()
            return None

        frame_id = self._frame_id
        self._frame_id += 1

        # 创建 FrameData 并检测
        try:
            frame_data, detection = self._process_frame(frame_id, img_path)
            return frame_data, detection
        except Exception as e:
            self.stats.failed_frames += 1
            if self.on_error:
                self.on_error(frame_id, str(img_path), str(e))
            return None

    def reset(self):
        """重置模拟器到初始状态"""
        self.state = SimulatorState.IDLE
        self._frame_id = 0
        self._current_index = 0
        self._image_list = []
        self.stats = SimulatorStats()
        self._inference_times = []
        print("  [Simulator] RESET")

    def get_statistics(self) -> dict:
        """获取运行统计摘要"""
        return self.stats.to_dict()

    # ======================== 内部方法 ========================

    def _run_loop(self):
        """主运行循环"""
        import cv2

        while self.state in (SimulatorState.RUNNING, SimulatorState.PAUSED):
            # 暂停等待
            if self.state == SimulatorState.PAUSED:
                time.sleep(0.1)
                continue

            # 选择下一张图片
            img_path = self._select_next_image()
            if img_path is None:
                self.state = SimulatorState.COMPLETED
                break

            frame_id = self._frame_id
            self._frame_id += 1

            # 处理帧
            try:
                frame_data, detection = self._process_frame(frame_id, img_path)
                self.stats.processed_frames += 1
                if detection.has_fire:
                    self.stats.fire_frames += 1
                if detection.has_smoke:
                    self.stats.smoke_frames += 1
                if not detection.has_fire and not detection.has_smoke:
                    self.stats.normal_frames += 1

                # 回调通知
                if self.on_frame:
                    self.on_frame(frame_data, detection)

            except Exception as e:
                self.stats.failed_frames += 1
                if self.on_error:
                    self.on_error(frame_id, str(img_path), str(e))
                # 跳过损坏图片，继续下一帧
                continue

            # 进度通知
            total = self.stats.total_frames
            if self.on_progress and total > 0:
                self.on_progress(frame_id, total, frame_id / total * 100)

            # 等待间隔（补偿推理时间）
            self._wait_interval()

            # 检查是否达到最大帧数
            if self.max_frames and self._frame_id >= self.max_frames:
                self.state = SimulatorState.COMPLETED
                break

        # 完成
        if self.state == SimulatorState.COMPLETED:
            self._finalize()

    def _select_next_image(self) -> Optional[Path]:
        """根据模式选择下一张图片"""
        images = self._image_list
        if not images:
            return None

        if self.mode == "random":
            idx = random.randint(0, len(images) - 1)
            self._current_index = idx
        else:
            # sequential
            if self._current_index >= len(images):
                if self.loop:
                    self._current_index = 0
                else:
                    return None
            idx = self._current_index
            self._current_index += 1

        return images[idx]

    def _process_frame(self, frame_id: int, img_path: Path) -> tuple:
        """
        处理一帧：读取图片 → YOLO检测 → 转换为统一格式
        返回 (FrameData, Detection)
        """
        import cv2

        # 读取图片（兼容中文路径）
        img = _cv_read_image(str(img_path))
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        h, w = img.shape[:2]

        # 模拟时间戳
        sim_timestamp = frame_id * self.interval_seconds

        # 创建 FrameData
        frame_data = FrameData(
            frame_id=frame_id,
            timestamp=sim_timestamp,
            image=img,
            image_path=str(img_path),
            source_type="validation_simulator",
            source_name=self.image_dir.parent.name,  # 数据集 split 名称
        )

        # YOLO 检测（通过依赖注入的 detector）
        t0 = time.time()
        if self.detector is not None:
            bboxes = self.detector(img)
        else:
            # 没有注入 detector 则返回空结果
            bboxes = []
        t_infer = (time.time() - t0) * 1000
        self._inference_times.append(t_infer)

        # 创建 Detection
        detection = Detection(
            bboxes=bboxes,
            image_path=str(img_path),
            timestamp=sim_timestamp,
            wall_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            inference_time_ms=t_infer,
            image_width=w,
            image_height=h,
        )

        return frame_data, detection

    def _wait_interval(self):
        """等待指定间隔（补偿推理时间）"""
        elapsed = self._inference_times[-1] / 1000 if self._inference_times else 0
        wait = max(0, self.interval_seconds - elapsed)
        if wait > 0:
            time.sleep(wait)

    def _finalize(self):
        """模拟结束，计算统计并回调"""
        elapsed = time.time() - self._wall_start_time if self._wall_start_time > 0 else 0
        infer_times = self._inference_times

        self.stats.total_simulated_duration = self._frame_id * self.interval_seconds
        self.stats.avg_inference_time_ms = sum(infer_times) / len(infer_times) if infer_times else 0.0
        self.stats.processed_frames = self._frame_id - self.stats.failed_frames

        # 打印摘要
        self._print_summary()

        # 回调
        if self.on_complete:
            self.on_complete(self.stats)

    def _print_summary(self):
        """打印运行统计"""
        s = self.stats
        sep = "=" * 50
        print(f"\n{sep}")
        print(f"  Validation Simulator 完成")
        print(f"{sep}")
        print(f"  总帧数:     {s.total_frames:>6}")
        print(f"  已处理:     {s.processed_frames:>6}")
        print(f"  失败帧:     {s.failed_frames:>6}")
        print(f"  含火焰帧:   {s.fire_frames:>6}")
        print(f"  含烟雾帧:   {s.smoke_frames:>6}")
        print(f"  正常帧:     {s.normal_frames:>6}")
        print(f"  平均推理:   {s.avg_inference_time_ms:.1f} ms")
        print(f"  模拟时长:   {s.total_simulated_duration:.1f}s")
        print(f"{sep}")


# ======================== 命令行入口 ========================

def _create_default_detector():
    """创建默认的 YOLO 检测器"""
    from ultralytics import YOLO

    # 查找模型
    candidates = [
        _ROOT / "models" / "best.pt",
        _ROOT / "yolo11n.pt",
    ]
    model_path = None
    for p in candidates:
        if p.exists():
            model_path = p
            break

    if not model_path:
        print("  [WARN] 未找到模型文件，使用空检测器")
        return lambda img: []

    print(f"  加载模型: {model_path.name}...")
    model = YOLO(str(model_path))
    conf = load_config().get("model", {}).get("confidence", 0.5)

    def detect_fn(image):
        results = model(image, conf=conf, verbose=False)
        r = results[0]
        bboxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cid = int(cls_ids[i])
                bboxes.append(BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=float(confs[i]),
                    class_id=cid,
                    class_name=names.get(cid, f"class_{cid}"),
                ))
        return bboxes

    return detect_fn


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="FireGuardian M5 - Validation Simulator")
    parser.add_argument("--mode", choices=["sequential", "random"], default=None)
    parser.add_argument("--interval", type=float, default=None, help="帧间隔（秒）")
    parser.add_argument("--max", type=int, default=10, help="最大处理帧数（默认10，全量指定-1）")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--dir", type=str, default=None, help="图片目录路径")

    args = parser.parse_args()

    # 创建检测器
    detector = _create_default_detector()

    # 创建模拟器
    sim = ValidationStreamSimulator(
        image_dir=args.dir,
        detector=detector,
        interval_seconds=args.interval,
        mode=args.mode,
        loop=args.loop,
        max_frames=None if args.max == -1 else args.max,
    )

    # 注册进度回调
    def on_progress(current, total, percent):
        if current % max(1, total // 20) == 0 or current == total:
            print(f"  [{percent:5.1f}%] 帧 {current}/{total}")

    sim.on_progress = on_progress

    # 开始
    sim.start()
    return 0


if __name__ == "__main__":
    # threading 仅在命令行运行时需要
    import threading
    sys.exit(main())
