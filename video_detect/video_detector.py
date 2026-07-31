"""
============================================================
M4 - 视频检测模块 (Video Detector)
功能：
  - 支持 mp4/avi/mov 格式视频检测
  - 逐帧 YOLO 推理，实时绘制检测框
  - 实时显示 FPS 和处理进度
  - 保存检测结果视频 + 检测日志

可独立运行: python video_detect/video_detector.py <视频路径>
============================================================
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ----- 环境修复 -----
_ROOT = Path(__file__).resolve().parent.parent
_ULTRA_CONFIG = _ROOT / ".venv" / "ultralytics_config"
_ULTRA_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

# ----- Ultralytics -----
from ultralytics import YOLO

# ----- 项目导入 -----
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import load_config, BoundingBox, Detection, ensure_dir


# ======================== 常量 ========================
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

# 模型查找优先级
_MODEL_CANDIDATES = [
    _ROOT / "models" / "best.pt",
    _ROOT / "yolo11n.pt",
]


# ======================== 数据类 ========================

@dataclass
class VideoDetectionResult:
    """视频检测结果"""
    video_path: str
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    fps: float = 0.0
    processing_time: float = 0.0
    fire_frames: int = 0
    smoke_frames: int = 0
    detections_per_frame: List[int] = field(default_factory=list)
    output_path: str = ""

    def print_summary(self):
        """打印检测概要"""
        sep = "=" * 50
        print(f"\n{sep}")
        print(f"  视频检测完成")
        print(f"{sep}")
        print(f"  视频: {Path(self.video_path).name}")
        print(f"  总帧数: {self.total_frames}")
        print(f"  处理帧: {self.processed_frames}")
        print(f"  检测目标: {self.total_detections} 个")
        print(f"  含火焰帧: {self.fire_frames}")
        print(f"  含烟雾帧: {self.smoke_frames}")
        print(f"  处理速度: {self.fps:.1f} FPS")
        print(f"  处理耗时: {self.processing_time:.1f}s")
        print(f"  输出视频: {Path(self.output_path).name}")
        print(f"{sep}")


# ======================== 检测器类 ========================

class VideoDetector:
    """视频检测器"""

    def __init__(self, model_path: Optional[str | Path] = None,
                 conf_threshold: Optional[float] = None,
                 iou_threshold: Optional[float] = None,
                 frame_skip: int = 1):
        """
        初始化视频检测器

        参数:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU 阈值
            frame_skip: 跳帧检测（1=逐帧，2=隔帧）
        """
        cfg = load_config()
        model_cfg = cfg.get("model", {})
        video_cfg = cfg.get("video_detect", {})

        # 模型路径
        if model_path is None:
            model_path = self._find_model()
        self.model_path = Path(model_path)

        self.conf_threshold = conf_threshold if conf_threshold is not None else model_cfg.get("confidence", 0.5)
        self.iou_threshold = iou_threshold if iou_threshold is not None else model_cfg.get("iou", 0.45)
        self.device = model_cfg.get("device", 0)
        self.frame_skip = frame_skip
        self.show_fps = video_cfg.get("show_fps", True)

        # 加载模型
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        print(f"  加载模型: {self.model_path.name}...")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        print(f"  模型就绪: {self.model.task}, {len(self.class_names)} 类")

        # 输出配置
        self.output_dir = ensure_dir(str(_ROOT / video_cfg.get("output_dir", "runs/video_detect")))

    def _find_model(self) -> Path:
        for p in _MODEL_CANDIDATES:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"未找到模型文件。请将模型放置在:\n"
            f"  {_MODEL_CANDIDATES[0]} 或 {_MODEL_CANDIDATES[1]}"
        )

    def detect(self, video_path: str | Path) -> VideoDetectionResult:
        """
        检测视频中的目标

        参数:
            video_path: 视频文件路径

        返回:
            VideoDetectionResult 对象
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        if video_path.suffix.lower() not in _VIDEO_EXTENSIONS:
            raise ValueError(f"不支持的视频格式: {video_path.suffix}")

        import cv2

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        # 视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 输出视频 Writer
        output_name = f"{video_path.stem}_detect.mp4"
        output_path = self.output_dir / output_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps // self.frame_skip, (orig_w, orig_h))

        result = VideoDetectionResult(
            video_path=str(video_path),
            total_frames=total_frames,
            output_path=str(output_path),
        )

        print(f"\n  视频: {video_path.name}  ({orig_w}x{orig_h}, {fps:.0f}FPS, {total_frames}帧, {total_frames/fps:.1f}s)")
        print(f"  跳帧: {self.frame_skip} | 阈值: conf={self.conf_threshold}")
        print(f"  输出: {output_path.name}")
        print(f"  {'-' * 45}")

        frame_idx = 0
        processed = 0
        t_start = time.time()
        progress_interval = max(1, total_frames // 20)  # 每 5% 显示一次进度

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 跳帧
            if (frame_idx - 1) % self.frame_skip != 0:
                continue

            # YOLO 检测
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            result_data = results[0]

            processed += 1
            boxes_data = result_data.boxes
            frame_det_count = 0
            has_fire = False
            has_smoke = False

            if boxes_data is not None and len(boxes_data) > 0:
                xyxy = boxes_data.xyxy.cpu().numpy()
                confs = boxes_data.conf.cpu().numpy()
                cls_ids = boxes_data.cls.cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    cid = int(cls_ids[i])
                    conf = float(confs[i])
                    class_name = self.class_names.get(cid, f"class_{cid}")

                    frame_det_count += 1

                    # 绘制检测框
                    color = self._get_color(cid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 帧信息叠加
            elapsed = time.time() - t_start
            current_fps = processed / elapsed if elapsed > 0 else 0
            progress = frame_idx / total_frames * 100

            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Detections: {frame_det_count}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {current_fps:.1f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 进度显示
            if processed % progress_interval == 0 or frame_idx == total_frames:
                eta = (elapsed / processed) * (total_frames // self.frame_skip - processed) if processed > 0 else 0
                print(f"  [{progress:5.1f}%] 帧 {frame_idx}/{total_frames} | 目标 {frame_det_count} | "
                      f"{current_fps:.1f} FPS | ETA {eta:.0f}s")

            # 写入输出
            writer.write(frame)

            # 统计
            result.total_detections += frame_det_count
            result.detections_per_frame.append(frame_det_count)
            if has_fire:
                result.fire_frames += 1
            if has_smoke:
                result.smoke_frames += 1

        # 完成
        cap.release()
        writer.release()

        result.processed_frames = processed
        result.processing_time = time.time() - t_start
        result.fps = processed / result.processing_time if result.processing_time > 0 else 0

        return result

    def decode(self, video_path: str | Path):
        """
        逐帧检测生成器（不写输出视频，供 GUI 实时显示）

        参数:
            video_path: 视频文件路径

        产出:
            (frame, detection, frame_idx, total_frames)
              frame: BGR ndarray（未绘制）
              detection: Detection（含 bboxes / 面积 / 置信度）
              frame_idx: 1-based 帧号（含跳帧）
              total_frames: 视频总帧数
        """
        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        if video_path.suffix.lower() not in _VIDEO_EXTENSIONS:
            raise ValueError(f"不支持的视频格式: {video_path.suffix}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                # 跳帧
                if (frame_idx - 1) % self.frame_skip != 0:
                    continue

                h, w = frame.shape[:2]
                t0 = time.time()
                results = self.model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                )
                infer_ms = (time.time() - t0) * 1000
                result_data = results[0]

                bboxes = []
                boxes_data = result_data.boxes
                if boxes_data is not None and len(boxes_data) > 0:
                    xyxy = boxes_data.xyxy.cpu().numpy()
                    confs = boxes_data.conf.cpu().numpy()
                    cls_ids = boxes_data.cls.cpu().numpy().astype(int)
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        cid = int(cls_ids[i])
                        bboxes.append(BoundingBox(
                            x1=float(x1), y1=float(y1),
                            x2=float(x2), y2=float(y2),
                            confidence=float(confs[i]),
                            class_id=cid,
                            class_name=self.class_names.get(cid, f"class_{cid}"),
                        ))

                detection = Detection(
                    bboxes=bboxes,
                    image_path=str(video_path),
                    timestamp=(frame_idx / fps) if fps > 0 else float(frame_idx),
                    wall_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                    inference_time_ms=infer_ms,
                    image_width=w,
                    image_height=h,
                    frame_id=frame_idx,
                )
                yield frame, detection, frame_idx, total_frames
        finally:
            cap.release()

    def _get_color(self, class_id: int) -> tuple:
        """根据类别返回检测框颜色"""
        colors = {
            0: (0, 0, 255),    # fire → 红色
            1: (0, 165, 255),  # smoke → 橙色
        }
        return colors.get(class_id, (0, 255, 0))  # 默认绿色

    def detect_fire_optimized(self, video_path: str | Path) -> VideoDetectionResult:
        """
        针对火灾检测优化的模式
        - 使用更低置信度阈值
        - 关注 class 0 (fire) 和 class 1 (smoke)
        """
        return self.detect(video_path)


# ======================== 命令行入口 ========================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="FireGuardian M4 - 视频检测")
    parser.add_argument("input", nargs="?", default=None,
                        help="视频路径（默认: 项目中的 me.mp4）")
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值")
    parser.add_argument("--skip", type=int, default=1, help="跳帧检测 (1=逐帧)")
    parser.add_argument("--no-output", action="store_true", help="不保存输出视频")

    args = parser.parse_args()

    if args.input is None:
        default_video = _ROOT / "me.mp4"
        if default_video.exists():
            args.input = str(default_video)
        else:
            print("请指定视频路径")
            return 1

    detector = VideoDetector(conf_threshold=args.conf, frame_skip=args.skip)
    result = detector.detect(args.input)
    result.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
