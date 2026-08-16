"""
============================================================
FireGuardian V2 Phase 5-2 - 困难负样本抽取

从本地素材（烟花1/烟花2/me.mp4）中用当前模型提取
「被误检为 fire/smoke 的困难负样本」：
  - 检测到 fire 或 smoke 且 conf >= 阈值 的帧
  - 保存原图 + 空标签（YOLO 负样本）
  - 只读输入，不修改原视频/模型

输出：datasets/v2_negatives/fireworks_XX/frame_*.jpg + *.txt(空)
============================================================
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

import cv2
import torch

from video_detect.video_detector import VideoDetector


OUT_DIR = _ROOT / "datasets" / "v2_negatives"
VIDEO_DIR = _ROOT / "测试视频"
EXTRA_VIDEOS = [_ROOT / "me.mp4"]
CONF_THRESHOLD = 0.35
FRAME_SKIP = 6
MAX_PER_VIDEO = 400
MIN_GAP = 3  # 相邻选取帧最小间隔，避免连续帧冗余


def _collect_frames(path: Path, detector: VideoDetector) -> list:
    frames = []
    last_picked = -10**9
    for frame, detection, frame_idx, total in detector.decode(str(path)):
        fire_confs = [b.confidence for b in detection.fire_boxes]
        smoke_confs = [b.confidence for b in detection.smoke_boxes]
        best = max(fire_confs + smoke_confs, default=0.0)
        if best >= CONF_THRESHOLD and frame_idx - last_picked >= MIN_GAP:
            frames.append((frame_idx, frame))
            last_picked = frame_idx
        if len(frames) >= MAX_PER_VIDEO:
            break
    return frames


def main() -> int:
    videos = []
    if VIDEO_DIR.exists():
        videos = sorted(
            p for p in VIDEO_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in (".mp4", ".avi", ".mov")
        )
    videos = (videos + [p for p in EXTRA_VIDEOS if p.exists()])[:6]
    if not videos:
        print("未找到可用的视频素材")
        return 1

    detector = VideoDetector(frame_skip=FRAME_SKIP)
    if not torch.cuda.is_available():
        detector.device = "cpu"
        print("CUDA 不可用，使用 CPU 推理")

    total_saved = 0
    for video in videos:
        stem = video.stem
        safe = "".join(ch if ch.isalnum() else "_" for ch in stem)
        out_dir = OUT_DIR / f"fw_{safe}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"处理 {video.name} ...", flush=True)
        picked = _collect_frames(video, detector)
        saved = 0
        for idx, frame in picked:
            img_path = out_dir / f"frame_{idx:06d}.jpg"
            lbl_path = img_path.with_suffix(".txt")
            cv2.imwrite(str(img_path), frame)
            lbl_path.write_text("", encoding="utf-8")  # 空标签 = 负样本
            saved += 1
        total_saved += saved
        print(f"  {video.name}: 保存困难负样本 {saved} 张 -> {out_dir}")

    print(f"合计保存困难负样本: {total_saved} 张 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
