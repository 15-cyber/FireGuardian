"""
============================================================
M3 - 图片检测模块 (Image Detector)
功能：
  - 支持 jpg/png/bmp/jpeg 格式图片检测
  - YOLO 模型推理，输出检测框、类别、置信度
  - 绘制检测结果并保存标注图
  - 支持单张检测和批量检测（遍历文件夹）

可独立运行: python detect/image_detector.py [图片路径]
============================================================
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ----- 环境修复：在导入 ultralytics 前设置 -----
_ROOT = Path(__file__).resolve().parent.parent
_ULTRA_CONFIG = _ROOT / ".venv" / "ultralytics_config"
_ULTRA_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

# ----- Ultralytics 导入 -----
from ultralytics import YOLO

# ----- 项目内部导入 -----
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    BoundingBox,
    Detection,
    ensure_dir,
)


# ======================== 常量 ========================
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
_DEFAULT_CONFIDENCE = 0.5
_DEFAULT_IOU = 0.45

# 模型查找优先级
_MODEL_CANDIDATES = [
    _ROOT / "models" / "best.pt",
    _ROOT / "yolo11n.pt",
]


# ======================== 检测器类 ========================

class ImageDetector:
    """图片检测器"""

    def __init__(self, model_path: Optional[str | Path] = None,
                 conf_threshold: Optional[float] = None,
                 iou_threshold: Optional[float] = None):
        """
        初始化检测器

        参数:
            model_path: 模型路径，None 则自动查找 (models/best.pt → yolo11n.pt)
            conf_threshold: 置信度阈值，None 则从 config.yaml 读取
            iou_threshold: IoU 阈值，None 则从 config.yaml 读取
        """
        cfg = load_config()
        model_cfg = cfg.get("model", {})

        # 确定模型路径
        if model_path is None:
            model_path = self._find_model()
        self.model_path = Path(model_path)

        # 阈值
        self.conf_threshold = conf_threshold if conf_threshold is not None else model_cfg.get("confidence", _DEFAULT_CONFIDENCE)
        self.iou_threshold = iou_threshold if iou_threshold is not None else model_cfg.get("iou", _DEFAULT_IOU)
        self.device = model_cfg.get("device", 0)

        # 加载模型
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        print(f"  加载模型: {self.model_path.name}...")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        print(f"  模型就绪: {self.model.task}, {len(self.class_names)} 类")

        # 检测统计
        self.total_detections = 0
        self.total_inference_time = 0.0

    def _find_model(self) -> Path:
        """按优先级查找可用的模型文件"""
        for p in _MODEL_CANDIDATES:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"未找到模型文件。请先将模型放置在以下位置之一:\n"
            f"  {_MODEL_CANDIDATES[0]} (训练后保存)\n"
            f"  {_MODEL_CANDIDATES[1]} (预训练模型)"
        )

    def detect_single(self, image_path: str | Path,
                      save_result: bool = True,
                      output_dir: Optional[str | Path] = None) -> Detection:
        """
        检测单张图片

        参数:
            image_path: 图片路径
            save_result: 是否保存标注图
            output_dir: 输出目录，None 则使用 config 中的默认路径

        返回:
            Detection 对象，包含所有检测框和元数据
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 推理
        t0 = time.time()
        results = self.model(
            str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        elapsed = time.time() - t0
        self.total_inference_time += elapsed
        result = results[0]

        # 解析检测结果
        boxes_data = result.boxes
        bboxes: List[BoundingBox] = []

        if boxes_data is not None and len(boxes_data) > 0:
            xyxy = boxes_data.xyxy.cpu().numpy()
            confs = boxes_data.conf.cpu().numpy()
            cls_ids = boxes_data.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                cid = int(cls_ids[i])
                conf = float(confs[i])
                class_name = self.class_names.get(cid, f"class_{cid}")
                bboxes.append(BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=conf,
                    class_id=cid,
                    class_name=class_name,
                ))

        detection = Detection(
            bboxes=bboxes,
            image_path=str(image_path),
        )
        self.total_detections += len(bboxes)

        # 打印结果
        n = len(bboxes)
        print(f"  {image_path.name}: 检测到 {n} 个目标 ({elapsed*1000:.0f}ms)")

        for box in bboxes:
            print(f"    [{box.class_name}] conf={box.confidence:.3f} "
                  f"({box.x1:.0f},{box.y1:.0f} → {box.x2:.0f},{box.y2:.0f})")

        # 保存标注图
        if save_result:
            output_dir = self._get_output_dir(output_dir)
            self._save_annotated(result, image_path, output_dir)

        return detection

    def detect_batch(self, input_path: str | Path,
                     save_result: bool = True,
                     output_dir: Optional[str | Path] = None) -> List[Detection]:
        """
        批量检测（输入为文件夹时遍历所有图片）

        参数:
            input_path: 图片路径或文件夹路径
            save_result: 是否保存标注图
            output_dir: 输出目录

        返回:
            Detection 列表
        """
        input_path = Path(input_path)

        if input_path.is_file():
            return [self.detect_single(input_path, save_result, output_dir)]

        if not input_path.is_dir():
            raise NotADirectoryError(f"路径不是文件也不是目录: {input_path}")

        # 收集所有图片
        image_files = sorted([
            f for f in input_path.iterdir()
            if f.suffix.lower() in _IMAGE_EXTENSIONS
        ])

        if not image_files:
            print(f"  {input_path} 中没有图片文件")
            return []

        print(f"\n  批量检测: {len(image_files)} 张图片\n")
        all_detections = []
        for img_path in image_files:
            det = self.detect_single(img_path, save_result, output_dir)
            all_detections.append(det)

        # 汇总
        fire_count = sum(1 for d in all_detections for b in d.bboxes if b.class_name == "fire")
        smoke_count = sum(1 for d in all_detections for b in d.bboxes if b.class_name == "smoke")
        total_time = self.total_inference_time

        print(f"\n  {'=' * 45}")
        print(f"  批量检测完成")
        print(f"  图片: {len(all_detections)} 张")
        print(f"  火焰: {fire_count} 处 | 烟雾: {smoke_count} 处")
        print(f"  总耗时: {total_time:.2f}s | 平均: {total_time/max(len(all_detections),1)*1000:.0f}ms/张")
        print(f"  {'=' * 45}")

        return all_detections

    def _get_output_dir(self, output_dir: Optional[str | Path] = None) -> Path:
        """获取并确保输出目录存在"""
        if output_dir is None:
            cfg = load_config()
            output_dir = cfg.get("detect", {}).get("output_dir", "runs/detect")
        out = _ROOT / str(output_dir)
        ensure_dir(str(out))
        return out

    def _save_annotated(self, result, image_path: Path, output_dir: Path):
        """保存标注后的图片"""
        # ultralytics 的结果可以直接保存
        save_name = f"{image_path.stem}_detect{image_path.suffix}"
        save_path = str(output_dir / save_name)
        # 使用 ultralytics 内置的 save 方法
        result.save(filename=save_path)
        print(f"  标注图已保存: {output_dir.name}/{save_name}")

    def print_stats(self):
        """打印检测器统计信息"""
        print(f"\n  [检测器统计]")
        print(f"  模型: {self.model_path.name}")
        print(f"  总检测数: {self.total_detections}")
        print(f"  总推理时间: {self.total_inference_time:.2f}s")


# ======================== 命令行接口 ========================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="FireGuardian M3 - 图片检测")
    parser.add_argument("input", nargs="?", default=None,
                        help="图片路径或文件夹路径（默认: 项目中的 bus.jpg）")
    parser.add_argument("--conf", type=float, default=None,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=None,
                        help="IoU 阈值")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存标注图")

    args = parser.parse_args()

    # 默认图片
    if args.input is None:
        default_img = _ROOT / "bus.jpg"
        if default_img.exists():
            args.input = str(default_img)
        else:
            print("请指定图片路径，或确保项目目录下有 bus.jpg")
            return 1

    # 创建检测器
    try:
        detector = ImageDetector(
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    # 执行检测
    input_path = Path(args.input)
    if input_path.is_dir():
        detector.detect_batch(input_path, save_result=not args.no_save, output_dir=args.output)
    else:
        detector.detect_single(input_path, save_result=not args.no_save, output_dir=args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
