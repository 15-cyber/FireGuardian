"""
============================================================
FireGuardian 共享工具模块
包含：数据类定义、配置加载、公共函数
可独立运行测试: python utils/common.py
============================================================
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

import yaml


# ======================== 配置加载 ========================

_CONFIG_CACHE: dict | None = None


def load_config(reload: bool = False) -> dict:
    """加载全局配置文件（带缓存）"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or reload:
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


# ======================== 数据类定义 ========================

@dataclass
class BoundingBox:
    """检测框"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """检测框像素面积"""
        return self.width * self.height

    def area_ratio(self, image_area: float) -> float:
        """检测框占图片面积的比例"""
        if image_area <= 0:
            return 0.0
        return self.area / image_area


@dataclass
class Detection:
    """单帧检测结果"""
    bboxes: List[BoundingBox] = field(default_factory=list)
    image_path: str = ""
    timestamp: float = 0.0              # 模拟时间戳（秒）
    wall_time: str = ""                  # 真实运行时间
    inference_time_ms: float = 0.0       # 推理耗时（毫秒）
    image_width: int = 0
    image_height: int = 0

    @property
    def fire_boxes(self) -> List[BoundingBox]:
        return [b for b in self.bboxes if b.class_name == "fire"]

    @property
    def smoke_boxes(self) -> List[BoundingBox]:
        return [b for b in self.bboxes if b.class_name == "smoke"]

    @property
    def fire_area(self) -> float:
        return sum(b.area for b in self.fire_boxes)

    @property
    def smoke_area(self) -> float:
        return sum(b.area for b in self.smoke_boxes)

    @property
    def fire_area_ratio(self) -> float:
        img_area = self.image_width * self.image_height
        if img_area <= 0:
            return 0.0
        return self.fire_area / img_area

    @property
    def smoke_area_ratio(self) -> float:
        img_area = self.image_width * self.image_height
        if img_area <= 0:
            return 0.0
        return self.smoke_area / img_area

    @property
    def has_fire(self) -> bool:
        return len(self.fire_boxes) > 0

    @property
    def has_smoke(self) -> bool:
        return len(self.smoke_boxes) > 0

    @property
    def max_confidence(self) -> float:
        return max((b.confidence for b in self.bboxes), default=0.0)


@dataclass
class FrameData:
    """
    统一帧数据对象
    无论来源是 验证集/视频/摄像头/RTSP，都转换成此格式
    """
    frame_id: int
    timestamp: float                     # 模拟时间戳（秒）
    image: Any = None                    # OpenCV 图像矩阵 (numpy array)
    image_path: str = ""
    source_type: str = ""                # "validation_simulator" / "video" / "camera" / "rtsp"
    source_name: str = ""                # 来源名称（如验证集名、视频文件名）

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "source_type": self.source_type,
            "source_name": self.source_name,
        }


@dataclass
class SimulatorStats:
    """模拟器运行统计"""
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    fire_frames: int = 0
    smoke_frames: int = 0
    normal_frames: int = 0
    avg_inference_time_ms: float = 0.0
    total_simulated_duration: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FireEvent:
    """聚合后的火灾事件（M6 → M7 的接口数据）"""
    event_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    detections: List[Detection] = field(default_factory=list)
    fire_area: float = 0.0
    smoke_area: float = 0.0
    duration: float = 0.0
    growth_rate: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "detection_count": len(self.detections),
            "fire_area": self.fire_area,
            "smoke_area": self.smoke_area,
            "duration": self.duration,
            "growth_rate": self.growth_rate,
            "confidence": self.confidence,
        }


@dataclass
class AgentDecision:
    """Agent 决策结果（M7 的输出）"""
    danger_level: str = "low"
    reason: str = ""
    suggestion: str = ""
    analysis_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ======================== 路径工具 ========================

def project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path 对象"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ========== 独立运行测试 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("FireGuardian 共享工具模块 - 自检测试")
    print("=" * 50)

    cfg = load_config()
    proj_name = cfg["project"]["name"]
    print(f"\nConfig OK | {proj_name}")

    # BoundingBox
    box = BoundingBox(10, 20, 100, 200, 0.85, 0, "fire")
    print(f"BoundingBox OK | area={box.area:.0f}, area_ratio={box.area_ratio(640*480):.4f}")

    # Detection
    det = Detection(bboxes=[box], image_path="test.jpg", timestamp=1.0, inference_time_ms=18.5,
                    image_width=640, image_height=480)
    print(f"Detection OK | fire={det.has_fire}, w/h={det.image_width}x{det.image_height}")

    # FrameData（新增）
    fd = FrameData(frame_id=1, timestamp=0.5, image_path="test.jpg",
                   source_type="validation_simulator", source_name="valid")
    print(f"FrameData OK | {fd.source_type} | frame_id={fd.frame_id}")

    # SimulatorStats（新增）
    st = SimulatorStats(total_frames=100, processed_frames=98, fire_frames=30)
    print(f"SimulatorStats OK | {st.to_dict()}")

    # FireEvent
    evt = FireEvent(event_id="EVT-001", source="simulator", fire_area=0.3, duration=8.0)
    print(f"FireEvent OK | {evt.event_id}")

    # AgentDecision
    dec = AgentDecision(danger_level="high", reason="Fire detected", suggestion="Check")
    print(f"AgentDecision OK | {dec.danger_level}")

    root = project_root()
    print(f"\nProject root: {root}")
    print("\nAll shared components OK")
