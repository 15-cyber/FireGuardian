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
from typing import List, Optional

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
        return self.width * self.height


@dataclass
class Detection:
    """单次检测结果"""
    bboxes: List[BoundingBox] = field(default_factory=list)
    image_path: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

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


@dataclass
class FireEvent:
    """聚合后的火灾事件（M6 → M7 的接口数据）"""
    event_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""                # image / video / simulator
    detections: List[Detection] = field(default_factory=list)
    fire_area: float = 0.0
    smoke_area: float = 0.0
    duration: float = 0.0           # 持续时间（秒）
    growth_rate: float = 0.0        # 火势增长率
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
    danger_level: str = "low"       # low / medium / high
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

    # 测试配置加载
    cfg = load_config()
    print(f"\n✅ 配置加载成功，项目名称: {cfg['project']['name']}")

    # 测试数据类
    box = BoundingBox(x1=10, y1=20, x2=100, y2=200, confidence=0.85, class_id=0, class_name="fire")
    detection = Detection(bboxes=[box], image_path="test.jpg")
    print(f"✅ 检测结果: {len(detection.bboxes)} 个目标, 火焰面积: {detection.fire_area:.1f}")

    event = FireEvent(event_id="EVT-001", source="test", fire_area=0.3, smoke_area=0.1, duration=8.0)
    print(f"✅ 火灾事件: {event.event_id}, 危险评估输入已就绪")

    decision = AgentDecision(danger_level="high", reason="连续检测8秒且火势扩大", suggestion="立即检查现场")
    print(f"✅ Agent决策: 等级={decision.danger_level}")

    print(f"\n📁 项目根目录: {project_root()}")
    print("\n所有共享组件正常工作 ✅")
