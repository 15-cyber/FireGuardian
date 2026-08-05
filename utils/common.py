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
from enum import Enum
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


# ======================== 枚举 ========================

class EventStatus(str, Enum):
    """火灾事件状态"""
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    ENDED = "ended"
    DISCARDED = "discarded"


class GrowthTrend(str, Enum):
    """增长趋势"""
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    UNKNOWN = "unknown"


class DangerLevel(str, Enum):
    """危险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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

    def area_ratio(self, image_area: float) -> float:
        if image_area <= 0:
            return 0.0
        return self.area / image_area


def union_area_of_boxes(boxes: List[BoundingBox]) -> float:
    """
    计算同一组轴对齐矩形框的并集面积（重叠区域不重复计数）。
    采用扫描线（按 x 排序 + y 区间合并）算法，输入顺序无关。
    """
    if not boxes:
        return 0.0

    events = []
    for b in boxes:
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
        if x2 <= x1 or y2 <= y1:
            continue  # 无效框直接忽略
        events.append((x1, y1, y2, 1))
        events.append((x2, y1, y2, -1))
    if not events:
        return 0.0

    events.sort(key=lambda e: e[0])
    area = 0.0
    prev_x = events[0][0]
    active: List[tuple] = []
    for x, y1, y2, typ in events:
        if x > prev_x and active:
            # 合并当前活跃的 y 区间
            active_sorted = sorted(active, key=lambda iv: iv[0])
            merged_len = 0.0
            cur_l, cur_r = active_sorted[0]
            for l, r in active_sorted[1:]:
                if l > cur_r:
                    merged_len += cur_r - cur_l
                    cur_l, cur_r = l, r
                else:
                    cur_r = max(cur_r, r)
            merged_len += cur_r - cur_l
            area += merged_len * (x - prev_x)
        prev_x = x
        if typ == 1:
            active.append((y1, y2))
        else:
            try:
                active.remove((y1, y2))
            except ValueError:
                pass
    return area


@dataclass
class Detection:
    """单帧检测结果"""
    bboxes: List[BoundingBox] = field(default_factory=list)
    image_path: str = ""
    timestamp: float = 0.0
    wall_time: str = ""
    inference_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    frame_id: int = -1  # 帧号（-1 表示未提供，M8 截图元数据使用）
    fps: Optional[float] = None  # 输入源帧率（用于 frame_id → seconds 转换，None=未提供）

    @property
    def fire_boxes(self) -> List[BoundingBox]:
        return [b for b in self.bboxes if b.class_name == "fire"]

    @property
    def smoke_boxes(self) -> List[BoundingBox]:
        return [b for b in self.bboxes if b.class_name == "smoke"]

    @property
    def fire_area(self) -> float:
        return union_area_of_boxes(self.fire_boxes)

    @property
    def smoke_area(self) -> float:
        return union_area_of_boxes(self.smoke_boxes)

    @property
    def fire_area_ratio(self) -> float:
        img_area = self.image_width * self.image_height
        if img_area <= 0:
            return 0.0
        return min(self.fire_area / img_area, 1.0)

    @property
    def smoke_area_ratio(self) -> float:
        img_area = self.image_width * self.image_height
        if img_area <= 0:
            return 0.0
        return min(self.smoke_area / img_area, 1.0)

    @property
    def has_fire(self) -> bool:
        return len(self.fire_boxes) > 0

    @property
    def has_smoke(self) -> bool:
        return len(self.smoke_boxes) > 0

    @property
    def max_confidence(self) -> float:
        return max((b.confidence for b in self.bboxes), default=0.0)

    @property
    def avg_fire_confidence(self) -> float:
        if not self.fire_boxes:
            return 0.0
        return sum(b.confidence for b in self.fire_boxes) / len(self.fire_boxes)

    @property
    def avg_smoke_confidence(self) -> float:
        if not self.smoke_boxes:
            return 0.0
        return sum(b.confidence for b in self.smoke_boxes) / len(self.smoke_boxes)


@dataclass
class FrameData:
    """统一帧数据对象"""
    frame_id: int
    timestamp: float
    image: Any = None
    image_path: str = ""
    source_type: str = ""
    source_name: str = ""

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
    """火灾事件（M6 → M7 的统一接口数据）"""
    event_id: str = ""
    status: EventStatus = EventStatus.CANDIDATE

    start_timestamp: float = 0.0
    last_timestamp: float = 0.0
    end_timestamp: Optional[float] = None
    # ---- 时间统一字段（规范单位：秒，M7 决策统一使用）----
    fps: Optional[float] = None               # 输入源帧率（frame_id → seconds 转换用）
    start_time_seconds: float = 0.0           # 事件开始时间（秒）
    last_time_seconds: float = 0.0            # 最近一次更新时间（秒）
    end_time_seconds: Optional[float] = None  # 事件结束时间（秒）
    duration_seconds: float = 0.0             # 事件持续时长（秒）

    total_frames: int = 0
    positive_frames: int = 0
    consecutive_positive_frames: int = 0
    consecutive_negative_frames: int = 0

    max_fire_area_ratio: float = 0.0
    max_smoke_area_ratio: float = 0.0
    avg_fire_confidence: float = 0.0
    avg_smoke_confidence: float = 0.0

    growth_trend: str = GrowthTrend.UNKNOWN

    source_type: str = ""
    source_name: str = ""

    representative_frame_id: Optional[int] = None
    representative_image_path: Optional[str] = None

    metadata: dict = field(default_factory=dict)
    detections: List[Detection] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """
        事件持续时间（秒）。
        优先使用规范化字段 duration_seconds；未设置时回退到 timestamp 差值。
        """
        if self.duration_seconds > 0:
            return self.duration_seconds
        if self.end_timestamp is not None:
            return self.end_timestamp - self.start_timestamp
        if self.last_timestamp > 0:
            return self.last_timestamp - self.start_timestamp
        return 0.0

    @property
    def positive_ratio(self) -> float:
        """阳性帧比例（用于可信度计算）"""
        if self.total_frames <= 0:
            return 0.0
        return self.positive_frames / self.total_frames

    @property
    def is_active(self) -> bool:
        return self.status in (EventStatus.CONFIRMED, EventStatus.ACTIVE)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "status": self.status.value if isinstance(self.status, EventStatus) else str(self.status),
            "start_timestamp": self.start_timestamp,
            "last_timestamp": self.last_timestamp,
            "end_timestamp": self.end_timestamp,
            "fps": self.fps,
            "start_time_seconds": self.start_time_seconds,
            "last_time_seconds": self.last_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "duration_seconds": self.duration_seconds,
            "duration": self.duration,
            "total_frames": self.total_frames,
            "positive_frames": self.positive_frames,
            "positive_ratio": self.positive_ratio,
            "max_fire_area_ratio": self.max_fire_area_ratio,
            "max_smoke_area_ratio": self.max_smoke_area_ratio,
            "avg_fire_confidence": self.avg_fire_confidence,
            "avg_smoke_confidence": self.avg_smoke_confidence,
            "growth_trend": self.growth_trend.value if isinstance(self.growth_trend, GrowthTrend) else str(self.growth_trend),
            "source_type": self.source_type,
            "source_name": self.source_name,
        }


@dataclass
class FireDecision:
    """
    Agent 决策结果（M7 → M10/M11 的统一接口数据）
    包含危险等级、结构化原因、建议、评分和可信度
    """
    event_id: str = ""
    danger_level: str = DangerLevel.LOW
    score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    summary: str = ""
    decision_source: str = "rule_engine"
    generated_at: str = ""
    debug_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "danger_level": self.danger_level,
            "score": self.score,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "suggestions": self.suggestions,
            "summary": self.summary,
            "decision_source": self.decision_source,
            "generated_at": self.generated_at,
        }


@dataclass
class EventReportData:
    """M10 报告输入包：所有数据显式传入，报告生成器不自行查目录"""
    event: FireEvent
    decisions: List[FireDecision] = field(default_factory=list)
    screenshots: List[ScreenshotRecord] = field(default_factory=list)
    system_info: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)


@dataclass
class ScreenshotRecord:
    """M8 截图记录（统一结构化对象，供 M10 报告使用）"""
    event_id: str = ""
    reason: str = ""
    frame_id: int = -1
    simulated_timestamp: float = 0.0
    raw_image_path: str = ""
    annotated_image_path: str = ""
    metadata_path: str = ""
    fire_area_ratio: float = 0.0
    smoke_area_ratio: float = 0.0
    danger_level: Optional[str] = None
    decision_score: Optional[float] = None
    decision_confidence: Optional[float] = None
    decision_source: Optional[str] = None
    saved_at: str = ""
    write_status: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AgentDecision:
    """（兼容旧版，后续迁移到 FireDecision）"""
    danger_level: str = DangerLevel.LOW
    reason: str = ""
    suggestion: str = ""
    analysis_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ======================== 路径工具 ========================

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ========== 独立运行测试 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("FireGuardian 共享工具模块 - 自检测试")
    print("=" * 50)

    cfg = load_config()
    print(f"Config OK | {cfg['project']['name']}")

    print(f"EventStatus: {[s.value for s in EventStatus]}")
    print(f"GrowthTrend: {[g.value for g in GrowthTrend]}")

    # FireEvent
    evt = FireEvent(
        event_id="EVT-001",
        status=EventStatus.ACTIVE,
        start_timestamp=0.0,
        last_timestamp=5.0,
        total_frames=10,
        positive_frames=8,
        max_fire_area_ratio=0.15,
        growth_trend=GrowthTrend.INCREASING,
        source_type="simulator",
        source_name="valid",
    )
    print(f"FireEvent OK | {evt.event_id} | duration={evt.duration:.1f}s | positive_ratio={evt.positive_ratio:.2f}")

    # FireDecision (upgraded)
    dec = FireDecision(
        event_id="EVT-001",
        danger_level=DangerLevel.HIGH,
        score=0.57,
        confidence=0.82,
        reasons=["火焰面积较大", "持续超过阈值"],
        suggestions=["立即检查现场"],
        summary="大面积火焰持续扩大，属于高风险",
        decision_source="rule_engine",
        generated_at="2026-07-31 12:00:00",
    )
    print(f"FireDecision OK | {dec.danger_level} | score={dec.score} | conf={dec.confidence}")
    print(f"  reasons: {dec.reasons}")
    print(f"  suggestions: {dec.suggestions}")

    print("\nAll shared components OK")
