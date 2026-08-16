"""
============================================================
V2 Agent - 统一数据 Schema（Phase 1）

定义：
  - AgentAssessment：Agent 对事件的定性判断（与 risk_level 分离）
  - AgentResult：Agent 最终统一输出（JSON Schema 校验）
  - ToolCall / ToolResult：工具调用与结果的结构化载体

安全说明：所有来自 LLM 的 JSON 必须先经过 from_dict/validate 校验，
非法值一律抛 SchemaValidationError，由上层走 Fallback。
============================================================
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentAssessment(str, Enum):
    """Agent 定性判断，与 low/medium/high 风险等级分开。"""
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    CONFIRMED_RISK = "CONFIRMED_RISK"
    UNCERTAIN = "UNCERTAIN"


RISK_LEVELS: tuple = ("low", "medium", "high")
UNCERTAINTY_LEVELS: tuple = ("low", "medium", "high")


class SchemaValidationError(ValueError):
    """Agent JSON Schema 校验失败。"""


@dataclass
class AgentResult:
    """Agent 最终统一输出（V2 计划书 §12）。"""

    event_id: str = ""
    agent_assessment: str = AgentAssessment.UNCERTAIN.value
    risk_level: str = "medium"
    agreement_with_rule: bool = True
    possible_cause: str = ""
    evidence_summary: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    recommended_action: str = ""
    uncertainty: str = "medium"
    tools_used: List[str] = field(default_factory=list)
    agent_confidence: float = 0.0
    # ---- Memory 可选字段（Phase 3-B，向后兼容） ----
    memory_used: bool = False
    similar_event_count: int = 0
    memory_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        errors: List[str] = []
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            errors.append("event_id 必须为非空字符串")
        if self.agent_assessment not in AgentAssessment._value2member_map_:
            errors.append(f"agent_assessment 非法: {self.agent_assessment!r}")
        if self.risk_level not in RISK_LEVELS:
            errors.append(f"risk_level 非法: {self.risk_level!r}（应为 low/medium/high）")
        if self.uncertainty not in UNCERTAINTY_LEVELS:
            errors.append(f"uncertainty 非法: {self.uncertainty!r}（应为 low/medium/high）")
        if not isinstance(self.agreement_with_rule, bool):
            errors.append("agreement_with_rule 必须为布尔值")
        if not isinstance(self.agent_confidence, (int, float)) or isinstance(self.agent_confidence, bool):
            errors.append("agent_confidence 必须为数值")
        elif not (0.0 <= float(self.agent_confidence) <= 1.0):
            errors.append(f"agent_confidence 超出 [0,1]: {self.agent_confidence!r}")
        for name, value in (
            ("possible_cause", self.possible_cause),
            ("reasoning_summary", self.reasoning_summary),
            ("recommended_action", self.recommended_action),
        ):
            if not isinstance(value, str):
                errors.append(f"{name} 必须为字符串")
        for name, value in (
            ("evidence_summary", self.evidence_summary),
            ("tools_used", self.tools_used),
        ):
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                errors.append(f"{name} 必须为字符串列表")
        if not isinstance(self.memory_used, bool):
            errors.append("memory_used 必须为布尔值")
        if isinstance(self.similar_event_count, bool) or not isinstance(self.similar_event_count, int):
            errors.append("similar_event_count 必须为整数")
        elif self.similar_event_count < 0:
            errors.append("similar_event_count 不能为负数")
        if not isinstance(self.memory_summary, str):
            errors.append("memory_summary 必须为字符串")
        if errors:
            raise SchemaValidationError("; ".join(errors))
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "AgentResult":
        """从 LLM 输出 JSON 构造 AgentResult，缺失字段取默认值，非法值报错。"""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Agent 输出必须是 JSON 对象，收到: {type(data).__name__}")
        try:
            confidence = float(data.get("agent_confidence", 0.0))
        except (TypeError, ValueError):
            raise SchemaValidationError("agent_confidence 必须为数值") from None
        memory_used = data.get("memory_used", False)
        if memory_used is None:
            memory_used = False
        if not isinstance(memory_used, bool):
            raise SchemaValidationError("memory_used 必须为布尔值")
        similar_event_count_raw = data.get("similar_event_count", 0)
        if similar_event_count_raw is None:
            similar_event_count_raw = 0
        if isinstance(similar_event_count_raw, bool) or not isinstance(
            similar_event_count_raw, int
        ):
            raise SchemaValidationError("similar_event_count 必须为整数")
        similar_event_count = similar_event_count_raw
        memory_summary_raw = data.get("memory_summary", "")
        if memory_summary_raw is None:
            memory_summary_raw = ""
        if not isinstance(memory_summary_raw, str):
            raise SchemaValidationError("memory_summary 必须为字符串")
        result = cls(
            event_id=str(data.get("event_id", "")),
            agent_assessment=str(data.get("agent_assessment", AgentAssessment.UNCERTAIN.value)),
            risk_level=str(data.get("risk_level", "medium")),
            agreement_with_rule=bool(data.get("agreement_with_rule", True)),
            possible_cause=str(data.get("possible_cause", "")),
            evidence_summary=[str(v) for v in data.get("evidence_summary", [])],
            reasoning_summary=str(data.get("reasoning_summary", "")),
            recommended_action=str(data.get("recommended_action", "")),
            uncertainty=str(data.get("uncertainty", "medium")),
            tools_used=[str(v) for v in data.get("tools_used", [])],
            agent_confidence=confidence,
            memory_used=memory_used,
            similar_event_count=similar_event_count,
            memory_summary=memory_summary_raw,
        )
        result.validate()
        return result


@dataclass
class ToolCall:
    """模型发起的工具调用请求。"""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    """工具执行结果（结构化、可记录）。"""

    tool_name: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
