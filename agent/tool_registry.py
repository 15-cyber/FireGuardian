"""
============================================================
V2 Agent - ToolRegistry（Phase 1）

统一注册 Agent 可用工具（OpenAI Function Calling schema），
并提供工具调用的参数校验。

安全原则：模型输出的工具参数不可信，执行前必须经过 validate_call：
  event_id 格式白名单、数值范围检查、类型检查、未知键拒绝。
禁止 eval() / 任意文件系统访问。
============================================================
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class ToolValidationError(ValueError):
    """工具调用参数校验失败。"""


def _event_id_schema() -> dict:
    return {"type": "string", "description": "FireEvent 事件 ID"}


TOOL_GET_EVENT_SUMMARY: dict = {
    "type": "function",
    "function": {
        "name": "get_event_summary",
        "description": "获取指定 FireEvent 的完整摘要（状态、秒制时长、fire/smoke 统计、增长趋势）。",
        "parameters": {
            "type": "object",
            "properties": {"event_id": _event_id_schema()},
            "required": ["event_id"],
            "additionalProperties": False,
        },
    },
}

TOOL_GET_RULE_DECISION: dict = {
    "type": "function",
    "function": {
        "name": "get_rule_decision",
        "description": "获取指定 FireEvent 的 M7 规则决策（level/score/confidence/reasons）。",
        "parameters": {
            "type": "object",
            "properties": {"event_id": _event_id_schema()},
            "required": ["event_id"],
            "additionalProperties": False,
        },
    },
}

TOOL_GET_DETECTION_HISTORY: dict = {
    "type": "function",
    "function": {
        "name": "get_detection_history",
        "description": "获取指定 FireEvent 最近若干秒的检测快照（置信度/面积变化）。",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": _event_id_schema(),
                "window_seconds": {
                    "type": "number",
                    "description": "回溯窗口（秒），1~3600",
                    "minimum": 1,
                    "maximum": 3600,
                },
            },
            "required": ["event_id", "window_seconds"],
            "additionalProperties": False,
        },
    },
}

TOOL_GET_EVENT_EVIDENCE: dict = {
    "type": "function",
    "function": {
        "name": "get_event_evidence",
        "description": "获取指定 FireEvent 的证据截图目录与元数据（不传输图片本身）。",
        "parameters": {
            "type": "object",
            "properties": {"event_id": _event_id_schema()},
            "required": ["event_id"],
            "additionalProperties": False,
        },
    },
}

TOOL_GET_SIMILAR_EVENTS: dict = {
    "type": "function",
    "function": {
        "name": "get_similar_events",
        "description": (
            "查询已结束历史事件中与当前事件相似的记录（结构化检索，仅辅助证据）。"
            "返回相似度、匹配特征与历史最终等级。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": _event_id_schema(),
                "top_k": {
                    "type": "integer",
                    "description": "返回条数，1~10，默认 5",
                    "minimum": 1,
                    "maximum": 10,
                },
                "min_similarity": {
                    "type": "number",
                    "description": "最低相似度 0~1，默认 0.65",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["event_id"],
            "additionalProperties": False,
        },
    },
}


TOOL_DEFINITIONS: List[dict] = [
    TOOL_GET_EVENT_SUMMARY,
    TOOL_GET_RULE_DECISION,
    TOOL_GET_DETECTION_HISTORY,
    TOOL_GET_EVENT_EVIDENCE,
    TOOL_GET_SIMILAR_EVENTS,
]


class ToolRegistry:
    """工具注册表：注册、查询、schema 导出与参数校验。"""

    def __init__(self, definitions: Optional[List[dict]] = None):
        self._definitions: Dict[str, dict] = {}
        for definition in definitions if definitions is not None else TOOL_DEFINITIONS:
            self.register(definition)

    def register(self, definition: dict) -> None:
        if not isinstance(definition, dict):
            raise ToolValidationError("工具定义必须是 dict")
        name = definition.get("function", {}).get("name", "")
        if not name or not re.match(r"^[a-z_][a-z0-9_]*$", name):
            raise ToolValidationError(f"非法工具名: {name!r}")
        self._definitions[name] = definition

    def get(self, name: str) -> Optional[dict]:
        return self._definitions.get(name)

    def list(self) -> List[str]:
        return list(self._definitions.keys())

    def schemas(self) -> List[dict]:
        return [self._definitions[name] for name in self.list()]

    def validate_call(self, name: str, arguments: Any) -> None:
        """校验工具调用参数；非法时抛 ToolValidationError。"""
        definition = self._definitions.get(name)
        if definition is None:
            raise ToolValidationError(f"未知工具: {name!r}")
        if not isinstance(arguments, dict):
            raise ToolValidationError(f"工具 {name} 的参数必须是 JSON 对象")
        params = definition["function"]["parameters"]
        properties = params.get("properties", {})
        required = set(params.get("required", []))
        for key in arguments:
            if key not in properties:
                raise ToolValidationError(f"工具 {name} 存在未知参数: {key!r}")
        for key in required:
            if key not in arguments:
                raise ToolValidationError(f"工具 {name} 缺少必填参数: {key!r}")
        for key, value in arguments.items():
            self._validate_value(name, key, value, properties[key])

    @staticmethod
    def _validate_value(tool: str, key: str, value: Any, schema: dict) -> None:
        vtype = schema.get("type")
        if vtype == "string":
            if not isinstance(value, str):
                raise ToolValidationError(f"工具 {tool} 参数 {key} 必须为字符串")
            if key == "event_id" and not EVENT_ID_PATTERN.match(value):
                raise ToolValidationError(
                    f"工具 {tool} 参数 event_id 非法: {value!r}（仅允许字母/数字/_/-，≤64字符）"
                )
        elif vtype == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ToolValidationError(f"工具 {tool} 参数 {key} 必须为数字")
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and value < minimum:
                raise ToolValidationError(
                    f"工具 {tool} 参数 {key} 小于最小值 {minimum}: {value!r}"
                )
            if maximum is not None and value > maximum:
                raise ToolValidationError(
                    f"工具 {tool} 参数 {key} 大于最大值 {maximum}: {value!r}"
                )
        elif vtype == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolValidationError(f"工具 {tool} 参数 {key} 必须为整数")
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")
            if minimum is not None and value < minimum:
                raise ToolValidationError(
                    f"工具 {tool} 参数 {key} 小于最小值 {minimum}: {value!r}"
                )
            if maximum is not None and value > maximum:
                raise ToolValidationError(
                    f"工具 {tool} 参数 {key} 大于最大值 {maximum}: {value!r}"
                )
        elif vtype == "object":
            if not isinstance(value, dict):
                raise ToolValidationError(f"工具 {tool} 参数 {key} 必须为对象")
        else:
            raise ToolValidationError(f"工具 {tool} 参数 {key} 的 schema 类型不支持: {vtype!r}")
