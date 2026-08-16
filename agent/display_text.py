"""
============================================================
V2 Agent - 展示层文本 fallback（Phase 2.1）

用途：供未来 M10 报告 / M11 GUI 展示 AgentResult 时使用。
策略：空字段替换为明确文案，绝不改动原始 AgentResult；
      返回的展示 dict 额外附带 _display_fallbacks 标记哪些字段被替换。
约束：本模块只做只读展示转换，不修改 M10/M11/GUI/Hybrid。
============================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.schemas import AgentResult


_FALLBACK_TEXT = {
    "possible_cause": "暂无明确判断（证据不足或 Agent 未给出结论）",
    "reasoning_summary": "Agent 未给出推理摘要（可查看工具调用与证据记录）",
    "recommended_action": "未给出处置建议，请按当前风险等级执行标准流程",
}

_FALLBACK_EVIDENCE = "无引用证据（可查看截图/历史记录）"
_FALLBACK_TOOLS = "未调用工具"
_FALLBACK_CONFIDENCE = "未提供"


def format_agent_result_for_display(result: AgentResult) -> Dict[str, Any]:
    """生成 GUI/M10 展示文本；空字段使用 fallback 文案，原始 AgentResult 不变。"""
    display: Dict[str, Any] = result.to_dict()
    fallbacks: List[str] = []

    for field, text in _FALLBACK_TEXT.items():
        if not str(display.get(field) or "").strip():
            display[field] = text
            fallbacks.append(field)

    if not display.get("evidence_summary"):
        display["evidence_summary"] = [_FALLBACK_EVIDENCE]
        fallbacks.append("evidence_summary")

    if not display.get("tools_used"):
        display["tools_used"] = [_FALLBACK_TOOLS]
        fallbacks.append("tools_used")

    if float(display.get("agent_confidence", 0.0)) <= 0.0:
        display["agent_confidence_display"] = _FALLBACK_CONFIDENCE
        fallbacks.append("agent_confidence")
    else:
        display["agent_confidence_display"] = f"{float(display['agent_confidence']):.2f}"

    display["_display_fallbacks"] = fallbacks
    return display
