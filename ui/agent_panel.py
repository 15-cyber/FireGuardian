"""
============================================================
M11 - Agent Analysis Panel（Phase 4-B）

只读展示 AgentPresentationModel；GUI 线程只接收 Qt Signal 更新，
不调用 DeepSeek / Memory / Tool，不重新计算 final_level / similarity。

历史事件详情遵循：
  - 优先使用该事件自己的 AgentPresentationModel；
  - 无对应字段（如 model / thinking）显示「未提供」，
    不用当前全局运行配置冒充历史信息。
============================================================
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


_LEVEL_COLORS = {
    "high": "#c0392b",
    "medium": "#e67e22",
    "low": "#27ae60",
}


def _level_color(level: Any) -> str:
    return _LEVEL_COLORS.get(str(level).lower(), "#555555")


def _text(value: Any, fallback: str = "未提供") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


class AgentAnalysisPanel(QWidget):
    """Agent Analysis 展示面板（Rule / Agent / Memory / Hybrid Final / 详情）。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()
        self.reset()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.status_label = QLabel("Agent: -")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.tool_label = QLabel("Tool: -")
        self.tool_label.setStyleSheet("color: #444;")
        root.addWidget(self.status_label)
        root.addWidget(self.tool_label)

        box = QGroupBox("Agent Analysis")
        lay = QVBoxLayout(box)

        self.final_label = QLabel("Hybrid Final: -")
        self.final_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.rule_label = QLabel("Rule: -")
        self.agent_label = QLabel("Agent: -")
        self.memory_label = QLabel("Memory: -")
        self.reason_label = QLabel("Reason: -")
        self.reason_label.setWordWrap(True)
        lay.addWidget(self.final_label)
        lay.addWidget(self.rule_label)
        lay.addWidget(self.agent_label)
        lay.addWidget(self.memory_label)
        lay.addWidget(self.reason_label)

        self.explanation_label = QLabel("Explanation: -")
        self.explanation_label.setWordWrap(True)
        self.cause_label = QLabel("Possible Cause: -")
        self.cause_label.setWordWrap(True)
        self.action_label = QLabel("Recommended Action: -")
        self.action_label.setWordWrap(True)
        lay.addWidget(self.explanation_label)
        lay.addWidget(self.cause_label)
        lay.addWidget(self.action_label)

        self.similar_label = QLabel("Similar Events: -")
        self.similar_label.setWordWrap(True)
        self.similar_label.setStyleSheet("color: #555;")
        lay.addWidget(self.similar_label)

        self.fallback_note = QLabel("Agent 信息不完整，仅供参考")
        self.fallback_note.setStyleSheet("color: #b03a2e;")
        self.fallback_note.setVisible(False)
        lay.addWidget(self.fallback_note)

        self.detail_label = QLabel("Details: -")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: #777; font-size: 10px;")
        lay.addWidget(self.detail_label)

        root.addWidget(box, 1)

    # ---------- 状态更新 ----------

    def _set_level_label(self, label: QLabel, text: str, level: Any) -> None:
        label.setText(text)
        label.setStyleSheet(f"color: {_level_color(level)};")

    def _apply_status(self, status: str, error: Any = None) -> None:
        err = _text(error, "")
        lower_err = err.lower()
        if status == "ok":
            self.status_label.setText("Agent: Completed")
            self.tool_label.setText("")
        elif status == "invalid_output":
            self.status_label.setText("Agent: Fallback")
            self.tool_label.setText(f"Reason: {_text(error, 'invalid output')}")
        elif status == "unavailable":
            if "timeout" in lower_err:
                self.status_label.setText("Agent: Timeout")
            else:
                self.status_label.setText("Agent: Fallback")
            self.tool_label.setText(f"Reason: {_text(error, 'DeepSeek unavailable')}")
        else:
            self.status_label.setText(f"Agent: {_text(status)}")
            self.tool_label.setText("")

    def set_analyzing(self, payload: dict) -> None:
        self.status_label.setText("Agent: Analyzing...")
        call_count = payload.get("call_count")
        self.tool_label.setText(
            f"Trigger: {_text(payload.get('trigger'), '-')}"
            + (f" | Agent calls: {call_count}/4" if call_count is not None else "")
        )

    def set_tool_called(self, payload: dict) -> None:
        self.status_label.setText("Agent: Analyzing...")
        self.tool_label.setText(f"Tool: {_text(payload.get('tool_name'), '-')}")

    def set_failed(self, payload: dict) -> None:
        status = payload.get("status", "unavailable")
        self._apply_status(status, payload.get("error"))
        self.final_label.setText("Hybrid Final: -")
        self.reason_label.setText(f"Reason: {_text(payload.get('error'), 'Agent failed')}")

    def set_presentation(self, payload: dict) -> None:
        """用该事件自己的 AgentPresentationModel 更新展示。"""
        status = _text(payload.get("agent_status"), "unavailable")
        self._apply_status(status, payload.get("error"))

        rule_level = _text(payload.get("rule_level"), "-")
        agent_risk = _text(payload.get("agent_risk_level"), "-")
        agent_assess = _text(payload.get("agent_assessment"), "未提供")
        final_level = _text(payload.get("final_level"), "-")

        self._set_level_label(self.rule_label, f"Rule: {rule_level.upper()}", rule_level)
        self._set_level_label(
            self.agent_label, f"Agent: {agent_assess} ({agent_risk.upper()})", agent_risk
        )
        final_suffix = "" if status == "ok" else " (M7)"
        self._set_level_label(
            self.final_label, f"Hybrid Final: {final_level.upper()}{final_suffix}", final_level
        )

        memory_used = bool(payload.get("memory_used", False))
        similar_count = int(payload.get("similar_event_count", 0) or 0)
        memory_summary = _text(payload.get("memory_summary"), "")
        if memory_used and similar_count == 0 and "不可用" in memory_summary:
            self.memory_label.setText("Memory: Unavailable")
        elif memory_used:
            self.memory_label.setText(f"Memory: Used ({similar_count})")
        else:
            self.memory_label.setText("Memory: Not Used")

        self.reason_label.setText(f"Reason: {_text(payload.get('reason'), '未提供')}")
        self.explanation_label.setText(
            f"Explanation: {_text(payload.get('reasoning_summary'), '未提供')}"
        )
        self.cause_label.setText(
            f"Possible Cause: {_text(payload.get('possible_cause'), '未提供')}"
        )
        self.action_label.setText(
            f"Recommended Action: {_text(payload.get('recommended_action'), '未提供')}"
        )

        results = payload.get("memory_results") or []
        if results:
            lines = [
                f"  {r.get('event_id', '?')}  {float(r.get('similarity_score', 0.0)):.2f}"
                for r in results[:5]
            ]
            self.similar_label.setText("Similar Events:\n" + "\n".join(lines))
        else:
            self.similar_label.setText("Similar Events: -")

        tools = "、".join(payload.get("tools_used") or []) or "None"
        latency = payload.get("latency_ms")
        latency_text = (
            f"{float(latency) / 1000.0:.1f}s" if isinstance(latency, (int, float)) else "未提供"
        )
        token_usage = payload.get("token_usage") or {}
        total_tokens = token_usage.get("total_tokens")
        tokens_text = str(total_tokens) if total_tokens is not None else "未提供"
        self.detail_label.setText(
            f"Details: Model: 未提供 | Thinking: 未提供 | Status: {status.upper()} | "
            f"Tools: {tools} | Memory: "
            f"{('Used (' + str(similar_count) + ')' if memory_used else 'Not Used')} | "
            f"Latency: {latency_text} | Tokens: {tokens_text}"
        )
        self.fallback_note.setVisible(bool(payload.get("display_fallbacks")))

    def reset(self) -> None:
        self.status_label.setText("Agent: -")
        self.tool_label.setText("Tool: -")
        self.final_label.setText("Hybrid Final: -")
        self.rule_label.setText("Rule: -")
        self.agent_label.setText("Agent: -")
        self.memory_label.setText("Memory: -")
        self.reason_label.setText("Reason: -")
        self.explanation_label.setText("Explanation: -")
        self.cause_label.setText("Possible Cause: -")
        self.action_label.setText("Recommended Action: -")
        self.similar_label.setText("Similar Events: -")
        self.detail_label.setText("Details: -")
        self.fallback_note.setVisible(False)
