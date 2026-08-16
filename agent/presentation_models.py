"""
============================================================
V2 Agent - Presentation Model（Phase 4-A）

统一 M10 报告 / M11 GUI 的展示数据：
  - AgentPresentationModel：从 FireDecision + AgentOutcome +
    HybridDecision + Memory 结果一次性转换；
  - AgentTriggerGuard：单事件 Agent 调用去重/计数（confirmed /
    risk_upgrade / disagreement / ended，单事件不超过上限）；
  - run_agent_with_timeout：整体运行超时保护（超时 → unavailable，
    回退 M7，不无限等待）。

约束：展示层只读格式化，不重新计算 final_level / similarity。
============================================================
"""
from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.deepseek_agent import AgentOutcome, AgentStatus, FireGuardianLLMAgent
from agent.display_text import format_agent_result_for_display
from agent.hybrid_decision import HybridDecision
from utils.common import FireDecision, FireEvent


AGENT_RUN_TIMEOUT_SECONDS = 60.0
AGENT_MAX_CALLS_PER_EVENT = 4
AGENT_TRIGGERS = ("event_confirmed", "risk_upgrade", "disagreement", "event_ended")

_LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}


def _rank(level: Any) -> int:
    value = getattr(level, "value", level)
    return _LEVEL_RANK.get(str(value).lower(), 1)


class AgentTriggerGuard:
    """单事件 Agent 调用保护：触发去重 + 计数上限。"""

    def __init__(self, max_calls_per_event: int = AGENT_MAX_CALLS_PER_EVENT):
        self.max_calls = int(max_calls_per_event)
        self._counts: Dict[str, int] = {}
        self._done: set = set()
        self._levels: Dict[str, str] = {}

    @staticmethod
    def _key(event_id: str, trigger: str, current_level: Optional[str]) -> tuple:
        if trigger == "risk_upgrade":
            return (event_id, trigger, current_level)
        return (event_id, trigger, None)

    def allowed(
        self,
        event_id: str,
        trigger: str,
        current_level: Optional[Any] = None,
    ) -> bool:
        if trigger not in AGENT_TRIGGERS:
            return False
        if self._counts.get(event_id, 0) >= self.max_calls:
            return False
        level_str = str(getattr(current_level, "value", current_level)).lower() if current_level is not None else None
        if trigger == "risk_upgrade":
            if current_level is None:
                return False
            prev = self._levels.get(event_id)
            if prev is not None and _rank(level_str) <= _rank(prev):
                return False
        key = self._key(event_id, trigger, level_str)
        return key not in self._done

    def record(
        self,
        event_id: str,
        trigger: str,
        current_level: Optional[Any] = None,
    ) -> int:
        """记录一次调用，返回该事件累计调用次数。"""
        level_str = str(getattr(current_level, "value", current_level)).lower() if current_level is not None else None
        key = self._key(event_id, trigger, level_str)
        self._done.add(key)
        self._counts[event_id] = self._counts.get(event_id, 0) + 1
        if level_str is not None:
            self._levels[event_id] = level_str
        return self._counts[event_id]

    def count(self, event_id: str) -> int:
        return self._counts.get(event_id, 0)

    def trigger_calls(self, event_id: str) -> List[dict]:
        return [
            {"event_id": e, "trigger": t, "level": lv, "call_count": self._counts.get(e, 0)}
            for (e, t, lv) in self._done
            if e == event_id
        ]


def append_agent_trigger_log(
    event_id: str,
    trigger: str,
    call_count: int,
    path: Optional[Path] = None,
) -> None:
    """记录 event_id + trigger + call_count（禁止敏感信息）。"""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event_id": event_id,
        "trigger": trigger,
        "call_count": int(call_count),
    }
    target = path or (_ROOT / "logs" / "agent_trigger_calls.jsonl")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return


def run_agent_with_timeout(
    agent: FireGuardianLLMAgent,
    event: FireEvent,
    decision: Optional[FireDecision],
    trigger: str = "event_confirmed",
    timeout_seconds: float = AGENT_RUN_TIMEOUT_SECONDS,
) -> AgentOutcome:
    """整体超时保护：超时返回 unavailable，不无限等待、不阻塞调用方。"""
    holder: Dict[str, Any] = {}

    def _target() -> None:
        try:
            holder["outcome"] = agent.analyze(event, decision, trigger=trigger)
        except Exception as exc:  # noqa: BLE001 - analyze 不应抛异常，防御性兜底
            holder["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=float(timeout_seconds))
    if worker.is_alive():
        return AgentOutcome(
            status=AgentStatus.UNAVAILABLE,
            error=f"agent_timeout_exceeded ({timeout_seconds}s)",
        )
    if "error" in holder:
        return AgentOutcome(
            status=AgentStatus.UNAVAILABLE,
            error=f"agent_run_error: {holder['error']}",
        )
    return holder.get("outcome") or AgentOutcome(
        status=AgentStatus.UNAVAILABLE, error="agent_no_result"
    )


def extract_memory_results(outcome: AgentOutcome) -> List[dict]:
    """从工具执行记录中提取紧凑的相似事件摘要（event_id + score）。"""
    results: List[dict] = []
    for tc in outcome.tool_calls:
        if tc.get("name") != "get_similar_events" or not tc.get("success"):
            continue
        payload = (tc.get("result_summary") or {}).get("data") or {}
        for item in payload.get("results", [])[:10]:
            results.append(
                {
                    "event_id": item.get("event_id", ""),
                    "similarity_score": float(item.get("similarity_score", 0.0)),
                }
            )
    return results


@dataclass
class AgentPresentationModel:
    """M10/M11 共用展示模型（字段来自 §26，全部由已有结果转换）。"""

    event_id: str = ""
    agent_status: str = AgentStatus.UNAVAILABLE.value
    rule_level: str = "low"
    rule_score: float = 0.0
    rule_confidence: float = 0.0
    rule_reasons: List[str] = field(default_factory=list)
    agent_assessment: Optional[str] = None
    agent_risk_level: Optional[str] = None
    agreement: Optional[bool] = None
    memory_used: bool = False
    similar_event_count: int = 0
    memory_summary: str = ""
    memory_results: List[dict] = field(default_factory=list)
    possible_cause: str = ""
    reasoning_summary: str = ""
    recommended_action: str = ""
    evidence_summary: List[str] = field(default_factory=list)
    uncertainty: str = "medium"
    tools_used: List[str] = field(default_factory=list)
    final_level: str = "low"
    agreement_hybrid: bool = True
    override: bool = False
    requires_review: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)
    display_fallbacks: List[str] = field(default_factory=list)
    agent_available: bool = False

    @property
    def status_label(self) -> str:
        return str(self.agent_status).upper()

    def to_dict(self) -> dict:
        return asdict(self)


def build_agent_presentation_model(
    event_id: str,
    rule_decision: Optional[FireDecision],
    outcome: AgentOutcome,
    hybrid: HybridDecision,
    memory_results: Optional[List[dict]] = None,
) -> AgentPresentationModel:
    """从已有结果一次性转换展示模型（不做任何重算）。"""
    rule = rule_decision or FireDecision(event_id=event_id)
    display = (
        format_agent_result_for_display(outcome.agent_result)
        if outcome.agent_result is not None
        else {}
    )
    rule_level = getattr(rule.danger_level, "value", rule.danger_level)
    return AgentPresentationModel(
        event_id=event_id,
        agent_status=outcome.status.value,
        rule_level=str(rule_level).lower(),
        rule_score=float(rule.score or 0.0),
        rule_confidence=float(rule.confidence or 0.0),
        rule_reasons=list(rule.reasons or []),
        agent_assessment=display.get("agent_assessment") if outcome.agent_result else None,
        agent_risk_level=display.get("risk_level") if outcome.agent_result else None,
        agreement=display.get("agreement_with_rule") if outcome.agent_result else None,
        memory_used=bool(outcome.agent_result and outcome.agent_result.memory_used),
        similar_event_count=int(outcome.agent_result.similar_event_count)
        if outcome.agent_result
        else 0,
        memory_summary=str(outcome.agent_result.memory_summary) if outcome.agent_result else "",
        memory_results=list(memory_results or []),
        possible_cause=display.get("possible_cause", ""),
        reasoning_summary=display.get("reasoning_summary", ""),
        recommended_action=display.get("recommended_action", ""),
        evidence_summary=list(display.get("evidence_summary", [])),
        uncertainty=display.get("uncertainty", "medium"),
        tools_used=list(display.get("tools_used", [])),
        final_level=str(hybrid.final_level).lower(),
        agreement_hybrid=bool(hybrid.agreement),
        override=bool(hybrid.override),
        requires_review=bool(hybrid.requires_review),
        reason=hybrid.reason or "",
        latency_ms=float(outcome.latency_ms or 0.0),
        token_usage=dict(outcome.token_usage or {}),
        display_fallbacks=list(display.get("_display_fallbacks", [])),
        agent_available=outcome.status == AgentStatus.OK,
    )
