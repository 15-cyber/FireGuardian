"""
============================================================
V2 Agent - AgentLogger（Phase 2）

新增日志：
  logs/agent_decisions.jsonl   Agent 决策记录（含 Hybrid 最终结果）
  logs/agent_tool_calls.jsonl  工具调用记录
  logs/agent_memory.jsonl      Memory 检索记录（Phase 3-A）

安全约束（禁止记录）：
  - API Key / Authorization header
  - 完整敏感 prompt
日志写入失败不影响主流程（尽量静默，仅回退到应用日志）。
============================================================
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class AgentLogger:
    """Agent 决策与工具调用的 JSONL 记录器。"""

    def __init__(
        self,
        decisions_path: Optional[Path] = None,
        tool_calls_path: Optional[Path] = None,
        memory_path: Optional[Path] = None,
    ):
        self.decisions_path = Path(decisions_path) if decisions_path else _ROOT / "logs" / "agent_decisions.jsonl"
        self.tool_calls_path = Path(tool_calls_path) if tool_calls_path else _ROOT / "logs" / "agent_tool_calls.jsonl"
        self.memory_path = Path(memory_path) if memory_path else _ROOT / "logs" / "agent_memory.jsonl"

    def _append(self, path: Path, record: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # 日志失败不影响 Agent 主流程
            return

    def log_decision(
        self,
        *,
        event_id: str,
        model: str,
        trigger: str,
        rule_level: str,
        agent_assessment: Optional[str],
        agent_risk_level: Optional[str],
        final_level: str,
        agreement: bool,
        override: bool,
        requires_review: bool,
        tools_used: list,
        latency_ms: float,
        token_usage: dict,
        agent_status: str,
        error: Optional[str] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_id": event_id,
            "model": model,
            "trigger": trigger,
            "rule_level": rule_level,
            "agent_assessment": agent_assessment,
            "agent_risk_level": agent_risk_level,
            "final_level": final_level,
            "agreement": agreement,
            "override": override,
            "requires_review": requires_review,
            "tools_used": list(tools_used or []),
            "latency_ms": round(float(latency_ms), 1),
            "token_usage": dict(token_usage or {}),
            "agent_status": agent_status,
            "error": error,
        }
        self._append(self.decisions_path, record)

    def log_tool_call(
        self,
        *,
        event_id: str,
        trigger: str,
        tool_name: str,
        arguments: dict,
        result_summary: Any,
        latency_ms: float,
        success: bool,
        round_no: int,
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_id": event_id,
            "trigger": trigger,
            "tool_name": tool_name,
            "arguments": arguments,
            "result_summary": result_summary,
            "latency_ms": round(float(latency_ms), 1),
            "success": success,
            "round": round_no,
        }
        self._append(self.tool_calls_path, record)

    def log_memory(
        self,
        *,
        event_id: str,
        top_k: int,
        min_similarity: float,
        candidate_count: int,
        matched_count: int,
        results: list,
        latency_ms: float,
        success: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Memory 检索审计记录（只保存必要数据，禁止 Key / 完整 prompt / 大型原始列表）。"""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_id": event_id,
            "tool": "get_similar_events",
            "memory_used": True,
            "top_k": top_k,
            "min_similarity": min_similarity,
            "candidate_count": candidate_count,
            "matched_count": matched_count,
            "results": list(results or []),
            "latency_ms": round(float(latency_ms), 1),
            "success": success,
            "reason": reason,
        }
        self._append(self.memory_path, record)
