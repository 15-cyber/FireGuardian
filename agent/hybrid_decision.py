"""
============================================================
V2 Agent - HybridDecisionCoordinator（Phase 2）

安全融合规则（严格，M7 High 永不可被 LLM 降级）：
  Case 1: M7=high   Agent=low      → Final=high
  Case 2: M7=high   Agent=high     → Final=high
  Case 3: M7=medium Agent=high     → Final=high, requires_review=true
  Case 4: M7=medium Agent=medium   → Final=medium
  Case 5: M7=low    Agent=high     → Final=high, requires_review=true
  Case 6: Agent unavailable        → Final=M7
  Case 7: Agent invalid JSON       → Final=M7
============================================================
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.deepseek_agent import AgentOutcome, AgentStatus
from agent.schemas import AgentResult, RISK_LEVELS


class HybridValidationError(ValueError):
    pass


@dataclass
class HybridDecision:
    """Hybrid 融合结果。"""

    rule_level: str = "low"
    agent_level: Optional[str] = None
    final_level: str = "low"
    agreement: bool = True
    override: bool = False
    requires_review: bool = False
    reason: str = ""
    agent_status: str = AgentStatus.UNAVAILABLE.value

    def to_dict(self) -> dict:
        return {
            "rule_level": self.rule_level,
            "agent_level": self.agent_level,
            "final_level": self.final_level,
            "agreement": self.agreement,
            "override": self.override,
            "requires_review": self.requires_review,
            "reason": self.reason,
            "agent_status": self.agent_status,
        }


_LEVEL_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}


class HybridDecisionCoordinator:
    """规则引擎与 Agent 的安全融合协调器。"""

    @staticmethod
    def _validate_level(level: str, field_name: str) -> None:
        if level not in RISK_LEVELS:
            raise HybridValidationError(f"{field_name} 非法: {level!r}（应为 low/medium/high）")

    def decide(
        self,
        rule_level: str,
        agent_level: Optional[str] = None,
        agent_status: str = AgentStatus.OK.value,
    ) -> HybridDecision:
        self._validate_level(rule_level, "rule_level")
        if agent_status not in ("ok", "unavailable", "invalid_output"):
            raise HybridValidationError(f"agent_status 非法: {agent_status!r}")

        # Case 6 / Case 7：Agent 不可用或输出非法 → Final = M7
        if agent_status != AgentStatus.OK.value or agent_level is None:
            return HybridDecision(
                rule_level=rule_level,
                agent_level=agent_level,
                final_level=rule_level,
                agreement=True,
                override=False,
                requires_review=False,
                reason=f"Agent {agent_status}，Final 回退到 M7 规则等级",
                agent_status=agent_status,
            )

        self._validate_level(agent_level, "agent_level")
        rule_rank = _LEVEL_RANK[rule_level]
        agent_rank = _LEVEL_RANK[agent_level]
        final_rank = max(rule_rank, agent_rank)
        final_level = RISK_LEVELS[final_rank]

        override = agent_rank > rule_rank
        agreement = agent_rank == rule_rank
        requires_review = agent_level == "high" and rule_level != "high"

        if rule_level == "high":
            # Case 1 / Case 2：M7 high 永不可降级
            reason = (
                "M7=high，Final=high；M7 High 不可被 Agent 降级"
                if agent_level != "high"
                else "M7=high 与 Agent=high 一致，Final=high"
            )
        elif override:
            reason = (
                f"Agent=high 升级 M7={rule_level} → Final=high，requires_review=true"
                if agent_level == "high"
                else f"Agent={agent_level} 建议高于 M7={rule_level}，Final={final_level}"
            )
        else:
            reason = f"Agent={agent_level} 与 M7={rule_level} 一致，Final={final_level}"

        return HybridDecision(
            rule_level=rule_level,
            agent_level=agent_level,
            final_level=final_level,
            agreement=agreement,
            override=override,
            requires_review=requires_review,
            reason=reason,
            agent_status=agent_status,
        )


def run_hybrid_pipeline(
    rule_level: str,
    outcome: AgentOutcome,
    *,
    event_id: str = "",
    trigger: str = "event_confirmed",
    model: str = "",
    logger: Optional[AgentLogger] = None,
) -> HybridDecision:
    """Agent 输出 → Hybrid → 写入 Agent 决策日志。"""
    coordinator = HybridDecisionCoordinator()
    decision = coordinator.decide(
        rule_level=rule_level,
        agent_level=outcome.agent_level if outcome.status == AgentStatus.OK else None,
        agent_status=outcome.status.value,
    )
    if logger is not None:
        logger.log_decision(
            event_id=event_id,
            model=model,
            trigger=trigger,
            rule_level=rule_level,
            agent_assessment=outcome.agent_assessment,
            agent_risk_level=outcome.agent_level,
            final_level=decision.final_level,
            agreement=decision.agreement,
            override=decision.override,
            requires_review=decision.requires_review,
            tools_used=outcome.tools_used,
            latency_ms=outcome.latency_ms,
            token_usage=outcome.token_usage,
            agent_status=decision.agent_status,
            error=outcome.error,
        )
    return decision
