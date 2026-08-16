"""
============================================================
V2 Agent - FireGuardianLLMAgent（Phase 2）

主链路：
  M7 Rule Decision
    → DeepSeek Agent（主动取证，最多 3 轮 Tool Calling）
    → Agent Analysis
    → Hybrid Decision Coordinator
    → Final Decision

能力：
  1. 接收 EventContext（M6）与 FireDecision（M7）；
  2. 需要信息时主动调用工具（get_event_summary / get_rule_decision /
     get_detection_history / get_event_evidence）；
  3. 最多执行 3 轮 Tool Calling，最终输出严格 Schema JSON；
  4. API 失败 → AgentStatus.UNAVAILABLE；
  5. 非法 JSON / Schema 失败 → AgentStatus.INVALID_OUTPUT；
  6. Tool 失败不崩溃，继续分析或回退。

生产默认：deepseek-v4-flash，thinking = disabled。
============================================================
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_logger import AgentLogger
from agent.deepseek_client import (
    ChatResult,
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekEmptyContentError,
    DeepSeekJSONError,
)
from agent.event_context import build_event_context, summarize_event_context
from agent.prompts import load_prompt
from agent.schemas import AgentResult, SchemaValidationError, ToolResult
from agent.tool_executor import JsonlEventDataSource, ToolDataSource, ToolExecutor
from agent.tool_registry import ToolRegistry
from utils.common import FireDecision, FireEvent

# ---- 调用时机控制（Phase 2）----
AGENT_TRIGGERS = ("event_confirmed", "risk_upgrade", "disagreement", "event_ended")
MAX_LLM_CALLS_PER_EVENT = 4

FINAL_OUTPUT_MAX_TOKENS = 768


class AgentStatus(str, Enum):
    """Agent 结果状态。"""

    OK = "ok"
    UNAVAILABLE = "unavailable"
    INVALID_OUTPUT = "invalid_output"


@dataclass
class AgentOutcome:
    """Agent 分析的结构化结果（供 Hybrid 与日志使用）。"""

    status: AgentStatus = AgentStatus.UNAVAILABLE
    agent_result: Optional[AgentResult] = None
    error: Optional[str] = None
    tool_calls: List[dict] = field(default_factory=list)
    rounds: int = 0
    latency_ms: float = 0.0
    token_usage: dict = field(default_factory=dict)
    final_json_success: bool = False
    schema_success: bool = False
    reasoning_seen: bool = False
    reasoning_echoed: bool = False

    @property
    def agent_level(self) -> Optional[str]:
        return self.agent_result.risk_level if self.agent_result is not None else None

    @property
    def agent_assessment(self) -> Optional[str]:
        return self.agent_result.agent_assessment if self.agent_result is not None else None

    @property
    def tools_used(self) -> List[str]:
        return list(self.agent_result.tools_used) if self.agent_result is not None else []

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "agent_result": self.agent_result.to_dict() if self.agent_result else None,
            "error": self.error,
            "tool_calls": self.tool_calls,
            "rounds": self.rounds,
            "latency_ms": round(self.latency_ms, 1),
            "token_usage": self.token_usage,
            "final_json_success": self.final_json_success,
            "schema_success": self.schema_success,
            "reasoning_seen": self.reasoning_seen,
            "reasoning_echoed": self.reasoning_echoed,
        }


def should_call_agent(event_state: dict, trigger_type: str, calls_so_far: int = 0) -> bool:
    """统一控制 Agent 调用频率（Phase 2 计划书 §7）。

    只允许 4 类触发，且单事件最多 MAX_LLM_CALLS_PER_EVENT 次调用。
    """
    if trigger_type not in AGENT_TRIGGERS:
        return False
    if calls_so_far >= MAX_LLM_CALLS_PER_EVENT:
        return False
    if not isinstance(event_state, dict):
        return False
    return True


def _safe_parse_arguments(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class FireGuardianLLMAgent:
    """DeepSeek 驱动的火灾事件分析 Agent（Phase 2）。"""

    def __init__(
        self,
        client: Optional[DeepSeekClient] = None,
        data_source: Optional[ToolDataSource] = None,
        registry: Optional[ToolRegistry] = None,
        max_tool_rounds: int = 3,
        thinking_disabled: bool = True,
        logger: Optional[AgentLogger] = None,
        allowed_tool_names: Optional[List[str]] = None,
    ):
        self.client = client or DeepSeekClient()
        self.registry = registry or ToolRegistry()
        self.executor = ToolExecutor(
            data_source or JsonlEventDataSource(), self.registry
        )
        self.max_tool_rounds = int(max_tool_rounds)
        self.thinking_disabled = bool(thinking_disabled)
        self.logger = logger or AgentLogger()
        self.allowed_tool_names = (
            list(allowed_tool_names) if allowed_tool_names is not None else None
        )
        self._extra_body = (
            {"thinking": {"type": "disabled"}} if self.thinking_disabled else None
        )
        self.total_calls = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0

    def _tools_for_call(self) -> List[dict]:
        """返回本次调用允许的工具 schema（支持 Without Memory 实验臂）。"""
        schemas = self.registry.schemas()
        if self.allowed_tool_names is None:
            return schemas
        allowed = set(self.allowed_tool_names)
        return [d for d in schemas if d["function"]["name"] in allowed]

    # ---------- 内部工具 ----------

    @staticmethod
    def _accumulate_usage(target: dict, usage: dict) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            target[key] = target.get(key, 0) + int(usage.get(key, 0))

    @staticmethod
    def _build_memory_summary(data: Optional[dict], error: Optional[str]) -> str:
        if error:
            return f"历史查询失败：{error}"
        if not data:
            return ""
        if not data.get("memory_available", True):
            return f"历史记忆不可用（{data.get('reason') or 'unknown'}）"
        summary = data.get("summary") or {}
        matched = int(summary.get("matched_count", 0))
        results = data.get("results") or []
        if matched <= 0 or not results:
            return "未找到达到相似度阈值的历史事件"
        scores = [float(r.get("similarity_score", 0.0)) for r in results]
        return (
            f"检索到 {matched} 个相似历史事件，前 {len(results)} 个相似度 "
            f"{min(scores):.2f}~{max(scores):.2f}；已结合历史证据分析"
        )

    @staticmethod
    def _apply_memory_fields(
        agent_result: AgentResult,
        memory_used_in_run: bool,
        memory_data: Optional[dict],
        memory_error: Optional[str],
    ) -> None:
        """以工具实际执行为准回填 memory 字段（不依赖模型自述），随后重新校验。"""
        if not memory_used_in_run:
            agent_result.memory_used = False
            agent_result.similar_event_count = 0
            agent_result.memory_summary = ""
        else:
            agent_result.memory_used = True
            matched = 0
            if memory_data is not None:
                matched = int((memory_data.get("summary") or {}).get("matched_count", 0))
            agent_result.similar_event_count = matched
            if not (agent_result.memory_summary or "").strip():
                agent_result.memory_summary = FireGuardianLLMAgent._build_memory_summary(
                    memory_data, memory_error
                )
        agent_result.validate()

    def _final_json_call(self, messages: List[dict], usage: dict) -> ChatResult:
        """最终输出调用：不带 tools、tool_choice=none、严格 JSON。"""
        response = self.client.chat(
            messages,
            response_format={"type": "json_object"},
            tool_choice="none",
            max_tokens=FINAL_OUTPUT_MAX_TOKENS,
            extra_body=self._extra_body,
        )
        self._accumulate_usage(usage, response.usage)
        return response

    # ---------- 主入口 ----------

    def analyze(
        self,
        event: FireEvent,
        decision: Optional[FireDecision] = None,
        trigger: str = "event_confirmed",
    ) -> AgentOutcome:
        """分析一个火灾事件，返回 AgentOutcome（绝不抛异常）。"""
        t0 = time.perf_counter()
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        reasoning_seen = False
        reasoning_echoed = False
        outcome = AgentOutcome(status=AgentStatus.UNAVAILABLE)

        try:
            if trigger not in AGENT_TRIGGERS:
                raise ValueError(f"非法触发类型: {trigger!r}")

            ctx = build_event_context(event, decision)
            messages: List[dict] = [
                {"role": "system", "content": load_prompt("system_prompt")},
                {
                    "role": "user",
                    "content": (
                        f"事件上下文：{json.dumps(ctx, ensure_ascii=False)}\n\n"
                        f"事件摘要：{summarize_event_context(ctx)}\n\n"
                        "请分析当前事件。需要更多信息时主动调用工具；"
                        "如果 M7 规则决策已足够，可以直接输出最终 JSON。"
                    ),
                },
            ]

            tool_call_round = 0
            llm_calls = 0
            final: Optional[ChatResult] = None
            executed_tool_calls: List[dict] = []
            memory_used_in_run = False
            memory_data: Optional[dict] = None
            memory_error: Optional[str] = None

            while tool_call_round < self.max_tool_rounds:
                tool_call_round += 1
                llm_calls += 1
                response = self.client.chat_with_tools(
                    messages,
                    tools=self._tools_for_call(),
                    extra_body=self._extra_body,
                )
                self._accumulate_usage(usage, response.usage)
                if response.reasoning_content:
                    reasoning_seen = True

                if not response.tool_calls:
                    final = response
                    break

                # 组装 assistant 消息（thinking 模式下需回传 reasoning_content）
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": response.tool_calls,
                }
                if response.reasoning_content:
                    assistant_msg["reasoning_content"] = response.reasoning_content
                    reasoning_echoed = True
                messages.append(assistant_msg)

                for tc in response.tool_calls:
                    name = tc["function"]["name"]
                    arguments = _safe_parse_arguments(tc["function"]["arguments"])
                    if (
                        self.allowed_tool_names is not None
                        and name not in self.allowed_tool_names
                    ):
                        # 模型幻觉调用了未提供/被禁用的工具：拒绝执行，不视为 Memory 使用
                        tool_result = ToolResult(
                            tool_name=name,
                            success=False,
                            error=f"工具 {name} 不在允许列表（当前实验臂禁用）",
                            latency_ms=0.0,
                        )
                    else:
                        tool_result = self.executor.execute(name, arguments)
                        if name == "get_similar_events":
                            memory_used_in_run = True
                            if tool_result.success:
                                memory_data = tool_result.data
                            else:
                                memory_error = tool_result.error
                    executed_tool_calls.append(
                        {
                            "name": name,
                            "arguments": arguments,
                            "success": tool_result.success,
                            "result_summary": tool_result.to_dict(),
                            "latency_ms": tool_result.latency_ms,
                        }
                    )
                    self.logger.log_tool_call(
                        event_id=event.event_id,
                        trigger=trigger,
                        tool_name=name,
                        arguments=arguments,
                        result_summary=tool_result.to_dict(),
                        latency_ms=tool_result.latency_ms,
                        success=tool_result.success,
                        round_no=tool_call_round,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(tool_result.to_dict(), ensure_ascii=False),
                        }
                    )

                if tool_call_round >= self.max_tool_rounds:
                    # 达到最大工具轮数：强制输出最终 JSON
                    llm_calls += 1
                    final = self._final_json_call(messages, usage)
                    if final.reasoning_content:
                        reasoning_seen = True
                    break

            if final is None:
                llm_calls += 1
                final = self._final_json_call(messages, usage)
                if final.reasoning_content:
                    reasoning_seen = True

            # 解析 + Schema 校验
            parsed = self._parse_final(final)
            outcome.final_json_success = True
            agent_result = AgentResult.from_dict(parsed)
            self._apply_memory_fields(
                agent_result, memory_used_in_run, memory_data, memory_error
            )
            outcome.schema_success = True

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.total_calls += llm_calls
            self.total_latency_ms += latency_ms
            self.total_tokens += int(usage.get("total_tokens", 0))

            outcome.status = AgentStatus.OK
            outcome.agent_result = agent_result
            outcome.tool_calls = executed_tool_calls
            outcome.rounds = llm_calls
            outcome.latency_ms = latency_ms
            outcome.token_usage = usage
            outcome.reasoning_seen = reasoning_seen
            outcome.reasoning_echoed = reasoning_echoed
            return outcome

        except DeepSeekAPIError as exc:
            outcome.status = AgentStatus.UNAVAILABLE
            outcome.error = str(exc)
        except (DeepSeekJSONError, DeepSeekEmptyContentError) as exc:
            outcome.status = AgentStatus.INVALID_OUTPUT
            outcome.error = str(exc)
        except SchemaValidationError as exc:
            outcome.status = AgentStatus.INVALID_OUTPUT
            outcome.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - 任何异常都走安全回退
            outcome.status = AgentStatus.UNAVAILABLE
            outcome.error = f"unexpected_error: {exc}"

        outcome.rounds = llm_calls if "llm_calls" in locals() else outcome.rounds
        outcome.latency_ms = (time.perf_counter() - t0) * 1000.0
        outcome.token_usage = usage
        outcome.reasoning_seen = reasoning_seen
        outcome.reasoning_echoed = reasoning_echoed
        return outcome

    @staticmethod
    def _parse_final(final: ChatResult) -> dict:
        if final.tool_calls:
            raise DeepSeekJSONError("最终输出仍包含工具调用")
        return DeepSeekClient.parse_json_content(final.content)

    def stats(self) -> dict:
        return {
            "model": self.client.model,
            "total_calls": self.total_calls,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_tokens": self.total_tokens,
        }
