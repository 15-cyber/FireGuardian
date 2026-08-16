"""
============================================================
FireGuardian V2 Phase 1 - 最小真实 Agent Loop 冒烟测试

验证链路（最多 2 轮 LLM 调用）：
  DeepSeek(第1轮, 要求工具调用)
    → Tool Call (get_event_summary)
    → ToolExecutor 本地执行
    → Tool Result 按 tool message 格式回传
    → DeepSeek(第2轮, 输出最终 Agent JSON)
    → schemas.AgentResult Schema Validation

用法：
    python tools/v2_phase1_agent_loop_smoke.py

安全约束：
  - 真实 API 只在手动执行本脚本时调用；单元测试全部 Mock；
  - 不打印 API Key（只显示掩码）；不写日志；不执行 Phase 2 逻辑；
  - API 失败 / Tool 失败 / 非法 JSON 均返回明确失败状态，不异常退出。
============================================================
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.deepseek_client import (
    ChatResult,
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekEmptyContentError,
    DeepSeekJSONError,
)
from agent.event_context import build_event_context
from agent.prompts import load_prompt
from agent.schemas import AgentResult, SchemaValidationError
from agent.tool_executor import InMemoryDataSource, ToolDataSource, ToolExecutor
from agent.tool_registry import ToolRegistry
from utils.common import EventStatus, FireEvent, GrowthTrend

ROUND2_MAX_TOKENS = 768
THINKING_DISABLED_EXTRA = {"thinking": {"type": "disabled"}}


def _safe_parse_arguments(raw: str) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _accumulate_usage(target: dict, usage: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[key] = target.get(key, 0) + int(usage.get(key, 0))


def _failure(
    status: str,
    error: str,
    rounds: int,
    usage: dict,
    latency_ms: float,
    agent_result: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
) -> dict:
    return {
        "ok": False,
        "status": status,
        "rounds": rounds,
        "agent_result": agent_result,
        "diagnostics": diagnostics,
        "latency_ms": round(latency_ms, 1),
        "usage": usage,
        "error": error,
    }


def _build_diagnostics(
    second: ChatResult,
    max_tokens_used: int = ROUND2_MAX_TOKENS,
) -> dict:
    """第二轮诊断信息：不含 API Key、不含完整 prompt。"""
    content = second.content or ""
    usage = second.usage or {}
    return {
        "finish_reason": second.finish_reason,
        "response_format": {"type": "json_object"},
        "tool_choice": "none",
        "thinking": dict(THINKING_DISABLED_EXTRA),
        "max_tokens": max_tokens_used,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "raw_content_length": len(content),
        "raw_content_preview": content[:1000],
    }


def run_minimal_agent_loop(
    client: DeepSeekClient,
    data_source: ToolDataSource,
    event: FireEvent,
    registry: Optional[ToolRegistry] = None,
    max_rounds: int = 2,
) -> dict:
    """最小 Agent Loop：第 1 轮工具调用 + 第 2 轮最终 JSON，绝不抛异常。

    返回结构：
      ok / status / rounds / agent_result / latency_ms / usage / error
    status 取值：
      success / api_error / no_tool_call / unexpected_tool_call /
      tool_error / tool_round_exhausted / invalid_json / schema_error /
      unexpected_error
    """
    tool_registry = registry or ToolRegistry()
    executor = ToolExecutor(data_source, tool_registry)
    t0 = time.perf_counter()
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rounds = 0
    diagnostics: Optional[dict] = None
    try:
        ctx = build_event_context(event)
        system_prompt = load_prompt("system_prompt")
        messages: List[dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"事件上下文：{json.dumps(ctx, ensure_ascii=False)}\n\n"
                    "请先调用 get_event_summary 工具获取事件摘要，再继续分析。"
                ),
            },
        ]

        # ---------- 第 1 轮：要求模型调用工具 ----------
        rounds += 1
        first: ChatResult = client.chat_with_tools(
            messages,
            tools=tool_registry.schemas(),
            extra_body=THINKING_DISABLED_EXTRA,
        )
        _accumulate_usage(usage, first.usage)

        if not first.tool_calls:
            return _failure(
                "no_tool_call",
                "第 1 轮未返回任何工具调用",
                rounds, usage, (time.perf_counter() - t0) * 1000.0,
            )
        called_names = [tc["function"]["name"] for tc in first.tool_calls]
        if "get_event_summary" not in called_names:
            return _failure(
                "unexpected_tool_call",
                f"第 1 轮未调用 get_event_summary，收到: {called_names}",
                rounds, usage, (time.perf_counter() - t0) * 1000.0,
            )

        # ---------- 本地执行工具 ----------
        executed: List[dict] = []
        for tc in first.tool_calls:
            tool_result = executor.execute(
                tc["function"]["name"],
                _safe_parse_arguments(tc["function"]["arguments"]),
            )
            executed.append(
                {
                    "tool_call": tc,
                    "tool_result": tool_result.to_dict(),
                }
            )
            if not tool_result.success:
                return _failure(
                    "tool_error",
                    f"工具 {tc['function']['name']} 执行失败: {tool_result.error}",
                    rounds, usage, (time.perf_counter() - t0) * 1000.0,
                )

        # ---------- 回传 Tool Result（正确 tool message 格式） ----------
        messages.append(
            {
                "role": "assistant",
                "content": first.content or "",
                "tool_calls": first.tool_calls,
            }
        )
        for item in executed:
            tc = item["tool_call"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(item["tool_result"], ensure_ascii=False),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": load_prompt("final_output_prompt"),
            }
        )

        # ---------- 第 2 轮：要求输出最终 JSON ----------
        rounds += 1
        second: ChatResult = client.chat(
            messages,
            response_format={"type": "json_object"},
            tool_choice="none",
            max_tokens=ROUND2_MAX_TOKENS,
            extra_body=THINKING_DISABLED_EXTRA,
        )
        _accumulate_usage(usage, second.usage)
        diagnostics = _build_diagnostics(second)

        if second.tool_calls:
            return _failure(
                "tool_round_exhausted",
                f"第 {rounds} 轮仍返回工具调用（已达最大 {max_rounds} 轮）",
                rounds, usage, (time.perf_counter() - t0) * 1000.0,
                diagnostics=diagnostics,
            )
        try:
            second.parsed_json = client.parse_json_content(second.content)
        except (DeepSeekJSONError, DeepSeekEmptyContentError) as exc:
            raw = second.content or ""
            return _failure(
                "invalid_json",
                f"{exc} | 原始内容前300字符: {raw[:300]!r}",
                rounds, usage, (time.perf_counter() - t0) * 1000.0,
                diagnostics=diagnostics,
            )

        # ---------- Schema Validation ----------
        try:
            agent_result = AgentResult.from_dict(second.parsed_json)
        except SchemaValidationError as exc:
            return _failure(
                "schema_error",
                str(exc),
                rounds, usage, (time.perf_counter() - t0) * 1000.0,
                diagnostics=diagnostics,
            )
        return {
            "ok": True,
            "status": "success",
            "rounds": rounds,
            "agent_result": agent_result.to_dict(),
            "diagnostics": diagnostics,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "usage": usage,
            "error": None,
        }
    except DeepSeekAPIError as exc:
        return _failure(
            "api_error", str(exc), rounds, usage, (time.perf_counter() - t0) * 1000.0
        )
    except (DeepSeekJSONError, DeepSeekEmptyContentError) as exc:
        return _failure(
            "invalid_json", str(exc), rounds, usage, (time.perf_counter() - t0) * 1000.0
        )
    except SchemaValidationError as exc:
        return _failure(
            "schema_error", str(exc), rounds, usage, (time.perf_counter() - t0) * 1000.0
        )
    except Exception as exc:  # noqa: BLE001 - 冒烟测试不允许异常退出
        return _failure(
            "unexpected_error", str(exc), rounds, usage, (time.perf_counter() - t0) * 1000.0
        )


def _save_diagnostic_file(result: dict) -> str:
    """失败时保存诊断信息（原始 assistant content 前 1000 字符，不含 Key/完整 prompt）。"""
    path = _ROOT / "logs" / "agent_loop_diagnostic.json"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": result.get("status"),
        "rounds": result.get("rounds"),
        "latency_ms": result.get("latency_ms"),
        "usage": result.get("usage"),
        "diagnostics": result.get("diagnostics"),
        "error": result.get("error"),
        "note": "诊断文件不含 API Key 与完整 prompt",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def main() -> int:
    print("FireGuardian V2 Phase 1 - 最小 Agent Loop 真实冒烟测试")
    print("=" * 64)

    client = DeepSeekClient()
    print(f"model    : {client.model}")
    print(f"base_url : {client.base_url}")
    print(f"api_key  : {client._masked_key}")
    print("-" * 64)

    # 最小 mock FireEvent（M6 视角：秒制时长 + union area 面积特征）
    event = FireEvent(
        event_id="FE-LOOP-SMOKE-0001",
        status=EventStatus.CONFIRMED,
        start_time_seconds=0.0,
        last_time_seconds=3.5,
        duration_seconds=3.5,
        total_frames=30,
        positive_frames=22,
        max_fire_area_ratio=0.12,
        max_smoke_area_ratio=0.40,
        avg_fire_confidence=0.72,
        avg_smoke_confidence=0.81,
        growth_trend=GrowthTrend.STABLE,
        source_type="smoke",
        source_name="agent_loop_smoke",
    )
    data_source = InMemoryDataSource(
        summaries={
            event.event_id: {
                "found": True,
                "event_id": event.event_id,
                "summary": {
                    "status": "confirmed",
                    "duration_seconds": 3.5,
                    "total_frames": 30,
                    "positive_frames": 22,
                    "positive_ratio": 0.7333,
                    "max_fire_area_ratio": 0.12,
                    "max_smoke_area_ratio": 0.40,
                    "avg_fire_confidence": 0.72,
                    "avg_smoke_confidence": 0.81,
                    "growth_trend": "stable",
                    "source_type": "smoke",
                    "source_name": "agent_loop_smoke",
                },
            }
        }
    )
    registry = ToolRegistry()

    result = run_minimal_agent_loop(
        client=client,
        data_source=data_source,
        event=event,
        registry=registry,
        max_rounds=2,
    )

    print(f"状态      : {result['status']}")
    print(f"LLM 轮数  : {result['rounds']}")
    print(f"总耗时    : {result['latency_ms']}ms")
    print(f"Token     : {result['usage']}")
    if result["ok"]:
        diag = result.get("diagnostics") or {}
        print(f"finish_reason: {diag.get('finish_reason')} | "
              f"response_format={diag.get('response_format')} | "
              f"tool_choice={diag.get('tool_choice')} | "
              f"max_tokens={diag.get('max_tokens')} | "
              f"completion_tokens={diag.get('completion_tokens')}")
        print("最终 Agent JSON（已通过 Schema Validation）:")
        print(json.dumps(result["agent_result"], ensure_ascii=False, indent=2))
    else:
        print(f"失败环节  : {result['status']}")
        print(f"错误信息  : {result['error']}")
        diag = result.get("diagnostics")
        if diag:
            print(f"finish_reason: {diag.get('finish_reason')} | "
                  f"response_format={diag.get('response_format')} | "
                  f"tool_choice={diag.get('tool_choice')} | "
                  f"max_tokens={diag.get('max_tokens')} | "
                  f"completion_tokens={diag.get('completion_tokens')} | "
                  f"raw_content_length={diag.get('raw_content_length')}")
            preview = diag.get("raw_content_preview", "")
            if preview:
                print("原始 content 前 500 字符:")
                print(preview[:500])
            saved = _save_diagnostic_file(result)
            print(f"诊断文件已保存: {saved}")
    print("-" * 64)
    print(f"AGENT_LOOP_RESULT: {'PASS' if result['ok'] else 'FAIL'}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
