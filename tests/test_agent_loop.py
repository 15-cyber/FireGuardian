# -*- coding: utf-8 -*-
"""V2 Agent - 最小 Agent Loop 单元测试（Phase 1，全部 Mock，不消耗 API）。"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import openai
import httpx2

from agent import deepseek_client as _ds_client
from agent.deepseek_client import DeepSeekClient
from agent.tool_executor import InMemoryDataSource
from agent.tool_registry import ToolRegistry
from utils.common import EventStatus, FireEvent, GrowthTrend


def _load_loop_module():
    path = _ROOT / "tools" / "v2_phase1_agent_loop_smoke.py"
    spec = importlib.util.spec_from_file_location("v2_phase1_agent_loop_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOOP = _load_loop_module()


def _make_response(content=None, tool_calls=None, usage=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        choices=[choice],
        model="deepseek-v4-flash",
        usage=usage
        or SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


def _tool_call(name="get_event_summary", arguments='{"event_id": "FE-LOOP-0001"}', call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _fake_response(status_code: int):
    request = httpx2.Request("POST", "https://api.deepseek.com/chat/completions")
    return httpx2.Response(status_code, request=request)


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("出现了多余的 API 调用（超出预期）")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _patch_openai(responses):
    completions = _FakeCompletions(responses)

    def factory(*args, **kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    patcher = patch.object(_ds_client.openai, "OpenAI", factory)
    patcher.start()
    return completions, patcher


def _make_client(retry_attempts=0):
    return DeepSeekClient(
        api_key="sk-test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        retry_attempts=retry_attempts,
        timeout_seconds=5,
    )


def _make_event() -> FireEvent:
    return FireEvent(
        event_id="FE-LOOP-0001",
        status=EventStatus.CONFIRMED,
        duration_seconds=3.5,
        total_frames=30,
        positive_frames=22,
        max_fire_area_ratio=0.12,
        max_smoke_area_ratio=0.40,
        avg_fire_confidence=0.72,
        avg_smoke_confidence=0.81,
        growth_trend=GrowthTrend.STABLE,
    )


def _make_data_source():
    return InMemoryDataSource(
        summaries={
            "FE-LOOP-0001": {
                "found": True,
                "event_id": "FE-LOOP-0001",
                "summary": {"status": "confirmed", "duration_seconds": 3.5},
            }
        }
    )


def _final_json() -> str:
    return json.dumps(
        {
            "event_id": "FE-LOOP-0001",
            "agent_assessment": "SUSPICIOUS",
            "risk_level": "medium",
            "agreement_with_rule": True,
            "possible_cause": "疑似非火灾烟雾",
            "evidence_summary": ["持续烟雾，无持续火焰"],
            "reasoning_summary": "烟雾特征明显但火焰证据不足",
            "recommended_action": "继续观察并人工核查",
            "uncertainty": "medium",
            "tools_used": ["get_event_summary"],
            "agent_confidence": 0.78,
        }
    )


class TestAgentLoop(unittest.TestCase):
    def tearDown(self):
        patch.stopall()

    def test_loop_success_full_chain(self):
        completions, _ = _patch_openai(
            [
                _make_response(content=None, tool_calls=[_tool_call()]),
                _make_response(content=_final_json()),
            ]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rounds"], 2)
        self.assertIsNone(result["error"])
        self.assertEqual(result["agent_result"]["event_id"], "FE-LOOP-0001")
        self.assertEqual(result["agent_result"]["agent_assessment"], "SUSPICIOUS")
        self.assertEqual(result["usage"]["total_tokens"], 300)

        # 第 2 轮 messages 必须包含 assistant(tool_calls) + tool 回传消息
        round2_params = completions.calls[1]
        messages = round2_params["messages"]
        roles = [m["role"] for m in messages]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)
        assistant = next(m for m in messages if m["role"] == "assistant")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        tool_msg = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_1")
        payload = json.loads(tool_msg["content"])
        self.assertTrue(payload["success"])
        self.assertEqual(payload["tool_name"], "get_event_summary")
        # 第 2 轮调用参数：JSON Output、禁止工具、限制 max_tokens、严格输出 Prompt
        self.assertEqual(
            completions.calls[0]["extra_body"], {"thinking": {"type": "disabled"}}
        )
        self.assertEqual(
            round2_params["extra_body"], {"thinking": {"type": "disabled"}}
        )
        self.assertNotIn("tools", round2_params)
        self.assertEqual(round2_params["tool_choice"], "none")
        self.assertEqual(round2_params["response_format"], {"type": "json_object"})
        self.assertEqual(round2_params["max_tokens"], LOOP.ROUND2_MAX_TOKENS)
        final_prompt = messages[-1]["content"]
        self.assertIn("JSON", final_prompt)
        self.assertIn("只输出一个 JSON 对象", final_prompt)
        self.assertIn("禁止代码块", final_prompt)
        self.assertIn("不要再调用任何工具", final_prompt)

        # 诊断信息（不含 Key / 完整 prompt）
        diag = result["diagnostics"]
        self.assertEqual(diag["finish_reason"], "stop")
        self.assertEqual(diag["response_format"], {"type": "json_object"})
        self.assertEqual(diag["tool_choice"], "none")
        self.assertEqual(diag["thinking"], {"thinking": {"type": "disabled"}})
        self.assertEqual(diag["max_tokens"], LOOP.ROUND2_MAX_TOKENS)
        self.assertEqual(diag["completion_tokens"], 50)

    def test_loop_api_failure_returns_status(self):
        _patch_openai([openai.RateLimitError("rate", response=_fake_response(429), body=None)])
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "api_error")
        self.assertIsNotNone(result["error"])

    def test_loop_no_tool_call_returns_status(self):
        _patch_openai([_make_response(content="我直接回答")])
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "no_tool_call")

    def test_loop_unexpected_tool_call_returns_status(self):
        _patch_openai(
            [_make_response(content=None, tool_calls=[_tool_call(name="get_rule_decision")])]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "unexpected_tool_call")

    def test_loop_tool_failure_returns_status(self):
        # 工具参数非法（event_id 白名单拒绝）→ 工具失败，Loop 返回明确状态
        _patch_openai(
            [
                _make_response(
                    content=None,
                    tool_calls=[
                        _tool_call(),
                        _tool_call(
                            name="get_event_summary",
                            arguments='{"event_id": "bad id!"}',
                        )
                    ],
                )
            ]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "tool_error")
        self.assertIn("event_id", result["error"])

    def test_loop_invalid_json_returns_status(self):
        _patch_openai(
            [
                _make_response(content=None, tool_calls=[_tool_call()]),
                _make_response(content="not json {"),
            ]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "invalid_json")
        self.assertEqual(result["diagnostics"]["raw_content_preview"], "not json {")

    def test_loop_round2_empty_content_returns_status(self):
        _patch_openai(
            [
                _make_response(content=None, tool_calls=[_tool_call()]),
                _make_response(content=""),
            ]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "invalid_json")
        self.assertEqual(result["diagnostics"]["raw_content_length"], 0)

    def test_loop_round2_truncated_content_returns_status(self):
        truncated = '{"event_id": "FE-LOOP-0001", "risk_level": "med'
        _patch_openai(
            [
                _make_response(content=None, tool_calls=[_tool_call()]),
                _make_response(content=truncated, finish_reason="length"),
            ]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "invalid_json")
        self.assertEqual(result["diagnostics"]["finish_reason"], "length")
        self.assertEqual(result["diagnostics"]["raw_content_preview"], truncated)

    def test_loop_schema_failure_returns_status(self):
        _patch_openai(
            [
                _make_response(content=None, tool_calls=[_tool_call()]),
                _make_response(content='{"event_id": "FE-LOOP-0001", "risk_level": "extreme"}'),
            ]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "schema_error")
        self.assertIsNotNone(result["diagnostics"])

    def test_loop_second_round_tool_call_exhausts_rounds(self):
        _patch_openai(
            [
                _make_response(content=None, tool_calls=[_tool_call()]),
                _make_response(content=None, tool_calls=[_tool_call(call_id="call_2")]),
            ]
        )
        result = LOOP.run_minimal_agent_loop(
            client=_make_client(),
            data_source=_make_data_source(),
            event=_make_event(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "tool_round_exhausted")
        self.assertEqual(result["rounds"], 2)


if __name__ == "__main__":
    unittest.main()
