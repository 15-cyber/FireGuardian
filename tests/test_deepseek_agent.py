# -*- coding: utf-8 -*-
"""V2 Agent - FireGuardianLLMAgent 单元测试（Phase 2，全部 Mock，不消耗 API）。"""
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.deepseek_agent import (
    AgentStatus,
    FireGuardianLLMAgent,
    should_call_agent,
)
from agent.deepseek_client import ChatResult, DeepSeekAPIError
from agent.tool_executor import InMemoryDataSource
from utils.common import DangerLevel, EventStatus, FireDecision, FireEvent, GrowthTrend


def _make_event() -> FireEvent:
    return FireEvent(
        event_id="FE-001",
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


def _make_decision() -> FireDecision:
    return FireDecision(
        event_id="FE-001",
        danger_level=DangerLevel.MEDIUM,
        score=0.48,
        confidence=0.72,
        reasons=["持续烟雾，未见持续火焰"],
    )


def _make_data_source():
    return InMemoryDataSource(
        summaries={
            "FE-001": {
                "found": True,
                "event_id": "FE-001",
                "summary": {"status": "confirmed", "duration_seconds": 3.5},
            }
        },
        decisions={
            "FE-001": {
                "found": True,
                "event_id": "FE-001",
                "decision": {"level": "medium", "score": 0.48},
            }
        },
    )


def _tool_response(name="get_event_summary", reasoning=None, call_id="call_1"):
    return ChatResult(
        content=None,
        reasoning_content=reasoning,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": '{"event_id": "FE-001"}'},
            }
        ],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    )


def _final_response(content=None):
    content = content or json.dumps(
        {
            "event_id": "FE-001",
            "agent_assessment": "SUSPICIOUS",
            "risk_level": "medium",
            "agreement_with_rule": True,
            "possible_cause": "疑似非火灾烟雾",
            "evidence_summary": ["持续烟雾无火焰"],
            "reasoning_summary": "烟雾特征明显但火焰证据不足",
            "recommended_action": "继续观察并人工核查",
            "uncertainty": "medium",
            "tools_used": ["get_event_summary"],
            "agent_confidence": 0.78,
        }
    )
    return ChatResult(
        content=content,
        finish_reason="stop",
        usage={"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
    )


class FakeClient:
    """脚本化响应客户端，行为与 DeepSeekClient 接口一致。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _take(self, kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("出现了多余的 LLM 调用")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def chat_with_tools(self, messages, tools=None, **kwargs):
        return self._take({"messages": messages, "tools": tools, **kwargs})

    def chat(self, messages, **kwargs):
        return self._take({"messages": messages, **kwargs})

    model = "deepseek-v4-flash"


def _make_agent(client, thinking_disabled=True):
    return FireGuardianLLMAgent(
        client=client,
        data_source=_make_data_source(),
        max_tool_rounds=3,
        thinking_disabled=thinking_disabled,
    )


class TestFireGuardianLLMAgent(unittest.TestCase):
    def test_tool_then_final(self):
        client = FakeClient([_tool_response(), _final_response()])
        agent = _make_agent(client)
        outcome = agent.analyze(_make_event(), _make_decision())

        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertEqual(outcome.rounds, 2)  # 1 轮工具 + 1 轮最终输出
        self.assertTrue(outcome.final_json_success)
        self.assertTrue(outcome.schema_success)
        self.assertEqual(outcome.agent_level, "medium")
        self.assertEqual(outcome.agent_assessment, "SUSPICIOUS")
        self.assertEqual(len(outcome.tool_calls), 1)
        self.assertTrue(outcome.tool_calls[0]["success"])

        # 第 1 轮带 tools + thinking disabled
        round1 = client.calls[0]
        self.assertIn("tools", round1)
        self.assertEqual(round1["extra_body"], {"thinking": {"type": "disabled"}})
        # 第二轮消息包含 assistant(tool_calls) 与 tool 回传
        messages = client.calls[1]["messages"]
        roles = [m["role"] for m in messages]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)

    def test_no_tool_needed(self):
        client = FakeClient([_final_response()])
        agent = _make_agent(client)
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertEqual(outcome.rounds, 1)
        self.assertEqual(outcome.tool_calls, [])

    def test_max_three_tool_rounds_then_forced_final(self):
        responses = [
            _tool_response(call_id="c1"),
            _tool_response(call_id="c2"),
            _tool_response(call_id="c3"),
            _final_response(),
        ]
        client = FakeClient(responses)
        agent = _make_agent(client)
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertEqual(outcome.rounds, 4)  # 3 轮工具 + 1 轮强制最终输出
        self.assertEqual(len(outcome.tool_calls), 3)
        self.assertEqual(len(client.calls), 4)
        # 强制最终输出：不带 tools、tool_choice=none、JSON Output、max_tokens=768
        final_call = client.calls[3]
        self.assertNotIn("tools", final_call)
        self.assertEqual(final_call["tool_choice"], "none")
        self.assertEqual(final_call["response_format"], {"type": "json_object"})
        self.assertEqual(final_call["max_tokens"], 768)
        self.assertEqual(final_call["extra_body"], {"thinking": {"type": "disabled"}})

    def test_api_failure_returns_unavailable(self):
        client = FakeClient(
            [DeepSeekAPIError("DeepSeek API 调用失败: timeout")]
        )
        agent = _make_agent(client)
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.UNAVAILABLE)
        self.assertIsNone(outcome.agent_result)
        self.assertIn("timeout", outcome.error)

    def test_invalid_json_returns_invalid_output(self):
        client = FakeClient([_tool_response(), _final_response(content="not json {")])
        agent = _make_agent(client)
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.INVALID_OUTPUT)
        self.assertFalse(outcome.final_json_success)

    def test_schema_failure_returns_invalid_output(self):
        bad = '{"event_id": "FE-001", "risk_level": "extreme"}'
        client = FakeClient([_tool_response(), _final_response(content=bad)])
        agent = _make_agent(client)
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.INVALID_OUTPUT)
        self.assertTrue(outcome.final_json_success)
        self.assertFalse(outcome.schema_success)

    def test_tool_failure_does_not_crash(self):
        # 工具参数非法（event_id 白名单拒绝）→ 工具失败，但 Agent 继续
        tool_bad = ChatResult(
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_event_summary",
                        "arguments": '{"event_id": "bad id!"}',
                    },
                }
            ],
            finish_reason="tool_calls",
            usage={"total_tokens": 50},
        )
        client = FakeClient([tool_bad, _final_response()])
        agent = _make_agent(client)
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertFalse(outcome.tool_calls[0]["success"])
        self.assertIn("event_id", outcome.tool_calls[0]["result_summary"]["error"])

    def test_thinking_echo_when_present(self):
        tool_with_reasoning = _tool_response(reasoning="先思考一下……")
        client = FakeClient([tool_with_reasoning, _final_response()])
        agent = _make_agent(client, thinking_disabled=False)
        outcome = agent.analyze(_make_event(), _make_decision())
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertTrue(outcome.reasoning_seen)
        self.assertTrue(outcome.reasoning_echoed)
        round2_messages = client.calls[1]["messages"]
        assistant = next(m for m in round2_messages if m["role"] == "assistant")
        self.assertEqual(assistant["reasoning_content"], "先思考一下……")
        self.assertIsNone(client.calls[1]["extra_body"])  # thinking 未关闭


class TestShouldCallAgent(unittest.TestCase):
    def test_valid_triggers(self):
        state = {"status": "confirmed", "level": "medium"}
        for trigger in ("event_confirmed", "risk_upgrade", "disagreement", "event_ended"):
            self.assertTrue(should_call_agent(state, trigger))

    def test_invalid_trigger_rejected(self):
        self.assertFalse(should_call_agent({}, "frame_update"))
        self.assertFalse(should_call_agent({}, "event_confirmed_extra"))

    def test_max_calls_limit(self):
        self.assertTrue(should_call_agent({}, "disagreement", calls_so_far=3))
        self.assertFalse(should_call_agent({}, "disagreement", calls_so_far=4))

    def test_non_dict_state_rejected(self):
        self.assertFalse(should_call_agent(None, "event_confirmed"))


if __name__ == "__main__":
    unittest.main()
