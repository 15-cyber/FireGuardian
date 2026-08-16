# -*- coding: utf-8 -*-
"""V2 Agent - Memory Agent 集成单元测试（Phase 3-B，全部 Mock，不消耗 API）。"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent_memory import AgentMemory, MemoryAgentDataSource
from agent.deepseek_agent import AgentStatus, FireGuardianLLMAgent
from agent.deepseek_client import ChatResult
from agent.hybrid_decision import run_hybrid_pipeline
from agent.memory_normalizer import build_query_record
from agent.tool_executor import InMemoryDataSource
from tests.memory_case_defs import CASES
from tests.test_deepseek_agent import FakeClient, _final_response


def _tool_response(name, arguments, call_id="call_1"):
    return ChatResult(
        content=None,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    )


def _base_data_source(event_id: str) -> InMemoryDataSource:
    return InMemoryDataSource(
        summaries={
            event_id: {
                "found": True,
                "event_id": event_id,
                "summary": {"status": "confirmed", "duration_seconds": 3.0},
            }
        },
        decisions={
            event_id: {
                "found": True,
                "event_id": event_id,
                "decision": {"level": "medium", "score": 0.5},
            }
        },
    )


def _case(case_id: str):
    return next(c for c in CASES if c["case_id"] == case_id)


def _make_agent(client, case, with_memory=True, allowed_tool_names=None):
    event = case["event"]
    memory = AgentMemory(events_path=case["history"])
    if with_memory:
        data_source = MemoryAgentDataSource(
            base=_base_data_source(event.event_id),
            memory=memory,
            query_provider=lambda eid: (
                build_query_record(event, case["decision"]) if eid == event.event_id else None
            ),
        )
    else:
        data_source = _base_data_source(event.event_id)
    return FireGuardianLLMAgent(
        client=client,
        data_source=data_source,
        max_tool_rounds=3,
        thinking_disabled=True,
        allowed_tool_names=allowed_tool_names,
    )


class TestMemoryAgentIntegration(unittest.TestCase):
    def test_without_memory_tool_not_exposed(self):
        case = _case("case_04_fireworks")
        client = FakeClient([_final_response()])
        allowed = [
            "get_event_summary",
            "get_rule_decision",
            "get_detection_history",
            "get_event_evidence",
        ]
        agent = _make_agent(client, case, with_memory=False, allowed_tool_names=allowed)
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        names = [d["function"]["name"] for d in client.calls[0]["tools"]]
        self.assertEqual(sorted(names), sorted(allowed))
        self.assertNotIn("get_similar_events", names)
        self.assertFalse(outcome.agent_result.memory_used)
        self.assertEqual(outcome.agent_result.similar_event_count, 0)

    def test_agent_calls_memory_and_postfills_fields(self):
        case = _case("case_02_smoke_only")
        client = FakeClient(
            [
                _tool_response(
                    "get_similar_events",
                    '{"event_id": "EV-CASE-02", "top_k": 5, "min_similarity": 0.65}',
                ),
                _final_response(),
            ]
        )
        agent = _make_agent(client, case, with_memory=True)
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        memory_call = next(
            tc for tc in outcome.tool_calls if tc["name"] == "get_similar_events"
        )
        self.assertTrue(memory_call["success"])
        result = outcome.agent_result
        self.assertTrue(result.memory_used)
        self.assertGreater(result.similar_event_count, 0)
        self.assertIn("相似历史事件", result.memory_summary)

    def test_memory_empty_matched_zero(self):
        case = _case("case_05_normal")
        with tempfile.TemporaryDirectory() as tmp:
            empty_history = Path(tmp) / "empty_events.jsonl"
            empty_history.write_text("", encoding="utf-8")
            client = FakeClient(
                [
                    _tool_response(
                        "get_similar_events",
                        '{"event_id": "EV-CASE-05", "top_k": 5, "min_similarity": 0.65}',
                    ),
                    _final_response(),
                ]
            )
            memory = AgentMemory(events_path=empty_history)
            agent = FireGuardianLLMAgent(
                client=client,
                data_source=MemoryAgentDataSource(
                    base=_base_data_source(case["event"].event_id),
                    memory=memory,
                    query_provider=lambda eid: (
                        build_query_record(case["event"], case["decision"])
                        if eid == case["event"].event_id
                        else None
                    ),
                ),
                max_tool_rounds=3,
                thinking_disabled=True,
            )
            outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertTrue(outcome.agent_result.memory_used)
        self.assertEqual(outcome.agent_result.similar_event_count, 0)
        self.assertIn("未找到达到相似度阈值", outcome.agent_result.memory_summary)

    def test_memory_unavailable_history_missing(self):
        case = _case("case_01_real_fire")
        missing = Path(tempfile.gettempdir()) / "no_such_history_phase3b.jsonl"
        client = FakeClient(
            [
                _tool_response(
                    "get_similar_events",
                    '{"event_id": "EV-CASE-01", "top_k": 5}',
                ),
                _final_response(),
            ]
        )
        memory = AgentMemory(events_path=missing)
        agent = FireGuardianLLMAgent(
            client=client,
            data_source=MemoryAgentDataSource(
                base=_base_data_source(case["event"].event_id),
                memory=memory,
                query_provider=lambda eid: (
                    build_query_record(case["event"], case["decision"])
                    if eid == case["event"].event_id
                    else None
                ),
            ),
            max_tool_rounds=3,
            thinking_disabled=True,
        )
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertTrue(outcome.agent_result.memory_used)
        self.assertIn("历史记忆不可用", outcome.agent_result.memory_summary)

    def test_memory_tool_failure_agent_continues(self):
        case = _case("case_06_ambiguous")
        client = FakeClient(
            [
                _tool_response(
                    "get_similar_events",
                    '{"event_id": "NO-SUCH-EVENT", "top_k": 5}',
                ),
                _final_response(),
            ]
        )
        memory = AgentMemory(events_path=case["history"])
        agent = FireGuardianLLMAgent(
            client=client,
            data_source=MemoryAgentDataSource(
                base=_base_data_source(case["event"].event_id),
                memory=memory,
                query_provider=lambda eid: None,  # 当前事件不存在
            ),
            max_tool_rounds=3,
            thinking_disabled=True,
        )
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        memory_call = next(
            tc for tc in outcome.tool_calls if tc["name"] == "get_similar_events"
        )
        self.assertFalse(memory_call["success"])
        self.assertTrue(outcome.agent_result.memory_used)
        self.assertEqual(outcome.agent_result.similar_event_count, 0)
        self.assertIn("历史查询失败", outcome.agent_result.memory_summary)

    def test_m7_high_not_downgraded_with_memory(self):
        case = _case("case_01_real_fire")
        client = FakeClient([_final_response()])
        agent = _make_agent(client, case, with_memory=True)
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        # 假设 Agent 输出低等级，Hybrid 仍必须保持 M7=high
        outcome.agent_result.risk_level = "low"
        hybrid = run_hybrid_pipeline(
            rule_level=case["decision"].danger_level,
            outcome=outcome,
            event_id=case["event"].event_id,
            model="deepseek-v4-flash",
        )
        self.assertEqual(hybrid.final_level, "high")

    def test_model_provided_summary_preserved_count_overridden(self):
        case = _case("case_04_fireworks")
        final = json.loads(_final_response().content)
        final["memory_used"] = True
        final["memory_summary"] = "自定义摘要：与历史烟花模式相似"
        client = FakeClient(
            [
                _tool_response(
                    "get_similar_events",
                    '{"event_id": "EV-CASE-04", "top_k": 5, "min_similarity": 0.65}',
                ),
                ChatResult(
                    content=json.dumps(final, ensure_ascii=False),
                    finish_reason="stop",
                    usage={"total_tokens": 200},
                ),
            ]
        )
        agent = _make_agent(client, case, with_memory=True)
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertTrue(outcome.agent_result.memory_used)
        self.assertGreater(outcome.agent_result.similar_event_count, 0)
        self.assertEqual(outcome.agent_result.memory_summary, "自定义摘要：与历史烟花模式相似")

    def test_model_claims_memory_without_tool_cleared(self):
        case = _case("case_03_fire_smoke")
        final = json.loads(_final_response().content)
        final["memory_used"] = True
        final["similar_event_count"] = 9
        final["memory_summary"] = "模型谎称用了历史"
        client = FakeClient(
            [
                ChatResult(
                    content=json.dumps(final, ensure_ascii=False),
                    finish_reason="stop",
                    usage={"total_tokens": 200},
                )
            ]
        )
        agent = _make_agent(client, case, with_memory=True)
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        self.assertFalse(outcome.agent_result.memory_used)
        self.assertEqual(outcome.agent_result.similar_event_count, 0)
        self.assertEqual(outcome.agent_result.memory_summary, "")

    def test_hallucinated_memory_tool_blocked_without_memory_arm(self):
        """无 Memory 臂：模型幻觉调用 get_similar_events，必须拒绝执行。"""
        case = _case("case_04_fireworks")
        client = FakeClient(
            [
                _tool_response(
                    "get_similar_events",
                    '{"event_id": "EV-CASE-04", "top_k": 5}',
                ),
                _final_response(),
            ]
        )
        allowed = [
            "get_event_summary",
            "get_rule_decision",
            "get_detection_history",
            "get_event_evidence",
        ]
        agent = _make_agent(client, case, with_memory=False, allowed_tool_names=allowed)
        outcome = agent.analyze(case["event"], case["decision"])
        self.assertEqual(outcome.status, AgentStatus.OK)
        blocked = next(
            tc for tc in outcome.tool_calls if tc["name"] == "get_similar_events"
        )
        self.assertFalse(blocked["success"])
        self.assertIn("不在允许列表", blocked["result_summary"]["error"])
        self.assertFalse(outcome.agent_result.memory_used)
        self.assertEqual(outcome.agent_result.similar_event_count, 0)


if __name__ == "__main__":
    unittest.main()
