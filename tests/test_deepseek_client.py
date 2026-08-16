# -*- coding: utf-8 -*-
"""V2 Agent - DeepSeekClient 单元测试（Phase 1，全部 Mock，不消耗 API 额度）。"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import openai
import httpx2

from agent import deepseek_client
from agent.deepseek_client import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekConfigError,
    DeepSeekEmptyContentError,
    DeepSeekJSONError,
)


def _make_response(content=None, tool_calls=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="deepseek-v4-flash",
        usage=usage
        or SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_calls():
    return [
        SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(
                name="get_event_summary", arguments='{"event_id": "FE-001"}'
            ),
        )
    ]


def _fake_response(status_code: int):
    request = httpx2.Request("POST", "https://api.deepseek.com/chat/completions")
    return httpx2.Response(status_code, request=request)


def _rate_limit_error():
    return openai.RateLimitError(
        "rate limited", response=_fake_response(429), body=None
    )


def _server_error():
    return openai.APIStatusError(
        "server error", response=_fake_response(503), body=None
    )


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


def _patch_client(responses):
    completions = _FakeCompletions(responses)

    def factory(*args, **kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    patcher = patch.object(deepseek_client.openai, "OpenAI", factory)
    patcher.start()
    return completions, patcher


class TestDeepSeekClient(unittest.TestCase):
    def setUp(self):
        self.client = DeepSeekClient(
            api_key="sk-test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            retry_attempts=2,
            timeout_seconds=5,
        )

    def tearDown(self):
        patch.stopall()

    def test_chat_json_parses_valid_json(self):
        completions, _ = _patch_client([_make_response(content='{"ok": true}')])
        result = self.client.chat_json(
            [{"role": "user", "content": "hello"}]
        )
        self.assertEqual(result.parsed_json, {"ok": True})
        self.assertEqual(completions.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(self.client.total_calls, 1)
        self.assertEqual(self.client.total_tokens, 15)

    def test_chat_json_invalid_json_raises(self):
        completions, _ = _patch_client([_make_response(content="not json {")])
        with self.assertRaises(DeepSeekJSONError):
            self.client.chat_json([{"role": "user", "content": "hi"}])
        self.assertEqual(len(completions.calls), 1)  # 解析失败不重试

    def test_chat_json_empty_content_raises(self):
        _patch_client([_make_response(content="")])
        with self.assertRaises(DeepSeekEmptyContentError):
            self.client.chat_json([{"role": "user", "content": "hi"}])

    def test_retry_on_rate_limit_then_success(self):
        completions, _ = _patch_client(
            [_rate_limit_error(), _make_response(content='{"ok": 1}')]
        )
        result = self.client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(result.parsed_json, None)

    def test_retry_on_5xx_then_success(self):
        completions, _ = _patch_client([_server_error(), _make_response(content="ok")])
        result = self.client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(result.content, "ok")

    def test_error_message_never_leaks_api_key(self):
        _patch_client([Exception(f"authentication failed for key {self.client.api_key}")])
        with self.assertRaises(DeepSeekAPIError) as cm:
            self.client.chat([{"role": "user", "content": "hi"}])
        message = str(cm.exception)
        self.assertNotIn("sk-test-key", message)
        self.assertIn("sk-t***", message)

    def test_missing_api_key_raises_config_error(self):
        with patch.object(deepseek_client, "load_env_file", lambda root=None: None), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPSEEK_API_KEY", None)
            with self.assertRaises(DeepSeekConfigError):
                DeepSeekClient()

    def test_tool_calls_parsed(self):
        _patch_client([_make_response(content=None, tool_calls=_tool_calls())])
        registry_tools = [{"type": "function", "function": {"name": "get_event_summary"}}]
        result = self.client.chat_with_tools(
            [{"role": "user", "content": "查一下事件"}], tools=registry_tools
        )
        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call["function"]["name"], "get_event_summary")
        self.assertEqual(call["id"], "call_1")

    def test_repr_masks_key(self):
        self.assertNotIn("sk-test-key", repr(self.client))
        self.assertIn("sk-t***", repr(self.client))


if __name__ == "__main__":
    unittest.main()
