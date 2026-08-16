"""
============================================================
FireGuardian V2 Phase 1 - DeepSeek 真实 API 最小冒烟测试

用法：
    python tools/v2_phase1_api_smoke.py

覆盖：
  1. JSON Output（response_format=json_object）可用且可解析；
  2. Tool Calling（Function Calling）可用，模型能返回工具调用。

安全约束：
  - 不打印 API Key（只显示掩码）；
  - 不写入任何日志文件；
  - 不执行工具，只验证模型能发起工具调用（执行循环在 Phase 2）。
============================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.deepseek_client import DeepSeekClient
from agent.tool_registry import ToolRegistry


def main() -> int:
    client = DeepSeekClient()
    print(f"model    : {client.model}")
    print(f"base_url : {client.base_url}")
    print(f"api_key  : {client._masked_key}")
    print("-" * 60)

    # 1) JSON Output 冒烟
    result_json = client.chat_json(
        [
            {"role": "system", "content": "你是测试助手，只输出 JSON。"},
            {"role": "user", "content": '请输出 {"ok": true, "value": 42}'},
        ]
    )
    print(
        f"[1] JSON Output  : parsed={result_json.parsed_json is not None} "
        f"latency={result_json.latency_ms:.1f}ms tokens={result_json.usage}"
    )
    print(f"    content      : {result_json.parsed_json}")

    # 2) Tool Calling 冒烟（不执行工具）
    registry = ToolRegistry()
    result_tool = client.chat_with_tools(
        [
            {
                "role": "system",
                "content": "你是 FireGuardian 测试 Agent，需要调用工具获取事件信息。",
            },
            {
                "role": "user",
                "content": "请调用 get_event_summary 查询事件 FE-LOG-0001 的摘要。",
            },
        ],
        tools=registry.schemas(),
    )
    names = [tc["function"]["name"] for tc in result_tool.tool_calls]
    print(
        f"[2] Tool Calling  : tool_calls={names} "
        f"latency={result_tool.latency_ms:.1f}ms tokens={result_tool.usage}"
    )

    ok = result_json.parsed_json is not None and bool(names)
    print("-" * 60)
    print(f"SMOKE_RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
