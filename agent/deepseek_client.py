"""
============================================================
V2 Agent - DeepSeekClient（Phase 1）

通过 OpenAI 兼容接口访问 DeepSeek API：
  base_url: https://api.deepseek.com
  model:    deepseek-v4-flash

职责：
  - API 连接、超时、重试（timeout / 429 / 5xx / 连接错误）
  - JSON Output（response_format=json_object）与解析
  - Tool Calls（Function Calling）
  - 时延 / Token 统计
  - 错误处理（任何错误信息不得包含 API Key）

安全约束：API Key 只从环境变量或 .env 读取，禁止写入源码、yaml 或日志。
============================================================
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 未安装时仅失去 .env 加载能力
    load_dotenv = None

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class DeepSeekConfigError(RuntimeError):
    """配置缺失或非法（例如未找到 API Key）。"""


class DeepSeekAPIError(RuntimeError):
    """API 调用失败（已重试仍失败）。"""


class DeepSeekJSONError(ValueError):
    """模型返回的 JSON 无法解析。"""


class DeepSeekEmptyContentError(ValueError):
    """模型返回空 content。"""


@dataclass
class ChatResult:
    """一次聊天补全的结构化结果。"""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    parsed_json: Optional[dict] = None
    tool_calls: List[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    model: str = ""
    latency_ms: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens", 0))


def load_env_file(root: Optional[Path] = None) -> None:
    """加载项目根目录 .env（API Key 等敏感配置只从这里读取）。"""
    if load_dotenv is not None:
        load_dotenv((root or _ROOT) / ".env", override=False)


def load_llm_agent_config(config_path: Optional[Path] = None) -> dict:
    """加载 config/llm_agent.yaml（只含配置，不含 API Key）。"""
    path = Path(config_path) if config_path else _ROOT / "config" / "llm_agent.yaml"
    if not path.exists():
        raise FileNotFoundError(f"V2 Agent 配置不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("llm_agent", cfg)


def _env_or_default(name: str, default: str, root: Optional[Path] = None) -> str:
    load_env_file(root)
    return os.environ.get(name, "").strip() or default


class DeepSeekClient:
    """DeepSeek API 客户端（OpenAI 兼容）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[Path] = None,
        timeout_seconds: Optional[float] = None,
        retry_attempts: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        cfg = load_llm_agent_config(config_path)
        self.api_key = api_key or _env_or_default(
            cfg.get("api_key_env", "DEEPSEEK_API_KEY"), "", _ROOT
        )
        if not self.api_key:
            raise DeepSeekConfigError(
                "未找到 DEEPSEEK_API_KEY：请在项目根目录 .env 或环境变量中配置"
            )
        self.base_url = (
            base_url
            or _env_or_default(
                cfg.get("base_url_env", "DEEPSEEK_BASE_URL"),
                cfg.get("base_url_default", "https://api.deepseek.com"),
                _ROOT,
            )
        ).rstrip("/")
        self.model = model or _env_or_default(
            cfg.get("model_env", "DEEPSEEK_MODEL"),
            cfg.get("model_default", "deepseek-v4-flash"),
            _ROOT,
        )
        self.timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else float(cfg.get("timeout_seconds", 20))
        )
        self.retry_attempts = int(
            retry_attempts if retry_attempts is not None else cfg.get("retry_attempts", 2)
        )
        self.temperature = float(
            temperature if temperature is not None else cfg.get("temperature", 0.2)
        )
        self.max_tokens = int(max_tokens if max_tokens is not None else cfg.get("max_tokens", 2048))

        self._client: Optional[Any] = None
        self.total_calls = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0
        self._masked_key = (
            f"{self.api_key[:4]}***" if len(self.api_key) >= 4 else "***"
        )

    # ---------- 内部 ----------

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _sanitize(self, text: str) -> str:
        """移除错误文本中的 API Key，防止 Key 进入日志/异常。"""
        if not text:
            return ""
        return text.replace(self.api_key, self._masked_key)

    def _should_retry(self, error: BaseException, attempt: int) -> bool:
        if attempt >= self.retry_attempts:
            return False
        retryable = (
            getattr(openai, "RateLimitError", ()),
            getattr(openai, "APITimeoutError", ()),
            getattr(openai, "APIConnectionError", ()),
        )
        if isinstance(error, retryable):
            return True
        status_error = getattr(openai, "APIStatusError", None)
        if status_error is not None and isinstance(error, status_error):
            return int(getattr(error, "status_code", 0) or 0) >= 500
        return False

    # ---------- 公开接口 ----------

    def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ) -> ChatResult:
        """调用 DeepSeek，带重试；任何异常都包装为 DeepSeekAPIError 且不含 Key。"""
        t0 = time.perf_counter()
        last_error: Optional[BaseException] = None
        effective_max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        for attempt in range(self.retry_attempts + 1):
            try:
                params: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": effective_max_tokens,
                }
                if tools:
                    params["tools"] = tools
                if tool_choice is not None:
                    params["tool_choice"] = tool_choice
                if response_format is not None:
                    params["response_format"] = response_format
                if extra_body is not None:
                    params["extra_body"] = extra_body

                resp = self._get_client().chat.completions.create(**params)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.total_calls += 1
                self.total_latency_ms += latency_ms

                message = resp.choices[0].message
                usage = getattr(resp, "usage", None)
                usage_dict: Dict[str, int] = {}
                if usage is not None:
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = getattr(usage, key, None)
                        if value is not None:
                            usage_dict[key] = int(value)
                self.total_tokens += int(usage_dict.get("total_tokens", 0))

                tool_calls: List[dict] = []
                for tc in getattr(message, "tool_calls", None) or []:
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", ""),
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": getattr(tc.function, "name", ""),
                                "arguments": getattr(tc.function, "arguments", "{}"),
                            },
                        }
                    )
                return ChatResult(
                    content=getattr(message, "content", None),
                    reasoning_content=getattr(message, "reasoning_content", None),
                    tool_calls=tool_calls,
                    finish_reason=getattr(resp.choices[0], "finish_reason", None),
                    model=getattr(resp, "model", self.model),
                    latency_ms=latency_ms,
                    usage=usage_dict,
                )
            except Exception as exc:  # noqa: BLE001 - 统一包装为 DeepSeekAPIError
                last_error = exc
                if not self._should_retry(exc, attempt):
                    raise DeepSeekAPIError(
                        f"DeepSeek API 调用失败: {self._sanitize(str(exc))}"
                    ) from exc
                time.sleep(min(2 ** attempt, 8.0))
        raise DeepSeekAPIError(
            f"DeepSeek API 调用失败（已重试 {self.retry_attempts} 次）: "
            f"{self._sanitize(str(last_error))}"
        )

    def chat_json(self, messages: List[dict], **kwargs: Any) -> ChatResult:
        """调用 DeepSeek 并要求 JSON Output，返回已解析的 parsed_json。"""
        result = self.chat(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        result.parsed_json = self.parse_json_content(result.content)
        return result

    def chat_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        tool_choice: Optional[Any] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ) -> ChatResult:
        """调用 DeepSeek 并允许模型发起工具调用（Function Calling）。"""
        return self.chat(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

    @staticmethod
    def parse_json_content(content: Optional[str]) -> dict:
        """解析模型返回的 JSON（容忍 ```json 代码围栏）。"""
        if not content or not content.strip():
            raise DeepSeekEmptyContentError("模型返回空 content")
        text = content.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DeepSeekJSONError(f"模型返回非法 JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise DeepSeekJSONError(f"JSON 顶层必须是对象，收到: {type(parsed).__name__}")
        return parsed

    def stats(self) -> dict:
        return {
            "model": self.model,
            "total_calls": self.total_calls,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_tokens": self.total_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"DeepSeekClient(model={self.model}, base_url={self.base_url}, "
            f"api_key={self._masked_key})"
        )
