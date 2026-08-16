"""
V2 Agent - Prompt 加载器
Prompt 统一放在 agent/prompts/*.md，避免写死在 Python 代码中。
"""
from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """按名称加载 Prompt 文本（自动补 .md 后缀）。"""
    filename = name if name.endswith(".md") else f"{name}.md"
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt 文件不存在: {path}")
    return path.read_text(encoding="utf-8").strip()
