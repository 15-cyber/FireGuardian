"""
============================================================
FireGuardian GitHub 公开版安全审计

检查已跟踪文件：
  - Secrets（sk- 密钥样式、Authorization: Bearer、password=）
  - .env 是否仅 .env.example
  - 模型 .pt / 视频 .mp4 是否入库
  - 大文件（>1MB）
  - 本机绝对路径（Codex 项目 / 火焰数据集 / C:\\Users / F:\\ 等）
  - Git 历史中的密钥（sk-71f9 等真实 Key 片段）

用法：python tools/github_public_audit.py
============================================================
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

ABS_PATTERNS = [
    "Codex项目",
    "火焰数据集",
    r"C:\\Users",
    r"E:\\Codex",
    r"D:\\Python",
    r"F:\\", "F:/",
]

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{16,}",
    r"Authorization: Bearer",
    r"Bearer [A-Za-z0-9]{20,}",
    r"password\s*=\s*['\"]?[A-Za-z0-9]{8,}",
]


def _tracked() -> list:
    return subprocess.check_output(["git", "ls-files"], text=True).splitlines()


def _scan(files, patterns):
    hits = []
    for f in files:
        path = _ROOT / f
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for p in patterns:
            if re.search(p, text):
                hits.append((f, p))
                break
    return hits


def main() -> int:
    files = _tracked()
    results = {}

    results["secret_hits"] = _scan(files, SECRET_PATTERNS)
    results["env_files"] = [f for f in files if re.search(r"(^|/)\.env$", f)]
    results["dotenv_all"] = [f for f in files if ".env" in f]
    results["pt_files"] = [f for f in files if f.endswith(".pt")]
    results["mp4_files"] = [f for f in files if f.endswith(".mp4")]
    results["big_files"] = [
        (f, (_ROOT / f).stat().st_size)
        for f in files
        if (_ROOT / f).is_file() and (_ROOT / f).stat().st_size > 1 * 1024 * 1024
    ]
    results["abs_path_hits"] = _scan(files, ABS_PATTERNS)

    # Git 历史密钥（真实 Key 片段）
    history_hits = []
    for needle in ("sk-71f9", "DEEPSEEK_API_KEY="):
        try:
            out = subprocess.check_output(
                ["git", "log", "--all", "--no-textconv", "--oneline", "-S", needle,
                 "--", "*.py", "*.yaml", "*.md", "*.json", "*.jsonl"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            out = ""
        if out.strip():
            history_hits.append((needle, out.strip().splitlines()))
    results["history_hits"] = history_hits

    ok = (
        not results["secret_hits"]
        and not results["env_files"]
        and not results["pt_files"]
        and not results["mp4_files"]
        and not results["big_files"]
        and not results["abs_path_hits"]
        and not results["history_hits"]
    )

    def fmt(key, val, pass_text="PASS", fail_text="FAIL"):
        status = pass_text if not val else fail_text
        print(f"{key}: {status}")
        for item in (val or [])[:10]:
            print(f"    {item}")
        return status

    print("== FireGuardian GitHub Public Audit ==")
    fmt("Secrets", results["secret_hits"])
    fmt(".env (tracked)", results["env_files"])
    fmt(".env.example present", [] if ".env.example" in results["dotenv_all"] else [".env.example missing"])
    fmt("Model weights", results["pt_files"])
    fmt("Videos", results["mp4_files"])
    fmt("Large files", results["big_files"])
    fmt("Absolute paths", results["abs_path_hits"])
    fmt("History secrets", results["history_hits"])
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
