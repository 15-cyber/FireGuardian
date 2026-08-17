"""
============================================================
FireGuardian GitHub 公开版安全审计

检查已跟踪文件：
  - Secrets（sk- 密钥样式、Authorization 头 / Bearer 前缀、password=）
  - .env 是否仅 .env.example
  - 模型 .pt / 视频 .mp4 是否入库
  - 大文件（>1MB）
  - 本机绝对路径（Codex 项目 / 火焰 数据集 / C 盘用户目录 / F 盘等）
  - Git 历史中的密钥（sk- 长密钥样式 / 环境变量赋值）

用法：python tools/github_public_audit.py
============================================================
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_BS = chr(92)  # 反斜杠（运行时拼装，避免审计脚本自匹配）
ABS_PATTERNS = [
    "Codex" + "\u9879\u76ee",                  # Codex 项目
    "\u706b\u7130" + "\u6570\u636e\u96c6",     # 火焰 数据集
    "C:" + _BS + _BS + "Users",
    "E:" + _BS + _BS + "Codex",
    "D:" + _BS + _BS + "Python",
    "F:" + _BS + _BS,
    "F" + ":/",
]

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{16,}",
    "Authorization" + r": Bearer",
    r"Bearer [A-Za-z0-9]{20,}",
    r"password\s*=\s*['\"]?[A-Za-z0-9]{8,}",
]


def _tracked() -> list:
    return subprocess.check_output(
        ["git", "-c", "core.quotepath=false", "ls-files"],
        text=True, encoding="utf-8",
    ).splitlines()


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
    results["docx_files"] = [f for f in files if f.lower().endswith(".docx")]
    results["big_files"] = [
        (f, (_ROOT / f).stat().st_size)
        for f in files
        if (_ROOT / f).is_file() and (_ROOT / f).stat().st_size > 1 * 1024 * 1024
    ]
    results["abs_path_hits"] = _scan(files, ABS_PATTERNS)

    # Git 历史密钥（排除审计工具自身，避免自匹配）
    history_hits = []
    try:
        revs = subprocess.check_output(["git", "rev-list", "--all"], text=True).splitlines()
    except subprocess.CalledProcessError:
        revs = []
    for pattern in (r"sk-[A-Za-z0-9]{24,}", r"DEEPSEEK_API_KEY=.+"):
        for rev in revs:
            try:
                out = subprocess.check_output(
                    ["git", "grep", "-n", "-I", "-E", pattern, rev, "--", ".",
                     ":(exclude)tools/github_public_audit.py"],
                    text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                continue
            if out.strip():
                history_hits.append((pattern, out.strip().splitlines()[:3]))
                break
    results["history_hits"] = history_hits

    ok = (
        not results["secret_hits"]
        and not results["env_files"]
        and not results["pt_files"]
        and not results["mp4_files"]
        and not results["docx_files"]
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
    fmt("Office docs (.docx)", results["docx_files"])
    fmt("Large files", results["big_files"])
    fmt("Absolute paths", results["abs_path_hits"])
    fmt("History secrets", results["history_hits"])
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
