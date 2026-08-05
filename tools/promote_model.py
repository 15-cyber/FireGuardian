"""
============================================================
FireGuardian Baseline V1 - 模型晋升工具 (prepare only)
============================================================
按《9.FireGuardian Baseline V1 中等规模训练与精度评估任务.docx》第十五节：
只有在最终报告确认 Baseline V1 明显优于冒烟模型并得到人工确认后，才可执行。
默认只做预检（dry-run），不修改任何文件；加 --execute 才真正替换。

晋升流程：
1. 备份当前 models/best.pt -> models/best.backup_<ts>.pt
2. 验证候选模型类别为 fire/smoke（nc=2, names[0]=fire, names[1]=smoke）
3. 校验候选文件 SHA-256
4. 原子替换 models/best.pt（先写临时文件再 os.replace）
5. 写入模型版本记录 models/model_versions.jsonl
6. 提示重启 GUI 确认新模型加载

用法:
  python tools/promote_model.py [--candidate models/baseline_v1_best.pt] [--execute]
============================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

BEST = _ROOT / "models" / "best.pt"
VERSIONS = _ROOT / "models" / "model_versions.jsonl"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_model(candidate):
    """校验候选模型类别"""
    from ultralytics import YOLO
    model = YOLO(str(candidate))
    names = model.names
    nc = len(names)
    issues = []
    if nc != 2:
        issues.append("nc=%d 应为 2" % nc)
    if names.get(0) != "fire":
        issues.append("names[0]=%r 应为 fire" % names.get(0))
    if names.get(1) != "smoke":
        issues.append("names[1]=%r 应为 smoke" % names.get(1))
    return names, nc, issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=str(_ROOT / "models" / "baseline_v1_best.pt"))
    ap.add_argument("--execute", action="store_true", help="真正执行替换（默认只预检）")
    args = ap.parse_args()

    candidate = Path(args.candidate)
    print("候选模型:", candidate)
    if not candidate.exists():
        print("[错误] 候选模型不存在:", candidate)
        return 1
    if not BEST.exists():
        print("[错误] 当前 models/best.pt 不存在，无法晋升")
        return 1

    names, nc, issues = check_model(candidate)
    print("候选模型类别:", names, "| nc =", nc)
    if issues:
        print("[错误] 类别校验失败:")
        for i in issues:
            print("  -", i)
        return 1

    cand_sha = sha256_file(candidate)
    cur_sha = sha256_file(BEST)
    print("候选 SHA-256:", cand_sha[:16], "...")
    print("当前 SHA-256:", cur_sha[:16], "...")

    if args.execute:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = _ROOT / "models" / ("best.backup_%s.pt" % ts)
        n = 1
        while backup.exists():  # 避免覆盖已有历史备份
            backup = _ROOT / "models" / ("best.backup_%s_%d.pt" % (ts, n))
            n += 1
        shutil.copy2(str(BEST), str(backup))
        print("已备份:", backup, "SHA-256:", sha256_file(backup)[:16], "...")
        tmp = BEST.with_suffix(".pt.tmp")
        shutil.copy2(str(candidate), str(tmp))
        os.replace(str(tmp), str(BEST))
        new_sha = sha256_file(BEST)
        print("已替换 models/best.pt，SHA-256:", new_sha[:16], "...")
        assert new_sha == cand_sha, "替换后 SHA-256 不一致！"
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "candidate": str(candidate),
            "candidate_sha256": cand_sha,
            "backup": str(backup),
            "backup_sha256": sha256_file(backup),
            "names": {int(k): v for k, v in names.items()},
        }
        with open(VERSIONS, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("已写入版本记录:", VERSIONS)
        print("\n请重启 GUI 并确认新模型加载成功。")
        return 0

    print("\n[预检通过] 可以晋升，但任务书要求人工确认后执行。")
    print("确认后运行: python tools/promote_model.py --candidate %s --execute" % args.candidate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
