"""
============================================================
FireGuardian - 冒烟训练一次性入口 (Smoke Train Entry)
============================================================
按照《8.FireGuardian M2 首次真实冒烟训练任务.docx》执行：
- 必须通过 M2 YOLOTrainer 训练（不绕过，不使用 CLI yolo train）
- 数据集: datasets/fire_smoke_smoke.yaml（train 3000 / valid 600）
- 参数: weights=yolo11n.pt, epochs=3, batch=16, imgsz=640, device=0
- 输出: runs/smoke_train/smoke_v1/，并自动复制到 models/best.pt
- 记录每 epoch 指标到 tools/smoke_train_metrics.jsonl
- 汇总结果到 tools/smoke_train_summary.json

用法:
  python tools/smoke_train_entry.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from train.yolo_trainer import YOLOTrainer

SEED = 42
METRICS_PATH = Path(__file__).resolve().parent / "smoke_train_metrics.jsonl"
SUMMARY_PATH = Path(__file__).resolve().parent / "smoke_train_summary.json"


def main() -> int:
    # 固定随机种子（尽量复现）
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"[seed] 固定随机种子: {SEED}")

    metrics_file = []
    t_start = time.time()

    def on_epoch(info: dict):
        elapsed = round(time.time() - t_start, 1)
        rec = {
            "kind": info.get("kind"),
            "epoch": info.get("epoch"),
            "total_epochs": info.get("epochs"),
            "loss": info.get("loss"),
            "precision": info.get("precision"),
            "recall": info.get("recall"),
            "mAP50": info.get("mAP50"),
            "mAP50-95": info.get("mAP50-95"),
            "best_map": info.get("best_map"),
            "elapsed_s": elapsed,
            "eta_s": info.get("eta_s"),
        }
        metrics_file.append(rec)
        if info.get("kind") == "epoch":
            print(f"[记录] epoch {rec['epoch']} | loss {rec['loss']} | "
                  f"P {rec['precision']} | R {rec['recall']} | "
                  f"mAP50 {rec['mAP50']} | mAP50-95 {rec['mAP50-95']} | "
                  f"elapsed {elapsed}s | eta {rec['eta_s']}s")

    trainer = YOLOTrainer(
        weights="yolo11n.pt",
        dataset_yaml="datasets/fire_smoke_smoke.yaml",
        project="runs/smoke_train",
        name="smoke_v1",
    )

    best_path = trainer.train(
        epochs=3,
        batch=16,
        imgsz=640,
        device=0,
        callback=on_epoch,
    )

    total_seconds = round(time.time() - t_start, 1)
    print(f"\n[结果] best.pt 路径: {best_path}")
    print(f"[结果] 训练总耗时: {total_seconds}s")

    # 保存指标
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        for rec in metrics_file:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[保存] 指标已写入: {METRICS_PATH}")

    # 汇总
    epoch_recs = [r for r in metrics_file if r.get("kind") == "epoch"]
    summary = {
        "seed": SEED,
        "best_path": str(best_path),
        "total_seconds": total_seconds,
        "epochs": len(epoch_recs),
        "per_epoch": epoch_recs,
        "best": epoch_recs[-1] if epoch_recs else None,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[保存] 汇总已写入: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())