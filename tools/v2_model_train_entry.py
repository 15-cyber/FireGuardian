"""
============================================================
FireGuardian V2 Phase 5-3 - V2 模型训练入口

按 V2 计划书 §30 执行：
  - weights=yolo11n.pt（不用 best.pt 作为起点）
  - data=datasets/fire_smoke_v2_15k.yaml（片段级无泄漏 + 困难负样本）
  - epochs=40（patience=15），batch=16, imgsz=640, device=0, workers=4, seed=42
  - project=runs/v2_model, name=v2_15k
  - best/last 隔离输出: models/v2_15k_best.pt / models/v2_15k_last.pt（不覆盖 models/best.pt）
  - 逐轮指标 -> tools/v2_model_metrics.jsonl
  - 权威解析 -> tools/v2_model_training_summary.json
============================================================
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from train.yolo_trainer import YOLOTrainer

SEED = 42
EPOCHS = 40
PATIENCE = 15
METRICS_PATH = _ROOT / "tools" / "v2_model_metrics.jsonl"
SUMMARY_PATH = _ROOT / "tools" / "v2_model_training_summary.json"
BEST_TARGET = _ROOT / "models" / "v2_15k_best.pt"
LAST_TARGET = _ROOT / "models" / "v2_15k_last.pt"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_results_csv(csv_path):
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def g(key):
                try:
                    return float(row.get(key, 0.0) or 0.0)
                except ValueError:
                    return 0.0

            rows.append({
                "epoch": int(g("epoch")),
                "time_s": g("time"),
                "train_box_loss": g("train/box_loss"),
                "train_cls_loss": g("train/cls_loss"),
                "train_dfl_loss": g("train/dfl_loss"),
                "val_box_loss": g("val/box_loss"),
                "val_cls_loss": g("val/cls_loss"),
                "val_dfl_loss": g("val/dfl_loss"),
                "precision": g("metrics/precision(B)"),
                "recall": g("metrics/recall(B)"),
                "mAP50": g("metrics/mAP50(B)"),
                "mAP50-95": g("metrics/mAP50-95(B)"),
                "lr_pg0": g("lr/pg0") if "lr/pg0" in row else 0.0,
            })
    return rows


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print("[seed] 固定随机种子:", SEED)

    t_start = time.time()
    callback_records = []

    def on_epoch(info: dict):
        rec = {
            "kind": info.get("kind"),
            "epoch": info.get("epoch"),
            "total_epochs": EPOCHS,
            "loss": info.get("loss"),
            "precision": info.get("precision"),
            "recall": info.get("recall"),
            "mAP50": info.get("mAP50"),
            "mAP50-95": info.get("mAP50-95"),
            "best_map": info.get("best_map"),
            "lr": info.get("lr"),
            "gpu_mem_mb": info.get("gpu_mem_mb"),
            "elapsed_s": round(time.time() - t_start, 1),
            "eta_s": info.get("eta_s"),
        }
        callback_records.append(rec)
        if info.get("kind") == "epoch":
            print("[记录] epoch %s | P %.4f | R %.4f | mAP50 %.4f | mAP50-95 %.4f | lr %s | gpu %.0fMB | elapsed %.0fs" % (
                info.get("epoch"), info.get("precision", 0) or 0,
                info.get("recall", 0) or 0, info.get("mAP50", 0) or 0,
                info.get("mAP50-95", 0) or 0, info.get("lr"),
                info.get("gpu_mem_mb", 0) or 0, rec["elapsed_s"]))

    trainer = YOLOTrainer(
        weights="yolo11n.pt",
        dataset_yaml="datasets/fire_smoke_v2_15k.yaml",
        project="runs/v2_model",
        name="v2_15k",
    )
    print("数据集 yaml:", trainer.dataset_yaml)
    print(f"训练参数: epochs={EPOCHS} batch=16 imgsz=640 device=0 workers=4 patience={PATIENCE}")

    best_path = trainer.train(
        epochs=EPOCHS,
        batch=16,
        imgsz=640,
        device=0,
        patience=PATIENCE,
        best_target=BEST_TARGET,
        callback=on_epoch,
    )
    total_seconds = round(time.time() - t_start, 1)
    print("\n[结果] best.pt 输出:", best_path)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        for rec in callback_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[保存] 回调指标:", METRICS_PATH)

    save_dir = Path(trainer.model.trainer.save_dir)
    results_csv = save_dir / "results.csv"
    rows = parse_results_csv(results_csv)
    print("[解析] results.csv:", results_csv, "行数:", len(rows))

    cb_by_epoch = {r.get("epoch"): r for r in callback_records if r.get("kind") == "epoch"}
    merged = []
    for row in rows:
        cb = cb_by_epoch.get(row["epoch"], {})
        merged.append({
            "epoch": row["epoch"],
            "total_epochs": EPOCHS,
            "time_s": row["time_s"],
            "train_box_loss": row["train_box_loss"],
            "train_cls_loss": row["train_cls_loss"],
            "train_dfl_loss": row["train_dfl_loss"],
            "val_box_loss": row["val_box_loss"],
            "val_cls_loss": row["val_cls_loss"],
            "val_dfl_loss": row["val_dfl_loss"],
            "precision": row["precision"],
            "recall": row["recall"],
            "mAP50": row["mAP50"],
            "mAP50-95": row["mAP50-95"],
            "lr": row["lr_pg0"],
            "gpu_mem_mb": cb.get("gpu_mem_mb"),
            "elapsed_s": cb.get("elapsed_s"),
        })

    best = max(merged, key=lambda r: r["mAP50"]) if merged else None
    early_stopped = bool(merged) and merged[-1]["epoch"] < EPOCHS

    last_source = save_dir / "weights" / "last.pt"
    copies = []
    if BEST_TARGET.exists():
        copies.append({"file": str(BEST_TARGET), "sha256": sha256_file(BEST_TARGET)})
    if last_source.exists():
        shutil.copy2(str(last_source), str(LAST_TARGET))
        copies.append({"file": str(LAST_TARGET), "sha256": sha256_file(LAST_TARGET)})
    print("[校验] 模型副本:", json.dumps(copies, ensure_ascii=False))

    summary = {
        "seed": SEED,
        "dataset_yaml": "datasets/fire_smoke_v2_15k.yaml",
        "best_path": str(best_path),
        "best_target": str(BEST_TARGET),
        "last_target": str(LAST_TARGET),
        "total_seconds": total_seconds,
        "epochs_run": len(merged),
        "early_stopped": early_stopped,
        "best_epoch": best["epoch"] if best else None,
        "best": best,
        "per_epoch": merged,
        "copies_sha256": copies,
        "note": "结果以 results.csv 为权威；模型未覆盖 models/best.pt",
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[保存] 训练摘要:", SUMMARY_PATH)
    print("[完成] 总耗时: %.1fs" % total_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
