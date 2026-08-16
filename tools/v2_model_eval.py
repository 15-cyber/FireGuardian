"""
============================================================
FireGuardian V2 Phase 5-4 - V2 模型评估（复用 baseline 评估逻辑）

用法：python tools/v2_model_eval.py [--model models/v2_15k_best.pt]
输出：tools/v2_model_evaluation.json + runs/v2_model/v2_15k/{eval_curves,error_analysis}
============================================================
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tools.evaluate_baseline_v1 import (  # noqa: E402
    EVAL_CONF,
    collect_error_cases,
    compute_all_metrics,
    export_error_images,
    image_presence_confusion,
    load_test_set,
    negative_fp_evaluation,
    plot_curves,
    run_predictions,
    write_error_md,
)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(_ROOT / "models" / "v2_15k_best.pt"))
    ap.add_argument("--test-txt", default=str(_ROOT / "datasets" / "v2_15k_test.txt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(_ROOT / "tools" / "v2_model_evaluation.json"))
    args = ap.parse_args()

    t0 = time.time()
    model_path = Path(args.model)
    if not model_path.exists():
        print("[错误] 模型不存在:", model_path)
        return 1
    from ultralytics import YOLO

    print("[加载] 模型:", model_path)
    model = YOLO(str(model_path))

    images = load_test_set(Path(args.test_txt))
    print("[数据] test 图片:", len(images))
    n_neg = sum(1 for i in images if not i["gt"])
    print("[数据] 其中负样本:", n_neg)

    print("[预测] 全量推理 conf=0.001 ...")
    preds = run_predictions(model, images, conf=0.001, iou=0.5,
                            imgsz=args.imgsz, device=args.device, batch=args.batch)
    img_dims = []
    for r in images:
        from tools.evaluate_baseline_v1 import read_cv

        img = read_cv(r["image"])
        img_dims.append((img.shape[1], img.shape[0]) if img is not None else (640, 640))
    print("[预测] 完成，总框数:", sum(len(p) for p in preds))

    metrics = compute_all_metrics(images, preds, img_dims)
    print("  总体 mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f" % (
        metrics["mAP50"]["all"], metrics["mAP50-95"]["all"],
        metrics["overall"]["precision"], metrics["overall"]["recall"]))

    cm = image_presence_confusion(images, preds, EVAL_CONF)
    neg_eval = negative_fp_evaluation(images, preds, img_dims)
    for th, v in neg_eval["thresholds"].items():
        print("  conf=%s: 图级误报率=%.4f" % (th, v["image_level_fp_rate"]))

    cases = collect_error_cases(images, preds, img_dims)
    base_dir = _ROOT / "runs" / "v2_model" / "v2_15k"
    err_dir = base_dir / "error_analysis"
    saved = export_error_images(cases, err_dir)
    curve_dir = base_dir / "eval_curves"
    curve_files = plot_curves(metrics, curve_dir)
    write_error_md(cases, err_dir / "error_analysis.md")

    eval_result = {
        "model": str(model_path),
        "model_names": {int(k): v for k, v in model.names.items()},
        "test_set": {"images": len(images), "negative": n_neg},
        "eval_conf": EVAL_CONF,
        "overall": metrics["overall"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "per_class": metrics["per_class"],
        "confusion_matrix": cm,
        "negative_fp": neg_eval,
        "error_cases": {k: len(v) for k, v in cases.items()},
        "error_examples_saved": saved,
        "curves": curve_files,
        "duration_seconds": round(time.time() - t0, 1),
    }
    out_json = Path(args.out)
    out_json.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[保存] 评估结果:", out_json)
    print("[完成] 总耗时: %.1fs" % eval_result["duration_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
