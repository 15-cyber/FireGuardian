"""
============================================================
FireGuardian V2 Phase 5-4 - conf 阈值重分析

基于 v2 test 集上 V1 与 V2 的评估结果（同一测试集），比较
mAP / P / R / 负样本误报率，并给出 conf 建议。
输入：tools/v1_on_v2_test_evaluation.json
      tools/v2_model_evaluation.json
输出：tools/v2_conf_analysis.json / tools/v2_conf_analysis.md
============================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    v1 = json.loads((_ROOT / "tools" / "v1_on_v2_test_evaluation.json").read_text(encoding="utf-8"))
    v2 = json.loads((_ROOT / "tools" / "v2_model_evaluation.json").read_text(encoding="utf-8"))

    def row(name, model):
        return {
            "model": name,
            "mAP50": round(model["mAP50"]["all"], 4),
            "mAP50-95": round(model["mAP50-95"]["all"], 4),
            "precision": round(model["overall"]["precision"], 4),
            "recall": round(model["overall"]["recall"], 4),
            "negative_fp": {
                th: {
                    "image_level_fp_rate": round(v["image_level_fp_rate"], 4),
                    "fp_fire_images": v["fp_fire_images"],
                    "fp_smoke_images": v["fp_smoke_images"],
                }
                for th, v in model["negative_fp"]["thresholds"].items()
            },
        }

    rows = [row("v1_baseline", v1), row("v2_15k", v2)]

    # 简单 F1 参考（eval_conf=0.25）
    def f1(r):
        p, r_ = r["precision"], r["recall"]
        return 0.0 if (p + r_) == 0 else round(2 * p * r_ / (p + r_), 4)

    for r in rows:
        r["f1@0.25"] = f1(r)

    # conf 建议：生产默认 0.50（V2 图级误报率 0.84%，比 V1 的 2.61% 改善明显）
    v2_fp = rows[1]["negative_fp"]
    recommendation = {
        "production_conf": 0.50,
        "reason": (
            "V2 在 conf=0.50 时图级误报率 0.84%（V1 同测试集 2.61%），"
            "且保持生产默认不变更符合误报优先策略；"
            "如需更高召回可下调至 0.25（误报 2.52%）。"
        ),
        "alternative": {
            "conf_0_25_fp_rate": v2_fp["0.25"]["image_level_fp_rate"],
            "conf_0_40_fp_rate": v2_fp["0.4"]["image_level_fp_rate"],
            "conf_0_50_fp_rate": v2_fp["0.5"]["image_level_fp_rate"],
        },
    }
    result = {
        "test_set": v1["test_set"],
        "comparison": rows,
        "note": "V1 与 V2 在同一 v2 test 集（无泄漏）上评估；V1 基线原报告为旧泄漏测试集，不直接可比。",
        "recommendation": recommendation,
    }
    out_json = _ROOT / "tools" / "v2_conf_analysis.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# FireGuardian V2 conf 阈值重分析",
        "",
        "| 模型 | mAP50 | mAP50-95 | P@0.25 | R@0.25 | F1@0.25 | 负样本图级误报率(0.25/0.4/0.5) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fp = r["negative_fp"]
        lines.append(
            f"| {r['model']} | {r['mAP50']} | {r['mAP50-95']} | {r['precision']} | "
            f"{r['recall']} | {r['f1@0.25']} | "
            f"{fp['0.25']['image_level_fp_rate']} / {fp['0.4']['image_level_fp_rate']} / "
            f"{fp['0.5']['image_level_fp_rate']} |"
        )
    lines += [
        "",
        f"**建议：生产 conf 保持 {recommendation['production_conf']}**。",
        "",
        recommendation["reason"],
        "",
    ]
    ( _ROOT / "tools" / "v2_conf_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
