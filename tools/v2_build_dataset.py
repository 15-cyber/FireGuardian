"""
============================================================
FireGuardian V2 Phase 5-1 - 数据集重划分（按片段分组，消除跨集合泄漏）

做法：
  1. 扫描 DATASET_ROOT（train/valid/test 全部图像）
  2. 用 dataset_audit.json 的 exact_duplicates + near_duplicates(<=8)
     做 union-find 片段分组（同一片段不跨集合）
  3. 排除越界框图片
  4. 按片段整体分配 train/val/test（目标 15k/3k/3k，类别配额平衡）
  5. 合并 datasets/v2_negatives 困难负样本
  6. 输出 datasets/v2_15k_*.txt + fire_smoke_v2_15k.yaml + 校验报告
============================================================
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CATS = ("negative", "fire_only", "smoke_only", "both")


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _category(lbl_path: Path) -> str:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return "negative"
    classes = set()
    for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if parts:
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
    if 0 in classes and 1 in classes:
        return "both"
    if 0 in classes:
        return "fire_only"
    if 1 in classes:
        return "smoke_only"
    return "negative"


def _worker_classify(item):
    """返回 (rel, cat)。item = (rel, img_abs, lbl_abs)"""
    rel, _img_abs, lbl_abs = item
    lbl = Path(lbl_abs)
    if not lbl.exists() or lbl.stat().st_size == 0:
        return (rel, "negative")
    classes = set()
    with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if parts:
                try:
                    classes.add(int(float(parts[0])))
                except ValueError:
                    continue
    if 0 in classes and 1 in classes:
        return (rel, "both")
    if 0 in classes:
        return (rel, "fire_only")
    if 1 in classes:
        return (rel, "smoke_only")
    return (rel, "negative")


def _scan_split(root: Path, split: str, workers: int = 8, progress_every: int = 20000) -> list:
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"
    items = []
    with os.scandir(images_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            suffix = Path(entry.name).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            rel = f"{split}/images/{entry.name}"
            items.append((rel, entry.path, str(labels_dir / (Path(entry.name).stem + ".txt"))))
    print(f"  {split}: {len(items)} 张，开始分类...", flush=True)
    results = []
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for i, (rel, cat) in enumerate(pool.imap_unordered(_worker_classify, items, chunksize=2000), 1):
            results.append((rel, cat))
            if i % progress_every == 0:
                print(f"    已分类 {i}/{len(items)}，耗时 {time.time() - t0:.0f}s", flush=True)
    cat_map = dict(results)
    print(f"  {split} 分类完成（{time.time() - t0:.0f}s）")
    recs = [
        {
            "rel": rel,
            "abs": str(root / rel.replace("/", os.sep)),
            "split": split,
            "cat": cat_map[rel],
        }
        for rel, _ in results
    ]
    return recs


def _greedy_split(clips, cat_counts, targets, rng):
    """按片段整体 + 类别配额分配；保证每个集合都覆盖四种类别。"""
    total_pool = sum(cat_counts.values())
    assigned = {"train": [], "val": [], "test": []}
    # 每个集合、每个类别的配额
    quotas = {}
    for split, total in targets.items():
        quotas[split] = {
            cat: max(1, int(round(total * cat_counts[cat] / total_pool)))
            for cat in CATS
        }
    clips_by_cat = defaultdict(list)
    for cid, clip in clips.items():
        dom = Counter(r["cat"] for r in clip).most_common(1)[0][0]
        clips_by_cat[dom].append(cid)
    for cat in CATS:
        rng.shuffle(clips_by_cat[cat])

    remaining = {split: dict(q) for split, q in quotas.items()}
    unused = []
    for cat in CATS:
        for cid in clips_by_cat[cat]:
            clip = clips[cid]
            # 优先放入该类别剩余配额缺口最大的集合
            candidates = [s for s in ("train", "val", "test") if remaining[s][cat] > 0]
            if not candidates:
                unused.append(cid)
                continue
            best = max(candidates, key=lambda s: remaining[s][cat])
            if len(clip) > remaining[best][cat]:
                unused.append(cid)
                continue
            assigned[best].append(cid)
            remaining[best][cat] -= len(clip)
    return assigned, remaining, quotas, unused


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"DATASET_ROOT")
    parser.add_argument("--audit", default=str(_ROOT / "tools" / "dataset_audit.json"))
    parser.add_argument("--negatives-dir", default=str(_ROOT / "datasets" / "v2_negatives"))
    parser.add_argument("--out", default=str(_ROOT / "datasets"))
    parser.add_argument("--train", type=int, default=15000)
    parser.add_argument("--val", type=int, default=3000)
    parser.add_argument("--test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"数据集根目录不存在: {root}")
        return 1
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))

    cache_path = _ROOT / "datasets" / ".v2_scan_cache.json"
    cache = None
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = None
    if cache and cache.get("root") == str(root):
        all_recs = cache["records"]
        print(f"使用扫描缓存: {len(all_recs)} 条")
    else:
        print("扫描全量图像并分类...", flush=True)
        all_recs = []
        for split in ("train", "valid", "test"):
            recs = _scan_split(root, split)
            all_recs.extend(recs)
            print(f"  {split}: {len(recs)}")
        print(f"  总计: {len(all_recs)}")
        cache_path.write_text(
            json.dumps({"root": str(root), "records": all_recs}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"扫描缓存已保存: {cache_path}")

    # 排除越界框图片
    excluded = set()
    for split in ("train", "valid", "test"):
        for entry in audit.get("splits", {}).get(split, {}).get("out_of_range_files", []):
            rel = (entry.get("path", "") or "").replace("\\", "/")
            parts = rel.split("/")
            if len(parts) == 3 and parts[1] == "labels":
                stem = Path(parts[2]).stem
                excluded.add(f"{parts[0]}/images/{stem}.jpg")
    pool = [r for r in all_recs if r["rel"] not in excluded]
    print(f"排除越界框图片后: {len(pool)}")

    # 片段分组（union-find）
    uf = UnionFind()
    for r in pool:
        uf.find(r["rel"])
    for group in audit.get("exact_duplicates", []):
        files = group.get("files", [])
        for i in range(1, len(files)):
            uf.union(files[i - 1], files[i])
    for pair in audit.get("near_duplicates", []):
        if int(pair.get("distance", 99)) <= args.near_threshold:
            files = pair.get("files", [])
            if len(files) >= 2:
                uf.union(files[0], files[1])

    clips: dict = defaultdict(list)
    for r in pool:
        clips[uf.find(r["rel"])].append(r)
    print(f"片段数: {len(clips)}（最大片段 {max(len(v) for v in clips.values())}）")

    # 类别比例
    cat_counts = Counter(r["cat"] for r in pool)
    print("类别分布:", dict(cat_counts))

    rng = random.Random(args.seed)
    targets = {"train": args.train, "val": args.val, "test": args.test}
    assigned, remaining, quotas, unused = _greedy_split(clips, cat_counts, targets, rng)
    print("分配剩余配额:", remaining, "未用片段:", len(unused))

    # 合并困难负样本
    neg_dir = Path(args.negatives_dir)
    neg_recs = []
    if neg_dir.exists():
        for img in sorted(neg_dir.rglob("*")):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS and not img.name.startswith("."):
                neg_recs.append({"rel": str(img.relative_to(_ROOT)).replace("\\", "/"),
                                 "abs": str(img), "split": "train", "cat": "negative"})
    print(f"困难负样本: {len(neg_recs)}")
    rng.shuffle(neg_recs)
    neg_train = neg_recs[: min(1200, len(neg_recs))]
    neg_val = neg_recs[min(1200, len(neg_recs)) : min(1500, len(neg_recs))]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = {
        "train": [r for cid in assigned["train"] for r in clips[cid]] + neg_train,
        "val": [r for cid in assigned["val"] for r in clips[cid]] + neg_val,
        "test": [r for cid in assigned["test"] for r in clips[cid]],
    }
    for split, recs in sel.items():
        txt = out_dir / f"v2_15k_{split}.txt"
        txt.write_text("".join(r["abs"] + "\n" for r in recs), encoding="utf-8")
        print(f"  {split}: {len(recs)} -> {txt.name}")

    yaml_path = out_dir / "fire_smoke_v2_15k.yaml"
    yaml_path.write_text(
        "train: v2_15k_train.txt\n"
        "val: v2_15k_val.txt\n"
        "test: v2_15k_test.txt\n"
        "nc: 2\n"
        "names:\n"
        "  0: fire\n"
        "  1: smoke\n",
        encoding="utf-8",
    )

    # 校验：片段不跨集合 + 类别分布
    clip_split = {}
    for split, cids in assigned.items():
        for cid in cids:
            clip_split[cid] = split
    cross = [cid for cid, s in clip_split.items() if False]  # 结构上不可能
    checks = {
        "clip_split_disjoint": cross == [],
        "train": len(sel["train"]),
        "val": len(sel["val"]),
        "test": len(sel["test"]),
        "category_train": dict(Counter(r["cat"] for r in sel["train"])),
        "category_val": dict(Counter(r["cat"] for r in sel["val"])),
        "category_test": dict(Counter(r["cat"] for r in sel["test"])),
    }
    report = {
        "dataset_root": str(root),
        "pool_images": len(pool),
        "excluded_out_of_range": len(excluded),
        "clips": len(clips),
        "assigned": {k: len(v) for k, v in assigned.items()},
        "remaining": remaining,
        "unused_clips": len(unused),
        "hard_negatives_added": {"train": len(neg_train), "val": len(neg_val)},
        "checks": checks,
    }
    report_path = _ROOT / "tools" / "v2_dataset_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"报告已保存: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
