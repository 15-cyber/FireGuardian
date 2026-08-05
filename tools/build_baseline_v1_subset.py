"""
============================================================
FireGuardian Baseline V1 - 中等规模数据子集构建器 v3
============================================================
按《9.FireGuardian Baseline V1 中等规模训练与精度评估任务.docx》执行：
- 数据源: F:\火焰数据集\Dataset (train/valid/test)
- 目标: train 15000 / val 3000 / test 3000，共 21000
- 分层: fire-only 30% / smoke-only 30% / fire+smoke 20% / 负样本 20%
- 过滤: 排除审计报告中越界框对应的 288 张图片；损坏/标签异常（审计为 0）
- 防泄漏:
    * 并行分类 + 分层抽样
    * 只对最终抽样 + 替补做一次读取，同时计算 SHA-256 与 pHash(8x8 DCT)
    * 同一 SHA-256 完全重复组只保留 1 张（跨集合 + 集合内均无重复）
    * pHash 完全一致(距离 0)不得跨集合，冲突用同类替补替换
    * 汉明距离 1..8 的疑似近似重复只报告、不自动删除
- 固定种子 42；不修改/复制/删除原始图片和标签
- 进度写 runs/baseline_build.log（flush）与 runs/baseline_build.state

用法:
  python tools/build_baseline_v1_subset.py [--limit N] [--train-size 15000 ...]
============================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_ROOT = r"F:/火焰数据集/Dataset"
DEFAULT_AUDIT = os.path.join(_PROJECT_ROOT, "tools", "dataset_audit.json")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CATEGORIES = ("fire_only", "smoke_only", "both", "negative")
SPLITS = ("train", "val", "test")
# 数据集真实目录名：数据集使用 valid/，但清单/报告统一用 val/
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
RATIOS = {"fire_only": 0.30, "smoke_only": 0.30, "both": 0.20, "negative": 0.20}
STATE_FILE = os.path.join(_PROJECT_ROOT, "runs", "baseline_build.state")

# 在导入 ultralytics 前设置可写配置目录（避免读取用户 AppData 权限失败）
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(_PROJECT_ROOT, ".venv"))


def log(msg):
    print(msg, flush=True)


def state(msg):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")
    except Exception:
        pass
    log(msg)


def _classify_worker(abs_lbl):
    cls = set()
    try:
        with open(abs_lbl, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.split()
                if parts:
                    try:
                        cls.add(int(float(parts[0])))
                    except ValueError:
                        pass
    except Exception:
        return (abs_lbl, "other")
    if not cls:
        return (abs_lbl, "negative")
    if cls == {0}:
        return (abs_lbl, "fire_only")
    if cls == {1}:
        return (abs_lbl, "smoke_only")
    if cls == {0, 1}:
        return (abs_lbl, "both")
    return (abs_lbl, "other")


def _hash_worker(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        ph = _phash(data)
        return (path, sha, ph, True)
    except Exception:
        return (path, "", None, False)


def _phash(data):
    import cv2
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    f = np.asarray(small, dtype=np.float32)
    dct = cv2.dct(f)
    low = dct[:8, :8].ravel()[1:]
    med = float(np.median(low))
    h = 0
    for i, b in enumerate(low > med):
        if b:
            h |= (1 << i)
    return h


def build_audit_exclusions(audit, root):
    excl = set()
    for split in ("train", "valid", "test"):
        for entry in audit.get("splits", {}).get(split, {}).get("out_of_range_files", []):
            rel = (entry.get("path", "") or "").replace("\\", "/")
            parts = rel.split("/")
            if len(parts) == 3 and parts[1] == "labels":
                stem = Path(parts[2]).stem
                for ext in IMAGE_EXTS:
                    cand = "%s/images/%s%s" % (parts[0], stem, ext)
                    if os.path.exists(os.path.join(root, cand)):
                        excl.add(cand)
                        break
    return excl


def scan_images(root, limit=0):
    recs = []
    for split_key in SPLITS:
        split_dir = SPLIT_DIRS[split_key]
        img_dir = os.path.join(root, split_dir, "images")
        if not os.path.isdir(img_dir):
            continue
        names = sorted(os.listdir(img_dir))
        if limit:
            # 测试模式：种子随机抽样，避免按文件名排序只取到同一前缀（如 FireAndSmoke_*）
            rng = random.Random(42)
            names = rng.sample(names, min(limit, len(names)))
        for name in names:
            if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
                continue
            abs_path = os.path.join(img_dir, name)
            recs.append({
                "abs_path": abs_path,
                "rel_path": "%s/images/%s" % (split_dir, name),
                "split": split_key,
                "label_path": os.path.join(root, split_dir, "labels", Path(name).stem + ".txt"),
                "category": "",
                "sha256": "",
                "phash": None,
            })
    return recs


def classify_parallel(recs, workers):
    paths = [r["label_path"] for r in recs]
    cat_map = {}
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_classify_worker, p): i for i, p in enumerate(paths)}
        for fut in as_completed(futs):
            lbl, cat = fut.result()
            cat_map[lbl] = cat
            done += 1
            if done % 20000 == 0:
                state("  [classify] %d/%d (%.0fs)" % (done, len(paths), time.time() - t0))
    for r in recs:
        r["category"] = cat_map.get(r["label_path"], "other")
    return [r for r in recs if r["category"] in CATEGORIES]


def hash_paths(paths, workers):
    result = {}
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_hash_worker, p): p for p in paths}
        for fut in as_completed(futs):
            p, sha, ph, ok = fut.result()
            done += 1
            if done % 2000 == 0:
                state("  [hash+pHash] %d/%d (%.0fs)" % (done, len(paths), time.time() - t0))
            if ok:
                result[p] = (sha, ph)
    return result


def hamming(a, b):
    return (a ^ b).bit_count()


def parse_yolo_label_strict(abs_lbl):
    try:
        with open(abs_lbl, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        return False, "读取失败: %s" % e
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            return False, "字段数=%d" % len(parts)
        try:
            cid, cx, cy, w, h = [float(x) for x in parts]
        except ValueError:
            return False, "非数字"
        if int(cid) not in (0, 1):
            return False, "class_id=%s" % cid
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            return False, "中心越界"
        if w <= 0 or h <= 0:
            return False, "宽高非正"
        if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6:
            return False, "x 越界"
        if cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            return False, "y 越界"
    return True, ""



def write_subset_md(rep, path):
    lines = []
    a = lines.append
    a("# FireGuardian Baseline V1 数据子集构建报告")
    a("")
    a("- 工具: `%s`" % rep["meta"]["tool"])
    a("- 数据源: `%s`" % rep["meta"]["dataset_root"])
    a("- 审计报告: `%s`" % rep["meta"]["audit_json"])
    a("- 生成时间: %s | 耗时: %.1fs" % (rep["meta"]["generated_at"], rep["meta"]["duration_seconds"]))
    a("- 随机种子: %d | 并行 worker: %d" % (rep["meta"]["seed"], rep["meta"]["workers"]))
    if rep["meta"].get("limit"):
        a("- 测试模式 limit: %d（仅小规模验证，非正式结果）" % rep["meta"]["limit"])
    a("")
    a("## 一、规模与类别分布")
    a("")
    a("| 集合 | fire-only | smoke-only | fire+smoke | 负样本 | 合计(目标) | 合计(实际) |")
    a("|---|---|---|---|---|---|---|")
    for s in ("train", "val", "test"):
        cats = rep["categories"][s]
        a("| %s | %d | %d | %d | %d | %d | %d |" % (
            s, cats["fire_only"], cats["smoke_only"], cats["both"], cats["negative"],
            rep["targets"][s], rep["actual"][s]))
    if rep["shortfalls"]:
        a("")
        a("### 数量不足说明")
        for sf in rep["shortfalls"]:
            a("- `%s/%s`: 目标 %d，实际 %d" % (sf["split"], sf["category"], sf["target"], sf["actual"]))
    a("")
    a("## 二、过滤与防泄漏")
    a("")
    ex = rep["exclusions"]
    a("- 审计越界框图片排除: %d" % ex["audit_out_of_range_images"])
    a("- 哈希失败: %d" % ex["hash_failed"])
    a("- SHA-256 重复移除: 共 %d（跨集合 %d / 集合内 %d）" % (
        ex["sha256_duplicates_removed"]["total"],
        ex["sha256_duplicates_removed"]["cross_split"],
        ex["sha256_duplicates_removed"]["intra_split"]))
    a("- pHash 完全一致(距离0)替换: %d | 无替补删除: %d" % (
        ex["phash_identical_replaced"], ex.get("phash_identical_removed_no_spare", 0)))
    ph = rep["phash"]
    a("- pHash 统计总数: %d" % ph["phash_total"])
    a("- 处理后跨集合 pHash 完全一致配对: %d" % ph["cross_identical_pairs_after_replacement"])
    approx_key = [k for k in ph if k.startswith("approx_pairs_") and k != "approx_pairs_listed"]
    if approx_key:
        th = approx_key[0].split("to")[-1]
        a("- 疑似近似重复(汉明距离 1..%s): %d" % (th, ph[approx_key[0]]))
    if ph.get("approx_pairs_listed"):
        a("- 近似重复样例: %d 对（仅报告，不删除）" % len(ph["approx_pairs_listed"]))
    a("")
    a("## 三、构建后验证")
    a("")
    a("| 检查项 | 结果 | 详情 |")
    a("|---|---|---|")
    for chk in rep["validation"]["checks"]:
        a("| %s | %s | %s |" % (chk["name"], "PASS" if chk["pass"] else "FAIL", chk["detail"]))
    a("")
    a("## 四、结论")
    a("")
    a("最终结论: **%s**" % rep["conclusion"])
    a("")
    a("## 五、输出文件")
    a("")
    for k, v in rep["sample_files"].items():
        a("- `%s`: `%s`" % (k, v))
    a("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT)
    parser.add_argument("--audit-json", default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=15000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=0, help="测试用：每个 split 最多扫描 N 个文件")
    parser.add_argument("--output-dir", default=os.path.join(_PROJECT_ROOT, "datasets"))
    parser.add_argument("--phash-approx-threshold", type=int, default=8)
    args = parser.parse_args()

    t_start = time.time()
    root = os.path.abspath(args.dataset_root)
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    state("=== Baseline V1 子集构建 v3 ===")
    state("数据源: %s | limit=%d" % (root, args.limit))

    with open(args.audit_json, "r", encoding="utf-8") as f:
        audit = json.load(f)
    excl_rel = build_audit_exclusions(audit, root)
    state("审计排除越界图片: %d" % len(excl_rel))

    recs = scan_images(root, args.limit)
    recs = [r for r in recs if r["rel_path"] not in excl_rel]
    state("扫描候选(排除后): %d，开始并行分类..." % len(recs))

    recs = classify_parallel(recs, args.workers)
    state("分类完成: %s" % {c: sum(1 for r in recs if r["category"] == c) for c in CATEGORIES})

    # 按源数据集 split 分层：train 只抽 train/，val 只抽 valid/，test 只抽 test/
    pool = {s: {c: [] for c in CATEGORIES} for s in SPLITS}
    for r in recs:
        pool[r["split"]][r["category"]].append(r)
    for s in SPLITS:
        for c in CATEGORIES:
            rng.shuffle(pool[s][c])
    state("类别池: %s" % {s: {c: len(v) for c, v in pool[s].items()} for s in SPLITS})

    def quotas_for(size):
        q = {c: int(size * r) for c, r in RATIOS.items()}
        diff = size - sum(q.values())
        q["negative"] += diff
        return q

    quotas = {"train": quotas_for(args.train_size),
              "val": quotas_for(args.val_size),
              "test": quotas_for(args.test_size)}
    if args.limit:
        # 测试模式：按实际池子容量缩小目标
        for s in SPLITS:
            for c in CATEGORIES:
                quotas[s][c] = min(quotas[s][c], len(pool[s][c]) // 3)

    selected = {}
    pool_idx = {s: {c: 0 for c in CATEGORIES} for s in SPLITS}
    for s in SPLITS:
        selected[s] = {}
        for c in CATEGORIES:
            n = quotas[s][c]
            take = []
            while len(take) < n and pool_idx[s][c] < len(pool[s][c]):
                take.append(pool[s][c][pool_idx[s][c]])
                pool_idx[s][c] += 1
            selected[s][c] = take
    state("抽样结果: %s" % {s: {c: len(selected[s][c]) for c in CATEGORIES} for s in SPLITS})

    sel_all = [r for s in SPLITS for c in CATEGORIES for r in selected[s][c]]
    paths = [r["abs_path"] for r in sel_all]
    state("需哈希图片: %d" % len(paths))
    hm = hash_paths(paths, args.workers)
    for r in sel_all:
        sha, ph = hm.get(r["abs_path"], ("", None))
        r["sha256"] = sha
        r["phash"] = ph
    hash_failed = [p for p in paths if p not in hm]
    state("哈希失败: %d" % len(hash_failed))

    # SHA-256 去重
    # 替补候选批量预哈希：避免每取一个替补都新建一次进程池（单图哈希太慢）
    spare_buf = {}  # (split, cat) -> list[cand]

    def hash_spare_batch(cands):
        m = hash_paths([c["abs_path"] for c in cands], args.workers)
        for c in cands:
            sha, ph = m.get(c["abs_path"], ("", None))
            c["sha256"] = sha
            c["phash"] = ph

    def fill_spare_buf(split_key, cat, need):
        key = (split_key, cat)
        while len(spare_buf.get(key, [])) < need and pool_idx[split_key][cat] < len(pool[split_key][cat]):
            take = []
            while len(take) < max(need, 128) and pool_idx[split_key][cat] < len(pool[split_key][cat]):
                take.append(pool[split_key][cat][pool_idx[split_key][cat]])
                pool_idx[split_key][cat] += 1
            hash_spare_batch(take)
            spare_buf.setdefault(key, []).extend(take)

    def fetch_spare(split_key, cat, used_sha):
        key = (split_key, cat)
        buf = [c for c in spare_buf.get(key, []) if c["sha256"] and c["sha256"] not in used_sha]
        spare_buf[key] = buf
        if not buf:
            fill_spare_buf(split_key, cat, 64)
            buf = [c for c in spare_buf.get(key, []) if c["sha256"] and c["sha256"] not in used_sha]
            spare_buf[key] = buf
        if buf:
            return buf.pop(0)
        return None

    dup_removed = {"total": 0, "cross_split": 0, "intra_split": 0}
    iters = 0
    while True:
        iters += 1
        sha_map = {}
        conflict = None
        for s in SPLITS:
            for c in CATEGORIES:
                for r in selected[s][c]:
                    if not r["sha256"]:
                        continue
                    if r["sha256"] in sha_map:
                        conflict = (r, sha_map[r["sha256"]])
                        break
                    sha_map[r["sha256"]] = r
                if conflict:
                    break
            if conflict:
                break
        if conflict is None:
            break
        victim, keeper = conflict
        dup_removed["total"] += 1
        if victim["split"] != keeper["split"]:
            dup_removed["cross_split"] += 1
        else:
            dup_removed["intra_split"] += 1
        spare = fetch_spare(victim["split"], victim["category"], set(sha_map.keys()))
        if spare is None:
            idx = next(i for i, r in enumerate(selected[victim["split"]][victim["category"]])
                       if r["abs_path"] == victim["abs_path"])
            del selected[victim["split"]][victim["category"]][idx]
        else:
            idx = next(i for i, r in enumerate(selected[victim["split"]][victim["category"]])
                       if r["abs_path"] == victim["abs_path"])
            selected[victim["split"]][victim["category"]][idx] = spare
        if iters > 20000:
            state("  [warn] 去重迭代过多，停止")
            break
    state("SHA-256 去重移除: %s" % dup_removed)

    # 跨集合 pHash 完全一致（距离 0）
    sel_by_split = {s: [r for c in CATEGORIES for r in selected[s][c]] for s in SPLITS}
    identical = []
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            for a in sel_by_split[s1]:
                if a["phash"] is None:
                    continue
                for b in sel_by_split[s2]:
                    if b["phash"] is None:
                        continue
                    if a["phash"] == b["phash"]:
                        identical.append((a, b))
    state("跨集合 pHash 完全一致配对: %d" % len(identical))

    used_sha = set()
    for s in SPLITS:
        for r in sel_by_split[s]:
            used_sha.add(r["sha256"])
    replaced_phash = 0
    removed_phash_no_spare = 0
    PHASH_RETRY = 8  # 每个 victim 最多尝试的替补数
    for a, b in identical:
        victim = b
        # 同一张图可能出现在多个完全一致对中；先替换后其余对已不成立，需跳过
        sel_by_split = {s: [r for c in CATEGORIES for r in selected[s][c]] for s in SPLITS}
        if not any(r is victim for r in sel_by_split[victim["split"]]):
            continue
        if not any(r is a for r in sel_by_split[a["split"]]):
            continue
        done = False
        for _attempt in range(PHASH_RETRY):
            spare = fetch_spare(victim["split"], victim["category"], used_sha)
            if spare is None:
                break
            ok = True
            for s in SPLITS:
                for r in sel_by_split[s]:
                    if r["phash"] is not None and spare["phash"] is not None and r["phash"] == spare["phash"]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue  # 该替补与已选图 pHash 撞车，尝试下一个
            idx = next(i for i, r in enumerate(selected[victim["split"]][victim["category"]])
                       if r is victim)
            selected[victim["split"]][victim["category"]][idx] = spare
            used_sha.discard(victim["sha256"])
            used_sha.add(spare["sha256"])
            replaced_phash += 1
            done = True
            break
        if not done:
            # 无可用替补：删除该图，确保跨集合无 pHash 完全一致（数量偏差记入报告）
            try:
                idx = next(i for i, r in enumerate(selected[victim["split"]][victim["category"]])
                           if r is victim)
                del selected[victim["split"]][victim["category"]][idx]
                used_sha.discard(victim["sha256"])
                removed_phash_no_spare += 1
            except StopIteration:
                pass
    state("pHash 完全一致替换: %d | 无替补删除: %d" % (replaced_phash, removed_phash_no_spare))

    # 疑似近似（汉明距离 1..threshold，跨集合）
    sel_by_split = {s: [r for c in CATEGORIES for r in selected[s][c]] for s in SPLITS}
    approx_count = 0
    approx_pairs = []
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            for a in sel_by_split[s1]:
                if a["phash"] is None:
                    continue
                for b in sel_by_split[s2]:
                    if b["phash"] is None:
                        continue
                    d = hamming(a["phash"], b["phash"])
                    if 0 < d <= args.phash_approx_threshold:
                        approx_count += 1
                        if len(approx_pairs) < 30:
                            approx_pairs.append({"split_a": s1, "path_a": a["abs_path"],
                                                 "split_b": s2, "path_b": b["abs_path"],
                                                 "hamming": d})
    state("疑似近似重复(1..%d): %d" % (args.phash_approx_threshold, approx_count))

    # 写 txt / yaml
    def write_txt(path, recs):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(r["abs_path"] + "\n")

    train_txt = os.path.join(args.output_dir, "baseline_v1_train.txt")
    val_txt = os.path.join(args.output_dir, "baseline_v1_val.txt")
    test_txt = os.path.join(args.output_dir, "baseline_v1_test.txt")
    yaml_path = os.path.join(args.output_dir, "fire_smoke_baseline_v1.yaml")
    write_txt(train_txt, sel_by_split["train"])
    write_txt(val_txt, sel_by_split["val"])
    write_txt(test_txt, sel_by_split["test"])
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: baseline_v1_train.txt\n"
                "val: baseline_v1_val.txt\n"
                "test: baseline_v1_test.txt\n"
                "nc: 2\n"
                "names:\n"
                "  0: fire\n"
                "  1: smoke\n")
    state("已写出 txt/yaml")

    # 验证
    verification = {"all_pass": True, "checks": []}

    def check(name, ok, detail=""):
        verification["checks"].append({"name": name, "pass": bool(ok), "detail": str(detail)})
        if not ok:
            verification["all_pass"] = False
        state("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, detail))

    missing_img = missing_lbl = 0
    bad_fmt = []
    for s in SPLITS:
        for r in sel_by_split[s]:
            if not os.path.exists(r["abs_path"]):
                missing_img += 1
            if not os.path.exists(r["label_path"]):
                missing_lbl += 1
            else:
                ok, msg = parse_yolo_label_strict(r["label_path"])
                if not ok:
                    bad_fmt.append((r["abs_path"], msg))
    check("all_image_paths_exist", missing_img == 0, "missing=%d" % missing_img)
    check("all_label_paths_exist", missing_lbl == 0, "missing=%d" % missing_lbl)
    check("yolo_label_format", len(bad_fmt) == 0, "bad=%d" % len(bad_fmt))

    try:
        from ultralytics.data.utils import check_det_dataset
        info = check_det_dataset(yaml_path)
        check("check_det_dataset", True,
              "train=%d val=%d test=%d nc=%s names=%s" % (
                  len(info.get("train", [])), len(info.get("val", [])),
                  len(info.get("test", [])), info.get("nc"), info.get("names")))
    except Exception as e:
        check("check_det_dataset", False, str(e))

    try:
        from ultralytics.data.dataset import YOLODataset
        loaded = {}
        for s, txt in (("train", train_txt), ("val", val_txt), ("test", test_txt)):
            n = 0
            ds = YOLODataset(img_path=txt, imgsz=640, augment=False, batch_size=16,
                             data={"nc": 2, "names": {0: "fire", 1: "smoke"}})
            for i in range(min(64, len(ds))):
                ds[i]
                n += 1
            loaded[s] = n
        check("ultralytics_random_load_32", all(v >= 32 for v in loaded.values()), "loaded=%s" % loaded)
    except Exception as e:
        check("ultralytics_random_load_32", False, str(e))

    duration = time.time() - t_start
    flat = {s: {c: len(selected[s][c]) for c in CATEGORIES} for s in SPLITS}
    shortfalls = []
    for s in SPLITS:
        for c in CATEGORIES:
            if flat[s][c] < quotas[s][c]:
                shortfalls.append({"split": s, "category": c,
                                   "target": quotas[s][c], "actual": flat[s][c]})
    cross_identical_after = sum(
        1 for i, s1 in enumerate(SPLITS) for s2 in SPLITS[i + 1:]
        for a in sel_by_split[s1] for b in sel_by_split[s2]
        if a["phash"] is not None and a["phash"] == b["phash"])
    rep = {
        "meta": {"tool": "tools/build_baseline_v1_subset.py",
                 "dataset_root": root, "audit_json": args.audit_json,
                 "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "duration_seconds": round(duration, 1),
                 "seed": args.seed, "workers": args.workers,
                 "limit": args.limit},
        "targets": {s: sum(quotas[s].values()) for s in SPLITS},
        "actual": {s: sum(flat[s].values()) for s in SPLITS},
        "categories": flat,
        "exclusions": {"audit_out_of_range_images": len(excl_rel),
                       "hash_failed": len(hash_failed),
                       "sha256_duplicates_removed": dup_removed,
                       "phash_identical_replaced": replaced_phash,
                       "phash_identical_removed_no_spare": removed_phash_no_spare},
        "phash": {"phash_total": len(sel_by_split["train"]) + len(sel_by_split["val"]) + len(sel_by_split["test"]),
                  "cross_identical_pairs_after_replacement": cross_identical_after,
                  "approx_pairs_1to%d" % args.phash_approx_threshold: approx_count,
                  "approx_pairs_listed": approx_pairs},
        "shortfalls": shortfalls,
        "validation": verification,
        "conclusion": "SUITABLE" if verification["all_pass"] and not shortfalls else "SUITABLE_WITH_WARNINGS",
        "sample_files": {"train": train_txt, "val": val_txt, "test": test_txt, "yaml": yaml_path},
    }
    json_path = os.path.join(_THIS_DIR, "baseline_v1_subset_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    state("报告已写: %s" % json_path)
    md_path = os.path.join(_THIS_DIR, "baseline_v1_subset_report.md")
    write_subset_md(rep, md_path)
    state("报告已写: %s" % md_path)
    state("总耗时: %.1fs" % duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
