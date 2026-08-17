"""
============================================================
FireGuardian - 独立数据集审计工具 (Dataset Audit Tool)
功能（对应审计要求 1-15）：
  1. 扫描 train / valid / test 的图片数、标签数、匹配数
  2. 缺失标签与孤立标签
  3. 损坏、无法用 OpenCV/Pillow 解码的图片
  4. YOLO txt 严格格式校验（class_id center_x center_y width height）
  5. class_id 仅允许 0 / 1
  6. 坐标与宽高合法范围校验
  7. 空标签文件数量及对应图片数量
  8. fire、smoke 的图片级数量与标注框数量
  9. 同时含 fire 和 smoke 的图片数
  10. 不含任何目标的正常负样本数量
  11. train/valid/test 之间基于 SHA-256 的完全重复图片
  12. 可选感知哈希（aHash）近似重复图片（--phash）
  13. 输出异常文件清单（只读，不删除/修改原数据）
  14. 输出 dataset_audit.json 与 dataset_audit.md
  15. 给出是否适合开始 YOLO 冒烟训练的最终结论

用法：
  python tools/dataset_audit.py                                    # 默认数据集
  python tools/dataset_audit.py --dataset-root "DATASET_ROOT"
  python tools/dataset_audit.py --phash                            # 启用近似重复检测
  python tools/dataset_audit.py --limit 200                        # 每集合仅扫描前 200 个（测试用）
  python tools/dataset_audit.py --output-dir tools --workers 8

依赖：Python 3.8+ / opencv-python / numpy / Pillow（FireGuardian venv 已安装）
============================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# ---------------- 常量 ----------------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
LABEL_EXT = '.txt'
ALLOWED_CLASSES = {0, 1}
EPS = 1e-4  # 浮点容差
TOOL_VERSION = '1.0.0'
DEFAULT_ROOT = r'DATASET_ROOT'
DEFAULT_SPLITS = ('train', 'valid', 'test')

# 本数据集文件名前缀与标签内容的约定（仅供参考，不影响结论）
PREFIX_EXPECT = {
    'Fire': 'fire_only',
    'Smoke': 'smoke_only',
    'FireAndSmoke': 'both',
    'FireNorSmoke': 'empty',
}


# ---------------- 基础工具函数 ----------------

def _rel(*parts):
    """生成以 / 分隔的相对路径，便于跨平台展示与 JSON 存储。"""
    return os.path.join(*parts).replace(os.sep, '/')


def _stem_of(rel_path):
    return os.path.splitext(os.path.basename(rel_path))[0]


def _ahash(img_bgr, hash_size=16):
    """平均哈希（aHash）：返回 hash_size^2 位整数（默认 16x16=256 位）。"""

    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small > mean).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a, b):
    return bin(a ^ b).count('1')


def _box_in_range(cx, cy, w, h):
    """坐标与宽高是否合法：cx,cy 在 [0,1]；w,h 在 (0,1]；且框不越出图像。"""
    if not (-EPS <= cx <= 1.0 + EPS):
        return False
    if not (-EPS <= cy <= 1.0 + EPS):
        return False
    if not (0.0 < w <= 1.0 + EPS):
        return False
    if not (0.0 < h <= 1.0 + EPS):
        return False
    if cx - w / 2.0 < -EPS or cx + w / 2.0 > 1.0 + EPS:
        return False
    if cy - h / 2.0 < -EPS or cy + h / 2.0 > 1.0 + EPS:
        return False
    return True


# ---------------- 工作进程：图片扫描 ----------------

def scan_image_file(args):
    """扫描单张图片：尺寸、解码（OpenCV->Pillow 依次尝试）、SHA-256、（可选）感知哈希。
    任何单文件错误都被记录在返回字典中，绝不抛出中断扫描。"""
    split, abs_path, rel_path, do_phash = args
    rec = {
        'split': split, 'stem': _stem_of(rel_path), 'path': rel_path,
        'size_bytes': 0, 'decode_ok': False, 'decoder': None, 'error': None,
        'width': 0, 'height': 0, 'sha256': None, 'phash': None,
    }
    try:
        with open(abs_path, 'rb') as f:
            data = f.read()
        rec['size_bytes'] = len(data)
        rec['sha256'] = hashlib.sha256(data).hexdigest()
        import numpy as np
        import cv2
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            rec['decode_ok'] = True
            rec['decoder'] = 'opencv'
            rec['width'] = w
            rec['height'] = h
            if do_phash:
                rec['phash'] = _ahash(img)
        else:
            try:
                from PIL import Image
                import io
                im = Image.open(io.BytesIO(data))
                im.load()
                w, h = im.size
                rec['decode_ok'] = True
                rec['decoder'] = 'pillow'
                rec['width'] = w
                rec['height'] = h
            except Exception as e:
                rec['error'] = 'OpenCV 与 Pillow 均无法解码: %s' % e
    except Exception as e:
        rec['error'] = '读取失败: %s' % e
    return rec


# ---------------- 工作进程：标签扫描 ----------------

def scan_label_file(args):
    """扫描单个 YOLO 标签：严格格式 / class_id / 范围 / 类别计数。
    任何单文件错误都被记录在返回字典中，绝不抛出中断扫描。"""
    split, abs_path, rel_path = args
    rec = {
        'split': split, 'stem': _stem_of(rel_path), 'path': rel_path,
        'size_bytes': 0, 'empty': False,
        'bad_lines': 0, 'first_error': None,
        'boxes': 0, 'fire_boxes': 0, 'smoke_boxes': 0,
        'invalid_class_boxes': 0, 'out_of_range_boxes': 0, 'blank_lines': 0,
    }
    try:
        rec['size_bytes'] = os.path.getsize(abs_path)
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if not content.strip():
            rec['empty'] = True
            return rec
        for idx, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                rec['blank_lines'] += 1
                continue
            parts = stripped.split()
            if len(parts) != 5:
                rec['bad_lines'] += 1
                if rec['first_error'] is None:
                    rec['first_error'] = '第%d行: 列数=%d(应为5)' % (idx, len(parts))
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                rec['bad_lines'] += 1
                if rec['first_error'] is None:
                    rec['first_error'] = '第%d行: 含非数值列' % idx
                continue
            raw_cid = vals[0]
            cid = int(round(raw_cid))
            cx, cy, w, h = vals[1], vals[2], vals[3], vals[4]
            rec['boxes'] += 1
            if abs(raw_cid - cid) > 1e-9:
                # 非整数 class_id（如 0.5）→ 视为越界类别
                rec['invalid_class_boxes'] += 1
            elif cid == 0:
                rec['fire_boxes'] += 1
            elif cid == 1:
                rec['smoke_boxes'] += 1
            else:
                rec['invalid_class_boxes'] += 1
            if not _box_in_range(cx, cy, w, h):
                rec['out_of_range_boxes'] += 1
    except Exception as e:
        rec['bad_lines'] += 1
        if rec['first_error'] is None:
            rec['first_error'] = '读取失败: %s' % e
    return rec


# ---------------- 进度显示 ----------------

class Progress:
    """轻量进度显示：每 step 比例打印一次（不依赖第三方库）。"""

    def __init__(self, total, label, step=0.05, quiet=False):
        self.total = total
        self.label = label
        self.step = step
        self.quiet = quiet
        self.done = 0
        self.next_mark = step
        self.t0 = time.time()

    def update(self, n=1):
        self.done += n
        if self.quiet or self.total <= 0:
            return
        ratio = self.done / float(self.total)
        if ratio >= self.next_mark:
            self.next_mark += self.step
            sys.stdout.write('\r[%s] %5.1f%% (%d/%d) 耗时 %6.1fs'
                             % (self.label, ratio * 100.0, self.done, self.total,
                                time.time() - self.t0))
            sys.stdout.flush()
            if self.done >= self.total:
                sys.stdout.write('\n')
                sys.stdout.flush()


# ---------------- 枚举 ----------------

def enumerate_split(split_path, split):
    """枚举某集合的 images / labels，返回 {stem: (绝对路径, 相对路径)}。
    使用 os.scandir（Windows 下无需逐文件 stat，避免在慢速盘上耗时）。"""
    img_map = {}
    lbl_map = {}
    img_dir = os.path.join(split_path, 'images')
    lbl_dir = os.path.join(split_path, 'labels')
    if os.path.isdir(img_dir):
        with os.scandir(img_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                stem = os.path.splitext(entry.name)[0]
                img_map[stem] = (entry.path, _rel(split, 'images', entry.name))
    if os.path.isdir(lbl_dir):
        with os.scandir(lbl_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext != LABEL_EXT:
                    continue
                stem = os.path.splitext(entry.name)[0]
                lbl_map[stem] = (entry.path, _rel(split, 'labels', entry.name))
    return img_map, lbl_map
# ---------------- 聚合与检查逻辑 ----------------

def check_prefix(stem, lrec):
    """文件名前缀与标签内容一致性（信息项，仅供参考）。返回不一致说明或 None。"""
    prefix = stem.split('_')[0]
    exp = PREFIX_EXPECT.get(prefix)
    if exp is None or lrec is None:
        return None
    has_fire = lrec['fire_boxes'] > 0
    has_smoke = lrec['smoke_boxes'] > 0
    if exp == 'fire_only':
        return None if (has_fire and not has_smoke) else 'Fire_* 应只含 fire'
    if exp == 'smoke_only':
        return None if (has_smoke and not has_fire) else 'Smoke_* 应只含 smoke'
    if exp == 'both':
        return None if (has_fire and has_smoke) else 'FireAndSmoke_* 应同时含 fire 与 smoke'
    if exp == 'empty':
        return None if (not has_fire and not has_smoke) else 'FireNorSmoke_* 应为负样本'
    return None


def aggregate_split(split, img_map, lbl_map, img_scan, lbl_scan):
    """聚合单个集合的审计结果。"""
    d = {
        'images': len(img_map),
        'labels': len(lbl_map),
        'matched': 0,
        'missing_labels': [],        # 图片存在但标签缺失 → 图片相对路径
        'orphan_labels': [],         # 标签存在但图片缺失 → 标签相对路径
        'corrupt_images': [],        # {path, error, size_bytes}
        'label_errors': [],          # {path, error, bad_lines}
        'empty_labels': [],          # 空标签文件（相对路径）
        'empty_label_images': [],    # 空标签对应的图片（相对路径，无论图片是否可解码）
        'negative_images': [],       # 正常负样本：图片可解码 + 标签为空
        'fire_images': 0,
        'smoke_images': 0,
        'both_images': 0,
        'fire_boxes': 0,
        'smoke_boxes': 0,
        'invalid_class_boxes': 0,
        'out_of_range_boxes': 0,
        'blank_lines': 0,
        'bad_label_lines': 0,
        'out_of_range_files': [],    # {path, boxes}
        'invalid_class_files': [],   # {path, boxes}
        'prefix_mismatch': [],       # 信息项
        'total_size_mb': 0.0,
    }
    common = set(img_map) & set(lbl_map)
    d['matched'] = len(common)
    for stem in sorted(set(img_map) - set(lbl_map)):
        d['missing_labels'].append(img_map[stem][1])
    for stem in sorted(set(lbl_map) - set(img_map)):
        d['orphan_labels'].append(lbl_map[stem][1])
    for stem in common:
        irec = img_scan.get((split, stem))
        lrec = lbl_scan.get((split, stem))
        if irec is None or lrec is None:
            continue
        d['total_size_mb'] += irec['size_bytes'] / (1024.0 * 1024.0)
        if not irec['decode_ok']:
            d['corrupt_images'].append({
                'path': irec['path'],
                'error': irec['error'] or '解码失败',
                'size_bytes': irec['size_bytes'],
            })
        if lrec['empty']:
            d['empty_labels'].append(lrec['path'])
            d['empty_label_images'].append(irec['path'])
            if irec['decode_ok']:
                d['negative_images'].append(irec['path'])
        else:
            if lrec['bad_lines'] > 0:
                d['label_errors'].append({
                    'path': lrec['path'],
                    'error': lrec['first_error'] or '解析错误',
                    'bad_lines': lrec['bad_lines'],
                })
            if lrec['fire_boxes'] > 0:
                d['fire_images'] += 1
            if lrec['smoke_boxes'] > 0:
                d['smoke_images'] += 1
            if lrec['fire_boxes'] > 0 and lrec['smoke_boxes'] > 0:
                d['both_images'] += 1
        d['fire_boxes'] += lrec['fire_boxes']
        d['smoke_boxes'] += lrec['smoke_boxes']
        d['invalid_class_boxes'] += lrec['invalid_class_boxes']
        d['out_of_range_boxes'] += lrec['out_of_range_boxes']
        d['blank_lines'] += lrec['blank_lines']
        d['bad_label_lines'] += lrec['bad_lines']
        if lrec['out_of_range_boxes'] > 0:
            d['out_of_range_files'].append({'path': lrec['path'], 'boxes': lrec['out_of_range_boxes']})
        if lrec['invalid_class_boxes'] > 0:
            d['invalid_class_files'].append({'path': lrec['path'], 'boxes': lrec['invalid_class_boxes']})
        pm = check_prefix(stem, lrec)
        if pm:
            d['prefix_mismatch'].append({'path': irec['path'], 'detail': pm})
    d['total_size_mb'] = round(d['total_size_mb'], 2)
    return d


def find_exact_duplicates(img_scan):
    """基于 SHA-256 查找跨集合完全重复图片。返回 (跨集合重复组列表, 集合内部重复组数)。"""
    groups = defaultdict(list)
    for rec in img_scan.values():
        if rec['sha256']:
            groups[rec['sha256']].append(rec['path'])
    dups = []
    intra = 0
    for h, paths in groups.items():
        if len(paths) < 2:
            continue
        splits = {p.split('/')[0] for p in paths}
        if len(splits) >= 2:
            dups.append({'sha256': h, 'files': sorted(paths)})
        else:
            intra += 1
    dups.sort(key=lambda g: len(g['files']), reverse=True)
    return dups, intra


def find_near_duplicates(img_scan, threshold, windows=8):
    """基于 aHash 的近似重复检测：多窗口分桶（降低漏报）。

    对每张图的 256 位 aHash 取 8 个互不重叠的 16 位窗口作为桶键；
    同桶内两两计算完整 256 位汉明距离。两个汉明距离 <= threshold 的图，
    只要任一窗口完全一致即落入同桶（漏报概率约 0.02%）。
    返回 (近似重复对列表, 因桶过大而跳过的桶数)。"""
    n_bits = 256
    win_bits = 16
    buckets = defaultdict(list)
    for rec in img_scan.values():
        ph = rec['phash']
        if ph is None:
            continue
        for w in range(windows):
            offset = w * (n_bits // windows)
            key = (ph >> (n_bits - win_bits - offset)) & ((1 << win_bits) - 1)
            buckets[(w, key)].append((ph, rec['path']))
    pairs = []
    skipped = 0
    seen = set()
    for bucket in buckets.values():
        if len(bucket) > 300:
            skipped += 1
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                h1, p1 = bucket[i]
                h2, p2 = bucket[j]
                dist = _hamming(h1, h2)
                if dist == 0:
                    continue  # 完全相同哈希：交由精确重复章节覆盖
                if dist <= threshold:
                    pair = tuple(sorted((p1, p2)))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    pairs.append({'distance': dist, 'files': [p1, p2]})
    pairs.sort(key=lambda x: (x['distance'], x['files'][0]))
    return pairs, skipped


def build_overall(splits_data):
    """汇总各集合统计到总体。"""
    o = {
        'images': 0, 'labels': 0, 'matched': 0,
        'missing_labels': 0, 'orphan_labels': 0,
        'corrupt_images': 0,
        'label_files_with_errors': 0,
        'empty_labels': 0, 'empty_label_images': 0,
        'negative_images': 0,
        'fire_images': 0, 'smoke_images': 0, 'both_images': 0,
        'fire_boxes': 0, 'smoke_boxes': 0,
        'invalid_class_boxes': 0, 'out_of_range_boxes': 0,
        'blank_lines': 0, 'bad_label_lines': 0,
        'out_of_range_files': 0, 'invalid_class_files': 0,
        'total_size_mb': 0.0,
        'exact_dup_groups': 0, 'exact_dup_files': 0, 'exact_dup_extra': 0,
        'intra_split_dup_groups': 0,
        'near_dup_pairs': 0, 'near_dup_skipped_buckets': 0,
    }
    sum_keys = ('images', 'labels', 'matched',
                'fire_images', 'smoke_images', 'both_images', 'fire_boxes', 'smoke_boxes',
                'invalid_class_boxes', 'out_of_range_boxes', 'blank_lines', 'bad_label_lines')
    list_keys = ('missing_labels', 'orphan_labels', 'corrupt_images', 'empty_labels',
                 'empty_label_images', 'negative_images', 'out_of_range_files',
                 'invalid_class_files')
    for d in splits_data.values():
        for k in sum_keys:
            o[k] += d[k]
        for k in list_keys:
            o[k] += len(d[k])
        o['label_files_with_errors'] += len(d['label_errors'])
        o['total_size_mb'] += d['total_size_mb']
    o['total_size_mb'] = round(o['total_size_mb'], 2)
    return o


def build_conclusion(o, splits_data, phash_enabled):
    """给出是否适合开始 YOLO 冒烟训练的结论。"""
    reasons = []
    warnings = []
    ok = True
    for s, d in splits_data.items():
        if d['images'] == 0:
            ok = False
            reasons.append('%s 集合无图片' % s)
        if d['labels'] == 0:
            ok = False
            reasons.append('%s 集合无标签' % s)
        if d['matched'] == 0:
            ok = False
            reasons.append('%s 集合无匹配的图片-标签对' % s)
        elif d['images'] > 0:
            rate = d['matched'] / float(d['images'])
            if rate < 0.90:
                ok = False
                reasons.append('%s 图片-标签匹配率 %.1f%% 过低' % (s, rate * 100.0))
    if o['images'] == 0:
        ok = False
        reasons.append('未扫描到任何图片')
    if o['fire_boxes'] == 0 or o['smoke_boxes'] == 0:
        ok = False
        reasons.append('缺少 fire 或 smoke 类别的标注框')

    total = o['images'] or 1
    corrupt_rate = o['corrupt_images'] / float(total)
    if corrupt_rate > 0.10:
        ok = False
        reasons.append('损坏图片比例 %.2f%% 超过 10%%' % (corrupt_rate * 100.0))
    elif corrupt_rate > 0.01:
        warnings.append('损坏图片比例 %.2f%%' % (corrupt_rate * 100.0))

    lbl_total = o['labels'] or 1
    err_rate = o['label_files_with_errors'] / float(lbl_total)
    if err_rate > 0.10:
        ok = False
        reasons.append('标签解析错误文件比例 %.2f%% 超过 10%%' % (err_rate * 100.0))
    elif err_rate > 0.01:
        warnings.append('标签解析错误文件比例 %.2f%%' % (err_rate * 100.0))

    if o['invalid_class_boxes']:
        warnings.append('%d 个标注框 class_id 超出 {0,1}' % o['invalid_class_boxes'])
    if o['out_of_range_boxes']:
        warnings.append('%d 个标注框坐标/宽高越界' % o['out_of_range_boxes'])
    if o['missing_labels']:
        warnings.append('%d 张图片缺失标签' % o['missing_labels'])
    if o['orphan_labels']:
        warnings.append('%d 个孤立标签' % o['orphan_labels'])
    if o['exact_dup_files']:
        warnings.append('跨集合完全重复图片 %d 组 / %d 个文件' % (o['exact_dup_groups'], o['exact_dup_files']))
    if phash_enabled and o['near_dup_pairs']:
        warnings.append('近似重复图片对 %d 对' % o['near_dup_pairs'])

    if ok and warnings:
        level = 'SUITABLE_WITH_WARNINGS'
        conclusion = '可以开始 YOLO 冒烟训练，但存在以下警告：' + '；'.join(warnings[:8]) + '。'
    elif ok:
        level = 'SUITABLE'
        conclusion = '数据集结构健康，可以开始 YOLO 冒烟训练。'
    else:
        level = 'NOT_SUITABLE'
        conclusion = '不建议开始冒烟训练。原因：' + '；'.join(reasons[:8]) + '。'
    suggestion = ('建议先运行 1-3 个 epoch 的冒烟训练（如 epochs=3, batch=16, imgsz=640）验证 M2 训练链路与 '
                  'models/best.pt 产出，确认无误后再进行完整训练；test 集合保留用于最终评估。'
                  '注意：train/valid/test 之间存在大量近似重复（含部分完全重复）图片，训练/验证指标可能偏乐观，'
                  '正式实验前建议先做去重并重新划分集合。')
    return {'level': level, 'conclusion': conclusion,
            'reasons': reasons, 'warnings': warnings, 'suggestion': suggestion}

# ---------------- 报告输出 ----------------

def _pct(num, den):
    return '%.2f%%' % (num * 100.0 / den) if den else '0.00%'


def _short_list(items, cap=20, key=None):
    items = list(items)
    shown = items[:cap]
    return shown, max(0, len(items) - cap)


def _fmt_item(item, key=None):
    if isinstance(item, dict):
        if key is not None:
            return str(item.get(key, ''))
        return str(item.get('path', item))
    return str(item)


def write_json(path, report):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def write_markdown(path, report):
    meta = report['meta']
    o = report['overall']
    splits = report['splits']
    concl = report['conclusion']
    dups = report['exact_duplicates']
    near = report['near_duplicates']
    L = []
    A = L.append
    A('# FireGuardian 数据集审计报告')
    A('')
    A('- 数据集根目录：`%s`' % meta['dataset_root'])
    A('- 生成时间：%s' % meta['scanned_at'])
    A('- 工具：`tools/dataset_audit.py` v%s | 扫描耗时 %.1fs | 进程数 %d' % (
        meta['version'], meta['duration_seconds'], meta['workers']))
    A('- 感知哈希：%s（阈值 %d）' % ('启用' if meta['phash_enabled'] else '未启用',
                                  meta['phash_threshold']))
    A('')
    A('## 1. 最终结论')
    A('')
    level_map = {
        'SUITABLE': '可以开始 YOLO 冒烟训练',
        'SUITABLE_WITH_WARNINGS': '可以开始 YOLO 冒烟训练（存在警告）',
        'NOT_SUITABLE': '不建议开始冒烟训练',
    }
    A('**%s**' % level_map.get(concl['level'], concl['level']))
    A('')
    A('> %s' % concl['conclusion'])
    if concl['reasons']:
        A('')
        A('**阻断原因**')
        for r in concl['reasons']:
            A('- %s' % r)
    if concl['warnings']:
        A('')
        A('**警告**')
        for w in concl['warnings']:
            A('- %s' % w)
    A('')
    A('**建议**：%s' % concl['suggestion'])
    A('')
    A('## 2. 总体统计')
    A('')
    A('| 指标 | 数值 |')
    A('| --- | --- |')
    rows = [
        ('图片总数', o['images']),
        ('标签文件总数', o['labels']),
        ('匹配的图片-标签对', o['matched']),
        ('缺失标签（图片无标签）', o['missing_labels']),
        ('孤立标签（标签无图片）', o['orphan_labels']),
        ('损坏/不可解码图片', '%d (%s)' % (o['corrupt_images'], _pct(o['corrupt_images'], o['images']))),
        ('标签文件含解析错误', '%d (%s)' % (o['label_files_with_errors'], _pct(o['label_files_with_errors'], o['labels']))),
        ('空标签文件', o['empty_labels']),
        ('空标签对应图片', o['empty_label_images']),
        ('正常负样本（可解码+空标签）', o['negative_images']),
        ('fire 图片数 / 框数', '%d / %d' % (o['fire_images'], o['fire_boxes'])),
        ('smoke 图片数 / 框数', '%d / %d' % (o['smoke_images'], o['smoke_boxes'])),
        ('同时含 fire 与 smoke 的图片', o['both_images']),
        ('class_id 越界框数（非 0/1）', o['invalid_class_boxes']),
        ('坐标/宽高越界框数', o['out_of_range_boxes']),
        ('跨集合完全重复组 / 文件', '%d / %d' % (o['exact_dup_groups'], o['exact_dup_files'])),
        ('集合内部重复组（参考）', o['intra_split_dup_groups']),
        ('近似重复图片对（参考）', '%d 对%s' % (o['near_dup_pairs'], '（列表已截断）' if o.get('near_dup_truncated') else '')),
        ('数据总量', '%.1f MB' % o['total_size_mb']),
    ]
    for k, v in rows:
        A('| %s | %s |' % (k, v))
    A('')
    A('## 3. 各集合统计')
    for name, d in splits.items():
        A('')
        A('### %s' % name)
        A('')
        A('| 指标 | 数值 |')
        A('| --- | --- |')
        rows2 = [
            ('图片数', d['images']),
            ('标签数', d['labels']),
            ('匹配数', d['matched']),
            ('缺失标签', len(d['missing_labels'])),
            ('孤立标签', len(d['orphan_labels'])),
            ('损坏图片', '%d (%s)' % (len(d['corrupt_images']), _pct(len(d['corrupt_images']), d['images']))),
            ('标签解析错误文件', '%d (%s)' % (len(d['label_errors']), _pct(len(d['label_errors']), d['labels']))),
            ('空标签文件 / 对应图片', '%d / %d' % (len(d['empty_labels']), len(d['empty_label_images']))),
            ('正常负样本', len(d['negative_images'])),
            ('fire 图片 / 框', '%d / %d' % (d['fire_images'], d['fire_boxes'])),
            ('smoke 图片 / 框', '%d / %d' % (d['smoke_images'], d['smoke_boxes'])),
            ('双类图片', d['both_images']),
            ('class_id 越界框', d['invalid_class_boxes']),
            ('坐标/宽高越界框', d['out_of_range_boxes']),
            ('图片总大小', '%.1f MB' % d['total_size_mb']),
        ]
        for k, v in rows2:
            A('| %s | %s |' % (k, v))
        A('')
        A('**异常清单（每类最多展示 10 条）**')
        anomaly_groups = (
            ('缺失标签图片', d['missing_labels'], None),
            ('孤立标签', d['orphan_labels'], None),
            ('损坏图片', d['corrupt_images'], 'path'),
            ('标签解析错误', d['label_errors'], 'path'),
            ('坐标/宽高越界标签', d['out_of_range_files'], 'path'),
            ('class_id 越界标签', d['invalid_class_files'], 'path'),
        )
        if not any(items for _, items, _ in anomaly_groups):
            A('- 无异常。')
        else:
            for label, items, key in anomaly_groups:
                if not items:
                    continue
                shown, more = _short_list(items, 10, key)
                A('- **%s**：共 %d 个' % (label, len(items)))
                for it in shown:
                    A('  - `%s`' % _fmt_item(it, key))
                if more:
                    A('  - … 其余 %d 个见 dataset_audit.json' % more)
    A('')
    A('## 4. 类别分布（图片级 / 框级）')
    A('')
    A('| 集合 | fire 图片 | fire 框 | smoke 图片 | smoke 框 | 双类图片 | 负样本 |')
    A('| --- | --- | --- | --- | --- | --- | --- |')
    for name, d in splits.items():
        A('| %s | %d | %d | %d | %d | %d | %d |' % (
            name, d['fire_images'], d['fire_boxes'],
            d['smoke_images'], d['smoke_boxes'],
            d['both_images'], len(d['negative_images'])))
    A('')
    A('## 5. 跨集合完全重复图片（SHA-256）')
    A('')
    if not dups:
        A('- 未发现跨集合完全重复图片。')
    else:
        A('- 共 %d 组，涉及 %d 个文件（重复副本 %d 张）。' % (
            o['exact_dup_groups'], o['exact_dup_files'], o['exact_dup_extra']))
        for g in dups[:10]:
            A('- `%s`（%d 个文件）' % (g['sha256'], len(g['files'])))
            for f in g['files'][:6]:
                A('  - `%s`' % f)
            if len(g['files']) > 6:
                A('  - … 共 %d 个文件' % len(g['files']))
        if len(dups) > 10:
            A('- … 其余 %d 组见 dataset_audit.json' % (len(dups) - 10))
    A('')
    A('## 6. 近似重复图片（感知哈希）')
    A('')
    if not meta['phash_enabled']:
        A('- 未启用（可使用 `--phash` 开启，需要更长时间）。')
    elif not near:
        A('- 未发现近似重复对（汉明距离 ≤ %d）。' % meta['phash_threshold'])
    else:
        A('- 共 %d 对（16x16 aHash 汉明距离 ≤ %d；距离 0 的完全相同图由第 5 节覆盖）。' % (
            o['near_dup_pairs'], meta['phash_threshold']))
        if o.get('near_dup_skipped_buckets'):
            A('- 有 %d 个哈希桶因过大被跳过，实际近似重复对数可能略多。' % o['near_dup_skipped_buckets'])
        for p in near[:20]:
            A('- 距离 %d：`%s` ↔ `%s`' % (p['distance'], p['files'][0], p['files'][1]))
        if o.get('near_dup_truncated'):
            A('- 列表已截断，仅保留前 %d 对示例；总数见上方。' % o['near_dup_pairs_listed'])
        elif len(near) > 20:
            A('- … 其余 %d 对见 dataset_audit.json' % (len(near) - 20))
    A('')
    A('## 7. 文件名前缀与标签一致性（参考信息）')
    A('')
    total_mm = sum(len(d['prefix_mismatch']) for d in splits.values())
    if total_mm == 0:
        A('- 文件名前缀（Fire/Smoke/FireAndSmoke/FireNorSmoke）与标签内容完全一致。')
    else:
        A('- 存在 %d 处不一致（命名约定仅供参考，不影响结论）：' % total_mm)
        for name, d in splits.items():
            if not d['prefix_mismatch']:
                continue
            A('- **%s**：%d 处' % (name, len(d['prefix_mismatch'])))
            for it in d['prefix_mismatch'][:5]:
                A('  - `%s`：%s' % (it['path'], it['detail']))
    A('')
    A('## 8. 审计要求核对清单')
    A('')
    checks = [
        ('1', '各集合图片数/标签数/匹配数', '见第 2、3 节'),
        ('2', '缺失标签与孤立标签', '%d 缺失 / %d 孤立' % (o['missing_labels'], o['orphan_labels'])),
        ('3', '损坏/不可解码图片（OpenCV+Pillow）', '%d 张' % o['corrupt_images']),
        ('4', 'YOLO txt 严格格式（5 列）', '%d 个文件含错误行' % o['label_files_with_errors']),
        ('5', 'class_id 仅 0/1', '%d 个越界框' % o['invalid_class_boxes']),
        ('6', '坐标/宽高合法范围', '%d 个越界框' % o['out_of_range_boxes']),
        ('7', '空标签文件及对应图片', '%d / %d' % (o['empty_labels'], o['empty_label_images'])),
        ('8', 'fire/smoke 图片级与框级数量', 'fire %d/%d，smoke %d/%d' % (
            o['fire_images'], o['fire_boxes'], o['smoke_images'], o['smoke_boxes'])),
        ('9', '同时含 fire 和 smoke 的图片', '%d 张' % o['both_images']),
        ('10', '不含目标的正常负样本', '%d 张' % o['negative_images']),
        ('11', '跨集合完全重复（SHA-256）', '%d 组 / %d 文件' % (
            o['exact_dup_groups'], o['exact_dup_files'])),
        ('12', '近似重复（感知哈希，可选）', '%s：%d 对%s' % (
            '已启用' if meta['phash_enabled'] else '未启用', o['near_dup_pairs'],
            '（列表已截断）' if o.get('near_dup_truncated') else '')),
        ('13', '异常文件清单（只读，不删除/修改）', '见第 3 节与 dataset_audit.json'),
        ('14', '输出 dataset_audit.json / dataset_audit.md', '已生成'),
        ('15', '是否适合开始 YOLO 冒烟训练', concl['level']),
    ]
    A('| # | 检查项 | 结果 |')
    A('| --- | --- | --- |')
    for num, item, res in checks:
        A('| %s | %s | %s |' % (num, item, res))
    A('')
    A('---')
    A('*本报告由 `tools/dataset_audit.py` 自动生成；审计过程为只读操作，未复制、移动或删除任何数据。*')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ---------------- 命令行入口 ----------------

def parse_args():
    p = argparse.ArgumentParser(description='FireGuardian 独立数据集审计工具')
    p.add_argument('--dataset-root', default=DEFAULT_ROOT, help='数据集根目录（含 train/valid/test）')
    p.add_argument('--splits', default=','.join(DEFAULT_SPLITS), help='逗号分隔的集合名')
    p.add_argument('--workers', type=int, default=0, help='并行进程数（默认 min(8, CPU数)）')
    p.add_argument('--phash', action='store_true', help='启用感知哈希近似重复检测')
    p.add_argument('--phash-threshold', type=int, default=16, help='感知哈希汉明距离阈值（16x16 aHash，默认 16）')
    p.add_argument('--limit', type=int, default=0, help='每集合仅扫描前 N 个文件（测试用）')
    p.add_argument('--output-dir', default='', help='报告输出目录（默认脚本所在目录）')
    p.add_argument('--quiet', action='store_true', help='不显示进度')
    return p.parse_args()


def main():
    multiprocessing.freeze_support()
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    args = parse_args()
    for _dep in ('cv2', 'numpy', 'PIL'):
        try:
            __import__(_dep)
        except Exception as _e:
            print('[错误] 缺少依赖 %s（%s）。请使用 FireGuardian venv 运行，例如：' % (_dep, _e))
            print('       .venv\\Scripts\\python.exe tools/dataset_audit.py')
            return 1
    t0 = time.time()
    root = os.path.abspath(args.dataset_root)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    workers = args.workers or min(8, os.cpu_count() or 4)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'dataset_audit.json')
    md_path = os.path.join(out_dir, 'dataset_audit.md')

    print('=' * 62)
    print('FireGuardian 数据集审计工具 v%s' % TOOL_VERSION)
    print('数据集: %s' % root)
    print('集合: %s | 进程数: %d | 感知哈希: %s' % (
        ', '.join(splits), workers, '启用' if args.phash else '禁用'))
    print('=' * 62)

    def empty_report(msg):
        return {
            'meta': {
                'tool': 'tools/dataset_audit.py', 'version': TOOL_VERSION,
                'dataset_root': root, 'splits': splits,
                'scanned_at': datetime.now().isoformat(timespec='seconds'),
                'duration_seconds': round(time.time() - t0, 1),
                'workers': workers, 'limit_per_split': args.limit,
                'phash_enabled': args.phash, 'phash_threshold': args.phash_threshold,
            },
            'overall': {'images': 0, 'labels': 0, 'matched': 0, 'missing_labels': 0,
                        'orphan_labels': 0, 'corrupt_images': 0, 'label_files_with_errors': 0,
                        'empty_labels': 0, 'empty_label_images': 0, 'negative_images': 0,
                        'fire_images': 0, 'smoke_images': 0, 'both_images': 0,
                        'fire_boxes': 0, 'smoke_boxes': 0, 'invalid_class_boxes': 0,
                        'out_of_range_boxes': 0, 'blank_lines': 0, 'bad_label_lines': 0,
                        'total_size_mb': 0.0, 'exact_dup_groups': 0, 'exact_dup_files': 0,
                        'exact_dup_extra': 0, 'intra_split_dup_groups': 0,
                        'near_dup_pairs': 0, 'near_dup_skipped_buckets': 0},
            'splits': {},
            'exact_duplicates': [],
            'near_duplicates': [],
            'conclusion': {'level': 'NOT_SUITABLE', 'conclusion': msg,
                           'reasons': [msg], 'warnings': [], 'suggestion': ''},
        }

    if not os.path.isdir(root):
        print('[错误] 数据集根目录不存在: %s' % root)
        write_json(json_path, empty_report('数据集根目录不存在: %s' % root))
        write_markdown(md_path, empty_report('数据集根目录不存在: %s' % root))
        print('已输出: %s' % json_path)
        print('       %s' % md_path)
        return 1

    # ---- 1. 枚举 ----
    enum = {}
    for split in splits:
        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            print('[警告] 集合不存在，跳过: %s' % split)
            continue
        img_map, lbl_map = enumerate_split(split_path, split)
        enum[split] = (img_map, lbl_map)
        print('[枚举] %-5s 图片 %d | 标签 %d' % (split, len(img_map), len(lbl_map)))

    if not enum:
        print('[错误] 没有任何可扫描的集合')
        write_json(json_path, empty_report('没有任何可扫描的集合'))
        write_markdown(md_path, empty_report('没有任何可扫描的集合'))
        return 1

    img_tasks = []
    lbl_tasks = []
    for split, (img_map, lbl_map) in enum.items():
        imgs = sorted(img_map.items())
        if args.limit > 0:
            imgs = imgs[:args.limit]
        for stem, (full, rel) in imgs:
            img_tasks.append((split, full, rel, args.phash))
        lbls = sorted(lbl_map.items())
        if args.limit > 0:
            lbls = lbls[:args.limit]
        for stem, (full, rel) in lbls:
            lbl_tasks.append((split, full, rel))

    # ---- 2. 并行扫描图片 ----
    img_scan = {}
    if img_tasks:
        print('\n[阶段1/3] 图片扫描（解码 + SHA-256%s）...' % (' + 感知哈希' if args.phash else ''))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            prog = Progress(len(img_tasks), '图片扫描', quiet=args.quiet)
            for rec in ex.map(scan_image_file, img_tasks, chunksize=64):
                prog.update()
                img_scan[(rec['split'], rec['stem'])] = rec

    # ---- 3. 并行扫描标签 ----
    lbl_scan = {}
    if lbl_tasks:
        print('\n[阶段2/3] 标签扫描（格式/类别/范围校验）...')
        with ProcessPoolExecutor(max_workers=workers) as ex:
            prog = Progress(len(lbl_tasks), '标签扫描', quiet=args.quiet)
            for rec in ex.map(scan_label_file, lbl_tasks, chunksize=128):
                prog.update()
                lbl_scan[(rec['split'], rec['stem'])] = rec

    # ---- 4. 聚合与重复检测 ----
    print('\n[阶段3/3] 聚合统计与重复检测...')
    splits_data = {}
    for split, (img_map, lbl_map) in enum.items():
        splits_data[split] = aggregate_split(split, img_map, lbl_map, img_scan, lbl_scan)

    dups, intra_groups = find_exact_duplicates(img_scan)
    near_pairs, skipped_buckets = [], 0
    if args.phash:
        near_pairs, skipped_buckets = find_near_duplicates(img_scan, args.phash_threshold)

    overall = build_overall(splits_data)
    overall['exact_dup_groups'] = len(dups)
    overall['exact_dup_files'] = sum(len(g['files']) for g in dups)
    overall['exact_dup_extra'] = overall['exact_dup_files'] - overall['exact_dup_groups']
    overall['intra_split_dup_groups'] = intra_groups
    overall['near_dup_skipped_buckets'] = skipped_buckets
    near_total = len(near_pairs)
    near_listed = near_pairs[:10000]
    overall['near_dup_pairs'] = near_total
    overall['near_dup_pairs_listed'] = len(near_listed)
    overall['near_dup_truncated'] = near_total > len(near_listed)

    conclusion = build_conclusion(overall, splits_data, args.phash)

    report = {
        'meta': {
            'tool': 'tools/dataset_audit.py', 'version': TOOL_VERSION,
            'dataset_root': root, 'splits': splits,
            'scanned_at': datetime.now().isoformat(timespec='seconds'),
            'duration_seconds': round(time.time() - t0, 1),
            'workers': workers, 'limit_per_split': args.limit,
            'phash_enabled': args.phash, 'phash_threshold': args.phash_threshold,
        },
        'overall': overall,
        'splits': splits_data,
        'exact_duplicates': dups,
        'near_duplicates': near_listed,
        'conclusion': conclusion,
    }

    write_json(json_path, report)
    write_markdown(md_path, report)

    print('\n' + '=' * 62)
    print('审计完成，耗时 %.1f 秒' % (time.time() - t0))
    print('图片 %d | 标签 %d | 匹配 %d | 缺失 %d | 孤立 %d | 损坏 %d' % (
        overall['images'], overall['labels'], overall['matched'],
        overall['missing_labels'], overall['orphan_labels'], overall['corrupt_images']))
    print('fire 图片/框: %d/%d | smoke 图片/框: %d/%d | 双类: %d | 负样本: %d' % (
        overall['fire_images'], overall['fire_boxes'],
        overall['smoke_images'], overall['smoke_boxes'],
        overall['both_images'], overall['negative_images']))
    print('跨集合完全重复: %d 组 / %d 个文件' % (
        overall['exact_dup_groups'], overall['exact_dup_files']))
    if args.phash:
        print('近似重复对: %d 对' % overall['near_dup_pairs'])
    print('-' * 62)
    print('[结论] %s' % conclusion['conclusion'])
    print('报告已输出: %s' % json_path)
    print('           %s' % md_path)
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())
