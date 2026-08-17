"""
============================================================
FireGuardian - YOLO 冒烟训练子集生成工具 (Smoke Subset Builder)
============================================================

基于已有审计结果（默认 tools/dataset_audit.json），从数据集
train / valid 两个集合中按类别分层、SHA-256 去重、固定随机种子，
生成用于首次 YOLO 链路验证的小规模训练子集。

硬性约束：
- 只读原始数据集：不复制、不移动、不删除、不覆盖任何原图或原标签；
- 不修改 M1-M11 业务模块；
- train 与 valid 之间不允许存在 SHA-256 完全相同的图片；
- 同一 SHA-256 重复组只能进入一个集合；
- 不依据 16x16 aHash 近似结果自动删除图片（近似重复仅作参考信息）。

输出文件：
  datasets/smoke_train.txt          每行一个图片绝对路径
  datasets/smoke_val.txt            每行一个图片绝对路径
  datasets/fire_smoke_smoke.yaml    Ultralytics 数据集配置
  tools/smoke_subset_report.json    完整报告
  tools/smoke_subset_report.md      可读报告

用法示例：
  python tools/build_smoke_subset.py
  python tools/build_smoke_subset.py --seed 42 --train-size 3000 --valid-size 600
  python tools/build_smoke_subset.py --skip-hash           # 快速模式
  python tools/build_smoke_subset.py --no-ultralytics      # 跳过加载测试

依赖：Python 3.8+；ultralytics 仅用于加载测试（缺失时自动跳过）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

TOOL_VERSION = '1.0.0'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
LABEL_EXT = '.txt'
CLASS_NAMES = {0: 'fire', 1: 'smoke'}
CATEGORIES = ('fire_only', 'smoke_only', 'both', 'negative')
CATEGORY_RATIOS = {'fire_only': 0.30, 'smoke_only': 0.30, 'both': 0.20, 'negative': 0.20}
DEFAULT_ROOT = r'DATASET_ROOT'
DEFAULT_TARGET = {'train': 3000, 'valid': 600}
EPS = 1e-4

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(_PROJECT_ROOT, '.ultralytics'))

RECOMMENDED_TRAIN_ARGS = {
    'weights': 'yolo11n.pt',
    'epochs': 3,
    'batch': 16,
    'imgsz': 640,
    'device': 0,
    'workers': 4,
}


# ---------------- 基础工具函数 ----------------

def _rel(*parts):
    return os.path.join(*parts).replace(os.sep, '/')


def _stem_of(rel_path):
    return os.path.splitext(os.path.basename(rel_path))[0]


def _box_in_range(cx, cy, w, h):
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


def parse_yolo_label_strict(abs_lbl):
    """严格解析 YOLO 标签：5 列 / 数值 / class_id∈{0,1} / 坐标范围合法。"""
    rec = {'empty': False, 'fire_boxes': 0, 'smoke_boxes': 0,
           'bad_lines': 0, 'first_error': None}
    try:
        with open(abs_lbl, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        rec['bad_lines'] = 1
        rec['first_error'] = '读取失败: %s' % e
        return rec
    if not content.strip():
        rec['empty'] = True
        return rec
    for idx, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            rec['bad_lines'] += 1
            if rec['first_error'] is None:
                rec['first_error'] = '第%d行 列数=%d(应为5)' % (idx, len(parts))
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            rec['bad_lines'] += 1
            if rec['first_error'] is None:
                rec['first_error'] = '第%d行 含非数值列' % idx
            continue
        cid = int(round(vals[0]))
        cx, cy, w, h = vals[1], vals[2], vals[3], vals[4]
        if abs(vals[0] - cid) > 1e-9 or cid not in (0, 1) or not _box_in_range(cx, cy, w, h):
            rec['bad_lines'] += 1
            if rec['first_error'] is None:
                rec['first_error'] = '第%d行 非法(class_id=%s)' % (idx, vals[0])
            continue
        if cid == 0:
            rec['fire_boxes'] += 1
        else:
            rec['smoke_boxes'] += 1
    return rec


class Progress:
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
# ---------------- 枚举 / 扫描 ----------------

def enumerate_split(split_path, split):
    """返回 {stem: (abs_img, rel_img)} 与 {stem: (abs_lbl, rel_lbl)}。"""
    img_map, lbl_map = {}, {}
    img_dir = os.path.join(split_path, 'images')
    lbl_dir = os.path.join(split_path, 'labels')
    if os.path.isdir(img_dir):
        with os.scandir(img_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in IMAGE_EXTS:
                    stem = os.path.splitext(entry.name)[0]
                    img_map[stem] = (entry.path, _rel(split, 'images', entry.name))
    if os.path.isdir(lbl_dir):
        with os.scandir(lbl_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext == LABEL_EXT:
                    stem = os.path.splitext(entry.name)[0]
                    lbl_map[stem] = (entry.path, _rel(split, 'labels', entry.name))
    return img_map, lbl_map


def scan_candidate(args):
    """工作进程：严格校验标签，并按需计算图片 SHA-256。单文件错误不中断扫描。"""
    split, stem, abs_img, rel_img, abs_lbl, rel_lbl, do_hash = args
    rec = {'split': split, 'stem': stem, 'abs_img': abs_img, 'rel_img': rel_img,
           'abs_lbl': abs_lbl, 'rel_lbl': rel_lbl, 'size_bytes': 0,
           'sha256': None, 'label': None, 'error': None, 'error_kind': None}
    try:
        rec['size_bytes'] = os.path.getsize(abs_img)
    except Exception as e:
        rec['error'] = '图片 stat 失败: %s' % e
        rec['error_kind'] = 'image'
        return rec
    rec['label'] = parse_yolo_label_strict(abs_lbl)
    if rec['label']['bad_lines'] > 0:
        rec['error'] = rec['label']['first_error']
        rec['error_kind'] = 'label'
        return rec
    if do_hash:
        try:
            with open(abs_img, 'rb') as f:
                rec['sha256'] = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            rec['error'] = '图片读取失败: %s' % e
            rec['error_kind'] = 'image'
    return rec


def classify(rec):
    lbl = rec['label']
    if lbl['empty']:
        return 'negative'
    if lbl['fire_boxes'] > 0 and lbl['smoke_boxes'] > 0:
        return 'both'
    if lbl['fire_boxes'] > 0:
        return 'fire_only'
    if lbl['smoke_boxes'] > 0:
        return 'smoke_only'
    return 'negative'


# ---------------- 审计报告处理 ----------------

def _entry_stem(entry):
    rel = entry.get('path', '') if isinstance(entry, dict) else entry
    return _stem_of(rel)


def build_audit_exclusions(audit):
    """从审计报告构建排除集：越界框、损坏、标签异常、缺失/孤立。返回 (excl, counters)。"""
    excl = defaultdict(set)
    counters = {'out_of_range_files': 0, 'out_of_range_boxes': 0, 'corrupt_images': 0,
                'label_errors': 0, 'invalid_class_files': 0, 'missing_labels': 0,
                'orphan_labels': 0}
    if not isinstance(audit, dict) or 'splits' not in audit:
        return excl, counters
    for split, sd in audit.get('splits', {}).items():
        for it in sd.get('out_of_range_files', []):
            stem = _entry_stem(it)
            if stem:
                excl[split].add(stem)
                counters['out_of_range_files'] += 1
                counters['out_of_range_boxes'] += int(it.get('boxes', 0) or 0)
        for it in sd.get('corrupt_images', []):
            stem = _entry_stem(it)
            if stem:
                excl[split].add(stem)
                counters['corrupt_images'] += 1
        for it in sd.get('label_errors', []):
            stem = _entry_stem(it)
            if stem:
                excl[split].add(stem)
                counters['label_errors'] += 1
        for it in sd.get('invalid_class_files', []):
            stem = _entry_stem(it)
            if stem:
                excl[split].add(stem)
                counters['invalid_class_files'] += 1
        for it in sd.get('missing_labels', []):
            stem = _entry_stem(it)
            if stem:
                excl[split].add(stem)
                counters['missing_labels'] += 1
        for it in sd.get('orphan_labels', []):
            stem = _entry_stem(it)
            if stem:
                excl[split].add(stem)
                counters['orphan_labels'] += 1
    return excl, counters


def apply_audit_cross_dedup(audit, audit_excl, rng):
    """快速模式：依据审计报告的跨集合重复组去重，每组在 train/valid 中保留一个代表。"""
    groups = audit.get('exact_duplicates', []) if isinstance(audit, dict) else []
    cnt_groups = 0
    cnt_excluded = 0
    for g in groups:
        files = [f for f in g.get('files', []) if f.startswith(('train/', 'valid/'))]
        if len(files) <= 1:
            continue
        cnt_groups += 1
        rep = rng.choice(sorted(files))
        for f in files:
            if f != rep:
                split = f.split('/')[0]
                audit_excl[split].add(_stem_of(f))
                cnt_excluded += 1
    return cnt_groups, cnt_excluded
# ---------------- 去重 / 抽样 ----------------

def dedup_pools(candidates, rng):
    """按 SHA-256 精确去重：每组保留一个代表（种子随机）。返回 (kept, removed, stats)。"""
    groups = defaultdict(list)
    for rec in candidates:
        if rec['sha256']:
            groups[rec['sha256']].append(rec)
    stats = {'groups': 0, 'removed': 0, 'cross_groups': 0, 'cross_removed': 0,
             'intra_groups': 0, 'intra_removed': 0}
    keep_ids = set()
    for h, members in groups.items():
        if len(members) == 1:
            keep_ids.add(id(members[0]))
            continue
        stats['groups'] += 1
        ordered = sorted(members, key=lambda r: (r['split'], r['stem']))
        rep = rng.choice(ordered)
        keep_ids.add(id(rep))
        is_cross = len({r['split'] for r in ordered}) > 1
        if is_cross:
            stats['cross_groups'] += 1
            stats['cross_removed'] += len(ordered) - 1
        else:
            stats['intra_groups'] += 1
            stats['intra_removed'] += len(ordered) - 1
        stats['removed'] += len(ordered) - 1
    kept = [rec for rec in candidates if id(rec) in keep_ids or not rec['sha256']]
    removed = [rec for rec in candidates if id(rec) not in keep_ids and rec['sha256']]
    return kept, removed, stats


def ensure_hash(rec):
    if rec['sha256'] is None:
        try:
            with open(rec['abs_img'], 'rb') as f:
                rec['sha256'] = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            rec['sha256'] = 'unreadable:%s' % hashlib.md5(
                rec['abs_img'].encode('utf-8')).hexdigest()
    return rec


def verify_and_refill(sampled_by_split, leftover_by_split_cat, rng, max_iter=40):
    """确保最终样本内（含 train/valid 之间）无任何 SHA-256 重复；重复则剔除并同类别补充。"""
    notes = []
    for _ in range(max_iter):
        changed = False
        for lst in sampled_by_split.values():
            for r in lst:
                ensure_hash(r)
        groups = defaultdict(list)
        for split, lst in sampled_by_split.items():
            for r in lst:
                if r['sha256']:
                    groups[r['sha256']].append((split, r))
        dup = {h: g for h, g in groups.items() if len(g) > 1}
        if not dup:
            break
        for h, g in dup.items():
            g_sorted = sorted(g, key=lambda x: (x[0], x[1]['stem']))
            for split, r in g_sorted[1:]:
                sampled_by_split[split].remove(r)
                cat = classify(r)
                notes.append('去除样本内重复并补充: %s (hash=%s...)' % (r['rel_img'], h[:12]))
                pool = leftover_by_split_cat.get(split, {}).get(cat, [])
                if pool:
                    idx = rng.randrange(len(pool))
                    fill = pool.pop(idx)
                    ensure_hash(fill)
                    sampled_by_split[split].append(fill)
                    notes.append('  补充 -> %s [%s]' % (fill['rel_img'], split))
                else:
                    notes.append('  警告: %s/%s 补充池已耗尽，数量将低于目标' % (split, cat))
                changed = True
        if not changed:
            break
    return sampled_by_split, notes


def largest_remainder_quotas(size, ratios):
    raw = {c: size * ratios[c] for c in CATEGORIES}
    floors = {c: int(raw[c]) for c in CATEGORIES}
    quotas = dict(floors)
    remaining = size - sum(quotas.values())
    order = sorted(CATEGORIES, key=lambda c: (-(raw[c] - floors[c]), c))
    for c in order[:remaining]:
        quotas[c] += 1
    return quotas


def img_to_label_path(abs_img):
    lbl = abs_img.replace('/images/', '/labels/')
    return os.path.splitext(lbl)[0] + LABEL_EXT


# ---------------- 输出写入 ----------------

def write_txt(path, img_paths):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for p in img_paths:
            f.write(p.replace(os.sep, '/') + '\n')


def write_yaml(path):
    content = ('train: smoke_train.txt\n'
               'val: smoke_val.txt\n'
               'nc: 2\n'
               'names:\n'
               '  0: fire\n'
               '  1: smoke\n')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
# ---------------- 生成后验证 ----------------

def validate_outputs(train_recs, valid_recs):
    """检查所有图片路径存在、对应标签存在且可被 YOLO 严格读取。"""
    res = {'images_checked': 0, 'images_missing': [], 'labels_checked': 0,
           'labels_missing': [], 'label_errors': [], 'label_error_details': []}
    for rec in train_recs + valid_recs:
        img = rec['abs_img']
        res['images_checked'] += 1
        if not os.path.isfile(img):
            res['images_missing'].append(img)
            continue
        lbl = img_to_label_path(img)
        if not os.path.isfile(lbl):
            res['labels_missing'].append(lbl)
            continue
        res['labels_checked'] += 1
        p = parse_yolo_label_strict(lbl)
        if p['bad_lines'] > 0:
            res['label_errors'].append(lbl)
            res['label_error_details'].append({'path': lbl, 'error': p['first_error']})
    return res


def ultralytics_load_test(yaml_path):
    """运行一次 Ultralytics 数据集加载测试（不训练、不写 labels.cache）。

    1) check_det_dataset 校验 yaml 与路径；2) 用 ultralytics 的 verify_image_label
    实际读取前 N 张图片与标签，验证图片可解码、标签格式/类别/坐标合法。
    """
    res = {'status': 'SKIPPED', 'message': '', 'details': {}}
    try:
        from ultralytics.data.utils import check_det_dataset, verify_image_label, img2label_paths
        info = check_det_dataset(str(yaml_path))
        res['details'] = {
            'nc': info.get('nc'),
            'names': info.get('names'),
            'train': info.get('train'),
            'val': info.get('val'),
        }
        with open(info['train'], encoding='utf-8') as f:
            imgs = [ln.strip() for ln in f if ln.strip()][:32]
        os_imgs = [x.replace('/', os.sep) for x in imgs]
        lbls = img2label_paths(os_imgs)
        nm = nf = ne = nc = 0
        msgs = []
        for im, lb in zip(os_imgs, lbls):
            r = verify_image_label((im, lb, '', False, int(info.get('nc', 2)), 0, 0, False))
            nm += int(r[5]); nf += int(r[6]); ne += int(r[7]); nc += int(r[8])
            if r[9]:
                msgs.append(r[9])
        res['details']['load_test'] = {
            'checked': len(imgs), 'labels_found': nf, 'labels_missing': nm,
            'labels_empty': ne, 'corrupt': nc, 'messages': msgs[:10],
        }
        res['status'] = 'OK'
        res['message'] = ('check_det_dataset 通过；verify_image_label 实际加载 %d 张'
                          '（found=%d, missing=%d, empty=%d, corrupt=%d）'
                          % (len(imgs), nf, nm, ne, nc))
    except Exception as e:
        res['status'] = 'ERROR'
        res['message'] = '%s: %s' % (type(e).__name__, e)
    return res


# ---------------- 报告 ----------------

def build_report(args, audit, audit_counters, excluded_scan, dup_stats, pools,
                 sampled, leftover, shortfalls, notes, val_out, ult, t0):
    categories = {}
    boxes = {}
    for split in ('train', 'valid'):
        categories[split] = {c: 0 for c in CATEGORIES}
        boxes[split] = {'fire_boxes': 0, 'smoke_boxes': 0, 'total_boxes': 0}
        for rec in sampled[split]:
            categories[split][classify(rec)] += 1
            boxes[split]['fire_boxes'] += rec['label']['fire_boxes']
            boxes[split]['smoke_boxes'] += rec['label']['smoke_boxes']
        boxes[split]['total_boxes'] = boxes[split]['fire_boxes'] + boxes[split]['smoke_boxes']
    pool_sizes = {}
    for split in ('train', 'valid'):
        pool_sizes[split] = {c: len(pools.get(split, {}).get(c, [])) for c in CATEGORIES}

    issues = []
    warnings = []
    for split, size in (('train', args.train_size), ('valid', args.valid_size)):
        if len(sampled[split]) != size:
            issues.append('%s 实际数量 %d 未达到目标 %d' % (split, len(sampled[split]), size))
        if shortfalls.get(split):
            warnings.append('%s 存在类别数量不足: %s' % (split, shortfalls[split]))
    if val_out['images_missing'] or val_out['labels_missing'] or val_out['label_errors']:
        issues.append('验证发现缺失图片/标签或标签错误')
    if ult['status'] == 'ERROR':
        issues.append('Ultralytics 加载测试失败')
    if ult['status'] == 'SKIPPED':
        warnings.append('Ultralytics 加载测试未执行')
    if notes:
        n_dup = sum(1 for n in notes if n.startswith('去除样本内重复'))
        if n_dup:
            warnings.append('样本内 SHA-256 重复已剔除 %d 处并同类别补充（详见 notes）' % n_dup)
        else:
            warnings.append('抽样过程备注 %d 条（见 notes）' % len(notes))

    if issues:
        level = 'NOT_SUITABLE'
        conclusion_msg = '暂不建议开始冒烟训练: ' + '；'.join(issues)
    elif warnings:
        level = 'SUITABLE_WITH_WARNINGS'
        conclusion_msg = '可以开始 YOLO 冒烟训练，但需注意: ' + '；'.join(warnings)
    else:
        level = 'SUITABLE'
        conclusion_msg = '可以开始 YOLO 冒烟训练'

    command = ('.venv\\Scripts\\python.exe -m ultralytics yolo train '
               'model=%s data=%s epochs=%s batch=%s imgsz=%s device=%s workers=%s'
               % (RECOMMENDED_TRAIN_ARGS['weights'], 'datasets/fire_smoke_smoke.yaml',
                  RECOMMENDED_TRAIN_ARGS['epochs'], RECOMMENDED_TRAIN_ARGS['batch'],
                  RECOMMENDED_TRAIN_ARGS['imgsz'], RECOMMENDED_TRAIN_ARGS['device'],
                  RECOMMENDED_TRAIN_ARGS['workers']))

    return {
        'meta': {
            'tool': 'tools/build_smoke_subset.py', 'version': TOOL_VERSION,
            'dataset_root': args.dataset_root,
            'audit_json': os.path.basename(args.audit_json) if args.audit_json else None,
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'duration_seconds': round(time.time() - t0, 1),
            'seed': args.seed,
            'workers': args.workers or min(8, os.cpu_count() or 4),
            'skip_hash': args.skip_hash,
        },
        'targets': {'train': args.train_size, 'valid': args.valid_size},
        'ratios': dict(CATEGORY_RATIOS),
        'actual': {split: len(sampled[split]) for split in ('train', 'valid')},
        'categories': categories,
        'pool_sizes': pool_sizes,
        'boxes': boxes,
        'shortfalls': shortfalls,
        'notes': notes,
        'exclusions': {
            'audit': audit_counters,
            'scan_excluded': excluded_scan,
            'duplicates': dup_stats,
            'audit_reference': {
                'cross_split_groups': audit.get('overall', {}).get('exact_dup_groups', 0)
                if isinstance(audit, dict) else 0,
                'cross_split_files': audit.get('overall', {}).get('exact_dup_files', 0)
                if isinstance(audit, dict) else 0,
                'intra_split_groups': audit.get('overall', {}).get('intra_split_dup_groups', 0)
                if isinstance(audit, dict) else 0,
            },
        },
        'validation': {
            'images_checked': val_out['images_checked'],
            'images_missing': val_out['images_missing'],
            'labels_checked': val_out['labels_checked'],
            'labels_missing': val_out['labels_missing'],
            'label_errors': val_out['label_errors'],
            'label_error_details': val_out['label_error_details'],
        },
        'ultralytics_test': ult,
        'recommended_training': dict(RECOMMENDED_TRAIN_ARGS),
        'recommended_command': command,
        'conclusion': {'level': level, 'conclusion': conclusion_msg,
                       'issues': issues, 'warnings': warnings},
        'sample_files': {'train': [r['abs_img'] for r in sampled['train']],
                         'valid': [r['abs_img'] for r in sampled['valid']]},
    }

def write_markdown(path, rep):
    L = []
    A = L.append
    A('# FireGuardian YOLO 冒烟训练子集报告')
    A('')
    A('- 生成时间: %s' % rep['meta']['generated_at'])
    A('- 工具: %s v%s' % (rep['meta']['tool'], rep['meta']['version']))
    A('- 数据集根目录: %s' % rep['meta']['dataset_root'])
    A('- 审计报告: %s' % rep['meta']['audit_json'])
    A('- 随机种子: %d | 进程数: %d | 扫描耗时: %.1fs' % (
        rep['meta']['seed'], rep['meta']['workers'], rep['meta']['duration_seconds']))
    A('')
    A('## 1. 最终结论')
    A('')
    A('**%s**' % rep['conclusion']['conclusion'])
    if rep['conclusion']['issues']:
        A('')
        A('**问题**')
        for i in rep['conclusion']['issues']:
            A('- %s' % i)
    if rep['conclusion']['warnings']:
        A('')
        A('**警告**')
        for w in rep['conclusion']['warnings']:
            A('- %s' % w)
    A('')
    A('## 2. 样本规模')
    A('')
    A('| 集合 | 目标 | 实际 |')
    A('| --- | --- | --- |')
    for split in ('train', 'valid'):
        A('| %s | %d | %d |' % (split, rep['targets'][split], rep['actual'][split]))
    A('')
    A('## 3. 类别分布')
    A('')
    A('| 集合 | fire-only | smoke-only | fire+smoke | 负样本 | 合计 |')
    A('| --- | --- | --- | --- | --- | --- |')
    for split in ('train', 'valid'):
        c = rep['categories'][split]
        A('| %s | %d | %d | %d | %d | %d |' % (
            split, c['fire_only'], c['smoke_only'], c['both'], c['negative'],
            rep['actual'][split]))
    A('')
    A('比例目标: fire-only 30%% / smoke-only 30%% / fire+smoke 20%% / 负样本 20%%')
    A('')
    A('## 4. 标注框统计')
    A('')
    A('| 集合 | fire 框 | smoke 框 | 合计 |')
    A('| --- | --- | --- | --- |')
    for split in ('train', 'valid'):
        b = rep['boxes'][split]
        A('| %s | %d | %d | %d |' % (split, b['fire_boxes'], b['smoke_boxes'], b['total_boxes']))
    A('')
    A('## 5. 排除情况')
    A('')
    A('**审计报告排除（dataset_audit.json）**')
    A('')
    ec = rep['exclusions']['audit']
    A('- 越界标注框样本: %d 个标签文件 / %d 个框' % (
        ec['out_of_range_files'], ec['out_of_range_boxes']))
    A('- 损坏图片: %d' % ec['corrupt_images'])
    A('- 标签异常: %d（label_errors=%d, invalid_class=%d, missing=%d, orphan=%d）' % (
        ec['label_errors'] + ec['invalid_class_files'] + ec['missing_labels'] + ec['orphan_labels'],
        ec['label_errors'], ec['invalid_class_files'], ec['missing_labels'], ec['orphan_labels']))
    A('- 跨集合完全重复（审计参考）: %d 组 / %d 个文件' % (
        rep['exclusions']['audit_reference']['cross_split_groups'],
        rep['exclusions']['audit_reference']['cross_split_files']))
    A('- 集合内部重复组（审计参考）: %d' % rep['exclusions']['audit_reference']['intra_split_groups'])
    A('')
    A('**本次扫描排除**')
    A('')
    A('- 候选扫描中标签异常/读取失败: %d（label=%d, image=%d）' % (
        rep['exclusions']['scan_excluded']['label'] + rep['exclusions']['scan_excluded']['image'],
        rep['exclusions']['scan_excluded']['label'], rep['exclusions']['scan_excluded']['image']))
    if rep['exclusions']['audit'].get('skip_hash_cross_groups'):
        A('- 快速模式审计跨集合去重: %d 组 / 排除 %d 个文件（train/valid 池内）' % (
            rep['exclusions']['audit']['skip_hash_cross_groups'],
            rep['exclusions']['audit']['skip_hash_cross_excluded']))
    d = rep['exclusions']['duplicates']
    A('- SHA-256 精确重复组（候选池内）: %d 组 / 移除 %d 个文件' % (d['groups'], d['removed']))
    A('  - 跨 train/valid 重复组: %d 组 / 移除 %d 个文件' % (d['cross_groups'], d['cross_removed']))
    A('  - 集合内部重复组: %d 组 / 移除 %d 个文件' % (d['intra_groups'], d['intra_removed']))
    A('')
    if rep['shortfalls']:
        A('**类别数量不足（按实际数量调整）**')
        A('')
        A('```')
        A(json.dumps(rep['shortfalls'], ensure_ascii=False, indent=2))
        A('```')
        A('')
    if rep['notes']:
        A('**抽样备注**')
        A('')
        for n in rep['notes']:
            A('- %s' % n)
        A('')
    A('## 6. 生成文件')
    A('')
    A('- `datasets/smoke_train.txt`（%d 行）' % rep['actual']['train'])
    A('- `datasets/smoke_val.txt`（%d 行）' % rep['actual']['valid'])
    A('- `datasets/fire_smoke_smoke.yaml`')
    A('')
    A('```yaml')
    A('train: smoke_train.txt')
    A('val: smoke_val.txt')
    A('nc: 2')
    A('names:')
    A('  0: fire')
    A('  1: smoke')
    A('```')
    A('')
    A('## 7. 生成后验证')
    A('')
    v = rep['validation']
    A('- 图片路径检查: %d 张，缺失 %d' % (v['images_checked'], len(v['images_missing'])))
    A('- 标签路径检查: %d 个，缺失 %d' % (v['labels_checked'], len(v['labels_missing'])))
    A('- 标签可读性（严格 YOLO 格式）: %d 个，错误 %d' % (
        v['labels_checked'], len(v['label_errors'])))
    A('- Ultralytics 加载测试: %s — %s' % (rep['ultralytics_test']['status'],
                                           rep['ultralytics_test']['message']))
    if rep['ultralytics_test']['details']:
        det = rep['ultralytics_test']['details']
        A('  - nc=%s names=%s' % (det.get('nc'), det.get('names')))
    A('')
    A('## 8. 推荐的冒烟训练参数')
    A('')
    A('```')
    A(rep['recommended_command'])
    A('```')
    A('')
    A('| 参数 | 值 |')
    A('| --- | --- |')
    t = rep['recommended_training']
    for k, vv in t.items():
        A('| %s | %s |' % (k, vv))
    A('')
    A('> 说明：如本机无 CUDA GPU，请将 `device=0` 改为 `device=cpu`。')
    A('')
    A('## 9. 抽样文件完整清单')
    A('')
    for split in ('train', 'valid'):
        A('### %s (%d)' % (split, rep['actual'][split]))
        A('')
        for p in rep['sample_files'][split]:
            A('- %s' % p)
        A('')
    A('---')
    A('*本报告由 `tools/build_smoke_subset.py` 自动生成；审计与抽样过程为只读操作，'
      '未复制、移动或删除原始数据集中的任何文件。*')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
# ---------------- 命令行入口 ----------------

def parse_args():
    p = argparse.ArgumentParser(description='FireGuardian YOLO 冒烟训练子集生成工具')
    p.add_argument('--dataset-root', default=DEFAULT_ROOT, help='数据集根目录（含 train/valid）')
    p.add_argument('--audit-json', default=os.path.join(_THIS_DIR, 'dataset_audit.json'),
                   help='审计报告路径')
    p.add_argument('--train-size', type=int, default=DEFAULT_TARGET['train'])
    p.add_argument('--valid-size', type=int, default=DEFAULT_TARGET['valid'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0,
                   help='并行进程数（默认 min(8, CPU数)）')
    p.add_argument('--output-dir', default=os.path.join(_PROJECT_ROOT, 'datasets'),
                   help='txt/yaml 输出目录（默认项目 datasets/）')
    p.add_argument('--report-dir', default=_THIS_DIR,
                   help='报告输出目录（默认 tools/）')
    p.add_argument('--skip-hash', action='store_true',
                   help='不重新计算全量 SHA-256，仅按审计报告的跨集合重复组去重（快速模式）')
    p.add_argument('--no-ultralytics', action='store_true', help='跳过 Ultralytics 加载测试')
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
    t0 = time.time()
    rng = random.Random(args.seed)
    root = os.path.abspath(args.dataset_root)
    workers = args.workers or min(8, os.cpu_count() or 4)

    print('=' * 62)
    print('FireGuardian 冒烟训练子集生成工具 v%s' % TOOL_VERSION)
    print('数据集: %s' % root)
    print('目标: train %d / valid %d | 种子: %d | 进程数: %d | 哈希: %s' % (
        args.train_size, args.valid_size, args.seed, workers,
        '跳过(审计去重)' if args.skip_hash else '全量SHA-256'))
    print('=' * 62)

    audit = None
    if os.path.isfile(args.audit_json):
        with open(args.audit_json, encoding='utf-8') as f:
            audit = json.load(f)
        print('[审计] 已加载 %s' % args.audit_json)
    else:
        print('[警告] 未找到审计报告，跳过越界/损坏排除: %s' % args.audit_json)

    audit_excl, audit_counters = build_audit_exclusions(audit)
    if args.skip_hash and audit:
        g, f = apply_audit_cross_dedup(audit, audit_excl, rng)
        audit_counters['skip_hash_cross_groups'] = g
        audit_counters['skip_hash_cross_excluded'] = f
        print('[去重] 审计跨集合重复组 %d 组，排除 %d 个文件（快速模式）' % (g, f))

    enum = {}
    for split in ('train', 'valid'):
        sp = os.path.join(root, split)
        if not os.path.isdir(sp):
            print('[错误] 缺少集合目录: %s' % sp)
            return 1
        img_map, lbl_map = enumerate_split(sp, split)
        enum[split] = (img_map, lbl_map)
        print('[枚举] %-5s 图片 %d | 标签 %d' % (split, len(img_map), len(lbl_map)))

    tasks = []
    for split, (img_map, lbl_map) in enum.items():
        common = sorted(set(img_map) & set(lbl_map))
        for stem in common:
            if stem in audit_excl.get(split, set()):
                continue
            abs_img, rel_img = img_map[stem]
            abs_lbl, rel_lbl = lbl_map[stem]
            tasks.append((split, stem,
                          os.path.abspath(abs_img).replace(os.sep, '/'), rel_img,
                          os.path.abspath(abs_lbl).replace(os.sep, '/'), rel_lbl,
                          not args.skip_hash))
    print('[候选] 排除审计异常后待扫描: %d' % len(tasks))

    scan_results = []
    if tasks:
        print('\n[阶段1/2] 候选扫描（标签校验%s）...' % (' + SHA-256' if not args.skip_hash else ''))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            prog = Progress(len(tasks), '候选扫描', quiet=args.quiet)
            for rec in ex.map(scan_candidate, tasks, chunksize=64):
                prog.update()
                scan_results.append(rec)
        print()

    excluded_scan = {'label': 0, 'image': 0}
    pools = {}
    for rec in scan_results:
        if rec['error'] is not None:
            excluded_scan[rec['error_kind'] or 'label'] += 1
            continue
        pools.setdefault(rec['split'], {}).setdefault(classify(rec), []).append(rec)

    dup_stats = {'groups': 0, 'removed': 0, 'cross_groups': 0, 'cross_removed': 0,
                 'intra_groups': 0, 'intra_removed': 0}
    all_candidates = [r for sp in pools.values() for lst in sp.values() for r in lst]
    if not args.skip_hash:
        all_candidates, removed, dup_stats = dedup_pools(all_candidates, rng)
        pools = {}
        for rec in all_candidates:
            pools.setdefault(rec['split'], {}).setdefault(classify(rec), []).append(rec)
        print('[去重] 候选池 SHA-256 重复组 %d 组，移除 %d 个文件（跨集合 %d 组/%d 文件）' % (
            dup_stats['groups'], dup_stats['removed'],
            dup_stats['cross_groups'], dup_stats['cross_removed']))

    sampled = {}
    leftover = {}
    shortfalls = {}
    notes = []
    for split, size in (('train', args.train_size), ('valid', args.valid_size)):
        pool_by_cat = pools.get(split, {})
        quotas = largest_remainder_quotas(size, CATEGORY_RATIOS)
        chosen, left_by_cat = [], {}
        for cat in CATEGORIES:
            pool = pool_by_cat.get(cat, [])
            q = min(quotas[cat], len(pool))
            if q < quotas[cat]:
                shortfalls.setdefault(split, {})[cat] = quotas[cat] - q
            picked = rng.sample(pool, q) if q else []
            chosen.extend(picked)
            picked_ids = {id(r) for r in picked}
            left_by_cat[cat] = [r for r in pool if id(r) not in picked_ids]
        if len(chosen) < size:
            leftovers_all = [r for lst in left_by_cat.values() for r in lst]
            need = size - len(chosen)
            extra = rng.sample(leftovers_all, min(need, len(leftovers_all))) if leftovers_all else []
            chosen.extend(extra)
            extra_ids = {id(r) for r in extra}
            for cat in CATEGORIES:
                left_by_cat[cat] = [r for r in left_by_cat[cat] if id(r) not in extra_ids]
            if len(extra) < need:
                notes.append('%s: 目标 %d，可用候选不足，实际 %d' % (split, size, len(chosen)))
            else:
                notes.append('%s: 目标 %d，%d 张由其他类别补齐' % (split, size, need))
        sampled[split] = chosen
        leftover[split] = left_by_cat

    print('\n[阶段2/2] 样本内 SHA-256 重复校验与整理...')
    sampled, dup_notes = verify_and_refill(sampled, leftover, rng)
    notes.extend(dup_notes)

    for split in ('train', 'valid'):
        print('[抽样] %-5s 实际 %d | fire-only %d / smoke-only %d / both %d / negative %d' % (
            split, len(sampled[split]),
            sum(1 for r in sampled[split] if classify(r) == 'fire_only'),
            sum(1 for r in sampled[split] if classify(r) == 'smoke_only'),
            sum(1 for r in sampled[split] if classify(r) == 'both'),
            sum(1 for r in sampled[split] if classify(r) == 'negative')))

    out_dir = os.path.abspath(args.output_dir)
    report_dir = os.path.abspath(args.report_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    txt_train = os.path.join(out_dir, 'smoke_train.txt')
    txt_val = os.path.join(out_dir, 'smoke_val.txt')
    yaml_path = os.path.join(out_dir, 'fire_smoke_smoke.yaml')
    write_txt(txt_train, [r['abs_img'] for r in sampled['train']])
    write_txt(txt_val, [r['abs_img'] for r in sampled['valid']])
    write_yaml(yaml_path)

    print('\n[验证] 检查路径存在性与标签可读性...')
    val_out = validate_outputs(sampled['train'], sampled['valid'])
    print('[验证] 图片 %d（缺失 %d）| 标签 %d（缺失 %d，解析错误 %d）' % (
        val_out['images_checked'], len(val_out['images_missing']),
        val_out['labels_checked'], len(val_out['labels_missing']),
        len(val_out['label_errors'])))

    print('[验证] Ultralytics 数据集加载测试...')
    ult = ultralytics_load_test(yaml_path) if not args.no_ultralytics else {
        'status': 'SKIPPED', 'message': '--no-ultralytics 指定', 'details': {}}
    print('[验证] %s: %s' % (ult['status'], ult['message']))

    report = build_report(args, audit, audit_counters, excluded_scan, dup_stats, pools,
                          sampled, leftover, shortfalls, notes, val_out, ult, t0)
    json_path = os.path.join(report_dir, 'smoke_subset_report.json')
    md_path = os.path.join(report_dir, 'smoke_subset_report.md')
    write_json(json_path, report)
    write_markdown(md_path, report)

    print('=' * 62)
    print('完成，耗时 %.1fs' % (time.time() - t0))
    print('train %d | valid %d | 结论: %s' % (
        len(sampled['train']), len(sampled['valid']), report['conclusion']['level']))
    print('输出: %s' % txt_train)
    print('      %s' % txt_val)
    print('      %s' % yaml_path)
    print('      %s' % json_path)
    print('      %s' % md_path)
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())