"""
============================================================
M1 - 数据管理模块 (Data Manager)
功能：
  - 自动扫描数据集目录结构（快速，仅文件列表）
  - 检查 images / labels 一一对应关系
  - 统计类别分布（采样统计，避免大数据集卡顿）
  - 检查缺失标签 / 孤立标签
  - 输出完整数据统计报告

可独立运行: python train/data_manager.py
============================================================
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import load_config


# ======================== 支持的格式 ========================
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_LABEL_EXTENSION = ".txt"


# ======================== 数据类 ========================

@dataclass
class SplitStats:
    """单个数据集划分的统计"""
    name: str
    image_count: int = 0
    label_count: int = 0
    matched_count: int = 0
    missing_count: int = 0
    orphan_count: int = 0
    missing_samples: List[str] = field(default_factory=list)
    orphan_samples: List[str] = field(default_factory=list)
    class_distribution: Dict[int, int] = field(default_factory=dict)

    @property
    def total_instances(self) -> int:
        return sum(self.class_distribution.values())

    def summary_dict(self) -> dict:
        return {
            "split": self.name,
            "images": self.image_count,
            "labels": self.label_count,
            "matched": self.matched_count,
            "missing": self.missing_count,
            "orphan": self.orphan_count,
            "instances": self.total_instances,
            "class_dist": self.class_distribution,
        }


@dataclass
class DatasetReport:
    """完整数据集报告"""
    dataset_path: str
    splits: Dict[str, SplitStats] = field(default_factory=dict)
    total_images: int = 0
    total_labels: int = 0
    total_matched: int = 0
    total_missing: int = 0
    total_orphan: int = 0
    total_instances: int = 0
    class_names: Dict[int, str] = field(default_factory=dict)
    scan_time: float = 0.0
    label_scan_mode: str = ""

    def print_report(self) -> None:
        """打印报告到控制台"""
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  FireGuardian 数据集报告")
        print(f"  {self.dataset_path}")
        print(f"{sep}")
        print(f"  总图片:     {self.total_images:>8,}")
        print(f"  总标签:     {self.total_labels:>8,}")
        print(f"  有效配对:   {self.total_matched:>8,}")
        print(f"  缺失标签:   {self.total_missing:>8,}")
        print(f"  孤立标签:   {self.total_orphan:>8,}")
        print(f"  标注实例:   {self.total_instances:>8,}")
        print(f"  扫描耗时:   {self.scan_time:.1f}s {self.label_scan_mode}")

        # 全局类别分布
        if self.total_instances > 0:
            merged: Dict[int, int] = {}
            for sp in self.splits.values():
                for cid, cnt in sp.class_distribution.items():
                    merged[cid] = merged.get(cid, 0) + cnt
            print(f"{sep}")
            print(f"  类别分布:")
            for cid in sorted(merged.keys()):
                name = self.class_names.get(cid, f"class_{cid}")
                pct = merged[cid] / self.total_instances * 100
                bar_len = int(pct / 2)
                bar = "#" * bar_len + "-" * (50 - bar_len)
                print(f"     [{cid}] {name:<8s}  {merged[cid]:>8,}  ({pct:5.1f}%)  {bar}")
        print(f"{sep}")

        # 各 split 详情
        for name, sp in self.splits.items():
            print(f"\n  [{name.upper()}]")
            print(f"     图片: {sp.image_count:>8,}  标签: {sp.label_count:>8,}  配对: {sp.matched_count:>8,}")
            if sp.missing_count > 0:
                print(f"     ⚠ 缺失标签: {sp.missing_count} 个 (例如: {sp.missing_samples[:5]})")
            if sp.orphan_count > 0:
                print(f"     ⚠ 孤立标签: {sp.orphan_count} 个 (例如: {sp.orphan_samples[:5]})")
            if sp.class_distribution:
                parts = [f"cls{cid}={cnt}" for cid, cnt in sorted(sp.class_distribution.items())]
                print(f"     标注采样: {', '.join(parts)} (共 {sp.total_instances} 实例)")

        print(f"\n{sep}\n")

    def is_healthy(self) -> bool:
        return self.total_missing == 0 and self.total_orphan == 0 and self.total_instances > 0


# ======================== 快速文件扫描（纯字符串操作） ========================

def _scan_files(dir_path: str, extensions: Set[str]) -> Tuple[Dict[str, str], int]:
    """
    快速扫描目录，返回 {stem: filename} 字典。
    纯字符串操作，避免 Path 对象的 is_file() 开销。
    """
    result: Dict[str, str] = {}
    if not os.path.isdir(dir_path):
        return result, 0
    try:
        for name in os.listdir(dir_path):
            lower = name.lower()
            dot_idx = lower.rfind(".")
            if dot_idx == -1:
                continue
            ext = lower[dot_idx:]
            if ext in extensions:
                stem = name[:name.rfind(".")]
                result[stem] = name
    except PermissionError:
        pass
    return result, len(result)


def _count_label_instances(labels_dir: str, stems: Set[str],
                           max_read: Optional[int] = 2000) -> Dict[int, int]:
    """
    批量读取标签文件，统计类别分布。
    """
    class_dist: Dict[int, int] = {}
    count = 0
    for stem in stems:
        lbl_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.isfile(lbl_path):
            continue
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        class_dist[cid] = class_dist.get(cid, 0) + 1
        except Exception:
            pass
        count += 1
        if max_read is not None and count >= max_read:
            break
    return class_dist


def scan_split(
    dataset_path: str,
    split_name: str,
    analyze_labels: bool = False,
    sample_size: Optional[int] = 500,
) -> SplitStats:
    """
    扫描一个数据集划分（train/valid/test）
    """
    split_path = os.path.join(dataset_path, split_name)
    img_dir = os.path.join(split_path, "images")
    lbl_dir = os.path.join(split_path, "labels")

    stats = SplitStats(name=split_name)

    images, img_count = _scan_files(img_dir, _IMAGE_EXTENSIONS)
    labels, lbl_count = _scan_files(lbl_dir, {_LABEL_EXTENSION})

    stats.image_count = img_count
    stats.label_count = lbl_count

    image_stems = set(images.keys())
    label_stems = set(labels.keys())
    matched_stems = image_stems & label_stems
    stats.matched_count = len(matched_stems)

    missing_stems = image_stems - label_stems
    orphan_stems = label_stems - image_stems
    stats.missing_count = len(missing_stems)
    stats.orphan_count = len(orphan_stems)
    stats.missing_samples = sorted(missing_stems)[:10]
    stats.orphan_samples = sorted(orphan_stems)[:10]

    # 标签内容分析
    if analyze_labels and matched_stems:
        stats.class_distribution = _count_label_instances(
            lbl_dir, matched_stems, max_read=sample_size
        )

    return stats


def scan_dataset(
    dataset_path: str,
    splits: Optional[List[str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    analyze_labels: bool = True,
    label_sample_size: Optional[int] = 500,
) -> DatasetReport:
    """
    扫描完整数据集
    """
    if splits is None:
        splits = ["train", "valid", "test"]
    if class_names is None:
        class_names = {}

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    t_start = time.time()
    report = DatasetReport(
        dataset_path=os.path.abspath(dataset_path),
        class_names=class_names,
    )

    for split_name in splits:
        sp = scan_split(
            dataset_path,
            split_name,
            analyze_labels=analyze_labels,
            sample_size=label_sample_size,
        )
        report.splits[split_name] = sp
        report.total_images += sp.image_count
        report.total_labels += sp.label_count
        report.total_matched += sp.matched_count
        report.total_missing += sp.missing_count
        report.total_orphan += sp.orphan_count
        report.total_instances += sp.total_instances

    report.scan_time = time.time() - t_start
    mode_str = ""
    if analyze_labels:
        mode_str = f"(标签采样={label_sample_size})" if label_sample_size else "(标签全量)"
    report.label_scan_mode = mode_str

    return report


def verify_dataset_health(report: DatasetReport) -> List[str]:
    """返回数据集存在的问题列表，空列表=健康"""
    issues = []
    if report.total_images == 0:
        issues.append("数据集为空，未找到任何图片")
    if report.total_instances == 0:
        issues.append("未找到任何标注实例，请检查标签文件格式")
    if report.total_missing > 0:
        issues.append(f"有 {report.total_missing:,} 张图片缺少对应的标签文件")
    if report.total_orphan > 0:
        issues.append(f"有 {report.total_orphan:,} 个标签文件没有对应的图片")
    return issues


# ======================== 主入口 ========================

def main():
    cfg = load_config()
    dataset_path = cfg["dataset"]["path"]
    raw_names = cfg["dataset"].get("class_names", {})
    class_names = {int(k): v for k, v in raw_names.items()}

    print()
    print(f"  FireGuardian M1 - 数据管理模块")
    print(f"  {'=' * 42}")
    print(f"\n  数据集: {dataset_path}")

    report = scan_dataset(
        dataset_path,
        splits=["train", "valid", "test"],
        class_names=class_names,
        analyze_labels=True,
        label_sample_size=500,
    )

    report.print_report()

    issues = verify_dataset_health(report)
    if issues:
        print("  [WARN] 数据集存在以下问题:")
        for i in issues:
            print(f"    - {i}")
        return 1
    else:
        print("  [OK] 数据集健康，可以直接用于训练和检测！")
        return 0


if __name__ == "__main__":
    sys.exit(main())
