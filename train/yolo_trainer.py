"""
============================================================
M2 - YOLO 训练模块 (YOLO Trainer)
功能：
  - 加载预训练模型 (yolo11n.pt) 并在火灾数据集上微调
  - 实时显示训练指标：Epoch / Loss / Precision / Recall / mAP50 / mAP50-95
  - 自动保存 best.pt 到 models/
  - 训练历史可视化

可独立运行: python train/yolo_trainer.py
============================================================
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# 必须先设置此环境变量，再导入 ultralytics
_ROOT = Path(__file__).resolve().parent.parent
_ULTRA_CONFIG = _ROOT / ".venv" / "ultralytics_config"
_ULTRA_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ROOT / ".venv"))

# ---------- Ultralytics 导入 ----------
from ultralytics import YOLO

# ---------- 项目内部导入 ----------
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import load_config, ensure_dir


# ======================== 常量 ========================
_DATASET_YAML = _ROOT / "datasets" / "fire.yaml"
_MODEL_SOURCE = _ROOT / "yolo11n.pt"
_MODEL_TARGET_DIR = _ROOT / "models"


# ======================== 训练器类 ========================

class TrainingStopped(Exception):
    """用户请求停止训练时抛出（在 epoch 回调边界安全退出，由 GUI 捕获）"""


class YOLOTrainer:
    """YOLO 训练器，封装训练全流程"""

    def __init__(self, config_path: Optional[str | Path] = None,
                 weights: Optional[str | Path] = None,
                 project: Optional[str] = None,
                 name: Optional[str] = None,
                 dataset_yaml: Optional[str | Path] = None):
        self.cfg = load_config() if config_path is None else load_config(reload=True)
        self.train_cfg = self.cfg.get("train", {})
        self.model_cfg = self.cfg.get("model", {})

        # 训练参数
        self.epochs = self.train_cfg.get("epochs", 100)
        self.batch = self.train_cfg.get("batch", 16)
        self.imgsz = self.train_cfg.get("imgsz", 640)
        self.lr = self.train_cfg.get("lr", 0.01)
        self.patience = self.train_cfg.get("patience", 20)
        self.workers = self.train_cfg.get("workers", 4)
        self.pretrained = self.train_cfg.get("pretrained", True)
        # 可覆盖参数（GUI 传入，None 则使用配置）
        self.weights = Path(weights) if weights else _MODEL_SOURCE
        self.project = project if project else self.train_cfg.get("project", "train")
        self.exp_name = name if name else self.train_cfg.get("name", "exp")
        self._stop_requested = False
        self.device = self.model_cfg.get("device", 0)
        self.exist_ok = self.train_cfg.get("exist_ok", True)

        # 回调状态
        self._current_epoch = 0
        self._train_start_time: float = 0.0
        self._best_map = 0.0
        self._metrics_history: Dict[str, list] = {
            "epoch": [],
            "loss": [],
            "precision": [],
            "recall": [],
            "mAP50": [],
            "mAP50-95": [],
        }

        # 模型和数据集
        self.model: Optional[YOLO] = None
        self.dataset_yaml = str(Path(dataset_yaml) if dataset_yaml else _DATASET_YAML)
        self._validate_paths()

    def _validate_paths(self):
        """验证模型和数据集配置是否存在"""
        if not Path(self.dataset_yaml).exists():
            raise FileNotFoundError(
                f"数据集配置文件不存在: {self.dataset_yaml}\n"
                f"请确保已创建 datasets/fire.yaml"
            )
        if self.pretrained and not self.weights.exists():
            raise FileNotFoundError(
                f"预训练模型不存在: {self.weights}\n"
                f"请先下载 yolo11n.pt"
            )

    def _setup_callbacks(self, callback=None):
        """
        注册训练回调，实时显示指标

        参数:
            callback: 可选 callable(dict)，每 epoch / 结束时收到指标与状态
        """
        self._train_callback = callback
        from ultralytics.utils.callbacks import add_integration_callbacks

        def on_train_epoch_end(trainer):
            """每个 epoch 结束时记录指标（含停止请求检查）"""
            if self._stop_requested:
                raise TrainingStopped("用户请求停止训练")
            self._current_epoch = trainer.epoch

            # 收集当前指标
            if hasattr(trainer, "metrics") and trainer.metrics:
                m = trainer.metrics
                self._metrics_history["epoch"].append(trainer.epoch)
                self._metrics_history["loss"].append(m.get("loss", 0))
                self._metrics_history["precision"].append(m.get("metrics/precision(B)", 0))
                self._metrics_history["recall"].append(m.get("metrics/recall(B)", 0))
                self._metrics_history["mAP50"].append(m.get("metrics/mAP50(B)", 0))
                self._metrics_history["mAP50-95"].append(m.get("metrics/mAP50-95(B)", 0))

                # 更新最佳 mAP
                current_map = m.get("metrics/mAP50(B)", 0)
                if current_map > self._best_map:
                    self._best_map = current_map

                # 实时打印
                elapsed = time.time() - self._train_start_time
                eta = (elapsed / max(trainer.epoch, 1)) * (self.epochs - trainer.epoch)
                loss_val = m.get("loss", 0)
                p = m.get("metrics/precision(B)", 0)
                r = m.get("metrics/recall(B)", 0)
                map50 = m.get("metrics/mAP50(B)", 0)
                map95 = m.get("metrics/mAP50-95(B)", 0)

                print(
                    f"  Epoch {trainer.epoch:>3}/{self.epochs} | "
                    f"Loss: {loss_val:.4f} | "
                    f"P: {p:.4f} | "
                    f"R: {r:.4f} | "
                    f"mAP50: {map50:.4f} | "
                    f"mAP50-95: {map95:.4f} | "
                    f"ETA: {eta:.0f}s"
                )

                # 转发给 GUI / 外部监听
                if self._train_callback:
                    self._train_callback({
                        "kind": "epoch",
                        "epoch": int(trainer.epoch),
                        "epochs": int(self.epochs),
                        "loss": round(float(loss_val), 4),
                        "precision": round(float(p), 4),
                        "recall": round(float(r), 4),
                        "mAP50": round(float(map50), 4),
                        "mAP50-95": round(float(map95), 4),
                        "best_map": round(float(self._best_map), 4),
                        "eta_s": round(float(eta), 1),
                    })

        def on_train_end(trainer):
            """训练结束时打印总结"""
            total = time.time() - self._train_start_time
            print(f"\n  {'=' * 50}")
            print(f"  训练完成！总耗时: {total:.1f}s ({total/60:.1f} 分钟)")
            print(f"  最佳 mAP50: {self._best_map:.4f}")
            print(f"  最佳模型: {trainer.best}")
            print(f"  最后模型: {trainer.last}")
            print(f"  {'=' * 50}")

            # 转发给 GUI / 外部监听
            if self._train_callback:
                self._train_callback({
                    "kind": "done",
                    "best_map": round(float(self._best_map), 4),
                    "best": str(trainer.best),
                    "last": str(trainer.last),
                    "total_seconds": round(float(total), 1),
                })

        # 注册回调
        YOLO.add_callback("on_train_epoch_end", on_train_epoch_end)
        YOLO.add_callback("on_train_end", on_train_end)

    def train(self,
              epochs: Optional[int] = None,
              batch: Optional[int] = None,
              imgsz: Optional[int] = None,
              device: Optional[str | int] = None,
              weights: Optional[str | Path] = None,
              project: Optional[str] = None,
              name: Optional[str] = None,
              dataset_yaml: Optional[str | Path] = None,
              callback: Optional[callable] = None) -> Path:
        """
        执行训练（参数 None 时使用实例/配置默认值）

        参数:
            epochs/batch/imgsz/device: 训练参数覆盖
            weights: 预训练权重覆盖
            project/name: 输出目录覆盖
            callback: 每 epoch / 训练结束回调 callable(dict)

        返回:
            best_model_path: 最佳模型路径
        """
        # 参数覆盖（None → 实例默认值）
        epochs = epochs if epochs is not None else self.epochs
        batch = batch if batch is not None else self.batch
        imgsz = imgsz if imgsz is not None else self.imgsz
        device = device if device is not None else self.device
        weights = Path(weights) if weights else self.weights
        project = project if project else self.project
        exp_name = name if name else self.exp_name
        dataset_yaml = str(Path(dataset_yaml) if dataset_yaml else self.dataset_yaml)
        self._stop_requested = False

        print(f"\n  {'=' * 50}")
        print(f"  FireGuardian M2 - YOLO 训练")
        print(f"  {'=' * 50}")
        print(f"  模型: {weights}")
        print(f"  数据集: {self.dataset_yaml}")
        print(f"  配置: {epochs} epochs, batch={batch}, imgsz={imgsz}")
        print(f"  Device: {device}")
        print(f"  {'=' * 50}\n")

        # 加载模型
        model_path = str(weights)
        self.model = YOLO(model_path)

        # 设置回调
        self._setup_callbacks(callback)

        # 开始计时
        self._train_start_time = time.time()

        # 执行训练
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=self.lr,
            patience=self.patience,
            device=device,
            workers=self.workers,
            project=project,
            name=exp_name,
            exist_ok=self.exist_ok,
            pretrained=self.pretrained,
            amp=True,
            verbose=False,
        )

        # 同步实例参数（供 plot_metrics / 复制模型使用）
        self.project = project
        self.exp_name = exp_name
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device

        # 复制最佳模型到 models/best.pt
        best_source = _ROOT / project / exp_name / "weights" / "best.pt"
        ensure_dir(str(_MODEL_TARGET_DIR))
        best_target = _MODEL_TARGET_DIR / "best.pt"

        if best_source.exists():
            import shutil
            shutil.copy2(str(best_source), str(best_target))
            print(f"\n  ✅ 最佳模型已保存到: {best_target}")
        else:
            print(f"\n  ⚠ 未找到最佳模型: {best_source}")

        return best_target

    def request_stop(self):
        """请求停止训练（下一个 epoch 边界生效）"""
        self._stop_requested = True

    def plot_metrics(self, save_path: Optional[str | Path] = None):
        """绘制训练指标曲线图"""
        if not self._metrics_history["epoch"]:
            print("  暂无训练数据，无法绘图")
            return

        try:
            import matplotlib.pyplot as plt

            hist = self._metrics_history
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle("FireGuardian YOLO Training Metrics", fontsize=14)

            plots = [
                (axes[0, 0], "epoch", "loss", "Loss"),
                (axes[0, 1], "epoch", "precision", "Precision"),
                (axes[0, 2], "epoch", "recall", "Recall"),
                (axes[1, 0], "epoch", "mAP50", "mAP50"),
                (axes[1, 1], "epoch", "mAP50-95", "mAP50-95"),
            ]

            for ax, x_key, y_key, title in plots:
                if hist[x_key] and hist[y_key]:
                    ax.plot(hist[x_key], hist[y_key], "b-", linewidth=1.5)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel(title)
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)

            # 隐藏多余的子图
            axes[1, 2].axis("off")

            plt.tight_layout()

            if save_path:
                plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
                print(f"  指标图已保存: {save_path}")

            plt.show()
        except ImportError:
            print("  matplotlib 未安装，跳过绘图")


# ======================== 命令行接口 ========================

def main():
    """主入口：启动训练"""
    trainer = YOLOTrainer()
    best_path = trainer.train()
    print(f"\n  训练完成。使用以下模型进行检测: {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
