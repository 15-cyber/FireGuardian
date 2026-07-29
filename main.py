"""
============================================================
FireGuardian - 智能火灾监测系统
程序入口（仅负责启动，不超过 500 行）
============================================================
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from logs.logger import get_logger

log = get_logger("Main")


def main():
    log.info("=" * 50)
    log.info("FireGuardian 启动中...")
    log.info("=" * 50)

    # TODO: 后续在此初始化各模块并启动 GUI
    # from ui.main_window import MainWindow
    # from PyQt6.QtWidgets import QApplication

    log.info("FireGuardian 启动完成 ✅")


if __name__ == "__main__":
    main()
