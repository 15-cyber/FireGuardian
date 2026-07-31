"""
============================================================
FireGuardian - 智能火灾监测系统
程序入口（仅负责启动 GUI，不包含业务逻辑）
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

    from PyQt6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    log.info("FireGuardian GUI 已显示")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())