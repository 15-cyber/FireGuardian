
"""
============================================================
FireGuardian 统一路径管理 (PathManager)
功能：
  - 集中生成 screenshots / reports / logs 等输出路径
  - 路径基准统一来自 config.yaml，避免各模块自行拼接
  - 供 M8 截图、M10 报告等模块使用

可独立运行测试: python utils/path_manager.py
============================================================
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import load_config, project_root


def sanitize_dir_name(name: str) -> str:
    """将任意字符串转换为安全的目录名（仅保留字母/数字/中划线/下划线）"""
    safe = re.sub(r"[^A-Za-z0-9\-_]", "_", str(name))
    return safe.strip("_") or "event"


class PathManager:
    """
    统一输出路径管理器

    所有模块的输出目录都通过这里获取，保证：
      - 相对路径统一以项目根目录为基准
      - 修改输出位置只需改 config.yaml
      - 打包 / 清理 / 跨平台保持一致
    """

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or load_config()
        self._root = project_root()

    # ======================== 基础目录 ========================

    def _resolve(self, raw: Optional[str], default: str) -> Path:
        """相对路径以项目根目录为基准，绝对路径原样保留"""
        p = Path(raw) if raw else Path(default)
        if not p.is_absolute():
            p = self._root / p
        return p

    def screenshots_dir(self) -> Path:
        """截图根目录（M8）"""
        sc = self._cfg.get("screenshot", {})
        return self._resolve(sc.get("output_dir"), "screenshots")

    def screenshot_event_dir(self, event_id: str) -> Path:
        """单个事件截图目录（M8）：screenshots/event_<event_id>/"""
        return self.screenshots_dir() / f"event_{sanitize_dir_name(event_id)}"

    def reports_dir(self) -> Path:
        """报告根目录（M10）"""
        rep = self._cfg.get("report", {})
        return self._resolve(rep.get("output_dir"), "reports")

    def report_event_dir(self, event_id: str) -> Path:
        """单个事件报告目录（M10）：reports/event_<event_id>/"""
        return self.reports_dir() / f"event_{sanitize_dir_name(event_id)}"

    def logs_dir(self) -> Path:
        """日志目录（M9）"""
        log_cfg = self._cfg.get("log", {})
        raw = log_cfg.get("file", "logs/fireguardian.log")
        return self._resolve(raw, "logs").parent

    def runs_dir(self) -> Path:
        """运行输出根目录（检测/模拟/训练）"""
        for key, default in (
            ("detect", "runs/detect"),
            ("video_detect", "runs/video_detect"),
            ("simulator", "runs/simulator"),
        ):
            section = self._cfg.get(key, {})
            raw = section.get("output_dir")
            if raw:
                return self._resolve(raw, default)
        return self._root / "runs"


# ======================== 命令行测试 ========================

def main():
    print("\n  FireGuardian PathManager 测试")
    print("  ===============================================\n")

    pm = PathManager()
    paths = {
        "screenshots_dir": pm.screenshots_dir(),
        "screenshot_event_dir": pm.screenshot_event_dir("FE-20260731120000-ab12cd"),
        "reports_dir": pm.reports_dir(),
        "report_event_dir": pm.report_event_dir("FE-20260731120000-ab12cd"),
        "logs_dir": pm.logs_dir(),
        "runs_dir": pm.runs_dir(),
    }
    for name, p in paths.items():
        print(f"  {name:24s} -> {p}")

    # 安全目录名测试
    assert sanitize_dir_name("FE-20260731-001") == "FE-20260731-001"
    assert sanitize_dir_name("1:2") == "1_2"
    assert sanitize_dir_name("...") == "event"
    print("\n  PathManager 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
