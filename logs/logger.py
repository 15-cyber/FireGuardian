"""
============================================================
FireGuardian 日志模块 (M9)
所有模块统一通过此模块记录日志
可独立运行: python logs/logger.py
============================================================
"""
import sys
from pathlib import Path
from loguru import logger
import yaml


# ---------- 全局配置加载 ----------
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _log_cfg = _cfg.get("log", {})
else:
    _log_cfg = {}


# ---------- 日志配置参数 ----------
_LOG_LEVEL = _log_cfg.get("level", "INFO")
_LOG_FILE = Path(_log_cfg.get("file", "logs/fireguardian.log"))
_MAX_BYTES = _log_cfg.get("max_bytes", 10 * 1024 * 1024)
_BACKUP_COUNT = _log_cfg.get("backup_count", 5)
_FORMAT = _log_cfg.get(
    "format",
    "[{time:YYYY-MM-DD HH:mm:ss.SSS}] <level>{level:^8}</level> | {name} | {message}",
)
_COLORIZE = _log_cfg.get("colorize", True)


# ---------- 移除默认 handler，添加自定义 handler ----------
logger.remove()

# 控制台输出（带颜色）
logger.add(
    sys.stderr,
    level=_LOG_LEVEL,
    format=_FORMAT,
    colorize=_COLORIZE,
    backtrace=True,
    diagnose=False,
)

# 文件输出（自动轮转）
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    str(_LOG_FILE),
    level=_LOG_LEVEL,
    format=_FORMAT.replace("<level>{level:^8}</level>", "{level:^8}"),
    rotation=_MAX_BYTES,
    retention=_BACKUP_COUNT,
    encoding="utf-8",
    enqueue=True,
)


def get_logger(name: str = "FireGuardian"):
    """获取全局统一的日志记录器"""
    return logger.bind(name=name)


# ========== 独立运行测试 ==========
if __name__ == "__main__":
    log = get_logger("Test")
    log.debug("这是 DEBUG 级别的日志")
    log.info("这是 INFO 级别的日志")
    log.warning("这是 WARNING 级别的日志")
    log.error("这是 ERROR 级别的日志")
    log.success("日志模块初始化成功！")
    print(f"\n日志文件位置: {_LOG_FILE.resolve()}")
