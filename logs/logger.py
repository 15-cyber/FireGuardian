"""
============================================================
FireGuardian 日志模块 (M9) — 按核查文档完善
功能：
  - 系统日志与事件日志分离：
      logs/application.log  程序运行与模块状态
      logs/error.log        异常堆栈（ERROR 及以上）
      logs/events.jsonl     结构化火灾事件记录（JSON Lines）
  - 统一事件日志字段：time / session_id / level / module /
    source_module / event_type / event_id / frame_id / details
  - session_id：每次程序运行唯一编号（run_YYYYMMDD_HHMMSS）
  - 模块耗时统计：log_elapsed / timed 上下文计时器
  - 日志轮转（rotation / retention / compression）
  - 敏感信息脱敏（API Key / token 等不落盘）
  - Debug 开关：debug=true 时记录 DEBUG 级详细日志
  - 日志写入失败不导致程序退出
  - 程序关闭时输出统计摘要并保存 logs/run_summary.json

所有模块统一通过 get_logger() 记录系统日志，
通过 log_event() / log_decision() / log_elapsed() 记录结构化事件日志。

可独立运行: python logs/logger.py
============================================================
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from utils.common import load_config


# ======================== 配置加载 ========================

def _resolve_path(raw: str) -> Path:
    """相对路径以项目根目录为基准"""
    p = Path(raw)
    if not p.is_absolute():
        p = _ROOT / p
    return p


_cfg = load_config()
_log_cfg = _cfg.get("log", {})

_LOG_LEVEL = _log_cfg.get("level", "INFO")
_DEBUG = bool(_log_cfg.get("debug", False))
# Debug 开关：debug=true 时提升为 DEBUG 级（核查文档改动七）
_EFFECTIVE_LEVEL = "DEBUG" if _DEBUG else str(_LOG_LEVEL).upper()
_APP_FILE = _resolve_path(str(
    _log_cfg.get("application_file") or _log_cfg.get("file") or "logs/application.log"))
_ERR_FILE = _resolve_path(str(_log_cfg.get("error_file") or "logs/error.log"))
_EVENTS_FILE = _resolve_path(str(_log_cfg.get("events_file") or "logs/events.jsonl"))
_SUMMARY_FILE = _resolve_path(str(_log_cfg.get("run_summary_file") or "logs/run_summary.json"))

# 兼容旧配置 max_bytes / backup_count
_ROTATION = _log_cfg.get("rotation", _log_cfg.get("max_bytes", "10 MB"))
_RETENTION = _log_cfg.get("retention", _log_cfg.get("backup_count", "30 days"))
_COMPRESSION = _log_cfg.get("compression", "zip")
_FORMAT = _log_cfg.get(
    "format",
    "[{time:YYYY-MM-DD HH:mm:ss.SSS}] <level>{level:^8}</level> | {extra[name]} | {message}",
)
_COLORIZE = _log_cfg.get("colorize", True)
_SENSITIVE_KEYS = {
    str(k).lower()
    for k in _log_cfg.get("sensitive_keys",
                          ["api_key", "apikey", "token", "secret", "password", "authorization"])
}

# ======================== Session ID（核查文档改动一） ========================
# 程序启动自动生成一次；可用环境变量覆盖（便于测试/多实例区分）
SESSION_ID = os.environ.get("FIREGUARDIAN_SESSION_ID") or \
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

_START_DT = datetime.now()
_STARTED_AT = _START_DT.isoformat(timespec="seconds")


# ======================== 敏感信息脱敏 ========================

def mask_sensitive(data: Any, key: str = "") -> Any:
    """递归脱敏：字段名命中敏感词时替换为 ***（改进说明 4）"""
    if isinstance(data, dict):
        return {k: mask_sensitive(v, str(k)) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_sensitive(v, key) for v in data]
    if key.lower() in _SENSITIVE_KEYS and data not in (None, ""):
        return "***"
    return data


# ======================== 事件日志写入器 ========================

class EventLogWriter:
    """结构化事件日志写入器（events.jsonl），写入失败不影响主流程"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.counts: Dict[str, int] = {}        # 按 event_type 计数
        self.danger_counts: Dict[str, int] = {}  # 按危险等级计数

    def write(self, record: dict, danger_level: Optional[str] = None) -> bool:
        """追加一条 JSON 行；失败返回 False，不抛出异常"""
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                et = record.get("event_type", "unknown")
                self.counts[et] = self.counts.get(et, 0) + 1
                if danger_level:
                    self.danger_counts[danger_level] = self.danger_counts.get(danger_level, 0) + 1
            return True
        except Exception:
            return False

# ======================== 初始化 loguru ========================

logger.remove()

# 控制台输出（带颜色）
logger.add(
    sys.stderr,
    level=_EFFECTIVE_LEVEL,
    format=_FORMAT,
    colorize=_COLORIZE,
    backtrace=True,
    diagnose=False,
)

# 系统日志 application.log（自动轮转）
_APP_FILE.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    str(_APP_FILE),
    level=_EFFECTIVE_LEVEL,
    format=_FORMAT.replace("<level>{level:^8}</level>", "{level:^8}"),
    rotation=_ROTATION,
    retention=_RETENTION,
    compression=_COMPRESSION,
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=False,
)

# 错误日志 error.log（ERROR 及以上，含堆栈）
_ERR_FILE.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    str(_ERR_FILE),
    level="ERROR",
    format=_FORMAT.replace("<level>{level:^8}</level>", "{level:^8}"),
    rotation=_ROTATION,
    retention=_RETENTION,
    compression=_COMPRESSION,
    encoding="utf-8",
    enqueue=True,
    backtrace=True,
    diagnose=False,
)

# 事件日志写入器
_event_writer = EventLogWriter(_EVENTS_FILE)


# ======================== 统一日志接口 ========================

def get_logger(name: str = "FireGuardian"):
    """获取全局统一的日志记录器（系统日志）"""
    return logger.bind(name=name)


def _extract_danger(details: dict) -> Optional[str]:
    """从详情中提取危险等级（兼容 Enum / 字符串 / metadata 内字段）"""
    raw = details.get("danger_level")
    if raw in (None, ""):
        md = details.get("metadata")
        if isinstance(md, dict):
            raw = md.get("danger_level")
    if raw in (None, ""):
        return None
    if hasattr(raw, "value"):
        raw = raw.value
    return str(raw).strip().lower()


def log_event(event, event_type: str = "event_updated",
              module: str = "event_aggregator",
              frame_id: Optional[int] = None,
              details: Optional[dict] = None,
              level: str = "INFO",
              source_module: Optional[str] = None) -> bool:
    """
    将事件以统一字段写入 events.jsonl（改进说明 2 / 核查文档改动四）。

    参数:
        event: FireEvent 或带 event_id 的对象
        event_type: event_confirmed / event_updated / event_ended / ...
        module: 来源模块名（如 event_aggregator）
        frame_id: 帧号（默认取事件代表帧）
        details: 附加详情（默认使用事件 to_dict + metadata）
        source_module: 来源模块编号（如 M6），用于快速追溯
    """
    if event is None:
        return False
    event_id = getattr(event, "event_id", "")
    if frame_id is None:
        frame_id = getattr(event, "representative_frame_id", None)
        if frame_id is None and getattr(event, "detections", None):
            frame_id = getattr(event.detections[-1], "frame_id", None)
    if details is None:
        details = dict(event.to_dict()) if hasattr(event, "to_dict") else {}
        metadata = getattr(event, "metadata", None)
        if metadata:
            details["metadata"] = metadata
    details = mask_sensitive(details)
    danger = _extract_danger(details)

    record = {
        "time": datetime.now().isoformat(timespec="milliseconds"),
        "session_id": SESSION_ID,
        "level": str(level).upper(),
        "module": module,
        "source_module": source_module or module,
        "event_type": event_type,
        "event_id": event_id,
        "frame_id": frame_id,
        "details": details,
    }
    ok = _event_writer.write(record, danger_level=danger)
    if ok:
        get_logger(module).info(f"{event_type} | {event_id} | 危险等级 {danger or 'N/A'}")
    return ok


def log_event_confirmed(event, module: str = "event_aggregator",
                        source_module: Optional[str] = None) -> bool:
    return log_event(event, "event_confirmed", module, source_module=source_module)


def log_event_updated(event, module: str = "event_aggregator",
                      source_module: Optional[str] = None) -> bool:
    return log_event(event, "event_updated", module, source_module=source_module)


def log_event_ended(event, module: str = "event_aggregator",
                    source_module: Optional[str] = None) -> bool:
    return log_event(event, "event_ended", module, source_module=source_module)


def log_event_discarded(event, module: str = "event_aggregator",
                        source_module: Optional[str] = None) -> bool:
    return log_event(event, "event_discarded", module, source_module=source_module)


def log_decision(event, decision, module: str = "fire_decision_agent",
                 source_module: Optional[str] = "M7") -> bool:
    """记录 M7 决策到 events.jsonl（event_type=decision_made）"""
    if event is None or decision is None:
        return False
    details = decision.to_dict() if hasattr(decision, "to_dict") else {"decision": str(decision)}
    return log_event(event, "decision_made", module,
                     details=details, source_module=source_module)


# ======================== 模块耗时（核查文档改动二） ========================

def log_elapsed(module: str, elapsed_ms: float, action: str = "",
                event_id: str = "", details: Optional[dict] = None,
                source_module: Optional[str] = None) -> bool:
    """记录模块耗时到 events.jsonl（event_type=elapsed），供统计平均/最大耗时与 FPS"""
    merged = dict(details or {})
    merged["elapsed_ms"] = round(float(elapsed_ms), 3)
    if action:
        merged["action"] = action
    record = {
        "time": datetime.now().isoformat(timespec="milliseconds"),
        "session_id": SESSION_ID,
        "level": "INFO",
        "module": module,
        "source_module": source_module or module,
        "event_type": "elapsed",
        "event_id": event_id,
        "frame_id": None,
        "details": mask_sensitive(merged),
    }
    ok = _event_writer.write(record)
    if ok:
        get_logger(module).debug(
            f"elapsed | {module} | {action or 'run'} | {record['details']['elapsed_ms']}ms")
    return ok


@contextmanager
def timed(module: str, action: str = "", event_id: str = "",
          source_module: Optional[str] = None) -> Iterator[None]:
    """上下文计时器：with timed('M3', action='detect'): ... 自动记录耗时"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        log_elapsed(module, elapsed_ms, action=action, event_id=event_id,
                    source_module=source_module)


# ======================== 运行统计（核查文档改动五） ========================

_runtime_stats: Dict[str, int] = {}


def add_stat(key: str, value: int = 1):
    """累计运行统计，例如 add_stat('screenshots', 3) / add_stat('reports', 1)"""
    _runtime_stats[key] = _runtime_stats.get(key, 0) + value


def get_event_stats() -> Dict[str, int]:
    """事件日志统计（按 event_type 计数）"""
    return dict(_event_writer.counts)


def get_danger_stats() -> Dict[str, int]:
    """危险等级分布统计"""
    return dict(_event_writer.danger_counts)

# ======================== 桥接与退出总结 ========================

def attach_event_logger(aggregator, module: str = "event_aggregator",
                        source_module: str = "M6"):
    """
    将事件日志桥接到 M6 聚合器回调（链式挂接，不覆盖已有回调）。

    真实管线中 M8/M9 等多个模块可依次挂接：
        screenshot_mgr.attach(aggregator)
        attach_event_logger(aggregator)
    """
    _EVENT_CALLBACKS = (
        ("on_event_confirmed", "event_confirmed"),
        ("on_event_updated", "event_updated"),
        ("on_event_ended", "event_ended"),
        ("on_event_discarded", "event_discarded"),
    )
    for attr, event_type in _EVENT_CALLBACKS:
        prev_cb = getattr(aggregator, attr, None)

        def make_handler(prev, et):
            def handler(event):
                log_event(event, et, module, source_module=source_module)
                if prev is not None:
                    prev(event)
            return handler

        setattr(aggregator, attr, make_handler(prev_cb, event_type))
    return aggregator


def _atomic_write_json(target: Path, data: dict) -> bool:
    """先写 .tmp 再替换为正式文件，避免留下损坏的 JSON"""
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(target)
        return True
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def flush():
    """等待 loguru 异步队列写入完成（测试 / 退出前调用）"""
    try:
        logger.complete()
    except Exception:
        pass


def save_run_summary(path: Optional[Path] = None) -> Optional[Path]:
    """程序退出时保存运行摘要 JSON（核查文档改动五）"""
    target = Path(path) if path else _SUMMARY_FILE
    summary = {
        "session_id": SESSION_ID,
        "started_at": _STARTED_AT,
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round((datetime.now() - _START_DT).total_seconds(), 1),
        "debug": _DEBUG,
        "total_events": sum(_event_writer.counts.values()),
        "event_types": dict(_event_writer.counts),
        "danger_levels": dict(_event_writer.danger_counts),
        "runtime_stats": dict(_runtime_stats),
    }
    return target if _atomic_write_json(target, summary) else None


def print_summary():
    """程序关闭时输出统计摘要，并保存 run_summary.json"""
    summary_path = save_run_summary()
    stats = get_event_stats()
    total = sum(stats.values())
    print("\n" + "=" * 48)
    print("  FireGuardian 日志统计摘要")
    print("=" * 48)
    print(f"  session_id: {SESSION_ID}")
    print(f"  事件日志总数: {total}")
    for et, n in sorted(stats.items()):
        print(f"    {et}: {n}")
    print(f"  危险等级分布: {get_danger_stats()}")
    print(f"  运行统计: {_runtime_stats}")
    print(f"  系统日志: {_APP_FILE}")
    print(f"  错误日志: {_ERR_FILE}")
    print(f"  事件日志: {_EVENTS_FILE}")
    if summary_path:
        print(f"  运行摘要: {summary_path}")
    print("=" * 48)


# ======================== 独立运行测试 ========================

def main():
    print("\n  FireGuardian M9 - 日志模块测试（按核查文档）")
    print("  =====================================================\n")

    import json as _json
    from utils.common import (FireEvent, EventStatus, GrowthTrend,
                              FireDecision, DangerLevel)

    # ---------- 1. 系统日志 / 错误日志分离 + 堆栈 + Warning ----------
    log = get_logger("Test")
    log.info("M9 test: application log")
    log.warning("M9 test: warning")
    try:
        raise ValueError("M9 fake exception")
    except ValueError:
        log.exception("M9 caught exception with traceback")
    flush()

    app_text = _APP_FILE.read_text(encoding="utf-8")
    err_text = _ERR_FILE.read_text(encoding="utf-8")
    assert "M9 test: application log" in app_text, "application.log 缺少系统日志"
    assert "M9 test: warning" in app_text, "application.log 缺少 WARNING"
    assert "M9 fake exception" in err_text, "error.log 缺少异常"
    assert "Traceback" in err_text, "error.log 缺少堆栈"
    print("    系统/错误日志分离 + 堆栈 + Warning [OK]")

    # ---------- 2. 结构化事件日志 + session_id + source_module + 脱敏 ----------
    event = FireEvent(
        event_id="FE-LOG-0001",
        status=EventStatus.CONFIRMED,
        start_timestamp=0.0,
        last_timestamp=3.0,
        total_frames=8,
        positive_frames=6,
        consecutive_positive_frames=6,
        max_fire_area_ratio=0.12,
        max_smoke_area_ratio=0.05,
        avg_fire_confidence=0.85,
        avg_smoke_confidence=0.70,
        growth_trend=GrowthTrend.INCREASING,
        source_type="test",
        source_name="log_test",
        representative_frame_id=125,
        metadata={"danger_level": "medium", "api_key": "sk-1234567890"},
    )
    log_event_confirmed(event, source_module="M6")
    log_event_updated(event, source_module="M6")

    decision = FireDecision(
        event_id="FE-LOG-0001",
        danger_level=DangerLevel.HIGH,
        score=0.55,
        confidence=0.80,
        reasons=["火焰面积较大"],
        suggestions=["立即检查现场"],
        decision_source="rule_engine",
    )
    log_decision(event, decision, source_module="M7")

    event.status = EventStatus.ENDED
    log_event_ended(event, source_module="M6")
    flush()

    events_text = _EVENTS_FILE.read_text(encoding="utf-8")
    lines = [_json.loads(l) for l in events_text.splitlines() if l.strip()]
    recs = [l for l in lines if l.get("session_id") == SESSION_ID]
    assert len(recs) >= 4, f"本次运行事件日志不足: {len(recs)}"
    rec = recs[0]
    for key in ("time", "session_id", "level", "module", "source_module",
                "event_type", "event_id", "frame_id", "details"):
        assert key in rec, f"事件日志缺少字段: {key}"
    assert rec["session_id"] == SESSION_ID
    assert rec["event_id"] == "FE-LOG-0001"
    assert rec["frame_id"] == 125
    assert rec["event_type"] == "event_confirmed"
    assert rec["source_module"] == "M6"
    assert "sk-1234567890" not in events_text, "API Key 未脱敏"
    print("    事件统一字段 + session_id + source_module + 脱敏 [OK]")

    # ---------- 3. 模块耗时（log_elapsed / timed） ----------
    with timed("M3", action="detect", source_module="M3"):
        time.sleep(0.01)
    log_elapsed("M7", 5.2, action="decision", event_id="FE-LOG-0001", source_module="M7")
    flush()
    elapsed_stats = get_event_stats()
    assert elapsed_stats.get("elapsed", 0) >= 2, f"耗时日志不足: {elapsed_stats}"
    print("    模块耗时统计 [OK]")

    # ---------- 4. 写入失败不影响主流程 ----------
    probe = _APP_FILE.parent / "_m9_fail_probe.txt"
    probe.write_text("x", encoding="utf-8")
    bad_writer = EventLogWriter(probe / "sub" / "events.jsonl")  # 父路径是普通文件 → 必然失败
    assert bad_writer.write({"event_type": "test"}) is False
    probe.unlink()
    print("    日志写入失败不影响主流程 [OK]")

    # ---------- 5. 桥接 M6 聚合器 ----------
    from aggregator.event_aggregator import EventAggregator

    from utils.common import BoundingBox, Detection

    agg = EventAggregator()
    attach_event_logger(agg)

    def make_det(ts, has_fire=True):
        bboxes = []
        if has_fire:
            bboxes.append(BoundingBox(x1=100, y1=100, x2=200, y2=200,
                                      confidence=0.85, class_id=0, class_name="fire"))
        return Detection(bboxes=bboxes, image_path="sim", timestamp=ts,
                         image_width=640, image_height=480, frame_id=int(ts * 2))

    for i in range(5):
        agg.process(make_det(0.5 + i * 0.5, has_fire=True))
    for i in range(3):
        agg.process(make_det(3.0 + i * 0.5, has_fire=False))
    flush()

    stats = get_event_stats()
    assert stats.get("event_confirmed", 0) >= 1, f"缺少 event_confirmed: {stats}"
    assert stats.get("event_ended", 0) >= 1, f"缺少 event_ended: {stats}"
    print(f"    桥接 M6 回调 [OK] | 统计: {stats}")

    # ---------- 6. run_summary.json ----------
    add_stat("screenshots", 3)
    add_stat("reports", 1)
    summary_path = save_run_summary()
    assert summary_path is not None and summary_path.exists()
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("session_id", "started_at", "ended_at", "duration_seconds",
                "total_events", "event_types", "danger_levels", "runtime_stats"):
        assert key in summary, f"run_summary 缺少字段: {key}"
    assert summary["session_id"] == SESSION_ID
    assert summary["runtime_stats"].get("screenshots") == 3
    assert summary["runtime_stats"].get("reports") == 1
    print(f"    run_summary.json 保存 [OK] | 危险等级分布: {summary['danger_levels']}")

    # ---------- 7. 统计摘要 ----------
    print_summary()
    print("\n  M9 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())