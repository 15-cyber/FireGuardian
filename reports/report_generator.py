"""
============================================================
M10 - 火灾事件分析报告生成模块 (Report Generator)
功能（按检查文档实现）：
  - 输入完整事件包 EventReportData（event + decisions + screenshots + system/model info）
  - 输出三种格式到 reports/event_<event_id>/：
      report.md    人类可读
      report.pdf   正式文档（中文，需要中文字体）
      event.json   机器可读（原子写入）
      images/      截图副本（供 md / pdf 引用）
  - 报告内容：封面 / 事件摘要 / 基本信息 / 检测统计 / 检测趋势图 /
              事件时间线 / Agent 分析 / 图片证据（含说明）/ 配置快照 / 免责声明
  - 完整性检查：Event/Decision/Screenshot/Metadata/Timeline 缺数据只警告，绝不崩溃
  - 中文字体：配置指定或自动检测；不可用时只生成 md + json，并明确提示
  - 事件驱动：attach_event_report 挂接 M6 on_event_ended

可独立运行: python reports/report_generator.py
============================================================
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----- 项目导入 -----
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.common import (
    load_config,
    EventReportData,
    FireDecision,
    ScreenshotRecord,
)
from utils.path_manager import PathManager
from agent.presentation_models import AgentPresentationModel

# 常见中文字体候选（仅检测，不提交字体文件到仓库）
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
]

_DISCLAIMER_DEFAULT = (
    "本系统结果仅用于辅助监测与演示，不替代专业消防判断和企业应急预案。"
    "本报告由 FireGuardian 自动生成，检测结果仅供辅助参考，最终判断应结合现场实际情况。"
)

_REASON_LABELS = {
    "confirmed": "事件确认截图",
    "peak": "峰值截图",
    "danger_level_upgraded": "危险等级升级截图",
    "final": "事件结束截图",
}

_GIT_COMMIT_CACHE: Optional[str] = None


def _git_commit() -> str:
    """获取当前 Git 短提交号（不可用时返回 unknown，不影响报告生成）"""
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is None:
        try:
            import subprocess
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=str(_ROOT),
            )
            _GIT_COMMIT_CACHE = out.stdout.strip() or "unknown"
        except Exception:
            _GIT_COMMIT_CACHE = "unknown"
    return _GIT_COMMIT_CACHE


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _level_str(level) -> str:
    """危险等级枚举 → 字符串（low/medium/high），避免输出 DangerLevel.HIGH"""
    if level is None:
        return ""
    if hasattr(level, "value"):
        return str(level.value)
    return str(level)


# ======================== 报告生成器 ========================

class ReportGenerator:
    """火灾事件分析报告生成器（M10）"""

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or load_config()
        rep = self._cfg.get("report", {})

        self.path_mgr = PathManager(self._cfg)
        self.output_dir = self.path_mgr.reports_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.formats = [str(f).lower() for f in rep.get("formats", ["md", "pdf"])]
        self.copy_images = rep.get("copy_images", True)
        self.disclaimer = rep.get("disclaimer", _DISCLAIMER_DEFAULT)

        # 中文字体解析（改进说明 4：字体不可用时不崩溃，只生成 md + json）
        self.font_path: Optional[Path] = None
        configured = str(rep.get("pdf_font", "") or "")
        if configured:
            if Path(configured).exists():
                self.font_path = Path(configured)
            else:
                print(f"  [Report] 配置的中文字体不存在: {configured}")
        if self.font_path is None:
            for cand in _FONT_CANDIDATES:
                if Path(cand).exists():
                    self.font_path = Path(cand)
                    break

        self.total_reports = 0
        self.generated_paths: List[Path] = []

    @property
    def pdf_available(self) -> bool:
        """是否找到可用中文字体（决定是否生成 PDF）"""
        return self.font_path is not None

    # ======================== 主入口 ========================

    def generate(
        self,
        data: EventReportData,
        agent_presentation: Optional[AgentPresentationModel] = None,
    ) -> List[Path]:
        """生成一份事件报告，返回输出文件路径列表"""
        if data is None or data.event is None:
            return []

        warnings = self._check_completeness(data)
        for w in warnings:
            print(f"  [Report] [Warning] {w}")

        event_dir = self.path_mgr.report_event_dir(data.event.event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        images_dir = event_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        output: List[Path] = []

        summary = self._build_summary(data)
        version_info = self._version_info(data)
        config_snapshot = self._config_snapshot()

        # 1. event.json（机器可读，原子写入）
        json_path = self._write_event_json(event_dir, data, summary, version_info,
                                           config_snapshot, warnings, agent_presentation)
        if json_path:
            output.append(json_path)

        # 2. 复制截图到 images/，生成趋势图
        image_map, rec_map = self._collect_images(data, images_dir)
        trend_rel = self._make_trend_chart(data, images_dir)

        # 3. report.md
        md_path = self._write_markdown(event_dir, data, image_map, rec_map, summary,
                                       version_info, config_snapshot, warnings, trend_rel,
                                       agent_presentation)
        if md_path:
            output.append(md_path)

        # 4. report.pdf（无中文字体时明确提示并跳过）
        if "pdf" in self.formats:
            pdf_path = self._write_pdf(event_dir, data, image_map, rec_map, summary,
                                       version_info, config_snapshot, warnings, trend_rel,
                                       agent_presentation)
            if pdf_path:
                output.append(pdf_path)
            else:
                print("  [Report] PDF 未生成：缺少可用中文字体，已只输出 md + json")

        self.total_reports += 1
        self.generated_paths.extend(output)
        print(f"  [Report] 事件 {data.event.event_id} 报告已生成 ({len(output)} 个文件) -> {event_dir}")
        return output

    # ======================== 完整性检查 / 摘要 / 快照 ========================

    def _check_completeness(self, data: EventReportData) -> List[str]:
        """生成前完整性检查：缺数据只警告，绝不崩溃（检查文档改动七）"""
        warnings: List[str] = []
        if data is None or data.event is None:
            return ["Event 缺失：无法生成报告"]
        evt = data.event
        if not evt.metadata:
            warnings.append("Metadata 缺失：事件元数据为空")
        if not data.decisions:
            warnings.append("Decision 缺失：无 Agent 决策记录")
        if not data.screenshots:
            warnings.append("Screenshot 缺失：无截图证据")
        has_timeline = bool(evt.start_timestamp or evt.end_timestamp
                            or data.screenshots or data.decisions)
        if not has_timeline:
            warnings.append("Timeline 缺失：无时间线数据")
        if not evt.detections and not data.screenshots:
            warnings.append("检测序列缺失：趋势图将跳过")
        if not self.pdf_available:
            warnings.append("字体缺失：无可用中文字体，PDF 将跳过")
        return warnings

    def _build_summary(self, data: EventReportData) -> dict:
        """事件摘要：等级/时长/峰值/最终建议（检查文档改动二）"""
        evt = data.event
        danger = str(evt.metadata.get("danger_level", "N/A")) if evt.metadata else "N/A"
        final_advice = "无决策记录"
        if data.decisions:
            last = data.decisions[-1]
            danger = _level_str(last.danger_level)
            final_advice = (last.suggestions[0] if last.suggestions
                            else (last.summary or "无"))
        return {
            "event_level": danger,
            "duration_seconds": round(evt.duration, 1),
            "fire_peak_ratio": evt.max_fire_area_ratio,
            "smoke_peak_ratio": evt.max_smoke_area_ratio,
            "final_recommendation": final_advice,
        }

    def _config_snapshot(self) -> dict:
        """报告内记录当时模型参数（检查文档改动四）"""
        mdl = self._cfg.get("model", {})
        trn = self._cfg.get("train", {})
        return {
            "model": mdl.get("path", ""),
            "confidence": mdl.get("confidence", 0.5),
            "iou": mdl.get("iou", 0.45),
            "image_size": trn.get("imgsz", 640),
        }

    def _version_info(self, data: EventReportData) -> dict:
        """版本信息：FireGuardian 版本 / Git Commit / 模型版本（检查文档改动五）"""
        proj = self._cfg.get("project", {})
        model = (data.model_info or {}).get("model", "")
        return {
            "fireguardian_version": proj.get("version", "unknown"),
            "git_commit": _git_commit(),
            "model": model,
            "report_version": "1.0",
            "generated_at": _now_str(),
        }

    def _image_caption(self, rec: ScreenshotRecord) -> str:
        """图片证据说明：Fire Area / Smoke / Reason（检查文档改动六）"""
        parts = []
        if rec.fire_area_ratio is not None:
            parts.append(f"Fire Area: {rec.fire_area_ratio:.1%}")
        if rec.smoke_area_ratio is not None:
            parts.append(f"Smoke: {rec.smoke_area_ratio:.1%}")
        reason_txt = _level_str(rec.danger_level) if rec.danger_level else (rec.reason or "")
        parts.append(f"Reason: {reason_txt}")
        return " | ".join(parts)

    # ======================== 图片证据 / 趋势图 ========================

    def _collect_images(self, data: EventReportData,
                        images_dir: Path) -> Tuple[Dict[str, str], Dict[str, ScreenshotRecord]]:
        """把截图复制到报告 images/，返回 reason->相对路径 与 reason->截图记录"""
        image_map: Dict[str, str] = {}
        rec_map: Dict[str, ScreenshotRecord] = {}
        copied: set = set()
        for rec in data.screenshots:
            candidates = []
            if rec.annotated_image_path:
                candidates.append(rec.annotated_image_path)
            if rec.raw_image_path:
                candidates.append(rec.raw_image_path)
            for raw in candidates:
                src = Path(raw)
                if not src.exists() or str(src) in copied:
                    continue
                target = images_dir / src.name
                if not target.exists():
                    try:
                        import shutil
                        shutil.copy2(str(src), str(target))
                    except Exception as exc:
                        print(f"  [Report] 图片复制失败（跳过）: {exc}")
                        continue
                copied.add(str(src))
            chosen = None
            for raw in candidates:
                src = Path(raw)
                if src.exists() and str(src) in copied:
                    chosen = src
                    break
            if chosen is not None and rec.reason not in image_map:
                image_map[rec.reason] = f"images/{chosen.name}"
                rec_map[rec.reason] = rec
        return image_map, rec_map

    def _make_trend_chart(self, data: EventReportData, images_dir: Path) -> Optional[str]:
        """生成检测趋势图 images/trend.png，返回相对路径；数据不足或失败返回 None"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
        except Exception:
            print("  [Report] [Warning] matplotlib 不可用，趋势图跳过")
            return None

        # 火焰/烟雾面积比序列（优先 M6 检测序列，其次截图记录）
        series = []
        for det in sorted(data.event.detections, key=lambda d: d.timestamp):
            series.append((det.timestamp, det.fire_area_ratio, det.smoke_area_ratio))
        if not series:
            for rec in sorted(data.screenshots, key=lambda r: r.simulated_timestamp):
                series.append((rec.simulated_timestamp, rec.fire_area_ratio, rec.smoke_area_ratio))
        if not series:
            return None

        # 危险等级序列（来自升级截图）
        danger_map = {"low": 0, "medium": 1, "high": 2}
        dseries = []
        for rec in sorted(data.screenshots, key=lambda r: r.simulated_timestamp):
            lv = _level_str(rec.danger_level) if rec.danger_level else ""
            if lv in danger_map:
                dseries.append((rec.simulated_timestamp, danger_map[lv]))

        # 中文字体（与 PDF 同源检测）
        zh = False
        if self.font_path and self.font_path.exists():
            try:
                font_manager.fontManager.addfont(str(self.font_path))
                fam = font_manager.FontProperties(fname=str(self.font_path)).get_name()
                plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                zh = True
            except Exception:
                zh = False

        try:
            times = [p[0] for p in series]
            fire = [p[1] for p in series]
            smoke = [p[2] for p in series]
            fig, ax1 = plt.subplots(figsize=(8, 4.2), dpi=120)
            ax1.plot(times, fire, color="#d62728", linewidth=1.8,
                     label="火焰面积" if zh else "Fire Area")
            ax1.plot(times, smoke, color="#ff7f0e", linewidth=1.8,
                     label="烟雾面积" if zh else "Smoke Area")
            ax1.set_xlabel("时间 (s)" if zh else "Time (s)")
            ax1.set_ylabel("面积比例" if zh else "Area Ratio")
            ax1.set_ylim(bottom=0)
            ax1.grid(True, linestyle="--", alpha=0.4)
            ax1.legend(loc="upper left")

            if dseries:
                ax2 = ax1.twinx()
                ax2.step([p[0] for p in dseries], [p[1] for p in dseries],
                         where="post", color="#1f77b4", linewidth=1.6,
                         label="危险等级" if zh else "Danger Level")
                ax2.set_ylim(-0.5, 2.5)
                ax2.set_yticks([0, 1, 2])
                ax2.set_yticklabels(["低", "中", "高"] if zh else ["low", "medium", "high"])
                ax2.legend(loc="upper right")

            fig.suptitle("检测趋势" if zh else "Detection Trend", fontsize=13)
            fig.tight_layout()
            target = images_dir / "trend.png"
            fig.savefig(str(target), bbox_inches="tight")
            plt.close(fig)
            return "images/trend.png"
        except Exception as exc:
            print(f"  [Report] [Warning] 趋势图生成失败（跳过）: {exc}")
            return None
    # ======================== 事件时间线 ========================

    def _upgrade_levels(self, rec: ScreenshotRecord):
        """从升级截图元数据读取 previous/current 危险等级"""
        try:
            meta = json.loads(Path(rec.metadata_path).read_text(encoding="utf-8"))
            return (meta.get("previous_danger_level", "?"),
                    meta.get("current_danger_level", "?"))
        except Exception:
            return ("?", rec.danger_level or "?")

    def _decision_to_dict(self, d: FireDecision) -> dict:
        """决策转字典（危险等级枚举归一化为字符串）"""
        payload = d.to_dict()
        payload["danger_level"] = _level_str(d.danger_level)
        return payload

    def _build_timeline(self, data: EventReportData) -> List[str]:
        """构建事件时间线（改进说明：确认/升级/峰值/结束）"""
        evt = data.event
        entries: List[tuple] = []
        if evt.start_timestamp:
            entries.append((evt.start_timestamp, "事件开始（首次检测到目标）"))
        for rec in sorted(data.screenshots, key=lambda r: r.simulated_timestamp):
            t = rec.simulated_timestamp
            if rec.reason == "confirmed":
                entries.append((t, "事件确认"))
            elif rec.reason == "peak":
                peak = max(rec.fire_area_ratio, rec.smoke_area_ratio)
                entries.append((t, f"面积达到峰值 {peak:.1%}"))
            elif rec.reason == "danger_level_upgraded":
                prev_lv, cur_lv = self._upgrade_levels(rec)
                entries.append((t, f"危险等级 {prev_lv} → {cur_lv}"))
            elif rec.reason == "final":
                entries.append((t, "事件结束（最后阳性帧）"))
        if evt.end_timestamp and evt.end_timestamp > 0:
            entries.append((evt.end_timestamp, "事件结束"))

        lines = [f"- {t:.1f}s：{desc}" for t, desc in sorted(entries, key=lambda x: x[0])]
        for d in data.decisions:
            when = d.generated_at or ""
            lines.append(f"- {when}：Agent 判定 {_level_str(d.danger_level)} —— {d.summary or '无摘要'}")
        return lines or ["- （无时间线数据）"]
    def _agent_analysis_markdown(self, p: AgentPresentationModel) -> List[str]:
        """V2 Agent Analysis 的 Markdown 区块（展示层只读格式化）。"""
        lines = [
            f"| 项目 | 内容 |",
            f"|------|------|",
            f"| Agent Status | {p.status_label} |",
            f"| Rule Level | {p.rule_level.upper()} |",
            f"| Rule Score | {p.rule_score} |",
            f"| Rule Confidence | {p.rule_confidence} |",
            f"| Agent Assessment | {p.agent_assessment or '-'} |",
            f"| Agent Risk | {(p.agent_risk_level or '-').upper()} |",
            f"| Agreement With Rule | {'YES' if p.agreement else 'NO'} |",
            f"| Memory | {'Used (' + str(p.similar_event_count) + ')' if p.memory_used else 'Not Used'} |",
            f"| Final Decision | {p.final_level.upper()} |",
            f"| Override | {'YES' if p.override else 'NO'} |",
            f"| Requires Review | {'YES' if p.requires_review else 'NO'} |",
            f"| Latency | {p.latency_ms:.1f}ms |",
            f"| Tokens | {p.token_usage.get('total_tokens', 0)} |",
        ]
        if p.agent_status == "unavailable":
            lines += ["", "**说明：DeepSeek 不可用，最终决策使用 M7 规则结果。**"]
        elif p.agent_status == "invalid_output":
            lines += ["", "**说明：Agent 输出未通过 Schema 校验，系统自动回退 M7。**"]
        lines += ["", "**Rule Reasons:**", ""]
        for r in p.rule_reasons:
            lines.append(f"- {r}")
        if p.memory_used:
            lines += ["", "**Memory Evidence:**", "", p.memory_summary or "（无摘要）"]
            if p.memory_results:
                lines += ["", "相似事件："]
                for m in p.memory_results:
                    lines.append(f"- {m['event_id']}  {m['similarity_score']:.2f}")
        else:
            lines += ["", "**Memory: Not Used**"]
        lines += [
            "",
            "**Agent Explanation:**",
            "",
            p.reasoning_summary or "（无推理摘要）",
            "",
            "**Possible Cause:**",
            "",
            p.possible_cause or "暂无明确判断",
            "",
            "**Recommended Action:**",
            "",
            p.recommended_action or "未给出处置建议，请按当前风险等级执行标准流程",
            "",
            "**Uncertainty:**",
            "",
            p.uncertainty,
            "",
            "**Final Decision (Hybrid):**",
            "",
            f"Rule: {p.rule_level.upper()} | Agent: {(p.agent_risk_level or '-').upper()} | Final: {p.final_level.upper()}",
            "",
            p.reason or "",
            "",
            "**Tools Used:**",
            "",
        ]
        if p.tools_used:
            for t in p.tools_used:
                lines.append(f"- {t}")
        else:
            lines.append("- None")
        return lines

    # ======================== 输出生成 ========================

    def _write_event_json(self, event_dir: Path, data: EventReportData, summary: dict,
                          version_info: dict, config_snapshot: dict,
                          warnings: List[str],
                          agent_presentation: Optional[AgentPresentationModel] = None) -> Optional[Path]:
        """event.json：机器可读完整事件包（原子写入）"""
        evt = data.event
        payload = {
            "event": evt.to_dict(),
            "metadata": evt.metadata,
            "decisions": [self._decision_to_dict(d) for d in data.decisions],
            "screenshots": [r.to_dict() for r in data.screenshots],
            "agent_analysis": agent_presentation.to_dict() if agent_presentation else None,
            "system_info": data.system_info,
            "model_info": data.model_info,
            "executive_summary": summary,
            "config_snapshot": config_snapshot,
            "version_info": version_info,
            "warnings": warnings,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "disclaimer": self.disclaimer,
        }
        target = event_dir / "event.json"
        return target if self._atomic_write_json(target, payload) else None

    def _write_markdown(self, event_dir: Path, data: EventReportData,
                        image_map: Dict[str, str], rec_map: Dict[str, ScreenshotRecord],
                        summary: dict, version_info: dict, config_snapshot: dict,
                        warnings: List[str], trend_rel: Optional[str],
                        agent_presentation: Optional[AgentPresentationModel] = None) -> Optional[Path]:
        """report.md：人类可读报告"""
        evt = data.event
        status = evt.status.value if hasattr(evt.status, "value") else str(evt.status)
        trend = evt.growth_trend.value if hasattr(evt.growth_trend, "value") else str(evt.growth_trend)
        end_ts = f"{evt.end_timestamp:.1f}s" if evt.end_timestamp else "进行中"

        lines = [
            "# 火灾事件分析报告",
            "",
            f"> 事件编号：`{evt.event_id}`",
            f"> 生成时间：{version_info['generated_at']}",
            f"> FireGuardian {version_info['fireguardian_version']} | Git {version_info['git_commit']} | "
            f"模型 {version_info['model'] or 'N/A'} | 报告版本 {version_info['report_version']}",
            "",
        ]
        if warnings:
            lines += [
                "> **报告完整性说明（警告）：**",
            ]
            for w in warnings:
                lines.append(f"> - {w}")
            lines.append("")

        lines += [
            "## 1. 事件摘要",
            "",
            "| 项目 | 内容 |",
            "|------|------|",
            f"| 事件等级 | {summary['event_level']} |",
            f"| 持续时长 | {summary['duration_seconds']:.1f}s |",
            f"| 火焰峰值 | {summary['fire_peak_ratio']:.1%} |",
            f"| 烟雾峰值 | {summary['smoke_peak_ratio']:.1%} |",
            f"| 最终建议 | {summary['final_recommendation']} |",
            "",
            "## 2. 基本信息",
            "",
            "| 项目 | 内容 |",
            "|------|------|",
            f"| 事件编号 | {evt.event_id} |",
            f"| 检测来源 | {evt.source_type} / {evt.source_name} |",
            f"| 状态 | {status} |",
            f"| 开始时间 | {evt.start_timestamp:.1f}s |",
            f"| 结束时间 | {end_ts} |",
            f"| 持续时长 | {evt.duration:.1f}s |",
            f"| 总处理帧数 | {evt.total_frames} |",
            f"| 阳性帧数 | {evt.positive_frames} |",
            "",
            "## 3. 检测统计",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 火焰最大面积比例 | {evt.max_fire_area_ratio:.1%} |",
            f"| 烟雾最大面积比例 | {evt.max_smoke_area_ratio:.1%} |",
            f"| 火焰平均置信度 | {evt.avg_fire_confidence:.3f} |",
            f"| 烟雾平均置信度 | {evt.avg_smoke_confidence:.3f} |",
            f"| 增长趋势 | {trend} |",
            f"| 最高危险等级 | {summary['event_level']} |",
            "",
            "## 4. 检测趋势",
            "",
        ]
        if trend_rel:
            lines += [
                f"![检测趋势]({trend_rel})",
                "",
                "> 火焰面积 / 烟雾面积随时间变化曲线；蓝色阶梯线为危险等级（低/中/高）。",
                "",
            ]
        else:
            lines += ["（无检测序列，趋势图跳过）", ""]

        lines += [
            "## 5. 事件时间线",
            "",
            *self._build_timeline(data),
            "",
            "## 6. Agent 分析",
            "",
        ]
        if agent_presentation is None:
            lines += ["**Agent Analysis: Not available for this event.**", ""]
            lines += ["**Rule Decision (M7):**", ""]
            if not data.decisions:
                lines.append("（无决策记录）")
            else:
                d = data.decisions[-1]
                lines.append(f"- 结论摘要：{d.summary or '无'}")
                lines.append(f"- 危险等级：{_level_str(d.danger_level)}")
                lines.append(f"- 评分：{d.score} | 可信度：{d.confidence}")
                lines.append(f"- 决策来源：{d.decision_source}")
                lines.append("- 判断原因：")
                for r in d.reasons:
                    lines.append(f"  - {r}")
                lines.append("- 应急建议：")
                for s in d.suggestions:
                    lines.append(f"  - {s}")
        else:
            lines += self._agent_analysis_markdown(agent_presentation)
        lines += [
            "",
            "## 7. 图片证据",
            "",
        ]
        if image_map:
            for reason in ("confirmed", "peak", "danger_level_upgraded", "final"):
                rel = image_map.get(reason)
                rec = rec_map.get(reason)
                if rel:
                    label = _REASON_LABELS.get(reason, reason)
                    lines += [
                        f"### {label}",
                        "",
                        f"![{label}]({rel})",
                        "",
                    ]
                    if rec is not None:
                        lines += [
                            f"> {self._image_caption(rec)}",
                            "",
                        ]
        else:
            lines.append("（无截图证据）")
            lines.append("")
        lines += [
            "## 8. 配置快照",
            "",
            "| 参数 | 值 |",
            "|------|------|",
            f"| 模型 | {config_snapshot['model']} |",
            f"| conf | {config_snapshot['confidence']} |",
            f"| iou | {config_snapshot['iou']} |",
            f"| image_size | {config_snapshot['image_size']} |",
            "",
            "## 9. 免责声明",
            "",
            f"> {self.disclaimer}",
            "",
        ]

        target = event_dir / "report.md"
        try:
            target.write_text("\n".join(lines), encoding="utf-8")
            return target
        except Exception as exc:
            print(f"  [Report] report.md 写入失败: {exc}")
            return None
    def _write_pdf(self, event_dir: Path, data: EventReportData,
                   image_map: Dict[str, str], rec_map: Dict[str, ScreenshotRecord],
                   summary: dict, version_info: dict, config_snapshot: dict,
                   warnings: List[str], trend_rel: Optional[str],
                   agent_presentation: Optional[AgentPresentationModel] = None) -> Optional[Path]:
        """report.pdf：正式文档（封面/摘要/趋势图/配置快照；任何失败都不崩溃，降级为 md+json）"""
        if not self.pdf_available:
            return None
        try:
            from io import BytesIO
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                            Spacer, Table, TableStyle)

            font_path = str(self.font_path)
            if font_path.lower().endswith(".ttc"):
                pdfmetrics.registerFont(TTFont("FireFont", font_path, subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont("FireFont", font_path))
            base = "FireFont"

            cover_title = ParagraphStyle("cover_title", fontName=base, fontSize=30,
                                         leading=38, alignment=1, spaceAfter=6)
            cover_sub = ParagraphStyle("cover_sub", fontName=base, fontSize=16,
                                       leading=22, alignment=1, textColor=colors.grey)
            h1 = ParagraphStyle("h1", fontName=base, fontSize=18, leading=24,
                                alignment=1, spaceAfter=12)
            h2 = ParagraphStyle("h2", fontName=base, fontSize=14, leading=18,
                                spaceBefore=10, spaceAfter=6)
            body = ParagraphStyle("body", fontName=base, fontSize=10.5,
                                  leading=15, spaceAfter=4)
            caption = ParagraphStyle("caption", fontName=base, fontSize=9,
                                     leading=12, spaceAfter=8, textColor=colors.darkgrey)

            evt = data.event

            def make_table(rows):
                t = Table(rows, colWidths=[60 * mm, 110 * mm])
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), base),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]))
                return t

            story = []
            # ---- 封面（检查文档改动一） ----
            story.append(Spacer(1, 50 * mm))
            story.append(Paragraph("FireGuardian", cover_title))
            story.append(Paragraph("Fire Event Analysis Report", cover_sub))
            story.append(Spacer(1, 20 * mm))
            story.append(make_table([
                ["事件编号", evt.event_id],
                ["生成时间", version_info["generated_at"]],
                ["模型", version_info["model"] or "N/A"],
                ["FireGuardian Version", version_info["fireguardian_version"]],
                ["Git Commit", version_info["git_commit"]],
                ["报告版本", version_info["report_version"]],
            ]))
            story.append(PageBreak())

            # ---- 1. 事件摘要（检查文档改动二） ----
            story.append(Paragraph("1. 事件摘要", h2))
            story.append(make_table([
                ["事件等级", summary["event_level"]],
                ["持续时长", f"{summary['duration_seconds']:.1f}s"],
                ["火焰峰值", f"{summary['fire_peak_ratio']:.1%}"],
                ["烟雾峰值", f"{summary['smoke_peak_ratio']:.1%}"],
                ["最终建议", summary["final_recommendation"]],
            ]))

            # ---- 报告完整性警告（检查文档改动七） ----
            if warnings:
                story.append(Paragraph("报告完整性说明（警告）", h2))
                for w in warnings:
                    story.append(Paragraph(f"· {w}", body))

            # ---- 2. 基本信息 ----
            story.append(Paragraph("2. 基本信息", h2))
            status = evt.status.value if hasattr(evt.status, "value") else str(evt.status)
            end_ts = f"{evt.end_timestamp:.1f}s" if evt.end_timestamp else "进行中"
            story.append(make_table([
                ["事件编号", evt.event_id],
                ["检测来源", f"{evt.source_type} / {evt.source_name}"],
                ["状态", status],
                ["开始时间", f"{evt.start_timestamp:.1f}s"],
                ["结束时间", end_ts],
                ["持续时长", f"{evt.duration:.1f}s"],
                ["总处理帧数", str(evt.total_frames)],
                ["阳性帧数", str(evt.positive_frames)],
            ]))

            # ---- 3. 检测统计 ----
            story.append(Paragraph("3. 检测统计", h2))
            trend = evt.growth_trend.value if hasattr(evt.growth_trend, "value") else str(evt.growth_trend)
            story.append(make_table([
                ["火焰最大面积比例", f"{evt.max_fire_area_ratio:.1%}"],
                ["烟雾最大面积比例", f"{evt.max_smoke_area_ratio:.1%}"],
                ["火焰平均置信度", f"{evt.avg_fire_confidence:.3f}"],
                ["烟雾平均置信度", f"{evt.avg_smoke_confidence:.3f}"],
                ["增长趋势", trend],
                ["最高危险等级", summary["event_level"]],
            ]))

            # ---- 4. 检测趋势（检查文档改动三） ----
            story.append(Paragraph("4. 检测趋势", h2))
            if trend_rel:
                src = event_dir / trend_rel
                if src.exists():
                    from PIL import Image as PILImage
                    with PILImage.open(str(src)) as pil_img:
                        w, h = pil_img.size
                    max_w = 170 * mm
                    scale = min(max_w / w, 1.0)
                    story.append(Image(str(src), width=w * scale, height=h * scale))
                    story.append(Paragraph(
                        "火焰面积 / 烟雾面积随时间变化曲线；蓝色阶梯线为危险等级（低/中/高）。",
                        caption))
            else:
                story.append(Paragraph("（无检测序列，趋势图跳过）", body))

            # ---- 5. 事件时间线 ----
            story.append(Paragraph("5. 事件时间线", h2))
            for line in self._build_timeline(data):
                story.append(Paragraph(line, body))

            # ---- 6. Agent 分析 ----
            story.append(Paragraph("6. Agent 分析", h2))
            if agent_presentation is None:
                story.append(Paragraph("Agent Analysis: Not available for this event.", body))
                if not data.decisions:
                    story.append(Paragraph("（无决策记录）", body))
                else:
                    d = data.decisions[-1]
                    story.append(Paragraph(f"结论摘要：{d.summary or '无'}", body))
                    story.append(Paragraph(
                        f"危险等级：{_level_str(d.danger_level)} | 评分：{d.score} | 可信度：{d.confidence} | "
                        f"决策来源：{d.decision_source}", body))
                    story.append(Paragraph("判断原因：", body))
                    for r in d.reasons:
                        story.append(Paragraph(f"· {r}", body))
                    story.append(Paragraph("应急建议：", body))
                    for s in d.suggestions:
                        story.append(Paragraph(f"· {s}", body))
            else:
                p = agent_presentation
                story.append(Paragraph(f"Agent Status: {p.status_label}", body))
                if p.agent_status == "unavailable":
                    story.append(Paragraph("说明：DeepSeek 不可用，最终决策使用 M7 规则结果。", body))
                elif p.agent_status == "invalid_output":
                    story.append(Paragraph("说明：Agent 输出未通过 Schema 校验，系统自动回退 M7。", body))
                story.append(Paragraph(
                    f"Rule Decision: {p.rule_level.upper()} | Score: {p.rule_score} | "
                    f"Confidence: {p.rule_confidence}", body))
                story.append(Paragraph("Rule Reasons:", body))
                for r in p.rule_reasons:
                    story.append(Paragraph(f"· {r}", body))
                story.append(Paragraph(
                    f"Agent Assessment: {p.agent_assessment or '-'} | Agent Risk: "
                    f"{(p.agent_risk_level or '-').upper()} | Agreement: "
                    f"{'YES' if p.agreement else 'NO'}", body))
                if p.memory_used:
                    story.append(Paragraph(f"Memory: Used ({p.similar_event_count} similar events)", body))
                    story.append(Paragraph(p.memory_summary or "（无摘要）", body))
                    for m in p.memory_results[:5]:
                        story.append(Paragraph(f"· {m['event_id']}  {m['similarity_score']:.2f}", body))
                else:
                    story.append(Paragraph("Memory: Not Used", body))
                story.append(Paragraph("Agent Explanation:", body))
                story.append(Paragraph(p.reasoning_summary or "（无推理摘要）", body))
                story.append(Paragraph(f"Possible Cause: {p.possible_cause}", body))
                story.append(Paragraph(f"Recommended Action: {p.recommended_action}", body))
                story.append(Paragraph(f"Uncertainty: {p.uncertainty}", body))
                story.append(Paragraph(
                    f"Final Decision: {p.final_level.upper()} "
                    f"(Rule {p.rule_level.upper()} / Agent {(p.agent_risk_level or '-').upper()})", body))
                story.append(Paragraph(p.reason or "", body))
                tools_text = "、".join(p.tools_used) if p.tools_used else "None"
                story.append(Paragraph(f"Tools Used: {tools_text}", body))
                story.append(Paragraph(
                    f"Latency: {p.latency_ms:.1f}ms | Tokens: {p.token_usage.get('total_tokens', 0)}",
                    caption))

            # ---- 7. 图片证据（检查文档改动六：图片说明） ----
            if image_map:
                story.append(Paragraph("7. 图片证据", h2))
                for reason in ("confirmed", "peak", "danger_level_upgraded", "final"):
                    rel = image_map.get(reason)
                    rec = rec_map.get(reason)
                    if not rel:
                        continue
                    src = event_dir / rel
                    if not src.exists():
                        continue
                    data_bytes = src.read_bytes()
                    from PIL import Image as PILImage
                    with BytesIO(data_bytes) as bio:
                        pil_img = PILImage.open(bio)
                        w, h = pil_img.size
                    max_w = 140 * mm
                    scale = min(max_w / w, 1.0)
                    label = _REASON_LABELS.get(reason, reason)
                    story.append(Paragraph(label, h2))
                    story.append(Image(BytesIO(data_bytes), width=w * scale, height=h * scale))
                    if rec is not None:
                        story.append(Paragraph(self._image_caption(rec), caption))

            # ---- 8. 配置快照（检查文档改动四） ----
            story.append(Paragraph("8. 配置快照", h2))
            story.append(make_table([
                ["模型", config_snapshot["model"]],
                ["conf", str(config_snapshot["confidence"])],
                ["iou", str(config_snapshot["iou"])],
                ["image_size", str(config_snapshot["image_size"])],
            ]))

            # ---- 9. 免责声明（检查文档改动八） ----
            story.append(Paragraph("9. 免责声明", h2))
            story.append(Paragraph(self.disclaimer, body))

            target = event_dir / "report.pdf"
            with open(str(target), "wb") as f:
                doc = SimpleDocTemplate(f, pagesize=A4,
                                        title=f"火灾事件分析报告 - {evt.event_id}")
                doc.build(story)
            return target
        except Exception as exc:
            print(f"  [Report] report.pdf 生成失败（不影响 md/json）: {exc}")
            return None
    @staticmethod
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

    # ======================== 查询接口 ========================

    def get_generated_paths(self) -> List[Path]:
        return list(self.generated_paths)

    def get_stats(self) -> dict:
        return {
            "total_reports": self.total_reports,
            "output_dir": str(self.output_dir),
            "pdf_available": self.pdf_available,
            "font_path": str(self.font_path) if self.font_path else "",
        }

# ======================== 事件驱动接线 ========================

def attach_event_report(aggregator, generator: ReportGenerator,
                        screenshot_provider=None, decision_provider=None):
    """
    事件结束（M6 on_event_ended）时自动生成报告；链式挂接，不覆盖已有回调。

    参数:
        aggregator: M6 EventAggregator
        generator: ReportGenerator 实例
        screenshot_provider: callable(event) -> List[ScreenshotRecord]（可选）
        decision_provider: callable(event) -> List[FireDecision]（可选）
    """
    prev_cb = getattr(aggregator, "on_event_ended", None)

    def handler(event):
        shots = screenshot_provider(event) if screenshot_provider else []
        decisions = decision_provider(event) if decision_provider else []
        data = EventReportData(event=event, decisions=decisions, screenshots=shots)
        generator.generate(data)
        if prev_cb is not None:
            prev_cb(event)

    aggregator.on_event_ended = handler
    return aggregator


# ======================== 命令行测试 ========================

def main():
    print("\n  FireGuardian M10 - 报告生成模块测试")
    print("  ===============================================\n")

    from utils.common import (BoundingBox, Detection, FireEvent, EventStatus,
                              GrowthTrend, DangerLevel, FireDecision, EventReportData)

    test_img = _ROOT / "bus.jpg"
    if not test_img.exists():
        print("  未找到测试图片 bus.jpg")
        return 1
    img_w, img_h = 810, 1080

    def make_det(ratio, ts, fid, conf=0.85):
        side = int((ratio * img_w * img_h) ** 0.5)
        box = BoundingBox(x1=100, y1=100, x2=100 + side, y2=100 + side,
                          confidence=conf, class_id=0, class_name="fire")
        return Detection(bboxes=[box], image_path=str(test_img), timestamp=ts,
                         image_width=img_w, image_height=img_h, frame_id=fid)

    # ---------- 1. 端到端：M8 截图 + M7 决策 → M10 报告 ----------
    print("  场景 1: 完整事件包生成报告（md/pdf/json/images）\n")
    from utils.screenshot import ScreenshotManager

    mgr = ScreenshotManager()
    evt = FireEvent(
        event_id="FE-RPT-0001",
        status=EventStatus.CONFIRMED,
        start_timestamp=0.0,
        last_timestamp=8.0,
        total_frames=20,
        positive_frames=16,
        consecutive_positive_frames=16,
        max_fire_area_ratio=0.15,
        max_smoke_area_ratio=0.06,
        avg_fire_confidence=0.86,
        avg_smoke_confidence=0.72,
        growth_trend=GrowthTrend.INCREASING,
        source_type="validation_simulator",
        source_name="valid",
        detections=[make_det(0.05, 1.0, 1), make_det(0.15, 5.0, 5)],
        metadata={"danger_level": "low"},
    )
    mgr.on_event_confirmed(evt)
    evt.detections.append(make_det(0.15, 5.0, 5))
    mgr.on_event_updated(evt)                       # peak
    mgr.on_decision(evt, danger_level="medium")     # low → medium
    mgr.on_decision(evt, danger_level="high")       # medium → high
    evt.status = EventStatus.ENDED
    evt.end_timestamp = 8.0
    mgr.on_event_ended(evt)
    records = mgr.get_event_shot_records("FE-RPT-0001")
    assert len(records) >= 4, f"截图记录不足: {len(records)}"

    d1 = FireDecision(event_id="FE-RPT-0001", danger_level=DangerLevel.MEDIUM,
                      score=0.42, confidence=0.75, reasons=["火焰面积中等"],
                      summary="判定为中等风险：火焰面积中等",
                      suggestions=["安排人员现场确认"], decision_source="rule_engine",
                      generated_at="2026-07-31 14:30:00")
    d2 = FireDecision(event_id="FE-RPT-0001", danger_level=DangerLevel.HIGH,
                      score=0.58, confidence=0.82, reasons=["火焰面积较大", "趋势扩大"],
                      summary="判定为高风险：火焰面积较大且趋势扩大",
                      suggestions=["立即启动应急预案", "必要时联系119"],
                      decision_source="rule_engine",
                      generated_at="2026-07-31 14:31:00")

    gen = ReportGenerator()
    data = EventReportData(
        event=evt, decisions=[d1, d2], screenshots=records,
        system_info={"os": "Windows", "python": "3.11"},
        model_info={"model": "yolo11n.pt", "task": "detect"},
    )
    paths = gen.generate(data)
    assert len(paths) >= 2, f"至少应有 md+json: {paths}"

    event_dir = gen.path_mgr.report_event_dir("FE-RPT-0001")
    assert (event_dir / "report.md").exists()
    assert (event_dir / "event.json").exists()
    assert (event_dir / "images").is_dir()

    md = (event_dir / "report.md").read_text(encoding="utf-8")
    for must in ("事件摘要", "基本信息", "检测统计", "检测趋势", "事件时间线",
                 "Agent 分析", "图片证据", "配置快照", "免责声明"):
        assert must in md, f"report.md 缺少: {must}"
    assert "FireGuardian" in md and "Git" in md      # 版本信息
    assert "images/trend.png" in md                  # 趋势图引用
    assert "Fire Area:" in md                        # 图片说明
    assert gen.disclaimer in md
    assert "危险等级" in md          # 时间线含升级
    assert "面积达到峰值" in md      # 时间线含峰值
    assert (event_dir / "images" / "trend.png").exists(), "应有趋势图"

    import json as _json
    payload = _json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    for key in ("event", "decisions", "screenshots", "system_info", "model_info",
                "executive_summary", "config_snapshot", "version_info", "warnings",
                "generated_at", "disclaimer"):
        assert key in payload, f"event.json 缺少: {key}"
    assert len(payload["decisions"]) == 2
    assert len(payload["screenshots"]) >= 4
    assert payload["warnings"] == []
    assert payload["executive_summary"]["event_level"] == "high"

    img_files = list((event_dir / "images").iterdir())
    assert len(img_files) >= 1, "images/ 应包含截图副本"
    print(f"    report.md / event.json / images [{len(img_files)} 张] [OK]")

    if gen.pdf_available:
        assert (event_dir / "report.pdf").exists(), "应有 report.pdf"
        print(f"    report.pdf 已生成（字体: {gen.font_path.name}）[OK]")
    else:
        print("    中文字体不可用，按改进说明仅生成 md+json（PDF 跳过）[OK]")

    # ---------- 2. 事件驱动：M6 结束自动生成 ----------
    print("\n  场景 2: attach_event_report 事件驱动")
    from aggregator.event_aggregator import EventAggregator

    agg = EventAggregator()
    gen2 = ReportGenerator()

    def shot_provider(event):
        return [r for r in records if r.event_id == event.event_id]

    attach_event_report(agg, gen2, screenshot_provider=shot_provider)
    for i in range(5):
        agg.process(make_det(0.05 + i * 0.02, 0.5 + i * 0.5, i + 1))
    for i in range(3):
        agg.process(Detection(bboxes=[], image_path=str(test_img),
                              timestamp=3.0 + i * 0.5, image_width=img_w,
                              image_height=img_h, frame_id=10 + i))
    ended = agg.event_history
    assert len(ended) == 1
    evt2 = ended[0]
    assert (gen2.path_mgr.report_event_dir(evt2.event_id) / "report.md").exists()
    assert (gen2.path_mgr.report_event_dir(evt2.event_id) / "event.json").exists()
    print(f"    事件 {evt2.event_id} 结束后自动生成报告 [OK]")

    # ---------- 3. 数据不完整：缺截图/缺决策/缺元数据不崩溃 ----------
    print("\n  场景 3: 数据不完整（缺截图/决策/元数据）仍生成")
    evt3 = FireEvent(
        event_id="FE-RPT-0003",
        status=EventStatus.ENDED,
        start_timestamp=0.0,
        end_timestamp=3.0,
        total_frames=10,
        positive_frames=6,
        source_type="unit_test",
        source_name="incomplete",
    )
    gen3 = ReportGenerator()
    paths3 = gen3.generate(EventReportData(event=evt3))
    assert len(paths3) >= 2, f"不完整数据仍应生成 md+json: {paths3}"
    event_dir3 = gen3.path_mgr.report_event_dir("FE-RPT-0003")
    payload3 = _json.loads((event_dir3 / "event.json").read_text(encoding="utf-8"))
    assert payload3["warnings"], "不完整数据应产生警告"
    md3 = (event_dir3 / "report.md").read_text(encoding="utf-8")
    assert "缺失" in md3, "report.md 应包含缺失警告"
    assert not (event_dir3 / "images" / "trend.png").exists(), "无检测序列不应生成趋势图"
    print(f"    警告 {len(payload3['warnings'])} 条，报告仍生成 [OK]")

    print("\n  M10 测试完成 [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
