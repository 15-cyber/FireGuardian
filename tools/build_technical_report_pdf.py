"""
============================================================
FireGuardian V2 研发技术总结报告 - Markdown -> PDF 构建器

解析 docs/FireGuardian_V2_研发技术总结报告.md 并生成同内容 PDF。
支持：标题层级、段落、粗体、表格、代码块、列表、引用、图片、
静态目录（章节列表）。

用法：
  python tools/build_technical_report_pdf.py
输出：
  docs/FireGuardian_V2_研发技术总结报告.pdf
============================================================
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = _ROOT / "docs" / "FireGuardian_V2_研发技术总结报告.md"
PDF_PATH = _ROOT / "docs" / "FireGuardian_V2_研发技术总结报告.pdf"

FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\msyh.ttc", 0),
    (r"C:\Windows\Fonts\msyh.ttf", None),
    (r"C:\Windows\Fonts\simhei.ttf", None),
    (r"C:\Windows\Fonts\simsun.ttc", 0),
]


def _find_font():
    for path, sub in FONT_CANDIDATES:
        if Path(path).exists():
            return path, sub
    return None, None


def _render_inline(text: str) -> str:
    """粗体/行内代码 -> reportlab mini-markup。"""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='ReportFont'>\1</font>", text)
    return text


def main() -> int:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_path, subfont = _find_font()
    if font_path is None:
        print("[错误] 未找到可用中文字体")
        return 1
    if subfont is not None:
        pdfmetrics.registerFont(TTFont("ReportFont", font_path, subfontIndex=subfont))
    else:
        pdfmetrics.registerFont(TTFont("ReportFont", font_path))
    pdfmetrics.registerFontFamily("ReportFont", normal="ReportFont", bold="ReportFont",
                                  italic="ReportFont", boldItalic="ReportFont")

    accent = colors.HexColor("#1a5276")
    body = ParagraphStyle("body", fontName="ReportFont", fontSize=10.5, leading=16,
                          spaceAfter=5, wordWrap="CJK")
    h1 = ParagraphStyle("h1", fontName="ReportFont", fontSize=17, leading=22,
                        textColor=accent, spaceBefore=14, spaceAfter=8,
                        borderPadding=2, backColor=colors.HexColor("#eaf2f8"))
    h2 = ParagraphStyle("h2", fontName="ReportFont", fontSize=14, leading=19,
                        textColor=colors.HexColor("#1f618d"), spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("h3", fontName="ReportFont", fontSize=12, leading=17,
                        textColor=colors.HexColor("#2c3e50"), spaceBefore=8, spaceAfter=4)
    code = ParagraphStyle("code", fontName="ReportFont", fontSize=9, leading=13,
                          backColor=colors.HexColor("#f4f6f7"), borderColor=colors.HexColor("#d5dbdb"),
                          borderWidth=0.5, borderPadding=6, spaceBefore=4, spaceAfter=8, wordWrap="CJK")
    quote = ParagraphStyle("quote", fontName="ReportFont", fontSize=10, leading=15,
                           textColor=colors.HexColor("#555555"), leftIndent=12,
                           borderColor=colors.HexColor("#d5dbdb"), borderWidth=0.5,
                           borderPadding=6, spaceAfter=6, wordWrap="CJK")
    caption = ParagraphStyle("caption", fontName="ReportFont", fontSize=9, leading=12,
                             textColor=colors.grey, spaceBefore=2, spaceAfter=8)
    toc_style = ParagraphStyle("toc", fontName="ReportFont", fontSize=10.5, leading=18,
                               spaceAfter=2)

    md = MD_PATH.read_text(encoding="utf-8")
    lines = md.splitlines()

    # ---- 静态目录（H1 章节）----
    toc_entries = []
    for line in lines:
        m = re.match(r"^## (.+)$", line)
        if m:
            toc_entries.append(m.group(1).strip())

    story = []
    # 封面标题
    cover = ParagraphStyle("cover", fontName="ReportFont", fontSize=22, leading=30,
                           alignment=1, spaceAfter=8, textColor=accent)
    story.append(Paragraph("FireGuardian V2 研发技术总结报告", cover))
    story.append(Spacer(1, 6))
    story.append(Paragraph("基于 YOLO + 规则引擎 + DeepSeek Tool-Calling Agent 的智能火灾监控系统",
                           ParagraphStyle("sub", fontName="ReportFont", fontSize=12,
                                          alignment=1, textColor=colors.HexColor("#666666"),
                                          spaceAfter=12)))
    story.append(Paragraph("目录", h1))
    for i, entry in enumerate(toc_entries, 1):
        story.append(Paragraph(f"{i}. {_render_inline(entry)}", toc_style))
    story.append(PageBreak())

    # ---- 正文 ----
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "---":
            story.append(Spacer(1, 6))
            i += 1
            continue
        m = re.match(r"^### (.+)$", stripped)
        if m:
            story.append(Paragraph(_render_inline(m.group(1)), h3))
            i += 1
            continue
        m = re.match(r"^## (.+)$", stripped)
        if m:
            story.append(Paragraph(_render_inline(m.group(1)), h2))
            i += 1
            continue
        m = re.match(r"^# (.+)$", stripped)
        if m:
            story.append(Paragraph(_render_inline(m.group(1)), h1))
            i += 1
            continue
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过闭合围栏
            story.append(Paragraph("<br/>".join(_render_inline(c) for c in code_lines), code))
            continue
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = [header_cells]
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            # 按列内容长度计算加权列宽，保证总宽不超过版心、文本列不被压缩
            usable_width = A4[0] - 20 * mm - 20 * mm
            ncols = len(rows[0])
            col_lengths = [0] * ncols
            for row in rows:
                for ci, cell in enumerate(row):
                    if ci < ncols:
                        col_lengths[ci] = max(col_lengths[ci], len(str(cell)))
            weights = [max(0.9, min(3.5, length / 9.0)) for length in col_lengths]
            total_w = sum(weights)
            col_widths = [usable_width * w / total_w for w in weights]
            cell_style = ParagraphStyle(
                "tcell", fontName="ReportFont", fontSize=8.5, leading=12,
                spaceAfter=0, wordWrap="CJK",
            )
            rows_par = [
                [Paragraph(_render_inline(str(cell)), cell_style) for cell in row]
                for row in rows
            ]
            table = Table(rows_par, repeatRows=1, colWidths=col_widths)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d6eaf8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aab7c4")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(Spacer(1, 4))
            story.append(table)
            story.append(Spacer(1, 6))
            continue
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph("<br/>".join(_render_inline(q) for q in quote_lines), quote))
            continue
        if re.match(r"^\s*[-*] ", stripped):
            while i < n and re.match(r"^\s*[-*] ", lines[i].strip()):
                story.append(Paragraph("•  " + _render_inline(re.sub(r"^\s*[-*] ", "", lines[i].strip())), body))
                i += 1
            continue
        if re.match(r"^\s*\d+\.\s", stripped):
            while i < n and re.match(r"^\s*\d+\.\s", lines[i].strip()):
                num = re.match(r"^\s*(\d+)\.\s", lines[i].strip()).group(1)
                text = re.sub(r"^\s*\d+\.\s", "", lines[i].strip())
                story.append(Paragraph(f"{num}.  " + _render_inline(text), body))
                i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if m:
            alt, rel = m.group(1), m.group(2)
            img_path = _ROOT / rel
            if img_path.exists():
                from PIL import Image as PILImage
                with PILImage.open(str(img_path)) as im:
                    w, h = im.size
                max_w = 150 * mm
                scale = min(max_w / w, 1.0)
                story.append(Image(str(img_path), width=w * scale, height=h * scale))
                story.append(Paragraph(alt, caption))
            else:
                story.append(Paragraph(f"[图缺失: {rel}]", caption))
            i += 1
            continue
        # 普通段落
        story.append(Paragraph(_render_inline(stripped), body))
        i += 1

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="FireGuardian V2 研发技术总结报告")
    doc.build(story)
    print(f"[完成] PDF 已生成: {PDF_PATH}（{PDF_PATH.stat().st_size} bytes）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
