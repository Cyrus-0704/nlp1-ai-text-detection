"""Build the reviewable DOCX for the C technical report from Markdown.

The Markdown is the editable source. This script intentionally supports only the
subset of Markdown used by ``reports/technical_report_draft.md``: ATX headings,
paragraphs, bullet/numbered lists, fenced code blocks, blockquotes, pipe tables,
bold/italic/inline-code spans, and horizontal rules.

Usage:
    python reports/build_technical_report.py
    python reports/build_technical_report.py --input reports/technical_report_draft.md \
        --output reports/technical_report_v1_review.docx
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_EAST_ASIA = "宋体"
HEADING_EAST_ASIA = "黑体"
LATIN_BODY = "Times New Roman"
LATIN_HEADING = "Arial"
ACCENT = RGBColor(0x1F, 0x38, 0x64)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT_FILL = "F2F5FA"
HEADER_FILL = "1F3864"


def set_run_font(run, east_asia: str = BODY_EAST_ASIA, latin: str = LATIN_BODY,
                 size: float | None = None, bold: bool | None = None,
                 color: RGBColor | None = None) -> None:
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)
    rfonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def style_paragraph(p, space_before: float = 0, space_after: float = 4,
                    line: float = 1.35, first_indent: float | None = None,
                    left_indent: float | None = None) -> None:
    fmt = p.paragraph_format
    fmt.space_before = Pt(space_before)
    fmt.space_after = Pt(space_after)
    fmt.line_spacing = line
    if first_indent is not None:
        fmt.first_line_indent = Cm(first_indent)
    if left_indent is not None:
        fmt.left_indent = Cm(left_indent)


def add_bottom_border(paragraph, color: str = "BFBFBF", size: int = 6) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    ppr.append(pbdr)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run("第 ")
    set_run_font(run, size=8.5, color=GRAY)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    run2 = paragraph.add_run(" 页 / 共 ")
    set_run_font(run2, size=8.5, color=GRAY)
    fld_begin2 = OxmlElement("w:fldChar")
    fld_begin2.set(qn("w:fldCharType"), "begin")
    instr2 = OxmlElement("w:instrText")
    instr2.set(qn("xml:space"), "preserve")
    instr2.text = "NUMPAGES"
    fld_end2 = OxmlElement("w:fldChar")
    fld_end2.set(qn("w:fldCharType"), "end")
    run2._r.append(fld_begin2)
    run2._r.append(instr2)
    run2._r.append(fld_end2)
    run3 = paragraph.add_run(" 页")
    set_run_font(run3, size=8.5, color=GRAY)


def add_inline(paragraph, text: str, size: float = 10.5,
               color: RGBColor | None = None) -> None:
    """Add bold/italic/inline-code spans; strip Markdown links to their label."""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")
    for chunk in pattern.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            run = paragraph.add_run(chunk[2:-2])
            set_run_font(run, size=size, bold=True, color=color)
        elif chunk.startswith("`") and chunk.endswith("`") and len(chunk) > 2:
            run = paragraph.add_run(chunk[1:-1])
            set_run_font(run, east_asia=BODY_EAST_ASIA, latin="Consolas",
                         size=size - 0.5, color=RGBColor(0x7F, 0x1D, 0x1D))
        else:
            run = paragraph.add_run(chunk)
            set_run_font(run, size=size, color=color)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        line = lines[i].strip()
        if re.fullmatch(r"\|?[\s:|-]+\|?", line):
            i += 1
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    cant_split.set(qn("w:val"), "true")
    tr_pr.append(cant_split)


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False

    # Usable width is about 16.6 cm; give the first column more room.
    header_text = " ".join(rows[0]) if rows else ""
    if ncols == 7:
        widths = [4.6, 2.0, 2.0, 1.9, 2.4, 1.8, 1.8]
    elif ncols == 5 and "SHA-256" in header_text:
        widths = [3.0, 2.6, 2.6, 2.6, 5.8]
    elif ncols == 5 and "检查范围" in header_text:
        widths = [4.6, 3.2, 3.2, 3.0, 2.6]
    elif ncols == 4 and "SHA-256" in header_text:
        widths = [3.8, 2.8, 2.8, 7.2]
    elif ncols == 4:
        widths = [5.2, 3.8, 3.8, 3.6]
    elif ncols == 3:
        widths = [5.6, 6.0, 5.0]
    else:
        widths = [16.6 / ncols] * ncols

    for c_idx, width in enumerate(widths):
        if c_idx < ncols:
            table.columns[c_idx].width = Cm(width)

    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx]
        prevent_row_split(row)
        if r_idx == 0:
            set_repeat_header(row)
        for c_idx in range(ncols):
            cell = row.cells[c_idx]
            cell.width = Cm(widths[c_idx] if c_idx < len(widths) else 2.0)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            style_paragraph(p, space_before=0, space_after=0, line=1.05)
            text = row_data[c_idx] if c_idx < len(row_data) else ""
            add_inline(p, text, size=8.5,
                       color=RGBColor(0xFF, 0xFF, 0xFF) if r_idx == 0 else None)
            if r_idx == 0:
                set_cell_shading(cell, HEADER_FILL)
                for run in p.runs:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            elif r_idx % 2 == 0:
                set_cell_shading(cell, LIGHT_FILL)
    # Small space after the table.
    spacer = doc.add_paragraph()
    style_paragraph(spacer, space_before=0, space_after=2, line=1.0)


def add_code_block(doc: Document, text: str) -> None:
    for line in text.splitlines() or [""]:
        p = doc.add_paragraph()
        style_paragraph(p, space_before=0, space_after=0, line=1.05, left_indent=0.5)
        ppr = p._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "F3F3F3")
        ppr.append(shd)
        run = p.add_run(line if line else " ")
        set_run_font(run, east_asia=BODY_EAST_ASIA, latin="Consolas", size=9,
                     color=RGBColor(0x33, 0x33, 0x33))


def add_heading(doc: Document, level: int, text: str) -> None:
    sizes = {1: 16, 2: 13, 3: 11.5}
    p = doc.add_paragraph()
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style_paragraph(p, space_before=16, space_after=8, line=1.2)
        add_bottom_border(p, color="1F3864", size=8)
    elif level == 2:
        style_paragraph(p, space_before=12, space_after=5, line=1.2)
    else:
        style_paragraph(p, space_before=8, space_after=3, line=1.2)
    run = p.add_run(text)
    set_run_font(run, east_asia=HEADING_EAST_ASIA, latin=LATIN_HEADING,
                 size=sizes.get(level, 11), bold=True, color=ACCENT)
    # Keep headings with the following paragraph.
    p.paragraph_format.keep_with_next = True


def build(md_path: Path, docx_path: Path) -> int:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(0.9)

    normal = doc.styles["Normal"]
    normal.font.name = LATIN_BODY
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_EAST_ASIA)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(4)

    # Header and footer.
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("大模型生成文本检测技术报告 · 内部审阅稿")
    set_run_font(hr, size=8.5, color=GRAY)
    add_bottom_border(hp, color="D9D9D9", size=4)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(fp)

    # Core properties: keep anonymous.
    props = doc.core_properties
    props.title = "大模型生成文本检测技术报告"
    props.subject = "初赛第一版候选内部审阅稿"
    props.author = ""
    props.last_modified_by = ""
    props.comments = ""
    props.category = ""
    props.keywords = ""
    props.identifier = ""
    props.language = "zh-CN"
    props.created = datetime(2026, 10, 18, 0, 0, 0)
    props.modified = datetime(2026, 10, 18, 0, 0, 0)
    props.revision = 1

    title_done = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Title block: first H1 becomes the document title.
        if stripped.startswith("# ") and not title_done:
            title_done = True
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_paragraph(p, space_before=12, space_after=6, line=1.15)
            run = p.add_run(stripped[2:].strip())
            set_run_font(run, east_asia=HEADING_EAST_ASIA, latin=LATIN_HEADING,
                         size=22, bold=True, color=ACCENT)
            i += 1
            continue

        if stripped.startswith("# "):
            add_heading(doc, 1, stripped[2:].strip())
            i += 1
            continue
        if stripped.startswith("## "):
            text = stripped[3:].strip()
            if text.startswith("附录 A"):
                doc.add_page_break()
            add_heading(doc, 2, text)
            i += 1
            continue
        if stripped.startswith("### "):
            add_heading(doc, 3, stripped[4:].strip())
            i += 1
            continue

        # Metadata lines immediately under the title: centered, not bullets.
        if title_done and i < 12 and stripped.startswith("- "):
            metadata = stripped[2:].strip()
            p = doc.add_paragraph()
            if metadata.startswith("匿名说明"):
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                style_paragraph(p, space_before=4, space_after=4, line=1.25,
                                left_indent=1.1)
                add_inline(p, metadata, size=9, color=GRAY)
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                style_paragraph(p, space_before=0, space_after=1, line=1.2)
                add_inline(p, metadata, size=9.5, color=GRAY)
            i += 1
            continue

        # Table.
        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            add_table(doc, rows)
            continue

        # Fenced code.
        if stripped.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, "\n".join(code_lines))
            continue

        # Blockquote.
        if stripped.startswith(">"):
            text = stripped.lstrip(">").strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            style_paragraph(p, space_before=2, space_after=4, line=1.25)
            add_inline(p, text, size=9.5, color=GRAY)
            i += 1
            continue

        # Horizontal rule.
        if re.fullmatch(r"-{3,}", stripped):
            p = doc.add_paragraph()
            style_paragraph(p, space_before=2, space_after=2, line=1.0)
            add_bottom_border(p, color="D9D9D9", size=4)
            i += 1
            continue

        # Lists.
        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        number_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if bullet_match:
            p = doc.add_paragraph()
            style_paragraph(p, space_before=0, space_after=2, line=1.3,
                            left_indent=0.75, first_indent=-0.4)
            run = p.add_run("• ")
            set_run_font(run, size=10.5, color=ACCENT)
            add_inline(p, bullet_match.group(1))
            i += 1
            continue
        if number_match:
            p = doc.add_paragraph()
            style_paragraph(p, space_before=0, space_after=2, line=1.3,
                            left_indent=0.75, first_indent=-0.45)
            run = p.add_run(f"{number_match.group(1)}. ")
            set_run_font(run, size=10.5, bold=True, color=ACCENT)
            add_inline(p, number_match.group(2))
            i += 1
            continue

        # Subtitle line directly under the title.
        if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4 and i < 8:
            text = stripped.strip("*")
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_paragraph(p, space_before=0, space_after=4, line=1.2)
            run = p.add_run(text)
            set_run_font(run, east_asia=HEADING_EAST_ASIA, latin=LATIN_HEADING,
                         size=12, bold=True, color=ACCENT)
            i += 1
            continue

        # Normal paragraph.
        p = doc.add_paragraph()
        style_paragraph(p, space_before=0, space_after=4, line=1.35,
                        first_indent=0.0)
        add_inline(p, stripped)
        i += 1

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(docx_path)
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("reports/technical_report_draft.md"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/technical_report_v1_review.docx"))
    args = parser.parse_args()
    n = build(args.input, args.output)
    print(f"built {args.output} from {args.input} ({n} markdown lines)")


if __name__ == "__main__":
    main()
