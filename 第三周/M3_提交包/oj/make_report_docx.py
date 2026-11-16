#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 report/M2.md 渲染成课程模板 docx（只支持本项目用到的 Markdown 子集）。

用法（包根目录）：
    python oj/make_report_docx.py
输出：
    ../../第二周/第二周实验报告_数据与基准.docx
"""
from __future__ import annotations

import os
import re
import sys

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LAB = os.path.dirname(os.path.dirname(ROOT))
TEMPLATE = os.path.join(LAB, "第二周", "lab-report-template.docx")
MD = os.path.join(ROOT, "report", "M2.md")
OUT = os.path.join(LAB, "第二周", "第二周实验报告_数据与基准.docx")

CN_FONT = "宋体"
CN_HEAD = "黑体"
MONO = "Consolas"
BODY_PT = 10.5


def set_run_font(run, size=None, bold=None, color=None, mono=False):
    run.font.name = MONO if mono else CN_FONT
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), CN_FONT if not mono else MONO)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color:
        run.font.color.rgb = color


def add_runs(p, text, size=BODY_PT, base_bold=False):
    """解析 **粗体** 与 `行内代码`。"""
    tokens = re.split(r"(\*\*.+?\*\*|`.+?`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2])
            set_run_font(r, size, True)
        elif tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1])
            set_run_font(r, size - 0.5, base_bold, mono=True)
        else:
            r = p.add_run(tok)
            set_run_font(r, size, base_bold)


def shade(cell, hex_color="E7E6E6"):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:color"): "auto",
                                          qn("w:fill"): hex_color})
    tcPr.append(shd)


def add_table(doc, rows, widths=None):
    ncol = len(rows[0])
    t = doc.add_table(rows=0, cols=ncol)
    try:
        t.style = "Grid Table Light"
    except KeyError:
        t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cell = cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            add_runs(p, val, 9, base_bold=(i == 0))
            if i == 0:
                shade(cell, "DCE6F1")
            if widths and j < len(widths):
                cell.width = Cm(widths[j])
    # 表头跨页重复
    if len(t.rows):
        trPr = t.rows[0]._tr.get_or_add_trPr()
        trPr.append(trPr.makeelement(qn("w:tblHeader"), {qn("w:val"): "true"}))
    doc.add_paragraph()


def parse_md(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", code))
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                blocks.append(("table", rows))
            continue
        if s.startswith("# "):
            blocks.append(("h1", s[2:].strip()))
        elif s.startswith("## "):
            blocks.append(("h2", s[3:].strip()))
        elif s.startswith("### "):
            blocks.append(("h3", s[4:].strip()))
        elif s.startswith("> "):
            quote = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                quote.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append(("quote", quote))
            continue
        elif re.match(r"^\d+\.\s", s):
            m = re.match(r"^(\d+)\.\s+(.*)$", s)
            blocks.append(("num", (m.group(1), m.group(2))))
        elif s.startswith("- "):
            blocks.append(("bullet", s[2:].strip()))
        else:
            blocks.append(("p", s))
        i += 1
    return blocks


def main() -> int:
    if not os.path.exists(TEMPLATE):
        print("找不到模板：", TEMPLATE)
        return 1
    doc = docx.Document(TEMPLATE)

    # 填信息表
    info = doc.tables[0]
    values = [
        "中文匹配·⑩ 中文改写语义一致性核查系统（T4-PAR / pawsx_zh）",
        "2026年11月9日",
        "M2 · 数据与基准",
    ]
    for i, v in enumerate(values):
        cell = info.rows[i].cells[1]
        cell.text = ""
        p = cell.paragraphs[0]
        add_runs(p, v, 10.5)
        cell.width = Cm(12.5)

    # 删除标题/信息表之后的模板占位内容
    body = doc.element.body
    keep_until = None
    children = list(body)
    for idx, ch in enumerate(children):
        if ch.tag.endswith("}tbl"):
            keep_until = idx
            break
    for ch in children[keep_until + 1:]:
        if ch.tag.endswith("}sectPr"):
            continue
        body.remove(ch)

    # Normal 样式统一字体
    normal = doc.styles["Normal"]
    normal.font.name = CN_FONT
    normal.font.size = Pt(BODY_PT)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(3)

    for kind, payload in parse_md(MD):
        if kind == "h1":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, payload, 12, base_bold=True)
        elif kind == "h2":
            p = doc.add_paragraph(style="Heading 1")
            add_runs(p, payload, BODY_PT, base_bold=True)
            for r in p.runs:
                r.font.name = CN_HEAD
                r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), CN_HEAD)
        elif kind == "h3":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            add_runs(p, payload, BODY_PT, base_bold=True)
        elif kind == "quote":
            for line in payload:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.4)
                p.paragraph_format.space_after = Pt(1)
                add_runs(p, line, 9)
                for r in p.runs:
                    r.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
                    r.italic = True
        elif kind == "bullet":
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, payload)
        elif kind == "num":
            n, text = payload
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.55)
            p.paragraph_format.first_line_indent = Cm(-0.55)
            add_runs(p, f"{n}. {text}")
        elif kind == "code":
            for line in payload:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.left_indent = Cm(0.3)
                r = p.add_run(line)
                set_run_font(r, 8.5, mono=True)
                pPr = p._p.get_or_add_pPr()
                pPr.append(pPr.makeelement(qn("w:shd"), {qn("w:val"): "clear",
                                                         qn("w:color"): "auto",
                                                         qn("w:fill"): "F2F2F2"}))
            doc.add_paragraph()
        elif kind == "table":
            ncol = len(payload[0])
            widths = {2: [7.0, 10.0], 3: [4.4, 6.2, 6.4], 4: [3.6, 4.4, 4.5, 4.5],
                      5: [3.6, 2.6, 4.2, 2.9, 3.7], 6: [2.9, 3.0, 2.8, 3.0, 2.6, 2.7],
                      7: [2.8, 1.6, 1.6, 2.6, 2.7, 2.6, 2.7]}.get(ncol)
            add_table(doc, payload, widths)
        else:
            p = doc.add_paragraph()
            add_runs(p, payload)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc.save(OUT)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
