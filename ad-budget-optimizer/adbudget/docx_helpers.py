"""Small helpers for building the Word report (tables, bullets, figures)."""
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def _shade(cell, hex_fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc, header: list[str], rows: list[list], widths: list[float] | None = None):
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.autofit = False
    for c, h in enumerate(header):
        cell = table.rows[0].cells[c]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        _shade(cell, "2F5D8A")
    for r in rows:
        cells = table.add_row().cells
        for c, v in enumerate(r):
            cells[c].text = str(v)
    for row in table.rows:
        for c, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9.5)
            if widths:
                cell.width = Inches(widths[c])
    doc.add_paragraph()


def bullets(doc, items: list[str]):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def figure(doc, path, caption: str, width: float = 5.8):
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)
