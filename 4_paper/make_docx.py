"""LaTeX paper -> Word (.docx) with native equations, continuous line numbers, Times 12."""
import re, subprocess, os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

import os as _os
ROOT = _os.environ.get("THERMOSWARM_ROOT", "/home/claude")  # base folder for data/ and outputs
os.chdir(f"{ROOT}/paper_tex")
tex = open("thermoswarm.tex").read()

# --- bibliography: number in listed order, convert citations to [n] -------------
bib = re.findall(r"\\bibitem\[(.*?)\]\{(.*?)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", tex, re.S)
num = {k: i + 1 for i, (_, k, _) in enumerate(bib)}
auth = {k: re.sub(r"\(\d{4}\)$", "", lab).strip() for lab, k, _ in bib}


def citep(m):
    keys = [k.strip() for k in m.group(2).split(",")]
    pre = m.group(1)
    ns = sorted(num[k] for k in keys)
    inner = ", ".join(str(n) for n in ns)
    if pre:  # \citep[XGBoost;][]{chen2016}
        p = re.match(r"\[(.*?)\]\[(.*?)\]", pre)
        return f"[{p.group(1)} {inner}]" if p else f"[{inner}]"
    return f"[{inner}]"


tex = re.sub(r"\\citep((?:\[[^\]]*\])*)\{([^}]*)\}", citep, tex)
tex = re.sub(r"\\citet\{([^}]*)\}", lambda m: f"{auth[m.group(1)].replace('~', ' ')} [{num[m.group(1)]}]", tex)
refs = "\n\n".join(f"[{i + 1}] " + re.sub(r"\s+", " ", body).strip() for i, (_, _, body) in enumerate(bib))
tex = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
             lambda m: "\\section*{References}\n\n" + refs, tex, flags=re.S)
# --- cross-references from LaTeX's own .aux file ------------------------------------
aux = open("thermoswarm.aux").read()
lab = dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))
lab["app:params"] = "A"
def eqtag(m):
    body, key = m.group(1), m.group(2)
    return "\\begin{equation}" + body + "\\qquad\\qquad (" + lab[key] + ")\n\\end{equation}"
tex = re.sub(r"\\begin\{equation\}(.*?)\\label\{(eq:[^}]*)\}\s*\\end\{equation\}", eqtag, tex, flags=re.S)
def capfix(m):
    env, body = m.group(1), m.group(2)
    k = re.search(r"\\label\{([^}]*)\}", body)
    if k and k.group(1) in lab:
        word = "Figure" if env == "figure" else "Table"
        body = body.replace("\\caption{", "\\caption{\\textbf{" + word + " " + lab[k.group(1)] + ".} ", 1)
    return "\\begin{" + env + "}" + body + "\\end{" + env + "}"
tex = re.sub(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", capfix, tex, flags=re.S)
tex = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + lab[m.group(1)] + ")", tex)
tex = re.sub(r"Eq\.~\\ref\{([^}]*)\}", lambda m: "Eq. (" + lab[m.group(1)] + ")", tex)
tex = re.sub(r"\\ref\{([^}]*)\}", lambda m: lab[m.group(1)], tex)
tex = re.sub(r"\\label\{[^}]*\}", "", tex)
# title block before the abstract
tex = re.sub(r"\\begin\{center\}.*?\\end\{center\}", lambda m: "\\maketitle", tex, count=1, flags=re.S)
tex = tex.replace("\\begin{document}", "\\title{ThermoSwarm: Altitude-Resolved Storm-Time Thermospheric Density from Multi-Object Orbital Decay, with Physics-Informed Drag Forecasting}\n\\author{Darsh Chaudhary\\\\Pathways School Noida}\n\\date{}\n\\begin{document}")
tex = tex.replace("\\appendix\n\\section{Key Parameters}", "\\section*{Appendix A. Key Parameters}")
# --- simplifications for pandoc -----------------------------------------------------
tex = tex.replace("figs/fig2_map.pdf", "figs/fig2_map.png").replace("figs/fig3_validation.pdf", "figs/fig3_validation.png")
for f in ["fig4_storm_bias", "fig5_cases", "fig6_tau", "fig7_forecast"]:
    tex = tex.replace(f"figs/{f}.pdf", f"figs/{f}.png")
tex = tex.replace(r"\operatorname*{median}_{\substack{\text{quiet},\ |t'-t|\leq 30\,\mathrm{d}}} f(t')",
                  r"\mathrm{median}\{\, f(t') : \text{quiet},\ |t'-t| \leq 30\ \mathrm{d} \,\}")
tex = re.sub(r"\\subsubsection\{([^}]*)\}", r"\\textbf{\1.} ", tex)
tex = tex.replace("\\linenumbers", "")
open("paper_docx.tex", "w").write(tex)

for f in ["fig2_map", "fig3_validation", "fig4_storm_bias", "fig5_cases", "fig6_tau", "fig7_forecast"]:
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", f"figs/{f}.pdf", f"figs/{f}"], check=True)

# --- reference styles ------------------------------------------------------------------
subprocess.run(["pandoc", "-o", "ref.docx", "--print-default-data-file", "reference.docx"], check=True)
ref = Document("ref.docx"); STY = {s.name: s for s in ref.styles}; TNR = "Times New Roman"


def font(st, size=None, bold=None, italic=None):
    st.font.name = TNR
    rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ("ascii", "hAnsi", "cs", "eastAsia"): rf.set(qn("w:" + a), TNR)
    for a in ("asciiTheme", "hAnsiTheme", "cstheme", "eastAsiaTheme"):
        if rf.get(qn("w:" + a)) is not None: del rf.attrib[qn("w:" + a)]
    if size: st.font.size = Pt(size)
    if bold is not None: st.font.bold = bold
    if italic is not None: st.font.italic = italic
    st.font.color.rgb = RGBColor(0, 0, 0)


for n in ["Normal", "Body Text", "First Paragraph", "Compact", "Table", "Abstract", "Bibliography"]:
    if n in STY: font(STY[n], 11.5)
for n in ["Body Text", "First Paragraph"]:
    pf = STY[n].paragraph_format; pf.space_after = Pt(6); pf.line_spacing = 1.1
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for n, sz in [("Heading 1", 13.5), ("Heading 2", 12), ("Heading 3", 11.5), ("Heading 4", 11.5)]:
    font(STY[n], sz, bold=True, italic=(n == "Heading 4"))
    STY[n].paragraph_format.space_before = Pt(12 if n == "Heading 1" else 8); STY[n].paragraph_format.space_after = Pt(4)
for n in ["Title"]:
    font(STY[n], 17, bold=True); STY[n].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
for n in ["Author", "Date"]:
    if n in STY: font(STY[n], 12); STY[n].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
for n in ["Abstract Title"]:
    if n in STY: font(STY[n], 12, bold=True)
for n in ["Image Caption", "Table Caption", "Caption"]:
    if n in STY: font(STY[n], 10, italic=False)
if "Abstract" in STY:
    STY["Abstract"].paragraph_format.left_indent = Inches(0.4); STY["Abstract"].paragraph_format.right_indent = Inches(0.4)
    STY["Abstract"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
sec = ref.sections[0]; sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"): setattr(sec, m, Inches(1))
ref.save("ref.docx")

subprocess.run(["pandoc", "paper_docx.tex", "-f", "latex", "-o", "raw.docx", "--reference-doc=ref.docx",
                "--resource-path=.", "--number-sections"], check=True)

# --- post-process ------------------------------------------------------------------------
d = Document("raw.docx")
body = d.element.body
for sp in body.iter(qn("w:sectPr")):
    ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); ln.set(qn("w:distance"), "360")
    # lnNumType must come after pgMar/paperSrc etc.; insert before cols/docGrid if present
    anchor = sp.find(qn("w:cols")) or sp.find(qn("w:docGrid"))
    (anchor.addprevious(ln) if anchor is not None else sp.append(ln))
for t in d.tables:
    tblPr = t._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for e, sz in (("top", "8"), ("bottom", "8"), ("insideH", "2")):
        x = OxmlElement(f"w:{e}"); x.set(qn("w:val"), "single"); x.set(qn("w:sz"), sz); x.set(qn("w:color"), "000000"); b.append(x)
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None: tblPr.remove(old)
    tblPr.append(b)
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr(); trPr.append(OxmlElement("w:cantSplit"))
        if i == 0: trPr.append(OxmlElement("w:tblHeader"))
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.size = Pt(10); r.font.name = TNR
                    if i == 0: r.font.bold = True
for p in d.paragraphs:
    if p._p.find(".//" + qn("w:drawing")) is not None or p._p.find(".//" + qn("m:oMathPara")) is not None:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pPr = p._p.get_or_add_pPr(); s = OxmlElement("w:suppressLineNumbers"); pPr.append(s) if p._p.find(".//" + qn("w:drawing")) is not None else None
d.core_properties.author = "Darsh Chaudhary"; d.core_properties.title = "ThermoSwarm"
d.save(f"{ROOT}/ThermoSwarm_Research_Paper_v2.docx")
print("docx saved")
