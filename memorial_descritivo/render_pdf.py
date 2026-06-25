# -*- coding: utf-8 -*-
"""
============================================================================
 RENDERIZADOR PDF  —  consome os blocos de `conteudo.py` (reportlab)
============================================================================
Gera o PDF diretamente (sem depender de Word/LibreOffice), com o mesmo
layout corporativo do DOCX: logo no cabeçalho de todas as páginas, marca
d'água de fundo, rodapé com paginação "Página X de Y" e as mesmas tabelas.
Usa a fonte DejaVu (cobre ρ, Δ, ≤, √, ², ⁻⁵, ×, ° etc.).
============================================================================
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, HRFlowable, KeepTogether)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

import dados_projeto as D

# ── Fontes Unicode (cobrem os símbolos técnicos) ───────────────────────────
_DJV = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DJV_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("DejaVu", _DJV))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", _DJV_B))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

ACENTO = colors.HexColor("#" + D.EMPRESA["cor_destaque"])
TEXTO = colors.HexColor("#262626")
CINZA = colors.HexColor("#595959")
ZEBRA = colors.HexColor("#F2F4F8")
GRADE = colors.HexColor("#BFBFBF")
CINZA_CLARO = colors.HexColor("#D9D9D9")
BRANCO = colors.white

PAGE_W, PAGE_H = A4


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ── Estilos ────────────────────────────────────────────────────────────────
def _estilos():
    s = {}
    s["body"] = ParagraphStyle("body", fontName="DejaVu", fontSize=10, leading=13.5,
                               alignment=TA_JUSTIFY, textColor=TEXTO, spaceAfter=6)
    s["heading"] = ParagraphStyle("heading", fontName="DejaVu-Bold", fontSize=12,
                                  leading=15, textColor=ACENTO, spaceBefore=12,
                                  spaceAfter=2, keepWithNext=1)
    s["subsecao"] = ParagraphStyle("subsecao", fontName="DejaVu-Bold", fontSize=10,
                                   leading=13, textColor=TEXTO, spaceBefore=4,
                                   spaceAfter=2, keepWithNext=1)
    s["nota"] = ParagraphStyle("nota", fontName="DejaVu", fontSize=8, leading=10.5,
                               alignment=TA_JUSTIFY, textColor=CINZA, spaceAfter=6)
    s["bullet"] = ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.5, leading=12.5,
                                 alignment=TA_JUSTIFY, textColor=TEXTO, leftIndent=14,
                                 bulletIndent=2, spaceAfter=2)
    s["titulo"] = ParagraphStyle("titulo", fontName="DejaVu-Bold", fontSize=15,
                                 leading=18, alignment=TA_CENTER, textColor=BRANCO)
    s["assina"] = ParagraphStyle("assina", fontName="DejaVu", fontSize=10, leading=13,
                                 alignment=TA_CENTER, textColor=TEXTO)
    s["assina_b"] = ParagraphStyle("assina_b", fontName="DejaVu-Bold", fontSize=10,
                                   leading=13, alignment=TA_CENTER, textColor=TEXTO)
    return s


def _cell(text, *, size=7.5, bold=False, align="center", color=TEXTO):
    al = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}[align]
    st = ParagraphStyle(f"c{size}{bold}{align}", fontName="DejaVu-Bold" if bold else "DejaVu",
                        fontSize=size, leading=size + 2.5, alignment=al, textColor=color)
    return Paragraph(esc(text), st)


# ── Tabelas ────────────────────────────────────────────────────────────────
def _tabela_kv(linhas, larg=(5.4, 11.6), size=9.5):
    data = [[_cell(k, size=size, bold=True, align="left", color=ACENTO),
             _cell(v, size=size, align="left")] for k, v in linhas]
    t = Table(data, colWidths=[larg[0] * cm, larg[1] * cm])
    estilo = [("GRID", (0, 0), (-1, -1), 0.5, CINZA_CLARO),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("LEFTPADDING", (0, 0), (-1, -1), 4),
              ("RIGHTPADDING", (0, 0), (-1, -1), 4),
              ("TOPPADDING", (0, 0), (-1, -1), 3),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i in range(len(linhas)):
        estilo.append(("BACKGROUND", (0, i), (0, i), ZEBRA))
    t.setStyle(TableStyle(estilo))
    return t


def _tabela_generica(cabec, linhas, larguras, size=7.5, totais=None):
    head = [_cell(c, size=size, bold=True, align="center", color=BRANCO) for c in cabec]
    data = [head] + linhas
    if totais:
        data.append(totais)
    t = Table(data, colWidths=[w * cm for w in larguras], repeatRows=1)
    estilo = [("BACKGROUND", (0, 0), (-1, 0), ACENTO),
              ("GRID", (0, 0), (-1, -1), 0.5, GRADE),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("LEFTPADDING", (0, 0), (-1, -1), 3),
              ("RIGHTPADDING", (0, 0), (-1, -1), 3),
              ("TOPPADDING", (0, 0), (-1, -1), 2.5),
              ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    # zebra nas linhas ímpares (dados 1-based como no docx: n%2)
    for n in range(len(linhas)):
        if n % 2 == 1:
            estilo.append(("BACKGROUND", (0, n + 1), (-1, n + 1), ZEBRA))
    if totais:
        estilo.append(("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), CINZA_CLARO))
    t.setStyle(TableStyle(estilo))
    return t


def _tabela_previsao(b):
    cab = ["Ambiente", "Área (m²)", "Perím. (m)", "Iluminação (VA)",
           "Tomadas (un.)", "TUG (VA)"]
    larg = [4.6, 2.2, 2.4, 3.2, 2.4, 2.2]
    linhas = []
    for l in b["linhas"]:
        linhas.append([
            _cell(l["ambiente"], size=8.5, align="left"),
            _cell(f'{l["area"]:.1f}', size=8.5), _cell(f'{l["perimetro"]:.1f}', size=8.5),
            _cell(f'{l["ilum_va"]}', size=8.5), _cell(f'{l["n_tug"]}', size=8.5),
            _cell(f'{l["tug_va"]}', size=8.5)])
    totais = [_cell("TOTAL", size=8.5, bold=True, align="left"), _cell("", size=8.5),
              _cell("", size=8.5), _cell(f'{b["tot_ilum"]}', size=8.5, bold=True),
              _cell(f'{b["tot_tomadas"]}', size=8.5, bold=True),
              _cell(f'{b["tot_tug"]}', size=8.5, bold=True)]
    return _tabela_generica(cab, linhas, larg, size=8.5, totais=totais)


def _tabela_circuitos(b):
    cab = ["Circ.", "Descrição", "Sist./V", "Pot. (VA)", "IB (A)", "Fase (mm²)",
           "PE (mm²)", "Eletrod. (mm)", "Disjuntor", "DR (mA)", "ΔV (%)"]
    larg = [1.0, 4.6, 1.3, 1.4, 1.1, 1.2, 1.1, 1.4, 1.6, 1.2, 1.1]
    linhas = []
    for cir in b["circuitos"]:
        linhas.append([
            _cell(cir["id"], bold=True), _cell(cir["descricao"], align="left"),
            _cell(f'{cir["sistema"]} / {cir["tensao"]:.0f}'),
            _cell(f'{cir["potencia_va"]:.0f}'), _cell(f'{cir["ib"]:.1f}'),
            _cell(f'{cir["secao"]:.1f}'), _cell(f'{cir["pe"]:.1f}'),
            _cell(f'{cir["eletroduto"]}'), _cell(f'{cir["disjuntor"]} (C)'),
            _cell("30" if cir["dr"] else "–"), _cell(f'{cir["dv"]:.2f}')])
    return _tabela_generica(cab, linhas, larg, size=7.5)


def _capa_titulo(texto, st):
    t = Table([[Paragraph(esc(texto), st["titulo"])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACENTO),
                           ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return t


# ── Story ──────────────────────────────────────────────────────────────────
def _story(blocos):
    st = _estilos()
    flow = []
    for b in blocos:
        tp = b["tipo"]
        if tp == "capa_titulo":
            flow += [_capa_titulo(b["texto"], st), Spacer(1, 6)]
        elif tp == "capa_ident":
            flow += [_tabela_kv(b["linhas"], larg=(4.8, 12.2)), Spacer(1, 4)]
        elif tp == "secao":
            flow.append(KeepTogether([
                Paragraph(f'{b["num"]}&nbsp;&nbsp;&nbsp;{esc(b["titulo"])}', st["heading"]),
                HRFlowable(width="100%", thickness=1, color=ACENTO,
                           spaceBefore=1, spaceAfter=5)]))
        elif tp == "subsecao":
            flow.append(Paragraph(esc(b["titulo"]), st["subsecao"]))
        elif tp == "paragrafo":
            flow.append(Paragraph(esc(b["texto"]), st["body"]))
        elif tp in ("nota", "legenda"):
            flow.append(Paragraph(esc(b["texto"]), st["nota"]))
        elif tp == "bullets":
            for it in b["itens"]:
                flow.append(Paragraph(esc(it), st["bullet"], bulletText="•"))
            flow.append(Spacer(1, 4))
        elif tp == "tabela_kv":
            flow += [_tabela_kv(b["linhas"]), Spacer(1, 4)]
        elif tp == "tabela_previsao":
            flow += [_tabela_previsao(b), Spacer(1, 4)]
        elif tp == "tabela_circuitos":
            flow += [_tabela_circuitos(b), Spacer(1, 4)]
        elif tp == "espaco":
            flow.append(Spacer(1, b.get("pt", 12)))
        elif tp == "assinatura":
            flow.append(Paragraph("_" * 42, st["assina"]))
            for texto, bold in b["linhas"]:
                flow.append(Paragraph(esc(texto), st["assina_b"] if bold else st["assina"]))
    return flow


# ── Decorações de página (cabeçalho, rodapé, marca d'água) ─────────────────
def _decoracoes(c, doc, ctx):
    emp, proj = D.EMPRESA, D.PROJETO
    # marca d'água centralizada (atrás do texto)
    try:
        img = ImageReader(ctx["marca"])
        iw, ih = img.getSize()
        w = 16 * cm
        h = w * ih / iw
        c.drawImage(img, (PAGE_W - w) / 2, (PAGE_H - h) / 2, w, h,
                    mask="auto", preserveAspectRatio=True)
    except Exception:
        pass
    # logo no cabeçalho
    try:
        logo = ImageReader(ctx["logo"])
        lw0, lh0 = logo.getSize()
        lw = 4.4 * cm
        lh = lw * lh0 / lw0
        c.drawImage(logo, 2 * cm, PAGE_H - 0.7 * cm - lh, lw, lh,
                    mask="auto", preserveAspectRatio=True)
    except Exception:
        pass
    # identificação à direita
    c.setFillColor(ACENTO); c.setFont("DejaVu-Bold", 11)
    c.drawRightString(19 * cm, PAGE_H - 1.0 * cm, emp["nome"])
    c.setFillColor(CINZA); c.setFont("DejaVu", 8)
    c.drawRightString(19 * cm, PAGE_H - 1.4 * cm, emp["slogan"])
    c.drawRightString(19 * cm, PAGE_H - 1.75 * cm,
                      f'{proj["codigo_doc"]}  ·  Rev. {proj["revisao"]}')
    # filete do cabeçalho
    c.setStrokeColor(ACENTO); c.setLineWidth(1.1)
    c.line(2 * cm, PAGE_H - 2.2 * cm, 19 * cm, PAGE_H - 2.2 * cm)
    # rodapé: filete + textos
    c.setStrokeColor(GRADE); c.setLineWidth(0.6)
    c.line(2 * cm, 1.5 * cm, 19 * cm, 1.5 * cm)
    c.setFillColor(CINZA); c.setFont("DejaVu", 7.5)
    c.drawString(2 * cm, 1.12 * cm, f'{emp["cidade_uf"]}  ·  {emp["cnpj"]}')
    c.drawCentredString(10.5 * cm, 1.12 * cm, emp["contato"])


class NumberedCanvas(canvas.Canvas):
    """Adiciona 'Página X de Y' com o total conhecido ao final."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for idx, state in enumerate(self._saved, start=1):
            self.__dict__.update(state)
            self.setFont("DejaVu", 7.5)
            self.setFillColor(CINZA)
            self.drawRightString(19 * cm, 1.12 * cm, f"Página {idx} de {total}")
            super().showPage()
        super().save()


def gerar_pdf(ctx, blocos, caminho):
    doc = BaseDocTemplate(
        caminho, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.6 * cm, bottomMargin=1.8 * cm,
        title=D.PROJETO["titulo"], author=D.PROJETO["responsavel"],
        subject="Memorial Descritivo de Instalação Elétrica – ABNT NBR 5410/5419")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="F")
    doc.addPageTemplates([PageTemplate(
        id="main", frames=[frame],
        onPage=lambda c, d: _decoracoes(c, d, ctx))])
    doc.build(_story(blocos), canvasmaker=NumberedCanvas)
    return caminho
