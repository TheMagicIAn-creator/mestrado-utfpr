# -*- coding: utf-8 -*-
"""
============================================================================
 GERADOR DE MEMORIAL DESCRITIVO ELÉTRICO  (.docx + .pdf)
============================================================================
Modelo enxuto e corporativo, com BASE DE CÁLCULO RESPONSIVA: edite
`dados_projeto.py`, rode este arquivo e o memorial é recalculado e
reemitido em DOCX e PDF. Layout com logo no cabeçalho de todas as páginas
e marca d'água de fundo (logo substituível em assets/logo.png).

Fluxo:
    dados_projeto.py  →  calculos_nbr.py  →  contexto  →  conteudo.py
                                                       →  render_docx / render_pdf

Uso:
    python gerar_memorial.py

Saída:
    saida/Memorial_Descritivo_Eletrico.docx
    saida/Memorial_Descritivo_Eletrico.pdf
============================================================================
"""
import os
import sys

import dados_projeto as D
import calculos_nbr as C
import conteudo
from marca import gerar_logo_exemplo, derivar_marca_dagua
from render_docx import gerar_docx
from render_pdf import gerar_pdf

AQUI = os.path.dirname(os.path.abspath(__file__))


def construir_contexto(logo, marca):
    """Roda todos os cálculos da NBR e devolve o contexto do documento."""
    inst = D.INSTALACAO
    previsao, tot_ilum, tot_tug, tot_tomadas = C.quadro_previsao(D.AMBIENTES)
    mapa_ilum = {l["ambiente"]: l["ilum_va"] for l in previsao}
    mapa_tug = {l["ambiente"]: l["tug_va"] for l in previsao}

    circuitos = []
    for cir in D.CIRCUITOS_ILUMINACAO:
        pot = sum(mapa_ilum[a] for a in cir["ambientes"])
        circuitos.append(C.calcula_circuito(cir, "iluminacao", pot, inst))
    for cir in D.CIRCUITOS_TUG:
        pot = sum(mapa_tug[a] for a in cir["ambientes"])
        circuitos.append(C.calcula_circuito(cir, "tomada", pot, inst))
    tues_va = []
    for cir in D.CIRCUITOS_TUE:
        va = cir["potencia_W"] / cir.get("fp", 1.0)
        tues_va.append({"descricao": cir["descricao"], "va": va})
        circuitos.append(C.calcula_circuito(cir, "tue", va, inst))

    p_ilum_tug = tot_ilum + tot_tug
    p_tue = sum(t["va"] for t in tues_va)
    p_instalada = p_ilum_tug + p_tue
    demanda = C.calcula_demanda(p_ilum_tug, tues_va, D.DEMANDA, inst)
    ramal = C.dimensiona_ramal(demanda["i_entrada"], D.DEMANDA, demanda["ncond"])
    spda = C.avalia_spda(D.SPDA)
    dv_max = max(c["dv"] for c in circuitos)

    normas, criterios, protecao, materiais = conteudo.textos_apoio()
    return {
        "logo": logo, "marca": marca,
        "normas": normas, "criterios": criterios,
        "protecao": protecao, "materiais": materiais,
        "previsao": (previsao, tot_ilum, tot_tug, tot_tomadas),
        "circuitos": circuitos, "demanda": demanda, "ramal": ramal, "spda": spda,
        "p_ilum_tug": p_ilum_tug, "p_tue": p_tue, "p_instalada": p_instalada,
        "dv_max": dv_max,
    }


def preparar_branding():
    logo = D.EMPRESA["logo"]
    if not os.path.exists(logo):
        print(f"• Logo não encontrada em '{logo}'. Gerando logo de exemplo...")
        gerar_logo_exemplo(logo, D.EMPRESA["cor_destaque"])
    marca = os.path.join("assets", "marca_dagua.png")
    derivar_marca_dagua(logo, marca, D.EMPRESA.get("marca_dagua_intensidade", 0.10))
    print("• Logo e marca d'água prontas.")
    return logo, marca


def main():
    os.chdir(AQUI)
    logo, marca = preparar_branding()

    ctx = construir_contexto(logo, marca)
    print(f"• Cálculos NBR concluídos: {len(ctx['circuitos'])} circuitos · "
          f"P_inst = {ctx['p_instalada']:.0f} VA · "
          f"demanda = {ctx['demanda']['d_total']/1000:.2f} kVA · "
          f"ΔV_máx = {ctx['dv_max']:.2f} %.")

    blocos = conteudo.blocos(ctx)

    out_dir = os.path.join(AQUI, "saida")
    os.makedirs(out_dir, exist_ok=True)
    docx_path = os.path.join(out_dir, "Memorial_Descritivo_Eletrico.docx")
    pdf_path = os.path.join(out_dir, "Memorial_Descritivo_Eletrico.pdf")

    gerar_docx(ctx, blocos, docx_path)
    print(f"• DOCX salvo: {docx_path}")
    gerar_pdf(ctx, blocos, pdf_path)
    print(f"• PDF salvo:  {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
