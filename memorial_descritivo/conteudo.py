# -*- coding: utf-8 -*-
"""
============================================================================
 CONTEÚDO DO MEMORIAL  —  fonte única (texto + estrutura)
============================================================================
Monta a lista ORDENADA de "blocos" do documento a partir dos cálculos.
Tanto o renderizador DOCX quanto o PDF consomem exatamente estes blocos,
garantindo que os dois arquivos nunca divirjam.

Cada bloco é um dict com a chave "tipo". Tipos suportados:
  capa_titulo | capa_ident | secao | subsecao | paragrafo | nota |
  bullets | legenda | tabela_kv | tabela_previsao | tabela_circuitos |
  espaco | assinatura
============================================================================
"""
import dados_projeto as D
import calculos_nbr as C


def blocos(ctx):
    inst = D.INSTALACAO
    previsao, tot_ilum, tot_tug, tot_tomadas = ctx["previsao"]
    dem, ram, s = ctx["demanda"], ctx["ramal"], ctx["spda"]
    P = D.PROJETO

    B = []
    add = B.append

    # ── CAPA (enxuta) ──
    add({"tipo": "capa_titulo", "texto": P["titulo"]})
    add({"tipo": "capa_ident", "linhas": [
        ("Obra", P["obra"]),
        ("Cliente / Contratante", P["cliente"]),
        ("Endereço", P["endereco"]),
        ("Área construída", f'{P["area_construida_m2"]:.2f} m²'),
        ("Documento", f'{P["codigo_doc"]} – Revisão {P["revisao"]}'),
        ("Data de emissão", P["data"]),
        ("Responsável técnico", f'{P["responsavel"]} – {P["crea"]}'),
    ]})

    # ── 1. OBJETO ──
    add({"tipo": "secao", "num": "1.", "titulo": "OBJETO"})
    add({"tipo": "paragrafo", "texto":
        f'Este memorial descritivo apresenta as premissas, os critérios e o '
        f'dimensionamento da instalação elétrica de baixa tensão da obra '
        f'“{P["obra"]}”, situada à {P["endereco"]}. O documento integra '
        f'o projeto elétrico e tem por finalidade descrever as características '
        f'técnicas da instalação, a previsão de cargas, a divisão de circuitos, '
        f'o dimensionamento de condutores e dispositivos de proteção, o sistema '
        f'de aterramento e a análise da necessidade de proteção contra descargas '
        f'atmosféricas (SPDA).'})

    # ── 2. NORMAS ──
    add({"tipo": "secao", "num": "2.", "titulo": "NORMAS E DOCUMENTOS DE REFERÊNCIA"})
    add({"tipo": "paragrafo", "texto":
        "O projeto observa as prescrições das normas técnicas vigentes, com "
        "destaque para:"})
    add({"tipo": "bullets", "itens": ctx["normas"]})

    # ── 3. CARACTERÍSTICAS GERAIS ──
    add({"tipo": "secao", "num": "3.", "titulo": "CARACTERÍSTICAS GERAIS DA INSTALAÇÃO"})
    add({"tipo": "tabela_kv", "linhas": [
        ("Concessionária", inst["concessionaria"]),
        ("Tipo de fornecimento", inst["fornecimento"]),
        ("Tensões nominais", f'{inst["tensao_fn"]:.0f} V (F-N) / '
                             f'{inst["tensao_ff"]:.0f} V (F-F) – {inst["frequencia_hz"]} Hz'),
        ("Esquema de aterramento", inst["esquema_aterramento"]),
        ("Fator de potência de referência", f'{inst["fator_potencia"]:.2f}'),
        ("Temperatura ambiente de projeto", f'{inst["temperatura_ambiente_C"]} °C'),
    ]})

    # ── 4. PREVISÃO DE CARGAS ──
    add({"tipo": "secao", "num": "4.", "titulo": "PREVISÃO DE CARGAS"})
    add({"tipo": "paragrafo", "texto":
        "A previsão de carga de iluminação e de tomadas de uso geral (TUG) segue "
        "os critérios mínimos da ABNT NBR 5410:2004 (item 9.5.2): iluminação de "
        "100 VA para áreas até 6 m², acrescida de 60 VA a cada 4 m² inteiros "
        "excedentes; tomadas previstas pelo perímetro de cada ambiente, com "
        "600 VA por ponto nas três primeiras tomadas de áreas molhadas (cozinha, "
        "copa, área de serviço, banheiros) e 100 VA nas demais."})
    add({"tipo": "tabela_previsao", "linhas": previsao,
         "tot_ilum": tot_ilum, "tot_tug": tot_tug, "tot_tomadas": tot_tomadas})
    add({"tipo": "paragrafo", "texto":
        f'Às cargas acima somam-se as tomadas de uso específico (TUE) dos '
        f'equipamentos fixos. A potência total instalada resulta em '
        f'{ctx["p_instalada"]/1000:.2f} kW ({ctx["p_instalada"]:.0f} VA), assim '
        f'distribuída: iluminação + TUG = {ctx["p_ilum_tug"]:.0f} VA e '
        f'TUE = {ctx["p_tue"]:.0f} VA.'})

    # ── 5. DIVISÃO DE CIRCUITOS ──
    add({"tipo": "secao", "num": "5.",
         "titulo": "DIVISÃO DE CIRCUITOS E QUADRO DE DISTRIBUIÇÃO"})
    add({"tipo": "paragrafo", "texto":
        "Os circuitos terminais foram separados por função e por ambiente, com "
        "circuitos independentes para iluminação, tomadas de uso geral e cada "
        "tomada de uso específico, conforme a ABNT NBR 5410 (item 4.2.5). O "
        "quadro a seguir consolida o dimensionamento de cada circuito."})
    add({"tipo": "tabela_circuitos", "circuitos": ctx["circuitos"]})
    add({"tipo": "legenda", "texto":
        "Legenda — Sist.: FN = fase-neutro, FF = fase-fase; Fase/PE: seção "
        "nominal do condutor de fase e de proteção (cobre); Eletrod.: diâmetro "
        "nominal do eletroduto; DR: corrente diferencial-residual nominal; "
        "ΔV: queda de tensão do circuito. Disjuntores em curva C (ABNT NBR IEC "
        "60898)."})

    # ── 6. MEMORIAL DE CÁLCULO ──
    add({"tipo": "secao", "num": "6.", "titulo": "MEMORIAL DE CÁLCULO"})
    add({"tipo": "subsecao", "titulo": "6.1  Critérios e formulação"})
    add({"tipo": "bullets", "itens": ctx["criterios"]})

    add({"tipo": "subsecao", "titulo": "6.2  Demanda provável e ramal de entrada"})
    add({"tipo": "paragrafo", "texto":
        f'Aplicando os fatores de demanda adotados (iluminação + TUG = '
        f'{D.DEMANDA["fd_iluminacao_tug"]:.2f}; chuveiros/aquecedores = '
        f'{D.DEMANDA["fd_chuveiro"]:.2f}; demais TUE = '
        f'{D.DEMANDA["fd_outros_tue"]:.2f}), a demanda provável resulta em '
        f'{dem["d_total"]/1000:.2f} kVA, correspondente a uma corrente de entrada '
        f'de {dem["i_entrada"]:.1f} A. O ramal de entrada é dimensionado em '
        f'condutor de cobre de {ram["secao"]:.0f} mm² (PE de {ram["pe"]:.0f} mm²), '
        f'protegido por disjuntor geral de {ram["disjuntor"]} A.'})
    add({"tipo": "tabela_kv", "linhas": [
        ("Demanda – iluminação + TUG", f'{dem["d_ilum_tug"]:.0f} VA'),
        ("Demanda – uso específico (TUE)", f'{dem["d_tue"]:.0f} VA'),
        ("Demanda provável total",
         f'{dem["d_total"]:.0f} VA  ({dem["d_total"]/1000:.2f} kVA)'),
        ("Corrente de entrada", f'{dem["i_entrada"]:.1f} A'),
        ("Condutor do ramal de entrada (Cu)",
         f'{ram["secao"]:.0f} mm² (método {ram["metodo"]}) + PE {ram["pe"]:.0f} mm²'),
        ("Disjuntor geral", f'{ram["disjuntor"]} A'),
    ]})
    add({"tipo": "nota", "texto":
        "Observação: os fatores de demanda são exemplificativos. Devem ser "
        "substituídos pelos valores da norma de fornecimento da concessionária "
        "local (p.ex. COPEL NTC 901100, CEMIG ND-5.1, CPFL/Enel), que também "
        "define o padrão de entrada, a caixa de medição e o ramal de ligação."})

    add({"tipo": "subsecao", "titulo": "6.3  Queda de tensão"})
    add({"tipo": "paragrafo", "texto":
        f'A queda de tensão foi verificada para cada circuito pelo modelo '
        f'resistivo ΔV(%) = (k·ρ·L·IB)/(S·V)·100, com ρ = {C.RHO_CU} Ω·mm²/m '
        f'(cobre) e k = 2 para circuitos monofásicos/bifásicos e √3 para '
        f'trifásicos. A maior queda de tensão entre os circuitos terminais é de '
        f'{ctx["dv_max"]:.2f} %, atendendo ao limite de 4 % para circuitos '
        f'terminais e ao limite global de 7 % estabelecidos no item 6.2.7 da '
        f'ABNT NBR 5410.'})

    add({"tipo": "subsecao",
         "titulo": "6.4  Proteção contra sobrecorrente, choques e surtos"})
    add({"tipo": "bullets", "itens": ctx["protecao"]})

    add({"tipo": "subsecao", "titulo": "6.5  Aterramento e condutor de proteção"})
    add({"tipo": "paragrafo", "texto":
        f'Adota-se o esquema de aterramento {inst["esquema_aterramento"]}, com '
        f'barramento de equipotencialização principal (BEP) e condutor de '
        f'proteção (PE) acompanhando todos os circuitos. As seções do PE seguem a '
        f'Tabela 58 da ABNT NBR 5410 (S_PE = S_fase para S ≤ 16 mm²; 16 mm² para '
        f'16 < S ≤ 35 mm²; S_fase/2 acima). O eletrodo de aterramento e a '
        f'equipotencialização atendem ao item 6.4 da norma, com ligação à '
        f'malha/haste e ao DPS.'})

    # ── 7. SPDA ──
    add({"tipo": "secao", "num": "7.",
         "titulo": "PROTEÇÃO CONTRA DESCARGAS ATMOSFÉRICAS (SPDA)"})
    add({"tipo": "paragrafo", "texto":
        f'A necessidade de SPDA é avaliada segundo a ABNT NBR 5419:2015. Para a '
        f'edificação de {s["L"]:.0f} × {s["W"]:.0f} × {s["H"]:.0f} m, a área de '
        f'exposição equivalente é Ad = {s["Ad"]:.0f} m². Com densidade de '
        f'descargas Ng = {s["Ng"]:.1f} raios/km²/ano e fator de localização '
        f'Cd = {s["Cd"]:.2f}, a frequência estimada de eventos perigosos é '
        f'Nd = {s["Nd"]:.2e} eventos/ano.'})
    rec = ("recomenda-se a instalação de SPDA, devendo a classe e o método de "
           "captação ser definidos pelo gerenciamento de risco completo"
           if s["recomendado"] else
           "a estimativa preliminar não indica obrigatoriedade de SPDA")
    add({"tipo": "paragrafo", "texto":
        f'Pelo critério preliminar de frequência, {rec}. A decisão definitiva '
        f'deve resultar do gerenciamento de risco da ABNT NBR 5419-2, comparando '
        f'o risco R1 (perda de vida humana) ao risco tolerável RT = 1×10⁻⁵, '
        f'considerando ainda as medidas de proteção (DPS, equipotencialização e '
        f'aterramento) já previstas neste projeto.'})

    # ── 8. MATERIAIS ──
    add({"tipo": "secao", "num": "8.", "titulo": "ESPECIFICAÇÃO BÁSICA DE MATERIAIS"})
    add({"tipo": "bullets", "itens": ctx["materiais"]})

    # ── 9. CONSIDERAÇÕES FINAIS ──
    add({"tipo": "secao", "num": "9.", "titulo": "CONSIDERAÇÕES FINAIS"})
    add({"tipo": "paragrafo", "texto":
        "A execução deve seguir integralmente este projeto, as normas citadas e "
        "as boas práticas de engenharia, com mão de obra qualificada e sob "
        "observância da NR-10. Qualquer alteração em obra deve ser previamente "
        "submetida ao responsável técnico. Os materiais devem ser certificados "
        "(INMETRO, quando aplicável) e a instalação deve ser ensaiada "
        "(continuidade do PE, resistência de isolamento, funcionamento dos DR e "
        "resistência de aterramento) antes da energização."})

    # ── 10. RESPONSÁVEL TÉCNICO ──
    add({"tipo": "secao", "num": "10.", "titulo": "RESPONSÁVEL TÉCNICO"})
    add({"tipo": "espaco", "pt": 22})
    add({"tipo": "assinatura", "linhas": [
        (P["responsavel"], True),
        (P["titulo_prof"], False),
        (P["crea"], False),
        (P["art"], False),
    ]})
    return B


def textos_apoio():
    """Listas de normas, critérios, proteção e materiais (independem de layout)."""
    normas = [
        "ABNT NBR 5410:2004 – Instalações elétricas de baixa tensão;",
        "ABNT NBR 5419:2015 (Partes 1 a 4) – Proteção contra descargas atmosféricas;",
        "ABNT NBR IEC 60898 – Disjuntores para proteção de sobrecorrentes em "
        "instalações domésticas;",
        "ABNT NBR IEC 61008 / 61009 – Interruptores diferenciais-residuais (DR);",
        "ABNT NBR 5444 / ABNT NBR IEC 60617 – Símbolos gráficos para instalações "
        "elétricas prediais;",
        "ABNT NBR ISO/CIE 8995-1 – Iluminação (níveis de iluminância);",
        "Norma de fornecimento da concessionária local (padrão de entrada e medição);",
        "NR-10 – Segurança em instalações e serviços em eletricidade.",
    ]
    criterios = [
        "Corrente de projeto: IB = P/V para circuitos monofásicos/bifásicos e "
        "IB = P/(√3·V) para trifásicos;",
        "Seções mínimas (NBR 5410, 6.2.6.1.1): 1,5 mm² para iluminação e 2,5 mm² "
        "para circuitos de tomadas (cobre);",
        "Capacidade de condução de corrente (ampacidade): NBR 5410 Tabela 36, "
        "cobre/PVC, com IZ corrigida por temperatura (FCT) e agrupamento (FCA), "
        "exigindo IZ ≥ IB;",
        "Coordenação proteção × condutor (NBR 5410, 5.3.4): IB ≤ IN ≤ IZ e "
        "I2 ≤ 1,45·IZ;",
        "Condutor de proteção PE: NBR 5410, Tabela 58, em função da seção de fase;",
        "Queda de tensão: limites de 4 % nos circuitos terminais e 7 % global "
        "(NBR 5410, 6.2.7).",
    ]
    protecao = [
        "Sobrecorrente: cada circuito é protegido por disjuntor termomagnético "
        "(curva C, NBR IEC 60898) coordenado ao condutor (IB ≤ IN ≤ IZ);",
        "Choques elétricos: proteção por seccionamento automático e dispositivos "
        "diferenciais-residuais (DR) de alta sensibilidade (IΔn ≤ 30 mA) nos "
        "circuitos de tomadas de áreas molhadas, externas e de banheiros "
        "(NBR 5410, 5.1.3.2);",
        "Surtos: dispositivo de proteção contra surtos (DPS) classe II instalado "
        "na origem da instalação, junto ao quadro de distribuição (NBR 5410, 6.3.5);",
        "Seccionamento e comando: disjuntor geral no quadro de distribuição, com "
        "acessibilidade e identificação de todos os circuitos.",
    ]
    materiais = [
        "Condutores: cobre, isolação em PVC 70 °C / 750 V (ou EPR/XLPE quando "
        "indicado), com cores normalizadas — neutro azul-claro e PE verde-amarelo "
        "(NBR 5410, 6.1.5.3);",
        "Eletrodutos: PVC rígido antichama, embutidos, com taxa de ocupação "
        "≤ 40 % (NBR 5410, 6.2.11);",
        "Disjuntores: termomagnéticos curva C, padrão DIN, capacidade de "
        "interrupção compatível com o curto-circuito presumido;",
        "Dispositivos DR de 30 mA e DPS classe II no quadro de distribuição;",
        "Quadro de distribuição (QDC) com barramentos de neutro e de PE separados "
        "(esquema TN-S);",
        "Aterramento: haste(s) de aço-cobre e condutores de equipotencialização "
        "conforme NBR 5410, item 6.4.",
    ]
    return normas, criterios, protecao, materiais
