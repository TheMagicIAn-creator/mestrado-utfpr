# -*- coding: utf-8 -*-
"""
============================================================================
 MOTOR DE CÁLCULO  —  ABNT NBR 5410:2004 e NBR 5419:2015
============================================================================
Funções puras de engenharia. Recebem as entradas de `dados_projeto.py` e
devolvem os resultados dimensionados. Nenhum valor é fixo no documento:
tudo é recalculado a partir dos dados.

Resumo das cláusulas usadas (NBR 5410:2004):
  • 9.5.2.1  Previsão de carga de iluminação
  • 9.5.2.2  Previsão de tomadas (quantidade e potência)
  • 6.2.6    Seções mínimas dos condutores
  • 6.2.5 / Tab. 36  Capacidade de condução de corrente (ampacidade)
  • 6.2.7    Limites de queda de tensão
  • 5.3.4    Coordenação proteção × condutor (IB ≤ IN ≤ IZ)
  • 5.1.3.2  Proteção diferencial-residual (DR 30 mA)
  • 6.4 / Tab. 58  Condutor de proteção (PE)
NBR 5419-1: estimativa de Nd (frequência de eventos perigosos).
============================================================================
"""
import math

# ── Seções comerciais de condutores de cobre (mm²) ─────────────────────────
SECOES = [1.5, 2.5, 4, 6, 10, 16, 25, 35, 50, 70, 95, 120, 150, 185, 240]

# ── Capacidade de condução de corrente — NBR 5410:2004, Tabela 36 ──────────
#   Cobre, isolação PVC, condutor a 70 °C, ambiente 30 °C.
#   Chave: método de referência → {seção: (2 carregados, 3 carregados)}
AMPACIDADE = {
    "B1": {1.5:(17.5,15.5), 2.5:(24,21), 4:(32,28), 6:(41,36), 10:(57,50),
           16:(76,68), 25:(101,89), 35:(125,110), 50:(151,134), 70:(192,171),
           95:(232,207), 120:(269,239), 150:(309,275), 185:(353,314), 240:(415,370)},
    "B2": {1.5:(16.5,15), 2.5:(23,20), 4:(30,27), 6:(38,34), 10:(52,46),
           16:(69,62), 25:(90,80), 35:(111,99), 50:(133,118), 70:(168,149),
           95:(201,179), 120:(232,206), 150:(265,236), 185:(300,268), 240:(351,313)},
    "C":  {1.5:(19.5,17.5), 2.5:(27,24), 4:(36,32), 6:(46,41), 10:(63,57),
           16:(85,76), 25:(112,96), 35:(138,119), 50:(168,144), 70:(213,184),
           95:(258,223), 120:(299,259), 150:(344,299), 185:(392,341), 240:(461,403)},
    "D":  {1.5:(22,18), 2.5:(29,24), 4:(38,31), 6:(47,39), 10:(63,52),
           16:(81,67), 25:(104,86), 35:(125,103), 50:(148,122), 70:(183,151),
           95:(216,179), 120:(246,203), 150:(278,230), 185:(312,258), 240:(361,297)},
}

# ── Disjuntores comerciais — NBR IEC 60898 (A) ─────────────────────────────
DISJUNTORES = [6, 10, 16, 20, 25, 32, 40, 50, 63, 70, 80, 100, 125]

# ── Diâmetro/área externa aproximada do cabo isolado 750 V (mm²) ───────────
#   Para cálculo de ocupação do eletroduto (NBR 5410 6.2.11.1, taxa 40 %).
AREA_EXTERNA_CABO = {1.5:8.0, 2.5:10.0, 4:13.0, 6:16.0, 10:26.0, 16:36.0,
                     25:55.0, 35:73.0, 50:98.0, 70:140.0, 95:182.0, 120:225.0}
# Eletrodutos (DN nominal mm → área interna útil aproximada, mm²)
ELETRODUTOS = [(16, 130), (20, 200), (25, 330), (32, 560), (40, 900),
               (50, 1460), (60, 2010), (75, 3210)]

# Resistividade do cobre adotada para queda de tensão (Ω·mm²/m)
RHO_CU = 0.0178


# ════════════════════════════════════════════════════════════════════════
#  1) PREVISÃO DE CARGA (NBR 5410 9.5.2)
# ════════════════════════════════════════════════════════════════════════
def carga_iluminacao(area):
    """VA de iluminação (9.5.2.1): 100 VA até 6 m²; +60 VA a cada 4 m² inteiros."""
    if area <= 6:
        return 100
    return 100 + 60 * int((area - 6) // 4)


def previsao_tomadas(ambiente):
    """Quantidade e potência (VA) de tomadas de uso geral (9.5.2.2)."""
    area, perim = ambiente["area"], ambiente["perimetro"]
    umida = ambiente.get("categoria") == "umida"
    # Quantidade mínima de tomadas
    if umida:
        qtd = max(1, math.ceil(perim / 3.5))   # 1 a cada 3,5 m ou fração
    else:
        qtd = max(1, math.ceil(perim / 5.0))   # 1 a cada 5,0 m ou fração
    qtd = max(qtd, ambiente.get("min_tomadas", 1))
    # Potência atribuída
    if umida:
        # 600 VA por tomada até 3; 100 VA para as excedentes (por ambiente)
        potencia = 600 * min(qtd, 3) + 100 * max(qtd - 3, 0)
    else:
        potencia = 100 * qtd
    return qtd, potencia


def quadro_previsao(ambientes):
    """Tabela de previsão de carga por ambiente."""
    linhas, tot_ilum, tot_tug, tot_tomadas = [], 0, 0, 0
    for a in ambientes:
        ilum = carga_iluminacao(a["area"])
        qtd, pot_tug = previsao_tomadas(a)
        tot_ilum += ilum; tot_tug += pot_tug; tot_tomadas += qtd
        linhas.append({
            "ambiente": a["nome"], "area": a["area"], "perimetro": a["perimetro"],
            "ilum_va": ilum, "n_tug": qtd, "tug_va": pot_tug,
        })
    return linhas, tot_ilum, tot_tug, tot_tomadas


# ════════════════════════════════════════════════════════════════════════
#  2) CORRENTE, CONDUTOR, PROTEÇÃO E QUEDA DE TENSÃO (por circuito)
# ════════════════════════════════════════════════════════════════════════
def _tensao_e_ncond(circ, inst):
    """Retorna (tensão de referência, nº de condutores carregados, fp)."""
    sistema = circ["sistema"]
    fp = circ.get("fp", inst["fator_potencia"])
    if sistema == "FN":
        return inst["tensao_fn"], 2, fp
    if sistema == "FF":
        return inst["tensao_ff"], 2, fp
    if sistema == "3F":
        return inst["tensao_ff"], 3, fp
    raise ValueError(f"sistema inválido: {sistema}")


def corrente_projeto(potencia_va, tensao, sistema):
    """IB (A). Monofásico/bifásico: P/V. Trifásico: P/(√3·V)."""
    if sistema == "3F":
        return potencia_va / (math.sqrt(3) * tensao)
    return potencia_va / tensao


def secao_minima(tipo):
    """Seção mínima por tipo de circuito (6.2.6.1.1)."""
    return 1.5 if tipo == "iluminacao" else 2.5


def disjuntor_coordenado(ib, iz):
    """Menor disjuntor padrão com IB ≤ IN ≤ IZ (5.3.4). None se não couber."""
    for inn in DISJUNTORES:
        if ib <= inn <= iz:
            return inn
    return None


def dimensiona_condutor(ib, circ, tipo):
    """Seção do condutor E disjuntor coordenados.

    Escolhe a menor seção que (a) respeita a seção mínima, (b) tem IZ ≥ IB e
    (c) admite um disjuntor padrão com IB ≤ IN ≤ IZ. Assim a coordenação
    proteção × condutor (NBR 5410, 5.3.4) é sempre satisfeita: se o disjuntor
    padrão exceder a ampacidade da seção, a seção é aumentada.
    Retorna (seção, IZ corrigida, nº condutores carregados, disjuntor).
    """
    metodo = circ.get("metodo", "B1")
    fct, fca = circ.get("fct", 1.0), circ.get("fca", 1.0)
    ncond = 3 if circ["sistema"] == "3F" else 2
    col = 1 if ncond == 3 else 0
    tab = AMPACIDADE[metodo]
    smin = secao_minima(tipo)
    for s in SECOES:
        if s < smin or s not in tab:
            continue
        iz = tab[s][col] * fct * fca
        if iz < ib:                       # ampacidade insuficiente
            continue
        disj = disjuntor_coordenado(ib, iz)
        if disj is None:                  # disjuntor padrão não cabe → sobe seção
            continue
        return s, iz, ncond, disj
    # acima da tabela: maior seção disponível
    s = max(tab.keys())
    iz = tab[s][col] * fct * fca
    disj = disjuntor_coordenado(ib, iz) or DISJUNTORES[-1]
    return s, iz, ncond, disj


def secao_pe(secao_fase):
    """Condutor de proteção PE — NBR 5410 Tabela 58."""
    if secao_fase <= 16:
        return secao_fase
    if secao_fase <= 35:
        return 16
    return secao_fase / 2


def queda_tensao(ib, comprimento, secao, tensao, sistema):
    """ΔV (%) — modelo resistivo. FN/FF: 2·ρ·L·I/S ; 3F: √3·ρ·L·I/S."""
    if sistema == "3F":
        dv = math.sqrt(3) * RHO_CU * comprimento * ib / secao
    else:
        dv = 2 * RHO_CU * comprimento * ib / secao
    return dv / tensao * 100.0


def eletroduto(secao_fase, ncond_total):
    """DN do eletroduto pela taxa de ocupação de 40 % (NBR 5410 6.2.11.1)."""
    area_cabo = AREA_EXTERNA_CABO.get(secao_fase, 225.0)
    area_ocupada = area_cabo * ncond_total
    for dn, area_int in ELETRODUTOS:
        if area_ocupada <= 0.40 * area_int:
            return dn
    return ELETRODUTOS[-1][0]


def calcula_circuito(circ, tipo, potencia_va, inst):
    """Dimensiona um circuito terminal completo."""
    tensao, ncond_carregados, fp = _tensao_e_ncond(circ, inst)
    ib = corrente_projeto(potencia_va, tensao, circ["sistema"])
    secao, iz, ncond, disj = dimensiona_condutor(ib, circ, tipo)
    pe = secao_pe(secao)
    dv = queda_tensao(ib, circ["comprimento"], secao, tensao, circ["sistema"])
    # nº total de condutores no eletroduto: fases + neutro (FN/FF) + PE
    ncond_total = (2 if circ["sistema"] in ("FN", "FF") else 4) + 1  # +PE
    eldt = eletroduto(secao, ncond_total)
    return {
        "id": circ["id"], "descricao": circ["descricao"], "tipo": tipo,
        "sistema": circ["sistema"], "tensao": tensao, "potencia_va": potencia_va,
        "ib": ib, "secao": secao, "iz": iz, "pe": pe, "disjuntor": disj,
        "dr": circ.get("dr_30ma", False), "dv": dv, "comprimento": circ["comprimento"],
        "metodo": circ.get("metodo", "B1"), "eletroduto": eldt,
    }


# ════════════════════════════════════════════════════════════════════════
#  3) DEMANDA E RAMAL DE ENTRADA
# ════════════════════════════════════════════════════════════════════════
def calcula_demanda(p_ilum_tug, tues_va, demanda, inst):
    """Demanda provável (VA) e corrente de entrada (A)."""
    d_ilum_tug = demanda["fd_iluminacao_tug"] * p_ilum_tug
    d_tue = 0.0
    for t in tues_va:
        if "chuveiro" in t["descricao"].lower() or "aquecedor" in t["descricao"].lower():
            d_tue += demanda["fd_chuveiro"] * t["va"]
        else:
            d_tue += demanda["fd_outros_tue"] * t["va"]
    d_total = d_ilum_tug + d_tue
    # Corrente de entrada
    sistema = inst["sistema"]
    if sistema == "trifasico":
        i_entrada = d_total / (math.sqrt(3) * inst["tensao_ff"])
        ncond = 3
    elif sistema == "bifasico":
        i_entrada = d_total / (2 * inst["tensao_fn"])
        ncond = 2
    else:  # monofásico
        i_entrada = d_total / inst["tensao_fn"]
        ncond = 2
    return {"d_ilum_tug": d_ilum_tug, "d_tue": d_tue, "d_total": d_total,
            "i_entrada": i_entrada, "ncond": ncond}


def dimensiona_ramal(i_entrada, demanda, ncond):
    """Seção do ramal de entrada e disjuntor geral."""
    metodo = demanda.get("metodo_ramal", "B1")
    fct, fca = demanda.get("fct_ramal", 1.0), demanda.get("fca_ramal", 1.0)
    col = 1 if ncond == 3 else 0
    tab = AMPACIDADE[metodo]
    secao, iz, disj = None, None, None
    for s in SECOES:
        if s < 10 or s not in tab:   # ramal de entrada: mínimo usual 10 mm²
            continue
        iz_s = tab[s][col] * fct * fca
        if iz_s < i_entrada:
            continue
        d = disjuntor_coordenado(i_entrada, iz_s)
        if d is None:                # sobe seção até caber o disjuntor geral
            continue
        secao, iz, disj = s, iz_s, d
        break
    if secao is None:
        secao = max(tab.keys()); iz = tab[secao][col] * fct * fca
        disj = disjuntor_coordenado(i_entrada, iz) or DISJUNTORES[-1]
    return {"secao": secao, "iz": iz, "disjuntor": disj, "pe": secao_pe(secao),
            "metodo": metodo}


# ════════════════════════════════════════════════════════════════════════
#  4) SPDA — NBR 5419-1 (estimativa de Nd)
# ════════════════════════════════════════════════════════════════════════
def avalia_spda(spda):
    """Área de exposição equivalente Ad e frequência de eventos Nd."""
    L, W, H = spda["comprimento_m"], spda["largura_m"], spda["altura_m"]
    Ng, Cd = spda["Ng"], spda["Cd"]
    Ad = L * W + 2 * (3 * H) * (L + W) + math.pi * (3 * H) ** 2   # m²
    Nd = Ng * Ad * Cd * 1e-6                                       # eventos/ano
    recomendado = Nd > spda.get("limiar_Nd", 1.0e-3)
    return {"Ad": Ad, "Nd": Nd, "recomendado": recomendado,
            "L": L, "W": W, "H": H, "Ng": Ng, "Cd": Cd}
