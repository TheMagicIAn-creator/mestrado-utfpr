"""
pod_curva.py — Al IAdo PV

A curva **POD(a)** e o critério de viabilidade do ensaio, no formalismo dos
ensaios não destrutivos.

POR QUE ESTE MÓDULO EXISTE
==========================
Hoje o projeto tem `POD_mon` como **pontos** — a taxa de detecção medida em cada
magnitude de injeção da grade. Não há curva ajustada, não há `a₉₀`, e a `SMD`
aproxima o conceito de `a₉₀` sem intervalo de confiança.

Este módulo traz o arcabouço que a MIL-HDBK-1823A padroniza e que o guia
LS-POD da NASA adapta para amostra pequena — que é exatamente o caso aqui: não
existem 40 defeitos de tamanhos variados, existem três modos de falha e uma
grade de magnitudes.

FONTES
======
- Koshti, A. et al. *Guidebook for Limited Sample Probability of Detection
  (LS-POD) Demonstration for Signal-Response NDE Methods.* NASA/TM-20210018515,
  2021. Norma de origem: MIL-HDBK-1823A (2009); requisito: NASA-STD-5009B.
- Virkkunen, I.; Ylitalo, M. *Practical Experiences in POD Determination for
  Airframe ET Inspection.* 8th Int. Symp. on NDT in Aerospace, 2016.

O MAPEAMENTO PARA O PIPELINE
============================
    y_N    → escore do AE em janelas saudáveis ("ruído")
    y_F    → escore sob injeção numa magnitude fixa
    y_dec  → limiar operacional congelado
    POF    → taxa de falso positivo (FPR)
    POD    → POD_mon
    tamanho-alvo de defeito → magnitude de injeção declarada

Atenção ao que "ruído" significa na fonte: `y_N` deve exercitar **todas** as
fontes relevantes de variabilidade — não é ruído instrumental. No projeto, o
análogo é o conjunto de regimes de F0, e é por isso que a deriva entre
calibração e teste é tratada aqui como grandeza de primeira classe.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import math

import numpy as np

# Confiança dos limites de tolerância. 95% é o valor da MIL-HDBK-1823A e do
# LS-POD; mudar aqui muda o significado de todos os `k1`.
CONFIANCA = 0.95
# Alvos convencionais: 90% de POD (o "90" de a₉₀/₉₅) e 99% do lado do ruído
# (equivalente a POF de 1%).
P_POD = 0.90
P_POF = 0.99

FONTE = "NASA/TM-20210018515 (LS-POD); MIL-HDBK-1823A"


# ============================================================
# FATOR k1 — limite de tolerância unilateral pela t não-central
# ============================================================

def fator_k1(m: int, p: float = P_POD, confianca: float = CONFIANCA) -> float:
    """``k1 = t(α, m−1, δ)/√m`` com ``δ = z_p·√m`` — t NÃO central.

    Corrige a incerteza amostral do limite de tolerância. Quanto menor ``m``,
    maior ``k1`` — e portanto mais conservador o limite. É o que torna o LS-POD
    conservador em relação a um estudo completo: como ``m_LS < m_MH``, tem-se
    ``k1_LS > k1_MH``.

    Valores tabelados na fonte, que os testes reproduzem:
        k1 (p=0,90): m=10 → 2,355 · m=15 → 2,068 · m=20 → 1,926 · m=30 → 1,777
        k1 (p=0,99): n=40 → 2,941 · n=50 → 2,862 · n=60 → 2,807
    """
    from scipy.stats import nct, norm

    m = int(m)
    if m < 2:
        raise ValueError(f"m deve ser >= 2 para estimar dispersão, recebido {m}")
    if not 0.0 < p < 1.0:
        raise ValueError("p deve estar em (0, 1)")
    delta = norm.ppf(p) * math.sqrt(m)
    return float(nct.ppf(confianca, m - 1, delta) / math.sqrt(m))


def limite_pof(escores_saudaveis, p: float = P_POF,
               confianca: float = CONFIANCA) -> dict:
    """``y_(1/95 POF) = ȳ_N + k1N·s_N`` — o piso admissível do limiar.

    Abaixo dele, a taxa de falso positivo excede o requisito. É o **lado do
    ruído** do critério de viabilidade.
    """
    y = np.asarray(escores_saudaveis, dtype=float).reshape(-1)
    _exigir_amostra(y, "escores_saudaveis")
    n = int(y.size)
    k1 = fator_k1(n, p=p, confianca=confianca)
    return {
        "limite": float(y.mean() + k1 * y.std(ddof=1)),
        "media": float(y.mean()), "desvio": float(y.std(ddof=1)),
        "n": n, "k1": k1, "p": float(p), "confianca": float(confianca),
        "pof_alvo_pct": float((1.0 - p) * 100.0),
    }


def limite_pod(escores_falha, p: float = P_POD,
               confianca: float = CONFIANCA) -> dict:
    """``y_(90/95 POD) = ȳ_F − k1F·s_F`` — o teto admissível do limiar.

    Acima dele, a probabilidade de detecção cai abaixo do requisito. É o **lado
    da falha** do critério.
    """
    y = np.asarray(escores_falha, dtype=float).reshape(-1)
    _exigir_amostra(y, "escores_falha")
    m = int(y.size)
    k1 = fator_k1(m, p=p, confianca=confianca)
    return {
        "limite": float(y.mean() - k1 * y.std(ddof=1)),
        "media": float(y.mean()), "desvio": float(y.std(ddof=1)),
        "m": m, "k1": k1, "p": float(p), "confianca": float(confianca),
        "pod_alvo_pct": float(p * 100.0),
    }


def _exigir_amostra(y: np.ndarray, nome: str) -> None:
    if y.size < 2:
        raise ValueError(f"{nome} precisa de ao menos 2 valores")
    if not np.isfinite(y).all():
        raise ValueError(f"{nome} contém valor não finito")


# ============================================================
# CRITÉRIO DE VIABILIDADE
# ============================================================

def viabilidade(escores_saudaveis, escores_falha, limiar: float,
                p_pod: float = P_POD, p_pof: float = P_POF) -> dict:
    """O ensaio é viável se ``y_(1/95 POF) ≤ y_dec ≤ y_(90/95 POD)``.

    **Se a faixa for vazia, o ensaio FALHOU** — e isso é resultado, não erro de
    execução. Significa que as distribuições de saudável e de falha não estão
    suficientemente separadas: não existe limiar algum que satisfaça POD e POF
    ao mesmo tempo.

    É a formalização normativa do que `docs/decisao_fpr_1pct.md` já havia
    concluído empiricamente ao rejeitar o corte de FPR ≤ 1%.

    A fonte prescreve quatro remédios, todos mapeáveis ao projeto:
      1. aumentar a magnitude-alvo (declarar SMD maior em vez de prometer
         detecção da magnitude atual);
      2. aceitar POF maior, com decisão documentada;
      3. melhorar o lado do ruído (features, filtragem, cobertura de regime);
      4. aumentar `m` — mas com **espécimes novos**, não repetindo medição nos
         mesmos. No projeto: mais trajetórias independentes, não mais janelas
         da mesma trajetória.
    """
    pof = limite_pof(escores_saudaveis, p=p_pof)
    pod = limite_pod(escores_falha, p=p_pod)
    y_dec = float(limiar)
    faixa_existe = pof["limite"] <= pod["limite"]
    dentro = pof["limite"] <= y_dec <= pod["limite"]

    if not faixa_existe:
        veredito = "ensaio_falhou"
        leitura = (
            "NENHUM limiar satisfaz POD e POF simultaneamente: o piso do falso "
            f"positivo ({pof['limite']:.4f}) está ACIMA do teto da detecção "
            f"({pod['limite']:.4f}). As distribuições de saudável e de falha não "
            "estão suficientemente separadas. Isto é resultado do ensaio, não "
            "defeito de execução — ver os quatro remédios na docstring.")
    elif dentro:
        veredito = "viavel"
        leitura = (
            f"o limiar {y_dec:.4f} cai na faixa admissível "
            f"[{pof['limite']:.4f}; {pod['limite']:.4f}]: entrega "
            f"{pod['pod_alvo_pct']:.0f}% de POD com {pof['confianca']:.0%} de "
            f"confiança, mantendo POF abaixo de {pof['pof_alvo_pct']:.1f}%.")
    else:
        lado = "abaixo do piso de POF" if y_dec < pof["limite"] else "acima do teto de POD"
        veredito = "limiar_fora_da_faixa"
        leitura = (
            f"existe faixa admissível [{pof['limite']:.4f}; {pod['limite']:.4f}], "
            f"mas o limiar adotado ({y_dec:.4f}) está {lado}. O limiar pode ser "
            "movido para dentro da faixa sem mudar nada do detector.")

    return {
        "veredito": veredito, "leitura": leitura,
        "faixa_admissivel_existe": bool(faixa_existe),
        "limiar_dentro_da_faixa": bool(dentro),
        "y_dec": y_dec,
        "faixa": [pof["limite"], pod["limite"]],
        "limite_pof": pof, "limite_pod": pod,
        "fonte": FONTE, "evidence_level": "E2",
    }


# ============================================================
# DERIVA DE CAMPO — o achado dos regimes de F0, com fórmula
# ============================================================

# Margem de proteção contra alarme espúrio de deriva. O desvio-padrão do próprio
# limite y_(1/95 POF) é ≈ 11% de (y_(1/95 POF) − ȳ_N) para n = 40 (9% para 60;
# 7% para 100). A fonte adota 10%.
MARGEM_DERIVA = 0.10


def deriva_de_campo(escores_calibracao, escores_campo, limiar: float,
                    margem: float = MARGEM_DERIVA) -> dict:
    """Monitoramento de processo: o ruído em campo ainda é o da qualificação?

    Dois gatilhos da fonte (LS-POD §3.1):

    **Deriva** — se ``y_(1/95 POF, campo) > y_(1/95 POF) + margem·(y_(1/95 POF)
    − ȳ_N)``, o ruído aumentou além da variabilidade esperada e o processo deve
    ser investigado.

    **Invalidação** — se ``y_(1/95 POF, campo) > y_dec``, o requisito de POF
    **não está sendo cumprido**, e a inspeção deve parar até a causa ser
    identificada.

    No projeto, "campo" é o bloco de teste e "qualificação" é o de calibração.
    Isto dá métrica nomeada, fórmula e critério normativo ao achado de que
    calibração e teste estão em regimes de F0 diferentes — que até aqui era
    observação qualitativa.
    """
    base = limite_pof(escores_calibracao)
    campo = limite_pof(escores_campo)
    gatilho = base["limite"] + float(margem) * (base["limite"] - base["media"])
    derivou = campo["limite"] > gatilho
    invalidou = campo["limite"] > float(limiar)

    if invalidou:
        leitura = (
            f"INVALIDAÇÃO: o piso de falso positivo em campo ({campo['limite']:.4f}) "
            f"ultrapassa o próprio limiar adotado ({float(limiar):.4f}). O "
            "requisito de POF não está sendo cumprido no bloco de teste. Pela "
            "fonte, a resposta correta é investigar a causa — não reapertar o "
            "limiar, que trataria o sintoma.")
    elif derivou:
        leitura = (
            f"DERIVA: o piso em campo ({campo['limite']:.4f}) passou do gatilho "
            f"({gatilho:.4f}). O ruído aumentou além da variabilidade esperada; "
            "investigar cobertura de dados antes de mexer no detector.")
    else:
        leitura = (
            f"estável: o piso em campo ({campo['limite']:.4f}) está dentro do "
            f"gatilho ({gatilho:.4f}).")

    return {
        "derivou": bool(derivou), "invalidou": bool(invalidou),
        "leitura": leitura, "gatilho": float(gatilho), "margem": float(margem),
        "calibracao": base, "campo": campo, "y_dec": float(limiar),
        "fonte": FONTE,
    }


# ============================================================
# HIPÓTESES QUE PRECISAM VALER (LS-POD §1.7)
# ============================================================

def checar_normalidade(amostra, nome: str = "amostra") -> dict:
    """Hipótese 1 do LS-POD: os limites de tolerância assumem normalidade.

    A fonte é explícita — o método é **inválido** se `y_N` e `y_F` não forem
    aproximadamente normais, ou se não houver transformação que os torne. A
    sugestão dela é o log quando há valores próximos de zero.

    Aqui a checagem é obrigatória e vai para o artefato, para que nenhum limite
    de POF/POD seja reportado sem o veredito da hipótese que o sustenta.

    Devolve também a estimativa **empírica** do mesmo quantil, que não assume
    distribuição: se as três (normal, log-normal, empírica) levarem à mesma
    conclusão, ela é robusta à violação — e isso é o que se declara.
    """
    from scipy import stats

    x = np.asarray(amostra, dtype=float).reshape(-1)
    _exigir_amostra(x, nome)

    def _teste(v):
        if v.size < 3:
            return {"W": None, "p": None, "normal": None}
        W, pval = stats.shapiro(v)
        return {"W": float(W), "p": float(pval), "normal": bool(pval > 0.05)}

    positivos = x[x > 0]
    bruto = _teste(x)
    log = _teste(np.log(positivos)) if positivos.size >= 3 else {
        "W": None, "p": None, "normal": None}

    return {
        "nome": nome, "n": int(x.size),
        "assimetria": float(stats.skew(x)),
        "curtose_excesso": float(stats.kurtosis(x)),
        "shapiro_bruto": bruto,
        "shapiro_log": log,
        "vale": bool(bruto["normal"] or log["normal"]),
        "melhor_escala": ("bruto" if bruto["normal"]
                          else "log" if log["normal"] else "nenhuma"),
        "nota": ("se nenhuma escala é normal, os limites de tolerância são "
                 "aproximações; confira se a conclusão se sustenta também pelo "
                 "quantil empírico antes de reportá-la"),
    }


def limite_pof_empirico(escores_saudaveis, p: float = P_POF) -> float:
    """O mesmo quantil, sem assumir distribuição — contraprova do `limite_pof`.

    Serve para responder à objeção óbvia: "o limite depende de normalidade, e a
    normalidade foi violada". Se o quantil empírico leva à mesma conclusão, ela
    não é artefato da hipótese.
    """
    y = np.asarray(escores_saudaveis, dtype=float).reshape(-1)
    _exigir_amostra(y, "escores_saudaveis")
    return float(np.percentile(y, float(p) * 100.0))


def verificar_hipoteses(escores_por_magnitude: dict,
                        escores_saudaveis) -> dict:
    """O arcabouço POD é INVÁLIDO se estas condições não valerem.

    Devolve veredito por hipótese, para o artefato registrar em vez de supor.
    A mais importante é a **monotonicidade**: se o escore não cresce com a
    magnitude da injeção, não existe curva POD(a) a ajustar, e o arcabouço
    inteiro não se aplica.
    """
    y_n = np.asarray(escores_saudaveis, dtype=float).reshape(-1)
    magnitudes = sorted(float(a) for a in escores_por_magnitude)
    medias = [float(np.mean(escores_por_magnitude[a])) for a in magnitudes]

    monotono = all(b >= a for a, b in zip(medias, medias[1:]))
    # Saturação: a maior magnitude empata com a anterior dentro do ruído.
    satura = False
    if len(medias) >= 2:
        passo_final = medias[-1] - medias[-2]
        satura = bool(passo_final <= 0.01 * max(abs(medias[-1]), 1e-12))
    piso_zero = bool(np.count_nonzero(y_n <= 0) > 0.01 * y_n.size)

    normalidade = checar_normalidade(y_n, "escores_saudaveis")

    return {
        "normalidade": normalidade,
        "monotonicidade": {
            "vale": bool(monotono),
            "medias_por_magnitude": dict(zip(map(str, magnitudes), medias)),
            "nota": ("o escore deve crescer com a magnitude injetada; se não "
                     "cresce, não há curva POD(a) a ajustar"),
        },
        "saturacao_no_topo": {
            "detectada": satura,
            "nota": ("escore que empata entre as duas maiores magnitudes indica "
                     "saturação — a curva POD não pode ser extrapolada acima"),
        },
        "piso_artificial_em_zero": {
            "detectado": piso_zero,
            "n_zeros": int(np.count_nonzero(y_n <= 0)),
            "nota": ("valores censurados em zero destroem a normalidade da "
                     "cauda inferior do lado saudável"),
        },
        # Normalidade violada NÃO impede aplicar: torna o limite
        # aproximado, e exige a contraprova pelo quantil empírico.
        "aplicavel": bool(monotono and not piso_zero),
        "limites_sao_aproximados": bool(not normalidade["vale"]),
        "fonte": FONTE,
    }
