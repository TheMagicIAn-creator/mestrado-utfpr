"""
confiabilidade.py — Al IAdo PV

As funções fechadas da distribuição de Weibull de 2 parâmetros, como **número**.

POR QUE ESTE MÓDULO EXISTE
==========================
`R(t)` e `h(t)` já eram calculadas — mas **dentro do código de plotagem**
(`graficos_rul.py`), inline, e o resultado ia só para o PNG. Consequência: o
agente não conseguia responder "qual a confiabilidade em t = 40?" com número, a
dissertação não tinha valor para tabelar, e a banca não tinha o que conferir.
Era o que o pesquisador chamou de "muito resultado e pouca margem
interpretativa".

Aqui as funções ficam disponíveis como dado. `graficos_rul.py` passa a importar
daqui em vez de recalcular — fonte única.

AS FUNÇÕES, E O QUE CADA UMA RESPONDE
=====================================
Com forma ``β`` (beta) e escala ``η`` (eta), para ``t > 0``:

    f(t) = (β/η)·(t/η)^(β−1)·exp(−(t/η)^β)     densidade
    F(t) = 1 − exp(−(t/η)^β)                    probabilidade de já ter falhado
    R(t) = exp(−(t/η)^β)                        probabilidade de sobreviver
    h(t) = (β/η)·(t/η)^(β−1)                    taxa de falha INSTANTÂNEA
    H(t) = (t/η)^β                              taxa acumulada, = −ln R(t)
    B_p  = η·(−ln(1−p))^(1/β)                   quantil: idade em que p% falhou
    MTTF = η·Γ(1 + 1/β)                         vida média

A leitura de engenharia está em ``h(t)``, não em ``MTTF``:

    β < 1  →  h decrescente  →  mortalidade infantil; trocar não ajuda
    β = 1  →  h constante    →  falha aleatória; manutenção preventiva é inútil
    β > 1  →  h crescente    →  desgaste; existe intervalo ótimo de troca

**Mas essa leitura só se sustenta se o IC de β não cruzar 1.** Se cruzar, a
afirmação honesta é "não se distingue de taxa constante" — ver
``classificar_forma``.

UNIDADE
=======
Neste projeto o eixo NÃO é tempo físico. É a magnitude de injeção em que a
detecção se confirma (ver docs/auditoria_total_src.md §3). As funções aqui são
puramente matemáticas e não sabem disso; quem chama é responsável por rotular.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import math

import numpy as np

# Referências das fórmulas, para citação na dissertação:
# - Nketiah, E. A.; Chenlong, L.; Yingchuan, J.; Dwumah, B. (2021). Parameter
#   estimation of the Weibull Distribution. IJAERS 8(9):210-224.
# - Genschel, U.; Meeker, W. Q. (2009). A comparison of maximum likelihood and
#   median rank regression for Weibull estimation. Iowa State University.
FONTE_FORMULAS = "Weibull 2P; ver Nketiah et al. (2021), IJAERS 8(9)"
FONTE_POSICOES_CENSURADAS = (
    "NIST/SEMATECH e-Handbook, secao 8.2.1.5: Kaplan-Meier modificado"
)


def _validar(beta: float, eta: float) -> tuple[float, float]:
    b, e = float(beta), float(eta)
    if not (b > 0 and math.isfinite(b)):
        raise ValueError(f"beta deve ser positivo e finito, recebido {beta!r}")
    if not (e > 0 and math.isfinite(e)):
        raise ValueError(f"eta deve ser positivo e finito, recebido {eta!r}")
    return b, e


def _t(t) -> np.ndarray:
    v = np.asarray(t, dtype=float)
    if np.any(v < 0):
        raise ValueError("t não pode ser negativo")
    return v


def confiabilidade(t, beta: float, eta: float) -> np.ndarray:
    """``R(t) = exp(−(t/η)^β)`` — probabilidade de sobreviver além de ``t``.

    É a **curva de confiabilidade**. Vale 1 em t=0 e decresce monotonicamente.
    Em ``t = η`` vale ``e^(−1) ≈ 0,368`` para **qualquer** β — é por isso que η
    se chama vida característica.
    """
    b, e = _validar(beta, eta)
    return np.exp(-((_t(t) / e) ** b))


def acumulada(t, beta: float, eta: float) -> np.ndarray:
    """``F(t) = 1 − exp(−(t/η)^β)`` — probabilidade de ter falhado até ``t``.

    Usa ``-expm1(-H)`` em vez de ``1 - exp(-H)``. Não é preciosismo: para
    ``H`` pequeno (idade muito abaixo de η, ou β grande), ``exp(-H)`` vale
    ``1 - ε`` e a subtração de 1 perde **toda** a precisão significativa por
    cancelamento. Com β = 5,4 e t = η/24, a forma ingênua erra em ordens de
    grandeza. ``expm1`` é construída para esse caso.
    """
    b, e = _validar(beta, eta)
    return -np.expm1(-((_t(t) / e) ** b))


def densidade(t, beta: float, eta: float) -> np.ndarray:
    """``f(t)`` — densidade de probabilidade. É a "distribuição de Weibull".

    Em ``t = 0`` a densidade diverge para β < 1, vale ``1/η`` para β = 1 e vale
    0 para β > 1 — o que dá à família as formas características.
    """
    b, e = _validar(beta, eta)
    v = _t(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = (b / e) * (v / e) ** (b - 1.0) * np.exp(-((v / e) ** b))
    return np.where(np.isfinite(dens), dens, np.inf)


def taxa_falha(t, beta: float, eta: float) -> np.ndarray:
    """``h(t) = f(t)/R(t) = (β/η)·(t/η)^(β−1)`` — taxa de falha instantânea.

    É a grandeza que decide manutenção, e a que o MTTF esconde. Note que ela
    NÃO é probabilidade: é taxa, e pode passar de 1.
    """
    b, e = _validar(beta, eta)
    v = _t(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (b / e) * (v / e) ** (b - 1.0)
    return np.where(np.isfinite(h), h, np.inf)


def taxa_acumulada(t, beta: float, eta: float) -> np.ndarray:
    """``H(t) = (t/η)^β = −ln R(t)`` — risco acumulado até ``t``."""
    b, e = _validar(beta, eta)
    return (_t(t) / e) ** b


def quantil(p: float, beta: float, eta: float) -> float:
    """``B_p = η·(−ln(1−p))^(1/β)`` — a idade em que uma fração ``p`` falhou.

    ``quantil(0.10, ...)`` é o **B10**, ``quantil(0.01, ...)`` o **B1** e
    ``quantil(0.50, ...)`` a vida mediana.

    Para decisão de manutenção B10/B1 são melhores indicadores que o MTTF: a
    Weibull é assimétrica, e a média pode ficar acima de boa parte da população.
    """
    b, e = _validar(beta, eta)
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p deve estar em (0, 1), recebido {p!r}")
    return float(e * (-math.log1p(-p)) ** (1.0 / b))


def vida_media(beta: float, eta: float) -> float:
    """``MTTF = η·Γ(1 + 1/β)`` — a média da distribuição."""
    b, e = _validar(beta, eta)
    return float(e * math.gamma(1.0 + 1.0 / b))


def classificar_forma(
    beta: float,
    ic_beta: tuple[float, float] | None = None,
    *,
    eixo_tempo: bool = True,
) -> dict:
    """Leitura de engenharia de ``β``, **com a ressalva do intervalo**.

    O log do pipeline dizia "β > 1 → taxa de falha crescente (desgaste)". Isso
    só se sustenta se o IC95 de β **não cruzar 1**. Se cruzar, o dado não
    distingue desgaste de falha aleatória, e afirmar desgaste é ir além da
    evidência.

    Devolve ``regime``, ``leitura`` e ``conclusivo``. Quando ``conclusivo`` é
    False, o texto já vem com a ressalva embutida — para ninguém copiar a
    afirmação forte por engano.
    """
    b = float(beta)
    if not eixo_tempo:
        if b > 1.0:
            regime, leitura = "intensidade_deteccao_crescente", (
                "a intensidade parametrica do primeiro cruzamento aumenta com "
                "a magnitude injetada. Como o eixo nao e idade, isso NAO "
                "significa desgaste nem autoriza intervalo de manutencao")
        elif b < 1.0:
            regime, leitura = "intensidade_deteccao_decrescente", (
                "a intensidade parametrica do primeiro cruzamento diminui com "
                "a magnitude injetada. Como o eixo nao e idade, isso NAO "
                "significa mortalidade infantil")
        else:
            regime, leitura = "intensidade_deteccao_constante", (
                "a intensidade parametrica do primeiro cruzamento e constante "
                "na escala de magnitude; nao ha interpretacao de falha aleatoria")

        conclusivo = True
        if ic_beta is not None:
            lo, hi = float(ic_beta[0]), float(ic_beta[1])
            if lo <= 1.0 <= hi:
                conclusivo = False
                leitura = (
                    f"beta = {b:.2f}, mas o IC95 [{lo:.2f}; {hi:.2f}] cruza 1; "
                    "a forma da intensidade de deteccao nao e distinguivel de "
                    "constante. Nenhuma leitura fisica de desgaste e autorizada")
        return {
            "beta": b,
            "regime": regime,
            "leitura": leitura,
            "conclusivo": bool(conclusivo),
            "eixo_tempo": False,
            "inferencia_manutencao_autorizada": False,
        }

    if b > 1.0:
        regime, leitura = "desgaste", (
            "taxa de falha crescente: o risco aumenta com a idade, e existe "
            "intervalo de substituição preventiva que compensa")
    elif b < 1.0:
        regime, leitura = "mortalidade_infantil", (
            "taxa de falha decrescente: falhas se concentram no início; "
            "substituir item em serviço tende a piorar")
    else:
        regime, leitura = "aleatorio", (
            "taxa de falha constante: falha sem memória de idade; "
            "substituição preventiva não reduz a taxa")

    conclusivo = True
    if ic_beta is not None:
        lo, hi = float(ic_beta[0]), float(ic_beta[1])
        if lo <= 1.0 <= hi:
            conclusivo = False
            leitura = (
                f"β = {b:.2f}, mas o IC95 [{lo:.2f}; {hi:.2f}] CRUZA 1 — o dado "
                "não distingue desgaste de falha aleatória. Não afirmar regime.")
    return {
        "beta": b,
        "regime": regime,
        "leitura": leitura,
        "conclusivo": bool(conclusivo),
        "eixo_tempo": True,
        "inferencia_manutencao_autorizada": bool(conclusivo),
    }


def grade_tempo(t_max: float, n: int = 200, t_min: float | None = None) -> np.ndarray:
    """Grade para amostrar as curvas.

    Começa acima de zero porque ``h(t)`` e ``f(t)`` divergem em t=0 quando
    β < 1 — plotar ou tabelar o infinito não informa nada.
    """
    t_max = float(t_max)
    if not t_max > 0:
        raise ValueError("t_max deve ser positivo")
    piso = float(t_min) if t_min is not None else max(t_max / 1000.0, 1e-6)
    return np.linspace(piso, t_max, int(n))


def curvas(beta: float, eta: float, t_max: float, n: int = 200) -> dict:
    """Todas as funções amostradas numa grade — o bloco que vai para o JSON.

    É o que permite responder "qual a confiabilidade em t = 40?" com número, em
    vez de apontar para um PNG.
    """
    t = grade_tempo(t_max, n)
    return {
        "t": t.tolist(),
        "R": confiabilidade(t, beta, eta).tolist(),
        "F": acumulada(t, beta, eta).tolist(),
        "f": densidade(t, beta, eta).tolist(),
        "h": taxa_falha(t, beta, eta).tolist(),
        "H": taxa_acumulada(t, beta, eta).tolist(),
        "n_pontos": int(n),
        "fonte": FONTE_FORMULAS,
    }


def marcos(beta: float, eta: float) -> dict:
    """Os pontos que decidem manutenção, num só lugar."""
    q01 = quantil(0.01, beta, eta)
    q10 = quantil(0.10, beta, eta)
    q50 = quantil(0.50, beta, eta)
    media = vida_media(beta, eta)
    return {
        "q01": q01,
        "q10": q10,
        "q50": q50,
        "media": media,
        # Aliases tradicionais. O chamador decide se o eixo permite nomes de
        # vida; no experimento a_det eles sao apenas quantis de detectabilidade.
        "b1": q01,
        "b10": q10,
        "vida_mediana": q50,
        "mttf": media,
        "eta": float(eta),
        "R_em_eta": float(math.exp(-1.0)),
        "nota_eta": ("eta e a escala caracteristica: R(eta) = exp(-1) "
                     "aprox. 0,368 para qualquer beta; 63,2% dos eventos "
                     "modelados ja ocorreram"),
    }


def mediana_de_posto(n: int) -> np.ndarray:
    """Aproximação de Bernard: ``(i − 0,3)/(n + 0,4)``, para o papel de Weibull.

    Estimativa não paramétrica de F(t) na i-ésima ordem estatística. É o que
    posiciona os pontos no probability plot — o gráfico canônico da área, que
    mostra visualmente se o ajuste presta.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n deve ser positivo")
    i = np.arange(1, n + 1, dtype=float)
    return (i - 0.3) / (n + 0.4)


def posicoes_probabilidade_censuradas(
    tempos, eventos
) -> tuple[np.ndarray, np.ndarray, str]:
    """Posicoes de probabilidade pelo Kaplan-Meier modificado do NIST.

    Usa o tamanho total da amostra e devolve pontos apenas nos eventos. Isso e
    essencial quando ha censura: normalizar os postos por ``n_eventos`` faria,
    por exemplo, 12 deteccoes em 31 cenarios parecerem quase 100% da populacao.
    Eventos sao ordenados antes de censuras empatadas, a convencao usual para o
    conjunto em risco.
    """
    t = np.asarray(tempos, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    if len(t) != len(obs):
        raise ValueError("tempos e eventos devem ter o mesmo comprimento")
    if not len(t):
        return np.asarray([]), np.asarray([]), FONTE_POSICOES_CENSURADAS

    ordem = np.lexsort((~obs, t))
    t_ord, obs_ord = t[ordem], obs[ordem]
    n = len(t_ord)
    sobrevivencia = (n + 0.7) / (n + 0.4)
    pontos_t: list[float] = []
    pontos_f: list[float] = []
    for posto, (tempo, evento) in enumerate(zip(t_ord, obs_ord), start=1):
        if not evento:
            continue
        sobrevivencia *= (n - posto + 0.7) / (n - posto + 1.7)
        pontos_t.append(float(tempo))
        pontos_f.append(float(1.0 - sobrevivencia))
    return (
        np.asarray(pontos_t),
        np.asarray(pontos_f),
        FONTE_POSICOES_CENSURADAS,
    )


def diagnostico_papel_weibull(
    tempos, eventos, beta: float, eta: float
) -> dict:
    """Diagnostico descritivo do ajuste no papel de Weibull censurado.

    ``R2`` e RMSE sao triagem visual, nao testes formais de aderencia. O valor
    compara os pontos censura-aware com a reta imposta pelo MLE de dois
    parametros; pode ser negativo quando o modelo e pior que a media dos pontos.
    """
    t, f_emp, metodo = posicoes_probabilidade_censuradas(tempos, eventos)
    x, y = eixos_papel_weibull(t, f_emp)
    if len(x) < 3:
        return {
            "n_pontos": int(len(x)),
            "r2": None,
            "rmse": None,
            "metodo_posicoes": metodo,
        }
    y_ajuste = float(beta) * (x - np.log(float(eta)))
    residuos = y - y_ajuste
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residuos**2)) / ss_total if ss_total > 0 else None
    return {
        "n_pontos": int(len(x)),
        "r2": float(r2) if r2 is not None else None,
        "rmse": float(np.sqrt(np.mean(residuos**2))),
        "metodo_posicoes": metodo,
    }


def curva_kaplan_meier(
    tempos, eventos
) -> tuple[np.ndarray, np.ndarray]:
    """Curva produto-limite, preservando censura à direita."""
    t = np.asarray(tempos, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    if len(t) != len(obs):
        raise ValueError("tempos e eventos devem ter o mesmo comprimento")
    pontos_t = [0.0]
    pontos_s = [1.0]
    sobrevivencia = 1.0
    for tempo in np.unique(t):
        em_risco = int(np.sum(t >= tempo))
        n_eventos = int(np.sum((t == tempo) & obs))
        if em_risco and n_eventos:
            sobrevivencia *= 1.0 - n_eventos / em_risco
        pontos_t.append(float(tempo))
        pontos_s.append(float(sobrevivencia))
    return np.asarray(pontos_t), np.asarray(pontos_s)


def margem_restrita_km(
    atual: float,
    tempos,
    eventos,
    horizonte: float | None = None,
) -> float:
    """Margem residual média de Kaplan-Meier até o horizonte observado."""
    t = np.asarray(tempos, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    if len(t) == 0 or len(t) != len(obs):
        return float("nan")

    tau = float(np.max(t) if horizonte is None else horizonte)
    t0 = float(max(atual, 0.0))
    if not np.isfinite(tau) or t0 >= tau:
        return 0.0

    sobrevivencia = 1.0
    inicio = 0.0
    area = 0.0
    sobrevivencia_t0: float | None = None
    for tempo in np.unique(t):
        fim = min(float(tempo), tau)
        if fim > inicio:
            if inicio <= t0 < fim:
                sobrevivencia_t0 = sobrevivencia
            area += sobrevivencia * max(fim - max(inicio, t0), 0.0)
        if tempo >= tau:
            inicio = tau
            break
        em_risco = int(np.sum(t >= tempo))
        n_eventos = int(np.sum((t == tempo) & obs))
        if em_risco > 0:
            sobrevivencia *= 1.0 - n_eventos / em_risco
        inicio = float(tempo)

    if inicio < tau:
        if inicio <= t0 < tau:
            sobrevivencia_t0 = sobrevivencia
        area += sobrevivencia * max(tau - max(inicio, t0), 0.0)
    if sobrevivencia_t0 is None:
        sobrevivencia_t0 = sobrevivencia
    if sobrevivencia_t0 <= 0:
        return 0.0
    return float(max(area / sobrevivencia_t0, 0.0))


def eixos_papel_weibull(t, f_emp) -> tuple[np.ndarray, np.ndarray]:
    """Linearização do papel de Weibull: ``x = ln t``, ``y = ln(−ln(1−F))``.

    Nessa escala a Weibull vira **reta** de inclinação β. Desvio sistemático da
    reta é evidência de que a família não serve — informação que o RMSE contra
    Kaplan-Meier, sozinho, não dá.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f_emp, dtype=float)
    ok = (t > 0) & (f > 0) & (f < 1)
    return np.log(t[ok]), np.log(-np.log1p(-f[ok]))
