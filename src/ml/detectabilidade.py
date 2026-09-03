"""Em que magnitude cada modelo passa a detectar: `a_det`, POD e Weibull.

POR QUE ESTA ETAPA EXISTE
=========================
A E3 diz se o modelo detecta a falha. Não diz a partir de QUE magnitude. Esta
etapa varre a severidade injetada e registra, por trajetória, o menor `a` em
que a detecção se confirma. É o segundo eixo da comparação entre os dois
autoencoders: não "quantas falhas cada um pega", mas "quão cedo".

O EIXO NÃO É TEMPO
==================
`a_det` é fração da assinatura nominal. Não é hora, ciclo, nem vida
consumida. A Weibull ajustada aqui descreve a dispersão da MAGNITUDE de
detecção entre trajetórias — não confiabilidade, não taxa de falha, não RUL.
Os símbolos coincidem com os da confiabilidade física e as grandezas não;
`docs/nomenclatura_deteccao.md` mantém a separação.

A CONFIRMAÇÃO
=============
Um único cruzamento do limiar não é detecção: com escore ruidoso perto do
limiar, o primeiro cruzamento é sorteio, e `a_det` viraria uma medida da
variância do ruído. Exige-se uma sequência de `n` pontos consecutivos acima do
limiar, e `a_det` é o `a` do PRIMEIRO ponto dessa sequência.

CENSURA À DIREITA
=================
Trajetória que não cruza até `a=1` não tem `a_det`: ela é censurada, não é
"a_det = 1". Tratá-la como detectada em 1,0 puxaria qualquer estatística para
baixo e inventaria detecções que não houve. A censura entra no ajuste e
limita quais percentis empíricos podem ser lidos.

O QUE NÃO SE PUBLICA
====================
Percentil paramétrico de ajuste rejeitado. Uma execução anterior publicou o
`a10` da Weibull 2P na coluna vizinha a "2P adotada: não" — o número existia,
o ajuste que o produziu tinha sido reprovado. Aqui o percentil empírico vem
primeiro e sempre; o paramétrico só acompanha quando o ajuste é adotado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Pontos consecutivos acima do limiar para a detecção contar.
CONFIRMACOES_PADRAO = 3

# Critérios de adoção do resumo paramétrico. Os três precisam valer.
MINIMO_DETECTADAS = 10
MINIMO_R2_PAPEL = 0.90
MINIMO_VALORES_DISTINTOS = 4


@dataclass(frozen=True)
class Detectabilidade:
    """Resultado da varredura de um modelo sobre um modo de injeção."""

    modelo: str
    injecao: str
    a_dets: np.ndarray
    eventos: np.ndarray = field(repr=False)
    n_trajetorias: int
    grade: tuple[float, ...]

    @property
    def n_detectadas(self) -> int:
        return int(np.count_nonzero(self.eventos))

    @property
    def fracao_detectada(self) -> float:
        return self.n_detectadas / self.n_trajetorias if self.n_trajetorias else 0.0

    @property
    def censura_presente(self) -> bool:
        return self.n_detectadas < self.n_trajetorias


def varrer_trajetoria(
    scorer,
    janela,
    injetar,
    limiar: float,
    grade,
    *,
    confirmacoes: int = CONFIRMACOES_PADRAO,
) -> float | None:
    """Menor `a` cuja detecção se confirma; ``None`` se nunca confirma.

    Pontua a grade inteira de uma vez: o scorer recebe uma lista, e chamá-lo
    por ponto desperdiçaria o lote. Devolve o `a` do primeiro ponto de uma
    sequência de `confirmacoes` cruzamentos consecutivos.
    """
    valores = tuple(float(a) for a in grade)
    if not valores:
        raise ValueError("A grade de severidade está vazia")
    if confirmacoes < 1:
        raise ValueError("A confirmação exige ao menos um ponto")
    if confirmacoes > len(valores):
        raise ValueError(
            f"A grade tem {len(valores)} pontos e a confirmação pede "
            f"{confirmacoes}: nenhuma detecção seria possível"
        )

    escores = np.asarray(scorer([injetar(janela, a) for a in valores]), dtype=float)
    if escores.shape != (len(valores),):
        raise ValueError(
            f"O scorer devolveu {escores.shape} para {len(valores)} janelas"
        )

    acima = escores > float(limiar)
    corridos = 0
    for indice, cruzou in enumerate(acima):
        corridos = corridos + 1 if cruzou else 0
        if corridos >= confirmacoes:
            return valores[indice - confirmacoes + 1]
    return None


def detectabilidade_do_modelo(
    modelo: str,
    injecao: str,
    scorer,
    janelas,
    injetar,
    limiar: float,
    grade,
    *,
    confirmacoes: int = CONFIRMACOES_PADRAO,
) -> Detectabilidade:
    """Varre todas as trajetórias e devolve `a_det` com marca de censura."""
    if not janelas:
        raise ValueError("A varredura exige ao menos uma trajetória")
    achados = [
        varrer_trajetoria(
            scorer, janela, injetar, limiar, grade, confirmacoes=confirmacoes
        )
        for janela in janelas
    ]
    eventos = np.asarray([achado is not None for achado in achados], dtype=bool)
    # Censurada entra como 1,0 SÓ para o ajuste saber até onde observou; o
    # vetor de eventos é o que impede que isso vire uma detecção.
    a_dets = np.asarray(
        [achado if achado is not None else 1.0 for achado in achados], dtype=float
    )
    return Detectabilidade(
        modelo=modelo,
        injecao=injecao,
        a_dets=a_dets,
        eventos=eventos,
        n_trajetorias=len(janelas),
        grade=tuple(float(a) for a in grade),
    )


# ── percentis empíricos, com o limite que a censura impõe ──────────────────

def percentis_empiricos(
    resultado: Detectabilidade,
    quantis: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> dict[str, float | None]:
    """Percentis lidos dos dados, não de um ajuste.

    Devolve ``None`` para o quantil que a fração detectada não alcança: com
    62% das trajetórias detectadas não existe `a90` observado, e inventá-lo a
    partir do ajuste seria publicar extrapolação como medida.
    """
    detectados = np.sort(resultado.a_dets[resultado.eventos])
    fracao = resultado.fracao_detectada
    saida: dict[str, float | None] = {}
    for quantil in quantis:
        rotulo = f"a{int(round(quantil * 100))}_empirico"
        if len(detectados) == 0 or quantil > fracao:
            saida[rotulo] = None
            continue
        # Posição dentro da amostra DETECTADA que corresponde ao quantil da
        # amostra inteira: com censura, o quantil q global é o quantil
        # q/fracao entre os detectados.
        posicao = min(quantil / fracao, 1.0)
        saida[rotulo] = float(np.quantile(detectados, posicao))
    return saida


def curva_pod(resultado: Detectabilidade) -> dict[str, list[float]]:
    """POD monótona: fração de trajetórias já detectadas até cada `a`."""
    grade = np.asarray(resultado.grade, dtype=float)
    detectados = resultado.a_dets[resultado.eventos]
    fracoes = [
        float(np.count_nonzero(detectados <= a) / resultado.n_trajetorias)
        for a in grade
    ]
    return {"a": grade.tolist(), "pod": fracoes}


# ── Weibull 2P com censura à direita ───────────────────────────────────────

def _log_verossimilhanca_forma(beta: float, detectados, censurados) -> float:
    """Derivada da log-verossimilhança em relação a beta, com censura.

    Zerar esta função é a condição de máximo do MLE da Weibull 2P. Escrevê-la
    diretamente evita otimizar em duas dimensões: dado beta, eta tem forma
    fechada.
    """
    todos = np.concatenate([detectados, censurados]) if len(censurados) else detectados
    potencias = todos**beta
    soma_potencias = float(np.sum(potencias))
    if soma_potencias <= 0.0:
        return float("nan")
    termo = float(np.sum(potencias * np.log(todos))) / soma_potencias
    return termo - 1.0 / beta - float(np.mean(np.log(detectados)))


def ajustar_weibull_2p(resultado: Detectabilidade) -> dict:
    """MLE da Weibull 2P sobre `a_det`, respeitando a censura à direita.

    Devolve sempre `convergiu` e `adotado`. `adotado` é o portão: enquanto ele
    for falso, nenhum percentil paramétrico pode ser publicado.
    """
    detectados = resultado.a_dets[resultado.eventos]
    censurados = resultado.a_dets[~resultado.eventos]
    base = {
        "beta": None,
        "eta": None,
        "convergiu": False,
        "adotado": False,
        "r2_papel": None,
        "n_detectadas": resultado.n_detectadas,
        "n_censuradas": int(len(censurados)),
        "valores_distintos": int(len(np.unique(detectados))),
        "motivo_da_rejeicao": None,
    }

    if len(detectados) < 2:
        base["motivo_da_rejeicao"] = "menos de duas trajetórias detectadas"
        return base
    if np.any(detectados <= 0.0):
        base["motivo_da_rejeicao"] = "a_det não positivo"
        return base

    try:
        from scipy.optimize import brentq
    except ModuleNotFoundError:  # pragma: no cover - scipy é dependência fixa
        base["motivo_da_rejeicao"] = "scipy indisponível"
        return base

    def equacao(beta: float) -> float:
        return _log_verossimilhanca_forma(beta, detectados, censurados)

    try:
        beta = float(brentq(equacao, 0.05, 100.0, xtol=1e-10))
    except (ValueError, RuntimeError):
        base["motivo_da_rejeicao"] = "MLE não convergiu no intervalo de beta"
        return base

    todos = (
        np.concatenate([detectados, censurados]) if len(censurados) else detectados
    )
    eta = float((np.sum(todos**beta) / len(detectados)) ** (1.0 / beta))
    base.update({"beta": beta, "eta": eta, "convergiu": True})

    r2 = _r2_papel_weibull(detectados, censurados)
    base["r2_papel"] = r2

    motivos = []
    if resultado.n_detectadas < MINIMO_DETECTADAS:
        motivos.append(
            f"apenas {resultado.n_detectadas} detecções (mínimo {MINIMO_DETECTADAS})"
        )
    if base["valores_distintos"] < MINIMO_VALORES_DISTINTOS:
        motivos.append(
            f"a_det assume só {base['valores_distintos']} valores distintos: o "
            "ajuste descreveria a grade de severidade, não a dispersão"
        )
    if r2 is None or r2 < MINIMO_R2_PAPEL:
        motivos.append(
            f"R² do papel de Weibull {r2 if r2 is None else round(r2, 4)} "
            f"abaixo de {MINIMO_R2_PAPEL}"
        )
    if motivos:
        base["motivo_da_rejeicao"] = "; ".join(motivos)
        return base

    base["adotado"] = True
    return base


def _r2_papel_weibull(detectados: np.ndarray, censurados: np.ndarray) -> float | None:
    """Aderência no papel de Weibull, com posto ajustado para censura.

    Usa o ajuste de Johnson sobre a ordem combinada e as posições medianas de
    Benard. É a mesma leitura que se faz à mão no papel de Weibull, e cai
    quando os pontos deixam de ser uma reta.
    """
    n = len(detectados) + len(censurados)
    if len(np.unique(detectados)) < 3:
        return None

    marcados = [(valor, True) for valor in detectados]
    marcados += [(valor, False) for valor in censurados]
    marcados.sort(key=lambda par: (par[0], not par[1]))

    posto_anterior = 0.0
    postos: list[float] = []
    valores: list[float] = []
    for indice, (valor, detectado) in enumerate(marcados):
        if not detectado:
            continue
        incremento = (n + 1.0 - posto_anterior) / (1.0 + (n - indice))
        posto = posto_anterior + incremento
        posto_anterior = posto
        postos.append(posto)
        valores.append(valor)

    if len(valores) < 3:
        return None

    postos_array = np.asarray(postos, dtype=float)
    mediana_benard = (postos_array - 0.3) / (n + 0.4)
    dentro = (mediana_benard > 0.0) & (mediana_benard < 1.0)
    if np.count_nonzero(dentro) < 3:
        return None

    x = np.log(np.asarray(valores, dtype=float)[dentro])
    y = np.log(-np.log(1.0 - mediana_benard[dentro]))
    if np.ptp(x) <= 0.0:
        return None

    coeficientes = np.polyfit(x, y, 1)
    residuos = y - np.polyval(coeficientes, x)
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return None
    return float(1.0 - float(np.sum(residuos**2)) / total)


def percentil_parametrico(ajuste: dict, quantil: float) -> float | None:
    """Percentil da Weibull ajustada — ``None`` enquanto o ajuste não for adotado.

    O portão é aqui, e não no chamador, porque foi um chamador que publicou o
    `a10` de um ajuste rejeitado.
    """
    if not ajuste.get("adotado"):
        return None
    beta, eta = ajuste["beta"], ajuste["eta"]
    return float(eta * (-math.log(1.0 - quantil)) ** (1.0 / beta))


def resumo(resultado: Detectabilidade) -> dict:
    """Linha publicável: empírico primeiro, paramétrico só se adotado."""
    ajuste = ajustar_weibull_2p(resultado)
    linha = {
        "modelo": resultado.modelo,
        "injecao": resultado.injecao,
        "n_trajetorias": resultado.n_trajetorias,
        "n_detectadas": resultado.n_detectadas,
        "fracao_detectada": resultado.fracao_detectada,
        "censura_presente": resultado.censura_presente,
        "eixo_nao_e_tempo": True,
        "evidence_level": "E2",
        **percentis_empiricos(resultado),
        "weibull_2p_adotada": bool(ajuste["adotado"]),
        "weibull_2p_convergiu": bool(ajuste["convergiu"]),
        "weibull_2p_beta": ajuste["beta"],
        "weibull_2p_eta": ajuste["eta"],
        "weibull_2p_r2_papel": ajuste["r2_papel"],
        "weibull_2p_motivo_da_rejeicao": ajuste["motivo_da_rejeicao"],
        "a10_parametrico": percentil_parametrico(ajuste, 0.10),
        "a50_parametrico": percentil_parametrico(ajuste, 0.50),
    }
    return linha


__all__ = [
    "CONFIRMACOES_PADRAO",
    "Detectabilidade",
    "MINIMO_DETECTADAS",
    "MINIMO_R2_PAPEL",
    "MINIMO_VALORES_DISTINTOS",
    "ajustar_weibull_2p",
    "curva_pod",
    "detectabilidade_do_modelo",
    "percentil_parametrico",
    "percentis_empiricos",
    "resumo",
    "varrer_trajetoria",
]
