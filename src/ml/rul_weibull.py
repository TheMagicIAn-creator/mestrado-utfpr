"""Weibull 2P exploratória da magnitude de primeiro cruzamento E2.

Cada trajetória mantém uma janela saudável F0 e aumenta a assinatura sintética
de 0 a 1. `a_det` é a primeira magnitude com excedência persistente do limiar.
O eixo não é tempo: beta é adimensional, e eta, média e a10 são frações da
assinatura nominal. MTTF, B10 e RUL existem somente como aliases legados.

`S_D(a)` descreve não detecção e `h_D(a)` a intensidade de primeiro cruzamento;
nenhuma das duas representa confiabilidade ou taxa de falha física. F1-F7 não
entram nesta etapa e permanecem reservados à validação experimental E3.
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError as _erro:  # execução direta: python src/ml/<arquivo>.py
    # Só trata a ausência do PACOTE `src` (rodar o arquivo direto, sem a raiz no
    # sys.path). Qualquer outra dependência faltando é repassada: reimportar não
    # a faria aparecer, e o retry produzia um traceback DUPLO com a causa real
    # ("No module named 'dotenv'") enterrada no meio — foi o que aconteceu com o
    # venv desativado em 15/08/2026.
    if (_erro.name or "").split(".")[0] != "src":
        raise
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("rul_weibull")
_log = _adaptar_log(_logger)


import json
import pickle
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.ml.estilo_graficos import (
    COR_ALERTA, COR_TEXTO_SEC, TAM, aplicar_estilo, salvar_figura,
)

aplicar_estilo()
from pathlib import Path
from scipy.stats import weibull_min
from scipy.special import gamma as gamma_func, gammaincc
from scipy.optimize import minimize
from typing import TYPE_CHECKING

from src.ml.gpvs_principal import (
    carregar_normalizacao_baseline,
    extrair_janela,
    JANELA,
    FS,
    normalizar_vetores_f0,
    preparar_janelas_holdout,
)
from src.ml.injecao_falhas import (
    FUNCOES_FALHA, FALHAS,
)
from src.ml.confiabilidade import curva_kaplan_meier, margem_restrita_km

# Alias histórico: a grandeza é margem residual em magnitude, não RUL.
rul_restrita_km = margem_restrita_km

if TYPE_CHECKING:
    import torch

    from src.ml.autoencoder import Autoencoder

# ── Caminhos ─────────────────────────────────────────────────
RAIZ        = Path(__file__).parent.parent.parent
PASTA_AE    = RAIZ / "resultados" / "autoencoder"

# ── Parâmetros de simulação ───────────────────────────────────
# ``None`` usa todo o holdout elegível. O teto histórico de 100 pegava as
# primeiras janelas ordenadas e, no GPVS atual, selecionava somente F0L.
N_TRAJ: int | None = None
N_STEPS = 501     # Δa=0,002; resolução principal da magnitude a_inj em [0, 1]
N_STEPS_SENSIBILIDADE = (101, 251, 501)

# ── O EIXO NÃO É TEMPO ──────────────────────────────────────────────────────
# Até 08/08/2026 este módulo chamava o eixo de TTF (time to failure) e a unidade
# de "passo sintético de degradação". Os dois nomes prometiam tempo e entregavam
# outra coisa: o que a trajetória varre é a MAGNITUDE DA ASSINATURA INJETADA,
# de 0 a 1,0, e o que se registra é a magnitude em que a detecção se confirma.
# Não há taxa de degradação de campo que converta magnitude em hora — converter
# seria inventar o número mais importante da seção.
#
# O eixo passa a se chamar `a_det`, na mesma família de `a_inj` (a magnitude
# injetada, src/ml/injecao_falhas.py) e do tamanho de defeito `a` da curva
# POD(a) do MIL-HDBK-1823A. Ganho concreto: `a_det` e a SMD passam a estar na
# MESMA unidade, então beta/eta/a10 do Weibull e a SMD podem ser lidos na
# mesma régua. Em passos isso era impossível.
#
# Leitura de a10 = 0,12: em 10% das trajetórias a falha já é detectada com 12%
# da assinatura nominal. Antes se lia "B10 = 14 passos", que não significa nada
# fora do experimento.
A_DET_UNIDADE = "a_det_fracao_da_assinatura_nominal"
A_DET_MIN = 0.0
A_DET_MAX = 1.0
# Alias de compatibilidade para as chaves antigas do JSON. Repetido como
# LITERAL de propósito: o manifesto de proveniência lê estas constantes por AST,
# sem importar o módulo (pipeline._parametros_do_fonte), e `literal_eval` não
# resolve referência a outro nome — escrever `= A_DET_UNIDADE` forçaria o
# import pesado no caminho de leitura. Um teste garante que os dois batem.
TTF_UNIDADE = "a_det_fracao_da_assinatura_nominal"
TEMPO_FISICO_CALIBRADO = False
TEMPO_FISICO_NOTA = (
    "O eixo do Weibull é a_det: a fração da assinatura nominal (a_inj) em que a "
    "detecção se confirma, em [0; 1]. NÃO é tempo. A janela de aquisição tem "
    "duração física conhecida, mas o avanço de magnitude não tem taxa de campo "
    "calibrada; portanto não converter para horas, dias ou anos. β é "
    "adimensional; eta, media(a_det) e a10 estão em fração de assinatura; "
    "MTTF/B10/RUL são apenas aliases legados."
)
BATCH_INFERENCIA = 16
N_BOOTSTRAP = 1000
N_BOOTSTRAP_ADERENCIA = 250
N_BOOTSTRAP_MODO = 500
MIN_EVENTOS_WEIBULL = 10
MAX_CENSURA_RUL_PCT = 50.0
MIN_R2_PAPEL_WEIBULL = 0.90
MIN_NIVEIS_ADERENCIA = 8
ALFA_ADERENCIA = 0.05
MAX_VARIACAO_RELATIVA_GRADE = 0.10
# A persistência é uma largura no eixo físico do experimento, não uma contagem
# de pontos. Assim, refinar a grade não muda a definição do detector.
PERSISTENCIA_MAGNITUDE = 0.02
# Alias legado: calculado para a grade principal e mantido nos manifestos.
PERSISTENCIA_CRUZAMENTO = 11
AJUSTE_WEIBULL_METODO = "mle_interval_censored_grid_right_censored"


def selecionar_trajetorias_holdout(
    janelas: list[pd.DataFrame],
    n_max: int | None = N_TRAJ,
) -> list[pd.DataFrame]:
    """Seleciona trajetórias com cobertura determinística de cada ensaio F0.

    O caminho canônico usa todas as janelas elegíveis. Se um teto for aplicado
    em uma execução exploratória, a seleção é uniforme dentro de cada ensaio e
    a alocação é proporcional, evitando o viés histórico das primeiras linhas.
    """
    if n_max is None or len(janelas) <= n_max:
        return list(janelas)
    if n_max <= 0:
        raise ValueError("n_max deve ser positivo ou None")

    grupos: dict[str, list[pd.DataFrame]] = {}
    for janela in janelas:
        ensaio = str(janela.attrs.get("ensaio", "sem_ensaio"))
        grupos.setdefault(ensaio, []).append(janela)

    total = len(janelas)
    alocacao = {
        ensaio: int(np.floor(n_max * len(grupo) / total))
        for ensaio, grupo in grupos.items()
    }
    for ensaio in sorted(grupos):
        if alocacao[ensaio] == 0 and n_max >= len(grupos):
            alocacao[ensaio] = 1
    while sum(alocacao.values()) < n_max:
        ensaio = max(
            grupos,
            key=lambda nome: (
                n_max * len(grupos[nome]) / total - alocacao[nome],
                len(grupos[nome]) - alocacao[nome],
                nome,
            ),
        )
        if alocacao[ensaio] >= len(grupos[ensaio]):
            break
        alocacao[ensaio] += 1

    selecionadas: list[pd.DataFrame] = []
    for ensaio in sorted(grupos):
        grupo = grupos[ensaio]
        quantidade = min(alocacao[ensaio], len(grupo))
        if quantidade <= 0:
            continue
        posicoes = np.linspace(0, len(grupo) - 1, quantidade).round().astype(int)
        selecionadas.extend(grupo[int(pos)] for pos in np.unique(posicoes))
    return selecionadas


def metadados_tempo_rul() -> dict:
    """Metadados que impedem ler a magnitude de injeção como tempo físico."""
    return {
        "a_det_unidade": A_DET_UNIDADE,
        "a_det_intervalo": [A_DET_MIN, A_DET_MAX],
        "a_det_passos_da_grade": N_STEPS,
        "a_det_por_passo": 1.0 / (N_STEPS - 1),
        # Aliases mantidos para não quebrar leitores antigos do JSON. Apontam
        # para a MESMA unidade — que agora é magnitude, não passo de tempo.
        "ttf_unidade": A_DET_UNIDADE,
        "rul_unidade": A_DET_UNIDADE,
        "eixo": "magnitude_da_assinatura_injetada",
        "eixo_nao_e_tempo": True,
        "grandeza_primaria": "magnitude_primeiro_cruzamento_detector",
        "rul_fisica_disponivel": False,
        "confiabilidade_fisica_disponivel": False,
        "tempo_fisico_calibrado": TEMPO_FISICO_CALIBRADO,
        "passo_tempo_fisico_horas": None,
        "fs_hz": FS,
        "janela_amostras": JANELA,
        "janela_aquisicao_s": float(JANELA / FS),
        "nota": TEMPO_FISICO_NOTA,
    }


def _json_seguro(valor):
    """Converte NaN/inf em null para manter o relatório JSON estrito."""
    if isinstance(valor, dict):
        return {k: _json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_json_seguro(v) for v in valor]
    if isinstance(valor, (float, np.floating)) and not np.isfinite(valor):
        return None
    return valor


# ============================================================
# VARREDURA DE MAGNITUDE — extraída para src/ml/varredura_a_det.py
# ============================================================
# A varredura produz o dado (`a_det`); este módulo o MODELA (Weibull). Eram a
# mesma coisa até 15/08/2026, quando a varredura ganhou um segundo consumidor
# (`weibull_por_modelo`, que a usa para qualquer detector) e o módulo passou de
# mil linhas. Reexportado para não quebrar quem já importava daqui.
from src.ml.varredura_a_det import (  # noqa: E402
    a_det_da_grade,
    calcular_erros_batch,
    gerar_a_det,
    gerar_ttf,
    passos_persistencia,
    selecionar_janelas_baseline_normais,
)


# ============================================================


def classificar_desfechos(a_dets: np.ndarray, eventos: np.ndarray) -> dict:
    """Separa INDETECTABILIDADE NO TETO de censura à direita genuína.

    O módulo chamava toda não detecção de "censura à direita" e reportava
    `censura_pct`. O nome é enganoso, e a diferença é metodológica, não de
    vocabulário:

    - **Censura à direita genuína** é acompanhamento interrompido: o evento
      ocorreria depois, mas paramos de observar. É o que o MLE censurado
      pressupõe, e sob essa hipótese extrapolar a Weibull além do último dado é
      legítimo.
    - **Indetectabilidade no teto** é outra coisa: a grade de magnitude foi
      varrida INTEIRA, até `a_inj = 1,0`, e o detector não confirmou. Não há
      "depois" a observar dentro do experimento. Tratar isso como censura só é
      admissível se admitirmos que a falha real pode ter assinatura MAIOR que a
      nominal — hipótese defensável, mas hipótese, e que precisa estar escrita.

    No desenho atual toda não detecção é do segundo tipo. A função devolve as
    duas contagens separadas para que o artefato não continue chamando as duas
    coisas pelo mesmo nome, e para expor `pod_mon_no_teto` — a probabilidade de
    detecção na magnitude máxima, que é o elo direto com a curva POD e com o
    `D_mon` da retroalimentação da FMECA.
    """
    a = np.asarray(a_dets, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    if len(a) != len(obs):
        raise ValueError("a_dets e eventos devem ter o mesmo comprimento.")

    n = len(a)
    no_teto = (~obs) & np.isclose(a, A_DET_MAX)
    censura_genuina = (~obs) & ~no_teto

    return {
        "n_traj": int(n),
        "n_detectadas": int(obs.sum()),
        "n_indetectaveis_no_teto": int(no_teto.sum()),
        "n_censura_genuina": int(censura_genuina.sum()),
        "pod_mon_no_teto": float(obs.mean()) if n else float("nan"),
        "indetectabilidade_pct": float(no_teto.mean() * 100.0) if n else float("nan"),
        "tratamento_no_ajuste": "right_censored",
        "hipotese_declarada": (
            "As trajetórias não detectadas são tratadas como censura à direita "
            "em a_inj = 1,0. Isso PRESSUPÕE que a assinatura real possa exceder "
            "a nominal; dentro da grade varrida elas são simplesmente NÃO "
            "DETECTADAS. Sem essa hipótese, o Weibull descreve apenas a "
            "subpopulação detectável; qualquer resumo paramétrico é condicional."
        ),
    }


def _ajuste_weibull_censurado(
    ttfs: np.ndarray,
    eventos: np.ndarray,
    passo_grade: float | None = None,
) -> tuple[float, float, bool]:
    """MLE 2P: cruzamentos por célula da grade e censura à direita no horizonte."""
    tempos = np.asarray(ttfs, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    tempos = np.clip(tempos, 1e-12, None)
    if int(obs.sum()) < MIN_EVENTOS_WEIBULL:
        return float("nan"), float("nan"), False

    delta_a = float(1.0 / (N_STEPS - 1) if passo_grade is None else passo_grade)
    if not np.isfinite(delta_a) or delta_a <= 0:
        raise ValueError("passo_grade deve ser positivo e finito")

    def neg_log_likelihood(log_params: np.ndarray) -> float:
        beta, eta = np.exp(log_params)
        ll = 0.0
        if obs.any():
            direita = tempos[obs]
            esquerda = np.maximum(0.0, direita - delta_a)
            z_direita = np.power(direita / eta, beta)
            z_esquerda = np.power(esquerda / eta, beta)
            diferenca = np.maximum(z_direita - z_esquerda, np.finfo(float).tiny)
            # log(S(esquerda) - S(direita)), sem cancelamento numérico.
            log_massa_intervalo = -z_esquerda + np.log(-np.expm1(-diferenca))
            ll += float(np.sum(log_massa_intervalo))
        if (~obs).any():
            z_censura = np.power(tempos[~obs] / eta, beta)
            ll += float(np.sum(-z_censura))
        return float(-ll) if np.isfinite(ll) else 1e30

    # O chute de eta acompanha a escala a_det, em vez do antigo eixo em passos.
    observados = tempos[obs]
    beta_ini = 2.0
    eta_ini = max(float(np.median(observados)), 1e-4)
    ajuste = minimize(
        neg_log_likelihood,
        x0=np.log([beta_ini, eta_ini]),
        method="L-BFGS-B",
        bounds=[(np.log(0.05), np.log(50.0)), (np.log(1e-6), np.log(1e5))],
    )
    beta, eta = np.exp(ajuste.x)
    convergiu = bool(ajuste.success and np.isfinite(beta) and np.isfinite(eta))
    return float(beta), float(eta), convergiu


def teste_aderencia_weibull_quantizada(
    tempos: np.ndarray,
    eventos: np.ndarray,
    beta: float,
    eta: float,
    *,
    passo_grade: float,
    n_boot: int = N_BOOTSTRAP_ADERENCIA,
    seed: int = 20260812,
) -> dict:
    """Bootstrap paramétrico de aderência compatível com a grade observada.

    A estatística compara as posições empíricas agrupadas à CDF ajustada. Cada
    réplica nasce da Weibull 2P, é quantizada para o extremo direito da mesma
    grade, recebe a mesma regra de horizonte e é reajustada. O p-valor, assim,
    não pune o modelo apenas pelos empates produzidos pelo experimento.
    """
    from src.ml.confiabilidade import posicoes_probabilidade_censuradas

    t = np.asarray(tempos, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    t_emp, f_emp, metodo = posicoes_probabilidade_censuradas(t, obs)
    if len(t_emp) < 3 or n_boot <= 0:
        return {
            "metodo": "bootstrap_parametrico_cdf_quantizada",
            "estatistica": None,
            "p_value": None,
            "bootstrap_solicitados": int(max(n_boot, 0)),
            "bootstrap_validos": 0,
            "n_niveis": int(len(t_emp)),
            "posicoes_empiricas": metodo,
        }

    f_ajustada = weibull_min.cdf(t_emp, beta, loc=0, scale=eta)
    estatistica = float(np.mean((f_emp - f_ajustada) ** 2))
    rng = np.random.default_rng(seed)
    horizonte = float(np.max(t))
    estatisticas_boot: list[float] = []
    for _ in range(int(n_boot)):
        simulados_continuos = eta * rng.weibull(beta, size=len(t))
        eventos_sim = simulados_continuos <= horizonte
        simulados = np.ceil(simulados_continuos / passo_grade) * passo_grade
        simulados = np.clip(simulados, passo_grade, horizonte)
        beta_b, eta_b, ok = _ajuste_weibull_censurado(
            simulados, eventos_sim, passo_grade=passo_grade
        )
        if not ok:
            continue
        tb, fb, _ = posicoes_probabilidade_censuradas(simulados, eventos_sim)
        if len(tb) < 3:
            continue
        ajuste_b = weibull_min.cdf(tb, beta_b, loc=0, scale=eta_b)
        estatisticas_boot.append(float(np.mean((fb - ajuste_b) ** 2)))

    p_value = None
    if estatisticas_boot:
        p_value = float(
            (1 + np.sum(np.asarray(estatisticas_boot) >= estatistica))
            / (1 + len(estatisticas_boot))
        )
    return {
        "metodo": "bootstrap_parametrico_cdf_quantizada",
        "estatistica": estatistica,
        "p_value": p_value,
        "bootstrap_solicitados": int(n_boot),
        "bootstrap_validos": len(estatisticas_boot),
        "n_niveis": int(len(t_emp)),
        "posicoes_empiricas": metodo,
    }


def ajustar_weibull(
    ttfs: np.ndarray,
    eventos: np.ndarray | None = None,
    n_boot: int = 250,
    seed: int = 42,
    passo_grade: float | None = None,
    n_boot_aderencia: int = 0,
) -> dict:
    """Ajusta Weibull intervalar e estima incerteza por bootstrap."""
    tempos = np.asarray(ttfs, dtype=float)
    obs = (
        np.ones(len(tempos), dtype=bool)
        if eventos is None else np.asarray(eventos, dtype=bool)
    )
    if len(tempos) != len(obs):
        raise ValueError("ttfs e eventos devem ter o mesmo comprimento.")

    delta_a = float(
        1.0 / (N_STEPS - 1) if passo_grade is None else passo_grade
    )
    beta, eta, convergiu = _ajuste_weibull_censurado(
        tempos, obs, passo_grade=delta_a
    )
    if convergiu:
        mttf = eta * gamma_func(1 + 1 / beta)
        b10 = eta * (-np.log(0.90)) ** (1 / beta)
        km_t, km_s = curva_kaplan_meier(tempos, obs)
        weibull_s = weibull_min.sf(km_t, beta, loc=0, scale=eta)
        km_rmse = float(np.sqrt(np.mean((km_s - weibull_s) ** 2)))
        from src.ml import confiabilidade as cf

        diagnostico_papel = cf.diagnostico_papel_weibull(
            tempos, obs, beta, eta
        )
        teste_aderencia = teste_aderencia_weibull_quantizada(
            tempos,
            obs,
            beta,
            eta,
            passo_grade=delta_a,
            n_boot=n_boot_aderencia,
            seed=seed + 10_000,
        )
    else:
        mttf = b10 = km_rmse = float("nan")
        diagnostico_papel = {
            "n_pontos": 0,
            "n_eventos": int(obs.sum()),
            "n_niveis_distintos": int(np.unique(tempos[obs]).size),
            "r2": None, "rmse": None,
            "metodo_posicoes": None,
        }
        teste_aderencia = {
            "metodo": "bootstrap_parametrico_cdf_quantizada",
            "estatistica": None,
            "p_value": None,
            "bootstrap_solicitados": int(max(n_boot_aderencia, 0)),
            "bootstrap_validos": 0,
            "n_niveis": 0,
            "posicoes_empiricas": None,
        }

    amostras_boot: list[tuple[float, float, float, float]] = []
    if convergiu and n_boot > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, len(tempos), size=len(tempos))
            b, e, ok = _ajuste_weibull_censurado(
                tempos[idx], obs[idx], passo_grade=delta_a
            )
            if ok:
                amostras_boot.append((
                    b,
                    e,
                    e * gamma_func(1 + 1 / b),
                    e * (-np.log(0.90)) ** (1 / b),
                ))

    nomes = ("beta", "eta", "mttf", "b10")
    cis = {}
    if amostras_boot:
        matriz = np.asarray(amostras_boot)
        for i, nome in enumerate(nomes):
            cis[f"{nome}_ci95"] = [
                float(np.percentile(matriz[:, i], 2.5)),
                float(np.percentile(matriz[:, i], 97.5)),
            ]
    else:
        cis = {f"{nome}_ci95": [None, None] for nome in nomes}

    censura_pct = float((~obs).mean() * 100.0)
    horizonte = float(np.max(tempos)) if len(tempos) else 0.0
    rul_restrita_inicial = rul_restrita_km(0.0, tempos, obs, horizonte)
    alta_censura = censura_pct > MAX_CENSURA_RUL_PCT
    desfechos = classificar_desfechos(tempos, obs)
    r2_papel = diagnostico_papel.get("r2")
    triagem_compativel = bool(
        convergiu and r2_papel is not None
        and r2_papel >= MIN_R2_PAPEL_WEIBULL
    )
    p_aderencia = teste_aderencia.get("p_value")
    niveis_suficientes = int(np.unique(tempos[obs]).size) >= MIN_NIVEIS_ADERENCIA
    aderencia_aceitavel = (
        bool(p_aderencia >= ALFA_ADERENCIA)
        if p_aderencia is not None else triagem_compativel
    )
    resumo_parametrico_recomendado = bool(
        convergiu and not alta_censura and niveis_suficientes
        and aderencia_aceitavel
    )
    taxa_bootstrap = len(amostras_boot) / n_boot if n_boot > 0 else None
    n_niveis_distintos = int(np.unique(tempos[obs]).size)
    taxa_empates = (
        1.0 - n_niveis_distintos / int(obs.sum()) if obs.any() else None
    )
    status_aderencia = (
        "nao_estimavel"
        if not convergiu else
        "resolucao_insuficiente"
        if not niveis_suficientes else
        "nao_testado_formalmente_triagem_visual_compativel"
        if p_aderencia is None and triagem_compativel else
        "nao_testado_formalmente_triagem_visual_incompativel"
        if p_aderencia is None else
        "compativel_bootstrap_quantizado"
        if aderencia_aceitavel else
        "desvio_detectado_bootstrap_quantizado"
    )

    return {
        "beta": float(beta),
        "eta": float(eta),
        "mttf": float(mttf),
        "b10": float(b10),
        "fit_converged": convergiu,
        "fit_method": AJUSTE_WEIBULL_METODO,
        "event_observation": "interval_censored_on_a_det_grid",
        "right_censoring": "non_detection_at_observed_horizon",
        "a_det_grid_step": delta_a,
        "n_niveis_distintos": n_niveis_distintos,
        "taxa_empates": taxa_empates,
        "adequacy_method": (
            "RMSE Kaplan-Meier e papel Weibull com empates agrupados"
        ),
        "km_rmse": km_rmse,
        "diagnostico_papel_weibull": diagnostico_papel,
        "teste_aderencia_quantizada": teste_aderencia,
        "aderencia_alfa": ALFA_ADERENCIA,
        "aderencia_aceitavel": aderencia_aceitavel,
        "min_niveis_aderencia": MIN_NIVEIS_ADERENCIA,
        "niveis_suficientes_aderencia": niveis_suficientes,
        "status_aderencia": status_aderencia,
        "triagem_papel_r2_min": MIN_R2_PAPEL_WEIBULL,
        "triagem_papel_compativel": triagem_compativel,
        "triagem_papel_nota": (
            "R2 no papel de Weibull e triagem visual descritiva. O criterio "
            "principal usa bootstrap parametrico com a mesma quantizacao."
        ),
        "resumo_parametrico_recomendado": resumo_parametrico_recomendado,
        "n_traj": int(len(tempos)),
        "n_eventos": int(obs.sum()),
        "n_censurados": int((~obs).sum()),
        "censura_pct": censura_pct,
        # Onde a não detecção deixa de ser um número só: ver classificar_desfechos.
        "desfechos": desfechos,
        "a_det_unidade": A_DET_UNIDADE,
        "eixo_nao_e_tempo": True,
        # Aliases: a chave é a antiga, o valor já é a unidade nova.
        "ttf_unidade": TTF_UNIDADE,
        "rul_unidade": TTF_UNIDADE,
        "tempo_fisico_calibrado": TEMPO_FISICO_CALIBRADO,
        "min_eventos_exigidos": MIN_EVENTOS_WEIBULL,
        # Compatibilidade: indica disponibilidade da curva paramétrica. Alta
        # censura passa a ser ressalva explícita, não motivo para apagar a RUL.
        "rul_reportavel": resumo_parametrico_recomendado,
        "rul_parametrica_disponivel": bool(convergiu),
        "rul_parametrica_alta_incerteza": bool(convergiu and alta_censura),
        "margem_parametrica_disponivel": bool(convergiu),
        "margem_parametrica_reportavel": resumo_parametrico_recomendado,
        "rul_restrita_disponivel": bool(len(tempos) > 0),
        "rul_restrita_horizonte": horizonte,
        "rul_restrita_inicial": rul_restrita_inicial,
        "margem_restrita_disponivel": bool(len(tempos) > 0),
        "margem_restrita_horizonte": horizonte,
        "margem_restrita_inicial": rul_restrita_inicial,
        "media_a_det_parametrica": float(mttf),
        "a10_parametrico": float(b10),
        "a_det_mean_detectadas": (
            float(np.mean(tempos[obs])) if obs.any() else None
        ),
        "a_det_min": float(np.min(tempos)),
        "a_det_max": float(np.max(tempos)),
        # Aliases das mesmas grandezas, com os nomes antigos.
        "ttf_mean_observado": (
            float(np.mean(tempos[obs])) if obs.any() else None
        ),
        "ttf_min": float(np.min(tempos)),
        "ttf_max": float(np.max(tempos)),
        "bootstrap_solicitados": int(n_boot),
        "bootstrap_validos": len(amostras_boot),
        "bootstrap_taxa_validos": taxa_bootstrap,
        "bootstrap_unidade": "janela_holdout_sem_sobreposicao_de_amostras",
        "bootstrap_independencia_demonstrada": False,
        "bootstrap_nota": (
            "Janelas nao compartilham amostras, mas independencia temporal "
            "entre trajetorias nao foi demonstrada; ICs sao condicionais ao "
            "experimento E2."
        ),
        "media_a_det_parametrica_ci95": cis["mttf_ci95"],
        "a10_parametrico_ci95": cis["b10_ci95"],
        **cis,
        # ── Curvas e interpretação ──────────────────────────────────────────
        # Até 07/08/2026, R(t) e h(t) só existiam DENTRO do código de plotagem
        # e iam apenas para o PNG. O agente não conseguia responder "qual a
        # confiabilidade em t = 40?" com número, a dissertação não tinha valor
        # para tabelar e a banca não tinha o que conferir. Agora saem como dado.
        **_curvas_e_interpretacao(convergiu, beta, eta, horizonte, cis,
                                  desfechos),
    }


def motivo_nao_estimavel(desfechos: dict) -> str:
    """Por que a Weibull não saiu, COM os números — não só "não convergiu".

    O pesquisador reportou que o IGBT "sumia" dos gráficos. Sumia mesmo: os
    painéis ficavam sem β/η e a legenda dizia apenas "ajuste não estimável",
    sem dizer se faltou 1 evento ou 50, nem que a curva não paramétrica
    continuava válida. Um buraco silencioso num capítulo é pior que um número
    ruim — a banca não tem como distinguir "não deu" de "quebrou".

    Fonte única da frase: gráficos, relatório em Markdown e JSON usam esta.
    """
    n_det = int(desfechos.get("n_detectadas", 0))
    n_traj = int(desfechos.get("n_traj", 0))
    faltam = max(MIN_EVENTOS_WEIBULL - n_det, 0)
    pod = desfechos.get("pod_mon_no_teto")
    nao_det = int(desfechos.get("n_indetectaveis_no_teto", 0))

    partes = [
        f"Weibull não estimável: {n_det} detecções em {n_traj} trajetórias, "
        f"contra o mínimo de {MIN_EVENTOS_WEIBULL}."
    ]
    if faltam:
        partes.append(
            f"Faltou {faltam} evento{'s' if faltam > 1 else ''} — o critério "
            "NÃO foi afrouxado para produzir uma curva."
        )
    if pod is not None and np.isfinite(pod):
        partes.append(
            f"POD_mon no teto = {pod:.1%}: {nao_det} trajetórias não são "
            "detectadas nem com a assinatura inteira (a_inj = 1,0)."
        )
    partes.append(
        "A curva Kaplan-Meier continua válida e está no gráfico: é não "
        "paramétrica e não exige mínimo de eventos. O que falta é a "
        "EXTRAPOLAÇÃO paramétrica, não a descrição do observado."
    )
    return " ".join(partes)


def _curvas_e_interpretacao(convergiu: bool, beta: float, eta: float,
                            horizonte: float, cis: dict,
                            desfechos: dict | None = None) -> dict:
    """Bloco de curvas amostradas + leitura de engenharia, para o JSON.

    Separado de `ajustar_weibull` para caber no limite de linhas do módulo e
    para ser testável isoladamente. Ver src/ml/confiabilidade.py.
    """
    from src.ml import confiabilidade as cf

    if not convergiu or not (beta > 0 and eta > 0):
        return {
            "curvas": None,
            "marcos": None,
            "interpretacao": {
                "conclusivo": False,
                "leitura": (motivo_nao_estimavel(desfechos) if desfechos else
                            "ajuste não convergiu — sem curva de confiabilidade "
                            "nem leitura de regime de falha"),
                "km_continua_valida": True,
            },
        }

    # Estende o eixo além do horizonte observado para a curva mostrar a cauda,
    # mas o artefato registra até onde há OBSERVAÇÃO — o resto é extrapolação.
    t_max = float(horizonte)
    ic_beta = cis.get("beta_ci95") or [None, None]
    tem_ic = ic_beta[0] is not None and ic_beta[1] is not None

    marcos = cf.marcos(beta, eta)
    marcos.update({
        "a01": marcos["q01"],
        "a10": marcos["q10"],
        "a50": marcos["q50"],
        "media_a_det": marcos["media"],
        "semantica": "quantis da magnitude de primeiro cruzamento, nao vida",
    })
    return {
        "curvas": cf.curvas(beta, eta, t_max=t_max, n=200),
        "semantica_curvas": {
            "R": "P(a_det > a): ainda nao detectada",
            "F": "P(a_det <= a): detectada ate a magnitude a",
            "f": "densidade parametrica da magnitude de deteccao",
            "h": "intensidade parametrica de primeiro cruzamento por unidade de a",
        },
        "marcos": marcos,
        "horizonte_observado": float(horizonte),
        "nota_extrapolacao": (
            f"curvas publicadas limitadas ao dominio observado a <= {t_max:.1f}; "
            "quantis parametricos fora desse dominio sao extrapolativos"),
        "interpretacao": cf.classificar_forma(
            beta, ic_beta=tuple(ic_beta) if tem_ic else None,
            eixo_tempo=False),
    }


# ============================================================
# ESTIMATIVA DE RUL CONDICIONAL
# ============================================================

def rul_condicional(t_atual: float, beta: float, eta: float) -> float:
    """
    Margem residual esperada dado que a deteccao nao ocorreu ate ``t_atual``.

    Pela identidade da media residual (nao pela propriedade sem memoria, que a
    Weibull so possui quando beta=1):
      E[T - t | T > t] = integral_t^∞ R(s)/R(t) ds
                       = eta * exp(z) * Γ(1 + 1/beta, z) - t
      onde z = (t/eta)^beta e Γ(.,.) é a gama incompleta superior.
    """
    if beta <= 0 or eta <= 0:
        return float("nan")

    if t_atual <= 0:
        return eta * gamma_func(1 + 1 / beta)  # margem média desde a=0

    z = (t_atual / eta) ** beta
    s = 1 + 1 / beta

    if z > 700:
        # Aproximação assintótica evita overflow numérico para tempos extremos.
        return float((eta / beta) * (t_atual / eta) ** (1 - beta))

    gama_sup = gamma_func(s) * gammaincc(s, z)
    media_condicional = eta * np.exp(z) * gama_sup
    return float(max(media_condicional - t_atual, 0.0))


# Nome canonico; o alias antigo permanece para compatibilidade.
margem_condicional_weibull = rul_condicional


# ============================================================
# VISUALIZAÇÕES
# ============================================================

_EXPORTACOES_TARDIAS = (("src.ml.graficos_rul", (
    "plotar_ttf_histogramas", "plotar_confiabilidade",
    "plotar_intensidade_deteccao", "plotar_funcoes_distribuicao_weibull",
    "plotar_distribuicao_weibull", "plotar_rul",
    "plotar_sensibilidade_grade", "plotar_modos_operacao",
)),)


def __getattr__(nome: str):
    from src.core.importacao import resolver_exportacao_tardia

    return resolver_exportacao_tardia(nome, _EXPORTACOES_TARDIAS, globals())


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_rul_weibull() -> bool:
    from src.ml.rul_weibull_execucao import executar_rul_weibull as executar

    return executar()


def regenerar_graficos_weibull(pasta: Path = PASTA_AE) -> dict:
    from src.ml.rul_weibull_execucao import regenerar_graficos_weibull as regenerar

    return regenerar(pasta)



# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    executar_rul_weibull()
