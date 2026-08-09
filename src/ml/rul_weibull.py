"""
rul_weibull.py — Al IAdo PV / Fase 5
Estimativa de Vida Útil Remanescente (RUL) com Análise de Weibull.

Fundamentação metodológica:
  O dataset de Paderborn contém apenas dados saudáveis (sem falhas reais).
  A estratégia adotada — definida na metodologia da dissertação — é varrer a
  MAGNITUDE da assinatura de falha, com trajetórias sintéticas fundamentadas na
  FMECA do TCC (Torres, 2024).

  Cada trajetória simula um inversor cuja falha se agrava: a magnitude injetada
  `a_inj` cresce de 0 a 1,0 em N_STEPS pontos sobre a MESMA janela saudável.
  Registra-se `a_det`, a magnitude em que o Autoencoder confirma a anomalia
  (escore > limiar por PERSISTENCIA_CRUZAMENTO avaliações seguidas).

  ⚠️  O EIXO NÃO É TEMPO. Sem dados run-to-failure nem taxa de degradação de
  campo, não existe conversão de magnitude para hora, dia ou ano — e inventá-la
  seria inventar o número mais importante do capítulo. β é adimensional; η,
  MTTF e B10 saem em FRAÇÃO DA ASSINATURA NOMINAL. Os nomes MTTF/B10 são
  mantidos porque são os da distribuição, não porque haja tempo envolvido.

  Ganho de ter renomeado: `a_det` e a SMD da injeção (`a_inj,95`) passam a
  compartilhar a unidade, e podem ser lidos na mesma régua.

  Com N_TRAJ trajetórias por tipo de falha, ajusta-se a Weibull de 2 parâmetros
  aos `a_det` obtidos.

Distribuição de Weibull de 2 parâmetros:
  PDF : f(t) = (β/η) × (t/η)^(β−1) × exp(−(t/η)^β)
  CDF : F(t) = 1 − exp(−(t/η)^β)
  R(t): R(t) = exp(−(t/η)^β)         ← Função de Confiabilidade
  h(t): h(t) = (β/η) × (t/η)^(β−1)  ← Taxa de Falha

  Parâmetros:
    β (shape)  — inclinação de Weibull
                 β < 1: mortalidade infantil
                 β = 1: falhas aleatórias (exponencial)
                 β > 1: desgaste (esperado para degradação gradual)
    η (scale)  — vida característica: a em que R(a) = e^−1 ≈ 36,8%

"RUL" neste eixo:
  Para uma falha já agravada até a magnitude `a_atual` sem ter sido detectada,
  a "RUL" é a margem de magnitude ainda esperada até a detecção:
    RUL = E[a_det − a_atual | a_det > a_atual]
  calculada com a Weibull truncada em `a_atual`. É margem de assinatura, não
  vida remanescente em tempo.

Entrada:
  resultados/autoencoder/modelo_autoencoder.pt
  resultados/autoencoder/scaler.pkl
  resultados/autoencoder/limiar.json
  dados/brutos/Inverter_Data_Set.csv

Saída:
  resultados/autoencoder/weibull_ttf.png            (histogramas de a_det)
  resultados/autoencoder/weibull_confiabilidade.png (R(a) e h(a))
  resultados/autoencoder/weibull_distribuicao.png   (f, F e papel de Weibull)
  resultados/autoencoder/weibull_rul.png
  resultados/autoencoder/weibull_results.json
  resultados/autoencoder/weibull_tabela.csv

Uso:
  python src/ml/rul_weibull.py

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta: python src/ml/<arquivo>.py
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ml.estilo_graficos import (
    COR_ALERTA, COR_TEXTO_SEC, TAM, aplicar_estilo, salvar_figura,
)

aplicar_estilo()
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from scipy.stats import weibull_min
from scipy.special import gamma as gamma_func, gammaincc
from scipy.optimize import minimize
from typing import TYPE_CHECKING

from src.ml.features_ca   import extrair_janela, JANELA, FS
from src.ml.dados_avaliacao import carregar_paderborn_compacto, preparar_janelas_holdout
from src.ml.injecao_falhas import (
    FUNCOES_FALHA, FALHAS,
)

if TYPE_CHECKING:
    import torch

    from src.ml.autoencoder import Autoencoder

# ── Caminhos ─────────────────────────────────────────────────
RAIZ        = Path(__file__).parent.parent.parent
ARQUIVO_CSV = RAIZ / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_AE    = RAIZ / "resultados" / "autoencoder"

# ── Parâmetros de simulação ───────────────────────────────────
N_TRAJ  = 100    # teto; o n efetivo não excede janelas independentes do holdout
N_STEPS = 120    # pontos da grade de magnitude por trajetória (a_inj 0→1,0)

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
# MESMA unidade, então β/η/B10 do Weibull e a SMD da injeção podem ser lidos na
# mesma régua. Em passos isso era impossível.
#
# Leitura de B10 = 0,12: em 10% das trajetórias a falha já é detectada com 12%
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
    "adimensional; η, MTTF e B10 estão em fração de assinatura."
)
BATCH_INFERENCIA = 16
N_BOOTSTRAP = 250
MIN_EVENTOS_WEIBULL = 10
MAX_CENSURA_RUL_PCT = 50.0
PERSISTENCIA_CRUZAMENTO = 3


def a_det_da_grade(passo: int, n_steps: int = N_STEPS) -> float:
    """Converte índice da grade de magnitude em ``a_det`` ∈ [0; 1].

    A grade é ``np.linspace(0, 1, n_steps)``, logo o passo ``i`` corresponde a
    ``i/(n_steps−1)``. A conversão fica aqui, e não espalhada, para que o fator
    seja auditável num lugar só.
    """
    n = max(int(n_steps), 2)
    return float(np.clip(int(passo) / (n - 1), A_DET_MIN, A_DET_MAX))


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


def calcular_erros_batch(vetores: np.ndarray,
                         modelo: Autoencoder,
                         scaler,
                         device: torch.device,
                         estat_residuo: dict | None = None,
                         metodo: str = "mse") -> np.ndarray:
    """Normaliza um lote de features e retorna o ESCORE de anomalia por amostra.

    Escore via src/ml/escore_anomalia.py: MSE médio (padrão) ou localizado
    (`metodo="localizado"` + régua). Deve ser o MESMO escore que definiu o
    limiar (senão o TTF cruza uma régua de escala diferente).
    """
    from src.ml import escore_anomalia as ea

    vnorm = scaler.transform(vetores).astype(np.float32)
    residuos = ea.residuo_por_feature(modelo, vnorm, device)
    return ea.pontuar(residuos, estat_residuo, metodo)


def selecionar_janelas_baseline_normais(
    janelas: list[pd.DataFrame],
    modelo: Autoencoder,
    scaler,
    device: torch.device,
    colunas_feat: list[str],
    limiar: float,
    estat_residuo: dict | None = None,
    metodo: str = "mse",
) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
    """Remove trajetórias cuja janela saudável já nasce acima do limiar."""
    if not janelas:
        return [], np.asarray([], dtype=float), np.asarray([], dtype=bool)

    vetores = []
    for janela in janelas:
        feats = extrair_janela(janela)
        vetores.append([feats.get(coluna, 0.0) for coluna in colunas_feat])
    erros = calcular_erros_batch(
        np.asarray(vetores, dtype=np.float32), modelo, scaler, device,
        estat_residuo, metodo,
    )
    elegiveis = np.asarray(erros <= limiar, dtype=bool)
    return (
        [janela for janela, ok in zip(janelas, elegiveis) if ok],
        np.asarray(erros, dtype=float),
        elegiveis,
    )


# ============================================================
# TRAJETÓRIA DE DEGRADAÇÃO PROGRESSIVA
# ============================================================

def gerar_a_det(janela_saudavel: pd.DataFrame,
                modelo: Autoencoder,
                scaler,
                device: torch.device,
                colunas_feat: list,
                limiar: float,
                tipo_falha: str,
                n_steps: int,
                seed: int,
                batch_size: int = BATCH_INFERENCIA,
                persistencia: int = PERSISTENCIA_CRUZAMENTO,
                estat_residuo: dict | None = None,
                metodo: str = "mse") -> tuple[float, bool]:
    """
    Varre a magnitude da assinatura injetada e devolve ``(a_det, detectou)``.

    A magnitude ``a_inj`` cresce linearmente de 0 a 1,0 em ``n_steps`` pontos
    sobre a MESMA janela saudável — a trajetória representa um único ativo cuja
    falha se agrava, não um ativo diferente a cada ponto.

    ``a_det`` é a magnitude em que o escore permanece acima do limiar por
    ``persistencia`` avaliações consecutivas; a persistência evita declarar
    detecção por um pico isolado.

    Se nem em ``a_inj = 1,0`` o escore confirma, devolve ``(1.0, False)``. Isso
    NÃO é censura à direita no sentido usual — ver `classificar_desfechos`: a
    grade foi varrida INTEIRA, e o desfecho é indetectabilidade no teto, não
    interrupção do acompanhamento.

    Parâmetros:
        seed : reproduz o ruído sintético da família Contator AC. NÃO sorteia
               janela-base — a janela vem pronta do holdout (ver abaixo).
    """
    fn          = FUNCOES_FALHA[tipo_falha]
    magnitudes  = np.linspace(0.0, 1.0, n_steps)

    # A janela chega pronta do holdout temporal, com exatamente JANELA amostras.
    # Até 08/08/2026 este bloco sorteava um início com `rng.integers(0, n_disp)`
    # a partir de um DataFrame maior — mas o chamador sempre passou uma janela
    # já recortada, então `n_disp` valia 0 e o sorteio nunca ocorria. Código
    # morto que prometia uma aleatorização inexistente no nome do parâmetro
    # (`df_estavel`) e na docstring.
    if len(janela_saudavel) != JANELA:
        raise ValueError(
            f"Esperada uma janela de {JANELA} amostras, recebidas "
            f"{len(janela_saudavel)}. A janela vem de preparar_janelas_holdout."
        )
    janela_base = janela_saudavel.copy()

    modelo.eval()

    erros_trajetoria: list[float] = []
    for inicio_batch in range(0, n_steps, batch_size):
        fim_batch = min(inicio_batch + batch_size, n_steps)
        vetores = []

        for step in range(inicio_batch, fim_batch):
            sev = magnitudes[step]

            janela = janela_base.copy()

            if sev > 0.01:
                if tipo_falha == "contator_ac":
                    # Mantém a mesma realização de ruído e aumenta somente sua
                    # amplitude. Trocar o ruído a cada passo misturava evolução
                    # da degradação com variabilidade aleatória.
                    janela = fn(janela, float(sev), seed=seed * 10_000)
                else:
                    janela = fn(janela, float(sev))

            feats = extrair_janela(janela)
            vetores.append([feats.get(c, 0.0) for c in colunas_feat])

        erros = calcular_erros_batch(
            np.asarray(vetores, dtype=np.float32),
            modelo, scaler, device, estat_residuo, metodo
        )
        erros_trajetoria.extend(float(erro) for erro in erros)

    acima = np.asarray(erros_trajetoria) > limiar
    persistencia = max(int(persistencia), 1)
    if persistencia == 1:
        cruzamentos = np.flatnonzero(acima)
    elif len(acima) >= persistencia:
        confirmados = np.convolve(
            acima.astype(int), np.ones(persistencia, dtype=int), mode="valid"
        ) >= persistencia
        # O evento é registrado quando o critério fica confirmado, não no
        # primeiro ponto ainda isolado da sequência.
        cruzamentos = np.flatnonzero(confirmados) + persistencia - 1
    else:
        cruzamentos = np.asarray([], dtype=int)

    if len(cruzamentos) > 0:
        return a_det_da_grade(int(cruzamentos[0]), n_steps), True

    # Não confirmou nem no topo da grade. O desfecho é registrado em a_inj = 1,0
    # — a última magnitude REALMENTE aplicada. A versão anterior devolvia
    # `n_steps` (120), um índice fora da grade (que vai de 0 a 119): o desfecho
    # era carimbado num ponto do eixo onde nada foi medido.
    return A_DET_MAX, False


# Nome anterior. Mantido porque o eixo mudou de significado, não a mecânica —
# quem chamava `gerar_ttf` continua obtendo o mesmo experimento, agora com a
# saída em magnitude. Ver o bloco "O EIXO NÃO É TEMPO" no topo do módulo.
gerar_ttf = gerar_a_det


# ============================================================
# AJUSTE DE WEIBULL
# ============================================================

def curva_kaplan_meier(
    ttfs: np.ndarray, eventos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Curva de Kaplan-Meier simples, preservando censura à direita."""
    tempos = np.asarray(ttfs, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    pontos_t = [0.0]
    pontos_s = [1.0]
    sobrevivencia = 1.0
    for t in np.unique(tempos):
        em_risco = int(np.sum(tempos >= t))
        falhas = int(np.sum((tempos == t) & obs))
        if em_risco and falhas:
            sobrevivencia *= 1.0 - falhas / em_risco
        pontos_t.append(float(t))
        pontos_s.append(float(sobrevivencia))
    return np.asarray(pontos_t), np.asarray(pontos_s)


def rul_restrita_km(
    t_atual: float,
    ttfs: np.ndarray,
    eventos: np.ndarray,
    horizonte: float | None = None,
) -> float:
    """RUL média restrita até o horizonte observado, estimada por Kaplan-Meier.

    Diferentemente da extrapolação Weibull, esta medida não pressupõe uma forma
    paramétrica além do último acompanhamento. Por isso continua informativa
    com alta censura, desde que seja apresentada explicitamente como RUL
    restrita ao horizonte sintético do experimento.
    """
    tempos = np.asarray(ttfs, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    if len(tempos) == 0 or len(tempos) != len(obs):
        return float("nan")

    tau = float(np.max(tempos) if horizonte is None else horizonte)
    t0 = float(max(t_atual, 0.0))
    if not np.isfinite(tau) or t0 >= tau:
        return 0.0

    sobrevivencia = 1.0
    inicio = 0.0
    area = 0.0
    sobrevivencia_t0: float | None = None

    for tempo in np.unique(tempos):
        fim = min(float(tempo), tau)
        if fim > inicio:
            if inicio <= t0 < fim:
                sobrevivencia_t0 = sobrevivencia
            area += sobrevivencia * max(fim - max(inicio, t0), 0.0)
        if tempo >= tau:
            inicio = tau
            break

        em_risco = int(np.sum(tempos >= tempo))
        eventos_t = int(np.sum((tempos == tempo) & obs))
        if em_risco > 0:
            sobrevivencia *= 1.0 - eventos_t / em_risco
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
            "subpopulação detectável, e η/MTTF/B10 valem só para ela."
        ),
    }


def _ajuste_weibull_censurado(
    ttfs: np.ndarray, eventos: np.ndarray
) -> tuple[float, float, bool]:
    """MLE Weibull de dois parâmetros com contribuição de sobrevivência."""
    tempos = np.asarray(ttfs, dtype=float)
    obs = np.asarray(eventos, dtype=bool)
    tempos = np.clip(tempos, 1e-6, None)
    if int(obs.sum()) < MIN_EVENTOS_WEIBULL:
        return float("nan"), float("nan"), False

    def neg_log_likelihood(log_params: np.ndarray) -> float:
        beta, eta = np.exp(log_params)
        z = np.power(tempos / eta, beta)
        log_f_evento = (
            np.log(beta) + (beta - 1.0) * np.log(tempos)
            - beta * np.log(eta) - z
        )
        log_s_censura = -z
        ll = np.where(obs, log_f_evento, log_s_censura).sum()
        return float(-ll) if np.isfinite(ll) else 1e30

    # O eixo passou de passos (1..120) para a_det ∈ (0; 1]. O chute e os limites
    # de η eram dimensionados para a escala antiga: `max(mediana, 1.0)` forçava
    # η_ini = 1,0 (o TETO do novo eixo) e o limite inferior de 0,1 podia PRENDER
    # o ótimo, já que η de uma falha bem detectada fica abaixo disso. Ambos
    # passam a acompanhar a escala dos dados.
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


def ajustar_weibull(
    ttfs: np.ndarray,
    eventos: np.ndarray | None = None,
    n_boot: int = 250,
    seed: int = 42,
) -> dict:
    """Ajusta Weibull censurada e estima incerteza por bootstrap de trajetórias."""
    tempos = np.asarray(ttfs, dtype=float)
    obs = (
        np.ones(len(tempos), dtype=bool)
        if eventos is None else np.asarray(eventos, dtype=bool)
    )
    if len(tempos) != len(obs):
        raise ValueError("ttfs e eventos devem ter o mesmo comprimento.")

    beta, eta, convergiu = _ajuste_weibull_censurado(tempos, obs)
    if convergiu:
        mttf = eta * gamma_func(1 + 1 / beta)
        b10 = eta * (-np.log(0.90)) ** (1 / beta)
        km_t, km_s = curva_kaplan_meier(tempos, obs)
        weibull_s = weibull_min.sf(km_t, beta, loc=0, scale=eta)
        km_rmse = float(np.sqrt(np.mean((km_s - weibull_s) ** 2)))
    else:
        mttf = b10 = km_rmse = float("nan")

    amostras_boot: list[tuple[float, float, float, float]] = []
    if convergiu and n_boot > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, len(tempos), size=len(tempos))
            b, e, ok = _ajuste_weibull_censurado(tempos[idx], obs[idx])
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

    return {
        "beta": float(beta),
        "eta": float(eta),
        "mttf": float(mttf),
        "b10": float(b10),
        "fit_converged": convergiu,
        "adequacy_method": "RMSE descritivo entre Kaplan-Meier e Weibull",
        "km_rmse": km_rmse,
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
        "rul_reportavel": bool(convergiu),
        "rul_parametrica_disponivel": bool(convergiu),
        "rul_parametrica_alta_incerteza": bool(convergiu and alta_censura),
        "rul_restrita_disponivel": bool(len(tempos) > 0),
        "rul_restrita_horizonte": horizonte,
        "rul_restrita_inicial": rul_restrita_inicial,
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
        "bootstrap_validos": len(amostras_boot),
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
    t_max = max(float(horizonte) * 1.2, cf.quantil(0.99, beta, eta))
    ic_beta = cis.get("beta_ci95") or [None, None]
    tem_ic = ic_beta[0] is not None and ic_beta[1] is not None

    return {
        "curvas": cf.curvas(beta, eta, t_max=t_max, n=200),
        "marcos": cf.marcos(beta, eta),
        "horizonte_observado": float(horizonte),
        "nota_extrapolacao": (
            f"as curvas vão até {t_max:.1f}, mas só há observação até "
            f"{horizonte:.1f}; além disso é extrapolação do modelo"),
        "interpretacao": cf.classificar_forma(
            beta, ic_beta=tuple(ic_beta) if tem_ic else None),
    }


# ============================================================
# ESTIMATIVA DE RUL CONDICIONAL
# ============================================================

def rul_condicional(t_atual: float, beta: float, eta: float) -> float:
    """
    RUL esperado dado que o componente sobreviveu até t_atual.

    Pela propriedade de memória da Weibull:
      E[T - t | T > t] = integral_t^∞ R(s)/R(t) ds
                       = eta * exp(z) * Γ(1 + 1/beta, z) - t
      onde z = (t/eta)^beta e Γ(.,.) é a gama incompleta superior.
    """
    if beta <= 0 or eta <= 0:
        return float("nan")

    if t_atual <= 0:
        return eta * gamma_func(1 + 1 / beta)  # MTTF completo

    z = (t_atual / eta) ** beta
    s = 1 + 1 / beta

    if z > 700:
        # Aproximação assintótica evita overflow numérico para tempos extremos.
        return float((eta / beta) * (t_atual / eta) ** (1 - beta))

    gama_sup = gamma_func(s) * gammaincc(s, z)
    media_condicional = eta * np.exp(z) * gama_sup
    return float(max(media_condicional - t_atual, 0.0))


# ============================================================
# VISUALIZAÇÕES
# ============================================================

_EXPORTACOES_TARDIAS = (("src.ml.graficos_rul", (
    "plotar_ttf_histogramas", "plotar_confiabilidade",
    "plotar_distribuicao_weibull", "plotar_rul",
)),)


def __getattr__(nome: str):
    from src.core.importacao import resolver_exportacao_tardia

    return resolver_exportacao_tardia(nome, _EXPORTACOES_TARDIAS, globals())


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_rul_weibull() -> bool:
    from src.ml.graficos_rul import (
        plotar_confiabilidade,
        plotar_distribuicao_weibull,
        plotar_rul,
        plotar_ttf_histogramas,
    )

    _log("=" * 60)
    _log("  AL IADO PV — RUL COM WEIBULL")
    _log("=" * 60)
    _log(f"\n  Teto de trajetórias por falha: {N_TRAJ}")
    _log(f"  Grade de magnitude   : {N_STEPS} pontos (a_inj 0→1,0)")
    _log(f"  Eixo do Weibull      : a_det — fração da assinatura nominal, NÃO tempo")

    # ── 1. Carrega artefatos ─────────────────────────────────
    _log(f"\n📂 Carregando artefatos...")
    for arq in [PASTA_AE/"modelo_autoencoder.pt",
                PASTA_AE/"scaler.pkl",
                PASTA_AE/"limiar.json"]:
        if not arq.exists():
            _log(f"   ❌ {arq.name} não encontrado")
            return False

    import torch
    from src.ml.autoencoder import Autoencoder

    checkpoint = torch.load(PASTA_AE/"modelo_autoencoder.pt",
                            map_location="cpu", weights_only=False)
    from src.core.seguranca import carregar_pickle_com_sidecar

    scaler = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
    with open(PASTA_AE/"limiar.json", "r") as f:
        info_limiar = json.load(f)

    n_features   = checkpoint["n_features"]
    latente_dim  = checkpoint["latente_dim"]
    colunas_feat = checkpoint["colunas_feat"]
    limiar       = info_limiar["limiar"]   # OPERACIONAL (método escolhido)

    # Escore operacional (o MESMO que definiu o limiar): método + régua.
    # O TTF é o passo em que ESTE escore cruza ESTE limiar. Sem a régua
    # (artefato antigo), cai para MSE.
    from src.ml import escore_anomalia as ea

    metodo_escore = info_limiar.get("metodo_escore", "mse")
    estat_residuo = ea.carregar_estatistica(PASTA_AE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    _log(f"   ✅ Limiar={limiar:.4f} | device={device} | "
          f"escore={ea.descricao_metodo(metodo_escore, info_limiar.get('k_localizado', 5))}")

    # ── 2. Holdout temporal isolado ───────────────────────────
    _log(f"\n📂 Carregando dataset...")
    df = carregar_paderborn_compacto(ARQUIVO_CSV)
    janelas_holdout, meta_holdout = preparar_janelas_holdout(df)
    del df
    n_janelas_originais = len(janelas_holdout)
    janelas_holdout, erros_baseline, mascara_elegivel = selecionar_janelas_baseline_normais(
        janelas_holdout, modelo, scaler, device, colunas_feat, limiar,
        estat_residuo, metodo_escore
    )
    n_excluidas = int((~mascara_elegivel).sum())
    meta_holdout["filtro_baseline_ttf"] = {
        "criterio": "erro_reconstrucao_baseline <= limiar",
        "limiar": float(limiar),
        "n_janelas_antes": n_janelas_originais,
        "n_janelas_elegiveis": len(janelas_holdout),
        "n_janelas_excluidas": n_excluidas,
        "erros_baseline": [float(x) for x in erros_baseline],
    }
    if not janelas_holdout:
        _log("   ❌ Nenhuma janela saudável ficou abaixo do limiar para gerar TTF")
        return False
    n_traj_real = min(N_TRAJ, len(janelas_holdout))
    _log(f"   ✅ {n_janelas_originais} janelas não sobrepostas do teste")
    _log(f"   ✅ {len(janelas_holdout)} elegíveis; {n_excluidas} excluídas por anomalia em t=0")
    _log(f"   ✅ {n_traj_real} trajetórias independentes serão usadas")

    # ── 3. Gera TTFs por tipo de falha ───────────────────────
    _log(f"\n⚙️  Gerando trajetórias de degradação...")
    ttfs_dict = {}
    eventos_dict = {}

    for idx_falha, falha in enumerate(FALHAS):
        fid  = falha["id"]
        nome = falha["nome"]
        _log(f"\n   🔴 {nome} ({n_traj_real} trajetórias × {N_STEPS} passos)...")

        ttfs = []
        eventos = []
        for i in range(n_traj_real):
            janela_base = janelas_holdout[i]
            a_det, detectou = gerar_a_det(
                janela_base, modelo, scaler, device,
                colunas_feat, limiar, fid, N_STEPS, seed=i,
                estat_residuo=estat_residuo, metodo=metodo_escore
            )
            ttfs.append(a_det)
            eventos.append(detectou)
            if (i + 1) % 20 == 0:
                _log(f"      [{i+1:>3}/{n_traj_real}] a_det médio até agora: "
                      f"{np.mean(ttfs):.3f}", end="\r")

        ttfs = np.array(ttfs, dtype=float)
        eventos = np.asarray(eventos, dtype=bool)
        ttfs_dict[fid] = ttfs
        eventos_dict[fid] = eventos
        d = classificar_desfechos(ttfs, eventos)
        _log(f"\n      a_det: μ={ttfs.mean():.3f} ± {ttfs.std():.3f} | "
              f"min={ttfs.min():.3f} | max={ttfs.max():.3f}")
        _log(f"      POD_mon no teto (a_inj=1,0): {d['pod_mon_no_teto']:.1%} | "
             f"indetectáveis no teto: {d['n_indetectaveis_no_teto']}/{d['n_traj']}")

    # ── 4. Ajuste de Weibull ─────────────────────────────────
    _log(f"\n📐 Ajustando distribuição de Weibull...")
    params = {}
    for falha in FALHAS:
        fid = falha["id"]
        p = ajustar_weibull(
            ttfs_dict[fid], eventos_dict[fid], n_boot=N_BOOTSTRAP,
            seed=42 + len(params),
        )
        params[fid] = p
        npm_str = f"NPR={falha['npr']}"
        _log(f"\n   {falha['nome']} ({npm_str})")
        if p["fit_converged"]:
            _log(f"      β={p['beta']:.3f}  η={p['eta']:.1f}  "
                  f"MTTF={p['mttf']:.1f}  B10={p['b10']:.1f}")
            _log(f"      Censura={p['censura_pct']:.0f}% | "
                 f"RMSE(KM)={p['km_rmse']:.4f} | bootstrap={p['bootstrap_validos']}")
        else:
            _log(
                f"      ⚠️ Weibull não estimável: {p['n_eventos']} eventos; "
                f"mínimo configurado={MIN_EVENTOS_WEIBULL}. "
                "RUL restrita por Kaplan-Meier será mantida."
            )

    # ── 5. Visualizações ─────────────────────────────────────
    _log(f"\n📊 Gerando gráficos...")
    plotar_ttf_histogramas(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_confiabilidade(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_distribuicao_weibull(ttfs_dict, eventos_dict, params, PASTA_AE)
    plotar_rul(ttfs_dict, eventos_dict, params, PASTA_AE)

    # ── 6. Salva resultados ──────────────────────────────────
    # A montagem do artefato vive em src/ml/relatorio_weibull.py: este módulo
    # ficou com a matemática, aquele com a serialização. Ver o docstring de lá.
    from src.ml.relatorio_weibull import montar_relatorio

    relatorio, linhas_weibull = montar_relatorio(
        params=params, a_dets_dict=ttfs_dict, eventos_dict=eventos_dict,
        falhas=FALHAS, meta_holdout=meta_holdout,
        metadados_tempo=metadados_tempo_rul(), limiar=float(limiar),
        n_traj_max=N_TRAJ, n_traj_real=n_traj_real, n_steps=N_STEPS,
        a_det_unidade=A_DET_UNIDADE, ttf_unidade=TTF_UNIDADE,
        tempo_fisico_calibrado=TEMPO_FISICO_CALIBRADO,
        tempo_fisico_nota=TEMPO_FISICO_NOTA,
        min_eventos_weibull=MIN_EVENTOS_WEIBULL,
        max_censura_rul_pct=MAX_CENSURA_RUL_PCT,
        persistencia_cruzamento=PERSISTENCIA_CRUZAMENTO,
        json_seguro=_json_seguro,
    )

    arq_json = PASTA_AE / "weibull_results.json"
    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_json.name}")

    arq_tabela = PASTA_AE / "weibull_tabela.csv"
    pd.DataFrame(linhas_weibull).to_csv(arq_tabela, index=False)
    _log(f"   📋 {arq_tabela.name}")

    # ── 7. Resumo final ──────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  ANÁLISE DE WEIBULL E RUL CONCLUÍDA!")
    _log(f"\n  Valores em FRAÇÃO DA ASSINATURA NOMINAL (a_det), não em tempo.")
    _log(f"\n  {'Falha':<28} {'β':>6} {'η':>7} {'MTTF':>8} {'B10':>8} {'POD@1,0':>8}")
    _log(f"  {'-'*68}")
    for falha in FALHAS:
        fid = falha["id"]
        p   = params[fid]
        _log(f"  {falha['nome']:<28} "
              f"{p['beta']:>6.3f} {p['eta']:>7.3f} "
              f"{p['mttf']:>8.3f} {p['b10']:>8.3f} "
              f"{p['desfechos']['pod_mon_no_teto']:>7.1%}")

    # A leitura do β só vale se o IC95 não cruzar 1 — quem decide isso é
    # confiabilidade.classificar_forma, e a conclusão dela já vem no artefato.
    _log(f"\n  Interpretação do β (válida só quando o IC95 não cruza 1):")
    for falha in FALHAS:
        p = params[falha["id"]]
        interp = p.get("interpretacao") or {}
        if interp.get("leitura"):
            marca = "" if interp.get("conclusivo") else "⚠️  "
            _log(f"  {marca}{falha['nome']}: {interp['leitura']}")
    _log(f"\n  Fase 5 do pipeline de ML concluída!")
    _log(f"  Próximo passo: integração no orquestrador")
    _log(f"{'='*60}")
    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    executar_rul_weibull()
