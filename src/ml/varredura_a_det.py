"""
varredura_a_det.py — Al IAdo PV

A VARREDURA de magnitude: dada uma janela saudável, aumenta a assinatura da
falha de 0 a 1 e devolve `a_det`, a magnitude em que a detecção se confirma.

POR QUE ESTE MÓDULO EXISTE
==========================
Isto estava dentro de `rul_weibull.py`, junto com o AJUSTE da Weibull. São duas
coisas: uma produz o dado (`a_det`), a outra o modela. A separação ficou
necessária quando a varredura passou a ter DOIS consumidores —
`rul_weibull_execucao` (o Autoencoder denso do pipeline) e `weibull_por_modelo`
(qualquer detector, inclusive o AE-LSTM do Ibrahim) — e o módulo original
ultrapassou o limite de mil linhas que `tests/test_limites_arquitetura.py`
impõe.

`rul_weibull` reexporta os nomes daqui, então nada que já importava dele
precisa mudar.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.ml.gpvs_principal import (
    JANELA, normalizar_vetores_f0, vetor_de_features,
)
from src.ml.injecao_falhas import FUNCOES_FALHA

# ── Grade de magnitude ──────────────────────────────────────────────────────
# Estas constantes moram AQUI porque são da varredura, não do ajuste. Mas
# `rul_weibull` mantém espelhos LITERAIS delas: o manifesto de proveniência lê
# `N_STEPS`, `BATCH_INFERENCIA` e `PERSISTENCIA_MAGNITUDE` daquele módulo por
# AST, sem importá-lo, e `literal_eval` não resolve referência a outro nome.
# Um teste garante que espelho e fonte não divirjam.
A_DET_MIN = 0.0
A_DET_MAX = 1.0
N_STEPS = 501            # Δa = 0,002 na grade principal de a_inj em [0, 1]
BATCH_INFERENCIA = 16
# A persistência é uma largura no eixo FÍSICO do experimento, não uma contagem
# de pontos: refinar a grade não pode mudar a definição do detector.
PERSISTENCIA_MAGNITUDE = 0.02


def a_det_da_grade(passo: int, n_steps: int = N_STEPS) -> float:
    """Converte índice da grade de magnitude em ``a_det`` ∈ [0; 1]."""
    n = max(int(n_steps), 2)
    return float(np.clip(int(passo) / (n - 1), A_DET_MIN, A_DET_MAX))


def passos_persistencia(
    n_steps: int,
    largura_magnitude: float = PERSISTENCIA_MAGNITUDE,
) -> int:
    """Converte a largura de confirmação em número de pontos da grade."""
    n = max(int(n_steps), 2)
    largura = float(largura_magnitude)
    if not np.isfinite(largura) or largura < 0:
        raise ValueError("largura_magnitude deve ser finita e não negativa")
    return max(1, int(np.ceil(largura * (n - 1))) + 1)

if TYPE_CHECKING:
    import torch

    from src.ml.autoencoder import Autoencoder


def calcular_erros_batch(vetores: np.ndarray,
                         modelo: Autoencoder,
                         scaler,
                         device: torch.device,
                         estat_residuo: dict | None = None,
                         metodo: str = "mse",
                         normalizacao_baseline: dict | None = None,
                         ensaios: list[str] | np.ndarray | None = None) -> np.ndarray:
    """Normaliza um lote de features e retorna o ESCORE de anomalia por amostra.

    Escore via src/ml/escore_anomalia.py: MSE médio (padrão) ou localizado
    (`metodo="localizado"` + régua). Deve ser o MESMO escore que definiu o
    limiar (senão o TTF cruza uma régua de escala diferente).
    """
    from src.ml import escore_anomalia as ea

    if normalizacao_baseline is not None:
        if ensaios is None:
            raise ValueError("Normalização GPVS exige o ensaio de cada vetor")
        vetores = normalizar_vetores_f0(
            vetores, ensaios, normalizacao_baseline
        )
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
    normalizacao_baseline: dict | None = None,
) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
    """Remove trajetórias cuja janela saudável já nasce acima do limiar."""
    if not janelas:
        return [], np.asarray([], dtype=float), np.asarray([], dtype=bool)

    # Era `.get(coluna, 0.0)` aqui e `[c]` estrito na varredura, no MESMO módulo
    # e com o MESMO colunas_feat: o filtro de elegibilidade aceitaria em silêncio
    # a janela zerada que a varredura logo adiante rejeitaria com KeyError.
    vetores = [vetor_de_features(janela, colunas_feat) for janela in janelas]
    erros = calcular_erros_batch(
        np.asarray(vetores, dtype=np.float32), modelo, scaler, device,
        estat_residuo, metodo, normalizacao_baseline,
        [janela.attrs.get("ensaio") for janela in janelas],
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
                persistencia: int | None = None,
                estat_residuo: dict | None = None,
                metodo: str = "mse",
                normalizacao_baseline: dict | None = None,
                scorer=None) -> tuple[float, bool]:
    """
    Varre a magnitude da assinatura injetada e devolve ``(a_det, detectou)``.

    A magnitude ``a_inj`` cresce linearmente de 0 a 1,0 em ``n_steps`` pontos
    sobre a MESMA janela saudável — a trajetória representa um único ativo cuja
    falha se agrava, não um ativo diferente a cada ponto.

    ``a_det`` é a magnitude em que o escore permanece acima do limiar por uma
    largura fixa de magnitude. ``persistencia`` permite sobrescrever a contagem
    de pontos apenas em testes/compatibilidade; por padrão ela é derivada de
    :func:`passos_persistencia`, para que refinar a grade não mude o detector.

    Se nem em ``a_inj = 1,0`` o escore confirma, devolve ``(1.0, False)``. Isso
    NÃO é censura à direita no sentido usual — ver `classificar_desfechos`: a
    grade foi varrida INTEIRA, e o desfecho é indetectabilidade no teto, não
    interrupção do acompanhamento.

    Parâmetros:
        seed : reproduz o ruído sintético da família Contator AC. NÃO sorteia
               janela-base — a janela vem pronta do holdout (ver abaixo).
        scorer : opcional. ``Callable[[list[DataFrame]], np.ndarray]`` — a MESMA
               interface que `macro_comum` já exige dos dois métodos comparados.
               Quando fornecido, substitui o caminho interno de extração +
               `calcular_erros_batch`, que é específico do Autoencoder denso.
               É o que permite gerar `a_det` — e portanto Weibull, confiabilidade
               e intensidade — para QUALQUER detector, inclusive o AE-LSTM do
               Ibrahim. Sem ele, a cadeia de confiabilidade só sabia falar do
               modelo proposto, e comparar os dois em detectabilidade era
               impossível.
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

    if scorer is None and modelo is not None:
        modelo.eval()

    persistencia_efetiva = (
        passos_persistencia(n_steps)
        if persistencia is None else max(int(persistencia), 1)
    )
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

            if scorer is None:
                vetores.append(vetor_de_features(janela, colunas_feat))
            else:
                vetores.append(janela)

        if scorer is None:
            erros = calcular_erros_batch(
                np.asarray(vetores, dtype=np.float32),
                modelo, scaler, device, estat_residuo, metodo,
                normalizacao_baseline,
                [janela_base.attrs.get("ensaio")] * len(vetores),
            )
        else:
            # O scorer recebe as JANELAS, não o vetor de features: é ele que
            # sabe como featurizar para o seu próprio modelo. Mesma interface do
            # macro_comum, então denso e AE-LSTM entram por aqui sem adaptador.
            erros = np.asarray(scorer(vetores), dtype=float)
        erros_trajetoria.extend(float(erro) for erro in erros)
        acima = np.asarray(erros_trajetoria) > limiar
        if persistencia_efetiva == 1:
            cruzamentos = np.flatnonzero(acima)
        elif len(acima) >= persistencia_efetiva:
            confirmados = np.convolve(
                acima.astype(int),
                np.ones(persistencia_efetiva, dtype=int),
                mode="valid",
            ) >= persistencia_efetiva
            # O evento é registrado quando o critério fica confirmado, não no
            # primeiro ponto ainda isolado da sequência.
            cruzamentos = (
                np.flatnonzero(confirmados) + persistencia_efetiva - 1
            )
        else:
            cruzamentos = np.asarray([], dtype=int)

        if len(cruzamentos) > 0:
            return a_det_da_grade(int(cruzamentos[0]), n_steps), True

    # Não confirmou nem no topo da grade. O desfecho é registrado em a_inj = 1,0
    # — a última magnitude REALMENTE aplicada. A versão anterior devolvia
    # `n_steps`, um índice fora da grade (que vai de 0 a n_steps-1): o desfecho
    # era carimbado num ponto do eixo onde nada foi medido.
    return A_DET_MAX, False


# Nome anterior. Mantido porque o eixo mudou de significado, não a mecânica —
# quem chamava `gerar_ttf` continua obtendo o mesmo experimento, agora com a
# saída em magnitude. Ver o bloco "O EIXO NÃO É TEMPO" no topo do módulo.
gerar_ttf = gerar_a_det


# ============================================================
# AJUSTE DE WEIBULL
