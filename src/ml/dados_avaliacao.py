"""Recorte temporal canonico para avaliacao interna do pipeline CA.

O Autoencoder e desenvolvido nos blocos do conjunto Stender (Paderborn
University; bancada de acionamento, nao Bearing Dataset). Injecao,
validacao e prognostico sintetico usam exclusivamente o bloco final de teste.
As janelas retornadas nao se sobrepoem, reduzindo dependencia artificial nas
metricas e nos intervalos de confianca.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import RAIZ_PROJETO
from src.ml.features_ca import COLUNAS_CORRENTE, COLUNAS_TENSAO, JANELA
from src.ml.split_temporal import nome_protocolo_split, split_padrao_paderborn

ARQUIVO_FEATURES = (
    RAIZ_PROJETO / "dados" / "processados" / "features_paderborn.parquet"
)


def carregar_paderborn_compacto(arquivo_csv: Path) -> pd.DataFrame:
    """Le apenas os seis sinais CA usados no pipeline, em precisao float32.

    Eram sete ate 08/08/2026: `u_dc_k` saiu junto com a feature `tensao_dc_media`
    (ver "ESCOPO CA" em src/ml/features_ca.py). Nenhuma etapa CA le o barramento.
    """
    colunas = COLUNAS_CORRENTE + COLUNAS_TENSAO
    return pd.read_csv(
        arquivo_csv,
        usecols=colunas,
        dtype={coluna: np.float32 for coluna in colunas},
        memory_map=True,
    )


def _indices_sem_sobreposicao(
    df_features: pd.DataFrame,
    indices: np.ndarray,
    janela: int = JANELA,
) -> np.ndarray:
    """Mantem janelas temporalmente ordenadas sem amostras compartilhadas."""
    escolhidos: list[int] = []
    proximo_inicio = -1
    for idx in indices:
        inicio = int(df_features.iloc[int(idx)]["amostra_inicio"])
        if inicio >= proximo_inicio:
            escolhidos.append(int(idx))
            proximo_inicio = inicio + janela
    return np.asarray(escolhidos, dtype=int)


def _amostra_uniforme(indices: np.ndarray, n_max: int | None) -> np.ndarray:
    if n_max is None or len(indices) <= n_max:
        return indices
    if n_max <= 0:
        raise ValueError("n_max deve ser positivo ou None.")
    posicoes = np.linspace(0, len(indices) - 1, n_max).round().astype(int)
    return indices[np.unique(posicoes)]


def preparar_janelas_holdout(
    df_bruto: pd.DataFrame,
    arquivo_features: Path = ARQUIVO_FEATURES,
    n_max: int | None = None,
    janela: int = JANELA,
) -> tuple[list[pd.DataFrame], dict]:
    """Retorna janelas nao sobrepostas do bloco temporal de teste.

    O dataframe de features fornece os inicios exatos das janelas usadas pelo
    Autoencoder. Isso impede que validacao e injecao voltem, por engano, a um
    intervalo bruto que participou do treinamento ou da calibracao.
    """
    if not arquivo_features.exists():
        raise FileNotFoundError(f"Features nao encontradas: {arquivo_features}")

    df_features = pd.read_parquet(arquivo_features)
    if "amostra_inicio" not in df_features.columns:
        raise ValueError("features_paderborn.parquet nao contem amostra_inicio.")

    split = split_padrao_paderborn(len(df_features))
    indices_teste = _indices_sem_sobreposicao(
        df_features, split["teste"], janela=janela
    )
    indices_teste = _amostra_uniforme(indices_teste, n_max)

    janelas: list[pd.DataFrame] = []
    inicios: list[int] = []
    indices_validos: list[int] = []
    for idx in indices_teste:
        inicio = int(df_features.iloc[int(idx)]["amostra_inicio"])
        fim = inicio + janela
        if fim > len(df_bruto):
            continue
        janelas.append(df_bruto.iloc[inicio:fim].copy().reset_index(drop=True))
        inicios.append(inicio)
        indices_validos.append(int(idx))

    if not janelas:
        raise ValueError("O bloco de teste nao produziu janelas brutas validas.")

    meta = {
        "protocolo": nome_protocolo_split(split, prefixo="holdout_"),
        "estrategia_split": split.get("estrategia"),
        "n_blocos_split": split.get("n_blocos"),
        "split_limites": split["limites"],
        "purga_janelas": split["purge_janelas"],
        "nota_split": (
            "O bloco de teste NÃO é o sufixo temporal da série: são blocos "
            "intercalados, para que treino, calibração e teste cubram a mesma "
            "faixa de rotação. Ver src/ml/split_temporal.py."
        ),
        "sem_sobreposicao": True,
        "n_janelas_disponiveis": len(indices_teste),
        "n_janelas_usadas": len(janelas),
        "indices_features": indices_validos,
        "amostras_inicio": inicios,
    }
    return janelas, meta
