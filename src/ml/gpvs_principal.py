"""Adaptador canônico do GPVS-Faults para o pipeline principal de ML.

O conjunto GPVS é a única fonte de dados dos resultados principais. Os ensaios
F0L/F0M fornecem operação saudável para treino, calibração, teste e injeção E2;
F1-F7 ficam reservados para validação experimental E3.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.config import RAIZ_PROJETO
from src.core.logs import adaptar_logger_como_print, get_logger
from src.ml.estilo_graficos import PALETA, TAM, aplicar_estilo, salvar_figura
from src.ml.gpvs import (
    COLUNAS_FONTE,
    COLUNAS_I_AC,
    COLUNAS_PRIMARIAS,
    COLUNAS_V_AC,
    DOI_GPVS,
    FEATURE_COLUMNS,
    GRID_FREQUENCY_HZ,
    PASTA_GPVS,
    PURGE_WINDOWS,
    arquivos_gpvs,
    extrair_features_gpvs,
    split_f0,
)

aplicar_estilo()

_log = adaptar_logger_como_print(get_logger("gpvs_principal"))

FS = 10_000
F0 = GRID_FREQUENCY_HZ
JANELA = int(round(FS / F0))
SOBREPOSICAO = 0
HARMONICOS = [5, 7, 11, 13]
BASELINE_FRACTION = 0.50
BASELINE_MIN_WINDOWS = 30
BASELINE_IQR_FLOOR_FRACTION = 0.10
ARQUIVO_NORMALIZACAO = "normalizacao_baseline_gpvs.npz"
COLUNAS_CORRENTE = list(COLUNAS_I_AC)
COLUNAS_TENSAO = list(COLUNAS_V_AC)

ARQUIVO_FEATURES = (
    Path(RAIZ_PROJETO) / "dados" / "processados" / "features_gpvs.parquet"
)
PASTA_PROCESSADOS = ARQUIVO_FEATURES.parent
ENSAIOS_SAUDAVEIS = ("F0L", "F0M")
META_COLS = [
    "ensaio", "falha", "modo", "janela_idx", "amostra_inicio",
    "amostra_fim", "tempo_inicio_s", "tempo_fim_s", "tempo_centro_s", "fase",
]


def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _thd_um_ciclo(x: np.ndarray, max_harmonica: int = 40) -> float:
    espectro = np.abs(np.fft.rfft(np.asarray(x, dtype=float) - np.mean(x)))
    fundamental = max(float(espectro[1]), np.finfo(float).eps)
    limite = min(max_harmonica + 1, len(espectro))
    return float(np.sqrt(np.sum(espectro[2:limite] ** 2)) / fundamental)


def extrair_janela(janela_df: pd.DataFrame) -> dict[str, float]:
    """Extrai as mesmas 24 features de ``gpvs.extrair_features_gpvs``."""
    ausentes = [c for c in COLUNAS_PRIMARIAS if c not in janela_df.columns]
    if ausentes:
        raise ValueError(f"Janela GPVS sem colunas primárias: {ausentes}")
    janela = janela_df[COLUNAS_PRIMARIAS].to_numpy(dtype=float)
    if len(janela) != JANELA:
        raise ValueError(f"Janela GPVS deve ter {JANELA} amostras; recebeu {len(janela)}")
    if not np.isfinite(janela).all():
        raise ValueError("Janela GPVS contém NaN ou infinito")
    sinais = {c: janela[:, i] for i, c in enumerate(COLUNAS_PRIMARIAS)}

    features: list[float] = []
    for coluna in ("Ipv", "Vpv", "Vdc"):
        features.extend([float(np.median(sinais[coluna])), _iqr(sinais[coluna])])
    rms_i = np.sqrt(np.mean(janela[:, 3:6] ** 2, axis=0))
    rms_v = np.sqrt(np.mean(janela[:, 6:9] ** 2, axis=0))
    features.extend(rms_i.tolist())
    features.extend(rms_v.tolist())
    for coluna in COLUNAS_CORRENTE + COLUNAS_TENSAO:
        features.append(_thd_um_ciclo(sinais[coluna]))
    features.extend([
        float(np.std(rms_i) / max(float(np.mean(rms_i)), np.finfo(float).eps)),
        float(np.std(rms_v) / max(float(np.mean(rms_v)), np.finfo(float).eps)),
    ])
    potencia_ac = np.sum(janela[:, 3:6] * janela[:, 6:9], axis=1)
    potencia_dc = janela[:, 0] * janela[:, 1]
    features.extend([
        float(np.mean(potencia_ac)), float(np.std(potencia_ac)),
        float(np.median(potencia_dc)), _iqr(potencia_dc),
    ])
    return dict(zip(FEATURE_COLUMNS, features, strict=True))


def _intervalo(indices: np.ndarray) -> list[int]:
    return [int(indices[0]), int(indices[-1]) + 1] if len(indices) else []


def split_features_gpvs(df: pd.DataFrame) -> dict:
    """Split temporal por F0L/F0M com purga e quatro papéis disjuntos."""
    blocos = {chave: [] for chave in ("treino", "validacao", "calibracao", "teste")}
    limites = {chave: [] for chave in blocos}
    por_ensaio = {}
    for ensaio in ENSAIOS_SAUDAVEIS:
        posicoes = np.flatnonzero(df["ensaio"].to_numpy() == ensaio)
        if not len(posicoes):
            raise ValueError(f"Features saudáveis ausentes para {ensaio}")
        local = split_f0(len(posicoes), purge=PURGE_WINDOWS)
        mapa = {
            "treino": local.treino,
            "validacao": local.validacao,
            "calibracao": local.calibracao,
            "teste": local.teste,
        }
        por_ensaio[ensaio] = {}
        for chave, locais in mapa.items():
            globais = posicoes[locais]
            blocos[chave].append(globais)
            limites[chave].append(_intervalo(globais))
            por_ensaio[ensaio][chave] = globais.tolist()

    resultado = {
        chave: np.concatenate(partes).astype(int) for chave, partes in blocos.items()
    }
    todos = np.concatenate(list(resultado.values()))
    if len(np.unique(todos)) != len(todos):
        raise AssertionError("Split GPVS principal possui sobreposição")
    resultado.update({
        "estrategia": "temporal_por_ensaio_F0L_F0M",
        "n_blocos": 2,
        "limites": limites,
        "purge_janelas": PURGE_WINDOWS,
        "por_ensaio": por_ensaio,
        "distancia_sem_compartilhamento": 1,
    })
    return resultado


def _estatistica_baseline(
    matriz: np.ndarray,
    indices: np.ndarray,
    piso_iqr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bloco = np.asarray(matriz, dtype=float)[np.asarray(indices, dtype=int)]
    if len(bloco) < BASELINE_MIN_WINDOWS:
        raise ValueError(
            f"Baseline GPVS exige ao menos {BASELINE_MIN_WINDOWS} janelas"
        )
    mediana = np.median(bloco, axis=0)
    q25, q75 = np.percentile(bloco, [25, 75], axis=0)
    escala = np.maximum(q75 - q25, np.asarray(piso_iqr, dtype=float))
    return mediana, escala


def ajustar_normalizacao_f0(
    features: pd.DataFrame,
    split: dict,
) -> tuple[np.ndarray, dict]:
    """Normaliza F0L/F0M pelo baseline de treino de cada ensaio.

    O piso de IQR vem do conjunto de treino F0 combinado e impede amplificação
    arbitrária de features quase constantes em um ensaio específico.
    """
    matriz = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    treino = matriz[np.asarray(split["treino"], dtype=int)]
    q25, q75 = np.percentile(treino, [25, 75], axis=0)
    piso_iqr = np.maximum(
        (q75 - q25) * BASELINE_IQR_FLOOR_FRACTION,
        np.full(len(FEATURE_COLUMNS), 1e-6),
    )
    normalizada = np.empty_like(matriz, dtype=np.float32)
    baselines = {}
    for ensaio in ENSAIOS_SAUDAVEIS:
        posicoes = np.flatnonzero(features["ensaio"].to_numpy() == ensaio)
        indices_treino = np.asarray(split["por_ensaio"][ensaio]["treino"], dtype=int)
        mediana, escala = _estatistica_baseline(matriz, indices_treino, piso_iqr)
        normalizada[posicoes] = ((matriz[posicoes] - mediana) / escala).astype(np.float32)
        baselines[ensaio] = {"mediana": mediana, "escala": escala}
    return normalizada, {
        "feature_columns": list(FEATURE_COLUMNS),
        "iqr_floor": piso_iqr,
        "baselines": baselines,
        "baseline_fraction": BASELINE_FRACTION,
        "baseline_min_windows": BASELINE_MIN_WINDOWS,
        "iqr_floor_fraction": BASELINE_IQR_FLOOR_FRACTION,
    }


def salvar_normalizacao_baseline(normalizacao: dict, pasta: Path) -> Path:
    caminho = Path(pasta) / ARQUIVO_NORMALIZACAO
    ensaios = list(normalizacao["baselines"])
    np.savez_compressed(
        caminho,
        feature_columns=np.asarray(normalizacao["feature_columns"], dtype="U64"),
        iqr_floor=np.asarray(normalizacao["iqr_floor"], dtype=np.float64),
        ensaios=np.asarray(ensaios, dtype="U8"),
        medianas=np.vstack([
            normalizacao["baselines"][ensaio]["mediana"] for ensaio in ensaios
        ]).astype(np.float64),
        escalas=np.vstack([
            normalizacao["baselines"][ensaio]["escala"] for ensaio in ensaios
        ]).astype(np.float64),
        baseline_fraction=np.asarray([normalizacao["baseline_fraction"]]),
        baseline_min_windows=np.asarray([normalizacao["baseline_min_windows"]]),
        iqr_floor_fraction=np.asarray([normalizacao["iqr_floor_fraction"]]),
    )
    return caminho


def carregar_normalizacao_baseline(pasta: Path) -> dict:
    caminho = Path(pasta) / ARQUIVO_NORMALIZACAO
    if not caminho.exists():
        raise FileNotFoundError(f"Normalização de baseline ausente: {caminho}")
    with np.load(caminho, allow_pickle=False) as dados:
        ensaios = dados["ensaios"].astype(str).tolist()
        return {
            "feature_columns": dados["feature_columns"].astype(str).tolist(),
            "iqr_floor": dados["iqr_floor"],
            "baselines": {
                ensaio: {
                    "mediana": dados["medianas"][i],
                    "escala": dados["escalas"][i],
                }
                for i, ensaio in enumerate(ensaios)
            },
            "baseline_fraction": float(dados["baseline_fraction"][0]),
            "baseline_min_windows": int(dados["baseline_min_windows"][0]),
            "iqr_floor_fraction": float(dados["iqr_floor_fraction"][0]),
        }


def normalizar_vetores_f0(
    vetores: np.ndarray,
    ensaios: list[str] | np.ndarray,
    normalizacao: dict,
) -> np.ndarray:
    matriz = np.asarray(vetores, dtype=np.float32)
    nomes = np.asarray(ensaios).astype(str)
    if len(matriz) != len(nomes):
        raise ValueError("Cada vetor deve declarar seu ensaio GPVS")
    resultado = np.empty_like(matriz, dtype=np.float32)
    for ensaio in np.unique(nomes):
        if ensaio not in normalizacao["baselines"]:
            raise ValueError(f"Baseline F0 desconhecido: {ensaio}")
        mascara = nomes == ensaio
        base = normalizacao["baselines"][ensaio]
        resultado[mascara] = (
            (matriz[mascara] - base["mediana"]) / base["escala"]
        ).astype(np.float32)
    return resultado


def normalizar_comissionamento(
    features: pd.DataFrame,
    normalizacao: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Ajusta baseline inicial e reserva pré-falha tardio para especificidade."""
    pre = np.flatnonzero(features["fase"].eq("pre_falha").to_numpy())
    post = np.flatnonzero(features["fase"].eq("pos_falha").to_numpy())
    n_baseline = max(
        int(normalizacao["baseline_min_windows"]),
        int(np.floor(len(pre) * float(normalizacao["baseline_fraction"]))),
    )
    if n_baseline >= len(pre):
        raise ValueError("Pré-falha insuficiente para baseline e teste separados")
    baseline = pre[:n_baseline]
    pre_teste = pre[n_baseline:]
    matriz = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mediana, escala = _estatistica_baseline(
        matriz, baseline, normalizacao["iqr_floor"]
    )
    transformada = ((matriz - mediana) / escala).astype(np.float32)
    meta = {
        "n_baseline": int(len(baseline)),
        "n_pre_test": int(len(pre_teste)),
        "n_post_test": int(len(post)),
        "baseline_fraction_of_pre": float(len(baseline) / len(pre)),
    }
    return transformada, pre_teste, post, meta


def _ler_ensaio(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(
        caminho,
        usecols=COLUNAS_FONTE,
        dtype={c: np.float32 for c in COLUNAS_FONTE if c != "Time"},
    )


def executar_features_gpvs(
    diretorio: Path = PASTA_GPVS,
    pasta_saida: Path = PASTA_PROCESSADOS,
) -> bool:
    """Extrai e publica features dos dois ensaios saudáveis GPVS."""
    _log("=" * 60)
    _log("  AL IADO PV — FEATURES GPVS-Faults F0")
    _log("=" * 60)
    arquivos = arquivos_gpvs(diretorio)
    frames = []
    inventario = {}
    for ensaio in ENSAIOS_SAUDAVEIS:
        _log(f"\nExtraindo {ensaio}...")
        df = pd.read_csv(arquivos[ensaio])
        features, meta = extrair_features_gpvs(df, ensaio)
        frames.append(features)
        inventario[ensaio] = meta
    resultado = pd.concat(frames, ignore_index=True)
    matriz = resultado[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(matriz).all():
        raise ValueError("Features GPVS principais contêm valores não finitos")

    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    parquet = pasta_saida / "features_gpvs.parquet"
    stats = pasta_saida / "features_gpvs_stats.csv"
    qualidade_json = pasta_saida / "features_gpvs_qualidade.json"
    qualidade_png = pasta_saida / "features_gpvs_qualidade.png"
    resultado.to_parquet(parquet, index=False)
    resultado[FEATURE_COLUMNS].describe().T.to_csv(stats)

    qualidade = {
        "dataset": "GPVS-Faults",
        "doi": DOI_GPVS,
        "escopo": "somente F0L/F0M saudáveis",
        "n_ensaios": len(ENSAIOS_SAUDAVEIS),
        "n_janelas": int(len(resultado)),
        "n_features": len(FEATURE_COLUMNS),
        "n_nan": int(np.isnan(matriz).sum()),
        "n_inf": int(np.isinf(matriz).sum()),
        "n_duplicadas": int(resultado[FEATURE_COLUMNS].duplicated().sum()),
        "janelas_por_ensaio": resultado["ensaio"].value_counts().to_dict(),
        "inventario": inventario,
        "split": "temporal por ensaio; treino/validação/calibração/teste com purga",
    }
    qualidade_json.write_text(
        json.dumps(qualidade, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=TAM["painel_2"], layout="constrained")
    for cor, ensaio in zip(PALETA[:2], ENSAIOS_SAUDAVEIS, strict=True):
        bloco = resultado[resultado["ensaio"] == ensaio]
        axes[0].plot(bloco["tempo_centro_s"], bloco["ia_thd"], color=cor, label=ensaio)
        axes[1].hist(
            bloco["i_rms_unbalance"], bins=30, alpha=0.55, color=cor,
            density=True, label=ensaio,
        )
    axes[0].set(title="THD da corrente no domínio saudável", xlabel="Tempo do ensaio (s)", ylabel="THD de ia")
    axes[1].set(title="Desbalanceamento RMS saudável", xlabel="Desbalanceamento relativo", ylabel="Densidade")
    for ax in axes:
        ax.legend()
    fig.suptitle("GPVS-Faults F0: qualidade das features do pipeline principal")
    salvar_figura(
        fig, qualidade_png,
        "F0L e F0M permanecem separados no split temporal; nenhuma falha real participa do treino.",
    )
    _log(f"\n{len(resultado)} janelas × {len(FEATURE_COLUMNS)} features")
    _log(f"Artefatos: {parquet}")
    return True


def preparar_janelas_holdout(
    arquivo_features: Path = ARQUIVO_FEATURES,
    diretorio: Path = PASTA_GPVS,
    n_max: int | None = None,
) -> tuple[list[pd.DataFrame], dict]:
    """Retorna janelas F0 de teste, não sobrepostas e sem uso no limiar."""
    arquivo_features = Path(arquivo_features)
    if not arquivo_features.exists():
        raise FileNotFoundError(f"Features GPVS não encontradas: {arquivo_features}")
    features = pd.read_parquet(arquivo_features)
    split = split_features_gpvs(features)
    indices = split["teste"]
    if n_max is not None and len(indices) > n_max:
        pos = np.linspace(0, len(indices) - 1, n_max).round().astype(int)
        indices = indices[np.unique(pos)]

    arquivos = arquivos_gpvs(diretorio)
    cache: dict[str, pd.DataFrame] = {}
    janelas = []
    registros = []
    for indice in indices:
        row = features.iloc[int(indice)]
        ensaio = str(row["ensaio"])
        if ensaio not in cache:
            cache[ensaio] = _ler_ensaio(arquivos[ensaio])
        inicio, fim = int(row["amostra_inicio"]), int(row["amostra_fim"])
        janela = cache[ensaio].iloc[inicio:fim].copy().reset_index(drop=True)
        if len(janela) != JANELA:
            continue
        janela.attrs.update({"ensaio": ensaio, "indice_feature": int(indice)})
        janelas.append(janela)
        registros.append({"ensaio": ensaio, "indice_feature": int(indice), "amostra_inicio": inicio})
    if not janelas:
        raise ValueError("Holdout GPVS não produziu janelas válidas")
    meta = {
        "dataset": "GPVS-Faults",
        "doi": DOI_GPVS,
        "protocolo": "holdout_temporal_F0L_F0M_com_purga",
        "estrategia_split": split["estrategia"],
        "split_limites": split["limites"],
        "purga_janelas": split["purge_janelas"],
        "sem_sobreposicao": True,
        "n_janelas_disponiveis": int(len(split["teste"])),
        "n_janelas_usadas": len(janelas),
        "registros": registros,
        "nota_split": "Somente F0 saudável; F1-F7 são reservados à validação experimental E3.",
    }
    return janelas, meta


__all__ = [
    "ARQUIVO_FEATURES", "ARQUIVO_NORMALIZACAO", "BASELINE_FRACTION",
    "BASELINE_IQR_FLOOR_FRACTION", "BASELINE_MIN_WINDOWS",
    "COLUNAS_CORRENTE", "COLUNAS_TENSAO", "F0", "FS", "FEATURE_COLUMNS",
    "JANELA", "META_COLS", "ajustar_normalizacao_f0",
    "carregar_normalizacao_baseline", "executar_features_gpvs",
    "extrair_janela", "normalizar_comissionamento", "normalizar_vetores_f0",
    "preparar_janelas_holdout", "salvar_normalizacao_baseline",
    "split_features_gpvs",
]
