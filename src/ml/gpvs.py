"""Validacao experimental do detector de anomalias no GPVS-Faults.

O modulo mantem o GPVS como protocolo independente do conjunto Stender. Ele
implementa dois testes complementares:

1. transferencia estrita: AE e limiar sao ajustados apenas nos ensaios F0;
2. adaptacao local: cada ensaio usa somente o inicio pre-falha para ajustar um
   AE de normalidade, preservando blocos posteriores para calibracao e teste.

Um PCA de reconstrução usa exatamente o mesmo split adaptativo como baseline
linear. A unidade de inferencia academica e o ensaio (14 cenarios), nao cada
janela altamente autocorrelacionada.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from src.core.config import RAIZ_PROJETO
from src.core.tempo import agora_local

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = TensorDataset = None


PASTA_GPVS = Path(RAIZ_PROJETO) / "dados" / "brutos" / "gpvs" / "csv" / "CSV_Files"
# Isola o benchmark adaptativo dos artefatos canônicos congelados.
PASTA_SAIDA = Path(RAIZ_PROJETO) / "resultados" / "gpvs" / "legado_adaptativo"

COLUNAS_FONTE = [
    "Time", "Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc",
    "Iabc", "If", "Vabc", "Vf",
]
COLUNAS_PRIMARIAS = ["Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc"]
COLUNAS_DC = ["Ipv", "Vpv", "Vdc"]
COLUNAS_I_AC = ["ia", "ib", "ic"]
COLUNAS_V_AC = ["va", "vb", "vc"]

FEATURE_COLUMNS = [
    "Ipv_median", "Ipv_iqr", "Vpv_median", "Vpv_iqr", "Vdc_median", "Vdc_iqr",
    "ia_rms", "ib_rms", "ic_rms", "va_rms", "vb_rms", "vc_rms",
    "ia_thd", "ib_thd", "ic_thd", "va_thd", "vb_thd", "vc_thd",
    "i_rms_unbalance", "v_rms_unbalance", "p_ac_mean", "p_ac_std",
    "p_dc_median", "p_dc_iqr",
]

FALHAS = {
    0: "Sem falha",
    1: "Falha total em um IGBT",
    2: "Falha de 20% em sensor de fase",
    3: "Afundamentos intermitentes de tensao",
    4: "Sombreamento parcial nao uniforme (10-20%)",
    5: "Circuito aberto em 15% do arranjo PV",
    6: "Ganho PI do MPPT/IPPT reduzido em 20%",
    7: "Constante de tempo PI elevada em 20%",
}
FALHAS_CURTAS = {
    1: "Falha em IGBT", 2: "Sensor de fase: -20%",
    3: "Afundamento de tensao", 4: "Sombreamento parcial",
    5: "Arranjo PV: 15% aberto", 6: "Ganho PI: -20%",
    7: "Constante PI: +20%",
}
MODOS = {"L": "IPPT (potencia limitada)", "M": "MPPT (potencia maxima)"}
SEEDS_PADRAO = (13, 29, 42, 71, 101)
SHA256_ZIP_OFICIAL = "88cd20c848fee86752870cf9b198eab45568c31355685328dd75aba982bf1a63"
DOI_GPVS = "10.17632/n76t439f65.1"

GRID_FREQUENCY_HZ = 50.0
TARGET_FPR = 0.01
PURGE_WINDOWS = 2
SUSTAINED_WINDOWS = 5


@dataclass(frozen=True)
class SplitTemporal:
    treino: np.ndarray
    validacao: np.ndarray
    calibracao: np.ndarray
    teste: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "treino": self.treino.tolist(),
            "validacao": self.validacao.tolist(),
            "calibracao": self.calibracao.tolist(),
            "teste": self.teste.tolist(),
        }


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def arquivos_gpvs(diretorio: Path = PASTA_GPVS) -> dict[str, Path]:
    """Localiza exatamente F0L..F7M, aceitando a pasta raiz do ZIP."""
    diretorio = Path(diretorio)
    candidatos = [diretorio, diretorio / "CSV_Files"]
    base = next((p for p in candidatos if (p / "F0L.csv").exists()), diretorio)
    arquivos = {f"F{i}{modo}": base / f"F{i}{modo}.csv" for i in range(8) for modo in "LM"}
    ausentes = [nome for nome, caminho in arquivos.items() if not caminho.exists()]
    if ausentes:
        raise FileNotFoundError(
            "GPVS incompleto. Ausentes: " + ", ".join(ausentes) +
            f". Diretorio consultado: {diretorio}"
        )
    return arquivos


def identificar_ensaio(nome: str) -> tuple[int, str]:
    achado = re.fullmatch(r"F([0-7])([LM])(?:\.csv)?", Path(nome).name)
    if not achado:
        raise ValueError(f"Nome GPVS invalido: {nome}")
    return int(achado.group(1)), achado.group(2)


def inferir_taxa_amostragem(tempo) -> dict[str, float]:
    """Infere a taxa pelo vetor de tempo e valida monotonicidade/jitter."""
    t = np.asarray(tempo, dtype=float)
    if t.ndim != 1 or len(t) < 1000 or not np.isfinite(t).all():
        raise ValueError("Vetor Time deve ser finito, unidimensional e ter >= 1000 pontos")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Time nao e estritamente crescente")
    mediana = float(np.median(dt))
    fs = 1.0 / mediana
    if not 9_000 <= fs <= 11_000:
        raise ValueError(f"Taxa observada fora da faixa GPVS esperada: {fs:.3f} Hz")
    return {
        "dt_median_s": mediana,
        "dt_min_s": float(dt.min()),
        "dt_max_s": float(dt.max()),
        "fs_hz": fs,
        "sampling_period_us": mediana * 1e6,
    }


def validar_dataframe_gpvs(df: pd.DataFrame, nome: str = "ensaio") -> dict[str, float]:
    ausentes = [c for c in COLUNAS_FONTE if c not in df.columns]
    if ausentes:
        raise ValueError(f"{nome}: colunas ausentes: {ausentes}")
    valores = df[COLUNAS_FONTE].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError(f"{nome}: ha valores NaN ou infinitos")
    return inferir_taxa_amostragem(df["Time"].to_numpy(dtype=float))


def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _thd_um_ciclo(x: np.ndarray, max_harmonica: int = 40) -> float:
    """THD espectral em uma janela de um ciclo, sem incluir DC."""
    espectro = np.abs(np.fft.rfft(x - np.mean(x)))
    fundamental = max(float(espectro[1]), np.finfo(float).eps)
    limite = min(max_harmonica + 1, len(espectro))
    return float(np.sqrt(np.sum(espectro[2:limite] ** 2)) / fundamental)


def extrair_features_gpvs(
    df: pd.DataFrame,
    nome: str,
    *,
    grid_frequency_hz: float = GRID_FREQUENCY_HZ,
) -> tuple[pd.DataFrame, dict]:
    """Extrai 24 features fisicas em janelas nao sobrepostas de um ciclo."""
    falha, modo = identificar_ensaio(nome)
    amostragem = validar_dataframe_gpvs(df, nome)
    amostras_janela_reais = amostragem["fs_hz"] / grid_frequency_hz
    amostras_janela = int(round(amostras_janela_reais))
    erro_relativo = abs(amostras_janela - amostras_janela_reais) / amostras_janela_reais
    if erro_relativo > 0.01:
        raise ValueError("A taxa observada nao permite janela de um ciclo a 50 Hz")

    primarias = df[COLUNAS_PRIMARIAS].to_numpy(dtype=float)
    tempo = df["Time"].to_numpy(dtype=float)
    linhas = []
    n_completas = len(df) // amostras_janela
    ponto_falha = len(df) // 2 if falha else None
    tempo_falha = float(tempo[ponto_falha]) if ponto_falha is not None else None

    for indice in range(n_completas):
        inicio = indice * amostras_janela
        fim = inicio + amostras_janela
        janela = primarias[inicio:fim]
        sinais = {c: janela[:, i] for i, c in enumerate(COLUNAS_PRIMARIAS)}

        features: list[float] = []
        for coluna in COLUNAS_DC:
            features.extend([float(np.median(sinais[coluna])), _iqr(sinais[coluna])])

        rms_i = np.sqrt(np.mean(janela[:, 3:6] ** 2, axis=0))
        rms_v = np.sqrt(np.mean(janela[:, 6:9] ** 2, axis=0))
        features.extend(rms_i.tolist())
        features.extend(rms_v.tolist())
        for coluna in COLUNAS_I_AC + COLUNAS_V_AC:
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

        if ponto_falha is None:
            fase = "saudavel"
        elif fim <= ponto_falha:
            fase = "pre_falha"
        elif inicio >= ponto_falha:
            fase = "pos_falha"
        else:
            fase = "transicao"

        linhas.append({
            "ensaio": nome,
            "falha": falha,
            "modo": modo,
            "janela_idx": indice,
            "amostra_inicio": inicio,
            "amostra_fim": fim,
            "tempo_inicio_s": float(tempo[inicio]),
            "tempo_fim_s": float(tempo[fim - 1]),
            "tempo_centro_s": float((tempo[inicio] + tempo[fim - 1]) / 2),
            "fase": fase,
            **dict(zip(FEATURE_COLUMNS, features, strict=True)),
        })

    resultado = pd.DataFrame(linhas)
    if resultado.empty or not np.isfinite(resultado[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError(f"{nome}: features vazias ou nao finitas")
    metadados = {
        **amostragem,
        "rows": int(len(df)),
        "windows": int(len(resultado)),
        "window_samples": amostras_janela,
        "window_duration_nominal_s": amostras_janela / amostragem["fs_hz"],
        "grid_frequency_hz": grid_frequency_hz,
        "fault_sample": ponto_falha,
        "fault_time_s": tempo_falha,
        "discarded_tail_samples": int(len(df) - n_completas * amostras_janela),
    }
    return resultado, metadados


def _split_fracoes(
    indices: np.ndarray,
    fracoes: tuple[float, float, float],
    *,
    purge: int = PURGE_WINDOWS,
) -> SplitTemporal:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 60 or np.any(np.diff(indices) != 1):
        raise ValueError("O split exige pelo menos 60 janelas consecutivas")
    n = len(indices)
    a, b, c = (int(n * f) for f in fracoes)
    split = SplitTemporal(
        treino=indices[:a],
        validacao=indices[a + purge:b],
        calibracao=indices[b + purge:c],
        teste=indices[c + purge:],
    )
    blocos = [split.treino, split.validacao, split.calibracao, split.teste]
    if any(len(bloco) < 10 for bloco in blocos):
        raise ValueError("Split GPVS gerou bloco com menos de 10 janelas")
    unidos = np.concatenate(blocos)
    if len(np.unique(unidos)) != len(unidos):
        raise AssertionError("Split GPVS possui sobreposicao")
    return split


def split_f0(n_janelas: int, *, purge: int = PURGE_WINDOWS) -> SplitTemporal:
    """F0: 50% treino, 15% validacao, 15% calibracao, restante teste."""
    return _split_fracoes(np.arange(n_janelas), (0.50, 0.65, 0.80), purge=purge)


def split_adaptativo(features: pd.DataFrame, *, purge: int = PURGE_WINDOWS) -> SplitTemporal:
    """Pre-falha: 40% treino, 15% validacao, 20% calibracao, restante teste."""
    pre = features.index[features["fase"].eq("pre_falha")].to_numpy(dtype=int)
    return _split_fracoes(pre, (0.40, 0.55, 0.75), purge=purge)


def _exigir_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch e necessario para executar a validacao GPVS")


def _treinar_ae(
    x_treino: np.ndarray,
    x_validacao: np.ndarray,
    *,
    seed: int,
    epochs: int,
    paciencia: int,
) -> tuple[object, dict]:
    _exigir_torch()
    from src.ml.autoencoder import Autoencoder

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    modelo = Autoencoder(x_treino.shape[1], latente_dim=4, dropout=0.1).to(device)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    criterio = nn.MSELoss()
    tensor_treino = torch.as_tensor(x_treino, dtype=torch.float32)
    tensor_val = torch.as_tensor(x_validacao, dtype=torch.float32)
    gerador = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(tensor_treino), batch_size=32, shuffle=True, generator=gerador,
    )
    melhor_loss = math.inf
    melhor_epoca = 0
    melhor_estado = None
    sem_melhora = 0
    epocas_executadas = 0

    for epoca in range(1, epochs + 1):
        epocas_executadas = epoca
        modelo.train()
        for (batch,) in loader:
            otimizador.zero_grad()
            loss = criterio(modelo(batch), batch)
            loss.backward()
            otimizador.step()
        modelo.eval()
        with torch.no_grad():
            loss_val = float(criterio(modelo(tensor_val), tensor_val).item())
        if loss_val < melhor_loss - 1e-7:
            melhor_loss = loss_val
            melhor_epoca = epoca
            melhor_estado = {
                nome: valor.detach().clone() for nome, valor in modelo.state_dict().items()
            }
            sem_melhora = 0
        else:
            sem_melhora += 1
        if sem_melhora >= paciencia:
            break

    if melhor_estado is None:
        raise RuntimeError("Treino GPVS nao produziu estado valido")
    modelo.load_state_dict(melhor_estado)
    return modelo, {
        "seed": seed,
        "best_epoch": melhor_epoca,
        "epochs_run": epocas_executadas,
        "best_validation_mse": melhor_loss,
    }


def _score_ae(modelo, x: np.ndarray) -> np.ndarray:
    _exigir_torch()
    modelo.eval()
    tensor = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        return ((modelo(tensor) - tensor) ** 2).mean(dim=1).cpu().numpy()


def _score_pca(modelo: PCA, x: np.ndarray) -> np.ndarray:
    reconstruido = modelo.inverse_transform(modelo.transform(x))
    return np.mean((x - reconstruido) ** 2, axis=1)


def _limiar_p99(scores: np.ndarray) -> float:
    if len(scores) < 20 or not np.isfinite(scores).all():
        raise ValueError("Calibracao do limiar exige >=20 scores finitos")
    return float(np.percentile(scores, 99))


def _metricas(
    features: pd.DataFrame,
    indice_anomalia: np.ndarray,
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    *,
    sustained_windows: int = SUSTAINED_WINDOWS,
) -> dict:
    pre_indices = np.asarray(pre_indices, dtype=int)
    post_indices = np.asarray(post_indices, dtype=int)
    if not len(pre_indices) or not len(post_indices):
        raise ValueError("Metricas exigem janelas pre e pos-falha")
    y = np.concatenate([np.zeros(len(pre_indices)), np.ones(len(post_indices))])
    score = np.concatenate([indice_anomalia[pre_indices], indice_anomalia[post_indices]])
    pred = indice_anomalia > 1.0
    fpr = float(np.mean(pred[pre_indices]))
    tpr = float(np.mean(pred[post_indices]))
    auc = float(roc_auc_score(y, score))

    delay = None
    primeiro_pos = int(post_indices[0])
    tempo_falha = float(features.iloc[primeiro_pos]["tempo_inicio_s"])
    for deslocamento in range(0, len(post_indices) - sustained_windows + 1):
        bloco = post_indices[deslocamento:deslocamento + sustained_windows]
        if np.all(np.diff(bloco) == 1) and bool(np.all(pred[bloco])):
            delay = float(features.iloc[int(bloco[0])]["tempo_inicio_s"] - tempo_falha)
            break
    return {
        "n_pre_test": int(len(pre_indices)),
        "n_post_test": int(len(post_indices)),
        "pre_fpr": fpr,
        "specificity": 1.0 - fpr,
        "post_tpr": tpr,
        "auc": auc,
        "balanced_accuracy": float((tpr + 1.0 - fpr) / 2),
        "sustained_detection": delay is not None,
        "detection_delay_s": delay,
        "median_pre_index": float(np.median(indice_anomalia[pre_indices])),
        "median_post_index": float(np.median(indice_anomalia[post_indices])),
    }


def _bootstrap_media(
    valores: np.ndarray,
    *,
    seed: int = 20260809,
    n_boot: int = 20_000,
) -> dict[str, float]:
    valores = np.asarray(valores, dtype=float)
    if not len(valores) or not np.isfinite(valores).all():
        raise ValueError("Bootstrap exige valores finitos")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(valores), size=(n_boot, len(valores)))
    medias = valores[indices].mean(axis=1)
    return {
        "mean": float(valores.mean()),
        "ci95_low": float(np.percentile(medias, 2.5)),
        "ci95_high": float(np.percentile(medias, 97.5)),
        "n_experiments": int(len(valores)),
        "bootstrap_resamples": n_boot,
    }


def _carregar_features(diretorio: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    features: dict[str, pd.DataFrame] = {}
    inventario = {}
    for nome, caminho in arquivos_gpvs(diretorio).items():
        df = pd.read_csv(caminho)
        feat, meta = extrair_features_gpvs(df, nome)
        features[nome] = feat
        inventario[nome] = {
            **meta,
            "path": str(caminho.relative_to(RAIZ_PROJETO)).replace("\\", "/"),
            "sha256": _sha256(caminho),
            "size_bytes": int(caminho.stat().st_size),
        }
    return features, inventario


def _pre_post(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pre = features.index[features["fase"].eq("pre_falha")].to_numpy(dtype=int)
    post = features.index[features["fase"].eq("pos_falha")].to_numpy(dtype=int)
    return pre, post


def _transferencia_estrita(
    features: dict[str, pd.DataFrame],
    *,
    seeds: tuple[int, ...],
    epochs: int,
    paciencia: int,
) -> tuple[dict[str, np.ndarray], list[dict], list[dict]]:
    indices_por_ensaio: dict[str, list[np.ndarray]] = {
        f"F{i}{modo}": [] for i in range(1, 8) for modo in "LM"
    }
    treinos = []
    f0_testes = []
    for modo in "LM":
        f0 = features[f"F0{modo}"]
        x0 = f0[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        split = split_f0(len(f0))
        for seed in seeds:
            scaler = RobustScaler().fit(x0[split.treino])
            z0 = scaler.transform(x0).astype(np.float32)
            modelo, meta = _treinar_ae(
                z0[split.treino], z0[split.validacao],
                seed=seed, epochs=epochs, paciencia=paciencia,
            )
            scores0 = _score_ae(modelo, z0)
            limiar = _limiar_p99(scores0[split.calibracao])
            f0_testes.append({
                "modo": modo,
                "seed": seed,
                "n_test": int(len(split.teste)),
                "fpr": float(np.mean(scores0[split.teste] > limiar)),
                "threshold": limiar,
            })
            treinos.append({"modo": modo, **meta})
            for falha in range(1, 8):
                nome = f"F{falha}{modo}"
                x = features[nome][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
                score = _score_ae(modelo, scaler.transform(x).astype(np.float32))
                indices_por_ensaio[nome].append(score / limiar)
    ensemble = {
        nome: np.median(np.vstack(valores), axis=0)
        for nome, valores in indices_por_ensaio.items()
    }
    return ensemble, treinos, f0_testes


def _adaptacao_local(
    features: dict[str, pd.DataFrame],
    *,
    seeds: tuple[int, ...],
    epochs: int,
    paciencia: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict], dict[str, SplitTemporal]]:
    ae_indices: dict[str, np.ndarray] = {}
    pca_indices: dict[str, np.ndarray] = {}
    treinos = []
    splits = {}
    for falha in range(1, 8):
        for modo in "LM":
            nome = f"F{falha}{modo}"
            feat = features[nome]
            x = feat[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            split = split_adaptativo(feat)
            splits[nome] = split
            indices_seed = []
            for seed in seeds:
                scaler = RobustScaler().fit(x[split.treino])
                z = scaler.transform(x).astype(np.float32)
                modelo, meta = _treinar_ae(
                    z[split.treino], z[split.validacao],
                    seed=seed, epochs=epochs, paciencia=paciencia,
                )
                scores = _score_ae(modelo, z)
                limiar = _limiar_p99(scores[split.calibracao])
                indices_seed.append(scores / limiar)
                treinos.append({
                    "ensaio": nome,
                    "threshold": limiar,
                    "n_train": int(len(split.treino)),
                    "n_validation": int(len(split.validacao)),
                    "n_calibration": int(len(split.calibracao)),
                    **meta,
                })
            ae_indices[nome] = np.median(np.vstack(indices_seed), axis=0)

            scaler_pca = RobustScaler().fit(x[split.treino])
            z_pca = scaler_pca.transform(x)
            pca = PCA(n_components=4, random_state=42).fit(z_pca[split.treino])
            score_pca = _score_pca(pca, z_pca)
            limiar_pca = _limiar_p99(score_pca[split.calibracao])
            pca_indices[nome] = score_pca / limiar_pca
    return ae_indices, pca_indices, treinos, splits


def _resumir_macros(cenarios: list[dict]) -> dict:
    resumo = {}
    for protocolo in ("strict_ae", "adaptive_ae", "adaptive_pca"):
        resumo[protocolo] = {}
        for escopo, filtrados in {
            "all": cenarios,
            "L": [r for r in cenarios if r["mode"] == "L"],
            "M": [r for r in cenarios if r["mode"] == "M"],
        }.items():
            resumo[protocolo][escopo] = {
                metrica: _bootstrap_media(
                    np.array([r[f"{protocolo}_{metrica}"] for r in filtrados]),
                    seed=20260809 + sum(ord(c) for c in protocolo + escopo + metrica),
                )
                for metrica in ("auc", "post_tpr", "specificity", "balanced_accuracy")
            }
    return resumo


def _salvar_tabela_md(df: pd.DataFrame, caminho: Path) -> None:
    colunas = [
        "experiment", "fault_type", "mode",
        "strict_ae_auc", "strict_ae_specificity",
        "adaptive_ae_auc", "adaptive_ae_post_tpr", "adaptive_ae_specificity",
        "adaptive_pca_auc", "adaptive_pca_post_tpr", "adaptive_pca_specificity",
    ]
    nomes = {
        "experiment": "Ensaio", "fault_type": "Falha", "mode": "Modo",
        "strict_ae_auc": "AUC AE estrito", "strict_ae_specificity": "Esp. AE estrito",
        "adaptive_ae_auc": "AUC AE adapt.", "adaptive_ae_post_tpr": "Sens. AE adapt.",
        "adaptive_ae_specificity": "Esp. AE adapt.", "adaptive_pca_auc": "AUC PCA",
        "adaptive_pca_post_tpr": "Sens. PCA", "adaptive_pca_specificity": "Esp. PCA",
    }
    tabela = df[colunas].rename(columns=nomes).copy()
    numericas = [c for c in tabela.columns if c not in ("Ensaio", "Falha", "Modo")]
    for coluna in numericas:
        tabela[coluna] = tabela[coluna].map(lambda v: f"{float(v):.3f}")
    cabecalho = "| " + " | ".join(tabela.columns) + " |"
    separador = "| " + " | ".join("---" for _ in tabela.columns) + " |"
    linhas = [
        "| " + " | ".join(str(valor).replace("|", "\\|") for valor in linha) + " |"
        for linha in tabela.itertuples(index=False, name=None)
    ]
    caminho.write_text("\n".join([cabecalho, separador, *linhas]) + "\n", encoding="utf-8")


def _gerar_relatorio(resultado: dict, caminho: Path) -> None:
    macro = resultado["macro_summary"]
    ae = macro["adaptive_ae"]["all"]
    pca = macro["adaptive_pca"]["all"]
    strict = macro["strict_ae"]["all"]
    cenarios = resultado["scenario_results"]
    detectados = sum(r["adaptive_ae_sustained_detection"] for r in cenarios)
    linhas = [
        "# Validacao experimental GPVS-Faults (E3 de bancada)",
        "",
        f"Gerado em `{resultado['created_at']}`. Dataset: DOI {DOI_GPVS}.",
        "",
        "## Resultado principal",
        "",
        (
            "A transferencia estrita do limiar F0 nao e operacional: a especificidade "
            f"macro foi {strict['specificity']['mean']:.3f} "
            f"(IC95% {strict['specificity']['ci95_low']:.3f}-"
            f"{strict['specificity']['ci95_high']:.3f}) por causa do deslocamento "
            "entre ensaios. A taxa de deteccao isolada desse protocolo nao deve ser citada."
        ),
        "",
        (
            "Com adaptacao local usando somente o inicio saudavel, o AE obteve AUC macro "
            f"{ae['auc']['mean']:.3f} (IC95% {ae['auc']['ci95_low']:.3f}-"
            f"{ae['auc']['ci95_high']:.3f}), sensibilidade {ae['post_tpr']['mean']:.3f} "
            f"e especificidade {ae['specificity']['mean']:.3f}. Houve deteccao sustentada "
            f"em {detectados}/14 ensaios."
        ),
        "",
        (
            "O baseline PCA, sob o mesmo split, obteve AUC macro "
            f"{pca['auc']['mean']:.3f}, sensibilidade {pca['post_tpr']['mean']:.3f} "
            f"e especificidade {pca['specificity']['mean']:.3f}."
        ),
        "",
        "## Leitura por modo de falha",
        "",
        "- F1, F2 e F5 sao os cenarios mais detectaveis no limiar p99.",
        "- F3 e intermitente; AUC e sensibilidade devem ser lidas separadamente.",
        "- F4, F6 e F7 permanecem limitacoes do detector no limiar operacional.",
        "- Resultado nulo foi preservado; nao houve selecao de cenarios por desempenho.",
        "",
        "## Protocolo",
        "",
        "- 24 features de sensores primarios em janelas nao sobrepostas de um ciclo de 50 Hz.",
        "- Taxa inferida de cada vetor `Time` (aprox. 10 kHz); o manual declara 9,9989 us,",
        "  mas os CSVs observados apresentam aproximadamente 99,9969 us.",
        "- Gargalo linear 4D, cinco sementes, indice final pela mediana dos scores",
        "  normalizados por seus limiares p99.",
        "- Split temporal com purga; nenhuma janela pos-falha entra em scaler, treino,",
        "  early stopping ou calibracao.",
        "- IC95% por bootstrap de ensaios, nao de janelas.",
        "",
        "## Limites de evidencia",
        "",
        "E3 aqui significa validacao experimental externa em bancada. Nao e validacao de",
        "campo, nao estima prevalencia industrial, nao identifica causa automaticamente e",
        "nao fornece tempos de vida para Weibull/RUL fisico.",
        "",
        "## Fontes",
        "",
        f"- GPVS-Faults: https://doi.org/{DOI_GPVS}",
        "- Bakdi et al. (2021): https://doi.org/10.1016/j.ijepes.2020.106457",
    ]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _plotar(resultados: pd.DataFrame, scores: pd.DataFrame, macros: dict, pasta: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    from src.ml.estilo_graficos import (
        COR_ALERTA, COR_METODO, COR_NEUTRA, COR_REFERENCIA, COR_TEXTO_SEC,
        PALETA, aplicar_estilo, salvar_figura,
    )

    aplicar_estilo()
    saidas = []

    fig, axes = plt.subplots(7, 2, figsize=(14, 21), sharex=False)
    for falha in range(1, 8):
        for coluna, modo in enumerate("LM"):
            ax = axes[falha - 1, coluna]
            nome = f"F{falha}{modo}"
            dados = scores[scores["experiment"].eq(nome)]
            valor = np.log10(np.clip(dados["adaptive_ae_index"].to_numpy(), 1e-4, None))
            ax.plot(dados["time_center_s"], valor, color=COR_METODO, linewidth=1.0)
            pos = dados[dados["phase"].eq("post_fault")]
            if not pos.empty:
                ax.axvline(float(pos["time_start_s"].iloc[0]), color=COR_ALERTA,
                           linestyle="--", linewidth=1.1)
            ax.axhline(0.0, color=COR_TEXTO_SEC, linestyle=":", linewidth=1.0)
            ax.set_title(f"{nome} - {FALHAS_CURTAS[falha]}", fontsize=9)
            ax.set_ylabel("log10(indice)")
            if falha == 7:
                ax.set_xlabel("Tempo (s)")
    fig.subplots_adjust(hspace=0.72, wspace=0.23, top=0.955, bottom=0.045)
    fig.suptitle("GPVS-Faults: indice de anomalia do AE adaptativo", y=0.985)
    caminho = pasta / "gpvs_series_temporais.png"
    salvar_figura(
        fig, caminho,
        nota="Linha horizontal: limiar p99. Linha vertical: inicio pos-falha. E3 de bancada; cinco sementes.",
    )
    saidas.append(caminho)

    labels = resultados["experiment"].tolist()[::-1]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
    configs = [
        ("auc", "AUC"), ("post_tpr", "Sensibilidade pos-falha"),
        ("specificity", "Especificidade pre-falha"),
    ]
    metodos = [("strict_ae", "AE estrito", COR_REFERENCIA),
               ("adaptive_ae", "AE adaptativo", COR_METODO),
               ("adaptive_pca", "PCA adaptativo", COR_NEUTRA)]
    altura = 0.23
    for ax, (metrica, titulo) in zip(axes, configs, strict=True):
        for j, (prefixo, rotulo, cor) in enumerate(metodos):
            valores = resultados[f"{prefixo}_{metrica}"].to_numpy()[::-1]
            ax.barh(y + (j - 1) * altura, valores, height=altura, label=rotulo, color=cor)
        ax.set_xlim(0, 1.02)
        ax.set_title(titulo)
        ax.set_xlabel("Proporcao")
    axes[0].set_yticks(y, labels)
    handles, legendas = axes[1].get_legend_handles_labels()
    fig.legend(handles, legendas, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    fig.subplots_adjust(top=0.88, bottom=0.09, wspace=0.20)
    fig.suptitle("Desempenho por ensaio: transferencia estrita e adaptacao local", y=0.985)
    caminho = pasta / "gpvs_metricas_por_cenario.png"
    salvar_figura(fig, caminho, nota="Cada barra resume um ensaio independente; nenhuma janela foi embaralhada entre blocos.")
    saidas.append(caminho)

    matriz_strict = resultados.pivot(index="fault", columns="mode", values="strict_ae_specificity").sort_index()
    matriz_adapt = resultados.pivot(index="fault", columns="mode", values="adaptive_ae_specificity").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
    for ax, matriz, titulo in zip(
        axes, (matriz_strict, matriz_adapt),
        ("Transferencia estrita F0", "AE com adaptacao local"), strict=True,
    ):
        imagem = ax.imshow(matriz.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(2), ["IPPT (L)", "MPPT (M)"])
        ax.set_yticks(range(7), [f"F{i}" for i in matriz.index])
        ax.set_title(titulo)
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                v = float(matriz.iloc[i, j])
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=9)
    fig.colorbar(imagem, ax=axes, label="Especificidade pre-falha", fraction=0.035, pad=0.04)
    fig.suptitle("Deslocamento entre ensaios no GPVS-Faults")
    caminho = pasta / "gpvs_transferencia_estrita.png"
    salvar_figura(fig, caminho, nota="A baixa especificidade do limiar F0 invalida seu uso direto nos demais ensaios.")
    saidas.append(caminho)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    metricas = [("auc", "AUC"), ("post_tpr", "Sensibilidade"), ("specificity", "Especificidade")]
    x = np.arange(len(metricas))
    largura = 0.24
    for j, (prefixo, rotulo, cor) in enumerate(metodos):
        medias, baixos, altos = [], [], []
        for metrica, _ in metricas:
            item = macros[prefixo]["all"][metrica]
            medias.append(item["mean"])
            baixos.append(item["mean"] - item["ci95_low"])
            altos.append(item["ci95_high"] - item["mean"])
        pos = x + (j - 1) * largura
        ax.bar(pos, medias, largura, label=rotulo, color=cor)
        ax.errorbar(pos, medias, yerr=np.vstack([baixos, altos]), fmt="none",
                    ecolor="#0b0b0b", capsize=3, linewidth=1)
        for xp, valor in zip(pos, medias, strict=True):
            ax.text(xp, min(valor + 0.035, 1.045), f"{valor:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [rotulo for _, rotulo in metricas])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Media macro por ensaio")
    ax.set_title("GPVS-Faults: estimativas macro e IC95% por bootstrap de ensaios")
    ax.legend(loc="lower left")
    caminho = pasta / "gpvs_macro_comparacao.png"
    salvar_figura(fig, caminho, nota="IC95% reamostra os 14 ensaios, preservando a dependencia entre janelas de cada ensaio.")
    saidas.append(caminho)
    return saidas


def executar_validacao_gpvs(
    diretorio: Path = PASTA_GPVS,
    pasta_saida: Path = PASTA_SAIDA,
    *,
    seeds: tuple[int, ...] = SEEDS_PADRAO,
    epochs: int = 300,
    paciencia: int = 30,
) -> dict:
    """Executa a validacao completa e grava somente artefatos versionaveis."""
    _exigir_torch()
    diretorio = Path(diretorio)
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    features, inventario = _carregar_features(diretorio)
    strict, treinos_strict, f0_testes = _transferencia_estrita(
        features, seeds=seeds, epochs=epochs, paciencia=paciencia,
    )
    adaptive, pca, treinos_adapt, splits = _adaptacao_local(
        features, seeds=seeds, epochs=epochs, paciencia=paciencia,
    )

    cenarios = []
    linhas_scores = []
    for falha in range(1, 8):
        for modo in "LM":
            nome = f"F{falha}{modo}"
            feat = features[nome]
            pre, post = _pre_post(feat)
            split = splits[nome]
            linha = {
                "experiment": nome,
                "fault": falha,
                "fault_type": FALHAS[falha],
                "mode": modo,
                "mode_name": MODOS[modo],
                "n_windows": int(len(feat)),
                "n_pre_fault": int(len(pre)),
                "n_post_fault": int(len(post)),
                "n_train": int(len(split.treino)),
                "n_validation": int(len(split.validacao)),
                "n_calibration": int(len(split.calibracao)),
            }
            for prefixo, indice, pre_teste in (
                ("strict_ae", strict[nome], pre),
                ("adaptive_ae", adaptive[nome], split.teste),
                ("adaptive_pca", pca[nome], split.teste),
            ):
                metricas = _metricas(feat, indice, pre_teste, post)
                linha.update({f"{prefixo}_{chave}": valor for chave, valor in metricas.items()})
            cenarios.append(linha)

            for idx, registro in feat.iterrows():
                fase = {
                    "pre_falha": "pre_fault", "pos_falha": "post_fault",
                    "transicao": "transition", "saudavel": "healthy",
                }[registro["fase"]]
                linhas_scores.append({
                    "experiment": nome,
                    "fault": falha,
                    "mode": modo,
                    "window_index": int(registro["janela_idx"]),
                    "time_start_s": float(registro["tempo_inicio_s"]),
                    "time_center_s": float(registro["tempo_centro_s"]),
                    "phase": fase,
                    "strict_ae_index": float(strict[nome][idx]),
                    "adaptive_ae_index": float(adaptive[nome][idx]),
                    "adaptive_pca_index": float(pca[nome][idx]),
                })

    macros = _resumir_macros(cenarios)
    sampling_us = [item["sampling_period_us"] for item in inventario.values()]
    resultado = {
        "schema_version": 1,
        "evidence_level": "E3",
        "evidence_scope": "validacao experimental externa em bancada; nao e campo",
        "created_at": agora_local().isoformat(),
        "dataset": {
            "name": "GPVS-Faults",
            "doi": DOI_GPVS,
            "official_zip_sha256": SHA256_ZIP_OFICIAL,
            "n_experiments": 16,
            "n_fault_experiments": 14,
            "files": inventario,
            "observed_sampling_period_us_min": float(min(sampling_us)),
            "observed_sampling_period_us_max": float(max(sampling_us)),
            "manual_sampling_period_us": 9.9989,
            "sampling_note": (
                "Os CSVs observados indicam ~99.997 us (~10 kHz), dez vezes o valor "
                "declarado no ReadMe; o pipeline infere a taxa pelo vetor Time."
            ),
        },
        "protocol": {
            "window": "um ciclo de 50 Hz, nao sobreposto",
            "features": FEATURE_COLUMNS,
            "excluded_source_columns": ["Iabc", "If", "Vabc", "Vf"],
            "excluded_reason": "estimativas derivadas e redundantes; sensores primarios preservam auditabilidade",
            "threshold": "p99 em bloco temporal de calibracao",
            "target_fpr": TARGET_FPR,
            "purge_windows": PURGE_WINDOWS,
            "sustained_detection_windows": SUSTAINED_WINDOWS,
            "seeds": list(seeds),
            "ae_architecture": "24-16-4-16-24; gargalo linear; dropout 0.1",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "epochs_max": epochs,
            "patience": paciencia,
            "strict_f0_split": "50% treino, 15% validacao, 15% calibracao, restante teste",
            "adaptive_split": "40% treino, 15% validacao, 20% calibracao, restante pre-falha teste",
            "baseline": "PCA 4 componentes, mesmo scaler/split/limiar adaptativo",
            "aggregation": "mediana do indice MSE/limiar em cinco sementes",
            "confidence_intervals": "bootstrap de ensaios (20000 reamostragens)",
        },
        "macro_summary": macros,
        "scenario_results": cenarios,
        "f0_healthy_test": f0_testes,
        "training_runs": {"strict": treinos_strict, "adaptive": treinos_adapt},
        "limitations": [
            "A transferencia estrita sofre deslocamento entre ensaios.",
            "Janelas do mesmo ensaio sao autocorrelacionadas; ICs usam o ensaio como unidade.",
            "Falhas foram introduzidas em bancada, nao observadas em campo.",
            "O detector sinaliza desvio e nao prova causalidade do componente.",
            "O dataset nao contem tempos de vida independentes para Weibull ou RUL fisico.",
        ],
    }

    df_cenarios = pd.DataFrame(cenarios)
    df_scores = pd.DataFrame(linhas_scores)
    json_path = pasta_saida / "validacao_gpvs_e3.json"
    csv_path = pasta_saida / "validacao_gpvs_cenarios.csv"
    tabela_path = pasta_saida / "validacao_gpvs_cenarios.md"
    scores_path = pasta_saida / "validacao_gpvs_scores.csv"
    relatorio_path = pasta_saida / "relatorio_validacao_gpvs.md"
    json_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    df_cenarios.to_csv(csv_path, index=False)
    df_scores.to_csv(scores_path, index=False)
    _salvar_tabela_md(df_cenarios, tabela_path)
    _gerar_relatorio(resultado, relatorio_path)
    figuras = _plotar(df_cenarios, df_scores, macros, pasta_saida)

    saidas = [json_path, csv_path, tabela_path, scores_path, relatorio_path, *figuras]
    manifesto_path = _salvar_manifesto_gpvs(diretorio, resultado, saidas)
    return {
        "ok": True,
        "resultado": resultado,
        "outputs": [str(p) for p in saidas],
        "manifest": str(manifesto_path),
    }


def _salvar_manifesto_gpvs(diretorio: Path, resultado: dict, saidas: list[Path]) -> Path:
    from src.ml.proveniencia import gerar_manifesto, salvar_manifesto

    entradas = {nome: caminho for nome, caminho in arquivos_gpvs(diretorio).items()}
    manifesto = gerar_manifesto(
        "validacao_gpvs_e3",
        Path(__file__),
        parameters=resultado["protocol"],
        input_artifacts=entradas,
        outputs=saidas,
        code_dependencies={
            "src/ml/autoencoder.py": Path(__file__).with_name("autoencoder.py"),
            "src/ml/estilo_graficos.py": Path(__file__).with_name("estilo_graficos.py"),
        },
        evidence_level="E3",
    )
    return salvar_manifesto(manifesto)


def regenerar_graficos_gpvs(
    diretorio: Path = PASTA_GPVS,
    pasta_saida: Path = PASTA_SAIDA,
) -> dict:
    """Regenera figuras e manifesto a partir dos artefatos tabulares salvos."""
    diretorio = Path(diretorio)
    pasta_saida = Path(pasta_saida)
    json_path = pasta_saida / "validacao_gpvs_e3.json"
    csv_path = pasta_saida / "validacao_gpvs_cenarios.csv"
    tabela_path = pasta_saida / "validacao_gpvs_cenarios.md"
    scores_path = pasta_saida / "validacao_gpvs_scores.csv"
    relatorio_path = pasta_saida / "relatorio_validacao_gpvs.md"
    obrigatorios = [json_path, csv_path, scores_path]
    ausentes = [str(p) for p in obrigatorios if not p.exists()]
    if ausentes:
        raise FileNotFoundError("Artefatos GPVS ausentes: " + ", ".join(ausentes))
    resultado = json.loads(json_path.read_text(encoding="utf-8"))
    cenarios = pd.read_csv(csv_path)
    scores = pd.read_csv(scores_path)
    _salvar_tabela_md(cenarios, tabela_path)
    _gerar_relatorio(resultado, relatorio_path)
    figuras = _plotar(cenarios, scores, resultado["macro_summary"], pasta_saida)
    saidas = [json_path, csv_path, tabela_path, scores_path, relatorio_path, *figuras]
    manifesto = _salvar_manifesto_gpvs(diretorio, resultado, saidas)
    return {"ok": True, "outputs": [str(p) for p in saidas], "manifest": str(manifesto)}
__all__ = [
    "FEATURE_COLUMNS", "FALHAS", "MODOS", "PASTA_GPVS", "PASTA_SAIDA",
    "SEEDS_PADRAO", "arquivos_gpvs", "executar_validacao_gpvs", "extrair_features_gpvs",
    "identificar_ensaio", "inferir_taxa_amostragem",
    "regenerar_graficos_gpvs", "split_adaptativo", "split_f0", "validar_dataframe_gpvs",
]
