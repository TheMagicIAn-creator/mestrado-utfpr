"""Selecao reproduzivel do autoencoder denso V2 em dados saudaveis GPVS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from src.core.config import RAIZ_PROJETO
from src.core.seguranca import gravar_sidecar_sha256
from src.core.tempo import agora_local
from src.ml.estatistica import intervalo_wilson
from src.ml.gpvs import DOI_GPVS, FEATURE_COLUMNS
from src.ml.gpvs_principal import (
    ajustar_normalizacao_f0,
    salvar_normalizacao_baseline,
    split_features_gpvs,
)

from .graficos import plotar_calibracao, plotar_selecao_modelo
from .modelo import (
    ARQUITETURAS_CANDIDATAS,
    FAMILIAS_FEATURES,
    AutoencoderDenso,
    configurar_seed,
    pesos_por_familia,
    pontuar_residuos,
    residuos_quadraticos,
    treinar,
)

ARQUIVO_FEATURES = Path(RAIZ_PROJETO) / "dados" / "processados" / "features_gpvs.parquet"
PASTA_SAIDA = Path(RAIZ_PROJETO) / "resultados" / "v2" / "autoencoder"
SEEDS = (13, 29, 42, 71, 101)
ALPHA = 0.01
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PACIENCIA = 30
TOLERANCIA_COMPLEXIDADE = 0.02


@dataclass
class DadosSaudaveis:
    frame: pd.DataFrame
    treino: np.ndarray
    validacao: np.ndarray
    calibracao: np.ndarray
    teste: np.ndarray
    indices: dict[str, np.ndarray]
    scaler: RobustScaler
    normalizacao_baseline: dict
    split: dict


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def preparar_dados(arquivo: Path = ARQUIVO_FEATURES) -> DadosSaudaveis:
    frame = pd.read_parquet(arquivo)
    if list(c for c in frame.columns if c in FEATURE_COLUMNS) != list(FEATURE_COLUMNS):
        raise ValueError("Parquet nao segue a ordem canonica das 24 features GPVS")
    split = split_features_gpvs(frame)
    normalizada, baseline = ajustar_normalizacao_f0(frame, split)
    scaler = RobustScaler()
    treino = scaler.fit_transform(normalizada[split["treino"]]).astype(np.float32)
    transformar = lambda papel: scaler.transform(  # noqa: E731
        normalizada[split[papel]]
    ).astype(np.float32)
    return DadosSaudaveis(
        frame=frame,
        treino=treino,
        validacao=transformar("validacao"),
        calibracao=transformar("calibracao"),
        teste=transformar("teste"),
        indices={
            papel: np.asarray(split[papel], dtype=int)
            for papel in ("treino", "validacao", "calibracao", "teste")
        },
        scaler=scaler,
        normalizacao_baseline=baseline,
        split=split,
    )


def limiar_ordem_finita(scores: np.ndarray, alpha: float = ALPHA) -> dict:
    """Quantil split-conformal por ordem estatistica, sem interpolacao."""

    valores = np.sort(np.asarray(scores, dtype=float))
    if valores.ndim != 1 or not len(valores) or not np.isfinite(valores).all():
        raise ValueError("Calibracao exige vetor finito e nao vazio")
    if not 0 < alpha < 1:
        raise ValueError("alpha deve pertencer ao intervalo aberto (0, 1)")
    ordem = int(math.ceil((len(valores) + 1) * (1.0 - alpha)))
    ordem = min(max(ordem, 1), len(valores))
    limiar = float(valores[ordem - 1])
    return {
        "method": "finite_sample_order_statistic",
        "comparison": "score > threshold",
        "alpha_nominal": float(alpha),
        "tail_nominal_pct": float(alpha * 100),
        "n_calibration": int(len(valores)),
        "order_one_based": ordem,
        "sample_percentile_pct": float(100 * ordem / len(valores)),
        "coverage_rank_over_n_plus_one": float(ordem / (len(valores) + 1)),
        "empirical_resolution_pct": float(100 / len(valores)),
        "threshold": limiar,
        "n_strictly_above_calibration": int(np.sum(valores > limiar)),
    }


def resumo_excedencia(scores: np.ndarray, limiar: float) -> dict:
    valores = np.asarray(scores, dtype=float)
    n = len(valores)
    excedencias = int(np.sum(valores > limiar))
    baixo, alto = intervalo_wilson(excedencias, n)
    return {
        "n": n,
        "n_above": excedencias,
        "taxa_pct": float(100 * excedencias / n),
        "ic95_low_pct": float(100 * baixo),
        "ic95_high_pct": float(100 * alto),
    }


def selecionar_arquitetura(
    execucoes: pd.DataFrame,
    tolerancia: float = TOLERANCIA_COMPLEXIDADE,
) -> tuple[str, pd.DataFrame]:
    """Menor rede dentro de 2% da melhor mediana de validacao."""

    obrigatorias = {"arquitetura", "seed", "perda_validacao", "n_parametros"}
    if not obrigatorias.issubset(execucoes.columns):
        raise ValueError(f"Colunas ausentes: {sorted(obrigatorias - set(execucoes))}")
    resumo = (
        execucoes.groupby("arquitetura", as_index=False)
        .agg(
            mediana_validacao=("perda_validacao", "median"),
            media_validacao=("perda_validacao", "mean"),
            desvio_validacao=("perda_validacao", "std"),
            n_parametros=("n_parametros", "first"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["mediana_validacao", "n_parametros"])
        .reset_index(drop=True)
    )
    melhor = float(resumo["mediana_validacao"].min())
    candidatas = resumo[
        resumo["mediana_validacao"] <= melhor * (1.0 + tolerancia)
    ]
    escolhida = str(
        candidatas.sort_values(["n_parametros", "mediana_validacao"]).iloc[0][
            "arquitetura"
        ]
    )
    resumo["dentro_faixa_2pct"] = resumo["mediana_validacao"].le(
        melhor * (1.0 + tolerancia)
    )
    resumo["selecionada"] = resumo["arquitetura"].eq(escolhida)
    return escolhida, resumo


def seed_representativo(execucoes: pd.DataFrame, arquitetura: str) -> int:
    bloco = execucoes[execucoes["arquitetura"].eq(arquitetura)].copy()
    if bloco.empty:
        raise ValueError(f"Arquitetura ausente: {arquitetura}")
    mediana = float(bloco["perda_validacao"].median())
    bloco["distancia_mediana"] = (bloco["perda_validacao"] - mediana).abs()
    return int(bloco.sort_values(["distancia_mediana", "seed"]).iloc[0]["seed"])


def _salvar_pickle(objeto, caminho: Path) -> None:
    with caminho.open("wb") as arquivo:
        pickle.dump(objeto, arquivo)
    gravar_sidecar_sha256(caminho)


def _salvar_json(caminho: Path, dados: dict) -> None:
    caminho.write_text(
        json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def executar_experimento(
    arquivo_features: Path = ARQUIVO_FEATURES,
    pasta_saida: Path = PASTA_SAIDA,
    *,
    epochs: int = EPOCHS,
) -> dict:
    """Seleciona arquitetura/seed em F0 e publica o detector congelado."""

    import torch

    inicio_total = time.perf_counter()
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    dados = preparar_dados(Path(arquivo_features))
    pesos = pesos_por_familia(FEATURE_COLUMNS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    execucoes = []
    estados: dict[tuple[str, int], dict] = {}

    for arquitetura in ARQUITETURAS_CANDIDATAS:
        for seed in SEEDS:
            inicio = time.perf_counter()
            configurar_seed(seed)
            modelo = AutoencoderDenso(len(FEATURE_COLUMNS), arquitetura)
            resultado = treinar(
                modelo,
                dados.treino,
                dados.validacao,
                pesos,
                seed=seed,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                paciencia=PACIENCIA,
                device=device,
            )
            estados[(arquitetura.id, seed)] = resultado.state_dict
            execucoes.append(
                {
                    "arquitetura": arquitetura.id,
                    "seed": seed,
                    "n_parametros": modelo.n_parametros,
                    "razao_parametros_por_janela": modelo.n_parametros
                    / len(dados.treino),
                    "melhor_epoca": resultado.melhor_epoca,
                    "epocas_executadas": len(resultado.historico_validacao),
                    "perda_treino_na_melhor_epoca": resultado.historico_treino[
                        resultado.melhor_epoca - 1
                    ],
                    "perda_validacao": resultado.melhor_validacao,
                    "duracao_s": time.perf_counter() - inicio,
                }
            )

    frame_execucoes = pd.DataFrame(execucoes)
    escolhida_id, frame_resumo = selecionar_arquitetura(frame_execucoes)
    seed_canonico = seed_representativo(frame_execucoes, escolhida_id)
    arquitetura = next(a for a in ARQUITETURAS_CANDIDATAS if a.id == escolhida_id)

    avaliacoes_seed = []
    scores_por_seed = {}
    for seed in SEEDS:
        modelo = AutoencoderDenso(len(FEATURE_COLUMNS), arquitetura).to(device)
        modelo.load_state_dict(estados[(escolhida_id, seed)])
        resid_cal = residuos_quadraticos(modelo, dados.calibracao, device=device)
        resid_teste = residuos_quadraticos(modelo, dados.teste, device=device)
        scores_cal = pontuar_residuos(resid_cal, pesos)
        scores_teste = pontuar_residuos(resid_teste, pesos)
        info_limiar = limiar_ordem_finita(scores_cal)
        limiar = float(info_limiar["threshold"])
        avaliacoes_seed.append(
            {
                "seed": seed,
                "threshold": limiar,
                "calibration": resumo_excedencia(scores_cal, limiar),
                "healthy_test": resumo_excedencia(scores_teste, limiar),
            }
        )
        scores_por_seed[seed] = (scores_cal, scores_teste)
        torch.save(
            {
                "schema_version": 2,
                "state_dict": estados[(escolhida_id, seed)],
                "n_features": len(FEATURE_COLUMNS),
                "feature_columns": list(FEATURE_COLUMNS),
                "architecture": arquitetura.como_dict(),
                "seed": seed,
                "score_method": "family_balanced_mse",
                "feature_weights": pesos.tolist(),
            },
            pasta_saida / f"modelo_seed_{seed}.pt",
        )

    can_seed_info = next(item for item in avaliacoes_seed if item["seed"] == seed_canonico)
    scores_cal, scores_teste = scores_por_seed[seed_canonico]
    modelo_canonico = AutoencoderDenso(len(FEATURE_COLUMNS), arquitetura).to(device)
    modelo_canonico.load_state_dict(estados[(escolhida_id, seed_canonico)])
    checkpoint_canonico = pasta_saida / "modelo_autoencoder_v2.pt"
    torch.save(
        {
            "schema_version": 2,
            "state_dict": estados[(escolhida_id, seed_canonico)],
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": list(FEATURE_COLUMNS),
            "architecture": arquitetura.como_dict(),
            "seed": seed_canonico,
            "score_method": "family_balanced_mse",
            "feature_weights": pesos.tolist(),
        },
        checkpoint_canonico,
    )

    pca = PCA(n_components=arquitetura.latente, svd_solver="full")
    pca.fit(dados.treino)
    pca_scores = {}
    for papel in ("treino", "validacao", "calibracao", "teste"):
        matriz = getattr(dados, papel)
        reconstruida = pca.inverse_transform(pca.transform(matriz))
        pca_scores[papel] = pontuar_residuos((matriz - reconstruida) ** 2, pesos)
    pca_limiar = limiar_ordem_finita(pca_scores["calibracao"])

    _salvar_pickle(dados.scaler, pasta_saida / "scaler.pkl")
    _salvar_pickle(pca, pasta_saida / "pca_baseline.pkl")
    baseline_path = salvar_normalizacao_baseline(
        dados.normalizacao_baseline, pasta_saida
    )
    frame_execucoes.to_csv(pasta_saida / "selecao_execucoes.csv", index=False)
    frame_resumo.to_csv(pasta_saida / "selecao_resumo.csv", index=False)

    linhas_scores = []
    for metodo, por_papel in (
        (
            "autoencoder_v2",
            {
                "calibracao": scores_cal,
                "teste": scores_teste,
            },
        ),
        ("pca", pca_scores),
    ):
        for papel, scores in por_papel.items():
            if papel not in dados.indices:
                continue
            for indice_global, score in zip(
                dados.indices[papel], scores, strict=True
            ):
                registro = dados.frame.iloc[int(indice_global)]
                linhas_scores.append(
                    {
                        "method": metodo,
                        "role": papel,
                        "experiment": registro["ensaio"],
                        "window_index": int(registro["janela_idx"]),
                        "time_center_s": float(registro["tempo_centro_s"]),
                        "score": float(score),
                    }
                )
    pd.DataFrame(linhas_scores).to_csv(
        pasta_saida / "escores_saudaveis.csv", index=False
    )

    limiar_json = {
        "schema_version": 2,
        "dataset": {"name": "GPVS-Faults", "doi": DOI_GPVS},
        "score_method": "family_balanced_mse",
        "feature_groups": {
            nome: list(colunas) for nome, colunas in FAMILIAS_FEATURES.items()
        },
        "feature_weights": dict(zip(FEATURE_COLUMNS, pesos.tolist(), strict=True)),
        "canonical_seed": seed_canonico,
        "canonical": {
            **limiar_ordem_finita(scores_cal),
            "calibration": can_seed_info["calibration"],
            "healthy_test": can_seed_info["healthy_test"],
        },
        "seed_robustness": avaliacoes_seed,
        "pca_baseline": {
            **pca_limiar,
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "calibration": resumo_excedencia(
                pca_scores["calibracao"], pca_limiar["threshold"]
            ),
            "healthy_test": resumo_excedencia(
                pca_scores["teste"], pca_limiar["threshold"]
            ),
        },
    }
    _salvar_json(pasta_saida / "limiar_v2.json", limiar_json)

    contrato = {
        "schema_version": 2,
        "created_at": agora_local().isoformat(),
        "dataset": {
            "name": "GPVS-Faults",
            "doi": DOI_GPVS,
            "features_file_sha256": _sha256(Path(arquivo_features)),
            "healthy_experiments": ["F0L", "F0M"],
            "fault_experiments_reserved": "F1L-F7M",
        },
        "sample_counts": {
            papel: int(len(getattr(dados, papel)))
            for papel in ("treino", "validacao", "calibracao", "teste")
        },
        "split": {
            "strategy": dados.split["estrategia"],
            "purge_windows": dados.split["purge_janelas"],
            "limits": dados.split["limites"],
        },
        "selection": {
            "seeds": list(SEEDS),
            "criterion": "minimum_median_healthy_validation_loss",
            "parsimony_rule": "fewest_parameters_within_2pct_of_best_median",
            "representative_seed_rule": "closest_to_selected_architecture_median",
            "fault_data_used": False,
            "selected_architecture": arquitetura.como_dict(),
            "canonical_seed": seed_canonico,
        },
        "training": {
            "epochs_max": epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "patience": PACIENCIA,
            "optimizer": "AdamW",
            "activation": "LeakyReLU",
            "latent_activation": "linear",
            "output_activation": "linear",
            "loss": "family_balanced_mse",
            "device": str(device),
        },
        "artifacts": {
            "model_sha256": _sha256(checkpoint_canonico),
            "scaler_sha256": _sha256(pasta_saida / "scaler.pkl"),
            "pca_sha256": _sha256(pasta_saida / "pca_baseline.pkl"),
            "baseline_normalization_sha256": _sha256(baseline_path),
        },
        "duration_s": time.perf_counter() - inicio_total,
    }
    _salvar_json(pasta_saida / "contrato_experimento.json", contrato)

    plotar_selecao_modelo(
        frame_execucoes,
        frame_resumo,
        escolhida_id,
        pasta_saida / "selecao_arquitetura.png",
    )
    plotar_calibracao(
        scores_cal,
        scores_teste,
        float(limiar_json["canonical"]["threshold"]),
        {
            "calibracao": can_seed_info["calibration"],
            "teste": can_seed_info["healthy_test"],
        },
        pasta_saida / "calibracao_limiar.png",
    )
    return {"contract": contrato, "threshold": limiar_json}


def regenerar_figuras_saudaveis(pasta_saida: Path = PASTA_SAIDA) -> None:
    """Reconstrói as figuras F0 a partir dos artefatos tabulares publicados."""

    pasta_saida = Path(pasta_saida)
    execucoes = pd.read_csv(pasta_saida / "selecao_execucoes.csv")
    resumo = pd.read_csv(pasta_saida / "selecao_resumo.csv")
    info = json.loads((pasta_saida / "limiar_v2.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(pasta_saida / "escores_saudaveis.csv")
    scores_ae = scores[scores["method"].eq("autoencoder_v2")]
    calibracao = scores_ae[scores_ae["role"].eq("calibracao")]["score"].to_numpy()
    teste = scores_ae[scores_ae["role"].eq("teste")]["score"].to_numpy()
    selecionadas = resumo[resumo["selecionada"].astype(bool)]
    if len(selecionadas) != 1:
        raise ValueError("Resumo deve declarar exatamente uma arquitetura selecionada")
    escolhida = str(selecionadas.iloc[0]["arquitetura"])
    plotar_selecao_modelo(
        execucoes,
        resumo,
        escolhida,
        pasta_saida / "selecao_arquitetura.png",
    )
    plotar_calibracao(
        calibracao,
        teste,
        float(info["canonical"]["threshold"]),
        {
            "calibracao": info["canonical"]["calibration"],
            "teste": info["canonical"]["healthy_test"],
        },
        pasta_saida / "calibracao_limiar.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    resultado = executar_experimento(epochs=args.epochs)
    selecao = resultado["contract"]["selection"]
    teste = resultado["threshold"]["canonical"]["healthy_test"]
    print(
        f"Selecionado {selecao['selected_architecture']['id']} / "
        f"seed {selecao['canonical_seed']} | FP teste = {teste['taxa_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
