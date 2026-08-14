"""Avaliação congelada do autoencoder V2 nos ensaios de falha GPVS-Faults."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from src.core.config import RAIZ_PROJETO
from src.core.seguranca import carregar_pickle_com_sidecar
from src.core.tempo import agora_local
from src.ml.estatistica import intervalo_wilson
from src.ml.gpvs import (
    DOI_GPVS,
    FALHAS,
    FEATURE_COLUMNS,
    MODOS,
    PASTA_GPVS,
    SUSTAINED_WINDOWS,
    arquivos_gpvs,
    extrair_features_gpvs,
)
from src.ml.gpvs_principal import (
    carregar_normalizacao_baseline,
    normalizar_comissionamento,
)

from .graficos import (
    plotar_contribuicoes_familias,
    plotar_curvas_macro,
    plotar_desempenho_por_ensaio,
    plotar_mapa_ponto_operacional,
    plotar_matrizes_confusao,
    plotar_series_temporais,
)
from .modelo import (
    FAMILIAS_FEATURES,
    Arquitetura,
    AutoencoderDenso,
    pontuar_residuos,
    residuos_quadraticos,
)

PASTA_AE = Path(RAIZ_PROJETO) / "resultados" / "v2" / "autoencoder"
ARQUIVO_CACHE = (
    Path(RAIZ_PROJETO) / "dados" / "processados" / "features_gpvs_falhas.parquet"
)
N_BOOTSTRAP = 20_000
SEED_BOOTSTRAP = 20260813


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def metricas_cenario(
    features: pd.DataFrame,
    indice_anomalia: np.ndarray,
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    *,
    persistencia: int = SUSTAINED_WINDOWS,
) -> dict:
    """Métricas binárias no ponto 1, com contagens e atraso sustentado."""

    pre = np.asarray(pre_indices, dtype=int)
    pos = np.asarray(post_indices, dtype=int)
    indice = np.asarray(indice_anomalia, dtype=float)
    if not len(pre) or not len(pos) or len(indice) != len(features):
        raise ValueError("Métricas exigem vetores pré/pós não vazios e alinhados")
    pred = indice > 1.0
    fp, tn = int(pred[pre].sum()), int((~pred[pre]).sum())
    tp, fn = int(pred[pos].sum()), int((~pred[pos]).sum())
    sensibilidade = tp / len(pos)
    especificidade = tn / len(pre)
    y = np.r_[np.zeros(len(pre), dtype=int), np.ones(len(pos), dtype=int)]
    scores = np.r_[indice[pre], indice[pos]]
    predicoes = np.r_[pred[pre], pred[pos]].astype(int)
    ci_sens = intervalo_wilson(tp, len(pos))
    ci_esp = intervalo_wilson(tn, len(pre))

    atraso = None
    tempo_falha = float(features.iloc[int(pos[0])]["tempo_inicio_s"])
    for deslocamento in range(len(pos) - persistencia + 1):
        bloco = pos[deslocamento : deslocamento + persistencia]
        if np.all(np.diff(bloco) == 1) and np.all(pred[bloco]):
            atraso = float(
                features.iloc[int(bloco[0])]["tempo_inicio_s"] - tempo_falha
            )
            break
    return {
        "n_pre_test": int(len(pre)),
        "n_post_test": int(len(pos)),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "auc_roc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "sensitivity": float(sensibilidade),
        "sensitivity_ci95_low": float(ci_sens[0]),
        "sensitivity_ci95_high": float(ci_sens[1]),
        "specificity": float(especificidade),
        "specificity_ci95_low": float(ci_esp[0]),
        "specificity_ci95_high": float(ci_esp[1]),
        "balanced_accuracy": float((sensibilidade + especificidade) / 2),
        "mcc": float(matthews_corrcoef(y, predicoes)),
        "sustained_detection": atraso is not None,
        "detection_delay_from_nominal_midpoint_s": atraso,
        "median_index_pre": float(np.median(indice[pre])),
        "median_index_post": float(np.median(indice[pos])),
    }


def _bootstrap_media(valores, *, seed: int, n_boot: int = N_BOOTSTRAP) -> dict:
    vetor = np.asarray(valores, dtype=float)
    if not len(vetor) or not np.isfinite(vetor).all():
        raise ValueError("Bootstrap exige métricas finitas")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vetor), size=(n_boot, len(vetor)))
    medias = vetor[indices].mean(axis=1)
    return {
        "mean": float(vetor.mean()),
        "ci95_low": float(np.percentile(medias, 2.5)),
        "ci95_high": float(np.percentile(medias, 97.5)),
        "n_experiments": int(len(vetor)),
        "bootstrap_resamples": int(n_boot),
    }


def resumir_macros(cenarios: pd.DataFrame) -> dict:
    """Resume por ensaio, preservando cada cenário como unidade amostral."""

    metricas = (
        "auc_roc",
        "average_precision",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "mcc",
    )
    resultado = {}
    for metodo, bloco in cenarios.groupby("method", sort=True):
        resultado[metodo] = {}
        for metrica in metricas:
            seed = SEED_BOOTSTRAP + sum(ord(c) for c in metodo + metrica)
            resultado[metodo][metrica] = _bootstrap_media(
                bloco[metrica], seed=seed
            )
        resultado[metodo]["sustained_detection"] = {
            "n": int(bloco["sustained_detection"].sum()),
            "total": int(len(bloco)),
            "rate": float(bloco["sustained_detection"].mean()),
        }
    return resultado


def diferenca_pareada(
    cenarios: pd.DataFrame,
    metodo_a: str,
    metodo_b: str,
    metrica: str,
    *,
    seed: int = SEED_BOOTSTRAP,
    n_boot: int = N_BOOTSTRAP,
) -> dict:
    """IC bootstrap da diferença A-B, reamostrando pares de ensaios."""

    tabela = cenarios.pivot(index="experiment", columns="method", values=metrica)
    if metodo_a not in tabela or metodo_b not in tabela:
        raise ValueError("Métodos ausentes para comparação pareada")
    delta = (tabela[metodo_a] - tabela[metodo_b]).to_numpy(dtype=float)
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Comparação pareada exige diferenças finitas")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    medias = delta[indices].mean(axis=1)
    baixo, alto = np.percentile(medias, (2.5, 97.5))
    return {
        "contrast": f"{metodo_a}_minus_{metodo_b}",
        "metric": metrica,
        "mean_difference": float(delta.mean()),
        "ci95_low": float(baixo),
        "ci95_high": float(alto),
        "n_pairs": int(len(delta)),
        "wins_a": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "wins_b": int(np.sum(delta < 0)),
        "bootstrap_resamples": int(n_boot),
        "ci_includes_zero": bool(baixo <= 0 <= alto),
    }


def _carregar_features_falha(
    diretorio: Path,
    arquivo_cache: Path,
) -> tuple[dict[str, pd.DataFrame], dict]:
    arquivos = arquivos_gpvs(diretorio)
    nomes = [f"F{i}{modo}" for i in range(1, 8) for modo in "LM"]
    arquivo_cache = Path(arquivo_cache)
    if arquivo_cache.exists():
        completo = pd.read_parquet(arquivo_cache)
        presentes = set(completo["ensaio"].astype(str).unique())
        if set(nomes).issubset(presentes) and set(FEATURE_COLUMNS).issubset(completo):
            frames = {
                nome: completo[completo["ensaio"].eq(nome)].reset_index(drop=True)
                for nome in nomes
            }
        else:
            completo = None
    else:
        completo = None

    if completo is None:
        partes = []
        frames = {}
        for nome in nomes:
            frame, _ = extrair_features_gpvs(pd.read_csv(arquivos[nome]), nome)
            frames[nome] = frame
            partes.append(frame)
        arquivo_cache.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(partes, ignore_index=True).to_parquet(arquivo_cache, index=False)

    inventario = {
        nome: {
            "sha256": _sha256(arquivos[nome]),
            "size_bytes": int(arquivos[nome].stat().st_size),
            "windows": int(len(frames[nome])),
        }
        for nome in nomes
    }
    return frames, inventario


def _carregar_modelo(caminho: Path):
    import torch

    checkpoint = torch.load(caminho, map_location="cpu", weights_only=False)
    arquitetura = Arquitetura.de_dict(checkpoint["architecture"])
    modelo = AutoencoderDenso(int(checkpoint["n_features"]), arquitetura)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    return modelo, checkpoint


def _linha_base(nome: str) -> dict:
    falha = int(nome[1])
    modo = nome[2]
    return {
        "experiment": nome,
        "fault": falha,
        "fault_type": FALHAS[falha],
        "mode": modo,
        "mode_name": MODOS[modo],
    }


def _papel_janela(
    n: int,
    baseline: np.ndarray,
    pre: np.ndarray,
    pos: np.ndarray,
) -> np.ndarray:
    papeis = np.full(n, "excluded", dtype="U12")
    papeis[np.asarray(baseline, dtype=int)] = "baseline"
    papeis[np.asarray(pre, dtype=int)] = "negative"
    papeis[np.asarray(pos, dtype=int)] = "positive"
    return papeis


def _resumir_contribuicoes(
    nome: str,
    contribuicoes: np.ndarray,
    pre: np.ndarray,
    pos: np.ndarray,
) -> dict:
    linha = _linha_base(nome)
    for familia, colunas in FAMILIAS_FEATURES.items():
        indices = [FEATURE_COLUMNS.index(coluna) for coluna in colunas]
        por_janela = contribuicoes[:, indices].sum(axis=1)
        mediana_pre = max(float(np.median(por_janela[pre])), np.finfo(float).tiny)
        mediana_pos = max(float(np.median(por_janela[pos])), np.finfo(float).tiny)
        linha[familia] = mediana_pos / mediana_pre
        linha[f"{familia}_median_pre"] = mediana_pre
        linha[f"{familia}_median_post"] = mediana_pos
    return linha


def _gerar_relatorio(resultado: dict, caminho: Path) -> None:
    macro = resultado["macro_summary"]
    ae = macro["autoencoder_v2"]
    pca = macro["pca"]
    diferencas = resultado["paired_comparison_autoencoder_v2_minus_pca"]
    linhas = [
        "# Avaliação experimental do autoencoder denso V2",
        "",
        f"Gerado em `{resultado['created_at']}` com o GPVS-Faults (DOI {DOI_GPVS}).",
        "",
        "## Resultado principal",
        "",
        (
            f"O autoencoder obteve AUC-ROC macro {ae['auc_roc']['mean']:.3f} "
            f"(IC95% {ae['auc_roc']['ci95_low']:.3f}-"
            f"{ae['auc_roc']['ci95_high']:.3f}), sensibilidade "
            f"{ae['sensitivity']['mean']:.3f} e especificidade "
            f"{ae['specificity']['mean']:.3f} no limiar congelado."
        ),
        (
            f"O baseline PCA obteve AUC-ROC macro {pca['auc_roc']['mean']:.3f}, "
            f"sensibilidade {pca['sensitivity']['mean']:.3f} e especificidade "
            f"{pca['specificity']['mean']:.3f}."
        ),
        (
            "Na comparação pareada, a diferença AE-PCA foi "
            f"{diferencas['auc_roc']['mean_difference']:+.3f} para AUC-ROC "
            f"(IC95% {diferencas['auc_roc']['ci95_low']:+.3f} a "
            f"{diferencas['auc_roc']['ci95_high']:+.3f}) e "
            f"{diferencas['balanced_accuracy']['mean_difference']:+.3f} para "
            "acurácia balanceada. O PCA teve AUC e especificidade maiores; o "
            "autoencoder teve sensibilidade maior. Para acurácia balanceada e "
            "MCC, os intervalos incluem zero, sem superioridade global."
        ),
        "",
        "## Protocolo",
        "",
        "- Arquitetura, semente, scaler e limiar foram congelados usando apenas F0L/F0M.",
        "- F1L-F7M foram abertos somente após o congelamento do detector.",
        "- A fonte situa a introdução manual da falha na metade do registro.",
        "- A linha de 50% é nominal; os CSVs não contêm um canal instrumentado de disparo.",
        "- A primeira metade do trecho anterior à fronteira ajusta apenas o baseline local.",
        "- A segunda metade anterior mede especificidade; o trecho posterior mede sensibilidade.",
        "- IC95% macro: bootstrap dos 14 ensaios, que são a unidade de inferência.",
        "",
        "## Interpretação",
        "",
        (
            "O GPVS valida detecção de mudanças de regime em bancada. Ele não contém "
            "tempos até falha, censura ou reparo e, portanto, não estima confiabilidade "
            "física, taxa de falha, Weibull temporal ou RUL."
        ),
        (
            "Somente F1 representa diretamente falha total em IGBT. Os demais modos "
            "não devem ser renomeados como falha de contator ou fusível da FMECA."
        ),
    ]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def executar_avaliacao(
    pasta_ae: Path = PASTA_AE,
    diretorio: Path = PASTA_GPVS,
    arquivo_cache: Path = ARQUIVO_CACHE,
) -> dict:
    """Avalia AE canônico, ensemble diagnóstico e PCA nos 14 ensaios."""

    pasta_ae = Path(pasta_ae)
    info_limiar = json.loads((pasta_ae / "limiar_v2.json").read_text(encoding="utf-8"))
    contrato = json.loads(
        (pasta_ae / "contrato_experimento.json").read_text(encoding="utf-8")
    )
    if contrato["selection"]["fault_data_used"] is not False:
        raise ValueError("Contrato não comprova isolamento dos ensaios de falha")
    seed_canonico = int(info_limiar["canonical_seed"])
    pesos = np.asarray(list(info_limiar["feature_weights"].values()), dtype=float)
    if list(info_limiar["feature_weights"]) != list(FEATURE_COLUMNS):
        raise ValueError("Ordem de pesos diverge das features canônicas")

    modelos = {}
    checkpoints = {}
    limites = {}
    for item in info_limiar["seed_robustness"]:
        seed = int(item["seed"])
        modelo, checkpoint = _carregar_modelo(pasta_ae / f"modelo_seed_{seed}.pt")
        if list(checkpoint["feature_columns"]) != list(FEATURE_COLUMNS):
            raise ValueError(f"Checkpoint da seed {seed} usa features incompatíveis")
        modelos[seed] = modelo
        checkpoints[seed] = checkpoint
        limites[seed] = float(item["threshold"])

    scaler = carregar_pickle_com_sidecar(pasta_ae / "scaler.pkl")
    pca = carregar_pickle_com_sidecar(pasta_ae / "pca_baseline.pkl")
    normalizacao = carregar_normalizacao_baseline(pasta_ae)
    limite_pca = float(info_limiar["pca_baseline"]["threshold"])
    features_por_ensaio, inventario = _carregar_features_falha(
        Path(diretorio), Path(arquivo_cache)
    )

    linhas_cenarios = []
    linhas_seeds = []
    linhas_scores = []
    linhas_contribuicoes = []
    for nome, features in features_por_ensaio.items():
        matriz, pre, pos, meta = normalizar_comissionamento(features, normalizacao)
        n_baseline = int(meta["n_baseline"])
        baseline = np.arange(n_baseline, dtype=int)
        papeis = _papel_janela(len(features), baseline, pre, pos)
        z = scaler.transform(matriz).astype(np.float32)
        indices_seed = {}
        residuos_canonicos = None
        for seed, modelo in modelos.items():
            residuos = residuos_quadraticos(modelo, z)
            indice = pontuar_residuos(residuos, pesos) / limites[seed]
            indices_seed[seed] = indice
            linha = {
                **_linha_base(nome),
                "seed": seed,
                **meta,
                **metricas_cenario(features, indice, pre, pos),
            }
            linhas_seeds.append(linha)
            if seed == seed_canonico:
                residuos_canonicos = residuos

        indice_canonico = indices_seed[seed_canonico]
        indice_ensemble = np.median(np.vstack(list(indices_seed.values())), axis=0)
        reconstruida_pca = pca.inverse_transform(pca.transform(z))
        indice_pca = pontuar_residuos((z - reconstruida_pca) ** 2, pesos) / limite_pca
        for metodo, indice in (
            ("autoencoder_v2", indice_canonico),
            ("autoencoder_ensemble", indice_ensemble),
            ("pca", indice_pca),
        ):
            linhas_cenarios.append(
                {
                    **_linha_base(nome),
                    "method": metodo,
                    "seed": seed_canonico if metodo == "autoencoder_v2" else None,
                    **meta,
                    **metricas_cenario(features, indice, pre, pos),
                }
            )
            for i, valor in enumerate(indice):
                linhas_scores.append(
                    {
                        **_linha_base(nome),
                        "method": metodo,
                        "window_index": int(features.iloc[i]["janela_idx"]),
                        "phase": str(features.iloc[i]["fase"]),
                        "evaluation_role": str(papeis[i]),
                        "time_center_s": float(features.iloc[i]["tempo_centro_s"]),
                        "time_from_nominal_midpoint_s": float(
                            features.iloc[i]["tempo_centro_s"]
                            - features.iloc[int(pos[0])]["tempo_inicio_s"]
                        ),
                        "anomaly_index": float(valor),
                    }
                )
        contribuicoes = residuos_canonicos * pesos
        linhas_contribuicoes.append(
            _resumir_contribuicoes(nome, contribuicoes, pre, pos)
        )

    cenarios = pd.DataFrame(linhas_cenarios)
    seeds = pd.DataFrame(linhas_seeds)
    scores = pd.DataFrame(linhas_scores)
    contribuicoes = pd.DataFrame(linhas_contribuicoes)
    macros = resumir_macros(cenarios)
    comparacao_pareada = {
        metrica: diferenca_pareada(
            cenarios,
            "autoencoder_v2",
            "pca",
            metrica,
            seed=SEED_BOOTSTRAP + sum(ord(c) for c in metrica),
        )
        for metrica in (
            "auc_roc",
            "average_precision",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "mcc",
        )
    }
    macros_seeds = {
        str(seed): resumir_macros(
            seeds[seeds["seed"].eq(seed)].assign(method="autoencoder_v2")
        )["autoencoder_v2"]
        for seed in sorted(modelos)
    }

    cenarios.to_csv(pasta_ae / "avaliacao_cenarios.csv", index=False)
    seeds.to_csv(pasta_ae / "avaliacao_seeds.csv", index=False)
    scores.to_csv(pasta_ae / "avaliacao_scores.csv", index=False)
    contribuicoes.to_csv(pasta_ae / "contribuicoes_familias.csv", index=False)
    resultado = {
        "schema_version": 2,
        "created_at": agora_local().isoformat(),
        "dataset": {
            "name": "GPVS-Faults",
            "doi": DOI_GPVS,
            "healthy_training_trials": ["F0L", "F0M"],
            "held_out_fault_trials": [f"F{i}{modo}" for i in range(1, 8) for modo in "LM"],
            "raw_inventory": inventario,
        },
        "protocol": {
            "inference_unit": "experiment",
            "window_duration_s": 0.02,
            "window_overlap": 0,
            "commissioning_baseline": "first_half_of_pre_fault",
            "negative_test": "second_half_of_pre_fault",
            "positive_test": "all_post_fault_windows",
            "fault_boundary_fraction": 0.50,
            "fault_boundary_source": (
                "Mendeley Data description: faults introduced manually halfway"
            ),
            "fault_boundary_semantics": (
                "nominal_record_midpoint_not_instrumented_trigger"
            ),
            "sustained_detection_windows": SUSTAINED_WINDOWS,
            "threshold_comparison": "anomaly_index > 1",
            "architecture_selected_without_fault_trials": True,
            "field_prevalence_estimated": False,
        },
        "canonical_seed": seed_canonico,
        "macro_summary": macros,
        "paired_comparison_autoencoder_v2_minus_pca": comparacao_pareada,
        "seed_robustness": macros_seeds,
        "limitations": [
            "Janelas do mesmo ensaio são autocorrelacionadas.",
            "Curvas ROC/PR por janela são descritivas; IC macro usa ensaios.",
            "GPVS não contém vida até falha, censura ou tempos de reparo.",
            "F1 é a única falha diretamente rotulada como IGBT.",
        ],
        "local_artifacts": {
            f"model_seed_{seed}_sha256": _sha256(
                pasta_ae / f"modelo_seed_{seed}.pt"
            )
            for seed in sorted(modelos)
        },
    }
    (pasta_ae / "avaliacao_experimental.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _gerar_relatorio(resultado, pasta_ae / "avaliacao_experimental.md")

    plotar_desempenho_por_ensaio(cenarios, pasta_ae / "desempenho_por_ensaio.png")
    plotar_mapa_ponto_operacional(cenarios, pasta_ae / "mapa_ponto_operacional.png")
    plotar_matrizes_confusao(cenarios, pasta_ae / "matrizes_confusao.png")
    plotar_curvas_macro(scores, macros, pasta_ae / "curvas_roc_pr_macro.png")
    plotar_series_temporais(scores, pasta_ae / "series_temporais.png")
    plotar_contribuicoes_familias(
        contribuicoes, pasta_ae / "contribuicoes_familias.png"
    )
    return resultado


def main() -> None:
    resultado = executar_avaliacao()
    macro = resultado["macro_summary"]["autoencoder_v2"]
    print(
        f"AE V2: AUC={macro['auc_roc']['mean']:.3f}, "
        f"sens={macro['sensitivity']['mean']:.3f}, "
        f"esp={macro['specificity']['mean']:.3f}"
    )


if __name__ == "__main__":
    main()
