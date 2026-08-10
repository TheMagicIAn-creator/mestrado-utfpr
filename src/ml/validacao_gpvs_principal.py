"""Validação E2/E3 do detector principal no GPVS-Faults.

O detector é ajustado uma única vez em F0L/F0M. Esta etapa primeiro executa a
validação sintética E2 orientada pela FMECA e depois aplica o mesmo modelo,
scaler, escore e limiar congelados aos ensaios reais F1L-F7M (E3). Cada ensaio
usa apenas a primeira metade pré-falha como baseline de comissionamento; não há
retreino do modelo nem recalibração do limiar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.config import RAIZ_PROJETO
from src.core.logs import adaptar_logger_como_print, get_logger
from src.core.seguranca import carregar_pickle_com_sidecar
from src.core.tempo import agora_local
from src.ml import escore_anomalia as ea
from src.ml.estatistica import intervalo_wilson
from src.ml.estilo_graficos import (
    COR_ALERTA,
    COR_METODO,
    COR_NEUTRA,
    COR_REFERENCIA,
    COR_TEXTO_SEC,
    aplicar_estilo,
    salvar_figura,
)
from src.ml.gpvs import (
    DOI_GPVS,
    FALHAS,
    FALHAS_CURTAS,
    FEATURE_COLUMNS,
    MODOS,
    PASTA_GPVS,
    PURGE_WINDOWS,
    SHA256_ZIP_OFICIAL,
    SUSTAINED_WINDOWS,
    arquivos_gpvs,
    extrair_features_gpvs,
)
from src.ml.gpvs_principal import (
    carregar_normalizacao_baseline,
    normalizar_comissionamento,
)
from src.ml.proveniencia import (
    gerar_manifesto,
    salvar_manifesto,
    sha256_arquivo_texto_normalizado,
)

aplicar_estilo()

PASTA_AE = Path(RAIZ_PROJETO) / "resultados" / "autoencoder"
PASTA_SAIDA = Path(RAIZ_PROJETO) / "resultados" / "gpvs"
N_BOOTSTRAP = 20_000
SEED_BOOTSTRAP = 20260809
# Espelho literal dos parâmetros da subetapa E2 para o manifesto do estágio
# composto. Testes de proveniência impedem divergência silenciosa.
SEVS_VALIDACAO = [0.30, 0.50, 1.00]
N_JANELAS_SAUDAVEL = 40
N_JANELAS_FALHA = 40
PREVALENCIA_RARA = 0.05

_log = adaptar_logger_como_print(get_logger("validacao_gpvs_principal"))


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _carregar_detector(pasta_ae: Path = PASTA_AE):
    import torch

    from src.ml.autoencoder import Autoencoder

    pasta_ae = Path(pasta_ae)
    checkpoint = torch.load(
        pasta_ae / "modelo_autoencoder.pt", map_location="cpu", weights_only=False
    )
    scaler = carregar_pickle_com_sidecar(pasta_ae / "scaler.pkl")
    info = json.loads((pasta_ae / "limiar.json").read_text(encoding="utf-8"))
    estatistica = ea.carregar_estatistica(pasta_ae)
    normalizacao = carregar_normalizacao_baseline(pasta_ae)
    colunas = list(checkpoint["colunas_feat"])
    if colunas != list(FEATURE_COLUMNS):
        raise ValueError(
            "O modelo principal não usa o contrato de 24 features do GPVS-Faults"
        )
    if (info.get("dataset") or {}).get("name") != "GPVS-Faults":
        raise ValueError("limiar.json não identifica o GPVS-Faults como dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(
        int(checkpoint["n_features"]), int(checkpoint["latente_dim"])
    ).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    return modelo, scaler, info, estatistica, normalizacao, device


def _pontuar(modelo, scaler, matriz: np.ndarray, info: dict, estatistica, device):
    matriz = np.asarray(matriz, dtype=np.float32)
    normalizada = scaler.transform(matriz).astype(np.float32)
    residuos = ea.residuo_por_feature(modelo, normalizada, device)
    metodo = info.get("score_method", info.get("metodo_escore", "mse"))
    top_k = int(info.get("top_k") or info.get("k_localizado") or 5)
    return ea.pontuar(residuos, estatistica, metodo=metodo, k=top_k)


def _metricas_cenario(
    features: pd.DataFrame,
    scores: np.ndarray,
    limiar: float,
    *,
    persistencia: int = SUSTAINED_WINDOWS,
    pre_indices: np.ndarray | None = None,
    post_indices: np.ndarray | None = None,
) -> dict:
    pre = (
        np.flatnonzero(features["fase"].eq("pre_falha").to_numpy())
        if pre_indices is None else np.asarray(pre_indices, dtype=int)
    )
    pos = (
        np.flatnonzero(features["fase"].eq("pos_falha").to_numpy())
        if post_indices is None else np.asarray(post_indices, dtype=int)
    )
    if not len(pre) or not len(pos):
        raise ValueError("O ensaio precisa conter janelas pré e pós-falha")
    scores = np.asarray(scores, dtype=float)
    pred = scores > float(limiar)
    fp = int(pred[pre].sum())
    tp = int(pred[pos].sum())
    especificidade = 1.0 - fp / len(pre)
    sensibilidade = tp / len(pos)
    y = np.r_[np.zeros(len(pre), dtype=int), np.ones(len(pos), dtype=int)]
    score_auc = np.r_[scores[pre], scores[pos]]

    atraso = None
    tempo_falha = float(features.iloc[int(pos[0])]["tempo_inicio_s"])
    for deslocamento in range(len(pos) - persistencia + 1):
        bloco = pos[deslocamento:deslocamento + persistencia]
        if np.all(np.diff(bloco) == 1) and np.all(pred[bloco]):
            atraso = float(
                features.iloc[int(bloco[0])]["tempo_inicio_s"] - tempo_falha
            )
            break

    ci_sens = intervalo_wilson(tp, len(pos))
    ci_esp = intervalo_wilson(len(pre) - fp, len(pre))
    return {
        "n_pre_fault": int(len(pre)),
        "n_post_fault": int(len(pos)),
        "false_positives_pre": fp,
        "true_positives_post": tp,
        "auc": float(roc_auc_score(y, score_auc)),
        "sensitivity": float(sensibilidade),
        "sensitivity_ci95_low": float(ci_sens[0]),
        "sensitivity_ci95_high": float(ci_sens[1]),
        "specificity": float(especificidade),
        "specificity_ci95_low": float(ci_esp[0]),
        "specificity_ci95_high": float(ci_esp[1]),
        "balanced_accuracy": float((sensibilidade + especificidade) / 2.0),
        "sustained_detection": atraso is not None,
        "detection_delay_s": atraso,
        "median_score_pre": float(np.median(scores[pre])),
        "median_score_post": float(np.median(scores[pos])),
    }


def _bootstrap_media(valores, *, seed: int, n_boot: int = N_BOOTSTRAP) -> dict:
    valores = np.asarray(valores, dtype=float)
    if not len(valores) or not np.isfinite(valores).all():
        raise ValueError("Bootstrap exige valores finitos")
    rng = np.random.default_rng(seed)
    amostras = valores[rng.integers(0, len(valores), size=(n_boot, len(valores)))]
    medias = amostras.mean(axis=1)
    return {
        "mean": float(valores.mean()),
        "ci95_low": float(np.percentile(medias, 2.5)),
        "ci95_high": float(np.percentile(medias, 97.5)),
        "n_experiments": int(len(valores)),
        "bootstrap_resamples": int(n_boot),
    }


def _resumir_macros(cenarios: list[dict]) -> dict:
    resumo = {}
    for escopo, linhas in {
        "all": cenarios,
        "L": [r for r in cenarios if r["mode"] == "L"],
        "M": [r for r in cenarios if r["mode"] == "M"],
    }.items():
        resumo[escopo] = {}
        for metrica in ("auc", "sensitivity", "specificity", "balanced_accuracy"):
            seed = SEED_BOOTSTRAP + sum(ord(c) for c in escopo + metrica)
            resumo[escopo][metrica] = _bootstrap_media(
                [r[metrica] for r in linhas], seed=seed
            )
    return {"canonical_ae": resumo}


def _salvar_tabela(df: pd.DataFrame, pasta: Path) -> tuple[Path, Path]:
    csv_path = pasta / "validacao_gpvs_cenarios.csv"
    md_path = pasta / "validacao_gpvs_cenarios.md"
    df.to_csv(csv_path, index=False)
    colunas = [
        "experiment", "fault_type", "mode", "auc", "sensitivity",
        "specificity", "balanced_accuracy", "detection_delay_s",
    ]
    tabela = df[colunas].copy().rename(columns={
        "experiment": "Ensaio", "fault_type": "Falha", "mode": "Modo",
        "auc": "AUC", "sensitivity": "Sensibilidade",
        "specificity": "Especificidade", "balanced_accuracy": "Acurácia balanceada",
        "detection_delay_s": "Atraso sustentado (s)",
    })
    for coluna in tabela.columns[3:]:
        tabela[coluna] = tabela[coluna].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.3f}"
        )
    cab = "| " + " | ".join(tabela.columns) + " |"
    sep = "| " + " | ".join("---" for _ in tabela.columns) + " |"
    linhas = [
        "| " + " | ".join(str(v).replace("|", "\\|") for v in linha) + " |"
        for linha in tabela.itertuples(index=False, name=None)
    ]
    md_path.write_text("\n".join([cab, sep, *linhas]) + "\n", encoding="utf-8")
    return csv_path, md_path


def _gerar_relatorio(resultado: dict, caminho: Path) -> None:
    macro = resultado["macro_summary"]["canonical_ae"]["all"]
    cenarios = resultado["scenario_results"]
    melhor = max(cenarios, key=lambda item: item["auc"])
    menor = min(cenarios, key=lambda item: item["auc"])
    detectados = sum(item["sustained_detection"] for item in cenarios)
    auc = macro["auc"]
    linhas = [
        "# Validação experimental GPVS-Faults (E3 de bancada)",
        "",
        f"Gerado em `{resultado['created_at']}`. Dataset: DOI {DOI_GPVS}.",
        "",
        "## Resultado principal",
        "",
        (
            "O Autoencoder canônico, treinado e calibrado somente nos ensaios "
            "saudáveis F0L/F0M, obteve AUC macro "
            f"{auc['mean']:.3f} (IC95% {auc['ci95_low']:.3f}-"
            f"{auc['ci95_high']:.3f}) nos 14 ensaios de falha. A sensibilidade "
            f"macro no limiar congelado foi {macro['sensitivity']['mean']:.3f} e "
            f"a especificidade pré-falha, {macro['specificity']['mean']:.3f}."
        ),
        "",
        f"Houve detecção sustentada em {detectados}/14 ensaios. O maior AUC ocorreu "
        f"em {melhor['experiment']} ({melhor['auc']:.3f}) e o menor em "
        f"{menor['experiment']} ({menor['auc']:.3f}). Esses extremos são descritivos; "
        "não houve seleção de cenários por desempenho.",
        "",
        "## Protocolo",
        "",
        "- Uma única fonte de dados: GPVS-Faults.",
        "- F0L/F0M: scaler, treino, early stopping, calibração do limiar e teste saudável.",
        (
            "- F1L-F7M: validação E3 com pesos e limiar congelados; a primeira "
            "metade pré-falha define o baseline de comissionamento e a segunda "
            "metade pré-falha mede a especificidade."
        ),
        "- 24 features de sensores primários em janelas não sobrepostas de um ciclo de 50 Hz.",
        (
            "- A taxa usada é inferida do vetor `Time` (aproximadamente 10 kHz); "
            "o período de 9,9989 µs informado no manual diverge dos CSVs e é "
            "mantido apenas como ressalva documental."
        ),
        "- A unidade do IC95% macro é o ensaio (bootstrap de 14 ensaios), não a janela.",
        "- IC95% por cenário é Wilson por janela e deve ser lido com cautela devido à autocorrelação.",
        "",
        "## Limites de evidência",
        "",
        "E3 significa validação experimental em bancada, não validação de campo. O "
        "detector sinaliza desvio do padrão saudável; não demonstra causalidade do "
        "componente e não transforma os ensaios em tempos de vida para Weibull físico.",
        "",
        "## Fonte",
        "",
        f"- GPVS-Faults: https://doi.org/{DOI_GPVS}",
    ]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _plotar(df: pd.DataFrame, scores: pd.DataFrame, macros: dict, pasta: Path) -> list[Path]:
    saidas = []
    fig, axes = plt.subplots(7, 2, figsize=(14, 21), layout="constrained")
    for falha in range(1, 8):
        for coluna, modo in enumerate("LM"):
            nome = f"F{falha}{modo}"
            ax = axes[falha - 1, coluna]
            bloco = scores[scores["experiment"].eq(nome)]
            indice_plot = np.maximum(
                bloco["anomaly_index"].to_numpy(dtype=float),
                np.finfo(float).tiny,
            )
            ax.plot(
                bloco["time_center_s"], indice_plot,
                color=COR_METODO, linewidth=1.0,
            )
            pos = bloco[bloco["phase"].eq("post_fault")]
            if not pos.empty:
                ax.axvline(float(pos["time_start_s"].iloc[0]), color=COR_ALERTA, linestyle="--", linewidth=1.1)
            ax.axhline(1.0, color=COR_REFERENCIA, linestyle=":", linewidth=1.1)
            ax.set_yscale("log")
            ax.set_title(f"{nome} - {FALHAS_CURTAS[falha]}", fontsize=9)
            ax.set_ylabel("Índice de anomalia")
            if falha == 7:
                ax.set_xlabel("Tempo (s)")
    fig.suptitle("GPVS-Faults: resposta do detector canônico congelado")
    caminho = pasta / "gpvs_series_temporais.png"
    salvar_figura(
        fig,
        caminho,
        (
            "Linha horizontal: limiar; linha vertical: início pós-falha. "
            "Pesos e limiar congelados, com baseline de comissionamento por ensaio."
        ),
    )
    saidas.append(caminho)

    labels = df["experiment"].tolist()[::-1]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True, layout="constrained")
    for ax, (metrica, titulo), cor in zip(
        axes,
        (("auc", "AUC"), ("sensitivity", "Sensibilidade pós-falha"), ("specificity", "Especificidade pré-falha")),
        (COR_METODO, COR_ALERTA, COR_NEUTRA),
        strict=True,
    ):
        valores = df[metrica].to_numpy()[::-1]
        ax.barh(y, valores, color=cor, height=0.62)
        ax.set_xlim(0, 1.02)
        ax.set_title(titulo)
        ax.set_xlabel("Proporção")
    axes[0].set_yticks(y, labels)
    fig.suptitle("Desempenho do mesmo detector nos 14 ensaios GPVS")
    caminho = pasta / "gpvs_metricas_por_cenario.png"
    salvar_figura(fig, caminho, "Cada linha representa um ensaio; F1-F7 não participam do ajuste do detector.")
    saidas.append(caminho)

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True, layout="constrained")
    for ax, metrica, titulo in zip(
        axes, ("specificity", "sensitivity"),
        ("Especificidade pré-falha", "Sensibilidade pós-falha"), strict=True,
    ):
        matriz = df.pivot(index="fault", columns="mode", values=metrica).sort_index()
        imagem = ax.imshow(matriz.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(2), ["IPPT (L)", "MPPT (M)"])
        ax.set_yticks(range(7), [f"F{i}" for i in matriz.index])
        ax.set_title(titulo)
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                valor = float(matriz.iloc[i, j])
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color="white" if valor < 0.55 else "black")
    fig.colorbar(imagem, ax=axes, label="Proporção", fraction=0.035, pad=0.04)
    fig.suptitle("Ponto operacional congelado: estabilidade e detecção")
    caminho = pasta / "gpvs_transferencia_estrita.png"
    salvar_figura(fig, caminho, "A especificidade quantifica o deslocamento saudável entre F0 e cada ensaio de falha.")
    saidas.append(caminho)

    itens = macros["canonical_ae"]["all"]
    metricas = [("auc", "AUC"), ("sensitivity", "Sensibilidade"), ("specificity", "Especificidade"), ("balanced_accuracy", "Acurácia balanceada")]
    medias = [itens[chave]["mean"] for chave, _ in metricas]
    baixos = [itens[chave]["mean"] - itens[chave]["ci95_low"] for chave, _ in metricas]
    altos = [itens[chave]["ci95_high"] - itens[chave]["mean"] for chave, _ in metricas]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    x = np.arange(len(metricas))
    ax.bar(x, medias, color=(COR_METODO, COR_ALERTA, COR_NEUTRA, COR_REFERENCIA), width=0.62)
    ax.errorbar(x, medias, yerr=np.vstack([baixos, altos]), fmt="none", ecolor=COR_TEXTO_SEC, capsize=4)
    ax.set_xticks(x, [rotulo for _, rotulo in metricas])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Média macro por ensaio")
    ax.set_title("Detector canônico GPVS: estimativas macro e IC95%")
    for posicao, valor in zip(x, medias, strict=True):
        ax.text(posicao, min(valor + 0.035, 1.045), f"{valor:.3f}", ha="center")
    caminho = pasta / "gpvs_macro_comparacao.png"
    salvar_figura(fig, caminho, "IC95% por bootstrap dos 14 ensaios; janelas não são tratadas como réplicas independentes.")
    saidas.append(caminho)
    return saidas


def _salvar_manifesto_e3(
    arquivos: dict[str, Path], outputs: list[Path]
) -> Path:
    entradas = {f"raw_{nome}": caminho for nome, caminho in arquivos.items()}
    entradas.update({
        "model": PASTA_AE / "modelo_autoencoder.pt",
        "scaler": PASTA_AE / "scaler.pkl",
        "threshold": PASTA_AE / "limiar.json",
        "baseline_normalization": PASTA_AE / "normalizacao_baseline_gpvs.npz",
    })
    dependencias = {
        "gpvs_contract": Path(__file__).with_name("gpvs.py"),
        "gpvs_principal": Path(__file__).with_name("gpvs_principal.py"),
        "anomaly_score": Path(__file__).with_name("escore_anomalia.py"),
        "statistics": Path(__file__).with_name("estatistica.py"),
        "autoencoder": Path(__file__).with_name("autoencoder.py"),
    }
    manifesto = gerar_manifesto(
        "validacao_gpvs_e3",
        Path(__file__),
        {
            "dataset": "GPVS-Faults",
            "fault_experiments": [
                f"F{i}{modo}" for i in range(1, 8) for modo in "LM"
            ],
            "commissioning_baseline_fraction": 0.50,
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_unit": "experiment",
            "sustained_detection_windows": SUSTAINED_WINDOWS,
        },
        entradas,
        outputs,
        code_dependencies=dependencias,
        evidence_level="E3",
    )
    return salvar_manifesto(manifesto)


def executar_validacao_gpvs_principal(
    diretorio: Path = PASTA_GPVS,
    pasta_saida: Path = PASTA_SAIDA,
) -> dict:
    """Aplica o detector canônico congelado aos 14 ensaios reais de falha."""
    diretorio, pasta_saida = Path(diretorio), Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    modelo, scaler, info, estatistica, normalizacao, device = _carregar_detector()
    limiar = float(info.get("score_threshold", info["limiar"]))
    metodo = info.get("score_method", info.get("metodo_escore", "mse"))
    arquivos = arquivos_gpvs(diretorio)
    cenarios, linhas_scores, inventario = [], [], {}

    for falha in range(1, 8):
        for modo in "LM":
            nome = f"F{falha}{modo}"
            _log(f"Validando {nome} com detector congelado...")
            caminho = arquivos[nome]
            features, meta = extrair_features_gpvs(pd.read_csv(caminho), nome)
            matriz, pre_teste, post_teste, meta_baseline = normalizar_comissionamento(
                features, normalizacao
            )
            scores = _pontuar(modelo, scaler, matriz, info, estatistica, device)
            metricas = _metricas_cenario(
                features, scores, limiar,
                pre_indices=pre_teste, post_indices=post_teste,
            )
            cenarios.append({
                "experiment": nome,
                "fault": falha,
                "fault_type": FALHAS[falha],
                "mode": modo,
                "mode_name": MODOS[modo],
                "n_windows": int(len(features)),
                **meta_baseline,
                **metricas,
            })
            inventario[nome] = {
                **meta,
                "path": str(caminho.relative_to(RAIZ_PROJETO)).replace("\\", "/"),
                "sha256": _sha256(caminho),
                "size_bytes": int(caminho.stat().st_size),
            }
            baseline_indices = set(
                np.flatnonzero(features["fase"].eq("pre_falha").to_numpy())
            ) - set(pre_teste.tolist())
            for indice, registro in features.iterrows():
                fase = {
                    "pre_falha": "pre_fault", "pos_falha": "post_fault",
                    "transicao": "transition", "saudavel": "healthy",
                }[registro["fase"]]
                if indice in baseline_indices:
                    fase = "commissioning_baseline"
                linhas_scores.append({
                    "experiment": nome,
                    "fault": falha,
                    "mode": modo,
                    "window_index": int(registro["janela_idx"]),
                    "time_start_s": float(registro["tempo_inicio_s"]),
                    "time_center_s": float(registro["tempo_centro_s"]),
                    "phase": fase,
                    "score": float(scores[indice]),
                    "score_threshold": limiar,
                    "anomaly_index": float(scores[indice] / limiar),
                })

    macros = _resumir_macros(cenarios)
    periodos_observados_us = [
        item["sampling_period_us"] for item in inventario.values()
    ]
    resultado = {
        "schema_version": 2,
        "evidence_level": "E3",
        "evidence_scope": "validação experimental externa em bancada; não é campo",
        "created_at": agora_local().isoformat(),
        "dataset": {
            "name": "GPVS-Faults", "doi": DOI_GPVS,
            "official_zip_sha256": SHA256_ZIP_OFICIAL,
            "n_fault_experiments": 14,
            "observed_sampling_period_us_min": float(
                min(periodos_observados_us)
            ),
            "observed_sampling_period_us_max": float(
                max(periodos_observados_us)
            ),
            "manual_sampling_period_us": 9.9989,
            "sampling_source_for_processing": "Time column in each CSV",
            "files": inventario,
        },
        "detector": {
            "canonical": True,
            "training_experiments": ["F0L", "F0M"],
            "evaluation_experiments": [f"F{i}{m}" for i in range(1, 8) for m in "LM"],
            "adaptation_per_experiment": False,
            "model_retraining_per_experiment": False,
            "threshold_recalibration_per_experiment": False,
            "commissioning_normalization_per_experiment": True,
            "score_method": metodo,
            "score_threshold": limiar,
            "top_k": info.get("top_k", info.get("k_localizado")),
            "model_sha256": _sha256(PASTA_AE / "modelo_autoencoder.pt"),
            "scaler_sha256": _sha256(PASTA_AE / "scaler.pkl"),
            "threshold_sha256": sha256_arquivo_texto_normalizado(
                PASTA_AE / "limiar.json"
            ),
            "baseline_normalization_sha256": _sha256(
                PASTA_AE / "normalizacao_baseline_gpvs.npz"
            ),
        },
        "protocol": {
            "window": "um ciclo de 50 Hz, sem sobreposição",
            "features": FEATURE_COLUMNS,
            "purge_windows_f0": PURGE_WINDOWS,
            "sustained_detection_windows": SUSTAINED_WINDOWS,
            "commissioning_baseline": {
                "fraction_of_pre_fault": normalizacao["baseline_fraction"],
                "minimum_windows": normalizacao["baseline_min_windows"],
                "iqr_floor_fraction_of_global_f0_train": normalizacao[
                    "iqr_floor_fraction"
                ],
                "specificity_evaluation": (
                    "segunda metade pre-falha, não usada na normalização"
                ),
            },
            "confidence_intervals": "bootstrap de ensaios; Wilson por cenário apenas descritivo",
            "bootstrap_resamples": N_BOOTSTRAP,
        },
        "healthy_f0_test": {
            "n_windows": info.get("n_janelas_teste"),
            "false_positive_pct": info.get("fp_test_pct"),
        },
        "macro_summary": macros,
        "scenario_results": cenarios,
        "limitations": [
            "Ensaios de bancada não estimam prevalência industrial.",
            "Janelas do mesmo ensaio são autocorrelacionadas.",
            "O detector indica anomalia e não prova a causa física.",
            "O GPVS não contém tempos de vida para Weibull/RUL físico.",
        ],
    }

    df_cenarios = pd.DataFrame(cenarios)
    df_scores = pd.DataFrame(linhas_scores)
    json_path = pasta_saida / "validacao_gpvs_e3.json"
    scores_path = pasta_saida / "validacao_gpvs_scores.csv"
    relatorio_path = pasta_saida / "relatorio_validacao_gpvs.md"
    json_path.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    scores_path.write_text(
        df_scores.to_csv(index=False, lineterminator="\n"), encoding="utf-8"
    )
    csv_path, md_path = _salvar_tabela(df_cenarios, pasta_saida)
    _gerar_relatorio(resultado, relatorio_path)
    figuras = _plotar(df_cenarios, df_scores, macros, pasta_saida)
    outputs = [
        json_path, csv_path, md_path, scores_path, relatorio_path, *figuras,
    ]
    manifesto_path = _salvar_manifesto_e3(arquivos, outputs)
    return {
        "ok": True,
        "resultado": resultado,
        "outputs": [str(p) for p in outputs],
        "manifest": str(manifesto_path),
    }


def executar_validacao_principal() -> bool:
    """Executa E2/FMECA e E3 real como uma única etapa de validação."""
    from src.ml.validacao import executar_validacao

    if not executar_validacao():
        return False
    return bool(executar_validacao_gpvs_principal().get("ok"))


if __name__ == "__main__":
    executar_validacao_principal()
