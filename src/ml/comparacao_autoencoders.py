"""Orquestra a comparação canônica entre Autoencoder Denso e AE-LSTM."""

from __future__ import annotations

import argparse
import json
import logging

from src.core.tempo import agora_local
from src.ml.avaliacao_comparativa import evaluate_e3
from src.ml.dados_gpvs import load_or_extract_features, prepare_healthy_data
from src.ml.modelos_autoencoder import SCORE_TOP_K
from src.ml.publicacao_comparacao import RESULTS_DIR, save_results
from src.ml.sensibilidade_escore import evaluate_score_threshold_sensitivity
from src.ml.treino_comparacao import (
    REFERENCE_SEED,
    STABILITY_SEEDS,
    THRESHOLD_PERCENTILE,
    train_models,
)


LOGGER = logging.getLogger(__name__)


def run(
    *,
    force_features: bool = False,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
    threshold_percentile: float = THRESHOLD_PERCENTILE,
    score_top_k: int = SCORE_TOP_K,
) -> dict:
    """Executa treino saudável, validação E3 e publicação rastreável."""

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if REFERENCE_SEED not in normalized_seeds:
        raise ValueError(f"A execução deve incluir a semente {REFERENCE_SEED}")
    LOGGER.info("Carregando e validando os 16 ensaios GPVS-Faults")
    healthy, faults, dataset_manifest = load_or_extract_features(
        force=force_features
    )
    LOGGER.info("Preparando os quatro blocos temporais saudáveis")
    prepared = prepare_healthy_data(healthy)
    LOGGER.info("Treinando os dois modelos nas sementes %s", normalized_seeds)
    runs = train_models(
        prepared,
        seeds=normalized_seeds,
        threshold_percentile=threshold_percentile,
        score_top_k=score_top_k,
    )
    LOGGER.info("Executando E3 nos 14 ensaios reais com modelos congelados")
    e3 = evaluate_e3(prepared, runs, faults)
    LOGGER.info("Calculando sensibilidade pré-fixada de top-k e limiar saudável")
    e3["score_threshold_sensitivity"] = evaluate_score_threshold_sensitivity(
        prepared,
        runs,
        faults,
    )
    LOGGER.info("Publicando dados-fonte, figuras e manifesto v2")
    saved = save_results(
        dataset_manifest,
        prepared,
        runs,
        e3,
        seeds=normalized_seeds,
    )
    return {
        "ok": True,
        "created_at": agora_local().isoformat(),
        "results_dir": str(RESULTS_DIR),
        "manifest": str(saved["manifest"]),
        "output_count": len(saved["outputs"]),
        "payload": saved["payload"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara Autoencoder Denso e AE-LSTM na base experimental"
    )
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(STABILITY_SEEDS),
        help="Sementes de estabilidade; deve incluir 42",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=THRESHOLD_PERCENTILE,
        help="Percentil saudável solicitado para o limiar (padrão: 99.9)",
    )
    parser.add_argument(
        "--score-top-k",
        type=int,
        default=SCORE_TOP_K,
        help="Quantidade de maiores erros por feature usados no escore (padrão: 5)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(
        force_features=args.force_features,
        seeds=tuple(args.seeds),
        threshold_percentile=args.threshold_percentile,
        score_top_k=args.score_top_k,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "payload"}, indent=2))


if __name__ == "__main__":
    main()
