"""Orquestra a comparação canônica Denso versus AE-LSTM no GPVS-Faults."""

from __future__ import annotations

import argparse
import json
import logging

from src.core.tempo import agora_local
from src.ml.avaliacao_comparativa import (
    E2_MAGNITUDE_STEPS,
    evaluate_e2,
    evaluate_e3,
)
from src.ml.dados_gpvs import load_or_extract_features, prepare_healthy_data
from src.ml.publicacao_comparacao import RESULTS_DIR, save_results
from src.ml.treino_comparacao import REFERENCE_SEED, STABILITY_SEEDS, train_models


LOGGER = logging.getLogger(__name__)


def run(
    *,
    force_features: bool = False,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
    e2_steps: int = E2_MAGNITUDE_STEPS,
) -> dict:
    """Executa treino saudável, E3 real, E2 sintética e publicação rastreável."""

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
    runs = train_models(prepared, seeds=normalized_seeds)
    LOGGER.info("Executando E3 nos 14 ensaios reais com modelos congelados")
    e3 = evaluate_e3(prepared, runs, faults)
    LOGGER.info("Executando E2 em %s níveis de magnitude", int(e2_steps))
    e2 = evaluate_e2(prepared, runs, n_steps=int(e2_steps))
    LOGGER.info("Publicando dados-fonte, figuras e manifesto v2")
    saved = save_results(
        dataset_manifest,
        prepared,
        runs,
        e3,
        e2,
        seeds=normalized_seeds,
        e2_steps=int(e2_steps),
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
        description="Compara Autoencoder Denso e AE-LSTM no GPVS-Faults"
    )
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--e2-steps", type=int, default=E2_MAGNITUDE_STEPS)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(STABILITY_SEEDS),
        help="Sementes de estabilidade; deve incluir 42",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(
        force_features=args.force_features,
        seeds=tuple(args.seeds),
        e2_steps=args.e2_steps,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "payload"}, indent=2))


if __name__ == "__main__":
    main()
