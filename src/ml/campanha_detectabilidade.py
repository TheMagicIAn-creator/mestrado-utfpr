"""Orquestra a campanha E2: a partir de que magnitude cada modelo detecta.

O QUE ESTA ETAPA FAZ
====================
Carrega os DOIS modelos congelados pela etapa `comparacao` (pesos, scaler e
limiar), injeta severidade crescente nos três itens da FMECA vigente sobre as
janelas saudáveis de holdout e registra, por trajetória, o `a_det` — o menor
`a` em que a detecção se confirma. Publica resumo, curvas POD e as âncoras.

O QUE ELA NÃO FAZ
=================
Não treina, não recalibra o limiar e não escolhe configuração olhando o
resultado da varredura. O detector avaliado aqui é bit a bit o que a E3 avaliou;
qualquer reajuste faria a varredura medir outro modelo. É por isso que o limiar
e o scaler vêm de `carregar_execucao_congelada`, e nada é reajustado.

DUAS DECISÕES DE MODELAGEM, EXPLÍCITAS
======================================
1. Normalização da janela injetada. A injeção perturba uma janela saudável F0;
   ela é normalizada pela baseline F0 do SEU ensaio de origem e depois pelo
   scaler congelado — o mesmo caminho que uma janela saudável percorre no
   treino. Um monitor calibrado em comissionamento saudável enxerga o desvio
   contra essa baseline.
2. Sequência do AE-LSTM. `a` é magnitude, não tempo. No regime sustentado, as
   `SEQUENCE_LENGTH` janelas do contexto causal estão todas na mesma
   severidade; a sequência é o quadro injetado repetido. É o análogo E2 da
   coorte `sustained` da E3, que é a que fundamenta a conclusão temporal.
   Trocar por um contexto saudável-para-injetado mediria a TRANSIÇÃO, que na
   E3 é suplementar.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.core.tempo import agora_local
from src.core.versao_repositorio import CloneDefasado, exigir_clone_atualizado
from src.ml.dados_gpvs import (
    DATASET_DIR,
    FEATURE_COLUMNS,
    PRIMARY_COLUMNS,
    WINDOW_SAMPLES,
    dataset_files,
    feature_vector,
    load_holdout_windows,
    load_or_extract_features,
    normalize_f0_vectors,
    prepare_healthy_data,
)
from src.ml.detectabilidade import (
    CONFIRMACOES_PADRAO,
    Detectabilidade,
    varrer_trajetoria,
)
from src.ml.injecao_e2 import (
    ESPECIFICACOES,
    GRADE_SEVERIDADE,
    METODO_INTERPOLACAO,
    EspecificacaoInjecao,
    distancia_ancora,
    injetor_de,
)
from src.ml.modelos_autoencoder import SEQUENCE_LENGTH, score_dense, score_lstm
from src.ml.treino_comparacao import (
    MODEL_IDS,
    carregar_execucao_congelada,
)

LOGGER = logging.getLogger(__name__)

# A âncora exige escala estritamente positiva; o piso protege features cujo IQR
# saudável degenera a zero sem alterar as demais.
IQR_FLOOR = 1e-9

SEQUENCIA_POLICY = "sustained_severity_repeated_frame"
NORMALIZACAO_POLICY = "f0_baseline_of_source_experiment_then_frozen_scaler"


@dataclass
class ResultadoInjecao:
    """A varredura de um item da FMECA pelos dois modelos, com a âncora."""

    especificacao: EspecificacaoInjecao
    por_modelo: dict[str, Detectabilidade]
    ancora: dict
    registros_falha: list[dict] = field(default_factory=list)


# ── transformação janela → escore, espelhando a E3 ─────────────────────────

def _escalar_janelas(
    janelas: list[pd.DataFrame],
    experimento: str,
    scaler,
    baseline: dict,
) -> np.ndarray:
    """feature_vector → baseline F0 do ensaio de origem → scaler congelado."""
    fvs = np.asarray([feature_vector(janela) for janela in janelas], dtype=np.float32)
    rotulos = np.asarray([experimento] * len(janelas))
    normalizadas = normalize_f0_vectors(fvs, rotulos, baseline)
    return scaler.transform(normalizadas).astype(np.float32)


def _sequencias_estacionarias(escaladas: np.ndarray) -> np.ndarray:
    """Uma sequência por ponto de severidade, com o quadro injetado sustentado.

    Repetir o quadro `SEQUENCE_LENGTH` vezes é o regime sustentado: todo o
    contexto causal está na mesma severidade. `score_lstm` lê o último passo,
    que é exatamente esse quadro.
    """
    matriz = np.asarray(escaladas, dtype=np.float32)
    return np.repeat(matriz[:, None, :], int(SEQUENCE_LENGTH), axis=1)


def _pontuador(execucao: dict, experimento: str, baseline: dict):
    """Scorer congelado para uma trajetória de um ensaio saudável de origem."""
    modelo = execucao["modelo"]
    scaler = execucao["scaler"]
    top_k = int(execucao["score_top_k"])
    model_id = execucao["model_id"]

    def scorer(janelas: list[pd.DataFrame]) -> np.ndarray:
        escaladas = _escalar_janelas(janelas, experimento, scaler, baseline)
        if model_id == "ae_denso":
            return score_dense(modelo, escaladas, top_k=top_k)
        return score_lstm(modelo, _sequencias_estacionarias(escaladas), top_k=top_k)

    return scorer


# ── janelas brutas de falha, para a âncora e a interpolação ────────────────

def _carregar_janelas_falha(
    fault_features: pd.DataFrame,
    experimentos: tuple[str, ...],
    *,
    fase: str = "post_fault",
    directory=DATASET_DIR,
) -> tuple[list[pd.DataFrame], list[dict]]:
    """Janelas brutas pós-falha dos ensaios reais de um item da FMECA."""
    files = dataset_files(directory)
    cache: dict[str, pd.DataFrame] = {}
    janelas: list[pd.DataFrame] = []
    registros: list[dict] = []
    for experimento in experimentos:
        linhas = (
            fault_features[
                fault_features["experiment"].eq(experimento)
                & fault_features["phase"].eq(fase)
            ]
            .sort_values("window_index")
        )
        if linhas.empty:
            raise ValueError(
                f"{experimento} não tem janelas de fase {fase} para a âncora"
            )
        if experimento not in cache:
            cache[experimento] = pd.read_csv(
                files[experimento],
                usecols=list(PRIMARY_COLUMNS),
                dtype={coluna: np.float32 for coluna in PRIMARY_COLUMNS},
            )
        bruto = cache[experimento]
        for _, linha in linhas.iterrows():
            inicio, fim = int(linha["sample_start"]), int(linha["sample_end"])
            janela = bruto.iloc[inicio:fim].copy().reset_index(drop=True)
            if len(janela) != WINDOW_SAMPLES:
                raise ValueError(
                    f"{experimento}: janela de falha incompleta em {inicio}:{fim}"
                )
            janelas.append(janela)
            registros.append(
                {"experiment": experimento, "window_index": int(linha["window_index"])}
            )
    return janelas, registros


def _escala_saudavel(prepared) -> np.ndarray:
    """IQR por feature do bloco saudável de TREINO, no espaço bruto das features."""
    indices = np.asarray(prepared.split["train"], dtype=int)
    bloco = (
        prepared.features.iloc[indices][list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    )
    q25, q75 = np.percentile(bloco, (25, 75), axis=0)
    return np.maximum(q75 - q25, IQR_FLOOR)


# ── a varredura de um item da FMECA ────────────────────────────────────────

def _varrer_injecao(
    especificacao: EspecificacaoInjecao,
    execucoes: dict[str, dict],
    holdout: list[pd.DataFrame],
    holdout_experimentos: list[str],
    janelas_falha: list[pd.DataFrame],
    baseline: dict,
    *,
    grade=GRADE_SEVERIDADE,
    confirmacoes: int = CONFIRMACOES_PADRAO,
) -> tuple[dict[str, Detectabilidade], list]:
    """Varre a severidade de um item pelos dois modelos congelados.

    Constrói o resultado por trajetória em vez de chamar
    `detectabilidade_do_modelo` porque cada trajetória tem seu próprio scorer
    (a baseline F0 é por ensaio de origem) e, na interpolação, seu próprio
    injetor (o extremo `a=1` é um ensaio real distinto). O núcleo autoritativo
    continua sendo `varrer_trajetoria`.
    """
    n = len(holdout)
    if not n:
        raise ValueError("A campanha exige ao menos uma janela de holdout")
    if especificacao.metodo == METODO_INTERPOLACAO:
        if not janelas_falha:
            raise ValueError(
                f"{especificacao.id} é interpolação e exige janelas reais de "
                "falha como extremo superior"
            )
        injetores = [
            injetor_de(
                especificacao, janela_falha=janelas_falha[indice % len(janelas_falha)]
            )
            for indice in range(n)
        ]
    else:
        injetor = injetor_de(especificacao)
        injetores = [injetor] * n

    por_modelo: dict[str, Detectabilidade] = {}
    for model_id in MODEL_IDS:
        execucao = execucoes[model_id]
        limiar = float(execucao["limiar"])
        achados = [
            varrer_trajetoria(
                _pontuador(execucao, holdout_experimentos[indice], baseline),
                holdout[indice],
                injetores[indice],
                limiar,
                grade,
                confirmacoes=confirmacoes,
            )
            for indice in range(n)
        ]
        eventos = np.asarray([achado is not None for achado in achados], dtype=bool)
        a_dets = np.asarray(
            [achado if achado is not None else 1.0 for achado in achados],
            dtype=float,
        )
        por_modelo[model_id] = Detectabilidade(
            modelo=model_id,
            injecao=especificacao.id,
            a_dets=a_dets,
            eventos=eventos,
            n_trajetorias=n,
            grade=tuple(float(a) for a in grade),
        )
    return por_modelo, injetores


def _ancora_da_injecao(
    injetores,
    holdout: list[pd.DataFrame],
    janelas_falha: list[pd.DataFrame],
    escala_saudavel: np.ndarray,
) -> dict:
    """Distância entre a assinatura em `a=1` e o ensaio real, em IQR saudável.

    A âncora independe do modelo: `feature_vector` da janela injetada e da real
    não passa por nenhum autoencoder.
    """
    injetadas_a1 = [injetores[indice](holdout[indice], 1.0) for indice in range(len(holdout))]
    return distancia_ancora(injetadas_a1, janelas_falha, escala_saudavel)


# ── orquestração ───────────────────────────────────────────────────────────

def run(
    *,
    directory=DATASET_DIR,
    grade=GRADE_SEVERIDADE,
    confirmacoes: int = CONFIRMACOES_PADRAO,
    publicar: bool = True,
    force_features: bool = False,
) -> dict:
    """Executa a campanha E2 completa e, por padrão, publica os artefatos.

    `force_features` só reextrai as features do GPVS; ele não retreina nem
    recalibra nada — os pesos, o scaler e o limiar continuam vindo congelados da
    etapa `comparacao`. O argumento existe porque `pipeline.executar_etapa`
    chama todo runner que exige GPVS com essa assinatura.
    """
    LOGGER.info("Carregando features e ensaios GPVS-Faults")
    healthy, faults, dataset_manifest = load_or_extract_features(
        force=force_features, directory=directory
    )
    prepared = prepare_healthy_data(healthy)
    LOGGER.info("Carregando janelas saudáveis de holdout para injeção")
    holdout, holdout_meta = load_holdout_windows(prepared, directory=directory)
    holdout_experimentos = [str(janela.attrs["experiment"]) for janela in holdout]
    baseline = prepared.baseline_normalization
    escala_saudavel = _escala_saudavel(prepared)

    LOGGER.info("Carregando os dois modelos congelados da etapa comparacao")
    execucoes = {model_id: carregar_execucao_congelada(model_id) for model_id in MODEL_IDS}

    resultados: list[ResultadoInjecao] = []
    for especificacao in ESPECIFICACOES:
        LOGGER.info("Varrendo a severidade do item %s", especificacao.id)
        janelas_falha, registros_falha = _carregar_janelas_falha(
            faults, especificacao.ensaios_reais, directory=directory
        )
        por_modelo, injetores = _varrer_injecao(
            especificacao,
            execucoes,
            holdout,
            holdout_experimentos,
            janelas_falha,
            baseline,
            grade=grade,
            confirmacoes=confirmacoes,
        )
        ancora = _ancora_da_injecao(
            injetores, holdout, janelas_falha, escala_saudavel
        )
        resultados.append(
            ResultadoInjecao(
                especificacao=especificacao,
                por_modelo=por_modelo,
                ancora=ancora,
                registros_falha=registros_falha,
            )
        )

    parametros = {
        "grade_severidade": [float(a) for a in grade],
        "confirmacoes": int(confirmacoes),
        "n_trajetorias": len(holdout),
        "holdout_experiments": sorted(set(holdout_experimentos)),
        "sequence_policy": SEQUENCIA_POLICY,
        "normalization_policy": NORMALIZACAO_POLICY,
        "threshold_source": "frozen_from_comparacao",
        "a_axis": "fraction_of_nominal_signature_not_time",
    }

    if not publicar:
        return {
            "ok": True,
            "created_at": agora_local().isoformat(),
            "resultados": resultados,
            "parametros": parametros,
            "holdout_meta": holdout_meta,
        }

    LOGGER.info("Publicando resumo, curvas POD, âncoras e manifesto E2")
    from src.ml.publicacao_detectabilidade import RESULTS_DIR, salvar_resultados

    saved = salvar_resultados(
        resultados,
        dataset_manifest=dataset_manifest,
        holdout_meta=holdout_meta,
        parametros=parametros,
    )
    return {
        "ok": True,
        "created_at": agora_local().isoformat(),
        "results_dir": str(RESULTS_DIR),
        "manifest": str(saved["manifest"]),
        "output_count": len(saved["outputs"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Campanha E2: magnitude de detecção dos dois autoencoders"
    )
    parser.add_argument(
        "--confirmacoes",
        type=int,
        default=CONFIRMACOES_PADRAO,
        help=(
            "Pontos consecutivos acima do limiar para a detecção contar "
            f"(padrão: {CONFIRMACOES_PADRAO})"
        ),
    )
    parser.add_argument(
        "--ignorar-versao",
        action="store_true",
        help=(
            "Executa mesmo com o clone atrás do remoto. Use apenas para "
            "reproduzir deliberadamente um commit anterior."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.ignorar_versao:
        try:
            exigir_clone_atualizado(progresso=LOGGER.warning)
        except CloneDefasado as erro:
            LOGGER.error("%s", erro)
            return 1
    resultado = run(confirmacoes=args.confirmacoes)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "IQR_FLOOR",
    "NORMALIZACAO_POLICY",
    "SEQUENCIA_POLICY",
    "ResultadoInjecao",
    "run",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
