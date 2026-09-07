"""Publicação rastreável da campanha E2 de detectabilidade por magnitude.

Escreve resumo, curvas POD e âncoras como dados-fonte auditáveis, mais o
contrato JSON, o relatório e o manifesto de proveniência da etapa. O empírico
vem sempre primeiro; o percentil paramétrico só aparece quando o ajuste Weibull
foi adotado, e o método de injeção viaja em cada linha.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.core.config import RAIZ_PROJETO
from src.ml.detectabilidade import (
    MINIMO_DETECTADAS,
    MINIMO_R2_PAPEL,
    MINIMO_VALORES_DISTINTOS,
    curva_pod,
    resumo,
)
from src.ml.injecao_e2 import contrato_da_especificacao
from src.ml.proveniencia import gerar_manifesto, salvar_manifesto
from src.ml.treino_comparacao import MODEL_IDS, MODEL_NAMES, MODEL_ROOT

ROOT = Path(RAIZ_PROJETO)
RESULTS_DIR = ROOT / "resultados" / "detectabilidade"
STAGE = "detectabilidade"
CODE_PATH = ROOT / "src" / "ml" / "campanha_detectabilidade.py"

# Colunas do contrato de injeção que descem ao CSV; a fundamentação longa fica
# só no JSON e no relatório.
_CAMPOS_CONTRATO_CSV = (
    "injection_id",
    "injection_name",
    "fmeca_scope",
    "injection_method",
    "physical_simulation",
    "a1_meaning",
)


def _limpar(valor):
    """Converte não-finitos em None em profundidade, para JSON estrito."""
    if isinstance(valor, dict):
        return {chave: _limpar(sub) for chave, sub in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_limpar(item) for item in valor]
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(_limpar(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def _contrato_csv(contrato: dict) -> dict:
    linha = {campo: contrato[campo] for campo in _CAMPOS_CONTRATO_CSV}
    linha["reference_experiments"] = ";".join(contrato["reference_experiments"])
    return linha


def _linhas_resumo(resultados) -> pd.DataFrame:
    linhas = []
    for resultado in resultados:
        contrato = contrato_da_especificacao(resultado.especificacao)
        base = _contrato_csv(contrato)
        for model_id in MODEL_IDS:
            det = resultado.por_modelo[model_id]
            linhas.append(
                {
                    **base,
                    "model": model_id,
                    "model_name": MODEL_NAMES[model_id],
                    **resumo(det),
                }
            )
    return pd.DataFrame(linhas)


def _linhas_pod(resultados) -> pd.DataFrame:
    linhas = []
    for resultado in resultados:
        for model_id in MODEL_IDS:
            det = resultado.por_modelo[model_id]
            curva = curva_pod(det)
            for a, pod in zip(curva["a"], curva["pod"], strict=True):
                linhas.append(
                    {
                        "injection_id": resultado.especificacao.id,
                        "model": model_id,
                        "a": float(a),
                        "pod": float(pod),
                    }
                )
    return pd.DataFrame(linhas)


def _linhas_ancora(resultados) -> pd.DataFrame:
    linhas = []
    for resultado in resultados:
        contrato = contrato_da_especificacao(resultado.especificacao)
        linhas.append(
            {
                "injection_id": resultado.especificacao.id,
                "injection_method": contrato["injection_method"],
                "fmeca_scope": contrato["fmeca_scope"],
                "reference_experiments": ";".join(contrato["reference_experiments"]),
                "physical_simulation": contrato["physical_simulation"],
                **resultado.ancora,
            }
        )
    return pd.DataFrame(linhas)


def _metodologia() -> dict:
    return {
        "a_axis": "fraction_of_nominal_signature_not_time",
        "a_det_is_not": ["time", "cycle", "consumed_life", "failure_rate", "RUL"],
        "empirical_before_parametric": True,
        "right_censoring": "trajectory_that_never_crosses_by_a1_is_censored_not_a_det_1",
        "weibull_adoption_gate": {
            "min_detected": MINIMO_DETECTADAS,
            "min_paper_r2": MINIMO_R2_PAPEL,
            "min_distinct_values": MINIMO_VALORES_DISTINTOS,
        },
        "weibull_meaning": "dispersion_of_detection_magnitude_not_physical_reliability",
    }


def _relatorio(resultados, parametros: dict) -> str:
    linhas = [
        "# Detectabilidade por magnitude (E2)",
        "",
        "## Escopo",
        "",
        "A varredura injeta severidade crescente nos três itens da FMECA vigente "
        "sobre as janelas saudáveis de holdout F0 e registra `a_det`, o menor `a` "
        "em que cada modelo congelado confirma a detecção. `a` é fração da "
        "assinatura nominal, não tempo: `a_det`, a POD e a Weibull descrevem a "
        "dispersão da MAGNITUDE de detecção, nunca confiabilidade, taxa de falha "
        "ou RUL. O detector é o mesmo congelado na etapa `comparacao` e avaliado "
        "na E3; a campanha não treina nem recalibra.",
        "",
        "## Método por item",
        "",
        "| Item | Escopo FMECA | Ensaios reais | Método | Simulação física |",
        "|---|---|---|---|---|",
    ]
    for resultado in resultados:
        contrato = contrato_da_especificacao(resultado.especificacao)
        linhas.append(
            "| {nome} | {escopo} | {ensaios} | {metodo} | {fisica} |".format(
                nome=contrato["injection_name"],
                escopo=contrato["fmeca_scope"],
                ensaios=", ".join(contrato["reference_experiments"]),
                metodo=contrato["injection_method"],
                fisica="sim" if contrato["physical_simulation"] else "não",
            )
        )
    linhas += [
        "",
        "## a_det empírico e âncora",
        "",
        "| Item | Modelo | Fração detectada | a10 | a50 | Weibull adotada | Âncora (IQR) |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for resultado in resultados:
        ancora = resultado.ancora["distancia_euclidiana_iqr"]
        for model_id in MODEL_IDS:
            det = resultado.por_modelo[model_id]
            linha_resumo = resumo(det)
            linhas.append(
                "| {item} | {modelo} | {frac:.2f} | {a10} | {a50} | {wb} | {anc:.2f} |".format(
                    item=resultado.especificacao.id,
                    modelo=model_id,
                    frac=linha_resumo["fracao_detectada"],
                    a10=_fmt(linha_resumo["a10_empirico"]),
                    a50=_fmt(linha_resumo["a50_empirico"]),
                    wb="sim" if linha_resumo["weibull_2p_adotada"] else "não",
                    anc=ancora,
                )
            )
    linhas += [
        "",
        "## Leitura da âncora",
        "",
        "Em `a=1` a distância entre a janela injetada e o ensaio real sai em "
        "unidades do IQR saudável. Nas assinaturas elétricas (IGBT, sensor), "
        "distância grande diz que a injeção é caricatura da falha medida — é "
        "informação, não defeito. Na interpolação de controle, `a=1` É um estado "
        "medido, então a distância próxima de zero é esperada e confirma que a "
        "injeção não simula física.",
        "",
        f"Trajetórias por item: {parametros['n_trajetorias']}. "
        f"Confirmação: {parametros['confirmacoes']} pontos consecutivos.",
        "",
    ]
    return "\n".join(linhas)


def _fmt(valor) -> str:
    return "—" if valor is None else f"{float(valor):.3f}"


def _payload(resultados, *, holdout_meta: dict, parametros: dict) -> dict:
    injecoes = []
    for resultado in resultados:
        contrato = contrato_da_especificacao(resultado.especificacao)
        injecoes.append(
            {
                **contrato,
                "anchor": resultado.ancora,
                "reference_windows": resultado.registros_falha,
                "models": {
                    model_id: resumo(resultado.por_modelo[model_id])
                    for model_id in MODEL_IDS
                },
            }
        )
    holdout_resumo = {
        chave: holdout_meta[chave]
        for chave in ("dataset", "doi", "protocol", "purge_windows", "n_windows")
        if chave in holdout_meta
    }
    return {
        "stage": STAGE,
        "evidence_level": "E2",
        "schema_version": 1,
        "parameters": parametros,
        "methodology": _metodologia(),
        "holdout": holdout_resumo,
        "injections": injecoes,
    }


def salvar_resultados(
    resultados,
    *,
    dataset_manifest: dict,
    holdout_meta: dict,
    parametros: dict,
    results_dir: Path = RESULTS_DIR,
) -> dict:
    """Escreve dados-fonte, contrato, relatório e manifesto da etapa E2."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    resumo_path = results_dir / "detectabilidade_resumo.csv"
    pod_path = results_dir / "pod_curvas.csv"
    ancora_path = results_dir / "ancoras.csv"
    _linhas_resumo(resultados).to_csv(resumo_path, index=False, lineterminator="\n")
    _linhas_pod(resultados).to_csv(pod_path, index=False, lineterminator="\n")
    _linhas_ancora(resultados).to_csv(ancora_path, index=False, lineterminator="\n")
    outputs.extend([resumo_path, pod_path, ancora_path])

    payload = _payload(resultados, holdout_meta=holdout_meta, parametros=parametros)
    outputs.append(_write_json(results_dir / "detectabilidade.json", payload))

    report_path = results_dir / "relatorio.md"
    report_path.write_text(_relatorio(resultados, parametros), encoding="utf-8", newline="\n")
    outputs.append(report_path)

    manifest = gerar_manifesto(
        stage=STAGE,
        code_path=CODE_PATH,
        parameters={
            **parametros,
            "weibull_adoption_gate": _metodologia()["weibull_adoption_gate"],
            "injections": [resultado.especificacao.id for resultado in resultados],
            "models": list(MODEL_IDS),
            "dataset_windows": dataset_manifest.get("windows")
            if isinstance(dataset_manifest, dict)
            else None,
        },
        input_artifacts={
            f"{model_id}_contract": MODEL_ROOT / model_id / "contrato.json"
            for model_id in MODEL_IDS
        },
        outputs=outputs,
        code_dependencies={
            "detectabilidade": ROOT / "src" / "ml" / "detectabilidade.py",
            "injection": ROOT / "src" / "ml" / "injecao_e2.py",
            "models": ROOT / "src" / "ml" / "modelos_autoencoder.py",
            "dataset": ROOT / "src" / "ml" / "dados_gpvs.py",
            "training": ROOT / "src" / "ml" / "treino_comparacao.py",
            "orchestration": CODE_PATH,
        },
        evidence_level="E2",
    )
    manifest_path = salvar_manifesto(manifest)
    return {"outputs": outputs, "manifest": manifest_path, "payload": payload}


__all__ = ["CODE_PATH", "RESULTS_DIR", "STAGE", "salvar_resultados"]
