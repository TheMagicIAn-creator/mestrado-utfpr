"""
pipeline.py - Al IAdo PV
Registro unico das etapas do pipeline de Machine Learning.

Este modulo centraliza ordem, nomes, artefatos, dependencias e execucao.
Interface, orquestrador e ferramentas devem consumir daqui em vez de
recriar listas paralelas de etapas.
"""

from __future__ import annotations

import ast
import gc
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

from src.core.config import RAIZ_PROJETO


@dataclass(frozen=True)
class PipelineStage:
    key: str
    label: str
    module: str
    function: str
    artifacts: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    input_artifacts: tuple[str, ...] = ()
    parameter_names: tuple[str, ...] = ()
    code_dependencies: tuple[str, ...] = ()
    evidence_level: str | None = None

    def paths(self) -> list[Path]:
        return [RAIZ_PROJETO / rel for rel in self.artifacts]

    def is_complete(self) -> bool:
        return all(path.exists() for path in self.paths())

    def load_runner(self) -> Callable[[], bool]:
        module = import_module(self.module)
        return getattr(module, self.function)

    def parameters(self) -> dict:
        if not self.parameter_names:
            return {}
        valores = _parametros_do_fonte(self.module, self.parameter_names)
        pendentes = [nome for nome in self.parameter_names if nome not in valores]
        if pendentes:
            module = import_module(self.module)
            for nome in pendentes:
                if hasattr(module, nome):
                    valores[nome] = getattr(module, nome)
        return {
            nome.lower(): _valor_manifesto(valores[nome])
            for nome in self.parameter_names
            if nome in valores
        }


def _parametros_do_fonte(module: str, nomes: tuple[str, ...]) -> dict:
    """Le constantes simples do arquivo sem importar modulos pesados."""
    caminho = RAIZ_PROJETO / Path(*module.split(".")).with_suffix(".py")
    if not caminho.exists():
        return {}
    procurados = set(nomes)
    valores = {}
    try:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    for node in arvore.body:
        if isinstance(node, ast.Assign):
            alvos = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for alvo in alvos:
                if alvo in procurados:
                    try:
                        valores[alvo] = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError):
                        pass
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            alvo = node.target.id
            if alvo in procurados and node.value is not None:
                try:
                    valores[alvo] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
    return valores


def _valor_manifesto(valor):
    """Normaliza constantes de etapa para JSON estável no manifesto."""
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    if isinstance(valor, (list, tuple)):
        return [_valor_manifesto(v) for v in valor]
    if isinstance(valor, dict):
        return {str(k): _valor_manifesto(v) for k, v in valor.items()}
    return str(valor)


GPVS_F0_INPUTS = tuple(
    f"dados/brutos/gpvs/csv/CSV_Files/F0{modo}.csv" for modo in "LM"
)
GPVS_FAULT_INPUTS = tuple(
    f"dados/brutos/gpvs/csv/CSV_Files/F{falha}{modo}.csv"
    for falha in range(1, 8)
    for modo in "LM"
)
GPVS_ALL_INPUTS = GPVS_F0_INPUTS + GPVS_FAULT_INPUTS


STAGES: dict[str, PipelineStage] = {
    "features_gpvs": PipelineStage(
        key="features_gpvs",
        label="Features GPVS-Faults F0",
        module="src.ml.gpvs_principal",
        function="executar_features_gpvs",
        parameter_names=("FS", "F0", "JANELA", "SOBREPOSICAO", "HARMONICOS"),
        code_dependencies=("src.ml.gpvs", "src.ml.estilo_graficos"),
        input_artifacts=GPVS_F0_INPUTS,
        artifacts=(
            "dados/processados/features_gpvs.parquet",
            "dados/processados/features_gpvs_stats.csv",
            "resultados/qualidade/features_gpvs_qualidade.json",
            "resultados/qualidade/features_gpvs_qualidade.png",
        ),
    ),
    "autoencoder": PipelineStage(
        key="autoencoder",
        label="Autoencoder",
        module="src.ml.autoencoder",
        function="executar_autoencoder",
        parameter_names=(
            "LATENTE_DIM", "EPOCHS", "BATCH_SIZE", "LR", "DROPOUT",
            "PACIENCIA", "SIGMA", "THRESHOLD_METHOD", "SEED",
            "TRAIN_RATIO", "VALIDATION_RATIO", "CALIBRATION_RATIO", "TEST_RATIO",
        ),
        code_dependencies=(
            "src.ml.escore_anomalia",
            "src.ml.gpvs",
            "src.ml.gpvs_principal",
            "src.ml.graficos_autoencoder",
            "src.ml.estilo_graficos",
        ),
        artifacts=(
            "resultados/autoencoder/modelo_autoencoder.pt",
            "resultados/autoencoder/scaler.pkl",
            "resultados/autoencoder/scaler.pkl.sha256",
            "resultados/autoencoder/normalizacao_baseline_gpvs.npz",
            "resultados/autoencoder/estatistica_residuo.npz",
            "resultados/autoencoder/limiar.json",
            "resultados/autoencoder/diagnostico_autoencoder.npz",
            "resultados/autoencoder/calibracao_autoencoder.csv",
            "resultados/autoencoder/calibracao_autoencoder.md",
            "resultados/autoencoder/curva_treino.png",
            "resultados/autoencoder/distribuicao_erro.png",
            "resultados/autoencoder/erro_temporal.png",
        ),
        depends_on=("features_gpvs",),
        input_artifacts=("dados/processados/features_gpvs.parquet",),
        evidence_level="E2",
    ),
    "injecao_falhas": PipelineStage(
        key="injecao_falhas",
        label="Injecao de Falhas",
        module="src.ml.injecao_falhas",
        function="executar_injecao_falhas",
        parameter_names=("A_INJ", "ALVO_SMD", "N_JANELAS_SMD"),
        code_dependencies=(
            "src.ml.gpvs",
            "src.ml.gpvs_principal",
            "src.ml.autoencoder",
            "src.ml.escore_anomalia",
            "src.ml.estatistica",
            "src.ml.estilo_graficos",
            "src.ml.diagnostico_escore",
        ),
        artifacts=(
            "resultados/autoencoder/injecao_falhas_resultados.png",
            "resultados/autoencoder/injecao_falhas_comparacao.png",
            "resultados/autoencoder/injecao_falhas_report.json",
            "resultados/autoencoder/injecao_smd_tabela.csv",
            "resultados/autoencoder/diagnostico_escore.png",
            "resultados/autoencoder/diagnostico_escore.json",
        ),
        depends_on=("autoencoder",),
        input_artifacts=GPVS_F0_INPUTS + (
            "dados/processados/features_gpvs.parquet",
            "resultados/autoencoder/modelo_autoencoder.pt",
            "resultados/autoencoder/scaler.pkl",
            "resultados/autoencoder/scaler.pkl.sha256",
            "resultados/autoencoder/normalizacao_baseline_gpvs.npz",
            "resultados/autoencoder/estatistica_residuo.npz",
            "resultados/autoencoder/limiar.json",
        ),
        evidence_level="E2",
    ),
    "validacao": PipelineStage(
        key="validacao",
        label="Validacao GPVS E2 + E3",
        module="src.ml.validacao_gpvs_principal",
        function="executar_validacao_principal",
        parameter_names=(
            "SEVS_VALIDACAO", "N_JANELAS_SAUDAVEL", "N_JANELAS_FALHA",
            "PREVALENCIA_RARA",
        ),
        code_dependencies=(
            "src.ml.gpvs",
            "src.ml.gpvs_principal",
            "src.ml.autoencoder",
            "src.ml.escore_anomalia",
            "src.ml.estatistica",
            "src.ml.injecao_falhas",
            "src.ml.retroalimentacao_fmeca",
            "src.ml.validacao",
            "src.ml.estilo_graficos",
        ),
        artifacts=(
            "resultados/autoencoder/validacao_roc.png",
            "resultados/autoencoder/validacao_pr.png",
            "resultados/autoencoder/validacao_matriz.png",
            "resultados/autoencoder/validacao_matrizes_severidades.png",
            "resultados/autoencoder/validacao_metricas.png",
            "resultados/autoencoder/validacao_tabela.csv",
            "resultados/autoencoder/validacao_tabela.md",
            "resultados/autoencoder/validacao_report.json",
            "resultados/autoencoder/retroalimentacao_fmeca.json",
            "resultados/autoencoder/retroalimentacao_fmeca.md",
            "resultados/gpvs/validacao_gpvs_e3.json",
            "resultados/gpvs/validacao_gpvs_cenarios.csv",
            "resultados/gpvs/validacao_gpvs_cenarios.md",
            "resultados/gpvs/validacao_gpvs_scores.csv",
            "resultados/gpvs/relatorio_validacao_gpvs.md",
            "resultados/gpvs/gpvs_series_temporais.png",
            "resultados/gpvs/gpvs_metricas_por_cenario.png",
            "resultados/gpvs/gpvs_transferencia_estrita.png",
            "resultados/gpvs/gpvs_macro_comparacao.png",
        ),
        depends_on=("injecao_falhas",),
        input_artifacts=GPVS_ALL_INPUTS + (
            "dados/processados/features_gpvs.parquet",
            "resultados/autoencoder/modelo_autoencoder.pt",
            "resultados/autoencoder/scaler.pkl",
            "resultados/autoencoder/scaler.pkl.sha256",
            "resultados/autoencoder/normalizacao_baseline_gpvs.npz",
            "resultados/autoencoder/estatistica_residuo.npz",
            "resultados/autoencoder/limiar.json",
        ),
        evidence_level="E2+E3",
    ),
    "rul_weibull": PipelineStage(
        key="rul_weibull",
        label="Detectabilidade E2 / Weibull",
        module="src.ml.rul_weibull",
        function="executar_rul_weibull",
        parameter_names=(
            "N_TRAJ", "N_STEPS", "N_STEPS_SENSIBILIDADE",
            "BATCH_INFERENCIA", "N_BOOTSTRAP", "N_BOOTSTRAP_ADERENCIA",
            "N_BOOTSTRAP_MODO",
            "MIN_EVENTOS_WEIBULL", "MAX_CENSURA_RUL_PCT",
            "MIN_R2_PAPEL_WEIBULL", "MIN_NIVEIS_ADERENCIA",
            "ALFA_ADERENCIA", "MAX_VARIACAO_RELATIVA_GRADE",
            "PERSISTENCIA_MAGNITUDE",
            "PERSISTENCIA_CRUZAMENTO", "AJUSTE_WEIBULL_METODO",
            "A_DET_UNIDADE", "TTF_UNIDADE",
            "TEMPO_FISICO_CALIBRADO",
        ),
        code_dependencies=(
            "src.ml.gpvs",
            "src.ml.gpvs_principal",
            "src.ml.autoencoder",
            "src.ml.escore_anomalia",
            "src.ml.estatistica",
            "src.ml.confiabilidade",
            "src.ml.graficos_rul",
            "src.ml.injecao_falhas",
            "src.ml.rul_weibull_execucao",
            "src.ml.relatorio_weibull",
            "src.ml.estilo_graficos",
            "src.ml.pod_curva",
            "scripts.relatorio_confiabilidade",
        ),
        artifacts=(
            "resultados/autoencoder/weibull_ttf.png",
            "resultados/autoencoder/weibull_confiabilidade.png",
            "resultados/autoencoder/weibull_intensidade_deteccao.png",
            "resultados/autoencoder/weibull_funcoes_distribuicao.png",
            "resultados/autoencoder/weibull_distribuicao.png",
            "resultados/autoencoder/weibull_rul.png",
            "resultados/autoencoder/weibull_sensibilidade_grade.png",
            "resultados/autoencoder/weibull_modos_operacao.png",
            "resultados/autoencoder/weibull_sensibilidade_grade.csv",
            "resultados/autoencoder/weibull_trajetorias_grade.csv",
            "resultados/autoencoder/weibull_results.json",
            "resultados/autoencoder/weibull_tabela.csv",
            "resultados/autoencoder/relatorio_confiabilidade.md",
            "resultados/autoencoder/relatorio_confiabilidade.json",
        ),
        depends_on=("validacao",),
        input_artifacts=GPVS_F0_INPUTS + (
            "dados/processados/features_gpvs.parquet",
            "resultados/autoencoder/modelo_autoencoder.pt",
            "resultados/autoencoder/scaler.pkl",
            "resultados/autoencoder/scaler.pkl.sha256",
            "resultados/autoencoder/normalizacao_baseline_gpvs.npz",
            "resultados/autoencoder/estatistica_residuo.npz",
            "resultados/autoencoder/diagnostico_autoencoder.npz",
            "resultados/autoencoder/limiar.json",
        ),
        evidence_level="E2",
    ),
}

ORDEM_ETAPAS_ML = list(STAGES.keys())
NOMES_ETAPAS = {key: stage.label for key, stage in STAGES.items()}
ARTEFATOS_ML = {key: list(stage.artifacts) for key, stage in STAGES.items()}

DATASET_GPVS = Path(
    os.getenv(
        "AL_IADO_DATASET_GPVS",
        str(RAIZ_PROJETO / "dados" / "brutos" / "gpvs" / "csv" / "CSV_Files"),
    )
).expanduser().resolve()

# Subconjunto leve e versionável necessário para consultar uma execução já
# concluída. Pesos, scalers e dados processados permanecem locais.
ARTEFATOS_PUBLICADOS: dict[str, tuple[str, ...]] = {
    "features_gpvs": ("resultados/manifestos/features_gpvs.json",),
    "autoencoder": (
        "resultados/manifestos/autoencoder.json",
        "resultados/autoencoder/limiar.json",
        "resultados/autoencoder/calibracao_autoencoder.csv",
        "resultados/autoencoder/calibracao_autoencoder.md",
        "resultados/autoencoder/distribuicao_erro.png",
    ),
    "injecao_falhas": (
        "resultados/manifestos/injecao_falhas.json",
        "resultados/autoencoder/injecao_falhas_report.json",
        "resultados/autoencoder/injecao_smd_tabela.csv",
    ),
    "validacao": (
        "resultados/manifestos/validacao.json",
        "resultados/autoencoder/validacao_report.json",
        "resultados/autoencoder/validacao_tabela.csv",
        "resultados/gpvs/validacao_gpvs_e3.json",
        "resultados/gpvs/validacao_gpvs_cenarios.csv",
        "resultados/gpvs/relatorio_validacao_gpvs.md",
    ),
    "rul_weibull": (
        "resultados/manifestos/rul_weibull.json",
        "resultados/autoencoder/weibull_results.json",
        "resultados/autoencoder/weibull_tabela.csv",
    ),
}

STAGE_ALIASES = {"features_ca": "features_gpvs"}


def _chave_canonica(key: str) -> str:
    return STAGE_ALIASES.get(key, key)


def get_stage(key: str) -> PipelineStage:
    key = _chave_canonica(key)
    try:
        return STAGES[key]
    except KeyError as exc:
        raise ValueError(f"Etapa desconhecida: {key}") from exc


def capacidade_recalculo_pipeline() -> dict:
    """Distingue o ambiente de cálculo local do modo de consulta do deploy."""
    esperados = [DATASET_GPVS / Path(relativo).name for relativo in GPVS_ALL_INPUTS]
    ausentes = [str(path) for path in esperados if not path.is_file()]
    disponivel = not ausentes
    return {
        "disponivel": disponivel,
        "modo": "calculo_local" if disponivel else "consulta_publicada",
        "dataset": str(DATASET_GPVS),
        "arquivos_esperados": len(esperados),
        "arquivos_ausentes": ausentes,
    }


def estado_resultados_publicados() -> dict[str, dict]:
    """Disponibilidade dos artefatos leves que o site pode consultar."""
    estados: dict[str, dict] = {}
    for key, relativos in ARTEFATOS_PUBLICADOS.items():
        caminhos = [RAIZ_PROJETO / relativo for relativo in relativos]
        presentes = [path for path in caminhos if path.is_file()]
        estados[key] = {
            "disponivel": len(presentes) == len(caminhos),
            "presentes": len(presentes),
            "esperados": len(caminhos),
        }
    return estados


def etapa_pendente(key: str) -> bool:
    return not get_stage(key).is_complete()


def features_ca_pendente() -> bool:
    """Alias histórico; o pipeline principal agora usa features GPVS."""
    return etapa_pendente("features_gpvs")


def features_gpvs_pendente() -> bool:
    return etapa_pendente("features_gpvs")


def autoencoder_pendente() -> bool:
    return etapa_pendente("autoencoder")


def injecao_falhas_pendente() -> bool:
    return etapa_pendente("injecao_falhas")


def validacao_pendente() -> bool:
    return etapa_pendente("validacao")


def rul_weibull_pendente() -> bool:
    return etapa_pendente("rul_weibull")


def pipeline_status() -> dict[str, bool]:
    """Retorna {etapa: pronto}. (Mantido para compatibilidade — booleano.)"""
    return {key: stage.is_complete() for key, stage in STAGES.items()}


# ── Proveniência: estado ready / stale / pending por etapa ──────────────────

def _code_path(stage: PipelineStage) -> str:
    """Caminho do arquivo-fonte da etapa (para o hash de código do manifesto)."""
    from importlib.util import find_spec

    try:
        spec = find_spec(stage.module)
        return spec.origin if spec and spec.origin else ""
    except Exception:
        return ""


def _code_dependencies(stage: PipelineStage) -> dict[str, str]:
    """Módulos científicos compartilhados que também invalidam a etapa."""
    from importlib.util import find_spec

    deps: dict[str, str] = {}
    for module in stage.code_dependencies:
        try:
            spec = find_spec(module)
        except Exception:
            continue
        if spec and spec.origin:
            deps[module] = spec.origin
    return deps


def _inputs_da_etapa(stage: PipelineStage) -> dict:
    """Entradas científicas realmente lidas pela etapa.

    `depends_on` define ordem de execução. `input_artifacts` define proveniência;
    separar os dois impede que uma alteração em PNG/Markdown invalide cálculos
    que não consomem esses arquivos.
    """
    return {
        f"input:{idx}:{Path(relativo).name}": str(RAIZ_PROJETO / relativo)
        for idx, relativo in enumerate(stage.input_artifacts)
    }


def registrar_manifesto(key: str, parameters: dict | None = None,
                        evidence_level: str | None = None) -> None:
    """Salva o manifesto de proveniência de uma etapa recém-concluída."""
    key = _chave_canonica(key)
    try:
        from src.ml.proveniencia import gerar_manifesto, salvar_manifesto

        stage = get_stage(key)
        parametros = parameters if parameters is not None else stage.parameters()
        manifesto = gerar_manifesto(
            key, _code_path(stage), parametros,
            _inputs_da_etapa(stage), [str(p) for p in stage.paths()],
            code_dependencies=_code_dependencies(stage),
            evidence_level=evidence_level or stage.evidence_level,
        )
        salvar_manifesto(manifesto)
    except Exception:
        # Manifesto é rastreabilidade, não deve derrubar a execução da etapa,
        # mas a falha é REGISTRADA (não silenciada).
        from src.core.logs import get_logger

        get_logger("pipeline").exception("falha ao registrar manifesto de %s", key)


def _data_do_manifesto(key: str) -> str:
    """`created_at` do manifesto da etapa, para carimbar de quando é o artefato.

    Existe para que a resposta de SKIP nunca apresente números sem dizer de que
    execução eles vêm. Ver docs/auditoria_total_src.md secao 2.
    """
    from src.ml.proveniencia import carregar_manifesto

    key = _chave_canonica(key)

    try:
        salvo = carregar_manifesto(key) or {}
    except Exception:  # noqa: BLE001 - diagnóstico nunca derruba a etapa
        return "data desconhecida"
    return str(salvo.get("created_at") or "data desconhecida")


def estado_etapa_completo(key: str) -> dict:
    """{'estado': ready|stale|pending, 'motivos': [...]} via manifesto."""
    from src.ml.proveniencia import estado_etapa

    key = _chave_canonica(key)
    stage = get_stage(key)
    return estado_etapa(
        key, [str(p) for p in stage.paths()],
        _code_path(stage), stage.parameters(), _inputs_da_etapa(stage),
        _code_dependencies(stage),
    )


def estado_pipeline() -> dict[str, dict]:
    """Estado de 3 valores (ready/stale/pending) de todas as etapas."""
    return {k: estado_etapa_completo(k) for k in ORDEM_ETAPAS_ML}


def status_markdown() -> str:
    rotulo = {
        "ready": "pronto",
        "stale": "desatualizado (stale)",
        "pending": "pendente",
    }
    estados = estado_pipeline()
    linhas = ["## Status do pipeline de ML\n"]
    for key in ORDEM_ETAPAS_ML:
        stage = STAGES[key]
        info = estados[key]
        status = rotulo.get(info["estado"], info["estado"])
        if info["estado"] == "stale" and info.get("motivos"):
            status += f" — {', '.join(info['motivos'])}"
        linhas.append(f"- {stage.label}: **{status}**")
    linhas.append(
        "\n_stale = artefato existe, mas código, parâmetros ou artefatos upstream mudaram._"
    )
    return "\n".join(linhas)


def artefatos_a_partir(etapa_inicial: str) -> list[Path]:
    """Lista artefatos da etapa e de todas as que dependem dela."""
    etapa_inicial = _chave_canonica(etapa_inicial)
    idx = ORDEM_ETAPAS_ML.index(etapa_inicial)
    artefatos = []
    for key in ORDEM_ETAPAS_ML[idx:]:
        artefatos.extend(STAGES[key].paths())
    return artefatos


def limpar_artefatos(etapa_inicial: str) -> list[Path]:
    """Apaga artefatos da etapa e de todas as que dependem dela."""
    removidos = []
    for path in artefatos_a_partir(etapa_inicial):
        if path.exists():
            path.unlink()
            removidos.append(path)
    return removidos


def _precisa_rodar(key: str) -> bool:
    """True se a etapa NÃO está ready (pending OU stale) — precisa (re)rodar."""
    try:
        return estado_etapa_completo(key).get("estado") != "ready"
    except Exception:
        return etapa_pendente(key)


def dependencias_pendentes(etapa: str) -> list[str]:
    etapa = _chave_canonica(etapa)
    stage = get_stage(etapa)
    return [dep for dep in stage.depends_on if _precisa_rodar(dep)]


def _rodar_stage(stage: PipelineStage, progresso=None) -> bool:
    """Executa etapas reais em subprocesso; fakes de teste seguem in-process."""
    if (
        not isinstance(stage, PipelineStage)
        or os.environ.get("AL_IADO_SEM_ISOLAMENTO") == "1"
        or os.environ.get("AL_IADO_PIPELINE_CHILD") == "1"
    ):
        return bool(stage.load_runner()())

    from src.core.seguranca import env_minimo_subprocesso

    cmd = [sys.executable, "-m", "src.ml.exec_etapa_isolada", stage.key]
    env = env_minimo_subprocesso(extras={
        "AL_IADO_PIPELINE_CHILD": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    proc = subprocess.Popen(
        cmd,
        cwd=str(RAIZ_PROJETO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ultimas: list[str] = []
    assert proc.stdout is not None
    for linha in proc.stdout:
        linha = linha.rstrip()
        if not linha:
            continue
        ultimas.append(linha)
        ultimas = ultimas[-30:]
        if progresso:
            progresso(linha)
    proc.wait()
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    detalhe = "\n".join(ultimas[-12:])
    raise RuntimeError(
        f"subprocesso da etapa {stage.label} terminou com código "
        f"{proc.returncode}.\n{detalhe}"
    )


def executar_etapa(etapa: str,
                   *,
                   force: bool = False,
                   auto_deps: bool = True,
                   progresso=None) -> dict:
    """
    Executa uma unica etapa, respeitando dependencias.
    Se force=True, limpa os artefatos da etapa antes de executar.
    Se auto_deps=True (default), roda dependencias pendentes automaticamente
    em vez de retornar erro — comportamento que o usuario espera ao pedir
    "rode a validacao" mesmo sem ter rodado features e autoencoder antes.
    """
    etapa = _chave_canonica(etapa)
    stage = get_stage(etapa)

    # Só pula quando a etapa está READY (artefatos presentes E manifesto
    # compatível: código, parâmetros e artefatos upstream inalterados). Se
    # estiver STALE (ex.: código da injeção/validação mudou após um git pull),
    # RE-EXECUTA — senão os artefatos/gráficos ficariam defasados. `pending`
    # também roda. Ver src/ml/proveniencia.estado_etapa.
    estado_atual = None
    if not force:
        completo = estado_etapa_completo(etapa)
        estado_atual = completo.get("estado")
        if estado_atual == "ready":
            # A mensagem precisa DIZER que nada foi recalculado, e desde quando.
            # Antes era so "ja esta pronto", e o chamador concatenava a tabela de
            # resultados logo abaixo -- o que lia como execucao fresca. Como o
            # treino e deterministico (semente fixa), o pesquisador nao tinha como
            # distinguir SKIP de recalculo olhando os arquivos.
            # Ver docs/auditoria_total_src.md secao 2.
            desde = _data_do_manifesto(etapa)
            return {
                "ok": True,
                "etapa": stage.label,
                "executou": False,
                "recalculou": False,
                "artefatos_de": desde,
                "mensagem": (
                    f"NAO recalculei. {stage.label} esta READY desde {desde}; "
                    "os numeros abaixo vem desse artefato, nao de uma execucao "
                    "agora. Para forcar, peca 'recalcule tudo do zero'."
                ),
            }
    pendentes = dependencias_pendentes(etapa)
    if pendentes:
        if not auto_deps:
            nomes = ", ".join(NOMES_ETAPAS[p] for p in pendentes)
            return {
                "ok": False,
                "etapa": stage.label,
                "executou": False,
                "mensagem": f"{stage.label} depende de: {nomes}.",
            }

        # Roda dependencias pendentes em ordem antes da etapa alvo.
        executadas_extra = []
        for dep_key in ORDEM_ETAPAS_ML:
            if dep_key == etapa:
                break
            if _precisa_rodar(dep_key):
                if progresso:
                    progresso(f"Pré-requisito: rodando {NOMES_ETAPAS[dep_key]}...")
                dep_stage = get_stage(dep_key)
                try:
                    ok_dep = _rodar_stage(dep_stage, progresso=progresso)
                except Exception as exc:
                    return {
                        "ok": False,
                        "etapa": dep_stage.label,
                        "executou": True,
                        "mensagem": (
                            f"Falha ao preparar pré-requisito '{dep_stage.label}' "
                            f"para {stage.label}: {exc}"
                        ),
                    }
                if not ok_dep:
                    return {
                        "ok": False,
                        "etapa": dep_stage.label,
                        "executou": True,
                        "mensagem": (
                            f"Pré-requisito '{dep_stage.label}' falhou — "
                            f"não consegui chegar até {stage.label}."
                        ),
                    }
                registrar_manifesto(dep_key)
                executadas_extra.append(dep_stage.label)

    # Só remove resultados antigos depois que todas as dependências necessárias
    # estão disponíveis. Assim uma falha de preparação não destrói a cópia
    # publicada que ainda pode ser consultada.
    if force or estado_atual == "stale":
        limpar_artefatos(etapa)

    if progresso:
        progresso(f"Executando: {stage.label}...")

    try:
        ok = _rodar_stage(stage, progresso=progresso)
    except Exception as exc:
        return {
            "ok": False,
            "etapa": stage.label,
            "executou": True,
            "mensagem": f"Erro ao executar {stage.label}: {exc}",
        }

    # Etapa concluída → registra manifesto de proveniência (rastreabilidade +
    # detecção de stale futura). Não derruba a execução se falhar.
    if ok:
        registrar_manifesto(etapa)

    msg_base = (
        f"{stage.label} concluido com sucesso."
        if ok else f"{stage.label} retornou falha."
    )
    if 'executadas_extra' in locals() and executadas_extra:
        msg_base = (
            f"Pré-requisitos executados: {', '.join(executadas_extra)}. "
            + msg_base
        )

    return {
        "ok": ok,
        "etapa": stage.label,
        "executou": True,
        "mensagem": msg_base,
    }


def regenerar_pipeline(etapa_inicial: str = "features_gpvs",
                       progresso=None) -> list[str]:
    """Refaz a etapa inicial e todas as etapas dependentes."""
    etapa_inicial = _chave_canonica(etapa_inicial)
    limpar_artefatos(etapa_inicial)
    return executar_pipeline_ml(etapa_inicial, force=False, progresso=progresso)


def executar_pipeline_ml(etapa_inicial: str = "features_gpvs",
                         *,
                         force: bool = False,
                         progresso=None) -> list[str]:
    """
    Executa o pipeline em ordem a partir de uma etapa.
    Sem force, etapas prontas sao puladas.
    """
    etapa_inicial = _chave_canonica(etapa_inicial)
    if force:
        limpar_artefatos(etapa_inicial)

    idx = ORDEM_ETAPAS_ML.index(etapa_inicial)
    resultados = []

    for key in ORDEM_ETAPAS_ML[idx:]:
        res = executar_etapa(key, force=False, progresso=progresso)
        # O pipeline completo roda no mesmo processo do Streamlit. Coleta entre
        # etapas evita reter dataframes/figuras grandes até a validação seguinte.
        gc.collect()
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None and getattr(torch_mod, "cuda", None):
            if torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        prefixo = "OK" if res["ok"] else "ERRO"
        if res["ok"] and not res.get("executou"):
            prefixo = "SKIP"
        msg = f"{prefixo} - {res['mensagem']}"
        resultados.append(msg)
        if progresso:
            progresso(msg)
        if not res["ok"]:
            break

    return resultados
