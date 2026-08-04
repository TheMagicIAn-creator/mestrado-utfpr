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
    parameter_names: tuple[str, ...] = ()
    code_dependencies: tuple[str, ...] = ()

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


STAGES: dict[str, PipelineStage] = {
    "features_ca": PipelineStage(
        key="features_ca",
        label="Features CA",
        module="src.ml.features_ca",
        function="executar_features_ca",
        parameter_names=("FS", "F0", "JANELA", "SOBREPOSICAO", "HARMONICOS"),
        code_dependencies=("src.ml.estilo_graficos",),
        artifacts=(
            "dados/processados/features_paderborn.parquet",
            "dados/processados/features_paderborn_stats.csv",
            "dados/processados/features_paderborn_qualidade.json",
            "dados/processados/features_paderborn_qualidade.png",
        ),
    ),
    "autoencoder": PipelineStage(
        key="autoencoder",
        label="Autoencoder",
        module="src.ml.autoencoder",
        function="executar_autoencoder",
        parameter_names=(
            "LATENTE_DIM", "EPOCHS", "BATCH_SIZE", "LR", "SIGMA",
            "THRESHOLD_METHOD", "SEED", "TRAIN_RATIO", "CALIB_RATIO",
            "TEST_RATIO",
        ),
        code_dependencies=(
            "src.ml.escore_anomalia",
            "src.ml.split_temporal",
            "src.ml.graficos_autoencoder",
            "src.ml.estilo_graficos",
        ),
        artifacts=(
            "resultados/autoencoder/modelo_autoencoder.pt",
            "resultados/autoencoder/scaler.pkl",
            "resultados/autoencoder/limiar.json",
            "resultados/autoencoder/diagnostico_autoencoder.npz",
            "resultados/autoencoder/curva_treino.png",
            "resultados/autoencoder/distribuicao_erro.png",
            "resultados/autoencoder/erro_temporal.png",
        ),
        depends_on=("features_ca",),
    ),
    "injecao_falhas": PipelineStage(
        key="injecao_falhas",
        label="Injecao de Falhas",
        module="src.ml.injecao_falhas",
        function="executar_injecao_falhas",
        parameter_names=("SEVERIDADES", "ALVO_SMD", "N_JANELAS_SMD"),
        code_dependencies=(
            "src.ml.features_ca",
            "src.ml.autoencoder",
            "src.ml.dados_avaliacao",
            "src.ml.escore_anomalia",
            "src.ml.estatistica",
            "src.ml.estilo_graficos",
        ),
        artifacts=(
            "resultados/autoencoder/injecao_falhas_resultados.png",
            "resultados/autoencoder/injecao_falhas_comparacao.png",
            "resultados/autoencoder/injecao_falhas_report.json",
            "resultados/autoencoder/injecao_smd_tabela.csv",
        ),
        depends_on=("autoencoder",),
    ),
    "validacao": PipelineStage(
        key="validacao",
        label="Validacao Interna E2",
        module="src.ml.validacao",
        function="executar_validacao",
        parameter_names=("SEVS_VALIDACAO", "N_JANELAS_SAUDAVEL", "N_JANELAS_FALHA"),
        code_dependencies=(
            "src.ml.features_ca",
            "src.ml.autoencoder",
            "src.ml.dados_avaliacao",
            "src.ml.escore_anomalia",
            "src.ml.estatistica",
            "src.ml.injecao_falhas",
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
        ),
        depends_on=("injecao_falhas",),
    ),
    "rul_weibull": PipelineStage(
        key="rul_weibull",
        label="RUL / Weibull",
        module="src.ml.rul_weibull",
        function="executar_rul_weibull",
        parameter_names=(
            "N_TRAJ", "N_STEPS", "BATCH_INFERENCIA", "N_BOOTSTRAP",
            "MIN_EVENTOS_WEIBULL", "MAX_CENSURA_RUL_PCT",
        ),
        code_dependencies=(
            "src.ml.features_ca",
            "src.ml.autoencoder",
            "src.ml.dados_avaliacao",
            "src.ml.escore_anomalia",
            "src.ml.estatistica",
            "src.ml.injecao_falhas",
            "src.ml.estilo_graficos",
        ),
        artifacts=(
            "resultados/autoencoder/weibull_ttf.png",
            "resultados/autoencoder/weibull_confiabilidade.png",
            "resultados/autoencoder/weibull_rul.png",
            "resultados/autoencoder/weibull_results.json",
            "resultados/autoencoder/weibull_tabela.csv",
        ),
        depends_on=("validacao",),
    ),
}

ORDEM_ETAPAS_ML = list(STAGES.keys())
NOMES_ETAPAS = {key: stage.label for key, stage in STAGES.items()}
ARTEFATOS_ML = {key: list(stage.artifacts) for key, stage in STAGES.items()}

DATASET_PADERBORN = Path(
    os.getenv(
        "AL_IADO_DATASET_PADERBORN",
        str(RAIZ_PROJETO / "dados" / "brutos" / "Inverter_Data_Set.csv"),
    )
).expanduser().resolve()

# Subconjunto leve e versionável necessário para consultar uma execução já
# concluída. Pesos, scalers e dados processados permanecem locais.
ARTEFATOS_PUBLICADOS: dict[str, tuple[str, ...]] = {
    "features_ca": ("resultados/manifestos/features_ca.json",),
    "autoencoder": (
        "resultados/manifestos/autoencoder.json",
        "resultados/autoencoder/limiar.json",
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
    ),
    "rul_weibull": (
        "resultados/manifestos/rul_weibull.json",
        "resultados/autoencoder/weibull_results.json",
        "resultados/autoencoder/weibull_tabela.csv",
    ),
}


def get_stage(key: str) -> PipelineStage:
    try:
        return STAGES[key]
    except KeyError as exc:
        raise ValueError(f"Etapa desconhecida: {key}") from exc


def capacidade_recalculo_pipeline() -> dict:
    """Distingue o ambiente de cálculo local do modo de consulta do deploy."""
    disponivel = DATASET_PADERBORN.is_file()
    return {
        "disponivel": disponivel,
        "modo": "calculo_local" if disponivel else "consulta_publicada",
        "dataset": str(DATASET_PADERBORN),
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
    return etapa_pendente("features_ca")


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
    """Todos os artefatos upstream, para detectar qualquer regeneração relevante."""
    inputs: dict[str, str] = {}
    for dep in stage.depends_on:
        for idx, path in enumerate(STAGES[dep].paths()):
            inputs[f"{dep}:{idx}:{path.name}"] = str(path)
    return inputs


def registrar_manifesto(key: str, parameters: dict | None = None,
                        evidence_level: str | None = None) -> None:
    """Salva o manifesto de proveniência de uma etapa recém-concluída."""
    try:
        from src.ml.proveniencia import gerar_manifesto, salvar_manifesto

        stage = get_stage(key)
        parametros = parameters if parameters is not None else stage.parameters()
        manifesto = gerar_manifesto(
            key, _code_path(stage), parametros,
            _inputs_da_etapa(stage), [str(p) for p in stage.paths()],
            code_dependencies=_code_dependencies(stage),
            evidence_level=evidence_level,
        )
        salvar_manifesto(manifesto)
    except Exception:
        # Manifesto é rastreabilidade, não deve derrubar a execução da etapa,
        # mas a falha é REGISTRADA (não silenciada).
        from src.core.logs import get_logger

        get_logger("pipeline").exception("falha ao registrar manifesto de %s", key)


def estado_etapa_completo(key: str) -> dict:
    """{'estado': ready|stale|pending, 'motivos': [...]} via manifesto."""
    from src.ml.proveniencia import estado_etapa

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
    env = env_minimo_subprocesso(extras={"AL_IADO_PIPELINE_CHILD": "1"})
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
    stage = get_stage(etapa)

    # Só pula quando a etapa está READY (artefatos presentes E manifesto
    # compatível: código, parâmetros e artefatos upstream inalterados). Se
    # estiver STALE (ex.: código da injeção/validação mudou após um git pull),
    # RE-EXECUTA — senão os artefatos/gráficos ficariam defasados. `pending`
    # também roda. Ver src/ml/proveniencia.estado_etapa.
    if force:
        limpar_artefatos(etapa)
    else:
        estado_atual = estado_etapa_completo(etapa).get("estado")
        if estado_atual == "ready":
            return {
                "ok": True,
                "etapa": stage.label,
                "executou": False,
                "mensagem": f"{stage.label} ja esta pronto.",
            }
        if estado_atual == "stale":
            # limpa a etapa (e o downstream, que também precisa regenerar).
            limpar_artefatos(etapa)

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


def regenerar_pipeline(etapa_inicial: str = "features_ca",
                       progresso=None) -> list[str]:
    """Refaz a etapa inicial e todas as etapas dependentes."""
    limpar_artefatos(etapa_inicial)
    return executar_pipeline_ml(etapa_inicial, force=False, progresso=progresso)


def executar_pipeline_ml(etapa_inicial: str = "features_ca",
                         *,
                         force: bool = False,
                         progresso=None) -> list[str]:
    """
    Executa o pipeline em ordem a partir de uma etapa.
    Sem force, etapas prontas sao puladas.
    """
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
