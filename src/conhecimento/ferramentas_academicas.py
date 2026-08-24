"""Ferramentas acadêmicas sobre literatura, dataset e resultados canônicos."""

from __future__ import annotations

import re

from src.ml.dados_gpvs import ALL_EXPERIMENTS, DATASET_DIR, DATASET_DOI, DATASET_NAME
from src.ml.pipeline import capacidade_recalculo_pipeline
from src.ml.resultados import resumir_resultados


def buscar_na_web(progresso=None, pergunta: str = "") -> dict:
    """Adapta a busca externa ao contrato de ferramenta do agente."""

    from src.conhecimento.web_search import buscar_web

    if progresso:
        progresso(f"Pesquisando na web: '{pergunta[:60]}'...")
    term = (pergunta or "").strip()
    for trigger in (
        "buscar na web",
        "pesquisar na web",
        "pesquise na web",
        "busque na web",
        "buscar online",
        "pesquisar online",
        "procure na internet",
        "procure online",
        "na internet",
        "na web",
        "buscar",
        "pesquisar",
        "procurar",
        "google",
        "googlar",
    ):
        term = re.sub(rf"\b{trigger}\b", "", term, flags=re.IGNORECASE)
    term = term.strip(" ,.;?!:")
    if not term:
        return {
            "ok": False,
            "etapa": "Busca na web",
            "mensagem": "Informe o tema que deseja pesquisar.",
            "imagens": [],
            "resposta_pronta": True,
        }
    result = buscar_web(term)
    return {
        "ok": bool(result["ok"]),
        "etapa": "Busca na web",
        "mensagem": result["mensagem"],
        "imagens": [],
        "resposta_pronta": False,
    }


def listar_base_bibliografica(progresso=None, pergunta: str = "") -> dict:
    """Lista deterministicamente todos os documentos indexados no ChromaDB."""

    if progresso:
        progresso("Lendo o catálogo completo da base de conhecimento...")
    try:
        import chromadb

        from src.conhecimento.agente import catalogo_literatura
        from src.core.config import NOME_COLECAO, PASTA_CHROMADB

        client = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        collection = client.get_collection(NOME_COLECAO)
        text = catalogo_literatura(collection)
    except Exception as exc:  # diagnóstico operacional
        return {
            "ok": False,
            "etapa": "Base bibliográfica",
            "mensagem": (
                "Não consegui ler o catálogo da base de conhecimento agora "
                f"({exc}). Verifique se o índice foi construído."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }
    return {
        "ok": True,
        "etapa": "Base bibliográfica",
        "mensagem": text,
        "imagens": [],
        "resposta_pronta": True,
    }


def consultar_comparacao_autoencoders(progresso=None, pergunta: str = "") -> dict:
    """Consulta a comparação publicada, sem treinar nem recalibrar modelos."""

    if progresso:
        progresso("Lendo a comparação Denso versus AE-LSTM...")
    return resumir_resultados(f"{pergunta} e3 denso lstm")


def consultar_datasets(progresso=None, pergunta: str = "") -> dict:
    """Explica o único dataset ativo e seus papéis experimentais."""

    if progresso:
        progresso("Validando o inventário GPVS-Faults...")
    capacity = capacidade_recalculo_pipeline()
    present = len(ALL_EXPERIMENTS) - len(capacity["arquivos_ausentes"])
    message = f"""## Dataset canônico

**{DATASET_NAME}** é o único conjunto de dados ativo no pipeline.

- DOI: https://doi.org/{DATASET_DOI}
- Diretório local: `{DATASET_DIR}`
- Ensaios encontrados: {present}/{len(ALL_EXPERIMENTS)}
- F0L/F0M: treino, validação, calibração e teste saudável em blocos temporais disjuntos.
- F1L-F7M: avaliação E3 nos 14 ensaios reais de bancada, sem retreino ou recalibração.

Paderborn, PMSM, PV Farms e telemetria residencial não fornecem amostras,
features, métricas ou modelos à publicação vigente. Permanecem apenas como
contexto bibliográfico quando presentes na base semântica.

O GPVS-Faults não contém tempos de vida e censura por ativo; portanto, não
estima confiabilidade física, RUL ou Weibull físico.
"""
    return {
        "ok": True,
        "etapa": "Dataset canônico",
        "mensagem": message,
        "imagens": [],
        "resposta_pronta": True,
    }


def comparar_abordagens_ml(progresso=None, pergunta: str = "") -> dict:
    """Delimita os dois modelos e a confiabilidade bibliográfica publicada."""

    message = """## Abordagens mantidas

| Domínio | Método | Evidência | Interpretação |
|---|---|---|---|
| GPVS saudável | Autoencoder Denso e AE-LSTM | ajuste do detector | aprendizagem não supervisionada da normalidade |
| GPVS F1L-F7M | os mesmos modelos congelados | E3 de bancada | discriminação experimental por ensaio |
| Literatura de confiabilidade | modelo exponencial | cenário bibliográfico | `R(t)`, `F(t)`, `f(t)` e `h(t)` no tempo físico |

O Autoencoder Denso e o AE-LSTM usam as mesmas 24 features, partições, sementes
e orçamento de treino. AUC-PR é a métrica E3 principal. Os cenários físicos
usam somente taxas bibliográficas diretas ou derivadas e não são inferidos da
base experimental.
"""
    return {
        "ok": True,
        "etapa": "Abordagens canônicas",
        "mensagem": message,
        "imagens": [],
        "resposta_pronta": True,
        "forcar_resposta_direta": True,
    }


__all__ = [
    "buscar_na_web",
    "comparar_abordagens_ml",
    "consultar_comparacao_autoencoders",
    "consultar_datasets",
    "listar_base_bibliografica",
]
