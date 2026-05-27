"""
resultados.py - Al IAdo PV
Leitura e resumo dos artefatos do pipeline de Machine Learning.

As respostas de resultados agora saem por prompt, no chat. Este modulo
centraliza os JSONs, tabelas e graficos para evitar dashboards paralelos.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.core.config import PASTA_CHROMADB, RAIZ_PROJETO

PASTA_AE = RAIZ_PROJETO / "resultados" / "autoencoder"


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(valor, casas: int = 3) -> str:
    if isinstance(valor, (int, float)):
        return f"{valor:.{casas}f}"
    return str(valor)


def _normalizar(texto: str) -> str:
    import unicodedata

    texto = texto.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _focos(pergunta: str) -> set[str]:
    txt = _normalizar(pergunta)
    focos = set()
    if any(t in txt for t in ("autoencoder", "limiar", "baseline", "reconstrucao")):
        focos.add("autoencoder")
    if any(t in txt for t in ("injecao", "falha", "falhas", "smd", "severidade")):
        focos.add("injecao")
    if any(t in txt for t in ("validacao", "auc", "f1", "recall", "precision", "roc", "matriz")):
        focos.add("validacao")
    if any(t in txt for t in ("weibull", "rul", "mttf", "b10", "beta", "eta", "confiabilidade")):
        focos.add("weibull")
    return focos


def _quer_imagens(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    return any(t in txt for t in (
        "grafico", "graficos", "imagem", "imagens", "figura", "figuras",
        "curva", "curvas", "plot", "plots", "roc", "matriz", "matrizes",
        "heatmap", "visual", "visualiza", "mostre", "mostra", "mostrar",
        "cade", "exibe", "exibir", "veja", "ver",
    ))


def _add_img(imagens: list[dict], nome: str, legenda: str) -> None:
    path = PASTA_AE / nome
    if path.exists():
        imagens.append({"path": str(path.resolve()), "caption": legenda})


def imagens_relevantes(pergunta: str = "") -> list[dict]:
    focos = _focos(pergunta)
    if not focos:
        focos = {"autoencoder", "injecao", "validacao", "weibull"}

    imagens = []
    if "autoencoder" in focos:
        _add_img(imagens, "curva_treino.png", "Autoencoder - curva de treinamento")
        _add_img(imagens, "distribuicao_erro.png", "Autoencoder - distribuicao do erro")
        _add_img(imagens, "erro_temporal.png", "Autoencoder - erro temporal")
    if "injecao" in focos:
        _add_img(imagens, "injecao_falhas_resultados.png", "Falhas sinteticas - erro por severidade")
        _add_img(imagens, "injecao_falhas_comparacao.png", "Falhas sinteticas - comparacao em escala log")
    if "validacao" in focos:
        _add_img(imagens, "validacao_roc.png", "Validacao - curvas ROC")
        _add_img(imagens, "validacao_matriz.png", "Validacao - matrizes de confusao")
        _add_img(imagens, "validacao_metricas.png", "Validacao - heatmap de metricas")
    if "weibull" in focos:
        _add_img(imagens, "weibull_ttf.png", "Weibull - distribuicao TTF")
        _add_img(imagens, "weibull_confiabilidade.png", "Weibull - funcoes de confiabilidade")
        _add_img(imagens, "weibull_rul.png", "Weibull - RUL condicional")
    return imagens


def _resumo_autoencoder() -> str | None:
    d = _json(PASTA_AE / "limiar.json")
    if not d:
        return None
    return (
        "## Autoencoder - modelo de normalidade\n\n"
        "| Métrica | Valor |\n"
        "|---|---:|\n"
        f"| Limiar p99 | {_fmt(d.get('limiar'), 4)} |\n"
        f"| Média baseline | {_fmt(d.get('mu'), 4)} |\n"
        f"| Desvio baseline | {_fmt(d.get('sigma'), 4)} |\n"
        f"| Falsos positivos validação | {_fmt(d.get('fp_val_pct'), 2)}% |\n"
        f"| Épocas treinadas | {d.get('epochs_treinadas', '-')} |\n\n"
        "Leitura rápida: o detector está calibrado por erro de reconstrução. "
        "Quanto maior a distância entre erro de falha e limiar, mais clara é a anomalia."
    )


def _resumo_injecao() -> str | None:
    d = _json(PASTA_AE / "injecao_falhas_report.json")
    if not d:
        return None

    linhas = [
        "## Injeção de falhas sintéticas\n",
        f"Limiar: **{_fmt(d.get('limiar'), 4)}**. "
        f"Baseline: **{_fmt(d.get('baseline_mean'), 4)} ± {_fmt(d.get('baseline_std'), 4)}**.\n",
        "| Falha | NPR | SMD | Erro na SMD | Margem |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for fid, falha in d.get("falhas", {}).items():
        smd = d.get("smd", {}).get(fid)
        erro = margem = "-"
        if smd is not None:
            res = falha.get("resultados", {}).get(str(smd), {})
            erro = _fmt(res.get("erro"), 4)
            margem = f"{_fmt(res.get('margem'), 2)}x"
        linhas.append(
            f"| {falha.get('nome', fid)} | {falha.get('npr') or '-'} | "
            f"{smd if smd is not None else '-'} | {erro} | {margem} |\n"
        )
    linhas.append(
        "\nLeitura rápida: a SMD é a menor severidade em que o Autoencoder cruza o limiar."
    )
    return "".join(linhas)


def _nome_falha(chave: str) -> str:
    base = re.sub(r"_sev.*$", "", chave)
    nomes = {
        "lcl": "Degradação Filtro LCL",
        "desbalanceamento": "Desbalanceamento de Fase",
        "sensor": "Falha de Sensor CA",
    }
    return nomes.get(base, base)


def _sev(chave: str) -> str:
    match = re.search(r"sev([0-9.]+)", chave)
    return match.group(1) if match else "-"


def _resumo_validacao() -> str | None:
    d = _json(PASTA_AE / "validacao_report.json")
    if not d:
        return None

    melhores = {}
    for chave, res in d.items():
        falha = _nome_falha(chave)
        if falha not in melhores or res.get("auc_roc", 0) > melhores[falha].get("auc_roc", 0):
            item = dict(res)
            item["chave"] = chave
            melhores[falha] = item

    linhas = [
        "## Validação formal\n\n",
        "| Falha | Severidade | AUC-ROC | F1 | Recall | Precision |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for falha, res in melhores.items():
        linhas.append(
            f"| {falha} | {_sev(res['chave'])} | {_fmt(res.get('auc_roc'))} | "
            f"{_fmt(res.get('f1'))} | {_fmt(res.get('recall'))} | "
            f"{_fmt(res.get('precision'))} |\n"
        )
    linhas.append(
        "\nLeitura rápida: AUC próximo de 1 indica separação muito forte entre "
        "comportamento saudável e falha injetada."
    )
    return "".join(linhas)


def _resumo_weibull() -> str | None:
    d = _json(PASTA_AE / "weibull_results.json")
    if not d:
        return None

    linhas = [
        "## RUL / Weibull\n\n",
        "| Falha | NPR | beta | eta | MTTF | B10 | Interpretação |\n",
        "|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for fid, falha in d.get("falhas", {}).items():
        p = falha.get("weibull", {})
        beta = p.get("beta")
        taxa = "desgaste progressivo" if isinstance(beta, (int, float)) and beta > 1 else "falha aleatória/infantil"
        linhas.append(
            f"| {falha.get('nome', fid)} | {falha.get('npr') or 'D=10'} | "
            f"{_fmt(beta)} | {_fmt(p.get('eta'), 1)} | {_fmt(p.get('mttf'), 1)} | "
            f"{_fmt(p.get('b10'), 1)} | {taxa} |\n"
        )
    linhas.append(
        "\nLeitura rápida: beta > 1 sustenta a hipótese de degradação progressiva, "
        "coerente com manutenção preditiva."
    )
    return "".join(linhas)


def resumir_resultados(pergunta: str = "", *, incluir_imagens: bool = True) -> dict:
    focos = _focos(pergunta)
    if not focos:
        focos = {"autoencoder", "injecao", "validacao", "weibull"}

    secoes = []
    if "autoencoder" in focos:
        secoes.append(_resumo_autoencoder())
    if "injecao" in focos:
        secoes.append(_resumo_injecao())
    if "validacao" in focos:
        secoes.append(_resumo_validacao())
    if "weibull" in focos:
        secoes.append(_resumo_weibull())

    secoes = [s for s in secoes if s]
    if not secoes:
        return {
            "ok": True,
            "etapa": "Consulta de resultados",
            "mensagem": (
                "Ainda não encontrei artefatos de resultado para essa solicitação. "
                "Peça para rodar a etapa correspondente do pipeline."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    imagens = imagens_relevantes(pergunta) if incluir_imagens and _quer_imagens(pergunta) else []
    mensagem = (
        "Aqui está o que já existe nos artefatos do pipeline.\n\n"
        + "\n\n".join(secoes)
    )
    if imagens:
        mensagem += "\n\nVou exibir os gráficos relevantes logo abaixo da resposta."

    return {
        "ok": True,
        "etapa": "Consulta de resultados",
        "mensagem": mensagem,
        "imagens": imagens,
        "resposta_pronta": True,
    }


def indexar_resultados_ml(modelo_embeddings) -> str:
    """Gera resumo dos resultados e indexa na memoria do agente."""
    from src.conhecimento.indexador import indexar_sessao

    saida = RAIZ_PROJETO / "notas" / "memorias" / "resultados-fase5-ml.md"
    resumo = resumir_resultados("", incluir_imagens=False)["mensagem"]
    conteudo = (
        "# Resultados da Fase 5 - Pipeline de ML\n\n"
        f"> Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        f"{resumo}\n"
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(conteudo, encoding="utf-8")

    try:
        indexar_sessao(saida, modelo_embeddings, PASTA_CHROMADB)
        return "Resultados indexados. O agente ja pode discuti-los no chat."
    except Exception as exc:
        return f"Resumo salvo, mas houve erro ao indexar: {exc}"
