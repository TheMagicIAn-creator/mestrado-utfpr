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
PASTA_EXPERIMENTOS = RAIZ_PROJETO / "resultados" / "experimentos"


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


_EXPERIMENTOS_ALIASES = {
    "ghoneim": "ghoneim",
    "francisti": "francisti",
    "ibrahim": "ibrahim",
    "sharma": "sharma",
    "ahirwar": "ahirwar",
    "stender": "stender",
}
_EXPERIMENTOS_ANOMALIA = {"francisti", "ibrahim", "sharma", "ahirwar"}


def _slug_modelo(nome: str) -> str:
    import unicodedata

    texto = unicodedata.normalize("NFD", nome.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_") or "modelo"


def _quer_resultado_experimentos(txt: str) -> bool:
    if any(autor in txt for autor in _EXPERIMENTOS_ALIASES):
        return True
    if any(t in txt for t in (
        "experimento", "experimentos", "artigo", "artigos", "benchmark",
        "comparacao", "comparar", "ppo",
        "experiment", "experiments", "paper", "papers", "comparison", "compare",
        "experimento", "experimentos", "articulo", "articulos", "comparacion",
        "comparar", "experience", "expérience", "article", "articles",
        "comparaison", "comparer",
    )):
        return True
    if any(t in txt for t in ("modelo", "modelos", "model", "models", "modele", "modeles", "modèle", "modèles")):
        return any(t in txt for t in (
            "anomalia", "anomalias", "detectaram", "detectou", "melhor",
            "confiavel", "robusto", "matriz", "metricas", "grafico", "graficos",
            "anomaly", "anomalies", "detected", "best", "matrix",
            "reliable", "robust", "metrics", "chart", "charts", "plot", "plots",
            "anomalia", "anomalias", "detectadas", "mejor", "confiable", "robusto", "matriz",
            "metricas", "grafico", "graficos",
            "anomalie", "anomalies", "detectees", "détectées", "meilleur", "fiable", "robuste",
            "matrice", "metriques", "métriques", "graphique", "graphiques",
        ))
    return False


def _experimentos_pedidos(pergunta: str = "") -> list[str]:
    txt = _normalizar(pergunta)
    encontrados = []
    for nome, key in _EXPERIMENTOS_ALIASES.items():
        pos = txt.find(nome)
        if pos >= 0:
            encontrados.append((pos, key))
    pedidos = [key for _, key in sorted(encontrados)]
    if pedidos:
        return pedidos
    if any(t in txt for t in ("anomalia", "anomalias", "anomaly", "anomalies", "anomalie")):
        return ["francisti", "ibrahim", "sharma", "ahirwar"]
    if any(t in txt for t in ("classificacao", "supervision", "classification", "clasificacion")):
        return ["ghoneim"]
    return []


def _arquivos_experimentos(pergunta: str = "") -> list[Path]:
    arquivos = sorted(PASTA_EXPERIMENTOS.glob("*/resultado.json"))
    pedidos_lista = _experimentos_pedidos(pergunta)
    pedidos = set(pedidos_lista)
    if pedidos:
        por_nome = {arq.parent.name: arq for arq in arquivos}
        arquivos = [por_nome[key] for key in pedidos_lista if key in por_nome]
    return arquivos


def _pede_matriz(txt: str) -> bool:
    return any(t in txt for t in ("matriz", "matrizes", "confusao", "matrix", "confusion", "matrice"))


def _pede_graficos_modelo(txt: str) -> bool:
    return any(t in txt for t in (
        "grafico", "graficos", "grafica", "graficas", "metricas", "metrica",
        "plot", "plots", "chart", "charts", "figure", "figures",
        "imagen", "imagenes", "figura", "figuras",
        "graphique", "graphiques", "metriques", "métriques",
    ))


def _pede_anomalias(txt: str) -> bool:
    return any(t in txt for t in (
        "anomalias detectadas", "detectaram", "detectou",
        "detected anomalies", "detected", "detectadas", "detecto",
        "anomalies detectees", "anomalies détectées", "detectees", "détectées",
    ))


def _pede_melhor(txt: str) -> bool:
    return any(t in txt for t in (
        "melhor", "best", "mejor", "meilleur",
        "mais confiavel", "confiavel", "more reliable", "reliable",
        "mas confiable", "confiable", "plus fiable", "fiable",
        "robusto", "robust", "robuste",
    ))


def _contem_termo(txt: str, termo: str) -> bool:
    if len(termo) <= 3 or " " in termo:
        return bool(re.search(rf"\b{re.escape(termo)}\b", txt))
    return termo in txt


def _modelo_citado(txt: str, modelo: str) -> bool:
    nome = _normalizar(modelo)
    slug = _slug_modelo(modelo).replace("_", " ")
    aliases = {nome, slug}
    mapa = {
        "isolation forest": ("isolation forest", "iforest"),
        "ppo": ("ppo", "rl"),
        "random forest": ("random forest", "rf", "bosque aleatorio", "foret aleatoire", "forêt aléatoire"),
        "z-score": ("z-score", "z score", "zscore"),
        "ae-lstm": ("ae-lstm", "ae lstm", "autoencoder lstm"),
        "facebook prophet": ("facebook prophet", "prophet"),
        "hibrido": ("hibrido", "voto"),
        "svm": ("svm",),
        "knn": ("knn",),
        "ann": ("ann", "mlp"),
        "rnn": ("rnn",),
        "cnn": ("cnn",),
        "adaboost": ("adaboost",),
        "naive bayes": ("naive bayes", "bayes"),
        "regressao logistica": ("regressao logistica", "logistica"),
    }
    for chave, valores in mapa.items():
        if chave in nome or chave in slug:
            aliases.update(valores)
    return any(alias and alias in txt for alias in aliases)


def _modelos_citados(txt: str, modelos: dict) -> set[str]:
    return {modelo for modelo in modelos if _modelo_citado(txt, modelo)}


def _focos(pergunta: str) -> set[str]:
    txt = _normalizar(pergunta)
    focos = set()
    quer_experimentos = _quer_resultado_experimentos(txt)

    if any(t in txt for t in ("autoencoder", "limiar", "baseline", "reconstrucao")):
        focos.add("autoencoder")
    if any(t in txt for t in (
        "injecao", "falha", "falhas", "smd", "severidade",
        "injection", "fault", "failure", "severity",
        "inyeccion", "falla", "fallas", "severidad",
        "injection", "defaillance", "defaillances", "severite", "sévérité",
    )):
        focos.add("injecao")
    if any(t in txt for t in ("validacao", "auc", "f1", "recall", "precision", "roc", "matriz")) and (
        not quer_experimentos or "validacao" in txt or "autoencoder" in txt
    ):
        focos.add("validacao")
    if any(t in txt for t in (
        "weibull", "rul", "mttf", "b10", "beta", "eta", "confiabilidade",
        "reliability", "remaining useful life", "confiabilidad", "fiabilite",
    )):
        focos.add("weibull")
    if quer_experimentos:
        focos.add("experimentos")
    return focos


def _quer_imagens(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    return any(_contem_termo(txt, t) for t in (
        "grafico", "graficos", "imagem", "imagens", "figura", "figuras",
        "curva", "curvas", "plot", "plots", "roc", "matriz", "matrizes",
        "heatmap", "visual", "visualiza", "mostre", "mostra", "mostrar",
        "cade", "exibe", "exibir", "veja", "ver",
        "chart", "charts", "image", "images", "figure", "figures",
        "curve", "curves", "matrix", "show", "display", "see",
        "grafico", "graficos", "imagen", "imagenes", "figura", "curva",
        "matriz", "muestra", "mostrar", "ver",
        "graphique", "graphiques", "image", "images", "figure", "courbe",
        "matrice", "montre", "affiche", "voir",
    ))


def _add_img(
    imagens: list[dict],
    nome: str | Path,
    legenda: str,
    pasta: Path = PASTA_AE,
    grupo: str | None = None,
    ordem: int = 0,
    tipo: str = "grafico",
    ordem_grupo: int = 0,
) -> None:
    path = nome if isinstance(nome, Path) else pasta / nome
    if path.exists():
        resolvido = str(path.resolve())
        if not any(img["path"] == resolvido for img in imagens):
            imagens.append({
                "path": resolvido,
                "caption": legenda,
                "group": grupo or "Resultados",
                "group_order": ordem_grupo,
                "order": ordem,
                "kind": tipo,
            })


def _add_grafico_modelo(
    imagens: list[dict],
    pasta: Path,
    modelo: str,
    dados_modelo: dict,
    chave: str,
    legenda: str,
    grupo: str,
    ordem: int,
    ordem_grupo: int,
) -> None:
    caminho = dados_modelo.get(chave)
    tipo = "matriz" if chave == "grafico_matriz_confusao" else "modelo"
    if caminho:
        _add_img(
            imagens,
            Path(caminho),
            legenda,
            grupo=grupo,
            ordem=ordem,
            tipo=tipo,
            ordem_grupo=ordem_grupo,
        )
        return

    sufixo = "metricas" if chave == "grafico_metricas" else "matriz_confusao"
    _add_img(
        imagens,
        pasta / f"modelo_{_slug_modelo(modelo)}_{sufixo}.png",
        legenda,
        grupo=grupo,
        ordem=ordem,
        tipo=tipo,
        ordem_grupo=ordem_grupo,
    )


def imagens_relevantes(pergunta: str = "") -> list[dict]:
    txt = _normalizar(pergunta)
    focos = _focos(pergunta)
    if not focos:
        focos = {"autoencoder", "injecao", "validacao", "weibull", "experimentos"}

    imagens = []
    if "autoencoder" in focos:
        _add_img(imagens, "curva_treino.png", "Autoencoder - curva de treinamento", grupo="Autoencoder", ordem=10, ordem_grupo=10)
        _add_img(imagens, "distribuicao_erro.png", "Autoencoder - distribuicao do erro", grupo="Autoencoder", ordem=20, ordem_grupo=10)
        _add_img(imagens, "erro_temporal.png", "Autoencoder - erro temporal", grupo="Autoencoder", ordem=30, ordem_grupo=10)
    if "injecao" in focos:
        _add_img(imagens, "injecao_falhas_resultados.png", "Falhas sinteticas - erro por severidade", grupo="Injecao de falhas", ordem=10, tipo="comparacao", ordem_grupo=20)
        _add_img(imagens, "injecao_falhas_comparacao.png", "Falhas sinteticas - comparacao em escala log", grupo="Injecao de falhas", ordem=20, tipo="comparacao", ordem_grupo=20)
    if "validacao" in focos:
        _add_img(imagens, "validacao_roc.png", "Validacao - curvas ROC", grupo="Validacao", ordem=10, tipo="comparacao", ordem_grupo=30)
        _add_img(imagens, "validacao_pr.png", "Validacao - curvas Precision-Recall", grupo="Validacao", ordem=15, tipo="comparacao", ordem_grupo=30)
        _add_img(imagens, "validacao_matriz.png", "Validacao - matrizes de confusao", grupo="Validacao", ordem=20, tipo="matriz", ordem_grupo=30)
        _add_img(imagens, "validacao_metricas.png", "Validacao - heatmap de metricas", grupo="Validacao", ordem=30, tipo="comparacao", ordem_grupo=30)
    if "weibull" in focos:
        _add_img(imagens, "weibull_ttf.png", "Weibull - distribuicao TTF", grupo="Weibull / RUL", ordem=10, ordem_grupo=40)
        _add_img(imagens, "weibull_confiabilidade.png", "Weibull - funcoes de confiabilidade", grupo="Weibull / RUL", ordem=20, ordem_grupo=40)
        _add_img(imagens, "weibull_rul.png", "Weibull - RUL condicional", grupo="Weibull / RUL", ordem=30, ordem_grupo=40)
    if "experimentos" in focos:
        pede_matriz = _pede_matriz(txt)
        pede_graficos = _pede_graficos_modelo(txt)
        somente_matriz = pede_matriz and not pede_graficos
        melhor_apenas = _pede_melhor(txt)
        pede_anomalias = _pede_anomalias(txt)

        for idx_exp, resultado in enumerate(_arquivos_experimentos(pergunta)):
            pasta = resultado.parent
            nome = pasta.name
            dados = _json(resultado) or {}
            modelos = dados.get("modelos", {})
            melhor = dados.get("melhor_modelo")
            modelos_pedidos = _modelos_citados(txt, modelos)
            grupo = dados.get("referencia") or f"Experimento {nome}"

            if not somente_matriz:
                _add_img(
                    imagens,
                    pasta / "comparacao_metricas.png",
                    f"{grupo} - comparacao de metricas",
                    grupo=grupo,
                    ordem=0,
                    tipo="comparacao",
                    ordem_grupo=100 + idx_exp,
                )
                if pede_anomalias or nome in _EXPERIMENTOS_ANOMALIA:
                    _add_img(
                        imagens,
                        pasta / "anomalias_detectadas.png",
                        f"{grupo} - anomalias detectadas",
                        grupo=grupo,
                        ordem=1,
                        tipo="comparacao",
                        ordem_grupo=100 + idx_exp,
                    )

            for idx_modelo, (modelo, dados_modelo) in enumerate(modelos.items()):
                if not dados_modelo.get("disponivel", True):
                    continue
                if melhor_apenas and melhor and modelo != melhor:
                    continue
                if modelos_pedidos and modelo not in modelos_pedidos:
                    continue
                if not somente_matriz:
                    _add_grafico_modelo(
                        imagens,
                        pasta,
                        modelo,
                        dados_modelo,
                        "grafico_metricas",
                        f"{grupo} - resultado individual ({modelo})",
                        grupo,
                        100 + idx_modelo * 2,
                        100 + idx_exp,
                    )
                if pede_matriz:
                    _add_grafico_modelo(
                        imagens,
                        pasta,
                        modelo,
                        dados_modelo,
                        "grafico_matriz_confusao",
                        f"{grupo} - matriz de confusao ({modelo})",
                        grupo,
                        101 + idx_modelo * 2,
                        100 + idx_exp,
                    )
    return sorted(
        imagens,
        key=lambda img: (
            int(img.get("group_order", 0) or 0),
            int(img.get("order", 0) or 0),
            str(img.get("caption", "")),
        ),
    )


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

    meta = d.get("__meta__", {})
    melhores = {}
    for chave, res in d.items():
        if chave.startswith("__"):
            continue  # bloco de metadados (evidence_level, limiar), não é falha
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


def _resumo_experimentos(pergunta: str = "") -> str | None:
    arquivos = _arquivos_experimentos(pergunta)
    if not arquivos:
        return None

    txt = _normalizar(pergunta)
    metricas = ("accuracy", "precision", "recall", "f1", "auc", "specificity")
    linhas = [
        "## Experimentos por artigo\n\n",
        "| Experimento | Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias detectadas |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]

    linhas_modelos = []
    destaques_melhor = []
    origens = []
    for arq in arquivos:
        d = _json(arq)
        if not d:
            continue
        exp = d.get("referencia") or d.get("experimento") or arq.parent.name
        origem = d.get("origem_dados") or {}
        if origem.get("descricao"):
            origens.append((exp, origem["descricao"]))
        if _pede_melhor(txt) and d.get("melhor_modelo"):
            destaques_melhor.append(
                f"- **{exp}**: {d.get('melhor_modelo')} "
                f"({_fmt(d.get('metrica_principal'))}={_fmt(d.get('melhor_valor'))})"
            )
        modelos = d.get("modelos", {})
        modelos_pedidos = _modelos_citados(txt, modelos)
        for modelo, m in modelos.items():
            if modelos_pedidos and modelo not in modelos_pedidos:
                continue
            if not m.get("disponivel", True):
                motivo = m.get("motivo", "indisponivel")
                linhas_modelos.append({
                    "anomalias": None,
                    "exp": exp,
                    "modelo": modelo,
                    "linha": f"| {exp} | {modelo} ({motivo}) | - | - | - | - | - | - | - |\n",
                })
                continue
            valores = [_fmt(m.get(chave)) if m.get(chave) is not None else "-" for chave in metricas]
            anomalias = m.get("anomalias_detectadas", "-")
            linhas_modelos.append({
                "anomalias": anomalias if isinstance(anomalias, int) else None,
                "exp": exp,
                "modelo": modelo,
                "linha": (
                    f"| {exp} | {modelo} | {valores[0]} | {valores[1]} | {valores[2]} | "
                    f"{valores[3]} | {valores[4]} | {valores[5]} | {anomalias} |\n"
                ),
            })

    if _pede_anomalias(txt):
        linhas_modelos.sort(key=lambda item: item["anomalias"] if item["anomalias"] is not None else -1, reverse=True)
    linhas.extend(item["linha"] for item in linhas_modelos)

    if _pede_anomalias(txt):
        candidatos = [item for item in linhas_modelos if item["anomalias"] is not None]
        if candidatos:
            topo = candidatos[0]
            linhas.append(
                f"\nDestaque: quem mais marcou anomalias foi **{topo['modelo']}** "
                f"em **{topo['exp']}**, com **{topo['anomalias']}** detecções no ponto de operação.\n"
            )
    if destaques_melhor:
        linhas.append("\nMelhor modelo pelo criterio salvo:\n" + "\n".join(destaques_melhor) + "\n")
    if origens:
        vistos = set()
        linhas.append("\nOrigem dos dados usados nestes resultados:\n")
        for exp, descricao in origens:
            chave = (exp, descricao)
            if chave in vistos:
                continue
            vistos.add(chave)
            linhas.append(f"- **{exp}**: {descricao}\n")

    linhas.append(
        "\nLeitura rapida: AUC alto mede separacao por score. Para operacao real, "
        "olhe junto F1/accuracy e a coluna de anomalias detectadas; AUC ou recall "
        "alto com poucas ou zero anomalias detectadas indica que o modelo pode "
        "estar ranqueando bem, mas operando conservador demais no ponto escolhido."
    )
    return "".join(linhas)


def resumir_resultados(pergunta: str = "", *, incluir_imagens: bool = True) -> dict:
    focos = _focos(pergunta)
    if not focos:
        focos = {"autoencoder", "injecao", "validacao", "weibull", "experimentos"}

    secoes = []
    if "autoencoder" in focos:
        secoes.append(_resumo_autoencoder())
    if "injecao" in focos:
        secoes.append(_resumo_injecao())
    if "validacao" in focos:
        secoes.append(_resumo_validacao())
    if "weibull" in focos:
        secoes.append(_resumo_weibull())
    if "experimentos" in focos:
        secoes.append(_resumo_experimentos(pergunta))

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
