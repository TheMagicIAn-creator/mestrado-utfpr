"""
resultados.py - Al IAdo PV
Leitura e resumo dos artefatos do pipeline de Machine Learning.

As respostas de resultados agora saem por prompt, no chat. Este modulo
centraliza os JSONs, tabelas e graficos para evitar dashboards paralelos.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.core.config import PASTA_CHROMADB, RAIZ_PROJETO
from src.core.formatacao import fmt_num
from src.core.tempo import agora_local

PASTA_AE = RAIZ_PROJETO / "resultados" / "autoencoder"
PASTA_EXPERIMENTOS = RAIZ_PROJETO / "resultados" / "experimentos"


def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(valor, casas: int = 3) -> str:
    """Formatação canônica — delega a src.core.formatacao (política única)."""
    if isinstance(valor, str):
        return valor  # rótulos passam intactos (compatibilidade)
    return fmt_num(valor, casas)


def _fmt_excedencia(info: dict | None, fallback_pct=None) -> str:
    if not isinstance(info, dict):
        return f"{_fmt(fallback_pct, 2)}%" if fallback_pct is not None else "-"
    count = info.get("count")
    n = info.get("n")
    taxa = info.get("rate_pct")
    low = info.get("ci95_low_pct")
    high = info.get("ci95_high_pct")
    if count is None or n is None or taxa is None:
        return f"{_fmt(fallback_pct, 2)}%" if fallback_pct is not None else "-"
    return (
        f"{count}/{n} = {_fmt(taxa, 2)}% "
        f"[{_fmt(low, 2)}; {_fmt(high, 2)}]"
    )


def _normalizar(texto: str) -> str:
    import unicodedata

    texto = texto.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


_EXPERIMENTOS_ALIASES = {
    "francisti": "francisti",
    "ibrahim": "ibrahim",
}
_EXPERIMENTOS_ANOMALIA = {"francisti", "ibrahim"}


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
        return ["francisti", "ibrahim"]
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


def _variante_comparacao_metricas(txt: str) -> tuple[str, str]:
    if any(t in txt for t in ("pontos", "dot plot", "dotplot", "lollipop", "dispersao")):
        return "comparacao_metricas_pontos.png", "comparacao por pontos"
    if any(t in txt for t in (
        "barras", "barra horizontal", "bar chart", "bar plot",
        "grafico de colunas", "gráfico de colunas",
    )):
        return "comparacao_metricas_barras.png", "comparacao em barras horizontais"
    return "comparacao_metricas.png", "comparacao de metricas"


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


def _pede_origem_dados(txt: str) -> bool:
    return any(t in txt for t in (
        "origem", "dados usados", "dataset local", "datasets locais",
        "recalculos", "recalculado", "recalculados", "recalculei",
        "replicacao", "replicacoes", "replicado", "replicados",
        "metodologia dos artigos", "somente os artefatos", "artefatos recalculados",
        "repositorio", "local", "proveniencia", "proveniência",
        "origin", "local dataset", "local datasets", "replication",
        "replications", "recalculated", "repository", "provenance",
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
        caminho_qualidade = RAIZ_PROJETO / "dados" / "processados" / "features_paderborn_qualidade.png"
        _add_img(imagens, caminho_qualidade, "Features - diagnostico espectral e F0", grupo="Qualidade dos dados", ordem=5, ordem_grupo=5)
        _add_img(imagens, "curva_treino.png", "Autoencoder - convergencia treino/calibracao", grupo="Autoencoder", ordem=10, ordem_grupo=10)
        _add_img(imagens, "distribuicao_erro.png", "Autoencoder - distribuicao MSE e ECDF", grupo="Autoencoder", ordem=20, ordem_grupo=10)
        _add_img(imagens, "erro_temporal.png", "Autoencoder - erro MSE temporal por split", grupo="Autoencoder", ordem=30, ordem_grupo=10)
    if "injecao" in focos:
        _add_img(imagens, "injecao_falhas_resultados.png", "Falhas sinteticas - erro por severidade", grupo="Injecao de falhas", ordem=10, tipo="comparacao", ordem_grupo=20)
        _add_img(imagens, "injecao_falhas_comparacao.png", "Falhas sinteticas - taxa de deteccao e IC95%", grupo="Injecao de falhas", ordem=20, tipo="comparacao", ordem_grupo=20)
    if "validacao" in focos:
        _add_img(imagens, "validacao_roc.png", "Validacao - curvas ROC", grupo="Validacao", ordem=10, tipo="comparacao", ordem_grupo=30)
        _add_img(imagens, "validacao_pr.png", "Validacao - curvas Precision-Recall", grupo="Validacao", ordem=15, tipo="comparacao", ordem_grupo=30)
        _add_img(imagens, "validacao_matriz.png", "Validacao - matrizes de confusao", grupo="Validacao", ordem=20, tipo="matriz", ordem_grupo=30)
        _add_img(imagens, "validacao_matrizes_severidades.png", "Validacao - matrizes por falha e severidade", grupo="Validacao", ordem=25, tipo="comparacao", ordem_grupo=30)
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
        pedido_comparativo = any(t in txt for t in (
            "comparar", "compare", "comparacao", "comparison",
            "comparacion", "comparaison", "versus", " vs ",
        ))
        pede_individuais = (not pedido_comparativo) or any(t in txt for t in (
            "resultado individual", "resultados individuais",
            "grafico de cada modelo", "gráfico de cada modelo",
            "graficos por modelo", "gráficos por modelo",
            "todos os graficos", "todos os gráficos", "graficos e matrizes",
        ))

        for idx_exp, resultado in enumerate(_arquivos_experimentos(pergunta)):
            pasta = resultado.parent
            nome = pasta.name
            dados = _json(resultado) or {}
            modelos = dados.get("modelos", {})
            melhor = dados.get("melhor_modelo")
            modelos_pedidos = _modelos_citados(txt, modelos)
            grupo = dados.get("referencia") or f"Experimento {nome}"

            if not somente_matriz:
                arquivo_comparacao, rotulo_comparacao = _variante_comparacao_metricas(txt)
                caminho_comparacao = pasta / arquivo_comparacao
                if not caminho_comparacao.exists():
                    caminho_comparacao = pasta / "comparacao_metricas.png"
                _add_img(
                    imagens,
                    caminho_comparacao,
                    f"{grupo} - {rotulo_comparacao}",
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
                if not somente_matriz and pede_individuais:
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
    metodo = d.get("score_method") or d.get("metodo_escore") or "mse"
    percentil = d.get("threshold_effective_percentile", d.get("percentil_limiar"))
    if percentil is not None:
        ponto_operacao = f"{metodo} / percentil efetivo {_fmt(percentil, 1)}"
    else:
        ponto_operacao = str(metodo)
    fp_score = (d.get("fp_score_operacional") or {}).get("teste")
    fp_mse = (d.get("fp_mse_p99") or {}).get("teste")
    alvo_fpr = d.get("threshold_target_fpr_pct")
    resolucao_fpr = d.get("threshold_sample_resolution_pct")
    politica = d.get("threshold_policy") or "percentil legado"
    nota_resolucao = ""
    if d.get("threshold_target_resolvable") is False:
        nota_resolucao = (
            " O alvo está abaixo da resolução da calibração; zero eventos "
            "observados não certifica a taxa de campo."
        )
    return (
        "## Autoencoder - modelo de normalidade\n\n"
        "| Métrica | Valor |\n"
        "|---|---:|\n"
        f"| Escore operacional | {ponto_operacao} |\n"
        f"| Política do limiar | {politica} |\n"
        f"| FPR-alvo na calibração | {_fmt(alvo_fpr, 2)}% |\n"
        f"| Resolução amostral da calibração | {_fmt(resolucao_fpr, 2)}% |\n"
        f"| Limiar operacional | {_fmt(d.get('score_threshold', d.get('limiar')), 4)} |\n"
        f"| Referência MSE p99 | {_fmt(d.get('mse_p99', d.get('limiar_p99')), 4)} |\n"
        f"| Média baseline | {_fmt(d.get('mu'), 4)} |\n"
        f"| Desvio baseline | {_fmt(d.get('sigma'), 4)} |\n"
        f"| Janelas de treino | {d.get('n_janelas_treino', '-')} |\n"
        f"| Janelas de calibração | {d.get('n_janelas_calibracao', '-')} |\n"
        f"| Janelas de teste | {d.get('n_janelas_teste', '-')} |\n"
        f"| FP teste - escore operacional | {_fmt_excedencia(fp_score, d.get('fp_test_pct', d.get('fp_val_pct')))} |\n"
        f"| FP teste - referência MSE p99 | {_fmt_excedencia(fp_mse)} |\n"
        f"| Épocas treinadas | {d.get('epochs_treinadas', '-')} |\n\n"
        "Leitura rápida: o detector usa o escore operacional registrado em "
        "`limiar.json`; os gráficos principais de reconstrução permanecem na "
        "escala MSE e são acompanhados por `calibracao_autoencoder.md`."
        + nota_resolucao
    )


def _resumo_injecao() -> str | None:
    d = _json(PASTA_AE / "injecao_falhas_report.json")
    if not d:
        return None

    linhas = [
        "## Injeção de falhas sintéticas\n",
        f"Limiar: **{_fmt(d.get('limiar'), 4)}**. "
        f"Baseline: **{_fmt(d.get('baseline_mean'), 4)} ± {_fmt(d.get('baseline_std'), 4)}**.\n",
        "| Falha | NPR | SMD95 | Taxa (IC95%) | n | Erro mediano |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    nao_detectadas = []
    for fid, falha in d.get("falhas", {}).items():
        smd = d.get("smd", {}).get(fid)
        erro = taxa_txt = n_txt = "-"
        smd_txt = "⚠️ alvo não atingido"
        if smd is not None:
            res = falha.get("resultados", {}).get(str(smd), {})
            erro = _fmt(res.get("erro_mediano", res.get("erro")), 4)
            taxa_txt = (
                f"{_fmt(res.get('taxa_deteccao'), 3)} "
                f"[{_fmt(res.get('taxa_ci_low'), 3)}; {_fmt(res.get('taxa_ci_high'), 3)}]"
            )
            n_txt = str(res.get("n", "-"))
            smd_txt = str(smd)
        else:
            resultados_sev = falha.get("resultados", {})
            if resultados_sev:
                sev_max = max(resultados_sev, key=lambda s: float(s))
                pico = resultados_sev[sev_max]
                nao_detectadas.append(
                    f"- **{falha.get('nome', fid)}**: taxa máxima de detecção "
                    f"{_fmt(pico.get('taxa_deteccao'), 3)} na severidade {sev_max}; "
                    "o alvo probabilístico de 95% não foi atingido."
                )
        linhas.append(
            f"| {falha.get('nome', fid)} | {falha.get('npr') or '-'} | "
            f"{smd_txt} | {taxa_txt} | {n_txt} | {erro} |\n"
        )
    linhas.append(
        "\nLeitura rápida: SMD95 é a menor severidade cuja taxa pontual de detecção "
        "atinge 95%; o intervalo de Wilson mostra a incerteza dessa estimativa."
    )
    if nao_detectadas:
        linhas.append(
            "\n\n⚠️ **Falha(s) sem SMD nesta execução** (achado relevante, "
            "não omitir na dissertação):\n" + "\n".join(nao_detectadas)
        )
    return "".join(linhas)


def _nome_falha(chave: str) -> str:
    base = re.sub(r"_sev.*$", "", chave)
    nomes = {
        "contator_ac": "Contator AC",
        "igbt": "IGBT",
        "fusivel_ac": "Fusível AC",
    }
    return nomes.get(base, base)


def _sev(chave: str) -> str:
    match = re.search(r"sev([0-9.]+)", chave)
    return match.group(1) if match else "-"


def _resumo_validacao() -> str | None:
    d = _json(PASTA_AE / "validacao_report.json")
    if not d:
        return None

    itens = []
    ordem_falhas = {"Contator AC": 0, "IGBT": 1, "Fusível AC": 2}
    for chave, res in d.items():
        if chave.startswith("__"):
            continue
        item = dict(res)
        item["chave"] = chave
        item["falha"] = _nome_falha(chave)
        itens.append(item)
    itens.sort(key=lambda item: (
        ordem_falhas.get(item["falha"], 99), float(_sev(item["chave"])),
    ))

    linhas = [
        "## Validação sintética interna E2\n\n",
        "| Falha | Sev. | AUC-ROC (IC95%) | Recall (IC95%) | FNR | Especificidade | n/classe |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    cego = []
    for res in itens:
        falha = res["falha"]
        rec = res.get("recall")
        if isinstance(rec, (int, float)) and rec < 0.1:
            cego.append(f"{falha} (sev. {_sev(res['chave'])})")
        linhas.append(
            f"| {falha} | {_sev(res['chave'])} | {_fmt(res.get('auc_roc'))} "
            f"[{_fmt(res.get('auc_roc_ci_low'))}; {_fmt(res.get('auc_roc_ci_high'))}] | "
            f"{_fmt(rec)} [{_fmt(res.get('recall_ci_low'))}; {_fmt(res.get('recall_ci_high'))}] | "
            f"{_fmt(res.get('fnr'))} | {_fmt(res.get('specificity'))} | "
            f"{res.get('n_pos', '-')} |\n"
        )
    leitura = [
        "\n**Leitura honesta:** a AUC mede a separação por *ranking* (independe "
        "do limiar). No PONTO DE OPERAÇÃO (limiar operacional congelado), o recall pode "
        "ser bem menor que a AUC sugere."
    ]
    if cego:
        leitura.append(
            f" Atenção ao baixo recall em **{', '.join(cego)}**: o limiar "
            "conservador perde a maior parte dessas falhas."
        )
    leitura.append(
        " As linhas mostram todas as severidades, sem escolher apenas a melhor "
        "AUC. O bloco de teste é temporalmente isolado e sem sobreposição, mas a "
        "falha continua sintética: não é desempenho industrial."
    )
    linhas.append("".join(leitura))
    return "".join(linhas)


def _resumo_weibull() -> str | None:
    d = _json(PASTA_AE / "weibull_results.json")
    if not d:
        return None

    linhas = [
        "## RUL / Weibull\n\n",
        "| Falha | NPR | Eventos/Censura | beta (IC95%) | eta (IC95%) | MTTF (IC95%) | B10 (IC95%) | RUL restrita inicial | Status |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for fid, falha in d.get("falhas", {}).items():
        p = falha.get("weibull", {})
        beta = p.get("beta")
        def valor_ci(nome: str, casas: int = 1) -> str:
            valor = p.get(nome)
            ci = p.get(f"{nome}_ci95") or [None, None]
            return f"{_fmt(valor, casas)} [{_fmt(ci[0], casas)}; {_fmt(ci[1], casas)}]"

        status_mapa = {
            "exploratorio_descritivo": "exploratório",
            "exploratorio_alta_censura": "Weibull incerta; KM restrita disponível",
            "nao_estimavel": "não estimável",
            "nao_estimavel_parametrico_rul_restrita": (
                "Weibull não estimável; KM restrita disponível"
            ),
        }
        status = status_mapa.get(
            falha.get("status_ajuste"),
            "exploratório" if p.get("fit_converged") else "não estimável",
        )
        linhas.append(
            f"| {falha.get('nome', fid)} | {falha.get('npr')} | "
            f"{p.get('n_eventos', '-')}/{p.get('n_censurados', '-')} | "
            f"{valor_ci('beta', 2)} | {valor_ci('eta')} | {valor_ci('mttf')} | "
            f"{valor_ci('b10')} | {_fmt(p.get('rul_restrita_inicial'))} | {status} |\n"
        )
    linhas.append(
        "\n**Separação obrigatória das estimativas:** a coluna **RUL restrita "
        "inicial** é exclusivamente a média residual **não paramétrica de "
        "Kaplan-Meier**, truncada no horizonte observado. Ela nunca deve ser "
        "descrita como RUL Weibull. A curva Weibull do gráfico é a estimativa "
        "paramétrica/extrapolativa e só existe quando o ajuste convergiu.\n\n"
        "**Leitura obrigatória:** a censura agora é preservada e os intervalos "
        "vêm de bootstrap, mas os tempos continuam sendo passos de degradação "
        "sintética E2. A RUL por Kaplan-Meier é restrita ao horizonte observado; "
        "a RUL Weibull é extrapolativa e recebe ressalva quando há alta censura. "
        "MTTF, B10 e RUL descrevem o experimento computacional e "
        "não podem ser apresentados como vida útil física ou de campo. O NPR "
        "prioriza risco na FMECA; ele **não determina** quantos eventos o "
        "experimento sintético produzirá e não explica causalmente a censura."
    )
    return "".join(linhas)


def _resumo_experimentos(pergunta: str = "") -> str | None:
    arquivos = _arquivos_experimentos(pergunta)
    if not arquivos:
        return None

    txt = _normalizar(pergunta)
    metricas = ("accuracy", "precision", "recall", "f1", "auc", "specificity")
    linhas = ["## Experimentos por artigo\n\n"]

    linhas_modelos = []
    destaques_melhor = []
    origens = []
    evidencias = []
    for arq in arquivos:
        d = _json(arq)
        if not d:
            continue
        exp = d.get("referencia") or d.get("experimento") or arq.parent.name
        origem = d.get("origem_dados") or {}
        if origem.get("descricao"):
            origens.append((exp, origem["descricao"]))
        if d.get("evidence_level") or d.get("evidence_note"):
            evidencias.append((exp, d.get("evidence_level"), d.get("evidence_note")))
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
                    "dados": m,
                    "linha": f"| {exp} | {modelo} ({motivo}) | - | - | - | - | - | - | - |\n",
                })
                continue
            valores = [_fmt(m.get(chave)) if m.get(chave) is not None else "-" for chave in metricas]
            anomalias = m.get("anomalias_detectadas", "-")
            linhas_modelos.append({
                "anomalias": anomalias if isinstance(anomalias, int) else None,
                "exp": exp,
                "modelo": modelo,
                "dados": m,
                "linha": (
                    f"| {exp} | {modelo} | {valores[0]} | {valores[1]} | {valores[2]} | "
                    f"{valores[3]} | {valores[4]} | {valores[5]} | {anomalias} |\n"
                ),
            })

    metrica_pedida = next(
        (
            met
            for met in ("auc", "f1", "recall", "precision", "accuracy", "specificity")
            if re.search(rf"\b{re.escape(met)}\b", txt)
        ),
        None,
    )
    if _pede_anomalias(txt) and not metrica_pedida:
        linhas_modelos.sort(
            key=lambda item: item["anomalias"] if item["anomalias"] is not None else -1,
            reverse=True,
        )
    if _pede_anomalias(txt) and metrica_pedida:
        ordenados = sorted(
            linhas_modelos,
            key=lambda item: (
                item["dados"].get(metrica_pedida)
                if isinstance(item["dados"].get(metrica_pedida), (int, float))
                else -1
            ),
            reverse=True,
        )
        linhas.extend([
            f"| Rank | Experimento | Modelo | {metrica_pedida.upper()} | Detectadas | Reais | Taxa marcada | Recall |\n",
            "|---:|---|---|---:|---:|---:|---:|---:|\n",
        ])
        for rank, item in enumerate(ordenados, 1):
            dados_modelo = item["dados"]
            linhas.append(
                f"| {rank} | {item['exp']} | {item['modelo']} | "
                f"{_fmt(dados_modelo.get(metrica_pedida))} | "
                f"{_fmt(dados_modelo.get('anomalias_detectadas'), 0)} | "
                f"{_fmt(dados_modelo.get('anomalias_reais'), 0)} | "
                f"{_fmt(dados_modelo.get('taxa_anomalias_detectadas'))} | "
                f"{_fmt(dados_modelo.get('recall'))} |\n"
            )
    elif _pede_anomalias(txt):
        linhas.extend([
            "| Experimento | Modelo | Detectadas | Reais | Taxa marcada | Recall |\n",
            "|---|---|---:|---:|---:|---:|\n",
        ])
        for item in linhas_modelos:
            dados_modelo = item["dados"]
            linhas.append(
                f"| {item['exp']} | {item['modelo']} | "
                f"{_fmt(dados_modelo.get('anomalias_detectadas'), 0)} | "
                f"{_fmt(dados_modelo.get('anomalias_reais'), 0)} | "
                f"{_fmt(dados_modelo.get('taxa_anomalias_detectadas'))} | "
                f"{_fmt(dados_modelo.get('recall'))} |\n"
            )
    elif metrica_pedida:
        ordenados = sorted(
            linhas_modelos,
            key=lambda item: (
                item["dados"].get(metrica_pedida)
                if isinstance(item["dados"].get(metrica_pedida), (int, float))
                else -1
            ),
            reverse=True,
        )
        linhas.extend([
            f"| Rank | Experimento | Modelo | {metrica_pedida.upper()} |\n",
            "|---:|---|---|---:|\n",
        ])
        for rank, item in enumerate(ordenados, 1):
            linhas.append(
                f"| {rank} | {item['exp']} | {item['modelo']} | "
                f"{_fmt(item['dados'].get(metrica_pedida))} |\n"
            )
    else:
        linhas.extend([
            "| Experimento | Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias detectadas |\n",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|\n",
        ])
        linhas.extend(item["linha"] for item in linhas_modelos)

    if _pede_anomalias(txt):
        candidatos = [item for item in linhas_modelos if item["anomalias"] is not None]
        if candidatos:
            topo = max(candidatos, key=lambda item: item["anomalias"])
            linhas.append(
                f"\nDestaque: quem mais marcou anomalias foi **{topo['modelo']}** "
                f"em **{topo['exp']}**, com **{topo['anomalias']}** detecções no ponto de operação.\n"
            )
            linhas.append(
                "Essa contagem depende do limiar e não define, sozinha, o melhor "
                "modelo: avalie junto AUC, recall, falsos positivos e o protocolo.\n"
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

    if _pede_origem_dados(txt):
        linhas.append("\nSeparacao entre artigo e recalculo local:\n")
        linhas.append(
            "- **Metodologia dos artigos**: define quais familias de modelos entram "
            "no benchmark (por exemplo, Isolation Forest, AE-LSTM, SVM, "
            "RNN/CNN ou hibrido). Isso e inspiracao metodologica, nao copia de "
            "metricas publicadas.\n"
        )
        linhas.append(
            "- **Recalculado no repositorio**: metricas, matrizes de confusao, "
            "graficos e contagens de anomalias sao gerados a partir dos artefatos "
            "locais em `resultados/experimentos/<autor>/resultado.json`.\n"
        )
        linhas.append(
            "- **Dados locais**: Francisti e Ibrahim usam features "
            "locais do Paderborn extraidas de "
            "`dados/brutos/Inverter_Data_Set.csv`; como esse dataset e saudavel, "
            "o ground truth de anomalia vem de falhas sinteticas do pipeline.\n"
        )
        linhas.append(
            "- **Nao e validacao industrial**: esses experimentos sao E1 "
            "(benchmark exploratorio). Eles ajudam a comparar abordagens, mas "
            "nao substituem validacao externa em bancada/campo.\n"
        )
    elif evidencias:
        vistos_ev = set()
        linhas.append("\nNivel de evidencia:\n")
        for exp, nivel, nota in evidencias:
            chave = (exp, nivel, nota)
            if chave in vistos_ev:
                continue
            vistos_ev.add(chave)
            linhas.append(f"- **{exp}**: {nivel or '-'} - {nota or 'sem nota'}\n")

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

    # Gráficos sempre ficam disponíveis em antevisão sob demanda + download,
    # mas só são renderizados inline quando o pedido exige vê-los no chat.
    imagens = imagens_relevantes(pergunta) if incluir_imagens else []
    mostrar_inline = _quer_imagens(pergunta)
    for img in imagens:
        img["inline"] = mostrar_inline

    mensagem = (
        "Aqui está o que já existe nos artefatos do pipeline.\n\n"
        + "\n\n".join(secoes)
    )
    if imagens:
        if mostrar_inline:
            mensagem += "\n\nGráficos relevantes logo abaixo."
        else:
            mensagem += (
                f"\n\n{len(imagens)} gráfico(s) disponível(is) logo abaixo. "
                "Use **Visualizar** para abrir uma antevisão responsiva sem baixar, "
                'ou peça "mostre os gráficos" para inseri-los na conversa.'
            )

    return {
        "ok": True,
        "etapa": "Consulta de resultados",
        "mensagem": mensagem,
        "imagens": imagens,
        "resposta_pronta": False,
    }


def indexar_resultados_ml(modelo_embeddings) -> str:
    """Gera resumo dos resultados e indexa na memoria do agente."""
    from src.conhecimento.indexador import indexar_sessao

    saida = RAIZ_PROJETO / "notas" / "memorias" / "resultados-fase5-ml.md"
    resumo = resumir_resultados("", incluir_imagens=False)["mensagem"]
    conteudo = (
        "# Resultados da Fase 5 - Pipeline de ML\n\n"
        f"> Gerado em {agora_local().strftime('%d/%m/%Y %H:%M %Z')}\n\n"
        f"{resumo}\n"
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(conteudo, encoding="utf-8")

    try:
        indexar_sessao(saida, modelo_embeddings, PASTA_CHROMADB)
        return "Resultados indexados. O agente ja pode discuti-los no chat."
    except Exception as exc:
        return f"Resumo salvo, mas houve erro ao indexar: {exc}"
