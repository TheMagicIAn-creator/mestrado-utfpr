"""Graficos e comparacoes visuais dos experimentos por artigo."""

from __future__ import annotations

from src.ml.experimentos_artigos import (
    ExperimentoArtigo,
    METRICAS_GRAFICO,
    Path,
)

def _slug_modelo(nome: str) -> str:
    """Nome estavel para arquivos de artefatos por modelo."""
    import re
    import unicodedata

    texto = unicodedata.normalize("NFD", nome.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    return texto or "modelo"


def _registrar_grafico_modelo(modelo: dict, chave: str, caminho: Path) -> None:
    from src.core.utils import to_project_relative_path

    modelo.setdefault("graficos", [])
    rel = to_project_relative_path(caminho)  # relativo ao projeto (portável)
    if rel not in modelo["graficos"]:
        modelo["graficos"].append(rel)
    modelo[chave] = rel


def _grafico_metricas_modelo(exp: ExperimentoArtigo, nome: str, modelo: dict, plt, np) -> Path | None:
    metricas = [
        met for met in METRICAS_GRAFICO
        if isinstance(modelo.get(met), (int, float))
    ]
    if not metricas:
        return None

    from src.ml.estilo_graficos import COR_METODO, COR_NEUTRA, TAM

    valores = [float(modelo[met]) for met in metricas]
    fig, ax = plt.subplots(figsize=TAM["unico"])
    y = np.arange(len(metricas))
    cor = COR_METODO if "autoencoder" in nome.lower() else COR_NEUTRA
    barras = ax.barh(y, valores, color=cor, height=0.58)
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels([met.replace("_", " ").upper() for met in metricas])
    ax.invert_yaxis()
    ax.set_xlabel("Métrica (0–1)")
    ax.set_title(f"{exp.referencia} - {nome}")
    ax.grid(axis="x", alpha=0.25)
    for barra, valor in zip(barras, valores):
        ax.text(
            min(1.03, valor + 0.015),
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    linhas = []
    if isinstance(modelo.get("anomalias_detectadas"), int):
        linhas.append(f"Anomalias detectadas: {modelo['anomalias_detectadas']}")
    if isinstance(modelo.get("anomalias_reais"), int):
        linhas.append(f"Anomalias reais: {modelo['anomalias_reais']}")
    if modelo.get("ponto_operacao"):
        rotulos_ponto = {
            "limiar_otimo_score": "limiar otimizado",
            "decisao_nativa_modelo": "decisao nativa",
        }
        linhas.append(f"Ponto: {rotulos_ponto.get(modelo['ponto_operacao'], modelo['ponto_operacao'])}")
    if linhas:
        fig.text(
            0.02,
            0.03,
            "\n".join(linhas),
            ha="left",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.15 if linhas else 0, 1, 1))
    caminho = exp.pasta() / f"modelo_{_slug_modelo(nome)}_metricas.png"
    fig.savefig(caminho)
    plt.close(fig)
    _registrar_grafico_modelo(modelo, "grafico_metricas", caminho)
    return caminho


def _grafico_matriz_modelo(exp: ExperimentoArtigo, nome: str, modelo: dict, plt, np) -> Path | None:
    if not modelo.get("matriz_confusao"):
        return None

    cm = np.asarray(modelo["matriz_confusao"], dtype=int)
    if cm.ndim != 2 or cm.size == 0:
        return None

    from src.ml.estilo_graficos import tam_matriz

    labels = modelo.get("classes") or [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=tam_matriz(len(labels)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matriz de confusao - {nome}")
    ax.set_xlabel("predito")
    ax.set_ylabel("real")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    limite = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cor = "white" if cm[i, j] > limite else "#111111"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=cor)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    caminho = exp.pasta() / f"modelo_{_slug_modelo(nome)}_matriz_confusao.png"
    fig.savefig(caminho)
    plt.close(fig)
    _registrar_grafico_modelo(modelo, "grafico_matriz_confusao", caminho)
    return caminho


def _grafico_comparacao(exp: ExperimentoArtigo, resultado: dict) -> list[Path]:
    """Gera PNGs comparativos e artefatos individuais por modelo."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from src.ml.estilo_graficos import (
            COR_METODO,
            COR_NEUTRA,
            PALETA,
            TAM,
            aplicar_estilo,
            tam_barras_h,
        )

        aplicar_estilo()
    except Exception:
        return []

    exp.pasta().mkdir(parents=True, exist_ok=True)
    graficos: list[Path] = []
    modelos = [
        (nome, m)
        for nome, m in resultado.get("modelos", {}).items()
        if m.get("disponivel", True)
    ]

    for nome, modelo in modelos:
        graf = _grafico_metricas_modelo(exp, nome, modelo, plt, np)
        if graf:
            graficos.append(graf)
        graf = _grafico_matriz_modelo(exp, nome, modelo, plt, np)
        if graf:
            graficos.append(graf)

    metricas = [
        met for met in METRICAS_GRAFICO
        if any(isinstance(m.get(met), (int, float)) for _, m in modelos)
    ]
    if modelos and metricas:
        nomes = [n for n, _ in modelos]
        matriz = np.asarray([
            [
                float(m.get(met)) if isinstance(m.get(met), (int, float)) else np.nan
                for met in metricas
            ]
            for _, m in modelos
        ])

        # Visão densa padrão: valores comparáveis na mesma escala, sem colunas
        # estreitas nem legendas que disputem espaço com os modelos.
        fig, ax = plt.subplots(figsize=(max(9.0, 1.25 * len(metricas)), max(4.5, 0.75 * len(nomes))))
        im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metricas)))
        ax.set_xticklabels([met.replace("_", " ").upper() for met in metricas])
        ax.set_yticks(range(len(nomes)))
        ax.set_yticklabels(nomes)
        ax.grid(False)
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                valor = matriz[i, j]
                if np.isnan(valor):
                    rotulo, cor_texto = "–", "#52514e"
                else:
                    rotulo = f"{valor:.3f}"
                    cor_texto = "white" if valor >= 0.58 else "#0b0b0b"
                ax.text(j, i, rotulo, ha="center", va="center", color=cor_texto, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Métrica (0–1)")
        ax.set_title(f"{exp.referencia} - comparação multimétrica")
        fig.tight_layout()
        caminho = exp.pasta() / "comparacao_metricas.png"
        fig.savefig(caminho)
        plt.close(fig)
        graficos.append(caminho)

        # Alternativa 1: dot plots em pequenos múltiplos. Boa para perceber
        # diferenças pequenas sem transformar cada métrica em uma cor.
        ncols = min(3, len(metricas))
        nrows = int(np.ceil(len(metricas) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=TAM["painel_6"], squeeze=False)
        y = np.arange(len(nomes))
        cores_modelos = [
            COR_METODO if "autoencoder" in nome.lower() else PALETA[i % len(PALETA)]
            for i, nome in enumerate(nomes)
        ]
        for indice, met in enumerate(metricas):
            ax = axes.flat[indice]
            valores = matriz[:, indice]
            ax.hlines(y, 0, valores, color="#d5d4cd", linewidth=1.2)
            ax.scatter(valores, y, c=cores_modelos, s=44, zorder=3)
            ax.set_xlim(0, 1.03)
            ax.set_title(met.replace("_", " ").upper(), fontsize=10)
            ax.set_yticks(y)
            ax.set_yticklabels(nomes if indice % ncols == 0 else [])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
        for ax in axes.flat[len(metricas):]:
            ax.axis("off")
        fig.suptitle(f"{exp.referencia} - comparação por pontos", y=1.01)
        fig.tight_layout()
        caminho = exp.pasta() / "comparacao_metricas_pontos.png"
        fig.savefig(caminho)
        plt.close(fig)
        graficos.append(caminho)

        # Alternativa 2: barras horizontais em pequenos múltiplos, usada
        # somente quando o prompt pedir barras explicitamente.
        fig, axes = plt.subplots(nrows, ncols, figsize=TAM["painel_6"], squeeze=False)
        for indice, met in enumerate(metricas):
            ax = axes.flat[indice]
            valores = matriz[:, indice]
            ax.barh(y, valores, color=cores_modelos, height=0.55)
            ax.set_xlim(0, 1.03)
            ax.set_title(met.replace("_", " ").upper(), fontsize=10)
            ax.set_yticks(y)
            ax.set_yticklabels(nomes if indice % ncols == 0 else [])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
        for ax in axes.flat[len(metricas):]:
            ax.axis("off")
        fig.suptitle(f"{exp.referencia} - comparação em barras horizontais", y=1.01)
        fig.tight_layout()
        caminho = exp.pasta() / "comparacao_metricas_barras.png"
        fig.savefig(caminho)
        plt.close(fig)
        graficos.append(caminho)

    itens_anomalia = [
        (nome, int(m["anomalias_detectadas"]))
        for nome, m in modelos
        if isinstance(m.get("anomalias_detectadas"), int)
    ]
    if itens_anomalia:
        nomes = [n for n, _ in itens_anomalia]
        valores = [v for _, v in itens_anomalia]
        y = np.arange(len(nomes))
        reais = [
            int(m["anomalias_reais"])
            for _, m in modelos
            if isinstance(m.get("anomalias_reais"), int)
        ]
        referencia_real = reais[0] if reais and len(set(reais)) == 1 else None
        maior = max(valores + [1])
        if referencia_real is not None:
            fig, (ax, ax_taxa) = plt.subplots(1, 2, figsize=TAM["painel_2"])
        else:
            fig, ax = plt.subplots(figsize=tam_barras_h(len(nomes)))
            ax_taxa = None
        barras = ax.barh(y, valores, color=PALETA[:len(nomes)], height=0.58)
        ax.set_yticks(y)
        ax.set_yticklabels(nomes)
        ax.invert_yaxis()
        ax.set_xlim(0, maior * 1.18)
        ax.set_xlabel("Anomalias marcadas no ponto de operação")
        ax.set_title(f"{exp.referencia} - detecções por modelo")
        for barra, v in zip(barras, valores):
            ax.text(
                v + maior * 0.012,
                barra.get_y() + barra.get_height() / 2,
                str(v),
                ha="left",
                va="center",
                fontsize=9,
            )
        ax.grid(axis="x", alpha=0.25)
        if ax_taxa is not None and referencia_real:
            taxas = [100.0 * valor / referencia_real for valor in valores]
            barras_taxa = ax_taxa.barh(
                y, taxas, color=PALETA[:len(nomes)], height=0.58
            )
            ax_taxa.set_yticks(y)
            ax_taxa.set_yticklabels([])
            ax_taxa.invert_yaxis()
            ax_taxa.set_xlim(0, max(100.0, max(taxas) * 1.15))
            ax_taxa.axvline(100, color="#c43d3d", linestyle="--", linewidth=1.5)
            ax_taxa.set_xlabel("Cobertura das anomalias reais (%)")
            ax_taxa.set_title(f"Referência: {referencia_real} anomalias reais")
            for barra, taxa in zip(barras_taxa, taxas):
                ax_taxa.text(
                    taxa + 1.0,
                    barra.get_y() + barra.get_height() / 2,
                    f"{taxa:.1f}%",
                    ha="left",
                    va="center",
                    fontsize=9,
                )
            ax_taxa.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        caminho = exp.pasta() / "anomalias_detectadas.png"
        fig.savefig(caminho)
        plt.close(fig)
        graficos.append(caminho)

    melhor = resultado.get("melhor_modelo")
    modelo_cm = resultado.get("modelos", {}).get(melhor, {})
    if not modelo_cm.get("matriz_confusao"):
        for _, m in modelos:
            if m.get("matriz_confusao"):
                modelo_cm = m
                break
    if modelo_cm.get("matriz_confusao"):
        caminho_individual = modelo_cm.get("grafico_matriz_confusao")
        if caminho_individual and Path(caminho_individual).exists():
            caminho = exp.pasta() / "matriz_confusao.png"
            import shutil
            shutil.copyfile(caminho_individual, caminho)
            graficos.append(caminho)

    return graficos
