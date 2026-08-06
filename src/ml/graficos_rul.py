"""Graficos academicos de TTF, confiabilidade e RUL."""

from __future__ import annotations

from src.ml.rul_weibull import (
    COR_ALERTA,
    COR_TEXTO_SEC,
    FALHAS,
    Path,
    TAM,
    _log,
    curva_kaplan_meier,
    np,
    plt,
    rul_condicional,
    rul_restrita_km,
    salvar_figura,
    weibull_min,
)

def plotar_ttf_histogramas(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """TTFs observados e censurados com ajuste Weibull, quando estimável."""
    n_falhas = len(FALHAS)
    fig, axes = plt.subplots(
        1, n_falhas, figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle("TTF sintético — falhas observadas e censura à direita")

    for ax, falha in zip(axes, FALHAS):
        fid  = falha["id"]
        nome = falha["nome"]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        p    = params[fid]
        observados = ttfs[eventos]
        censurados = ttfs[~eventos]

        horizonte = float(max(ttfs))
        bins = np.linspace(0.0, horizonte, 13)
        MIN_EVENTOS_HIST = 5
        if len(observados) >= MIN_EVENTOS_HIST:
            ax.hist(
                observados, bins=bins, density=False, alpha=0.72,
                color=falha["cor"], edgecolor="white",
                label=f"Eventos observados (n={len(observados)})",
            )
        elif len(observados):
            # Pouquíssimos eventos: uma barra solitária parece defeito e engana.
            # Mostra um aviso claro de amostra insuficiente (mantendo as linhas
            # de mediana/censura) em vez de um histograma degenerado.
            ax.set_ylim(0, 1)
            ax.text(
                0.5, 0.6,
                f"amostra insuficiente\npara histograma (n={len(observados)})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=COR_TEXTO_SEC,
            )
        if len(observados):
            ax.axvline(
                float(np.median(observados)), color="0.35", linestyle="--",
                linewidth=1.3, label=f"Mediana observada={np.median(observados):.0f}",
            )

        # Curva de Weibull ajustada SOBRE o histograma (quando estimável). A
        # densidade f(t) integra 1 sobre TODAS as trajetórias; escala-se para a
        # contagem multiplicando por (n_total × largura_do_bin). n_total inclui
        # as censuradas: só assim a curva bate com o histograma dos eventos
        # observados (que é a fração de f(t) à esquerda do horizonte). Sob alta
        # censura, a maior parte da massa fica à direita da censura — a curva
        # baixa perto do histograma é o próprio sinal de que o ajuste extrapola.
        if p["fit_converged"] and len(observados):
            largura_bin = float(bins[1] - bins[0])
            t_grid = np.linspace(0.0, horizonte, 400)
            densidade = weibull_min.pdf(
                t_grid, p["beta"], loc=0, scale=p["eta"]
            )
            escala = len(ttfs) * largura_bin
            ax.plot(
                t_grid, densidade * escala, color="black", linewidth=2.2,
                label=f"Weibull ajustada (β={p['beta']:.2f}, η={p['eta']:.1f})",
            )

        if len(censurados):
            ax.axvline(
                horizonte, color=COR_ALERTA, linestyle="--", linewidth=2,
                label=f"Censura em {horizonte:.0f} (n={len(censurados)})",
            )

        if p["fit_converged"] and p["b10"] <= horizonte:
            ax.axvline(
                p["b10"], color="#2a78d6", linestyle=":", linewidth=1.7,
                label=f"B10 paramétrico={p['b10']:.1f}",
            )

        npm_str = f"NPR={falha['npr']}"
        if p["fit_converged"]:
            incerto = p["rul_parametrica_alta_incerteza"]
            # β sob censura alta é ARTEFATO (poucos eventos empilhados na borda,
            # η extrapolando além do horizonte) — não é propriedade física. Marca
            # explicitamente para não ser lido como um β confiável.
            beta_txt = f"β={p['beta']:.2f}" + (" (não confiável*)" if incerto else "")
            ajuste = (
                f"{beta_txt} · η={p['eta']:.1f} · censura={p['censura_pct']:.0f}%"
                + ("\n*censura alta → β é artefato, não vida útil"
                   if incerto else "")
            )
        else:
            ajuste = (f"Weibull não estimável · censura={p['censura_pct']:.0f}%\n"
                      "RUL restrita por Kaplan-Meier disponível")
        ax.set_title(f"{nome} ({npm_str})\n{ajuste}", fontsize=9)
        ax.set_xlabel("TTF (passos de degradação)")
        ax.set_ylabel("Número de trajetórias")
        ax.set_xlim(0, horizonte * 1.05)
        ax.legend(fontsize=8)

    arq = pasta / "weibull_ttf.png"
    salvar_figura(
        fig, arq,
        "E2 ilustrativo: trajetórias sintéticas, sem equivalência com tempo físico ou vida útil de campo.",
    )
    _log(f"   📊 {arq.name}")


def plotar_confiabilidade(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Funções de confiabilidade R(t) e taxa de falha h(t)."""
    fig, axes = plt.subplots(
        2, 3, figsize=TAM["painel_6"], layout="constrained"
    )
    fig.suptitle("Confiabilidade sintética — Kaplan-Meier e Weibull censurada")

    for col, falha in enumerate(FALHAS):
        fid  = falha["id"]
        p    = params[fid]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        t    = np.linspace(0.1, max(ttfs) * 1.2, 300)

        # Confiabilidade R(t)
        ax_r = axes[0][col]
        km_t, km_s = curva_kaplan_meier(ttfs, eventos)
        ax_r.step(km_t, km_s, where="post", color="black", linewidth=1.5,
                  label="Kaplan-Meier")
        if p["fit_converged"]:
            R = weibull_min.sf(t, p["beta"], loc=0, scale=p["eta"])
            ax_r.plot(t, R, color=falha["cor"], linewidth=2, label="Weibull")
            ax_r.fill_between(t, R, alpha=0.12, color=falha["cor"])
        ax_r.set_ylim([0, 1.05])
        ax_r.set_xlabel("t (passos)")
        ax_r.set_ylabel("R(t) = P(T > t)")
        npm_str = f"NPR={falha['npr']}"
        titulo_ajuste = (
            f"β={p['beta']:.2f}, η={p['eta']:.1f}, RMSE-KM={p['km_rmse']:.3f}"
            + ("\nALTA CENSURA — extrapolação incerta"
               if p["rul_parametrica_alta_incerteza"] else "")
            if p["fit_converged"] else "ajuste não estimável"
        )
        ax_r.set_title(f"{falha['nome']} ({npm_str})\n{titulo_ajuste}", fontsize=9)
        ax_r.legend(fontsize=8)

        # Taxa de falha h(t)
        ax_h = axes[1][col]
        if p["fit_converged"]:
            H = weibull_min.pdf(t, p["beta"], loc=0, scale=p["eta"]) / np.maximum(
                weibull_min.sf(t, p["beta"], loc=0, scale=p["eta"]), 1e-10
            )
            ax_h.plot(t, H, color=falha["cor"], linewidth=2)
            beta_desc = ("crescente ↑" if p["beta"] > 1.1
                         else "constante →" if p["beta"] > 0.9
                         else "decrescente ↓")
            ax_h.set_title(f"Taxa de falha sintética h(t)\nβ={p['beta']:.2f} — {beta_desc}", fontsize=9)
        else:
            ax_h.text(0.5, 0.5, "Sem eventos suficientes\npara estimar h(t)",
                      transform=ax_h.transAxes, ha="center", va="center",
                      color=COR_TEXTO_SEC)
            ax_h.set_title("Taxa de falha não estimável")
        ax_h.set_xlabel("t (passos)")
        ax_h.set_ylabel("h(t)")

    arq = pasta / "weibull_confiabilidade.png"
    salvar_figura(
        fig, arq,
        "Curvas em passos sintéticos; o ajuste descreve o experimento computacional, não confiabilidade de campo.",
    )
    _log(f"   📊 {arq.name}")


def plotar_rul(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """RUL restrita e paramétrica, sem ocultar componentes censurados."""
    fig, axes = plt.subplots(
        1, len(FALHAS), figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "RUL sintética por componente — estimativa restrita e extrapolação Weibull"
    )

    for ax, falha in zip(axes, FALHAS):
        fid = falha["id"]
        p = params[fid]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        horizonte = float(np.max(ttfs))
        t_pontos = np.linspace(0.0, horizonte * 0.8, 41)
        ruls_km = [
            rul_restrita_km(t, ttfs, eventos, horizonte) for t in t_pontos
        ]

        ax.plot(
            t_pontos, ruls_km, color=falha["cor"], linewidth=2.6,
            label="RUL restrita KM",
        )
        ax.fill_between(t_pontos, ruls_km, color=falha["cor"], alpha=0.12)
        ax.set_xlabel("Tempo atual (passos sintéticos)")
        ax.set_ylabel("RUL restrita ao horizonte")
        ax.set_xlim(0, horizonte * 0.8)
        ax.set_ylim(0, max(horizonte, max(ruls_km, default=0.0)) * 1.05)

        eixos_legenda = [ax]
        if p["fit_converged"]:
            ruls_param = [
                rul_condicional(t, p["beta"], p["eta"]) for t in t_pontos
            ]
            # Com alta censura, a extrapolação pode ser várias vezes maior que
            # o horizonte observado. Um eixo próprio mantém as duas estimativas
            # visíveis e impede que a escala paramétrica achate a curva KM.
            ax_param = ax.twinx() if p["rul_parametrica_alta_incerteza"] else ax
            ax_param.plot(
                t_pontos, ruls_param, color="black", linewidth=1.8,
                linestyle="--", label="RUL Weibull",
            )
            if ax_param is not ax:
                ax_param.set_ylabel("RUL Weibull extrapolada")
                ax_param.tick_params(axis="y", colors=COR_TEXTO_SEC)
                eixos_legenda.append(ax_param)

        status = (
            "Weibull com alta incerteza"
            if p["rul_parametrica_alta_incerteza"]
            else "Weibull + KM"
            if p["fit_converged"]
            else "Somente KM restrita"
        )
        ax.set_title(
            f"{falha['nome']} (NPR={falha['npr']})\n"
            f"eventos={p['n_eventos']} · censura={p['censura_pct']:.0f}% · {status}",
            fontsize=9,
        )
        handles, labels = [], []
        for eixo in eixos_legenda:
            h, l = eixo.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        ax.legend(handles, labels, fontsize=8, loc="best")

    arq = pasta / "weibull_rul.png"
    salvar_figura(
        fig, arq,
        "E2 ilustrativo. KM é restrita ao horizonte observado; Weibull extrapola e exige cautela, especialmente sob alta censura.",
    )
    _log(f"   📊 {arq.name}")
