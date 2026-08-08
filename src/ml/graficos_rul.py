"""Graficos academicos de a_det, confiabilidade e RUL.

O eixo NAO e tempo: e a magnitude da assinatura injetada em que a deteccao se
confirma, em [0; 1]. Ver o bloco "O EIXO NAO E TEMPO" em src/ml/rul_weibull.py.
"""

from __future__ import annotations

from src.ml.confiabilidade import (
    acumulada,
    confiabilidade,
    densidade,
    eixos_papel_weibull,
    mediana_de_posto,
    taxa_falha,
)
from src.ml.rul_weibull import (
    COR_ALERTA,
    COR_TEXTO_SEC,
    FALHAS,
    TAM,
    Path,
    _log,
    curva_kaplan_meier,
    np,
    plt,
    rul_condicional,
    rul_restrita_km,
    salvar_figura,
)


def plotar_ttf_histogramas(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """a_det detectados e indetectáveis no teto, com ajuste Weibull."""
    n_falhas = len(FALHAS)
    fig, axes = plt.subplots(
        1, n_falhas, figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle("a_det — magnitude de detecção, e indetectabilidade no teto")

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
            # `t_grid` começa acima de zero: f(0) diverge quando β < 1, e
            # plotar o infinito não informa nada.
            t_grid = np.linspace(max(horizonte / 400.0, 1e-6), horizonte, 400)
            f_ajustada = densidade(t_grid, p["beta"], p["eta"])
            escala = len(ttfs) * largura_bin
            ax.plot(
                t_grid, f_ajustada * escala, color="black", linewidth=2.2,
                label=f"Weibull ajustada (β={p['beta']:.2f}, η={p['eta']:.3f})",
            )

        if len(censurados):
            ax.axvline(
                horizonte, color=COR_ALERTA, linestyle="--", linewidth=2,
                label=f"Não detectadas em $a$={horizonte:.2f} (n={len(censurados)})",
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
                f"{beta_txt} · η={p['eta']:.3f} · indetect.={p['censura_pct']:.0f}%"
                + ("\n*indetectabilidade alta → β é artefato da borda"
                   if incerto else "")
            )
        else:
            ajuste = (f"Weibull não estimável · indetect.={p['censura_pct']:.0f}%\n"
                      "RUL restrita por Kaplan-Meier disponível")
        ax.set_title(f"{nome} ({npm_str})\n{ajuste}", fontsize=9)
        ax.set_xlabel("$a_{det}$ (fração da assinatura nominal)")
        ax.set_ylabel("Número de trajetórias")
        ax.set_xlim(0, horizonte * 1.05)
        ax.legend(fontsize=8)

    arq = pasta / "weibull_ttf.png"
    salvar_figura(
        fig, arq,
        "E2 ilustrativo: eixo em magnitude de assinatura, sem equivalência com tempo físico ou vida útil de campo.",
    )
    _log(f"   📊 {arq.name}")


def plotar_confiabilidade(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Funções de confiabilidade R(t) e taxa de falha h(t)."""
    fig, axes = plt.subplots(
        2, 3, figsize=TAM["painel_6"], layout="constrained"
    )
    fig.suptitle("$R(a)$ — probabilidade de a falha AINDA não ter sido detectada")

    for col, falha in enumerate(FALHAS):
        fid  = falha["id"]
        p    = params[fid]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        # O piso acompanha a escala: era 0,1 fixo, o que num eixo que agora vai de
        # 0 a 1 apagaria os primeiros 10% da curva.
        t    = np.linspace(max(ttfs) / 300.0, max(ttfs) * 1.2, 300)

        # Confiabilidade R(t)
        ax_r = axes[0][col]
        km_t, km_s = curva_kaplan_meier(ttfs, eventos)
        ax_r.step(km_t, km_s, where="post", color="black", linewidth=1.5,
                  label="Kaplan-Meier")
        if p["fit_converged"]:
            # Fonte única: src/ml/confiabilidade.py. Antes era weibull_min
            # inline aqui, e por isso o valor nunca saía do PNG.
            R = confiabilidade(t, p["beta"], p["eta"])
            ax_r.plot(t, R, color=falha["cor"], linewidth=2, label="Weibull")
            ax_r.fill_between(t, R, alpha=0.12, color=falha["cor"])
        ax_r.set_ylim([0, 1.05])
        ax_r.set_xlabel("$a$ (fração da assinatura nominal)")
        ax_r.set_ylabel("$R(a) = P(a_{det} > a)$ — ainda não detectada")
        npm_str = f"NPR={falha['npr']}"
        titulo_ajuste = (
            f"β={p['beta']:.2f}, η={p['eta']:.3f}, RMSE-KM={p['km_rmse']:.3f}"
            + ("\nALTA CENSURA — extrapolação incerta"
               if p["rul_parametrica_alta_incerteza"] else "")
            if p["fit_converged"] else "ajuste não estimável"
        )
        ax_r.set_title(f"{falha['nome']} ({npm_str})\n{titulo_ajuste}", fontsize=9)
        ax_r.legend(fontsize=8)

        # Taxa de falha h(t)
        ax_h = axes[1][col]
        if p["fit_converged"]:
            H = taxa_falha(t, p["beta"], p["eta"])
            ax_h.plot(t, H, color=falha["cor"], linewidth=2)
            beta_desc = ("crescente ↑" if p["beta"] > 1.1
                         else "constante →" if p["beta"] > 0.9
                         else "decrescente ↓")
            ax_h.set_title(f"Taxa de detecção h(a)\nβ={p['beta']:.2f} — {beta_desc}", fontsize=9)
        else:
            ax_h.text(0.5, 0.5, "Sem detecções suficientes\npara estimar h(a)",
                      transform=ax_h.transAxes, ha="center", va="center",
                      color=COR_TEXTO_SEC)
            ax_h.set_title("Taxa de falha não estimável")
        ax_h.set_xlabel("$a$ (fração da assinatura nominal)")
        ax_h.set_ylabel("$h(a)$")

    arq = pasta / "weibull_confiabilidade.png"
    salvar_figura(
        fig, arq,
        "Eixo em fração da assinatura nominal (a_det), NAO em tempo; descreve o experimento computacional, não confiabilidade de campo.",
    )
    _log(f"   📊 {arq.name}")


def plotar_rul(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Margem de magnitude até detectar, restrita (KM) e paramétrica."""
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
        ax.set_xlabel("$a$ já atingida sem detecção")
        ax.set_ylabel("margem de magnitude até detectar")
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
            f"detectadas={p['n_eventos']} · indetect. no teto={p['censura_pct']:.0f}% · {status}",
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


def plotar_distribuicao_weibull(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """f(t), F(t) e o papel de Weibull — as figuras que faltavam.

    O pesquisador apontou não ter visto "distribuição de Weibull". As curvas
    R(t) e h(t) existiam (`plotar_confiabilidade`), mas a **densidade** e a
    **acumulada** não eram desenhadas em lugar nenhum, e o **papel de Weibull**
    — o gráfico canônico da área — também não.

    O papel de Weibull é o que mais informa sobre a qualidade do ajuste: na
    escala `ln t × ln(−ln(1−F))` a distribuição vira RETA de inclinação β.
    Desvio sistemático da reta é evidência de que a família não serve — algo
    que o RMSE contra Kaplan-Meier, sozinho, não revela.
    """
    fig, axes = plt.subplots(3, len(FALHAS), figsize=TAM["painel_9"],
                             layout="constrained")
    fig.suptitle("Distribuição de Weibull ajustada — densidade, acumulada e "
                 "papel de Weibull")

    for col, falha in enumerate(FALHAS):
        fid = falha["id"]
        p = params[fid]
        ttfs = np.asarray(ttfs_dict[fid], dtype=float)
        eventos = np.asarray(eventos_dict[fid], dtype=bool)
        cor = falha["cor"]

        ax_f, ax_F, ax_pw = axes[0][col], axes[1][col], axes[2][col]
        ax_f.set_title(f"{falha['nome']} (NPR={falha['npr']})", fontsize=10)

        if not p["fit_converged"]:
            for ax in (ax_f, ax_F, ax_pw):
                ax.text(0.5, 0.5, "ajuste não estimável", ha="center",
                        va="center", transform=ax.transAxes, color=COR_TEXTO_SEC)
            continue

        beta, eta = p["beta"], p["eta"]
        t = np.linspace(max(float(ttfs.max()) / 400.0, 1e-6),
                        float(ttfs.max()) * 1.2, 400)

        # ── f(t): a densidade, com os eventos observados ao fundo ──
        ax_f.plot(t, densidade(t, beta, eta), color=cor, linewidth=2)
        ax_f.fill_between(t, densidade(t, beta, eta), alpha=0.12, color=cor)
        if eventos.any():
            ax_f.plot(ttfs[eventos], np.zeros(int(eventos.sum())), "|",
                      color=COR_TEXTO_SEC, markersize=10, alpha=0.7)
        ax_f.set_ylabel("f(t)")
        ax_f.set_xlabel("$a_{det}$ (fração da assinatura nominal)")

        # ── F(t): acumulada paramétrica contra mediana de posto ──
        ax_F.plot(t, acumulada(t, beta, eta), color=cor, linewidth=2,
                  label="Weibull")
        if eventos.any():
            obs = np.sort(ttfs[eventos])
            ax_F.plot(obs, mediana_de_posto(len(obs)), "o", color="black",
                      markersize=4, label="mediana de posto")
        ax_F.set_ylim([0, 1.05])
        ax_F.set_ylabel("F(t) = P(T ≤ t)")
        ax_F.set_xlabel("$a_{det}$ (fração da assinatura nominal)")
        ax_F.legend(fontsize=8)

        # ── papel de Weibull: a reta é o teste visual do ajuste ──
        if eventos.sum() >= 3:
            obs = np.sort(ttfs[eventos])
            x_p, y_p = eixos_papel_weibull(obs, mediana_de_posto(len(obs)))
            ax_pw.plot(x_p, y_p, "o", color="black", markersize=4,
                       label="observado")
            x_r = np.linspace(x_p.min(), x_p.max(), 50)
            # reta do ajuste: y = β·(ln t − ln η)
            ax_pw.plot(x_r, beta * (x_r - np.log(eta)), color=cor, linewidth=2,
                       label=f"ajuste (β={beta:.2f})")
            ax_pw.legend(fontsize=8)
        else:
            ax_pw.text(0.5, 0.5, "eventos insuficientes", ha="center",
                       va="center", transform=ax_pw.transAxes,
                       color=COR_TEXTO_SEC)
        ax_pw.set_xlabel("ln t")
        ax_pw.set_ylabel("ln(−ln(1−F))")

    arq = pasta / "weibull_distribuicao.png"
    salvar_figura(
        fig, arq,
        "E2 ilustrativo. No papel de Weibull (linha 3) a distribuição é RETA de "
        "inclinação β; desvio sistemático indica que a família não descreve os "
        "dados — o que o RMSE contra Kaplan-Meier sozinho não mostra.",
    )
    _log(f"   📊 {arq.name}")
