"""Graficos academicos da magnitude de primeiro cruzamento do detector.

O eixo NAO e tempo: e a magnitude da assinatura injetada em que a deteccao se
confirma, em [0; 1]. Ver o bloco "O EIXO NAO E TEMPO" em src/ml/rul_weibull.py.
"""

from __future__ import annotations

from src.ml.confiabilidade import (
    acumulada,
    confiabilidade,
    densidade,
    eixos_papel_weibull,
    posicoes_probabilidade_censuradas,
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
    motivo_nao_estimavel,
    np,
    plt,
    margem_condicional_weibull,
    rul_restrita_km,
    salvar_figura,
)


def _aviso_nao_estimavel(p: dict, largura: int = 34) -> str:
    """A frase completa, quebrada para caber num título de painel.

    Painel sem beta/eta e legenda dizendo so "nao estimavel" e um buraco mudo:
    nao distingue "faltou 1 evento" de "quebrou". Ver
    rul_weibull.motivo_nao_estimavel.
    """
    import textwrap

    desfechos = p.get("desfechos") or {}
    if not desfechos:
        return "ajuste não estimável"
    n_det = desfechos.get("n_detectadas", "?")
    n_traj = desfechos.get("n_traj", "?")
    pod = desfechos.get("pod_mon_no_teto")
    cabeca = f"NÃO ESTIMÁVEL — {n_det}/{n_traj} detecções"
    if pod is not None:
        cabeca += f" (POD_mon={pod:.0%})"
    return "\n".join(textwrap.wrap(cabeca, largura))


def _texto_do_painel_vazio(p: dict) -> str:
    """Bloco explicativo para o painel que perderia a curva paramétrica."""
    import textwrap

    desfechos = p.get("desfechos") or {}
    if not desfechos:
        return "Sem detecções suficientes\npara estimar h(a)"
    return "\n".join(textwrap.wrap(motivo_nao_estimavel(desfechos), 38))


def plotar_ttf_histogramas(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """a_det detectados e indetectáveis no teto, com ajuste Weibull."""
    n_falhas = len(FALHAS)
    fig, axes = plt.subplots(
        1, n_falhas, figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Primeiro cruzamento confirmado do detector e indetectabilidade no teto"
    )

    for ax, falha in zip(axes, FALHAS):
        fid  = falha["id"]
        nome = falha["nome"]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        p    = params[fid]
        observados = ttfs[eventos]
        censurados = ttfs[~eventos]

        horizonte = float(max(ttfs))
        # Menos eventos, menos bins: com 9 eventos em 12 bins o histograma vira
        # código de barras e sugere ausência de estrutura onde só falta amostra.
        n_bins = int(min(13, max(5, len(observados) // 2 + 1)))
        bins = np.linspace(0.0, horizonte, n_bins)
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
                linewidth=1.3,
                label=f"Mediana observada={np.median(observados):.2f}",
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
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ax.plot(
                t_grid, f_ajustada * escala, color="black", linewidth=2.2,
                linestyle="-" if recomendada else "--",
                label=(f"Weibull 2P (β={p['beta']:.2f}, η={p['eta']:.3f})"
                       + ("" if recomendada else " — não recomendada")),
            )

        if len(censurados):
            ax.axvline(
                horizonte, color=COR_ALERTA, linestyle="--", linewidth=2,
                label=f"Não detectadas em $a$={horizonte:.2f} (n={len(censurados)})",
            )

        if (p.get("resumo_parametrico_recomendado", False)
                and p["b10"] <= horizonte):
            ax.axvline(
                p["b10"], color="#2a78d6", linestyle=":", linewidth=1.7,
                label=f"a10 paramétrico={p['b10']:.2f}",
            )

        npm_str = f"NPR={falha['npr']}"
        if p["fit_converged"]:
            incerto = p["rul_parametrica_alta_incerteza"]
            # β sob censura alta é ARTEFATO (poucos eventos empilhados na borda,
            # η extrapolando além do horizonte) — não é propriedade física. Marca
            # explicitamente para não ser lido como um β confiável.
            beta_txt = f"β={p['beta']:.2f}" + (" (alta indetect.*)" if incerto else "")
            r2 = (p.get("diagnostico_papel_weibull") or {}).get("r2")
            r2_txt = f" · R²pp={r2:.2f}" if r2 is not None else ""
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ajuste = (
                f"{beta_txt} · η={p['eta']:.3f}{r2_txt} · "
                f"indetect.={p['censura_pct']:.0f}%"
                + ("\nmodelo não recomendado para síntese"
                   if not recomendada else "")
            )
        else:
            ajuste = (_aviso_nao_estimavel(p)
                      + "\nKaplan-Meier permanece válida")
        ax.set_title(f"{nome} ({npm_str})\n{ajuste}", fontsize=9)
        ax.set_xlabel("$a_{det}$ (fração da assinatura nominal)")
        ax.set_ylabel("Número de trajetórias")
        ax.set_xlim(0, 1.02)
        ax.legend(fontsize=8)

    arq = pasta / "weibull_ttf.png"
    salvar_figura(
        fig, arq,
        "E2: cada trajetória é uma janela do holdout, não um ativo. Eixo em magnitude; sem equivalência com tempo ou vida útil.",
    )
    _log(f"   📊 {arq.name}")


def plotar_confiabilidade(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Sobrevivência e intensidade do primeiro cruzamento em ``a_det``."""
    fig, axes = plt.subplots(
        2, 3, figsize=TAM["painel_6"], layout="constrained"
    )
    fig.suptitle(
        "Curva de não detecção e intensidade do primeiro cruzamento (E2)"
    )

    for col, falha in enumerate(FALHAS):
        fid  = falha["id"]
        p    = params[fid]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        # O piso acompanha a escala: era 0,1 fixo, o que num eixo que agora vai de
        # 0 a 1 apagaria os primeiros 10% da curva.
        t = np.linspace(max(ttfs) / 300.0, float(max(ttfs)), 300)

        # Confiabilidade R(t)
        ax_r = axes[0][col]
        km_t, km_s = curva_kaplan_meier(ttfs, eventos)
        ax_r.step(km_t, km_s, where="post", color="black", linewidth=1.5,
                  label="Kaplan-Meier")
        if p["fit_converged"]:
            # Fonte única: src/ml/confiabilidade.py. Antes era weibull_min
            # inline aqui, e por isso o valor nunca saía do PNG.
            R = confiabilidade(t, p["beta"], p["eta"])
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ax_r.plot(
                t, R, color=falha["cor"], linewidth=2,
                linestyle="-" if recomendada else "--",
                label="Weibull 2P" + ("" if recomendada else " — cautela"),
            )
            ax_r.fill_between(t, R, alpha=0.12, color=falha["cor"])
        ax_r.set_ylim([0, 1.05])
        ax_r.set_xlabel("$a$ (fração da assinatura nominal)")
        ax_r.set_ylabel("$S_D(a)=P(a_{det}>a)$ — ainda não detectada")
        npm_str = f"NPR={falha['npr']}"
        titulo_ajuste = (
            f"β={p['beta']:.2f}, η={p['eta']:.3f}, "
            f"R²pp={(p.get('diagnostico_papel_weibull') or {}).get('r2', float('nan')):.2f}"
            + ("\nMODELO NÃO RECOMENDADO PARA SÍNTESE"
               if not p.get("resumo_parametrico_recomendado", False) else "")
            if p["fit_converged"] else _aviso_nao_estimavel(p)
        )
        ax_r.set_title(f"{falha['nome']} ({npm_str})\n{titulo_ajuste}", fontsize=9)
        ax_r.legend(fontsize=8)

        # Taxa de falha h(t)
        ax_h = axes[1][col]
        if p["fit_converged"] and p.get("resumo_parametrico_recomendado", False):
            H = taxa_falha(t, p["beta"], p["eta"])
            ax_h.plot(t, H, color=falha["cor"], linewidth=2)
            beta_desc = ("crescente ↑" if p["beta"] > 1.1
                         else "constante →" if p["beta"] > 0.9
                         else "decrescente ↓")
            ax_h.set_title(
                f"Intensidade de detecção $h_D(a)$\n"
                f"β={p['beta']:.2f} — {beta_desc}", fontsize=9
            )
        else:
            motivo = (
                _texto_do_painel_vazio(p) if not p["fit_converged"] else
                "Curva paramétrica omitida\nna síntese acadêmica:\n"
                "alta indetectabilidade ou\ndesvio no papel de Weibull"
            )
            ax_h.text(0.5, 0.5, motivo,
                      transform=ax_h.transAxes, ha="center", va="center",
                      color=COR_TEXTO_SEC, fontsize=7.5, wrap=True)
            ax_h.set_title("Intensidade não reportável", fontsize=9)
        # Eixo numérico num painel sem curva sugere dado que não existe.
        if p["fit_converged"] and p.get("resumo_parametrico_recomendado", False):
            ax_h.set_xlabel("$a$ (fração da assinatura nominal)")
            ax_h.set_ylabel("$h_D(a)$ por unidade de $a$")
        else:
            ax_h.set_xticks([])
            ax_h.set_yticks([])

    arq = pasta / "weibull_confiabilidade.png"
    salvar_figura(
        fig, arq,
        "S_D(a) descreve não detecção do algoritmo, não sobrevivência do componente. h_D(a) não é taxa de falha física.",
    )
    _log(f"   📊 {arq.name}")


def plotar_rul(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Margem residual de magnitude, restrita (KM) e paramétrica."""
    fig, axes = plt.subplots(
        1, len(FALHAS), figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Margem residual de magnitude até a detecção — não é RUL física"
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
            label="Margem restrita KM",
        )
        ax.fill_between(t_pontos, ruls_km, color=falha["cor"], alpha=0.12)
        ax.set_xlabel("$a$ já atingida sem detecção")
        ax.set_ylabel("margem de magnitude até detectar")
        ax.set_xlim(0, horizonte * 0.8)
        ax.set_ylim(0, max(horizonte, max(ruls_km, default=0.0)) * 1.05)

        eixos_legenda = [ax]
        if p["fit_converged"] and p.get("resumo_parametrico_recomendado", False):
            ruls_param = [
                margem_condicional_weibull(t, p["beta"], p["eta"])
                for t in t_pontos
            ]
            # Com alta censura, a extrapolação pode ser várias vezes maior que
            # o horizonte observado. Um eixo próprio mantém as duas estimativas
            # visíveis e impede que a escala paramétrica achate a curva KM.
            ax_param = ax.twinx() if p["rul_parametrica_alta_incerteza"] else ax
            ax_param.plot(
                t_pontos, ruls_param, color="black", linewidth=1.8,
                linestyle="--", label="Margem Weibull",
            )
            if ax_param is not ax:
                ax_param.set_ylabel("Margem Weibull extrapolada")
                ax_param.tick_params(axis="y", colors=COR_TEXTO_SEC)
                eixos_legenda.append(ax_param)

        status = (
            "Weibull não recomendada"
            if p["fit_converged"] and not p.get("resumo_parametrico_recomendado", False)
            else "Weibull + KM"
            if p.get("resumo_parametrico_recomendado", False)
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
        "E2. Margem em fração de assinatura, condicionada ao detector; não mede tempo restante até falha do componente.",
    )
    _log(f"   📊 {arq.name}")


def plotar_distribuicao_weibull(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """f(a), F_D(a) e papel de Weibull com posições censura-aware.

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
    fig.suptitle("Diagnóstico Weibull da magnitude de detecção — densidade, "
                 "acumulada e papel censura-aware")

    for col, falha in enumerate(FALHAS):
        fid = falha["id"]
        p = params[fid]
        ttfs = np.asarray(ttfs_dict[fid], dtype=float)
        eventos = np.asarray(eventos_dict[fid], dtype=bool)
        cor = falha["cor"]
        t_emp, f_emp, _ = posicoes_probabilidade_censuradas(ttfs, eventos)

        ax_f, ax_F, ax_pw = axes[0][col], axes[1][col], axes[2][col]
        ax_f.set_title(f"{falha['nome']} (NPR={falha['npr']})", fontsize=10)

        if not p["fit_converged"]:
            # Três painéis em branco era o que fazia o IGBT "sumir" do capítulo.
            # Só a DENSIDADE precisa de β e η; a acumulada empírica e o papel de
            # Weibull são construídos a partir dos eventos observados e seguem
            # informativos — o papel, em especial, é o que MOSTRA por que o
            # ajuste não fecha: poucos pontos, ou pontos fora de uma reta.
            # Título curto e no mesmo formato dos outros dois painéis: o
            # detalhe já está no corpo, e repeti-lo no título estourava a
            # linha e brigava com o suptitle.
            ax_f.set_title(f"{falha['nome']} (NPR={falha['npr']})\n"
                           f"f(t) exige β e η — não estimados", fontsize=10)
            ax_f.text(0.5, 0.5, _texto_do_painel_vazio(p), ha="center",
                      va="center", transform=ax_f.transAxes,
                      color=COR_TEXTO_SEC, fontsize=7.5)
            # Sem densidade, os eixos numéricos só sugerem dado que não existe.
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            if eventos.any():
                ax_F.plot(t_emp, f_emp, "o", color="black",
                          markersize=4, label="posição com censura")
                ax_F.legend(fontsize=8)
            else:
                ax_F.text(0.5, 0.5, "nenhuma detecção", ha="center",
                          va="center", transform=ax_F.transAxes,
                          color=COR_TEXTO_SEC)
            ax_F.set_ylim([0, 1.05])
            ax_F.set_ylabel("F(t) = P(T ≤ t)")
            ax_F.set_xlabel("$a_{det}$ (fração da assinatura nominal)")

            if eventos.sum() >= 3:
                x_p, y_p = eixos_papel_weibull(t_emp, f_emp)
                ax_pw.plot(x_p, y_p, "o", color="black", markersize=4,
                           label="observado (sem ajuste)")
                ax_pw.legend(fontsize=8)
            else:
                ax_pw.text(0.5, 0.5, "eventos insuficientes", ha="center",
                           va="center", transform=ax_pw.transAxes,
                           color=COR_TEXTO_SEC)
            ax_pw.set_xlabel("ln t")
            ax_pw.set_ylabel("ln(−ln(1−F))")
            continue

        beta, eta = p["beta"], p["eta"]
        t = np.linspace(max(float(ttfs.max()) / 400.0, 1e-6),
                        float(ttfs.max()), 400)
        recomendada = p.get("resumo_parametrico_recomendado", False)
        estilo = "-" if recomendada else "--"
        r2 = (p.get("diagnostico_papel_weibull") or {}).get("r2")
        ax_f.set_title(
            f"{falha['nome']} (NPR={falha['npr']})\n"
            f"R²pp={r2:.2f} · "
            + ("síntese recomendada" if recomendada else "não recomendada"),
            fontsize=9,
        )

        # ── f(t): a densidade, com os eventos observados ao fundo ──
        ax_f.plot(t, densidade(t, beta, eta), color=cor, linewidth=2,
                  linestyle=estilo)
        ax_f.fill_between(t, densidade(t, beta, eta), alpha=0.12, color=cor)
        if eventos.any():
            ax_f.plot(ttfs[eventos], np.zeros(int(eventos.sum())), "|",
                      color=COR_TEXTO_SEC, markersize=10, alpha=0.7)
        ax_f.set_ylabel("$f_D(a)$")
        ax_f.set_xlabel("$a_{det}$ (fração da assinatura nominal)")

        # ── F(t): acumulada paramétrica contra mediana de posto ──
        ax_F.plot(t, acumulada(t, beta, eta), color=cor, linewidth=2,
                  linestyle=estilo, label="Weibull 2P")
        if eventos.any():
            ax_F.plot(t_emp, f_emp, "o", color="black",
                      markersize=4, label="posição com censura")
        ax_F.set_ylim([0, 1.05])
        ax_F.set_ylabel(r"$F_D(a)=P(a_{det}\leq a)$")
        ax_F.set_xlabel("$a_{det}$ (fração da assinatura nominal)")
        ax_F.legend(fontsize=8)

        # ── papel de Weibull: a reta é o teste visual do ajuste ──
        if eventos.sum() >= 3:
            x_p, y_p = eixos_papel_weibull(t_emp, f_emp)
            ax_pw.plot(x_p, y_p, "o", color="black", markersize=4,
                       label="observado")
            x_r = np.linspace(x_p.min(), x_p.max(), 50)
            # reta do ajuste: y = β·(ln t − ln η)
            ax_pw.plot(x_r, beta * (x_r - np.log(eta)), color=cor, linewidth=2,
                       linestyle=estilo,
                       label=f"ajuste (β={beta:.2f})")
            ax_pw.legend(fontsize=8)
        else:
            ax_pw.text(0.5, 0.5, "eventos insuficientes", ha="center",
                       va="center", transform=ax_pw.transAxes,
                       color=COR_TEXTO_SEC)
        ax_pw.set_xlabel("ln $a_{det}$")
        ax_pw.set_ylabel("ln(−ln(1−F))")

    arq = pasta / "weibull_distribuicao.png"
    salvar_figura(
        fig, arq,
        "E2. Pontos usam Kaplan-Meier modificado com n total; linha tracejada indica modelo não recomendado. R²pp é triagem, não teste formal.",
    )
    _log(f"   📊 {arq.name}")
