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
    N_STEPS,
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


def _rotulo_posicoes_empiricas(eventos: np.ndarray) -> str:
    """Descreve a amostra sem sugerir censura quando ela não ocorreu."""
    eventos = np.asarray(eventos, dtype=bool)
    n_total = int(eventos.size)
    n_indetectaveis = int((~eventos).sum())
    if n_indetectaveis:
        return (
            "posição empírica "
            f"(n={n_total}; indetect. no teto={n_indetectaveis})"
        )
    return f"posição empírica (n={n_total}; sem indetectabilidade)"


def _limites_eixo_magnitude(valores: np.ndarray) -> tuple[float, float]:
    """Enquadra cada componente sem esconder a unidade comum ``a_det``.

    Os painéis são pequenos múltiplos de distribuições com escalas muito
    diferentes. O Fusível ocupa menos de 5% do domínio nominal; fixar todos os
    eixos em [0, 1] reduz seus seis valores observados a uma linha no canto.
    """
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]
    if not valores.size:
        return 0.0, 1.0

    minimo = float(valores.min())
    maximo = float(valores.max())
    passo_grade = 1.0 / (N_STEPS - 1)
    amplitude = max(maximo - minimo, 2.0 * passo_grade)
    margem = max(1.5 * passo_grade, 0.08 * amplitude)
    limite_inferior = max(0.0, minimo - margem)
    limite_superior = min(1.0, maximo + margem)
    if np.isclose(maximo, 1.0):
        limite_superior = 1.01
    return limite_inferior, limite_superior


def plotar_ttf_histogramas(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Frequência discreta de a_det e ajuste Weibull por componente."""
    n_falhas = len(FALHAS)
    fig, axes = plt.subplots(
        1, n_falhas, figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Distribuição discreta do primeiro cruzamento — validação sintética E2"
    )

    for ax, falha in zip(axes, FALHAS):
        fid  = falha["id"]
        nome = falha["nome"]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        p    = params[fid]
        observados = ttfs[eventos]
        nao_detectados = ttfs[~eventos]

        horizonte = float(max(ttfs))
        limite_x = _limites_eixo_magnitude(ttfs)
        passo_grade = 1.0 / (N_STEPS - 1)
        altura_maxima = 0.0
        if len(observados):
            magnitudes, contagens = np.unique(observados, return_counts=True)
            altura_maxima = float(contagens.max())
            ax.bar(
                magnitudes, contagens, width=0.82 * passo_grade,
                alpha=0.72, color=falha["cor"], edgecolor="white",
                linewidth=0.6,
                label=f"Frequência (n={len(observados)})",
            )
        else:
            ax.text(
                0.5, 0.55, "nenhuma detecção na grade observada",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=COR_TEXTO_SEC,
            )
        if len(observados):
            ax.axvline(
                float(np.median(observados)), color="0.35", linestyle="--",
                linewidth=1.3,
                label=f"Mediana={np.median(observados):.2f}",
            )

        # A densidade contínua é convertida em contagem esperada por ponto da
        # grade: f_D(a) × n_total × Δa. Assim a curva e as barras discretas usam
        # a mesma escala vertical sem depender de bins arbitrários.
        if p["fit_converged"] and len(observados):
            inicio_curva = max(limite_x[0], 1e-6)
            fim_curva = min(limite_x[1], 1.0)
            t_grid = np.linspace(inicio_curva, fim_curva, 400)
            f_ajustada = densidade(t_grid, p["beta"], p["eta"])
            escala = len(ttfs) * passo_grade
            altura_maxima = max(
                altura_maxima, float(np.nanmax(f_ajustada * escala))
            )
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ax.plot(
                t_grid, f_ajustada * escala, color="black", linewidth=2.2,
                linestyle="-" if recomendada else "--",
                label=("Weibull 2P"
                       + ("" if recomendada else " — não recomendada")),
            )

        if len(nao_detectados):
            ax.axvline(
                horizonte, color=COR_ALERTA, linestyle="--", linewidth=1.5,
                label="_nolegend_",
            )
            ax.scatter(
                [horizonte], [0.0], marker=">", s=64,
                facecolors="white", edgecolors=COR_ALERTA, linewidths=1.5,
                clip_on=False, zorder=5,
                label=(
                    f"Indetectáveis no teto $a$={horizonte:.2f} "
                    f"(n={len(nao_detectados)})"
                ),
            )

        if (p.get("resumo_parametrico_recomendado", False)
                and p["b10"] <= horizonte):
            ax.axvline(
                p["b10"], color="#2a78d6", linestyle=":", linewidth=1.7,
                label=f"a10={p['b10']:.2f}",
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
        ax.set_xlabel("Magnitude de detecção, $a_{det}$ (fração da assinatura nominal)")
        ax.set_ylabel("Trajetórias por ponto da grade")
        ax.set_xlim(*limite_x)
        if altura_maxima > 0:
            ax.set_ylim(0.0, altura_maxima * 1.30)
        ax.legend(fontsize=8, loc="upper right")

    arq = pasta / "weibull_ttf.png"
    salvar_figura(
        fig, arq,
        "Validação sintética E2; Δa=1/119. Escalas horizontais ajustadas por componente: compare magnitudes pelos valores dos eixos, não pela largura visual. Não há equivalência com tempo ou vida útil.",
    )
    _log(f"   📊 {arq.name}")


def plotar_confiabilidade(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Não detecção e intensidade do primeiro cruzamento em ``a_det``."""
    fig, axes = plt.subplots(
        2, 3, figsize=TAM["painel_6"], layout="constrained"
    )
    fig.suptitle(
        "Curva de não detecção e intensidade do primeiro cruzamento — E2 sintético"
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
        ax_r.set_xlabel("Magnitude da perturbação CA, $a$ (fração nominal)")
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
            ax_h.set_xlabel("Magnitude da perturbação CA, $a$ (fração nominal)")
            ax_h.set_ylabel("$h_D(a)$ por unidade de $a$")
        else:
            ax_h.set_xticks([])
            ax_h.set_yticks([])

    arq = pasta / "weibull_confiabilidade.png"
    salvar_figura(
        fig, arq,
        "Validação sintética E2. S_D(a) descreve não detecção do algoritmo, não sobrevivência do componente; h_D(a) não é taxa de falha física.",
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
        ax.set_xlabel("Magnitude $a$ já atingida sem detecção")
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
        "Validação sintética E2. Margem em fração de assinatura, condicionada ao detector; não mede tempo restante até falha do componente.",
    )
    _log(f"   📊 {arq.name}")


def plotar_distribuicao_weibull(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """f_D(a), F_D(a) e papel de Weibull com não detecções explícitas.

    O pesquisador apontou não ter visto "distribuição de Weibull". As curvas
    S_D(a) e h_D(a) existiam (`plotar_confiabilidade`), mas a **densidade** e a
    **acumulada** não eram desenhadas em lugar nenhum, e o **papel de Weibull**
    — o gráfico canônico da área — também não.

    O papel de Weibull é o que mais informa sobre a qualidade do ajuste: na
    escala `ln a_det × ln(−ln(1−F_D))` a distribuição vira RETA de inclinação β.
    Desvio sistemático da reta é evidência de que a família não serve — algo
    que o RMSE contra Kaplan-Meier, sozinho, não revela.
    """
    fig, axes = plt.subplots(3, len(FALHAS), figsize=TAM["painel_9"],
                             layout="constrained")
    fig.suptitle(
        "Ajuste de Weibull para severidade de detecção ($a_{det}$) — nível E2"
    )

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
                           f"$f_D(a)$ exige β e η — não estimados", fontsize=10)
            ax_f.text(0.5, 0.5, _texto_do_painel_vazio(p), ha="center",
                      va="center", transform=ax_f.transAxes,
                      color=COR_TEXTO_SEC, fontsize=7.5)
            # Sem densidade, os eixos numéricos só sugerem dado que não existe.
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            if eventos.any():
                ax_F.plot(t_emp, f_emp, "o", color="black",
                          markersize=4,
                          label=_rotulo_posicoes_empiricas(eventos))
                ax_F.legend(fontsize=8)
            else:
                ax_F.text(0.5, 0.5, "nenhuma detecção", ha="center",
                          va="center", transform=ax_F.transAxes,
                          color=COR_TEXTO_SEC)
            ax_F.set_ylim([0, 1.05])
            ax_F.set_ylabel(r"$F_D(a)=P(a_{det}\leq a)$")
            ax_F.set_xlabel(
                "Magnitude de detecção, $a_{det}$ (fração nominal)"
            )

            if eventos.sum() >= 3:
                x_p, y_p = eixos_papel_weibull(t_emp, f_emp)
                ax_pw.plot(x_p, y_p, "o", color="black", markersize=4,
                           label="observado (sem ajuste)")
                ax_pw.legend(fontsize=8)
            else:
                ax_pw.text(0.5, 0.5, "eventos insuficientes", ha="center",
                           va="center", transform=ax_pw.transAxes,
                           color=COR_TEXTO_SEC)
            ax_pw.set_xlabel("ln $a_{det}$")
            ax_pw.set_ylabel(r"ln($-\ln(1-F_D)$)")
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

        # ── f_D(a): densidade com os eventos observados ao fundo ──
        ax_f.plot(t, densidade(t, beta, eta), color=cor, linewidth=2,
                  linestyle=estilo)
        ax_f.fill_between(t, densidade(t, beta, eta), alpha=0.12, color=cor)
        if eventos.any():
            ax_f.plot(ttfs[eventos], np.zeros(int(eventos.sum())), "|",
                      color=COR_TEXTO_SEC, markersize=10, alpha=0.7)
        ax_f.set_ylabel("$f_D(a)$")
        ax_f.set_xlabel("Magnitude de detecção, $a_{det}$ (fração nominal)")

        # ── F_D(a): acumulada paramétrica contra posição empírica ──
        ax_F.plot(t, acumulada(t, beta, eta), color=cor, linewidth=2,
                  linestyle=estilo, label="Weibull 2P")
        if eventos.any():
            ax_F.plot(t_emp, f_emp, "o", color="black",
                      markersize=4,
                      label=_rotulo_posicoes_empiricas(eventos))
        ax_F.set_ylim([0, 1.05])
        ax_F.set_ylabel(r"$F_D(a)=P(a_{det}\leq a)$")
        ax_F.set_xlabel("Magnitude de detecção, $a_{det}$ (fração nominal)")
        ax_F.legend(fontsize=8)

        # ── papel de Weibull: a reta é o teste visual do ajuste ──
        if eventos.sum() >= 3:
            x_p, y_p = eixos_papel_weibull(t_emp, f_emp)
            ax_pw.plot(x_p, y_p, "o", color="black", markersize=4,
                       label="observado")
            x_r = np.linspace(x_p.min(), x_p.max(), 50)
            # reta do ajuste: y = β·(ln a_det − ln η)
            ax_pw.plot(x_r, beta * (x_r - np.log(eta)), color=cor, linewidth=2,
                       linestyle=estilo,
                       label=f"ajuste (β={beta:.2f})")
            ax_pw.legend(fontsize=8)
        else:
            ax_pw.text(0.5, 0.5, "eventos insuficientes", ha="center",
                       va="center", transform=ax_pw.transAxes,
                       color=COR_TEXTO_SEC)
        ax_pw.set_xlabel("ln $a_{det}$")
        ax_pw.set_ylabel(r"ln($-\ln(1-F_D)$)")

    arq = pasta / "weibull_distribuicao.png"
    salvar_figura(
        fig, arq,
        "Validação sintética E2. Pontos empíricos usam n total; linha tracejada indica modelo não recomendado. R²pp é triagem, não teste formal.",
    )
    _log(f"   📊 {arq.name}")
