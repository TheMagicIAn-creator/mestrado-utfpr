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
            "posição empírica agrupada "
            f"(n={n_total}; indetect. no teto={n_indetectaveis})"
        )
    return f"posição empírica agrupada (n={n_total}; sem indetectabilidade)"


def _rotulo_aderencia(p: dict) -> str:
    """Rótulo curto baseado no teste formal, sem transformar E2 em validação."""
    teste = p.get("teste_aderencia_quantizada") or {}
    p_value = teste.get("p_value")
    status = p.get("status_aderencia")
    if status == "resolucao_insuficiente":
        return "resolução insuficiente para síntese 2P"
    if p_value is None:
        return "ajuste 2P exploratório"
    if p.get("aderencia_aceitavel"):
        return f"compatível com 2P (bootstrap p={p_value:.3f})"
    return f"desvio da 2P (bootstrap p={p_value:.3f})"


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

        # O ajuste intervalar é comparado na mesma resolução do experimento:
        # P(a-delta < A <= a) vezes o número de trajetórias. Uma densidade
        # contínua sobre barras quantizadas sugeriria precisão inexistente.
        if p["fit_converged"] and len(observados):
            indice_inicial = max(1, int(np.ceil(limite_x[0] / passo_grade)))
            indice_final = min(
                N_STEPS - 1, int(np.floor(limite_x[1] / passo_grade))
            )
            t_grid = np.arange(indice_inicial, indice_final + 1) * passo_grade
            massa_ajustada = (
                acumulada(t_grid, p["beta"], p["eta"])
                - acumulada(
                    np.maximum(0.0, t_grid - passo_grade),
                    p["beta"], p["eta"],
                )
            )
            contagem_ajustada = len(ttfs) * massa_ajustada
            altura_maxima = max(
                altura_maxima, float(np.nanmax(contagem_ajustada))
            )
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ax.plot(
                t_grid, contagem_ajustada, color="black", linewidth=2.2,
                marker="o", markersize=2.8,
                linestyle="-" if recomendada else "--",
                label=("Massa Weibull 2P por célula"
                       + ("" if recomendada else " — exploratória")),
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
                + (f"\n{_rotulo_aderencia(p)}" if not recomendada else "")
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
        f"Validação sintética E2; Δa={1.0 / (N_STEPS - 1):.4f}. Escalas horizontais ajustadas por componente: compare magnitudes pelos valores dos eixos, não pela largura visual. Não há equivalência com tempo ou vida útil.",
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
        limite_x = _limites_eixo_magnitude(ttfs)
        inicio_t = max(limite_x[0], float(max(ttfs)) / 500.0, 1e-6)
        t = np.linspace(inicio_t, limite_x[1], 400)

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
                label=(
                    "Weibull 2P"
                    + ("" if recomendada else " — exploratória")
                ),
            )
            if recomendada:
                ax_r.fill_between(t, R, alpha=0.12, color=falha["cor"])
        ax_r.set_ylim([0, 1.05])
        ax_r.set_xlim(*limite_x)
        ax_r.set_xlabel("Magnitude da perturbação CA, $a$ (fração nominal)")
        ax_r.set_ylabel("$S_D(a)=P(a_{det}>a)$ — ainda não detectada")
        npm_str = f"NPR={falha['npr']}"
        titulo_ajuste = (
            f"β={p['beta']:.2f}, η={p['eta']:.3f}, "
            f"R²pp={(p.get('diagnostico_papel_weibull') or {}).get('r2', float('nan')):.2f}"
            + (f"\n{_rotulo_aderencia(p)}"
               if not p.get("resumo_parametrico_recomendado", False) else "")
            if p["fit_converged"] else _aviso_nao_estimavel(p)
        )
        ax_r.set_title(f"{falha['nome']} ({npm_str})\n{titulo_ajuste}", fontsize=9)
        ax_r.legend(fontsize=8)

        # Taxa de falha h(t)
        ax_h = axes[1][col]
        if p["fit_converged"]:
            H = taxa_falha(t, p["beta"], p["eta"])
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ax_h.plot(
                t, H, color=falha["cor"], linewidth=2,
                linestyle="-" if recomendada else "--",
                label=(
                    "Weibull 2P"
                    + ("" if recomendada else " — exploratória")
                ),
            )
            niveis_obs, contagens_obs = np.unique(
                np.asarray(ttfs)[np.asarray(eventos, dtype=bool)],
                return_counts=True,
            )
            if len(niveis_obs):
                tamanhos = 25.0 + 45.0 * np.sqrt(
                    contagens_obs / contagens_obs.max()
                )
                ax_h.scatter(
                    niveis_obs, np.zeros(len(niveis_obs)), s=tamanhos,
                    marker="|", color="black", alpha=0.65, clip_on=False,
                    label="suporte empírico de $a_{det}$",
                )
            h_max = float(np.nanmax(H)) if np.isfinite(H).any() else 1.0
            ax_h.set_ylim(0.0, max(1e-6, h_max * 1.10))
            ax_h.set_xlim(*limite_x)
            beta_desc = ("crescente ↑" if p["beta"] > 1.1
                         else "constante →" if p["beta"] > 0.9
                         else "decrescente ↓")
            ax_h.set_title(
                f"Intensidade de detecção $h_D(a)$\n"
                f"β={p['beta']:.2f} — {beta_desc}"
                + ("" if recomendada else " · exploratória")
                + " · escala local", fontsize=9
            )
            ax_h.legend(fontsize=7.5, loc="best")
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
        if p["fit_converged"]:
            ax_h.set_xlabel("Magnitude da perturbação CA, $a$ (fração nominal)")
            ax_h.set_ylabel("$h_D(a)$ por unidade de $a$")
        else:
            ax_h.set_xticks([])
            ax_h.set_yticks([])

    arq = pasta / "weibull_confiabilidade.png"
    salvar_figura(
        fig, arq,
        "E2 sintético. Eixos x e y dos painéis inferiores usam escalas locais ao componente; leia os valores, não a largura visual. S_D(a) e h_D(a) descrevem o detector, não vida ou taxa de falha física.",
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
        if p["fit_converged"]:
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
                linestyle="--",
                label=(
                    "Margem Weibull"
                    if p.get("resumo_parametrico_recomendado", False)
                    else "Margem Weibull exploratória"
                ),
            )
            if ax_param is not ax:
                ax_param.set_ylabel("Margem Weibull extrapolada")
                ax_param.tick_params(axis="y", colors=COR_TEXTO_SEC)
                eixos_legenda.append(ax_param)

        status = (
            "2P exploratória"
            if p["fit_converged"] and not p.get("resumo_parametrico_recomendado", False)
            else "2P adotada em E2"
            if p.get("resumo_parametrico_recomendado", False)
            else "Somente KM restrita"
        )
        p_aderencia = (
            p.get("teste_aderencia_quantizada") or {}
        ).get("p_value")
        p_txt = (
            f"p={p_aderencia:.3f}"
            if p_aderencia is not None else "p não estimado"
        )
        ax.set_title(
            f"{falha['nome']} (NPR={falha['npr']})\n"
            f"n={p['n_eventos']} · {p_txt} · {status}",
            fontsize=8.5,
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
        "Ajuste de Weibull para magnitude de detecção ($a_{det}$) — nível E2"
    )

    for col, falha in enumerate(FALHAS):
        fid = falha["id"]
        p = params[fid]
        ttfs = np.asarray(ttfs_dict[fid], dtype=float)
        eventos = np.asarray(eventos_dict[fid], dtype=bool)
        cor = falha["cor"]
        t_emp, f_emp, _ = posicoes_probabilidade_censuradas(ttfs, eventos)
        niveis_obs, contagens_obs = np.unique(
            ttfs[eventos], return_counts=True
        )
        tamanhos_pontos = (
            24.0 + 72.0 * np.sqrt(contagens_obs / contagens_obs.max())
            if len(contagens_obs) else np.asarray([])
        )
        limite_x = _limites_eixo_magnitude(ttfs)

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
            ax_F.set_xlim(*limite_x)
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
        t = np.linspace(
            max(limite_x[0], float(ttfs.max()) / 500.0, 1e-6),
            limite_x[1], 400,
        )
        recomendada = p.get("resumo_parametrico_recomendado", False)
        estilo = "-" if recomendada else "--"
        r2 = (p.get("diagnostico_papel_weibull") or {}).get("r2")
        ax_f.set_title(
            f"{falha['nome']} (NPR={falha['npr']})\n"
            f"R²pp={r2:.2f} · "
            + ("síntese 2P adotada" if recomendada else _rotulo_aderencia(p)),
            fontsize=9,
        )

        # ── f_D(a): distribuição EMPÍRICA em primeiro plano; a Weibull só
        # aparece como sobreposição, sempre tracejada quando a aderência foi
        # rejeitada. Assim, uma curva suave não se passa pelos dados medidos.
        if eventos.any():
            valores_evento = ttfs[eventos]
            bins = np.histogram_bin_edges(valores_evento, bins="fd")
            if len(bins) < 6:
                largura = max(
                    float(valores_evento.max() - valores_evento.min()),
                    2.0 / (N_STEPS - 1),
                )
                centro = float(np.median(valores_evento))
                bins = np.linspace(
                    max(0.0, centro - largura), centro + largura, 7
                )
            ax_f.hist(
                valores_evento, bins=bins, density=True,
                color=cor, edgecolor=cor, alpha=0.28,
                label=f"Histograma empírico (n={int(eventos.sum())})",
            )
        ax_f.plot(
            t, densidade(t, beta, eta), color=cor, linewidth=2,
            linestyle=estilo,
            label="Weibull 2P" + ("" if recomendada else " — exploratória"),
        )
        if eventos.any():
            ax_f.scatter(
                niveis_obs,
                np.zeros(len(niveis_obs)),
                s=tamanhos_pontos,
                marker="|",
                color=COR_TEXTO_SEC,
                alpha=0.75,
                label=f"{len(niveis_obs)} níveis observados",
            )
        ax_f.set_xlim(*limite_x)
        ax_f.legend(fontsize=7.2, loc="upper left")
        ax_f.set_ylabel("$f_D(a)$")
        ax_f.set_xlabel("Magnitude de detecção, $a_{det}$ (fração nominal)")

        # ── F_D(a): acumulada paramétrica contra posição empírica ──
        ax_F.plot(
            t, acumulada(t, beta, eta), color=cor, linewidth=2,
            linestyle=estilo,
            label="Weibull 2P" + ("" if recomendada else " — exploratória"),
        )
        if eventos.any():
            ax_F.step(
                t_emp, f_emp, where="post", color="black", linewidth=1.2,
                alpha=0.72,
            )
            ax_F.scatter(
                t_emp,
                f_emp,
                s=tamanhos_pontos,
                color="black",
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label=(
                    f"posição empírica: n={int(eventos.size)}, "
                    f"{len(t_emp)} níveis; tamanho = multiplicidade"
                ),
            )
        ax_F.set_ylim([0, 1.05])
        ax_F.set_xlim(*limite_x)
        ax_F.set_ylabel(r"$F_D(a)=P(a_{det}\leq a)$")
        ax_F.set_xlabel("Magnitude de detecção, $a_{det}$ (fração nominal)")
        ax_F.legend(fontsize=8)

        # ── papel de Weibull: a reta é o teste visual do ajuste ──
        if eventos.sum() >= 3:
            x_p, y_p = eixos_papel_weibull(t_emp, f_emp)
            ax_pw.scatter(
                x_p,
                y_p,
                s=tamanhos_pontos,
                color="black",
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
                label="observado; tamanho = multiplicidade",
            )
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
        f"E2 sintético; MLE intervalar (Δa={1.0 / (N_STEPS - 1):.4f}). Histograma, ECDF e pontos vêm das trajetórias GPVS; tamanho indica empates no mesmo nível. Linha tracejada = Weibull exploratória rejeitada/não adotada. Eixos a_det usam escala local.",
    )
    _log(f"   📊 {arq.name}")


def plotar_sensibilidade_grade(
    resultados_grade: dict[int, tuple[dict, dict]],
    pasta: Path,
) -> None:
    """Compara a ECDF de ``a_det`` sob todas as resoluções declaradas."""
    fig, axes = plt.subplots(
        1, len(FALHAS), figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Sensibilidade do primeiro cruzamento à resolução da grade — E2"
    )
    estilos = (":", "--", "-")
    for ax, falha in zip(axes, FALHAS):
        fid = falha["id"]
        for estilo, (n_steps, (a_dets, eventos)) in zip(
            estilos[-len(resultados_grade):], sorted(resultados_grade.items()),
            strict=True,
        ):
            observados = np.sort(a_dets[fid][eventos[fid]])
            ecdf = np.arange(1, len(observados) + 1) / len(a_dets[fid])
            ax.step(
                observados, ecdf, where="post", linewidth=1.8,
                linestyle=estilo, color=falha["cor"],
                label=(
                    f"{n_steps} pontos · Δa={1.0/(n_steps-1):.3f} · "
                    f"{len(np.unique(observados))} níveis"
                ),
            )
        ax.set_title(f"{falha['nome']} (NPR={falha['npr']})", fontsize=10)
        ax.set_xlabel("Magnitude de detecção, $a_{det}$")
        ax.set_ylabel("Probabilidade empírica acumulada")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7.5, loc="lower right")
    arq = pasta / "weibull_sensibilidade_grade.png"
    salvar_figura(
        fig,
        arq,
        "Mesmas janelas GPVS e largura de confirmação 0,02 em todas as grades; convergência visual separa efeito de resolução de efeito amostral.",
    )
    _log(f"   📊 {arq.name}")


def plotar_modos_operacao(
    ttfs_dict: dict,
    eventos_dict: dict,
    ensaios: np.ndarray,
    params: dict,
    pasta: Path,
) -> None:
    """Expõe a heterogeneidade F0L/F0M que a curva global mascara."""
    cores_modo = {"F0L": "#2a78d6", "F0M": "#d98f00"}
    fig, axes = plt.subplots(
        1, len(FALHAS), figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Primeiro cruzamento por modo operacional GPVS — E2 sintético"
    )
    for ax, falha in zip(axes, FALHAS):
        fid = falha["id"]
        linhas_status = []
        for modo in ("F0L", "F0M"):
            mascara = ensaios == modo
            tempos = ttfs_dict[fid][mascara]
            eventos = eventos_dict[fid][mascara]
            t_emp, f_emp, _ = posicoes_probabilidade_censuradas(
                tempos, eventos
            )
            ajuste = params[fid]["ajustes_por_modo"][modo]
            ax.step(
                t_emp, f_emp, where="mid", color=cores_modo[modo],
                linewidth=2.0, label=f"{modo} empírico (n={len(tempos)})",
            )
            if ajuste["fit_converged"]:
                t = np.linspace(
                    max(float(tempos.min()) / 2.0, 1e-6),
                    float(tempos.max()),
                    300,
                )
                adotado = ajuste.get("resumo_parametrico_recomendado", False)
                ax.plot(
                    t, acumulada(t, ajuste["beta"], ajuste["eta"]),
                    color=cores_modo[modo], linewidth=1.6,
                    linestyle="-" if adotado else "--",
                    label=f"{modo} Weibull 2P"
                    + ("" if adotado else " exploratória"),
                )
            p_value = (
                ajuste.get("teste_aderencia_quantizada") or {}
            ).get("p_value")
            linhas_status.append(
                f"{modo}: p={p_value:.3f}"
                if p_value is not None else f"{modo}: p indisponível"
            )
        ax.set_title(
            f"{falha['nome']} (NPR={falha['npr']})\n"
            + " · ".join(linhas_status),
            fontsize=9,
        )
        ax.set_xlabel("Magnitude de detecção, $a_{det}$")
        ax.set_ylabel("Probabilidade empírica acumulada")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7.2, loc="lower right")
    arq = pasta / "weibull_modos_operacao.png"
    salvar_figura(
        fig,
        arq,
        "F0L (IPPT) e F0M (MPPT) pertencem ao mesmo GPVS, mas são regimes distintos. Linhas tracejadas mantêm ajustes exploratórios visíveis; p vem de bootstrap paramétrico com quantização.",
    )
    _log(f"   📊 {arq.name}")
