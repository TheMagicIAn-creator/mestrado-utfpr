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
    intensidade_weibull,
    posicoes_probabilidade_censuradas,
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
    """Sobrevivência empírica do detector e referência Weibull em ``a_det``."""
    fig, axes = plt.subplots(
        1, len(FALHAS), figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Probabilidade de ainda não detectar — sobrevivência do detector, não confiabilidade física"
    )

    for ax, falha in zip(axes, FALHAS):
        fid  = falha["id"]
        p    = params[fid]
        ttfs = ttfs_dict[fid]
        eventos = eventos_dict[fid]
        limite_x = _limites_eixo_magnitude(ttfs)
        inicio_a = max(limite_x[0], float(max(ttfs)) / 500.0, 1e-6)
        a = np.linspace(inicio_a, limite_x[1], 400)

        km_t, km_s = curva_kaplan_meier(ttfs, eventos)
        ax.step(
            km_t, km_s, where="post", color="black", linewidth=1.7,
            label=f"Empírica Kaplan-Meier (n={len(ttfs)})",
        )
        if p["fit_converged"]:
            sobrevivencia = confiabilidade(a, p["beta"], p["eta"])
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ax.plot(
                a, sobrevivencia, color=falha["cor"], linewidth=2.2,
                linestyle="-" if recomendada else "--",
                label=(
                    "Weibull 2P"
                    + ("" if recomendada else " — exploratória")
                ),
            )
            if recomendada:
                ax.fill_between(
                    a, sobrevivencia, alpha=0.12, color=falha["cor"]
                )
        ax.set_ylim([0, 1.05])
        ax.set_xlim(*limite_x)
        ax.set_xlabel("Magnitude aplicada, $a$ (fração da assinatura nominal)")
        ax.set_ylabel("$S_D(a)=P(a_{det}>a)$")
        npm_str = f"NPR={falha['npr']}"
        titulo_ajuste = (
            f"β={p['beta']:.2f}, η={p['eta']:.3f}, "
            f"R²pp={(p.get('diagnostico_papel_weibull') or {}).get('r2', float('nan')):.2f}"
            + (f"\n{_rotulo_aderencia(p)}"
               if not p.get("resumo_parametrico_recomendado", False) else "")
            if p["fit_converged"] else _aviso_nao_estimavel(p)
        )
        ax.set_title(f"{falha['nome']} ({npm_str})\n{titulo_ajuste}", fontsize=9)
        ax.legend(fontsize=8, loc="best")

    arq = pasta / "weibull_confiabilidade.png"
    salvar_figura(
        fig, arq,
        "E2 sintético. A escada é a estimativa empírica de ainda não detecção; a linha tracejada é uma referência Weibull 2P rejeitada no ajuste global. O eixo é magnitude, não tempo ou vida útil.",
    )
    _log(f"   📊 {arq.name}")


def plotar_intensidade_deteccao(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Intensidade paramétrica do primeiro cruzamento, sem suporte fictício."""
    fig, axes = plt.subplots(
        1, len(FALHAS), figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Intensidade paramétrica do primeiro cruzamento — não é taxa de falha física"
    )

    for ax, falha in zip(axes, FALHAS):
        fid = falha["id"]
        p = params[fid]
        ttfs = ttfs_dict[fid]
        limite_x = _limites_eixo_magnitude(ttfs)
        inicio_a = max(limite_x[0], float(max(ttfs)) / 500.0, 1e-6)
        a = np.linspace(inicio_a, limite_x[1], 400)

        if p["fit_converged"]:
            intensidade = intensidade_weibull(a, p["beta"], p["eta"])
            recomendada = p.get("resumo_parametrico_recomendado", False)
            ax.plot(
                a, intensidade, color=falha["cor"], linewidth=2.4,
                linestyle="-" if recomendada else "--",
                label=(
                    "Weibull 2P"
                    + ("" if recomendada else " — exploratória")
                ),
            )
            h_max = (
                float(np.nanmax(intensidade))
                if np.isfinite(intensidade).any() else 1.0
            )
            ax.set_ylim(0.0, max(1e-6, h_max * 1.10))
            ax.set_xlim(*limite_x)
            beta_desc = ("crescente ↑" if p["beta"] > 1.1
                         else "constante →" if p["beta"] > 0.9
                         else "decrescente ↓")
            ax.set_title(
                f"{falha['nome']} (NPR={falha['npr']})\n"
                f"$h_D(a)$: β={p['beta']:.2f} — {beta_desc}"
                + ("" if recomendada else " · exploratória")
                + " · escala local",
                fontsize=9,
            )
            ax.legend(fontsize=8, loc="best")
        else:
            ax.text(
                0.5, 0.5, _texto_do_painel_vazio(p),
                transform=ax.transAxes, ha="center", va="center",
                color=COR_TEXTO_SEC, fontsize=8, wrap=True,
            )
            ax.set_title(
                f"{falha['nome']} (NPR={falha['npr']})\n"
                "intensidade não estimável",
                fontsize=9,
            )
        if p["fit_converged"]:
            ax.set_xlabel("Magnitude aplicada, $a$ (fração da assinatura nominal)")
            ax.set_ylabel("$h_D(a)$ por unidade de magnitude")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    arq = pasta / "weibull_intensidade_deteccao.png"
    salvar_figura(
        fig, arq,
        "E2 sintético. h_D(a) é uma função exclusivamente paramétrica e, como a Weibull 2P global foi rejeitada, aparece tracejada para auditoria exploratória. Não há pontos empíricos de taxa nem interpretação de desgaste.",
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


def _tamanhos_pontos_empates(contagens: np.ndarray) -> np.ndarray:
    """Escala visual estável para níveis agrupados no mesmo ``a_det``."""
    contagens = np.asarray(contagens, dtype=float)
    if not len(contagens):
        return np.asarray([])
    return 24.0 + 72.0 * np.sqrt(contagens / contagens.max())


def plotar_funcoes_distribuicao_weibull(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Separa densidade/PDF e distribuição acumulada/ECDF do papel Weibull."""
    fig, axes = plt.subplots(
        2, len(FALHAS), figsize=TAM["painel_6"], layout="constrained"
    )
    fig.suptitle(
        "Funções da magnitude de detecção — dados empíricos e referência Weibull 2P"
    )

    for col, falha in enumerate(FALHAS):
        fid = falha["id"]
        p = params[fid]
        ttfs = np.asarray(ttfs_dict[fid], dtype=float)
        eventos = np.asarray(eventos_dict[fid], dtype=bool)
        t_emp, f_emp, _ = posicoes_probabilidade_censuradas(ttfs, eventos)
        _, contagens_obs = np.unique(ttfs[eventos], return_counts=True)
        tamanhos = _tamanhos_pontos_empates(contagens_obs)
        limite_x = _limites_eixo_magnitude(ttfs)
        ax_f, ax_F = axes[0][col], axes[1][col]

        teste = p.get("teste_aderencia_quantizada") or {}
        p_valor = teste.get("p_value")
        p_texto = f"p={p_valor:.3f}" if p_valor is not None else "p não estimado"
        status = (
            "2P adotada"
            if p.get("resumo_parametrico_recomendado", False)
            else "2P não adotada"
        )
        ax_f.set_title(
            f"{falha['nome']} (NPR={falha['npr']})\n"
            f"n={len(ttfs)} · {p_texto} · {status}",
            fontsize=9,
        )

        if eventos.any():
            valores = ttfs[eventos]
            bins = np.histogram_bin_edges(valores, bins="fd")
            if len(bins) < 6:
                largura = max(
                    float(valores.max() - valores.min()),
                    2.0 / (N_STEPS - 1),
                )
                centro = float(np.median(valores))
                bins = np.linspace(max(0.0, centro - largura), centro + largura, 7)
            ax_f.hist(
                valores, bins=bins, density=True, color=falha["cor"],
                edgecolor=falha["cor"], alpha=0.28,
                label=f"Histograma empírico (n={int(eventos.sum())})",
            )

        if p["fit_converged"]:
            inicio_a = max(limite_x[0], float(ttfs.max()) / 500.0, 1e-6)
            a = np.linspace(inicio_a, limite_x[1], 400)
            recomendada = p.get("resumo_parametrico_recomendado", False)
            estilo = "-" if recomendada else "--"
            rotulo = "Weibull 2P" + ("" if recomendada else " — exploratória")
            ax_f.plot(
                a, densidade(a, p["beta"], p["eta"]),
                color=falha["cor"], linewidth=2.2, linestyle=estilo,
                label=rotulo,
            )
            ax_F.plot(
                a, acumulada(a, p["beta"], p["eta"]),
                color=falha["cor"], linewidth=2.2, linestyle=estilo,
                label=rotulo,
            )
        elif not eventos.any():
            ax_f.text(
                0.5, 0.5, _texto_do_painel_vazio(p),
                transform=ax_f.transAxes, ha="center", va="center",
                color=COR_TEXTO_SEC, fontsize=8,
            )

        if eventos.any():
            ax_F.step(
                t_emp, f_emp, where="post", color="black", linewidth=1.3,
                alpha=0.78,
            )
            ax_F.scatter(
                t_emp, f_emp, s=tamanhos, color="black", edgecolors="white",
                linewidths=0.5, zorder=3,
                label=(
                    f"Empírica: {len(t_emp)} níveis; "
                    "tamanho = multiplicidade"
                ),
            )
        else:
            ax_F.text(
                0.5, 0.5, "nenhuma detecção observada",
                transform=ax_F.transAxes, ha="center", va="center",
                color=COR_TEXTO_SEC,
            )

        ax_f.set_xlim(*limite_x)
        ax_f.set_xlabel("Magnitude de detecção, $a_{det}$ (fração nominal)")
        ax_f.set_ylabel("Densidade $f_D(a)$")
        ax_f.legend(fontsize=7.5, loc="best")
        ax_F.set_xlim(*limite_x)
        ax_F.set_ylim(0.0, 1.05)
        ax_F.set_xlabel("Magnitude de detecção, $a_{det}$ (fração nominal)")
        ax_F.set_ylabel(r"$F_D(a)=P(a_{det}\leq a)$")
        ax_F.legend(fontsize=7.5, loc="best")

    arq = pasta / "weibull_funcoes_distribuicao.png"
    salvar_figura(
        fig, arq,
        "E2 sintético. Barras, escadas e pontos vêm das 277 trajetórias GPVS F0; não há marcas auxiliares sobre o eixo x. Linhas tracejadas são referências Weibull 2P rejeitadas no ajuste global.",
    )
    _log(f"   📊 {arq.name}")


def plotar_distribuicao_weibull(
    ttfs_dict: dict, eventos_dict: dict, params: dict, pasta: Path
):
    """Papel de probabilidade: pontos empíricos e reta da Weibull 2P."""
    fig, axes = plt.subplots(
        1, len(FALHAS), figsize=TAM["painel_3"], layout="constrained"
    )
    fig.suptitle(
        "Papel de probabilidade Weibull — diagnóstico de linearidade da 2P"
    )

    for ax, falha in zip(axes, FALHAS):
        fid = falha["id"]
        p = params[fid]
        ttfs = np.asarray(ttfs_dict[fid], dtype=float)
        eventos = np.asarray(eventos_dict[fid], dtype=bool)
        t_emp, f_emp, _ = posicoes_probabilidade_censuradas(ttfs, eventos)
        _, contagens_obs = np.unique(ttfs[eventos], return_counts=True)
        tamanhos = _tamanhos_pontos_empates(contagens_obs)
        teste = p.get("teste_aderencia_quantizada") or {}
        p_valor = teste.get("p_value")
        p_texto = f"p={p_valor:.3f}" if p_valor is not None else "p não estimado"
        status = (
            "2P adotada"
            if p.get("resumo_parametrico_recomendado", False)
            else "2P rejeitada/não adotada"
        )
        r2 = (p.get("diagnostico_papel_weibull") or {}).get("r2")
        r2_texto = f"R²pp={r2:.2f}" if r2 is not None else "R²pp não estimado"
        ax.set_title(
            f"{falha['nome']} (NPR={falha['npr']})\n"
            f"{r2_texto} · bootstrap {p_texto} · {status}",
            fontsize=8.7,
        )

        if eventos.sum() >= 3:
            x_p, y_p = eixos_papel_weibull(t_emp, f_emp)
            ax.scatter(
                x_p, y_p, s=tamanhos, color="black", edgecolors="white",
                linewidths=0.5, zorder=3,
                label="Posições empíricas; tamanho = multiplicidade",
            )
            if p["fit_converged"]:
                x_reta = np.linspace(x_p.min(), x_p.max(), 80)
                recomendada = p.get("resumo_parametrico_recomendado", False)
                ax.plot(
                    x_reta,
                    p["beta"] * (x_reta - np.log(p["eta"])),
                    color=falha["cor"], linewidth=2.2,
                    linestyle="-" if recomendada else "--",
                    label=(
                        f"Reta Weibull 2P (β={p['beta']:.2f})"
                        + ("" if recomendada else " — exploratória")
                    ),
                )
            ax.legend(fontsize=7.5, loc="best")
        else:
            ax.text(
                0.5, 0.5, "eventos insuficientes para o papel Weibull",
                transform=ax.transAxes, ha="center", va="center",
                color=COR_TEXTO_SEC, fontsize=8,
            )

        ax.set_xlabel(r"$\ln(a_{det})$")
        ax.set_ylabel(r"$\ln[-\ln(1-F_D)]$")

    arq = pasta / "weibull_distribuicao.png"
    salvar_figura(
        fig, arq,
        f"E2 sintético; MLE intervalar (Δa={1.0 / (N_STEPS - 1):.4f}). Uma Weibull 2P adequada forma pontos aproximadamente lineares. A curvatura observada e o bootstrap p=0,004 rejeitam a 2P global; a reta permanece somente para diagnóstico.",
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
