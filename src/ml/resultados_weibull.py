"""Resumo textual dos resultados de detectabilidade Weibull E2."""

from __future__ import annotations

from src.ml.resultados import PASTA_AE, _fmt, _json


def resumo_weibull() -> str | None:
    d = _json(PASTA_AE / "weibull_results.json")
    if not d:
        return None

    tempo = d.get("__meta__", {}).get("tempo", {})
    protocolo = d.get("__meta__", {}).get("protocolo_avaliacao", {})
    amostragem = protocolo.get("amostragem_trajetorias_weibull", {})
    unidade = tempo.get("ttf_unidade", "passos de degradação sintética")
    linhas = [
        "## Detectabilidade E2 / Weibull\n\n",
        f"Unidade do eixo: `{unidade}`; tempo físico calibrado: "
        f"{'sim' if tempo.get('tempo_fisico_calibrado') else 'não'}.\n\n",
        f"Amostragem: **{amostragem.get('n_selecionadas', 'não informada')} "
        "janelas elegíveis**, método "
        f"`{amostragem.get('metodo', 'não informado')}`; por modo: "
        f"`{amostragem.get('por_ensaio', {})}`.\n\n",
        "| Falha | NPR | Detectadas/total | Níveis | beta (IC95%) | eta (IC95%) | p aderência | Grade estável | margem KM | Status |\n",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|\n",
    ]
    linhas_modos = [
        "\n### Estratificação por modo GPVS\n\n",
        "| Falha | Modo | n | beta | eta | p aderência | Grade estável | Uso 2P |\n",
        "|---|---|---:|---:|---:|---:|---|---|\n",
    ]
    for fid, falha in d.get("falhas", {}).items():
        p = falha.get("weibull", {})
        def valor_ci(nome: str, casas: int = 1) -> str:
            valor = p.get(nome)
            ci = p.get(f"{nome}_ci95") or [None, None]
            return f"{_fmt(valor, casas)} [{_fmt(ci[0], casas)}; {_fmt(ci[1], casas)}]"

        status_mapa = {
            "exploratorio_descritivo": "exploratório legado",
            "exploratorio_alta_censura": "legado; alta indetectabilidade",
            "exploratorio_detectabilidade": "exploratório E2",
            "nao_recomendado_alta_indetectabilidade": "ajuste exploratório; alta indetectabilidade",
            "nao_recomendado_desvio_papel_weibull": "ajuste exploratório; desvio de aderência",
            "resolucao_insuficiente_sintese_exploratoria": "ajuste exploratório; resolução insuficiente",
            "desvio_aderencia_sintese_exploratoria": "ajuste exploratório; desvio no bootstrap quantizado",
            "instabilidade_grade_sintese_exploratoria": "ajuste exploratório; sensível à resolução da grade",
            "nao_estimavel": "não estimável",
            "nao_estimavel_parametrico_rul_restrita": (
                "Weibull não estimável; KM restrita disponível"
            ),
        }
        status = status_mapa.get(
            falha.get("status_ajuste"),
            "exploratório" if p.get("fit_converged") else "não estimável",
        )
        teste_aderencia = p.get("teste_aderencia_quantizada") or {}
        p_aderencia = teste_aderencia.get("p_value")
        grade_estavel = (p.get("sensibilidade_grade") or {}).get("estavel")
        linhas.append(
            f"| {falha.get('nome', fid)} | {falha.get('npr')} | "
            f"{p.get('n_eventos', '-')}/{p.get('n_traj', '-')} | "
            f"{p.get('n_niveis_distintos', '-')} | "
            f"{valor_ci('beta', 2)} | {valor_ci('eta')} | "
            f"{_fmt(p_aderencia, 3)} | "
            f"{'sim' if grade_estavel is True else 'não' if grade_estavel is False else '-'} | "
            f"{_fmt(p.get('margem_restrita_inicial', p.get('rul_restrita_inicial')))} | "
            f"{status} |\n"
        )
        for modo, ajuste in sorted((p.get("ajustes_por_modo") or {}).items()):
            p_modo = (
                ajuste.get("teste_aderencia_quantizada") or {}
            ).get("p_value")
            estavel_modo = (
                ajuste.get("sensibilidade_grade") or {}
            ).get("estavel")
            linhas_modos.append(
                f"| {falha.get('nome', fid)} | {modo} | "
                f"{ajuste.get('n_traj', '-')} | {_fmt(ajuste.get('beta'), 3)} | "
                f"{_fmt(ajuste.get('eta'), 3)} | {_fmt(p_modo, 3)} | "
                f"{'sim' if estavel_modo is True else 'não' if estavel_modo is False else '-'} | "
                f"{'adotado em E2' if ajuste.get('resumo_parametrico_recomendado') else 'exploratório'} |\n"
            )
    if len(linhas_modos) > 3:
        linhas.extend(linhas_modos)
    linhas.append(
        "\n**Leitura obrigatória:** esta etapa modela a distribuição da "
        "**magnitude do primeiro cruzamento confirmado do detector**. A curva "
        "S_D(a) é probabilidade de ainda não detectar; h_D(a) é intensidade de "
        "detecção por unidade de magnitude. Nenhuma delas é confiabilidade ou "
        "taxa de falha do componente. A margem restrita de Kaplan-Meier não é "
        "RUL, pois não existe eixo temporal. MTTF, B10 e RUL permanecem apenas "
        "como aliases legados no JSON.\n\n"
        "O MLE usa censura por intervalo na grade de magnitude; os pontos do "
        "papel de Weibull usam Kaplan-Meier modificado, tamanho total da "
        "amostra e empates agrupados. A aderência é testada por bootstrap que "
        "repete a quantização; a estabilidade usa grades de 251 e 501 pontos. "
        "Os ICs vêm de bootstrap de janelas sem amostras "
        "compartilhadas, mas independência temporal não foi demonstrada. "
        "O NPR "
        "prioriza risco na FMECA; ele **não determina** quantos eventos o "
        "experimento sintético produzirá e não explica causalmente a "
        "indetectabilidade."
    )
    return "".join(linhas)
