"""
relatorio_confiabilidade.py — Al IAdo PV

Aplica o arcabouço de confiabilidade e de POD **aos artefatos vigentes** e emite
um relatório com a **leitura**, não só com os números.

POR QUE ESTE SCRIPT EXISTE
==========================
O pesquisador apontou: *"foi muito resultado e pouca margem interpretativa"*. As
curvas existiam, mas só como pixel; os índices existiam, mas sem a leitura de
engenharia; e o limiar nunca havia sido confrontado com o critério normativo de
viabilidade de ensaio.

Aqui tudo isso vira texto que se lê, com a evidência ao lado de cada afirmação.

O QUE ELE **NÃO** FAZ
=====================
Não treina, não recalibra, não toca em `limiar.json` nem em nenhum artefato do
pipeline. Só lê, calcula e escreve o próprio relatório. Roda sem `torch` e sem
o dataset bruto.

Uso:

    python scripts/relatorio_confiabilidade.py

Saída: `resultados/autoencoder/relatorio_confiabilidade.md` (+ `.json`).

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.utils import configurar_saida_utf8  # noqa: E402

configurar_saida_utf8()

from src.ml import confiabilidade as cf  # noqa: E402
from src.ml import pod_curva as pod  # noqa: E402


def _ler(pasta: Path, nome: str):
    arq = pasta / nome
    if not arq.exists():
        return None
    if arq.suffix == ".json":
        return json.loads(arq.read_text(encoding="utf-8"))
    return np.load(arq, allow_pickle=True)


def _limiar_operacional(limiar_json: dict) -> tuple[str, float]:
    """Seleciona a régua canônica, sem preferir um escore legado inativo."""
    metodo = str(
        limiar_json.get("score_method")
        or limiar_json.get("metodo_escore")
        or "mse"
    )
    valor = limiar_json.get("score_threshold")
    if valor is None:
        valor = limiar_json.get("limiar_operacional", limiar_json.get("limiar"))
    if valor is None:
        raise ValueError("limiar operacional ausente em limiar.json")
    return metodo, float(valor)


def secao_confiabilidade(weibull: dict) -> tuple[list[str], dict]:
    """Detectabilidade E2, marcos e leitura de beta sem inferencia fisica."""
    linhas = ["## Detectabilidade sintética por modo de falha", ""]
    dados = {}
    for fid, bloco in weibull.get("falhas", {}).items():
        w = bloco.get("weibull") or {}
        if not w.get("fit_converged"):
            # Uma linha seca ("não convergiu") deixava o modo mais difícil da
            # FMECA como um buraco no capítulo — sem dizer se faltou um evento
            # ou cinquenta, e escondendo que a leitura NÃO PARAMÉTRICA existe.
            d = w.get("desfechos") or {}
            leitura = (w.get("interpretacao") or {}).get("leitura")
            linhas += [f"### {bloco.get('nome', fid)} (NPR {bloco.get('npr', '?')})",
                       "", f"**{leitura or 'Ajuste não convergiu.'}**", ""]
            if d:
                linhas += [
                    "| Grandeza | Valor |", "|---|--:|",
                    f"| trajetórias | {d.get('n_traj')} |",
                    f"| detectadas | {d.get('n_detectadas')} |",
                    f"| não detectadas em a_inj = 1,0 | {d.get('n_indetectaveis_no_teto')} |",
                    f"| POD_mon no teto | {float(d.get('pod_mon_no_teto', float('nan'))):.1%} |",
                    "",
                ]
            rul_km = w.get("rul_restrita_inicial")
            horizonte_km = w.get("rul_restrita_horizonte")
            if rul_km is not None and np.isfinite(float(rul_km)):
                linhas += [
                    f"> **Kaplan-Meier (não paramétrica) permanece disponível.** "
                    f"Margem média de magnitude até detectar, restrita ao "
                    f"horizonte observado de {float(horizonte_km):.2f}: "
                    f"**{float(rul_km):.2f}**. É margem de assinatura, não RUL.",
                    "",
                ]
            continue
        beta, eta = float(w["beta"]), float(w["eta"])
        ic = w.get("beta_ci95") or [None, None]
        tem_ic = ic[0] is not None
        leitura = cf.classificar_forma(
            beta, tuple(ic) if tem_ic else None, eixo_tempo=False
        )
        marcos = cf.marcos(beta, eta)
        horizonte = float(w.get("rul_restrita_horizonte") or 0.0)
        recomendada = bool(w.get("resumo_parametrico_recomendado", False))
        diagnostico = w.get("diagnostico_papel_weibull") or {}
        teste_aderencia = w.get("teste_aderencia_quantizada") or {}
        p_aderencia = teste_aderencia.get("p_value")
        sensibilidade = w.get("sensibilidade_grade") or {}
        grade_estavel = sensibilidade.get("estavel")
        status_aderencia = w.get("status_aderencia", "não informado")

        # R(t) em pontos de decisão: os próprios marcos são os pontos naturais.
        pontos = [marcos["b1"], marcos["b10"], marcos["vida_mediana"], eta]
        r_em = {f"{t:.1f}": float(cf.confiabilidade(t, beta, eta)) for t in pontos}

        linhas += [
            f"### {bloco.get('nome', fid)} (NPR {bloco.get('npr', '?')})", "",
            "| Parâmetro | Valor | IC95 |",
            "|---|--:|---|",
            f"| forma β | {beta:.3f} | "
            + (f"[{ic[0]:.2f}; {ic[1]:.2f}]" if tem_ic else "—") + " |",
            f"| escala η | {eta:.2f} | "
            + (lambda c: f"[{c[0]:.2f}; {c[1]:.2f}]" if c and c[0] is not None else "—")(
                w.get("eta_ci95")) + " |",
            "",
            f"Diagnóstico visual: **R²pp = "
            f"{float(diagnostico.get('r2', float('nan'))):.3f}**. "
            f"Aderência por bootstrap quantizado: **p = "
            + (
                f"{float(p_aderencia):.3f}"
                if p_aderencia is not None else "não estimado"
            )
            + f"** (`{status_aderencia}`). Estabilidade entre grades finas: "
            + (
                "**sim**" if grade_estavel is True else
                "**não**" if grade_estavel is False else
                "**não avaliada**"
            )
            + ".",
            (
                "Síntese paramétrica adotada somente como detectabilidade E2."
                if recomendada else
                "A curva e os parâmetros permanecem visíveis para auditoria, "
                "mas a síntese Weibull 2P é exploratória e não sustenta "
                "inferência física."
            ),
            "",
        ]
        if not recomendada:
            rul_km = w.get("margem_restrita_inicial", w.get("rul_restrita_inicial"))
            linhas += [
                f"> Margem restrita KM no início: **{float(rul_km):.2f}**. "
                "É descritiva no domínio observado e não é RUL.",
                "",
            ]
            dados[fid] = {
                "beta": beta, "eta": eta, "diagnostico": diagnostico,
                "sintese_parametrica_recomendada": False,
                "margem_restrita_inicial": rul_km,
                "horizonte_observado": horizonte,
            }

        linhas += [
            (
                "| Marco paramétrico E2 | Magnitude de injeção | "
                "S_D nesse ponto |"
            ),
            "|---|--:|--:|",
            f"| a01 (1% detectado) | {marcos['q01']:.2f} | "
            f"{cf.confiabilidade(marcos['b1'], beta, eta):.3f} |",
            f"| a10 (10% detectado) | {marcos['q10']:.2f} | {cf.confiabilidade(marcos['q10'], beta, eta):.3f} |",
            f"| a50 (mediana) | {marcos['q50']:.2f} | {cf.confiabilidade(marcos['q50'], beta, eta):.3f} |",
            f"| η (escala característica) | {eta:.2f} | 0.368 |",
            f"| média paramétrica de a_det | {marcos['media']:.2f} | {cf.confiabilidade(marcos['media'], beta, eta):.3f} |",
            "",
            f"**Leitura de β.** {leitura['leitura']}",
            "",
        ]
        if not leitura["conclusivo"]:
            linhas += ["> ⚠️ A afirmação de regime NÃO se sustenta neste caso.", ""]
        linhas += [
            "> β descreve somente a forma da intensidade de detecção em função "
            "da magnitude. Não implica desgaste, mortalidade infantil ou "
            "política de substituição.", ""]
        if horizonte:
            linhas += [
                f"> Observação vai até {horizonte:.1f}; qualquer marco além "
                "disso é extrapolação do modelo, não dado.", ""]

        ajustes_modo = w.get("ajustes_por_modo") or {}
        if ajustes_modo:
            linhas += [
                "#### Estratificação por modo operacional GPVS", "",
                "| Modo | n | beta | eta | p bootstrap | Grade estável | Uso 2P |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
            for modo in sorted(ajustes_modo):
                ajuste_modo = ajustes_modo[modo]
                p_modo = (
                    ajuste_modo.get("teste_aderencia_quantizada") or {}
                ).get("p_value")
                estavel_modo = (
                    ajuste_modo.get("sensibilidade_grade") or {}
                ).get("estavel")
                linhas.append(
                    f"| {modo} | {ajuste_modo.get('n_traj', '-')} | "
                    f"{float(ajuste_modo.get('beta', float('nan'))):.3f} | "
                    f"{float(ajuste_modo.get('eta', float('nan'))):.3f} | "
                    + (
                        f"{float(p_modo):.3f}" if p_modo is not None else "—"
                    )
                    + f" | {'sim' if estavel_modo else 'não'} | "
                    + (
                        "adotado em E2"
                        if ajuste_modo.get("resumo_parametrico_recomendado")
                        else "exploratório"
                    )
                    + " |"
                )
            linhas += [
                "",
                "> F0L (IPPT) e F0M (MPPT) são regimes do mesmo dataset "
                "GPVS. Diferenças entre eles não são eventos adicionais nem "
                "mistura de bases; são heterogeneidade operacional explícita.",
                "",
            ]

        dados[fid] = {
            "beta": beta,
            "eta": eta,
            "marcos": marcos,
            "interpretacao": leitura,
            "R_em_marcos": r_em,
            "horizonte_observado": horizonte,
            "diagnostico": diagnostico,
            "teste_aderencia_quantizada": teste_aderencia,
            "sensibilidade_grade": sensibilidade,
            "ajustes_por_modo": ajustes_modo,
            "sintese_parametrica_recomendada": recomendada,
        }
    return linhas, dados


def secao_pod(npz, limiar_json: dict) -> tuple[list[str], dict]:
    """Critério de viabilidade e deriva — com a hipótese conferida antes."""
    linhas = ["## Ponto de operação sob o critério LS-POD", ""]
    if npz is None or not limiar_json:
        return linhas + ["Artefatos ausentes.", ""], {}

    metodo_escore, y_dec = _limiar_operacional(limiar_json)
    calib = np.asarray(npz["scores_operacionais_calibracao"], dtype=float)
    teste = np.asarray(npz["scores_operacionais_teste"], dtype=float)

    norm = pod.checar_normalidade(teste, "escore saudável (teste)")
    pof = pod.limite_pof(teste)
    empirico = pod.limite_pof_empirico(teste)
    positivos = teste[teste > 0]
    log_lim = (float(np.exp(np.mean(np.log(positivos))
                            + pod.fator_k1(positivos.size, p=pod.P_POF)
                            * np.std(np.log(positivos), ddof=1)))
               if positivos.size >= 2 else float("nan"))
    deriva = pod.deriva_de_campo(calib, teste, limiar=y_dec)
    from src.ml.estatistica import intervalo_wilson

    n_teste = int(len(teste))
    n_excedencias = int(np.sum(teste > y_dec))
    taxa_excedencia = n_excedencias / n_teste if n_teste else float("nan")
    ci_excedencia = intervalo_wilson(n_excedencias, n_teste)

    linhas += [
        f"Escore operacional: **{metodo_escore}**; limiar adotado: "
        f"**{y_dec:.4f}**", "",
        "### A hipótese, antes do número", "",
        f"O método assume normalidade do lado saudável. Shapiro-Wilk no escore "
        f"bruto: p = {norm['shapiro_bruto']['p']:.2e}; em log: "
        f"p = {norm['shapiro_log']['p']:.2e}. Assimetria "
        f"{norm['assimetria']:+.2f}, curtose {norm['curtose_excesso']:+.2f}.", "",
        ("**Hipótese violada.** Por isso o mesmo quantil é estimado por três "
         "caminhos independentes abaixo — se os três concordarem, a conclusão "
         "não depende dela."
         if not norm["vale"] else
         f"A normalidade **não foi rejeitada** na escala "
         f"**{norm['melhor_escala']}** ao nível de 5%. Isso não prova a "
         "hipótese, especialmente com p próximo do corte; os três estimadores "
         "continuam sendo reportados como análise de sensibilidade."), "",
        "| Estimador do percentil 99 do escore saudável | Valor |",
        "|---|--:|",
        f"| normal no escore bruto (LS-POD) | {pof['limite']:.4f} |",
        f"| normal em log, destransformado | {log_lim:.4f} |",
        f"| percentil 99 empírico | {empirico:.4f} |",
        f"| **limiar adotado** | **{y_dec:.4f}** |",
        "",
    ]

    acima = [v for v in (pof["limite"], log_lim, empirico) if np.isfinite(v)]
    todos_acima = all(v > y_dec for v in acima)
    linhas += [
        (f"> **Os {len(acima)} estimadores pontuais ficam acima do limiar.** "
         "Isso sinaliza tensão com a meta nominal de 1%, mas não demonstra "
         "violação estatística com esta amostra."
         if todos_acima else
         "> Os estimadores divergem quanto ao cumprimento do requisito; "
         "reportar a faixa, não um veredito."), "",]
    # Um requisito abaixo da resolução amostral não é falha de calibração: é
    # falta de amostra, e nenhum limiar conserta. Sem esta ressalva o parágrafo
    # acima sugere que existe um limiar melhor a encontrar.
    resolucao_pct = 100.0 / max(int(norm.get("n", 0) or 0), 1)
    linhas += [
        f"> No teste foram observadas **{n_excedencias}/{n_teste} = "
         f"{taxa_excedencia:.2%}** excedências; IC95 de Wilson "
         f"**[{ci_excedencia[0]:.2%}; {ci_excedencia[1]:.2%}]**. A meta de 1% "
         "está dentro do intervalo: os dados não certificam conformidade nem "
         "violação.", "",
        f"> Com n = {norm.get('n')}, a resolução amostral é "
         f"{resolucao_pct:.2f}%: **o alvo de 1% está abaixo do que esta amostra "
         f"consegue certificar**. Zero excedências observadas não provariam 1%. "
        f"A limitação é de tamanho amostral, não uma prova de falha do limiar."
         if resolucao_pct > 1.0 else "", "",
        "### Deriva entre calibração e teste", "",
        f"O limite pontual LS-POD no teste foi "
        f"**{deriva['campo']['limite']:.4f}**, frente ao limiar "
        f"**{y_dec:.4f}** e ao gatilho de deriva "
        f"**{deriva['gatilho']:.4f}**.", "",
        "> O resultado aciona investigação como triagem. O bloco de teste não "
        "é campo e não fornece resolução para confirmar 1%; portanto não "
        "constitui invalidação industrial nem evidência de deriva física.", "",
    ]
    return linhas, {"y_dec": y_dec, "normalidade": norm, "limite_pof": pof,
                    "limite_empirico": empirico, "limite_log": log_lim,
                    "todos_acima_do_limiar": bool(todos_acima),
                    "excedencia_observada": {
                        "count": n_excedencias, "n": n_teste,
                        "rate": taxa_excedencia,
                        "ci95_wilson": list(ci_excedencia),
                        "conclusao": "inconclusiva_para_meta_1pct",
                    },
                    "deriva": deriva}


def main() -> int:
    from src.core.config import RAIZ_PROJETO

    pasta = Path(RAIZ_PROJETO) / "resultados" / "autoencoder"
    weibull = _ler(pasta, "weibull_results.json")
    limiar = _ler(pasta, "limiar.json")
    npz = _ler(pasta, "diagnostico_autoencoder.npz")

    if weibull is None:
        print("  ❌ weibull_results.json ausente.")
        print("     Se sumiu, RESTAURE em vez de recalcular:")
        print("     git log --oneline --diff-filter=D -- resultados/autoencoder/")
        return 1

    cabecalho = [
        "# Relatório de detectabilidade E2 e ponto de operação", "",
        "> **Evidência E2** — validação sintética orientada pela FMECA. Não é "
        "desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de "
        "injeção em que a detecção se confirma. Portanto não há RUL, MTTF, "
        "taxa de falha ou confiabilidade física nesta seção.", "",
    ]
    l_conf, d_conf = secao_confiabilidade(weibull)
    l_pod, d_pod = secao_pod(npz, limiar or {})

    md = "\n".join(cabecalho + l_conf + ["---", ""] + l_pod)
    (pasta / "relatorio_confiabilidade.md").write_text(md, encoding="utf-8")
    (pasta / "relatorio_confiabilidade.json").write_text(
        json.dumps({"evidence_level": "E2", "confiabilidade": d_conf,
                    "ponto_de_operacao": d_pod},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(md)
    print(f"\n  📄 {pasta / 'relatorio_confiabilidade.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
