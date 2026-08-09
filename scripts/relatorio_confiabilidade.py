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


def secao_confiabilidade(weibull: dict) -> tuple[list[str], dict]:
    """Curvas, marcos e a leitura de β — com a ressalva do intervalo."""
    linhas = ["## Confiabilidade por modo de falha", ""]
    dados = {}
    for fid, bloco in weibull.get("falhas", {}).items():
        w = bloco.get("weibull") or {}
        if not w.get("fit_converged"):
            linhas += [f"### {bloco.get('nome', fid)}", "",
                       "Ajuste não convergiu — sem curva de confiabilidade.", ""]
            continue
        beta, eta = float(w["beta"]), float(w["eta"])
        ic = w.get("beta_ci95") or [None, None]
        tem_ic = ic[0] is not None
        leitura = cf.classificar_forma(beta, tuple(ic) if tem_ic else None)
        marcos = cf.marcos(beta, eta)
        horizonte = float(w.get("rul_restrita_horizonte") or 0.0)

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
            "| Marco | Magnitude de injeção | R nesse ponto |",
            "|---|--:|--:|",
            f"| B1 (1% detectado) | {marcos['b1']:.2f} | "
            f"{cf.confiabilidade(marcos['b1'], beta, eta):.3f} |",
            f"| B10 (10% detectado) | {marcos['b10']:.2f} | {cf.confiabilidade(marcos['b10'], beta, eta):.3f} |",
            f"| mediana | {marcos['vida_mediana']:.2f} | {cf.confiabilidade(marcos['vida_mediana'], beta, eta):.3f} |",
            f"| η (vida característica) | {eta:.2f} | 0.368 |",
            f"| MTTF | {marcos['mttf']:.2f} | {cf.confiabilidade(marcos['mttf'], beta, eta):.3f} |",
            "",
            f"**Leitura de β.** {leitura['leitura']}",
            "",
        ]
        if not leitura["conclusivo"]:
            linhas += ["> ⚠️ A afirmação de regime NÃO se sustenta neste caso.", ""]
        if marcos["b10"] < marcos["mttf"]:
            linhas += [
                f"> **B10 ({marcos['b10']:.1f}) < MTTF ({marcos['mttf']:.1f}).** "
                "A distribuição é assimétrica: a média fica acima de boa parte "
                "da população, e por isso B10/B1 são melhores indicadores de "
                "decisão de manutenção que o MTTF.", ""]
        if horizonte:
            linhas += [
                f"> Observação vai até {horizonte:.1f}; além disso as curvas "
                "são extrapolação do modelo, não dado.", ""]

        dados[fid] = {"beta": beta, "eta": eta, "marcos": marcos,
                      "interpretacao": leitura, "R_em_marcos": r_em,
                      "horizonte_observado": horizonte}
    return linhas, dados


def secao_pod(npz, limiar_json: dict) -> tuple[list[str], dict]:
    """Critério de viabilidade e deriva — com a hipótese conferida antes."""
    linhas = ["## Ponto de operação sob o critério LS-POD", ""]
    if npz is None or not limiar_json:
        return linhas + ["Artefatos ausentes.", ""], {}

    y_dec = float(limiar_json.get("limiar_localizado") or limiar_json["limiar"])
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

    linhas += [
        f"Limiar operacional adotado: **{y_dec:.4f}**", "",
        "### A hipótese, antes do número", "",
        f"O método assume normalidade do lado saudável. Shapiro-Wilk no escore "
        f"bruto: p = {norm['shapiro_bruto']['p']:.2e}; em log: "
        f"p = {norm['shapiro_log']['p']:.2e}. Assimetria "
        f"{norm['assimetria']:+.2f}, curtose {norm['curtose_excesso']:+.2f}.", "",
        ("**Hipótese violada.** Por isso o mesmo quantil é estimado por três "
         "caminhos independentes abaixo — se os três concordarem, a conclusão "
         "não depende dela."
         if not norm["vale"] else
         f"**Hipótese satisfeita** na escala **{norm['melhor_escala']}**. Os "
         "três estimadores abaixo continuam sendo reportados: quando eles "
         "concordam sob hipótese válida, a concordância confirma o método; "
         "quando divergem, é sinal de cauda que o teste de normalidade não "
         "pegou."), "",
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
        (f"> **Os {len(acima)} estimadores ficam acima do limiar adotado.** Pelo "
         "critério LS-POD, o requisito de falso positivo de 1% **não é cumprido "
         "no bloco de teste**. A conclusão não depende da hipótese de "
         "normalidade: o quantil empírico, que não assume distribuição, leva ao "
         "mesmo lugar."
         + ("" if norm["vale"] else
            " Isso importa aqui, porque a normalidade está VIOLADA — sem o "
            "estimador empírico o veredito seria discutível.")
         if todos_acima else
         "> Os estimadores divergem quanto ao cumprimento do requisito; "
         "reportar a faixa, não um veredito."), "",]
    # Um requisito abaixo da resolução amostral não é falha de calibração: é
    # falta de amostra, e nenhum limiar conserta. Sem esta ressalva o parágrafo
    # acima sugere que existe um limiar melhor a encontrar.
    resolucao_pct = 100.0 / max(int(norm.get("n", 0) or 0), 1)
    linhas += [
        (f"> ⚠️ Com n = {norm.get('n')}, a resolução amostral é "
         f"{resolucao_pct:.2f}%: **o alvo de 1% está abaixo do que esta amostra "
         f"consegue certificar**. Zero excedências observadas não provariam 1%. "
         f"O requisito não falha por calibração — falha por tamanho de amostra."
         if resolucao_pct > 1.0 else ""), "",
        "### Deriva entre calibração e teste", "",
        deriva["leitura"], "",
    ]
    return linhas, {"y_dec": y_dec, "normalidade": norm, "limite_pof": pof,
                    "limite_empirico": empirico, "limite_log": log_lim,
                    "todos_acima_do_limiar": bool(todos_acima),
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
        "# Relatório de confiabilidade e ponto de operação", "",
        "> **Evidência E2** — validação sintética orientada pela FMECA. Não é "
        "desempenho de campo (E3). O eixo NÃO é tempo físico: é a magnitude de "
        "injeção em que a detecção se confirma.", "",
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
