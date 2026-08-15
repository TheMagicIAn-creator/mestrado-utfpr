"""
macro_comum.py — Al IAdo PV

Avaliação COMUM (E2, orientada pela FMECA) e saída UNIFORME para os dois
macro-códigos de comparação:
  - src/ml/macro_proposto.py  → nosso método (AE denso + MSE p99)
  - src/ml/macro_ibrahim.py   → método do Ibrahim (AE-LSTM temporal)

Ideia: cada macro fornece apenas um SCORER — uma função que recebe uma lista de
janelas (DataFrames de sinal) e devolve um score de anomalia por janela (maior =
mais anômalo). Toda a avaliação (injeção FMECA por severidade, detecção, AUC) e
todos os gráficos/tabelas saem daqui, no MESMO formato — então a comparação é
maçã-com-maçã e a análise é uma só. Substitui o framework de experimentos, que
gerava tabelas de 33 colunas e matrizes enganosas.

Nada de torch aqui no topo: os imports pesados são LOCAIS, para as funções de
tabela/gráfico serem testáveis sem o dataset.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ============================================================
# AVALIAÇÃO (scorer plugável)
# ============================================================

PURGA = 2   # janelas descartadas na fronteira (sobreposição de 50% nas features)


# Fração do bloco de calibração usada para AJUSTAR o limiar; o restante mede o
# FP fora da amostra.
#
# CONSTANTE, não configurável, e isso é deliberado: a comparação com o Ibrahim
# só vale porque os dois métodos passam pelo MESMO protocolo E2. Um valor por
# ambiente permitiria rodar os dois lados em protocolos diferentes sem que o
# artefato registrasse — e 1,0 esvaziaria o bloco de validação, fazendo a
# calibração cair em silêncio no percentil mais conservador.
FRACAO_AJUSTE_LIMIAR = 0.8


def dividir_calibracao_avaliacao(janelas: list, frac_calib: float = 0.4,
                                 purga: int = PURGA) -> tuple[list, list]:
    """Separa o holdout saudável em bloco de CALIBRAÇÃO e bloco de AVALIAÇÃO.

    Blocos contíguos no tempo, com purga na fronteira. O limiar é fixado no
    primeiro; FP, AUC e injeção são medidos SÓ no segundo — que o método nunca
    viu. Sem isso, o FP reportado é ~alvo por construção (in-sample) e não
    evidencia generalização.
    """
    n = len(janelas)
    n_cal = max(1, int(n * frac_calib))
    return janelas[:n_cal], janelas[min(n, n_cal + purga):]


def calibrar_limiar(scorer, janelas_calib: list) -> tuple[float, float]:
    """Fixa o limiar operacional de UM método no bloco de calibração.

    Devolve ``(limiar, percentil)``. O limiar sai do bloco de CALIBRAÇÃO e é
    auto-ajustado ao FP alvo contra a cauda do próprio bloco; nunca enxerga o
    bloco de avaliação.

    Está numa função própria porque tem DOIS consumidores — `avaliar_deteccao`
    (AUC/SMD por severidade) e `macro_weibull` (varredura de magnitude por
    modelo). Duplicar a regra faria a mesma comparação sair com limiares
    calibrados por critérios diferentes, e a diferença entre modelos passaria a
    medir a discrepância entre as duas cópias.

    A calibração é por MÉTODO, e isso é obrigatório: o MSE de um autoencoder
    denso e o de um AE-LSTM vivem em escalas distintas. Um limiar único
    compararia unidades diferentes.
    """
    from src.ml import escore_anomalia as ea

    s_cal = np.asarray(scorer(janelas_calib), dtype=float)
    if len(s_cal) >= 10:
        corte = max(1, int(len(s_cal) * FRACAO_AJUSTE_LIMIAR))
        # fp_alvo_pct=None => usa ea.FP_ALVO. Antes havia um 1.0 literal aqui,
        # que ignorava AL_IADO_ESCORE_FP_ALVO e tornava a varredura de
        # calibração impossível de conduzir pelo macro.
        return ea.limiar_por_fp_alvo(s_cal[:corte], s_cal[corte:])
    return float(np.percentile(s_cal, 99)), 99.0


# Quantas vezes o limiar do macro pode divergir do limiar do pipeline antes de
# ser suspeito. Vinte é folgado de propósito: bloco de calibração diferente e
# alvo de FP auto-ajustado mudam o valor legitimamente. O que este guarda pega é
# ordem de grandeza — o caso real foi 61.000×.
FATOR_SUSPEITO_LIMIAR = 20.0


def conferir_escala_do_limiar(nome: str, limiar: float,
                              pasta_ae: Path | None = None) -> str | None:
    """Alerta quando o limiar do macro foge da escala do limiar do pipeline.

    POR QUE EXISTE
    ==============
    Em 15/08/2026 os dois macros pontuaram com features CRUAS num scaler
    ajustado sobre features normalizadas por comissionamento. O limiar do mesmo
    autoencoder saiu 0,8577 no pipeline e 52.577,8 no macro. Nada cruzou, e a
    tabela publicou POD_mon = 0,00 nas três falhas, para os DOIS modelos — com
    cara de resultado científico ("nenhum detector enxerga"), quando era defeito
    de representação de entrada.

    Nenhum teste de CI pega isso: depende do modelo treinado e do dataset, que
    só existem na máquina do pesquisador. Então o alerta mora no runtime, onde
    o dado está. Devolve a mensagem (ou None) em vez de levantar — um limiar
    fora de escala é fortíssimo indício, não prova, e abortar impediria uma
    investigação legítima.

    Só vale para o método PROPOSTO: o AE-LSTM tem escala própria e nenhum
    limiar de referência publicado.
    """
    import json

    pasta = Path(pasta_ae) if pasta_ae else (
        Path(__file__).parent.parent.parent / "resultados" / "autoencoder"
    )
    arquivo = pasta / "limiar.json"
    if not arquivo.is_file():
        return None
    try:
        referencia = float(json.loads(arquivo.read_text(encoding="utf-8"))["limiar"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not (referencia > 0 and float(limiar) > 0):
        return None

    razao = max(float(limiar) / referencia, referencia / float(limiar))
    if razao < FATOR_SUSPEITO_LIMIAR:
        return None
    return (
        f"⚠️  {nome}: limiar {limiar:.4g} contra {referencia:.4g} do pipeline "
        f"({razao:.0f}× de diferença). Escala assim indica que a representação "
        f"de entrada divergiu da canônica — confira se a normalização de "
        f"comissionamento (gpvs_principal.vetores_de_janelas) está sendo "
        f"aplicada antes do scaler."
    )


def avaliar_deteccao(nome: str, cor: str, scorer, janelas_calib: list,
                     janelas_aval: list, seed: int = 42) -> dict:
    """Avalia UM método sobre a injeção FMECA por severidade.

    scorer: callable(list[DataFrame]) -> np.ndarray de scores (1 por janela).
    `janelas_calib` fixa o limiar; `janelas_aval` (disjunto, nunca visto) dá o
    FP, os negativos do AUC e a base da injeção. Retorna dict uniforme.
    """
    from sklearn.metrics import roc_auc_score

    from src.ml.injecao_falhas import ALVO_SMD, FALHAS, FUNCOES_FALHA, SEVERIDADES
    from src.ml.estatistica import intervalo_wilson

    limiar, percentil = calibrar_limiar(scorer, janelas_calib)
    s_sau = np.asarray(scorer(janelas_aval), dtype=float)    # mede FP (não visto)
    fp = float((s_sau > limiar).mean() * 100.0)   # FP HONESTO: bloco não visto

    res = {"nome": nome, "cor": cor, "limiar": float(limiar),
           "percentil": float(percentil), "fp_pct": fp,
           "n_calib": len(janelas_calib), "n_aval": len(janelas_aval),
           "severidades": [float(s) for s in SEVERIDADES], "falhas": {}}

    scorer_cache_sev1: dict = {}      # escores em severidade 1.0, por falha
    for falha in FALHAS:
        fid, fn = falha["id"], FUNCOES_FALHA[falha["id"]]
        por_sev, scores_altos = {}, []
        for sev in SEVERIDADES:
            inj = [
                fn(j, float(sev), seed=20_000 + i) if fid == "contator_ac"
                else fn(j, float(sev))
                for i, j in enumerate(janelas_aval)
            ]
            s = np.asarray(scorer(inj), dtype=float)
            if sev >= 0.5:
                scores_altos.append(s)
            if float(sev) == 1.0:
                scorer_cache_sev1[fid] = s
            det = s > limiar
            lo, hi = intervalo_wilson(int(det.sum()), len(det))
            # TPR em FPR FIXO (10%), POR SEVERIDADE — ponto de operação lido da
            # própria ROC, sem depender do limiar calibrado (degenerado com
            # calibração pequena). É esta a curva comparável entre métodos.
            corte_fpr = float(np.quantile(s_sau, 0.90)) if len(s_sau) else float("inf")
            tpr_sev = float((s > corte_fpr).mean())
            por_sev[float(sev)] = {
                "taxa": float(det.mean()), "ci_low": lo, "ci_high": hi,
                "tpr_fpr10": tpr_sev,
                "erro_mediano": float(np.median(s)), "atinge_smd": bool(det.mean() >= ALVO_SMD),
            }
        # AUC por falha: saudável (0) × severidades altas agregadas (1)
        s_alto = np.concatenate(scores_altos) if scores_altos else np.array([])
        if len(s_alto):
            y = np.r_[np.zeros(len(s_sau)), np.ones(len(s_alto))]
            auc = float(roc_auc_score(y, np.r_[s_sau, s_alto]))
        else:
            auc = float("nan")
        # SMD @FPR=10% — a MENOR severidade em que o método atinge o alvo de
        # detecção (95%) no ponto de operação de 10% de FPR. Discrimina onde os
        # métodos realmente diferem: em severidade 1.0 todos saturam em 100%.
        # Menor SMD = detecta a falha mais cedo = melhor.
        smd = None
        for sev in SEVERIDADES:
            if por_sev[float(sev)]["tpr_fpr10"] >= ALVO_SMD:
                smd = float(sev)
                break
        tpr_sev1 = por_sev[1.0]["tpr_fpr10"] if 1.0 in por_sev else float("nan")
        res["falhas"][fid] = {"nome": falha["nome"], "npr": falha["npr"],
                              "cor": falha["cor"], "auc": auc,
                              "smd_fpr10": smd, "tpr_fpr10": tpr_sev1,
                              "por_sev": por_sev}
    return res


# ============================================================
# SAÍDA UNIFORME — tabela ENXUTA (5 colunas, não 33)
# ============================================================

def _dados_severidade(falha: dict, severidade: float) -> dict:
    """Le uma severidade tanto do resultado em memoria quanto do JSON."""
    por_sev = falha["por_sev"]
    return por_sev.get(severidade, por_sev.get(str(severidade), {}))


def tabela_enxuta(resultados: list[dict]) -> str:
    """Tabela Markdown compacta, com as métricas COMPARÁVEIS entre métodos.

    AUC e SMD@FPR=10% independem do limiar — são as colunas de comparação. A
    detecção no limiar calibrado fica como referência operacional, mas com
    poucas janelas de calibração ela é conservadora demais (p99 de ~17 amostras
    ≈ máximo) e NÃO deve ser usada para ranquear métodos.
    """
    linhas = ["| Método | Falha (NPR) | AUC | SMD @FPR=10% | TPR @FPR=10%, sev=1.0 |",
              "|---|---|---|---|---|"]
    for r in resultados:
        for fid, f in r["falhas"].items():
            tpr = f.get("tpr_fpr10", float("nan"))
            smd = f.get("smd_fpr10")
            smd_txt = f"{smd:.2f}" if smd is not None else "não atinge"
            linhas.append(
                f"| {r['nome']} | {f['nome']} (NPR={f['npr']}) | "
                f"{f['auc']:.3f} | {smd_txt} | {tpr * 100:.0f}% |"
            )
    return "\n".join(linhas)


def salvar_saidas(resultados: list[dict], pasta: Path, prefixo: str = "comparacao") -> dict:
    """Grava tabela (md+csv) e o gráfico de detecção por severidade (uniforme)."""
    import csv
    import json

    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)

    (pasta / f"{prefixo}_tabela.md").write_text(tabela_enxuta(resultados), encoding="utf-8")
    with (pasta / f"{prefixo}_tabela.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "metodo", "falha", "npr", "auc", "smd_fpr10",
            "tpr_fpr10_sev1", "deteccao_limiar_sev1", "fp_pct",
        ])
        for r in resultados:
            for fid, f in r["falhas"].items():
                w.writerow([r["nome"], f["nome"], f["npr"], round(f["auc"], 4),
                            f.get("smd_fpr10"),
                            round(f.get("tpr_fpr10", float("nan")), 4),
                            round(_dados_severidade(f, 1.0).get("taxa", float("nan")), 4),
                            round(r["fp_pct"], 2)])
    (pasta / f"{prefixo}_resultado.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    caminho_png = plotar_deteccao_severidade(resultados, pasta, prefixo)
    return {
        "tabela_md": pasta / f"{prefixo}_tabela.md",
        "tabela_csv": pasta / f"{prefixo}_tabela.csv",
        "resultado_json": pasta / f"{prefixo}_resultado.json",
        "grafico": caminho_png,
    }


def plotar_deteccao_severidade(resultados: list[dict], pasta: Path,
                               prefixo: str = "comparacao") -> Path:
    """Detecção × severidade, um painel por falha, com os métodos sobrepostos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.ml.estilo_graficos import COR_ALERTA, TAM, aplicar_estilo, salvar_figura

    aplicar_estilo()
    falhas = list(resultados[0]["falhas"].keys())
    sevs = resultados[0]["severidades"]
    marcadores = ["o-", "s--", "^:", "d-."]
    fig, axes = plt.subplots(1, len(falhas), figsize=TAM["painel_3"],
                             layout="constrained", sharey=True)
    if len(falhas) == 1:
        axes = [axes]
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    fig.suptitle("Comparação da detecção por severidade a FPR=10% (E2)", y=1.01)
    for ax, fid in zip(axes, falhas):
        info = resultados[0]["falhas"][fid]
        for j, r in enumerate(resultados):
            f = r["falhas"][fid]
            # curva no ponto de operação FPR=10% (comparável); cai para a taxa
            # no limiar se o resultado for de uma versão anterior do pipeline.
            y = []
            for severidade in sevs:
                dados = _dados_severidade(f, severidade)
                y.append(dados.get("tpr_fpr10", dados["taxa"]) * 100)
            ax.plot(sevs, y, marcadores[j % len(marcadores)],
                    label=f"{r['nome']} (AUC={f['auc']:.2f})")
        ax.axhline(95, color=COR_ALERTA, linestyle="--", linewidth=1.4, label="Alvo SMD 95%")
        ax.set_title(f"{info['nome']} (NPR={info['npr']})", fontsize=10)
        ax.set_xlabel("Severidade")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Taxa de detecção (%)")
    axes[0].legend(fontsize=8)
    caminho = pasta / f"{prefixo}_deteccao_severidade.png"
    salvar_figura(fig, caminho,
                  "E2 sintético (injeção FMECA no sinal); ponto de operação FPR=10% da ROC; mesmo banco para os dois métodos.")
    return caminho
