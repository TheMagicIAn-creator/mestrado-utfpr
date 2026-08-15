"""
weibull_por_modelo.py — Al IAdo PV

Detectabilidade E2 (`a_det` → Weibull → S_D, f_D, h_D, papel de Weibull) para
**qualquer** detector, não só o Autoencoder denso.

POR QUE ESTE MÓDULO EXISTE
==========================
O pesquisador pediu, literalmente, as curvas "pertinente a cada modelo, a cada
simulação" — comparando o AE denso com o AE-LSTM do Ibrahim.

Não dava. A cadeia de confiabilidade (`rul_weibull_execucao`) carrega UM
checkpoint fixo de `resultados/autoencoder/` e itera sobre `FALHAS` (Contator
AC, IGBT, Fusível AC). O laço é por COMPONENTE, nunca por MODELO. E o AE-LSTM
nunca gerou `a_det`: grep por "weibull" nos quatro macro-códigos e em
`modelos_anomalia` devolve zero. Ele produzia AUC, SMD e matriz de confusão, e
parava aí.

O que faltava não era matemática — `confiabilidade.py` já tem as quatro funções
e `graficos_rul.py` já as desenha. Faltava a varredura de magnitude aceitar um
detector arbitrário.

COMO RESOLVE
============
`rul_weibull.gerar_a_det` ganhou um parâmetro `scorer`: um
``Callable[[list[DataFrame]], np.ndarray]``. É a MESMA interface que
`macro_comum` já exige dos dois métodos comparados — então o denso e o AE-LSTM
entram por aqui sem adaptador nenhum.

Este módulo só orquestra: varre as trajetórias por falha, ajusta a Weibull com
`rul_weibull.ajustar_weibull` (censura intervalar na grade, já implementada) e
devolve o bloco pronto. Nenhuma fórmula é reimplementada aqui — duplicar
`confiabilidade.py` seria criar uma segunda fonte para a mesma curva, que é
exatamente o problema que o projeto já teve com `confiabilidade_fisica_v2`.

O QUE ELE NÃO FAZ
=================
Não promove nada a E3, e não converte magnitude em tempo. `a_det` continua sendo
fração da assinatura nominal, `S_D` é não detecção e `h_D` é intensidade de
primeiro cruzamento — nunca confiabilidade ou taxa de falha física.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import numpy as np

from src.ml.injecao_falhas import FALHAS
from src.ml.rul_weibull import (
    N_STEPS,
    ajustar_weibull,
    classificar_desfechos,
    gerar_a_det,
    selecionar_trajetorias_holdout,
)

# Semente base das trajetórias. Fixa de propósito: a comparação entre modelos só
# é justa se os DOIS virem exatamente as mesmas realizações de ruído injetado.
SEED_TRAJETORIAS = 20_000


def trajetorias_por_falha(
    scorer,
    limiar: float,
    janelas: list,
    falha_id: str,
    n_steps: int = N_STEPS,
    seed_base: int = SEED_TRAJETORIAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Varre a magnitude em cada janela e devolve ``(a_det, detectou)``.

    Uma trajetória por janela do holdout: a janela representa um ativo, e a
    assinatura da falha cresce sobre ELA. Trocar de janela a cada passo
    misturaria degradação com variação operacional.
    """
    a_dets, eventos = [], []
    for i, janela in enumerate(janelas):
        a_det, detectou = gerar_a_det(
            janela,
            modelo=None, scaler=None, device=None, colunas_feat=None,
            limiar=float(limiar),
            tipo_falha=falha_id,
            n_steps=int(n_steps),
            seed=seed_base + i,
            scorer=scorer,
        )
        a_dets.append(a_det)
        eventos.append(detectou)
    return np.asarray(a_dets, dtype=float), np.asarray(eventos, dtype=bool)


def detectabilidade_do_modelo(
    nome: str,
    scorer,
    limiar: float,
    janelas: list,
    n_steps: int = N_STEPS,
    n_max_trajetorias: int | None = None,
    n_boot: int = 0,
) -> dict:
    """Bloco de detectabilidade E2 completo para UM detector.

    Devolve, por falha da FMECA: os `a_det`, os desfechos (detectada ×
    indetectável no teto) e o ajuste Weibull com as curvas amostradas — o mesmo
    formato que `rul_weibull` já produz para o modelo proposto, para que
    gráficos e relatório não precisem saber de qual modelo vieram.

    `n_boot=0` por padrão: o bootstrap é caro e a comparação entre modelos se
    faz primeiro pelos pontos. Quem quiser IC passa o valor.
    """
    selecionadas = selecionar_trajetorias_holdout(janelas, n_max_trajetorias)
    passo = 1.0 / (max(int(n_steps), 2) - 1)

    por_falha = {}
    for falha in FALHAS:
        fid = falha["id"]
        a_det, eventos = trajetorias_por_falha(
            scorer, limiar, selecionadas, fid, n_steps=n_steps
        )
        ajuste = ajustar_weibull(
            a_det, eventos, n_boot=n_boot, passo_grade=passo
        )
        por_falha[fid] = {
            "nome": falha["nome"],
            "npr": falha["npr"],
            "a_dets": a_det.tolist(),
            "eventos_observados": eventos.tolist(),
            "desfechos": classificar_desfechos(a_det, eventos),
            "weibull": ajuste,
        }

    return {
        "modelo": nome,
        "limiar": float(limiar),
        "n_trajetorias": len(selecionadas),
        "n_steps": int(n_steps),
        "a_det_por_passo": passo,
        "evidence_level": "E2",
        "eixo_nao_e_tempo": True,
        "nota": (
            "a_det é fração da assinatura nominal injetada, não tempo. S_D(a) é "
            "probabilidade de AINDA NÃO detectar e h_D(a) é intensidade de "
            "primeiro cruzamento — nenhuma das duas é confiabilidade ou taxa de "
            "falha física. Comparável entre modelos porque as trajetórias usam "
            "as mesmas janelas e as mesmas realizações de ruído."
        ),
        "falhas": por_falha,
    }


def comparar_detectabilidade(modelos: list[dict]) -> dict:
    """Junta os blocos de dois ou mais modelos numa tabela comparável.

    `modelos` é uma lista de saídas de `detectabilidade_do_modelo`. A comparação
    é por falha, nos marcos que decidem manutenção — a10 e a mediana — mais o
    POD_mon no teto, que é o elo com a curva POD e com o `D_mon` da FMECA.
    """
    if not modelos:
        raise ValueError("comparar_detectabilidade exige ao menos um modelo")

    linhas = []
    for bloco in modelos:
        for fid, dados in bloco["falhas"].items():
            w = dados["weibull"]
            linhas.append({
                "modelo": bloco["modelo"],
                "falha": dados["nome"],
                "falha_id": fid,
                "npr": dados["npr"],
                "n_trajetorias": bloco["n_trajetorias"],
                "detectadas": dados["desfechos"]["n_detectadas"],
                "pod_mon_no_teto": dados["desfechos"]["pod_mon_no_teto"],
                "ajuste_convergiu": bool(w.get("fit_converged")),
                # Menor é melhor: o detector confirma com menos assinatura.
                "a10": w.get("b10"),
                "a_det_mediana": w.get("vida_mediana") or w.get("mediana"),
                "beta": w.get("beta"),
                "eta": w.get("eta"),
                "evidence_level": "E2",
            })
    return {"linhas": linhas, "n_modelos": len(modelos)}
