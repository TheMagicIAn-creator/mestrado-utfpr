"""
comparacao_literatura.py - Al IAdo PV
Comparação do MÉTODO PROPOSTO (Autoencoder do pipeline principal) com a
literatura (Francisti e Ibrahim), em pé de igualdade.

Por que este módulo existe
--------------------------
Os AUCs do pipeline principal (validacao_report.json) e dos experimentos
por artigo NÃO são diretamente comparáveis: o pipeline injeta falha no
SINAL bruto (E2) e os experimentos injetam no espaço de FEATURES (E1),
com agregações diferentes. Comparar esses números lado a lado seria
indefensável em banca.

A solução: pontuar o Autoencoder JÁ TREINADO no MESMO banco de teste dos
experimentos (preparar_dados_anomalia, seed=42 — split temporal com purga
e injeção FMEA idênticos). Aí todos os métodos veem exatamente as mesmas
janelas e o mesmo ground truth, e o AUC vira genuinamente comparável.

Duplo reporte (honestidade metodológica):
- E2 nativo — a validação real do método (injeção no sinal), reportada à
  parte a partir do validacao_report.json;
- E1 banco comum — este módulo; é um teste MAIS FRACO para o Autoencoder
  (features prontas, sem o efeito completo da falha no sinal), usado só
  para o "apples-to-apples" contra a literatura.

Regras:
- NUNCA treina nada: usa o modelo salvo em resultados/autoencoder/. Se
  não existir, avisa "rode o pipeline principal primeiro".
- Lê os resultado.json dos experimentos; se faltarem, avisa "rode os
  experimentos primeiro" (inclui só o que existir, com aviso).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.core.config import RAIZ_PROJETO
from src.core.formatacao import fmt_metrica, tabela_markdown

PASTA_AE = RAIZ_PROJETO / "resultados" / "autoencoder"
PASTA_EXPERIMENTOS = RAIZ_PROJETO / "resultados" / "experimentos"
PASTA_COMPARACAO = RAIZ_PROJETO / "resultados" / "comparacao"

NOME_METODO = "Autoencoder (método proposto)"
_EXPERIMENTOS = ("francisti", "ibrahim")
_ARTEFATOS_AE = ("modelo_autoencoder.pt", "scaler.pkl", "limiar.json")


def _artefatos_ae_faltando() -> list[str]:
    return [nome for nome in _ARTEFATOS_AE if not (PASTA_AE / nome).exists()]


# ============================================================
# PONTUAÇÃO DO AUTOENCODER NO BANCO COMUM
# ============================================================

def _pontuar_autoencoder(dados: dict) -> dict:
    """
    Erro de reconstrução do Autoencoder salvo sobre dados["X_te"] do banco
    comum. O X_te vem escalonado pelo scaler DO BANCO; revertemos com
    inverse_transform, remapeamos as colunas para a ordem do Autoencoder
    (coluna ausente vira 0.0, a mesma convenção do pipeline) e aplicamos
    o scaler DO AUTOENCODER.
    """
    import numpy as np
    import torch

    from src.core.seguranca import carregar_pickle_com_sidecar
    from src.ml.autoencoder import Autoencoder

    checkpoint = torch.load(PASTA_AE / "modelo_autoencoder.pt",
                            map_location="cpu", weights_only=False)
    scaler_ae = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
    colunas_feat = checkpoint["colunas_feat"]

    modelo = Autoencoder(checkpoint["n_features"], checkpoint["latente_dim"])
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    # banco comum → espaço bruto → ordem de colunas do Autoencoder
    X_bruto = dados["scaler"].inverse_transform(dados["X_te"])
    nomes = list(dados["nomes"])
    idx = {nome: j for j, nome in enumerate(nomes)}
    n = len(X_bruto)
    X_ae = np.zeros((n, len(colunas_feat)), dtype=np.float64)
    ausentes = []
    for k, col in enumerate(colunas_feat):
        j = idx.get(col)
        if j is None:
            ausentes.append(col)      # fica 0.0 (convenção do pipeline)
        else:
            X_ae[:, k] = X_bruto[:, j]

    vnorm = scaler_ae.transform(X_ae).astype(np.float32)
    with torch.inference_mode():
        x = torch.from_numpy(vnorm)
        erros = ((x - modelo(x)) ** 2).mean(dim=1).numpy()

    from sklearn.metrics import roc_auc_score

    y_te = np.asarray(dados["y_te"]).astype(int)
    auc = float(roc_auc_score(y_te, erros))

    # AUC por família: normais + anômalas daquela família
    tipos = np.asarray(dados["tipos_te"])
    n_norm = int((y_te == 0).sum())
    auc_por_falha = {}
    for fam in dict.fromkeys(tipos.tolist()):
        mask_anom = np.r_[np.zeros(n_norm, dtype=bool), tipos == fam]
        mask = (y_te == 0) | mask_anom
        auc_por_falha[str(fam)] = float(roc_auc_score(y_te[mask], erros[mask]))

    return {
        "auc": auc,
        "auc_por_falha": auc_por_falha,
        "colunas_ausentes": ausentes,
        "n_te": int(n),
    }


# ============================================================
# LEITURA DOS EXPERIMENTOS E DO E2 NATIVO
# ============================================================

def _linhas_experimentos(n_te_banco: int) -> tuple[list[dict], list[str]]:
    """Extrai (linhas AUC, avisos) dos resultado.json de Francisti/Ibrahim."""
    linhas, avisos = [], []
    for key in _EXPERIMENTOS:
        arq = PASTA_EXPERIMENTOS / key / "resultado.json"
        if not arq.exists():
            avisos.append(
                f"Experimento '{key}' sem resultado salvo — rode-o primeiro "
                f"('rode o experimento do {key}')."
            )
            continue
        d = json.loads(arq.read_text(encoding="utf-8"))
        for nome, m in (d.get("modelos") or {}).items():
            if not isinstance(m, dict) or not m.get("disponivel", False):
                continue
            if not isinstance(m.get("auc"), (int, float)):
                continue
            if isinstance(m.get("amostras"), int) and m["amostras"] != n_te_banco:
                avisos.append(
                    f"'{nome}' ({key}) foi avaliado em {m['amostras']} janelas, "
                    f"mas o banco comum atual tem {n_te_banco} — reexecute o "
                    "experimento para garantir o MESMO teste."
                )
            linhas.append({
                "metodo": nome,
                "papel": "baseline" if key == "francisti" else "concorrente",
                "fonte": d.get("referencia", key),
                "auc": float(m["auc"]),
                "evidencia": "E1",
            })
    return linhas, avisos


def _e2_nativo() -> dict | None:
    """AUCs do método no protocolo nativo (injeção no sinal, E2)."""
    arq = PASTA_AE / "validacao_report.json"
    if not arq.exists():
        return None
    d = json.loads(arq.read_text(encoding="utf-8"))
    por_caso = {
        k: float(v["auc_roc"])
        for k, v in d.items()
        if isinstance(v, dict) and isinstance(v.get("auc_roc"), (int, float))
    }
    return por_caso or None


# ============================================================
# APRESENTAÇÃO (tabela + gráfico)
# ============================================================

def _tabela_md(linhas: list[dict]) -> str:
    ordenadas = sorted(linhas, key=lambda x: x["auc"], reverse=True)
    corpo = []
    for li in ordenadas:
        nome = f"**{li['metodo']}**" if li["papel"] == "proposto" else li["metodo"]
        corpo.append([nome, li["papel"], fmt_metrica(li["auc"]),
                      li["evidencia"], li["fonte"]])
    return tabela_markdown(
        ["Método", "Papel", "AUC", "Evidência", "Fonte"], corpo,
        alinhamentos=["e", "e", "d", "e", "e"],
    )


def _grafico(linhas: list[dict], info_banco: str, destino: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    from src.ml.estilo_graficos import (
        COR_METODO, COR_NEUTRA, COR_REFERENCIA, COR_TEXTO_SEC,
        aplicar_estilo, rotular_barras, tam_barras_h,
    )

    aplicar_estilo()
    ordenadas = sorted(linhas, key=lambda x: x["auc"])  # maior AUC no topo
    nomes = [li["metodo"] for li in ordenadas]
    valores = [li["auc"] for li in ordenadas]
    cores = [COR_METODO if li["papel"] == "proposto" else COR_NEUTRA
             for li in ordenadas]

    fig, ax = plt.subplots(figsize=tam_barras_h(len(ordenadas)))
    barras = ax.barh(nomes, valores, color=cores, height=0.55)

    # linha do acaso com rótulo ancorado ACIMA da área de plot (x em dados,
    # y em fração dos eixos) — não colide com barras nem com o título
    ax.axvline(0.5, color=COR_REFERENCIA, ls="--", lw=1.2)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.5, 1.01, "acaso (0,5)", transform=trans,
            ha="center", va="bottom", fontsize=8, color=COR_TEXTO_SEC)

    ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUC-ROC no banco comum (maior = melhor)")
    # título à esquerda com respiro; subtítulo (banco) logo abaixo dele
    ax.set_title("Método proposto vs. literatura — mesmo teste, mesma injeção",
                 loc="left", pad=30)
    ax.text(0, 1.055, info_banco, transform=ax.transAxes,
            fontsize=8.5, color=COR_TEXTO_SEC, va="bottom")
    ax.grid(axis="y", visible=False)
    rotular_barras(ax, barras, horizontal=True)
    fig.savefig(destino)
    plt.close(fig)


# ============================================================
# ORQUESTRAÇÃO
# ============================================================

def comparar_com_literatura(progresso=None) -> dict:
    """
    Monta a comparação completa. Retorna dict com ok/mensagem, tabela_md,
    grafico (path relativo), avisos e o bloco de metodologia; grava
    resultados/comparacao/comparacao_literatura.{json,png}.
    """
    faltando = _artefatos_ae_faltando()
    if faltando:
        return {
            "ok": False,
            "mensagem": (
                "O método proposto ainda não tem modelo salvo "
                f"(faltam: {', '.join(faltando)}). Rode o pipeline principal "
                "primeiro ('rode o pipeline') e tente de novo — a comparação "
                "usa o Autoencoder JÁ treinado, nunca treina sozinha."
            ),
        }

    if progresso:
        progresso("Preparando o banco comum (split temporal + injeção FMEA)...")
    from src.ml.protocolos_artigos import preparar_dados_anomalia

    dados = preparar_dados_anomalia(seed=42)

    if progresso:
        progresso("Pontuando o Autoencoder no banco comum...")
    ae = _pontuar_autoencoder(dados)

    linhas = [{
        "metodo": NOME_METODO,
        "papel": "proposto",
        "fonte": "pipeline principal (este trabalho)",
        "auc": ae["auc"],
        "evidencia": "E1",
    }]
    linhas_exp, avisos = _linhas_experimentos(ae["n_te"])
    linhas.extend(linhas_exp)
    if not linhas_exp:
        return {
            "ok": False,
            "mensagem": (
                "Nenhum experimento da literatura tem resultado salvo. Rode "
                "'rode os experimentos de anomalia' primeiro — sem eles não "
                "há com o que comparar. "
                f"(AUC do método no banco comum já calculado: {ae['auc']:.3f})"
            ),
        }
    if ae["colunas_ausentes"]:
        avisos.append(
            f"{len(ae['colunas_ausentes'])} feature(s) do Autoencoder não "
            "existem no banco comum e entraram como 0.0 — verifique se "
            "features_ca e experimentos usam o MESMO parquet."
        )

    info_banco = (
        f"Banco comum E1: injeção FMEA no espaço de features, sev="
        f"{dados['injecao']['severidade']}, {ae['n_te']} janelas de teste, "
        f"split temporal com purga (seed 42)"
    )

    PASTA_COMPARACAO.mkdir(parents=True, exist_ok=True)
    arq_png = PASTA_COMPARACAO / "comparacao_literatura.png"
    _grafico(linhas, info_banco, arq_png)

    relatorio = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "E1",
        "evidence_note": (
            "Comparação no banco comum (injeção no espaço de features). "
            "Para o método proposto este é um teste MAIS FRACO que o E2 "
            "nativo (injeção no sinal) — reportado à parte em e2_nativo."
        ),
        "banco_comum": {**dados["split"], **dados["injecao"], "seed": 42},
        "linhas": linhas,
        "auc_por_falha_metodo": ae["auc_por_falha"],
        "e2_nativo": _e2_nativo(),
        "avisos": avisos,
    }
    (PASTA_COMPARACAO / "comparacao_literatura.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")

    from src.core.utils import to_project_relative_path

    return {
        "ok": True,
        "tabela_md": _tabela_md(linhas),
        "grafico": to_project_relative_path(arq_png),
        "info_banco": info_banco,
        "auc_por_falha_metodo": ae["auc_por_falha"],
        "e2_nativo": relatorio["e2_nativo"],
        "avisos": avisos,
    }
