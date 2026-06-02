"""
classificador_pv_infer.py — Al IAdo PV / Sprint 3 (6.2/6.4)

Persistência e inferência do classificador supervisionado PV Farms (domínio CC).
Treina o melhor modelo (Random Forest), salva os artefatos e permite classificar
uma amostra com VALIDAÇÃO de colunas, retornando classe, probabilidade, aviso de
domínio e versão do modelo.

NUNCA afirma diagnóstico CA: PV Farms é classificação de falhas CC conhecidas.

Artefatos (em resultados/classificacao_pv/):
    modelo_classificador.pkl, scaler.pkl, feature_columns.json,
    class_mapping.json, metricas.json
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.core.config import RAIZ_PROJETO

PASTA = Path(RAIZ_PROJETO) / "resultados" / "classificacao_pv"
NOMES_CLASSES = {
    0: "Normal", 1: "F1 - String", 2: "F2 - String-Terra", 3: "F3 - String-String",
}
AVISO_DOMINIO = (
    "Classificação de falhas CC (PV Farms). NÃO diagnostica falhas CA do "
    "inversor nem substitui o pipeline de anomalia (Paderborn)."
)


def treinar_e_salvar_de(X_tr, y_tr, X_te, y_te, pasta: Path = PASTA) -> dict:
    """Treina RF, salva artefatos e métricas. Genérico (testável com fixture)."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    from src.ml.experimentos_artigos import _metricas_classificacao

    pasta.mkdir(parents=True, exist_ok=True)
    colunas = list(X_tr.columns)
    classes = sorted(int(c) for c in set(y_tr))

    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr_s, y_tr)
    metricas = _metricas_classificacao(list(y_te), list(clf.predict(Xte_s)))

    joblib.dump(clf, pasta / "modelo_classificador.pkl")
    with open(pasta / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (pasta / "feature_columns.json").write_text(
        json.dumps(colunas, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "class_mapping.json").write_text(
        json.dumps({str(c): NOMES_CLASSES.get(c, f"Classe {c}") for c in classes},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "metricas.json").write_text(
        json.dumps({**{k: v for k, v in metricas.items() if k != "modelo"},
                    "evidence_level": "E1", "dominio": "CC", "modelo": "Random Forest"},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        from src.ml.pipeline import registrar_manifesto  # noqa: F401
    except Exception:
        pass
    return {"ok": True, "n_features": len(colunas), "classes": classes,
            "metricas": metricas}


def treinar_e_salvar(pasta: Path = PASTA) -> dict:
    """Treina/salva a partir do dataset PV Farms local."""
    from src.ml.classificador_pv import carregar_dados

    X_tr, y_tr, X_te, y_te = carregar_dados()
    return treinar_e_salvar_de(X_tr, y_tr, X_te, y_te, pasta)


def carregar(pasta: Path = PASTA) -> dict | None:
    """Carrega os artefatos do classificador, ou None se ausentes."""
    import joblib

    arq_modelo = pasta / "modelo_classificador.pkl"
    arq_cols = pasta / "feature_columns.json"
    if not (arq_modelo.exists() and arq_cols.exists()):
        return None
    with open(pasta / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return {
        "modelo": joblib.load(arq_modelo),
        "scaler": scaler,
        "colunas": json.loads(arq_cols.read_text(encoding="utf-8")),
        "classes": json.loads((pasta / "class_mapping.json").read_text(encoding="utf-8")),
    }


def classificar(amostra, pasta: Path = PASTA) -> dict:
    """
    Classifica uma amostra (dict {coluna: valor} ou lista ordenada).
    Valida colunas (ausentes/extras), retorna classe, probabilidade, aviso de
    domínio, versão e limitações. NÃO levanta exceção em erro de validação.
    """
    import numpy as np

    art = carregar(pasta)
    if art is None:
        return {"ok": False, "erro": "Classificador não treinado. "
                "Rode 'treinar_classificador_pv' primeiro.", "aviso": AVISO_DOMINIO}

    colunas = art["colunas"]
    if isinstance(amostra, dict):
        faltando = [c for c in colunas if c not in amostra]
        extras = [c for c in amostra if c not in colunas]
        if faltando:
            return {"ok": False, "erro": f"Colunas ausentes: {faltando[:8]}"
                    f"{'…' if len(faltando) > 8 else ''}", "aviso": AVISO_DOMINIO}
        if extras:
            return {"ok": False, "erro": f"Colunas extras não suportadas: "
                    f"{extras[:8]}", "aviso": AVISO_DOMINIO}
        vetor = [float(amostra[c]) for c in colunas]
    else:
        vetor = list(amostra)
        if len(vetor) != len(colunas):
            return {"ok": False, "erro": f"Esperado {len(colunas)} features, "
                    f"recebido {len(vetor)}.", "aviso": AVISO_DOMINIO}
        vetor = [float(v) for v in vetor]

    X = art["scaler"].transform(np.array(vetor, dtype=float).reshape(1, -1))
    clf = art["modelo"]
    pred = int(clf.predict(X)[0])
    probas = clf.predict_proba(X)[0]
    idx = list(clf.classes_).index(pred)
    classe_nome = art["classes"].get(str(pred), NOMES_CLASSES.get(pred, f"Classe {pred}"))
    return {
        "ok": True,
        "classe": pred,
        "classe_nome": classe_nome,
        "probabilidade": float(probas[idx]),
        "dominio": "CC",
        "aviso": AVISO_DOMINIO,
        "evidence_level": "E1",
        "limitacoes": ["importância de feature ≠ causalidade",
                       "treinado em PV Farms (CC), não no inversor real (CA)"],
    }
