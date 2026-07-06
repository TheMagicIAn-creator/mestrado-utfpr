"""
classificador_pv_infer.py — Al IAdo PV / Sprint 3 (6.2/6.4)

Persistência e inferência do classificador supervisionado PV Farms (domínio CC).
Treina o melhor modelo (Random Forest), salva os artefatos e permite classificar
uma amostra com VALIDAÇÃO de colunas, retornando classe, probabilidade, aviso de
domínio e versão do modelo.

NUNCA afirma diagnóstico CA: PV Farms é classificação de falhas CC conhecidas.

Artefatos (em resultados/classificacao_pv/):
    modelo_classificador.pkl, scaler.pkl, feature_columns.json,
    class_mapping.json, dataset_manifest.json, training_manifest.json,
    metricas.json, metricas.csv, matriz_confusao.png, importancia_features.png
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
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


def _dist_classes(y) -> dict:
    contagem = y.value_counts().sort_index() if hasattr(y, "value_counts") else {}
    return {str(k): int(v) for k, v in dict(contagem).items()}


def _nome_classe(classe) -> str:
    try:
        return NOMES_CLASSES.get(int(classe), f"Classe {classe}")
    except (TypeError, ValueError):
        return str(classe)


def _plotar_matriz_confusao(cm, classes: list[str], destino: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.ml.estilo_graficos import TAM, aplicar_estilo

    aplicar_estilo()
    fig, ax = plt.subplots(figsize=TAM["quadrado"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("PV Farms - matriz de confusao (E1)")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i, linha in enumerate(cm):
        for j, valor in enumerate(linha):
            ax.text(j, i, int(valor), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def _plotar_importancia_features(clf, colunas: list[str], destino: Path) -> None:
    if not hasattr(clf, "feature_importances_"):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from src.ml.estilo_graficos import aplicar_estilo, tam_barras_h

    aplicar_estilo()
    serie = pd.Series(clf.feature_importances_, index=colunas).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=tam_barras_h(len(serie)))
    serie.sort_values().plot(kind="barh", ax=ax, color="#2F80ED")
    ax.set_title("PV Farms - importancia global de features (E1)")
    ax.set_xlabel("Importancia relativa")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def treinar_e_salvar_de(
    X_tr,
    y_tr,
    X_te,
    y_te,
    pasta: Path = PASTA,
    source_paths: dict | None = None,
) -> dict:
    """Treina RF, salva artefatos e métricas. Genérico (testável com fixture)."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    from src.ml.experimentos_artigos import _metricas_classificacao
    from src.ml.proveniencia import sha256_arquivo

    pasta.mkdir(parents=True, exist_ok=True)
    colunas = list(X_tr.columns)
    classes = sorted(int(c) for c in set(y_tr))

    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr_s, y_tr)
    y_pred = clf.predict(Xte_s)
    metricas = _metricas_classificacao(list(y_te), list(y_pred))
    metricas_salvas = {
        **{k: v for k, v in metricas.items() if k != "modelo"},
        "evidence_level": "E1",
        "dominio": "CC",
        "modelo": "Random Forest",
    }

    source_paths = source_paths or {}
    dataset_manifest = {
        "dataset": "PV Farms",
        "dominio": "CC",
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "n_features": int(len(colunas)),
        "class_distribution_train": _dist_classes(y_tr),
        "class_distribution_test": _dist_classes(y_te),
        "train_sha256": sha256_arquivo(source_paths.get("train", "")) or "",
        "test_sha256": sha256_arquivo(source_paths.get("test", "")) or "",
        "feature_columns": colunas,
        "evidence_level": "E1",
        "limitations": [
            "dominio CC; nao diagnostica falhas CA do inversor",
            "benchmark supervisionado com rotulos conhecidos",
        ],
    }
    training_manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset": "PV Farms",
        "dominio": "CC",
        "modelo": "Random Forest",
        "parameters": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
        "scaler": "StandardScaler",
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "n_features": int(len(colunas)),
        "feature_columns": colunas,
        "evidence_level": "E1",
        "outputs": [
            "modelo_classificador.pkl",
            "scaler.pkl",
            "feature_columns.json",
            "class_mapping.json",
            "metricas.json",
            "metricas.csv",
            "matriz_confusao.png",
            "importancia_features.png",
        ],
    }

    joblib.dump(clf, pasta / "modelo_classificador.pkl")
    with open(pasta / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # Integridade: SHA-256 dos artefatos pickle — conferido em carregar().
    # Pickle executa código ao desserializar; o hash garante que só carregamos
    # exatamente o que este treino gravou.
    from src.core.seguranca import sha256_de_arquivo

    (pasta / "hashes.json").write_text(
        json.dumps({
            "modelo_classificador.pkl": sha256_de_arquivo(pasta / "modelo_classificador.pkl"),
            "scaler.pkl": sha256_de_arquivo(pasta / "scaler.pkl"),
        }, indent=2), encoding="utf-8")
    (pasta / "feature_columns.json").write_text(
        json.dumps(colunas, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "class_mapping.json").write_text(
        json.dumps({str(c): NOMES_CLASSES.get(c, f"Classe {c}") for c in classes},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "metricas.json").write_text(
        json.dumps(metricas_salvas, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([metricas_salvas]).to_csv(pasta / "metricas.csv", index=False)
    (pasta / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "training_manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    nomes_classes = [_nome_classe(c) for c in metricas.get("classes", [str(c) for c in classes])]
    _plotar_matriz_confusao(metricas["matriz_confusao"], nomes_classes, pasta / "matriz_confusao.png")
    _plotar_importancia_features(clf, colunas, pasta / "importancia_features.png")

    try:
        from src.ml.pipeline import registrar_manifesto  # noqa: F401
    except Exception:
        pass
    return {"ok": True, "n_features": len(colunas), "classes": classes,
            "metricas": metricas}


def treinar_e_salvar(pasta: Path = PASTA) -> dict:
    """Treina/salva a partir do dataset PV Farms local."""
    from src.ml.classificador_pv import CSV_TESTE, CSV_TREINO, carregar_dados

    X_tr, y_tr, X_te, y_te = carregar_dados()
    return treinar_e_salvar_de(
        X_tr,
        y_tr,
        X_te,
        y_te,
        pasta,
        source_paths={"train": CSV_TREINO, "test": CSV_TESTE},
    )


def carregar(pasta: Path = PASTA) -> dict | None:
    """Carrega os artefatos do classificador, ou None se ausentes.

    Segurança: quando ``hashes.json`` existe (gravado por ``treinar_e_salvar``),
    o SHA-256 de cada pickle é conferido ANTES de desserializar — um artefato
    trocado em disco levanta ``ValueError`` em vez de executar código
    arbitrário. Artefatos antigos sem hash carregam com aviso no log.
    """
    import joblib

    from src.core.seguranca import carregar_pickle_verificado, sha256_de_arquivo

    arq_modelo = pasta / "modelo_classificador.pkl"
    arq_cols = pasta / "feature_columns.json"
    if not (arq_modelo.exists() and arq_cols.exists()):
        return None

    arq_hashes = pasta / "hashes.json"
    if arq_hashes.exists():
        hashes = json.loads(arq_hashes.read_text(encoding="utf-8"))
        scaler = carregar_pickle_verificado(
            pasta / "scaler.pkl", hashes.get("scaler.pkl", ""))
        h_modelo = hashes.get("modelo_classificador.pkl", "")
        if sha256_de_arquivo(arq_modelo) != str(h_modelo).lower():
            raise ValueError(
                "Integridade violada em modelo_classificador.pkl: SHA-256 não "
                "confere com hashes.json. Retreine com treinar_e_salvar()."
            )
        modelo = joblib.load(arq_modelo)
    else:
        from src.core.logs import get_logger

        get_logger("classificador_pv_infer").warning(
            "Artefatos sem hashes.json (anteriores ao hardening); carregando "
            "sem verificação de integridade. Retreine para gerar os hashes."
        )
        with open(pasta / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        modelo = joblib.load(arq_modelo)

    return {
        "modelo": modelo,
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
    import pandas as pd

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

    X_df = pd.DataFrame([vetor], columns=colunas, dtype=float)
    X = art["scaler"].transform(X_df)
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
