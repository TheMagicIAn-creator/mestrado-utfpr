"""
classificador_pv.py — Al IAdo PV / Fase 5
Classificação supervisionada de falhas em sistemas fotovoltaicos.

Dataset: Fault Detection in Photovoltaic Farms
Referência: Ghoneim, Rashed & Elkalashy (2021)

Classes:
  Normal | F1 (string) | F2 (string-terra) | F3 (string-string)

Modelos comparados:
  Random Forest, XGBoost, LightGBM, SVM, Gradient Boosting

Uso:
  python src/ml/classificador_pv.py

Autor: Rodolfo Torres (UTFPR)
"""

try:
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta: python src/ml/<arquivo>.py
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("classificador_pv")


def _log(*args, sep=" ", end="\n", flush=None):
    """Progresso/sumário de ML vai para o ARQUIVO de log — o terminal
    fica silencioso quando rodando pelo app. Scripts manuais reativam o
    eco chamando habilitar_console() no bloco __main__. Linhas de
    progresso com \\r são rebaixadas a DEBUG."""
    texto = sep.join(str(a) for a in args)
    if not texto.strip():
        return
    if texto.startswith("\r"):
        _logger.debug(texto.strip())
        return
    _logger.info(texto.rstrip("\n"))



import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing      import StandardScaler
from sklearn.ensemble           import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm                import SVC
from sklearn.model_selection    import cross_val_score, StratifiedKFold
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS    = Path(__file__).parent.parent.parent / "dados" / "brutos"
PASTA_RESULT   = Path(__file__).parent.parent.parent / "resultados" / "classificacao_pv"
CSV_TREINO     = PASTA_DADOS / "train_data.csv"
CSV_TESTE      = PASTA_DADOS / "test_data.csv"

# Nomes amigáveis das classes (ajuste se o mapeamento for outro)
NOMES_CLASSES = {
    0: "Normal",
    1: "F1 - String",
    2: "F2 - String-Terra",
    3: "F3 - String-String"
}


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def carregar_dados() -> tuple:
    """
    Carrega os datasets de treino e teste.
    Separa features (X) e rótulo (y).
    """
    _log("📂 Carregando datasets...")

    # Separador é ponto-e-vírgula
    df_treino = pd.read_csv(CSV_TREINO, sep=";")
    df_teste  = pd.read_csv(CSV_TESTE,  sep=";")

    _log(f"   ✅ Treino: {len(df_treino)} instâncias")
    _log(f"   ✅ Teste : {len(df_teste)} instâncias")

    # Separa features e rótulo
    X_treino = df_treino.drop(columns=["class"])
    y_treino = df_treino["class"]
    X_teste  = df_teste.drop(columns=["class"])
    y_teste  = df_teste["class"]

    _log(f"   ✅ Features: {X_treino.shape[1]} colunas")
    _log(f"   ✅ Classes : {sorted(y_treino.unique())}")

    return X_treino, y_treino, X_teste, y_teste


# ============================================================
# ANÁLISE DA DISTRIBUIÇÃO DE CLASSES
# ============================================================

def analisar_classes(y_treino, y_teste):
    """Mostra o balanceamento das classes."""

    _log("\n📊 DISTRIBUIÇÃO DAS CLASSES")
    _log("=" * 60)

    dist_treino = y_treino.value_counts().sort_index()

    for classe, qtd in dist_treino.items():
        nome = NOMES_CLASSES.get(classe, f"Classe {classe}")
        pct  = qtd / len(y_treino) * 100
        _log(f"   {nome:25s}: {qtd:4d} ({pct:5.1f}%)")


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def preprocessar(X_treino, X_teste) -> tuple:
    """
    Normaliza as features com StandardScaler.

    Por que normalizar?
    Modelos como SVM são sensíveis à escala. Features com
    valores grandes (ex: tensão ~500) dominariam features
    pequenas (ex: variância ~0.001) sem normalização.
    """
    _log("\n⚙️  Pré-processando (normalização)...")

    scaler   = StandardScaler()
    X_treino_norm = scaler.fit_transform(X_treino)  # aprende e aplica
    X_teste_norm  = scaler.transform(X_teste)        # só aplica

    _log("   ✅ Features normalizadas (média 0, desvio 1)")

    return X_treino_norm, X_teste_norm, scaler


# ============================================================
# DEFINIÇÃO DOS MODELOS
# ============================================================

def criar_modelos() -> dict:
    """
    Cria o dicionário de modelos a comparar.
    XGBoost e LightGBM são opcionais — se não instalados, são pulados.
    """
    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", random_state=42, probability=True
        ),
    }

    # XGBoost (opcional)
    try:
        from xgboost import XGBClassifier
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=200, random_state=42,
            use_label_encoder=False, eval_metric="mlogloss"
        )
    except ImportError:
        _log("   ⚠️  XGBoost não instalado — pulando")

    # LightGBM (opcional)
    try:
        from lightgbm import LGBMClassifier
        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=200, random_state=42, verbose=-1
        )
    except ImportError:
        _log("   ⚠️  LightGBM não instalado — pulando")

    return modelos


# ============================================================
# TREINAMENTO E AVALIAÇÃO
# ============================================================

def treinar_e_avaliar(modelos, X_treino, y_treino, X_teste, y_teste) -> dict:
    """
    Treina cada modelo e avalia com:
      - Validação cruzada (5-fold) no treino
      - Métricas finais no conjunto de teste
    """
    _log("\n🤖 TREINANDO E AVALIANDO MODELOS")
    _log("=" * 60)

    # XGBoost precisa de classes começando em 0
    y_treino_ajust = y_treino - y_treino.min()
    y_teste_ajust  = y_teste  - y_teste.min()

    resultados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nome, modelo in modelos.items():
        _log(f"\n  ▶ {nome}")

        # Validação cruzada no treino
        scores_cv = cross_val_score(
            modelo, X_treino, y_treino_ajust,
            cv=cv, scoring="accuracy"
        )

        # Treina no conjunto completo de treino
        modelo.fit(X_treino, y_treino_ajust)

        # Avalia no teste
        y_pred = modelo.predict(X_teste)

        acc       = accuracy_score(y_teste_ajust, y_pred)
        precisao  = precision_score(y_teste_ajust, y_pred, average="macro")
        recall    = recall_score(y_teste_ajust, y_pred, average="macro")
        f1        = f1_score(y_teste_ajust, y_pred, average="macro")

        # Métricas ampliadas (4.1/4.2), via o schema único já testado:
        # MCC, balanced_accuracy, specificity (binária/macro-OvR rotulada), FPR/FNR.
        from src.ml.experimentos_artigos import _metricas_classificacao
        extra = _metricas_classificacao(list(y_teste_ajust), list(y_pred))

        resultados[nome] = {
            "modelo"                : modelo,
            "cv_media"              : scores_cv.mean(),
            "cv_desvio"             : scores_cv.std(),
            "acuracia"              : acc,
            "precisao"              : precisao,
            "recall"                : recall,
            "f1"                    : f1,
            "mcc"                   : extra["mcc"],
            "balanced_accuracy"     : extra["balanced_accuracy"],
            "specificity"           : extra["specificity"],
            "specificity_macro_ovr" : extra["specificity_macro_ovr"],
            "specificity_tipo"      : extra["specificity_tipo"],
            "false_positive_rate"   : extra["false_positive_rate"],
            "false_negative_rate"   : extra["false_negative_rate"],
            "y_pred"                : y_pred,
        }

        _log(f"     Validação Cruzada : {scores_cv.mean():.4f} (±{scores_cv.std():.4f})")
        _log(f"     Acurácia (teste)  : {acc:.4f}")
        _log(f"     Precisão (macro)  : {precisao:.4f}")
        _log(f"     Recall (macro)    : {recall:.4f}")
        _log(f"     F1-Score (macro)  : {f1:.4f}")
        _log(f"     MCC               : {extra['mcc']:.4f}")
        _log(f"     Balanced accuracy : {extra['balanced_accuracy']:.4f}")

    return resultados, y_teste_ajust


# ============================================================
# GRÁFICOS
# ============================================================

def plotar_comparacao(resultados: dict):
    """Gráfico de barras comparando os modelos."""

    PASTA_RESULT.mkdir(parents=True, exist_ok=True)

    nomes     = list(resultados.keys())
    metricas  = ["acuracia", "precisao", "recall", "f1"]
    rotulos   = ["Acurácia", "Precisão", "Recall", "F1-Score"]

    fig = go.Figure()

    for metrica, rotulo in zip(metricas, rotulos):
        valores = [resultados[n][metrica] for n in nomes]
        fig.add_trace(go.Bar(name=rotulo, x=nomes, y=valores))

    fig.update_layout(
        title="Comparação de Modelos — Classificação de Falhas PV",
        xaxis_title="Modelo",
        yaxis_title="Pontuação",
        barmode="group",
        template="plotly_dark",
        yaxis=dict(range=[0, 1.05])
    )

    caminho = PASTA_RESULT / "comparacao_modelos.html"
    fig.write_html(str(caminho))
    _log(f"\n✅ Gráfico salvo: {caminho.name}")


def plotar_matriz_confusao(resultados: dict, y_teste, melhor_modelo: str):
    """Matriz de confusão do melhor modelo."""

    y_pred  = resultados[melhor_modelo]["y_pred"]
    classes = sorted(np.unique(y_teste))
    nomes   = [NOMES_CLASSES.get(c + min(NOMES_CLASSES.keys()), f"C{c}") for c in classes]

    matriz = confusion_matrix(y_teste, y_pred)

    fig = px.imshow(
        matriz,
        x=nomes, y=nomes,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Previsto", y="Real", color="Casos"),
        title=f"Matriz de Confusão — {melhor_modelo}"
    )
    fig.update_layout(template="plotly_dark", height=500)

    caminho = PASTA_RESULT / "matriz_confusao.html"
    fig.write_html(str(caminho))
    _log(f"✅ Gráfico salvo: {caminho.name}")


def plotar_importancia(resultados: dict, nomes_features: list, melhor_modelo: str):
    """Importância das features do melhor modelo (se for baseado em árvore)."""

    modelo = resultados[melhor_modelo]["modelo"]

    if not hasattr(modelo, "feature_importances_"):
        _log(f"ℹ️  {melhor_modelo} não fornece importância de features.")
        return

    importancias = modelo.feature_importances_
    df_imp = pd.DataFrame({
        "feature"    : nomes_features,
        "importancia": importancias
    }).sort_values("importancia", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=df_imp["importancia"],
        y=df_imp["feature"],
        orientation="h",
        marker_color="#4CAF50"
    ))

    fig.update_layout(
        title=f"Top 15 Features Mais Importantes — {melhor_modelo}",
        xaxis_title="Importância",
        yaxis_title="Feature",
        template="plotly_dark",
        height=500
    )

    caminho = PASTA_RESULT / "importancia_features.html"
    fig.write_html(str(caminho))
    _log(f"✅ Gráfico salvo: {caminho.name}")


# ============================================================
# RELATÓRIO FINAL
# ============================================================

def gerar_relatorio(resultados: dict, melhor_modelo: str, y_teste):
    """Salva um relatório de texto com os resultados."""

    PASTA_RESULT.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_RESULT / "relatorio_classificacao.txt"

    linhas = []
    linhas.append("=" * 60)
    linhas.append("  RELATÓRIO — CLASSIFICAÇÃO DE FALHAS PV")
    linhas.append("  Al IAdo PV — Mestrado UTFPR")
    linhas.append("=" * 60)
    linhas.append("")
    linhas.append("COMPARAÇÃO DE MODELOS:")
    linhas.append("-" * 60)

    for nome, r in sorted(resultados.items(), key=lambda x: -x[1]["f1"]):
        linhas.append(f"\n{nome}")
        linhas.append(f"  Validação Cruzada : {r['cv_media']:.4f} (±{r['cv_desvio']:.4f})")
        linhas.append(f"  Acurácia          : {r['acuracia']:.4f}")
        linhas.append(f"  Precisão (macro)  : {r['precisao']:.4f}")
        linhas.append(f"  Recall (macro)    : {r['recall']:.4f}")
        linhas.append(f"  F1-Score (macro)  : {r['f1']:.4f}")

    linhas.append("")
    linhas.append("=" * 60)
    linhas.append(f"MELHOR MODELO: {melhor_modelo}")
    linhas.append(f"F1-Score: {resultados[melhor_modelo]['f1']:.4f}")
    linhas.append("=" * 60)

    caminho.write_text("\n".join(linhas), encoding="utf-8")
    _log(f"✅ Relatório salvo: {caminho.name}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_classificacao() -> bool:
    _log("=" * 60)
    _log("  AL IADO PV — CLASSIFICAÇÃO DE FALHAS FOTOVOLTAICAS")
    _log("=" * 60)

    # 1. Carrega dados
    X_treino, y_treino, X_teste, y_teste = carregar_dados()

    # 2. Analisa classes
    analisar_classes(y_treino, y_teste)

    # 3. Pré-processa
    X_treino_norm, X_teste_norm, scaler = preprocessar(X_treino, X_teste)

    # 4. Cria modelos
    _log("\n🔧 Criando modelos...")
    modelos = criar_modelos()
    _log(f"   ✅ {len(modelos)} modelos prontos: {', '.join(modelos.keys())}")

    # 5. Treina e avalia
    resultados, y_teste_ajust = treinar_e_avaliar(
        modelos, X_treino_norm, y_treino, X_teste_norm, y_teste
    )

    # 6. Identifica o melhor (por F1-Score)
    melhor = max(resultados, key=lambda n: resultados[n]["f1"])

    _log("\n" + "=" * 60)
    _log(f"  🏆 MELHOR MODELO: {melhor}")
    _log(f"     F1-Score: {resultados[melhor]['f1']:.4f}")
    _log("=" * 60)

    # 7. Gráficos
    _log("\n📈 Gerando gráficos...")
    plotar_comparacao(resultados)
    plotar_matriz_confusao(resultados, y_teste_ajust, melhor)
    plotar_importancia(resultados, list(X_treino.columns), melhor)

    # 8. Relatório
    gerar_relatorio(resultados, melhor, y_teste_ajust)

    _log("\n" + "=" * 60)
    _log("  PIPELINE CONCLUÍDO!")
    _log(f"  Resultados em: resultados/classificacao_pv/")
    _log("=" * 60)

    return True

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    executar_classificacao()