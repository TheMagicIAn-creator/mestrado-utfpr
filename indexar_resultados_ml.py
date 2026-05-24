"""
indexar_resultados_ml.py — Indexa os resultados da Fase 5
na base de conhecimento para o agente poder discuti-los.
"""
import json
from pathlib import Path
from datetime import datetime

RAIZ     = Path(__file__).parent
PASTA_AE = RAIZ / "resultados" / "autoencoder"
SAIDA    = RAIZ / "notas" / "memorias" / "resultados-fase5-ml.md"

linhas = ["# Resultados da Fase 5 — Pipeline de ML\n"]
linhas.append(f"> Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

# Autoencoder
arq = PASTA_AE / "limiar.json"
if arq.exists():
    d = json.loads(arq.read_text(encoding="utf-8"))
    linhas.append("\n## Autoencoder\n")
    linhas.append(f"O Autoencoder de detecção de anomalias foi treinado com "
                  f"limiar operacional (percentil 99) de {d['limiar']:.4f}. "
                  f"O erro de reconstrução baseline do inversor saudável tem "
                  f"média {d['mu']:.4f} e desvio {d['sigma']:.4f}. "
                  f"Foram treinadas {d.get('epochs_treinadas','?')} épocas.\n")

# Injeção de falhas
arq = PASTA_AE / "injecao_falhas_report.json"
if arq.exists():
    d = json.loads(arq.read_text(encoding="utf-8"))
    linhas.append("\n## Injeção de Falhas Sintéticas\n")
    for fid, f in d["falhas"].items():
        smd = d["smd"].get(fid)
        linhas.append(f"- {f['nome']} (NPR={f['npr']}): "
                      f"severidade mínima detectável = {smd}.\n")

# Validação
arq = PASTA_AE / "validacao_report.json"
if arq.exists():
    d = json.loads(arq.read_text(encoding="utf-8"))
    linhas.append("\n## Validação Formal\n")
    for chave, r in d.items():
        linhas.append(f"- {chave}: AUC={r['auc_roc']:.3f}, "
                      f"F1={r['f1']:.3f}, Recall={r['recall']:.3f}.\n")

# Weibull
arq = PASTA_AE / "weibull_results.json"
if arq.exists():
    d = json.loads(arq.read_text(encoding="utf-8"))
    linhas.append("\n## RUL — Análise de Weibull\n")
    for fid, f in d["falhas"].items():
        p = f["weibull"]
        linhas.append(f"- {f['nome']}: β={p['beta']:.2f}, η={p['eta']:.1f}, "
                      f"MTTF={p['mttf']:.1f}, B10={p['b10']:.1f}. "
                      f"Taxa de falha {'crescente' if p['beta']>1 else 'constante'}.\n")

SAIDA.parent.mkdir(parents=True, exist_ok=True)
SAIDA.write_text("".join(linhas), encoding="utf-8")
print(f"Resumo salvo: {SAIDA}")

# Indexa no ChromaDB
from sentence_transformers import SentenceTransformer
from src.core.config import MODELO_EMBEDDINGS, PASTA_CHROMADB
from src.conhecimento.indexador import indexar_sessao

modelo = SentenceTransformer(MODELO_EMBEDDINGS)
indexar_sessao(SAIDA, modelo, PASTA_CHROMADB)
print("Resultados indexados na base de conhecimento.")