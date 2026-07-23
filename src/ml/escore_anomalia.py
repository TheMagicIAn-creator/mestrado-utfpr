"""
escore_anomalia.py — Al IAdo PV / fonte ÚNICA do escore de anomalia do pipeline CA.

Dois escores, ambos calibrados para ~1% de falso positivo no bloco saudável:

- 'mse'        : média do erro de reconstrução sobre TODAS as ~109 features
                 (escore histórico). Dilui falhas localizadas no espectro.
- 'localizado' : média dos top-k |resíduos| PADRONIZADOS por feature — z do
                 |resíduo| contra a régua saudável, agregado pelos k maiores.
                 Sensível a falha concentrada em poucas features (harmônicos do
                 IGBT, perda de fase do Fusível).

Fundamentação (docs/auditoria_pipeline_ml.md §13): erro de reconstrução como
sinal de anomalia (Ibrahim, 2022, eq. 3); padronização por-feature do resíduo
(Francisti, 2025 — Z-score/Shewhart, aqui sobre o resíduo do AE); agregação
top-k como generalização robusta da regra de Shewhart / SPC multivariável e da
contribuição por feature (Narayanan, 2023).

Método OPERACIONAL padrão: 'localizado'. Para reproduzir EXATAMENTE o pipeline
antigo, defina a variável de ambiente:
    AL_IADO_ESCORE_ANOMALIA=mse
O k do top-k é configurável por AL_IADO_ESCORE_K (padrão 5; justificar por
varredura, ver §13). Este módulo é uma FOLHA: depende só de numpy/torch — nunca
de autoencoder/injecao/validacao/rul (evita ciclo de import).

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ── Configuração operacional (env, com padrões seguros) ─────────────────────
METODO_ESCORE = os.getenv("AL_IADO_ESCORE_ANOMALIA", "localizado").strip().lower()
K_LOCALIZADO = int(os.getenv("AL_IADO_ESCORE_K", "5"))

# Nome canônico do artefato com a régua por-feature (μ/σ do |resíduo| saudável).
ARQUIVO_ESTATISTICA = "estatistica_residuo.npz"


# ============================================================
# NÚCLEO — funções puras (testáveis sem dataset)
# ============================================================

def ajustar_estatistica_residuo(residuos_saudaveis: np.ndarray) -> dict:
    """μ/σ por-feature do |resíduo| no bloco saudável (a régua de padronização).

    residuos_saudaveis: (n_janelas, n_features) do resíduo (x − x_rec) em
    espaço normalizado. Retorna {"mu": (F,), "sigma": (F,)}.
    """
    abs_r = np.abs(np.asarray(residuos_saudaveis, dtype=float))
    mu = abs_r.mean(axis=0)
    sigma = abs_r.std(axis=0) + 1e-9
    return {"mu": mu, "sigma": sigma}


def escore_mse_medio(residuos: np.ndarray) -> np.ndarray:
    """Escore histórico: MSE médio sobre todas as features. (n,)→(n,)."""
    r = np.atleast_2d(np.asarray(residuos, dtype=float))
    return (r ** 2).mean(axis=1)


def escore_localizado(residuos: np.ndarray, stats: dict,
                      k: int = K_LOCALIZADO) -> np.ndarray:
    """Escore sensível a falha LOCALIZADA: média dos top-k |resíduos| z-padronizados.

    Não dilui um desvio concentrado em poucas features. residuos: (n, F).
    """
    r = np.abs(np.atleast_2d(np.asarray(residuos, dtype=float)))
    z = (r - stats["mu"]) / stats["sigma"]
    k = int(max(1, min(k, z.shape[1])))
    topk = np.partition(z, -k, axis=1)[:, -k:]
    return topk.mean(axis=1)


def pontuar(residuos: np.ndarray, stats: dict | None = None,
            metodo: str = METODO_ESCORE, k: int = K_LOCALIZADO) -> np.ndarray:
    """Escore operacional segundo `metodo`. Cai para 'mse' se faltar a régua.

    Fallback seguro: sem `stats` (régua ausente), retorna o MSE médio — o
    comportamento histórico —, nunca quebra por artefato faltando.
    """
    if metodo == "mse" or stats is None:
        return escore_mse_medio(residuos)
    return escore_localizado(residuos, stats, k=k)


# ============================================================
# INFERÊNCIA — resíduo por feature a partir do modelo
# ============================================================

def residuo_por_feature(modelo, X: np.ndarray, device) -> np.ndarray:
    """Resíduo (x − x_rec) por feature, em espaço normalizado. X: (n, F)→(n, F)."""
    import torch

    modelo.eval()
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        rec = modelo(t)
        return (t - rec).cpu().numpy()


def residuo_de_vetor(modelo, vetor_norm: np.ndarray, device) -> np.ndarray:
    """Resíduo por feature de UMA janela já normalizada. (1,F)/(F,)→(F,)."""
    v = np.asarray(vetor_norm, dtype=np.float32).reshape(1, -1)
    return residuo_por_feature(modelo, v, device).ravel()


# ============================================================
# PERSISTÊNCIA da régua por-feature (μ/σ)
# ============================================================

def salvar_estatistica(stats: dict, pasta: Path,
                       nome: str = ARQUIVO_ESTATISTICA) -> Path:
    caminho = Path(pasta) / nome
    np.savez(caminho, mu=np.asarray(stats["mu"], dtype=np.float64),
             sigma=np.asarray(stats["sigma"], dtype=np.float64))
    return caminho


def carregar_estatistica(pasta: Path,
                         nome: str = ARQUIVO_ESTATISTICA) -> dict | None:
    """Carrega a régua por-feature; None se ausente (→ fallback para MSE)."""
    caminho = Path(pasta) / nome
    if not caminho.exists():
        return None
    with np.load(caminho) as d:
        return {"mu": d["mu"], "sigma": d["sigma"]}


def descricao_metodo(metodo: str = METODO_ESCORE, k: int = K_LOCALIZADO) -> str:
    if metodo == "mse":
        return "MSE médio sobre todas as features (histórico)"
    return f"localizado: média dos top-{k} |resíduos| padronizados por feature"
