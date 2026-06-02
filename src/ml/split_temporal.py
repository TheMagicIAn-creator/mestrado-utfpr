"""
split_temporal.py — Al IAdo PV / Sprint 1 (integridade metodológica)

Divisão TEMPORAL em blocos contíguos (treino → validação → teste) com ZONA DE
PURGA entre os blocos, para o dataset de Paderborn.

Por que isto importa
--------------------
As features do Paderborn são extraídas em janelas com 50% de sobreposição. Uma
divisão ALEATÓRIA das janelas faria janelas quase idênticas (vizinhas, que
compartilham metade das amostras) caírem ao mesmo tempo em treino e validação —
vazamento temporal, que infla as métricas.

A divisão correta é por BLOCOS contíguos no tempo, descartando algumas janelas
na fronteira (purga) para que nenhuma janela de um bloco compartilhe amostras
com a janela do bloco seguinte.

Uso típico
----------
    sp = split_temporal_com_purga(n_janelas=457, train_ratio=0.60,
                                  val_ratio=0.20, test_ratio=0.20,
                                  purge_janelas=2)
    X_tr = X[sp["treino"]]
    X_val = X[sp["val"]]
    X_te = X[sp["teste"]]

Determinístico (sem aleatoriedade) → reprodutível por construção.
"""

from __future__ import annotations

import numpy as np

# Sobreposição padrão das janelas (50% em features_ca → 1 vizinho compartilha
# metade das amostras). purge >= overlap_janelas garante fronteira limpa.
PURGA_PADRAO = 2


def split_temporal_com_purga(
    n_janelas: int,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    purge_janelas: int = PURGA_PADRAO,
) -> dict:
    """
    Retorna os índices (ordem temporal) de treino, validação e teste, com
    blocos contíguos e purga entre eles.

    Lança ValueError se os ratios não somam 1, se houver janelas insuficientes
    para formar os três blocos, ou se algum parâmetro for inválido.

    Retorno:
        {
          "treino": np.ndarray[int], "val": np.ndarray[int], "teste": np.ndarray[int],
          "limites": {"treino": (ini, fim), "val": (...), "teste": (...)},
          "purge_janelas": int,
          "ratios": {"train": ..., "val": ..., "test": ...},
          "n_janelas": int,
        }
    """
    if n_janelas <= 0:
        raise ValueError("n_janelas deve ser > 0.")
    if purge_janelas < 0:
        raise ValueError("purge_janelas deve ser >= 0.")
    soma = train_ratio + val_ratio + test_ratio
    if abs(soma - 1.0) > 1e-6:
        raise ValueError(f"train+val+test deve somar 1.0 (soma={soma:.4f}).")
    for nome, r in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)):
        if not (0.0 < r < 1.0):
            raise ValueError(f"{nome}_ratio deve estar em (0,1); recebido {r}.")

    idx = np.arange(n_janelas, dtype=int)

    n_tr = int(np.floor(n_janelas * train_ratio))
    n_val = int(np.floor(n_janelas * val_ratio))

    fim_tr = n_tr
    ini_val = fim_tr + purge_janelas
    fim_val = ini_val + n_val
    ini_te = fim_val + purge_janelas
    fim_te = n_janelas

    treino = idx[0:fim_tr]
    val = idx[ini_val:fim_val]
    teste = idx[ini_te:fim_te]

    if len(treino) == 0 or len(val) == 0 or len(teste) == 0:
        raise ValueError(
            f"Janelas insuficientes ({n_janelas}) para 3 blocos com purga="
            f"{purge_janelas}. Reduza a purga ou agregue mais janelas."
        )

    return {
        "treino": treino,
        "val": val,
        "teste": teste,
        "limites": {
            "treino": (0, fim_tr),
            "val": (ini_val, fim_val),
            "teste": (ini_te, fim_te),
        },
        "purge_janelas": purge_janelas,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "n_janelas": n_janelas,
    }
