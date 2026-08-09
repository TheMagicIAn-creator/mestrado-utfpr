"""
split_temporal.py — Al IAdo PV / Sprint 1 (integridade metodológica)

Divisão TEMPORAL em blocos contíguos (treino → validação → teste) com ZONA DE
PURGA entre os blocos, para o conjunto Stender (Paderborn University).

Por que isto importa
--------------------
As features do conjunto Stender são extraídas em janelas com 50% de sobreposição. Uma
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
TRAIN_RATIO_PADRAO = 0.60
CALIB_RATIO_PADRAO = 0.20
TEST_RATIO_PADRAO = 0.20

# ── Estratégia de split: por que deixou de ser 3 blocos contíguos ───────────
#
# Três blocos contíguos pressupõem que o sinal seja aproximadamente
# ESTACIONÁRIO ao longo do tempo. O conjunto Stender não é: é uma bancada de
# acionamento que varre rotação em RAMPA. Fatiar a rampa em três dá três faixas
# de velocidade, não três amostras do mesmo processo.
#
# Medido em 09/08/2026 (scripts/diagnostico_limiar.py), com 224 janelas:
#
#     bloco         mediana de F0      IQR       n
#     treino           20,45 Hz      83,13      136
#     calibração       51,11 Hz       1,46       45
#     teste           100,08 Hz      17,84       43
#
# O IQR da calibração é 1,46 Hz: o bloco inteiro está parado num único regime.
# O limiar operacional era congelado ali e aplicado a um bloco que opera ao
# DOBRO da fundamental. FPR de 4,4% na calibração e 62,8% no teste — e o
# diagnóstico é explícito de que isso mede cobertura de dados, não erro de
# calibração.
#
# Note que o TREINO tem IQR de 83 Hz: o autoencoder viu a faixa inteira. Quem
# extrapolava era só o limiar.
#
# A correção é fatiar em MAIS blocos e distribuí-los alternadamente. A garantia
# anti-vazamento não muda de natureza — continua sendo purga em fronteira, só
# que agora em toda fronteira onde o destino muda.
#
# CUSTO METODOLÓGICO, que precisa estar escrito na dissertação: o teste deixa de
# ser "o futuro" e passa a ser generalização ENTRE REGIMES. Para detecção de
# anomalia em bancada de velocidade variável isso é mais adequado que previsão
# temporal — o inversor em campo não opera em rampa monotônica —, mas é uma
# afirmação diferente e não pode ser apresentada como a anterior.
ESTRATEGIA_SPLIT = "blocos_intercalados"
# 15 blocos sobre ~224 janelas dão ~15 janelas por bloco, e 15 divide 60/20/20
# em 9/3/3 exatos. Mais blocos cobrem melhor o regime mas gastam mais purga
# (uma fronteira a cada troca de destino); menos blocos voltam ao problema.
N_BLOCOS_PADRAO = 15


def split_temporal_com_purga(
    n_janelas: int,
    train_ratio: float = TRAIN_RATIO_PADRAO,
    val_ratio: float = CALIB_RATIO_PADRAO,
    test_ratio: float = TEST_RATIO_PADRAO,
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


def sequencia_de_destinos(n_blocos: int, ratios: dict) -> list:
    """Ordena os destinos dos blocos espalhando cada conjunto uniformemente.

    Não é sorteio: cada conjunto recebe posições ideais ``(k + 0,5)/c`` e a
    sequência sai da ordenação dessas posições. Determinístico e reprodutível
    por construção — a mesma propriedade que o split contíguo tinha.

    Com 15 blocos e 60/20/20 devolve
    ``T E T V T T E T V T T E T V T``: teste nas posições 1, 6, 11 e calibração
    em 3, 8, 13. Cada conjunto atravessa a rampa de rotação inteira.
    """
    if n_blocos < 3:
        raise ValueError(f"n_blocos deve ser >= 3; recebido {n_blocos}")

    alvo = {nome: n_blocos * r for nome, r in ratios.items()}
    conta = {nome: int(np.floor(v)) for nome, v in alvo.items()}
    # Cada conjunto precisa de pelo menos um bloco antes de repartir o resto.
    for nome in conta:
        conta[nome] = max(conta[nome], 1)
    while sum(conta.values()) > n_blocos:                     # ratios extremos
        maior = max(conta, key=lambda k: (conta[k], k))
        if conta[maior] <= 1:
            raise ValueError(
                f"n_blocos={n_blocos} é insuficiente para os ratios {ratios}")
        conta[maior] -= 1
    # Resto pelo maior resíduo; desempate por nome, para ser determinístico.
    sobra = n_blocos - sum(conta.values())
    ordem = sorted(alvo, key=lambda k: (-(alvo[k] - conta[k]), k))
    for nome in ordem[:sobra]:
        conta[nome] += 1

    posicoes = [((k + 0.5) / c, nome)
                for nome, c in conta.items() for k in range(c)]
    posicoes.sort(key=lambda p: (p[0], p[1]))
    return [nome for _, nome in posicoes]


def split_blocos_intercalados(
    n_janelas: int,
    train_ratio: float = TRAIN_RATIO_PADRAO,
    val_ratio: float = CALIB_RATIO_PADRAO,
    test_ratio: float = TEST_RATIO_PADRAO,
    purge_janelas: int = PURGA_PADRAO,
    n_blocos: int = N_BLOCOS_PADRAO,
) -> dict:
    """Divide em ``n_blocos`` contíguos e os distribui entre os três conjuntos.

    Preserva a proteção anti-vazamento do split contíguo: janelas com 50% de
    sobreposição nunca cruzam conjuntos, porque em toda fronteira onde o destino
    MUDA são descartadas ``purge_janelas`` janelas. Onde o destino não muda não
    há o que purgar — blocos vizinhos do mesmo conjunto seguem contíguos.

    Ganha cobertura de regime: cada conjunto atravessa a série inteira em vez de
    ocupar uma fatia dela. Ver o bloco de comentário no topo do módulo.

    Devolve o mesmo contrato de ``split_temporal_com_purga``, com ``limites``
    virando uma LISTA de intervalos por conjunto, e mais ``estrategia``,
    ``n_blocos`` e ``destinos``.
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
    if n_blocos > n_janelas:
        raise ValueError(
            f"n_blocos ({n_blocos}) não pode exceder n_janelas ({n_janelas}).")

    destinos = sequencia_de_destinos(
        n_blocos,
        {"treino": train_ratio, "val": val_ratio, "teste": test_ratio},
    )

    # Fronteiras dos blocos: np.linspace arredondado distribui o resto de forma
    # equilibrada, sem deixar um bloco final anão.
    cortes = np.linspace(0, n_janelas, n_blocos + 1).round().astype(int)

    conjuntos = {"treino": [], "val": [], "teste": []}
    limites = {"treino": [], "val": [], "teste": []}
    for b, destino in enumerate(destinos):
        ini, fim = int(cortes[b]), int(cortes[b + 1])
        # Purga só onde o destino muda — é onde janelas vizinhas cairiam em
        # conjuntos diferentes.
        if b > 0 and destinos[b - 1] != destino:
            ini += purge_janelas
        if ini >= fim:
            continue
        conjuntos[destino].extend(range(ini, fim))
        limites[destino].append((ini, fim))

    vazios = [nome for nome, idx in conjuntos.items() if not idx]
    if vazios:
        raise ValueError(
            f"Janelas insuficientes ({n_janelas}) para {n_blocos} blocos com "
            f"purga={purge_janelas}: {', '.join(vazios)} ficou vazio. Reduza "
            f"n_blocos ou a purga."
        )

    return {
        "treino": np.asarray(conjuntos["treino"], dtype=int),
        "val": np.asarray(conjuntos["val"], dtype=int),
        "teste": np.asarray(conjuntos["teste"], dtype=int),
        "limites": limites,
        "purge_janelas": purge_janelas,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "n_janelas": n_janelas,
        "estrategia": ESTRATEGIA_SPLIT,
        "n_blocos": n_blocos,
        "destinos": destinos,
    }


def split_padrao_paderborn(n_janelas: int) -> dict:
    """Split canonico usado por treino, calibracao e avaliacao do pipeline CA.

    Desde 09/08/2026 usa BLOCOS INTERCALADOS. O split de três blocos contíguos
    continua disponível em `split_temporal_com_purga` — é o que reproduz os
    resultados anteriores, e o protocolo E1 por artigo segue usando ele.
    """
    return split_blocos_intercalados(
        n_janelas=n_janelas,
        train_ratio=TRAIN_RATIO_PADRAO,
        val_ratio=CALIB_RATIO_PADRAO,
        test_ratio=TEST_RATIO_PADRAO,
        purge_janelas=PURGA_PADRAO,
        n_blocos=N_BLOCOS_PADRAO,
    )


def split_treino_val(n_janelas: int, val_frac: float = 0.2,
                     purge_janelas: int = PURGA_PADRAO):
    """
    Divisão TEMPORAL treino/validação em 2 blocos contíguos com purga (sem
    bloco de teste). Treino = bloco inicial; validação = bloco final; a purga
    descarta janelas na fronteira para que janelas sobrepostas não vazem entre
    os conjuntos. Retorna (idx_treino, idx_val) como np.ndarray. Determinístico.
    """
    if n_janelas <= 0:
        raise ValueError("n_janelas deve ser > 0.")
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac deve estar em (0,1); recebido {val_frac}.")
    if purge_janelas < 0:
        raise ValueError("purge_janelas deve ser >= 0.")

    idx = np.arange(n_janelas, dtype=int)
    n_val = int(np.floor(n_janelas * val_frac))
    fim_treino = n_janelas - n_val - purge_janelas
    if fim_treino <= 0 or n_val <= 0:
        raise ValueError(
            f"Janelas insuficientes ({n_janelas}) para treino/val com purga="
            f"{purge_janelas} e val_frac={val_frac}."
        )
    return idx[:fim_treino], idx[fim_treino + purge_janelas:]
