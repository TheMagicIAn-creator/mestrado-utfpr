"""
modelos_anomalia.py — Al IAdo PV

Scorer de detecção de anomalia NÃO-supervisionada usado pelo protocolo por
artigo do núcleo (Ibrahim). Modelagem de NORMALIDADE (sem rótulo de falha):

- ``_score_ae_lstm`` — Autoencoder-LSTM TEMPORAL (Ibrahim, 2022). A LSTM
  percorre o eixo do TEMPO — uma SEQUÊNCIA de janelas consecutivas — como no
  artigo (que modela a "correlação na série temporal"). A versão anterior
  rodava a LSTM sobre o eixo das FEATURES (ordem arbitrária), o que NÃO era
  fiel ao Ibrahim: era uma camada densa cara disfarçada de recorrente
  (ver docs/auditoria_pipeline_ml.md §10.1). Corrigido aqui.

  Escolha metodológica (documentada): como a injeção do protocolo é pontual
  (uma janela por vez), cada item é pontuado como "a janela ATUAL dado o
  histórico normal precedente" — o escore é o erro de reconstrução no ÚLTIMO
  passo da sequência. Assim o modelo é genuinamente temporal e o banco de
  teste permanece o MESMO dos outros modelos (comparável por AUC).

O Isolation Forest do Ibrahim vive direto no protocolo (é um
sklearn.ensemble.IsolationForest, não precisa de scorer). Removidos na
curadoria: scorers supervisionados/RL (Sharma) e o Facebook Prophet.

Por que este módulo existe (arquitetura): ``experimentos_artigos.py`` e
``protocolos_artigos.py`` precisavam deste scorer sem importar um do outro —
este módulo é uma FOLHA (depende só de numpy/torch). O import pesado é LOCAL.
NÃO importe experimentos_artigos nem protocolos_artigos aqui.
"""

from __future__ import annotations


# ---- Construção de sequências temporais (numpy puro) -----------------------

def sequencias_deslizantes(Xn, L: int):
    """Janelas deslizantes sobre o fluxo normal ORDENADO no tempo.

    Xn: (n, F) janelas normais em ordem temporal. Retorna (max(1,n-L+1), L, F).
    Se n < L, faz padding com a primeira janela para garantir 1 sequência.
    """
    import numpy as np

    Xn = np.asarray(Xn, dtype=np.float32)
    n, F = Xn.shape
    if n < L:
        pad = np.repeat(Xn[:1], L - n, axis=0)
        return np.concatenate([pad, Xn])[None, :, :]
    return np.stack([Xn[i - L + 1:i + 1] for i in range(L - 1, n)])


def sequencias_com_contexto(contexto_normal, itens, L: int):
    """Uma sequência por item: L-1 predecessores NORMAIS + o item no último passo.

    contexto_normal: (n, F) fluxo normal ordenado (o "histórico" real).
    itens: (n, F) janelas a pontuar (limpas OU injetadas), alinhadas 1:1 com a
    posição temporal de contexto_normal. Retorna (n, L, F). No início (i<L-1)
    faz padding com a primeira janela normal.
    """
    import numpy as np

    ctx = np.asarray(contexto_normal, dtype=np.float32)
    it = np.asarray(itens, dtype=np.float32)
    n, F = it.shape
    n_ctx = len(ctx)
    seqs = np.zeros((n, L, F), dtype=np.float32)
    for i in range(n):
        for t in range(L):
            if t == L - 1:
                seqs[i, t] = it[i]                 # último passo = a janela atual
            else:
                # passo do histórico; limitado ao intervalo válido do contexto
                # (o contexto pode ser menor que os itens — ex.: contexto vem do
                # bloco de calibração e os itens do bloco de avaliação).
                pos = min(max(i - (L - 1) + t, 0), n_ctx - 1)
                seqs[i, t] = ctx[pos]
    return seqs


# ---- Autoencoder-LSTM temporal (PyTorch) -----------------------------------

def _score_ae_lstm(seq_fit, seq_eval, epochs: int = 60, seed: int = 42):
    """AE-LSTM TEMPORAL: reconstrói sequências de janelas ao longo do TEMPO.

    seq_fit:  (m, L, F) sequências NORMAIS para treino.
    seq_eval: (k, L, F) sequências a pontuar.
    Retorna score (k,) = MSE de reconstrução no ÚLTIMO passo (a janela atual
    dado o histórico). Fiel ao Ibrahim: a LSTM percorre o eixo TEMPORAL.
    """
    return pontuar_ae_lstm(treinar_ae_lstm(seq_fit, epochs=epochs, seed=seed), seq_eval)


def treinar_ae_lstm(seq_fit, epochs: int = 60, seed: int = 42):
    """Treina o AE-LSTM temporal em sequências NORMAIS. Retorna o modelo treinado
    (para pontuar várias vezes sem re-treinar — usado pelos macro-códigos)."""
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    Xf = torch.tensor(np.asarray(seq_fit, dtype=np.float32))    # (m, L, F)
    n_feat = Xf.shape[2]

    class AELSTM(nn.Module):
        def __init__(self, n_feat, hid=32, lat=8):
            super().__init__()
            self.enc = nn.LSTM(n_feat, hid, batch_first=True)
            self.to_lat = nn.Linear(hid, lat)
            self.from_lat = nn.Linear(lat, hid)
            self.dec = nn.LSTM(hid, hid, batch_first=True)
            self.out = nn.Linear(hid, n_feat)

        def forward(self, x):                       # x: (B, L, F)
            _, (h, _) = self.enc(x)                 # h[-1]: (B, hid)
            lat = self.to_lat(h[-1])                # (B, lat)
            dec_in = self.from_lat(lat).unsqueeze(1).repeat(1, x.size(1), 1)
            dec_out, _ = self.dec(dec_in)           # (B, L, hid)
            return self.out(dec_out)                # (B, L, F)

    model = AELSTM(n_feat)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xf), Xf)
        loss.backward()
        opt.step()
    model.eval()
    return model


def pontuar_ae_lstm(model, seq_eval):
    """Erro de reconstrução no ÚLTIMO passo de cada sequência (a janela atual
    dado o histórico normal). Não re-treina."""
    import numpy as np
    import torch

    Xe = torch.tensor(np.asarray(seq_eval, dtype=np.float32))   # (k, L, F)
    with torch.no_grad():
        rec = model(Xe)
        return ((rec[:, -1, :] - Xe[:, -1, :]) ** 2).mean(dim=1).numpy()
