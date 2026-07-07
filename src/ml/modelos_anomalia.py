"""
modelos_anomalia.py — Al IAdo PV

Scorers de detecção de anomalia NÃO-supervisionada usados pelo protocolo por
artigo do núcleo (Ibrahim). Cada função recebe o dict ``dados`` e
devolve um vetor de score por janela do teste — quanto maior, mais anômalo.

Restam dois, ambos por modelagem de NORMALIDADE (sem rótulo de falha):
- ``_score_ae_lstm``  — Autoencoder-LSTM (erro de reconstrução)
- ``_score_prophet``  — Facebook Prophet (desvio da banda de incerteza)
Os scorers supervisionados/RL (RNN/CNN/PPO) foram removidos na curadoria do
mestrado junto com o experimento Sharma.

Por que este módulo existe (arquitetura):
- ``experimentos_artigos.py`` e ``protocolos_artigos.py`` precisavam destes
  scorers, mas importavam um do outro, criando um ciclo. Mantê-los num TERCEIRO
  módulo NEUTRO quebra o ciclo: este módulo é uma FOLHA — depende só de
  bibliotecas externas (numpy/torch/prophet), nunca de experimentos_artigos nem
  de protocolos_artigos.
- Os imports pesados são LOCAIS (dentro de cada função), então importar este
  módulo é barato e não dispara torch/prophet na carga.

NÃO importe experimentos_artigos nem protocolos_artigos aqui (recriaria o ciclo).
"""

from __future__ import annotations


# ---- Redes neurais compactas (PyTorch) -------------------------------------

def _score_ae_lstm(dados, epochs: int = 60, retornar_treino: bool = False):
    """Autoencoder-LSTM: erro de reconstrução como score (fit no normal).

    Com ``retornar_treino=True`` devolve ``(score_teste, score_treino)`` —
    o erro no próprio treino permite CONGELAR um limiar (ex.: p99) antes de
    olhar o teste, como nos protocolos por artigo.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    Xn = torch.tensor(dados["Xn_tr"], dtype=torch.float32)
    Xte = torch.tensor(dados["X_te"], dtype=torch.float32)
    n_feat = Xn.shape[1]

    class AELSTM(nn.Module):
        def __init__(self, hid=32, lat=8):
            super().__init__()
            self.enc = nn.LSTM(1, hid, batch_first=True)
            self.to_lat = nn.Linear(hid, lat)
            self.from_lat = nn.Linear(lat, hid)
            self.dec = nn.LSTM(hid, hid, batch_first=True)
            self.out = nn.Linear(hid, 1)

        def forward(self, x):
            seq = x.unsqueeze(-1)                       # (B, F, 1)
            _, (h, _) = self.enc(seq)
            lat = self.to_lat(h[-1])                    # (B, lat)
            dec_in = self.from_lat(lat).unsqueeze(1).repeat(1, seq.size(1), 1)
            dec_out, _ = self.dec(dec_in)
            return self.out(dec_out).squeeze(-1)        # (B, F)

    model = AELSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xn), Xn)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        rec = model(Xte)
        score_te = ((rec - Xte) ** 2).mean(dim=1).numpy()
        if not retornar_treino:
            return score_te
        rec_tr = model(Xn)
        score_tr = ((rec_tr - Xn) ** 2).mean(dim=1).numpy()
        return score_te, score_tr


# ---- Facebook Prophet (univariado sobre a feature mais informativa) --------

def _score_prophet(dados, interval_width: float = 0.80):
    """
    Prophet aplicado à feature de maior variância no normal: aprende o nível e
    a banda de incerteza; o score é o desvio do valor em relação à banda
    (score > 1 ⇒ fora da banda — decisão NATIVA do modelo, sem oráculo).
    ``interval_width`` controla a banda (protocolo Ibrahim usa 0,99).
    Univariado por natureza — resultado honesto e mais modesto que o multivar.
    """
    import logging

    import numpy as np
    import pandas as pd

    logging.getLogger("prophet").setLevel(logging.CRITICAL)
    logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
    from prophet import Prophet

    Xn = dados["Xn_tr"]
    Xte = dados["X_te"]
    # Protocolos passam a coluna a monitorar (feature sensível às falhas
    # FMEA); sem ela, cai no comportamento antigo (maior variância).
    j = dados.get("col_prophet")
    j = int(np.argmax(Xn.var(axis=0))) if j is None else int(j)
    yn = Xn[:, j]
    ds = pd.date_range("2020-01-01", periods=len(yn), freq="D")
    df = pd.DataFrame({"ds": ds, "y": yn})

    m = Prophet(weekly_seasonality=False, yearly_seasonality=False,
                daily_seasonality=False, interval_width=interval_width)
    m.fit(df)
    fc = m.predict(df)
    centro = float(np.mean(fc["yhat"].to_numpy()))
    meia_banda = float(np.mean((fc["yhat_upper"] - fc["yhat_lower"]).to_numpy()) / 2)
    meia_banda = meia_banda if meia_banda > 1e-9 else 1.0
    return np.abs(Xte[:, j] - centro) / meia_banda

