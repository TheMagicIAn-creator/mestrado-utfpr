"""
modelos_anomalia.py — Al IAdo PV

Scorer de detecção de anomalia NÃO-supervisionada usado pelo protocolo por
artigo do núcleo (Ibrahim). Recebe o dict ``dados`` e devolve um vetor de
score por janela do teste — quanto maior, mais anômalo.

Resta um, por modelagem de NORMALIDADE (sem rótulo de falha):
- ``_score_ae_lstm``  — Autoencoder-LSTM (erro de reconstrução)
Removidos na curadoria do mestrado: os scorers supervisionados/RL (RNN/CNN/PPO,
com o experimento Sharma) e o Facebook Prophet (pior detector do Ibrahim e
dependência instável em runtime). O Isolation Forest do Ibrahim vive direto no
protocolo (é só um sklearn.ensemble.IsolationForest, não precisa de scorer).

Por que este módulo existe (arquitetura):
- ``experimentos_artigos.py`` e ``protocolos_artigos.py`` precisavam deste
  scorer, mas importavam um do outro, criando um ciclo. Mantê-lo num TERCEIRO
  módulo NEUTRO quebra o ciclo: este módulo é uma FOLHA — depende só de
  bibliotecas externas (numpy/torch), nunca de experimentos_artigos nem
  de protocolos_artigos.
- O import pesado é LOCAL (dentro da função), então importar este módulo é
  barato e não dispara torch na carga.

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


