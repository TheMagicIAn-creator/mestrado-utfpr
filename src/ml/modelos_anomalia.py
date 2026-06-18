"""
modelos_anomalia.py — Al IAdo PV

Zoo de SCORERS de detecção de anomalia compartilhados pelos experimentos por
artigo e pelos protocolos de decisão. Cada função recebe o dict ``dados``
(preparado por experimentos_artigos/protocolos_artigos) e devolve um vetor de
score por janela do teste — quanto maior, mais anômalo.

Por que este módulo existe (arquitetura):
- ``experimentos_artigos.py`` e ``protocolos_artigos.py`` precisavam destes
  scorers, mas importavam um do outro, criando um ciclo. Extrair os scorers para
  um TERCEIRO módulo NEUTRO quebra o ciclo: este módulo é uma FOLHA — depende só
  de bibliotecas externas (numpy/torch/prophet/sklearn/sb3/gymnasium), nunca de
  experimentos_artigos nem de protocolos_artigos.
- Todos os imports pesados são LOCAIS (dentro de cada função), então importar
  este módulo é barato e não dispara torch/prophet na carga.

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


def _treinar_clf_torch(model, dados, epochs: int = 60):
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    Xtr = torch.tensor(dados["X_tr_sup"], dtype=torch.float32)
    ytr = torch.tensor(dados["y_tr_sup"], dtype=torch.float32)
    Xte = torch.tensor(dados["X_te"], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte)).numpy()


def _score_rnn_torch(dados):
    import torch.nn as nn

    n_feat = dados["X_te"].shape[1]

    class RNNClf(nn.Module):
        def __init__(self, hid=32):
            super().__init__()
            self.rnn = nn.LSTM(1, hid, batch_first=True)
            self.fc = nn.Linear(hid, 1)

        def forward(self, x):
            _, (h, _) = self.rnn(x.unsqueeze(-1))
            return self.fc(h[-1]).squeeze(-1)

    return _treinar_clf_torch(RNNClf(), dados)


def _score_cnn_torch(dados):
    import torch.nn as nn

    class CNNClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Linear(16 * 8, 1)

        def forward(self, x):
            z = self.conv(x.unsqueeze(1))
            return self.fc(z.flatten(1)).squeeze(-1)

    return _treinar_clf_torch(CNNClf(), dados)


# ---- Facebook Prophet (univariado sobre a feature mais informativa) --------

def _score_prophet(dados, interval_width: float = 0.80):
    """
    Prophet aplicado à feature de maior variância no normal: aprende o nível e
    a banda de incerteza; o score é o desvio do valor em relação à banda
    (score > 1 ⇒ fora da banda — decisão NATIVA do modelo, sem oráculo).
    ``interval_width`` controla a banda (protocolo Ibrahim/Ahirwar usa 0,99).
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


# ---- Isolation Forest auto-ajustado por RL (PPO) ---------------------------

def _ppo_buscar_contaminacao(Xn_fit, Xval, yval, timesteps: int = 600,
                             metrica: str = "auc") -> float:
    """
    Busca a 'contamination' do Isolation Forest via PPO (Sharma et al., 2026).
    Ambiente de 1 passo (bandit): ação → contamination; recompensa → métrica
    na VALIDAÇÃO fornecida pelo chamador (o teste nunca entra aqui).
    ``metrica``: "auc" (independente de limiar) ou "f1" (decisional).
    """
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import f1_score, roc_auc_score
    from stable_baselines3 import PPO

    def recompensa(cont):
        cont = float(np.clip(cont, 0.01, 0.45))
        iso = IsolationForest(n_estimators=120, contamination=cont, random_state=42)
        iso.fit(Xn_fit)
        if metrica == "f1":
            y_pred = (iso.predict(Xval) == -1).astype(int)
            return float(f1_score(yval, y_pred, zero_division=0))
        return float(roc_auc_score(yval, -iso.decision_function(Xval)))

    class EnvIForest(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            cont = 0.01 + (float(action[0]) + 1.0) / 2.0 * 0.44
            reward = recompensa(cont)
            return np.zeros(1, dtype=np.float32), reward, True, False, {"cont": cont}

    modelo = PPO("MlpPolicy", EnvIForest(), seed=42, verbose=0,
                 n_steps=64, batch_size=32)
    modelo.learn(total_timesteps=timesteps)

    obs = np.zeros(1, dtype=np.float32)
    accao, _ = modelo.predict(obs, deterministic=True)
    return float(np.clip(0.01 + (float(accao[0]) + 1.0) / 2.0 * 0.44, 0.01, 0.45))
