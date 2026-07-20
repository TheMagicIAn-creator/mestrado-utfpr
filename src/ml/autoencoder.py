"""
autoencoder.py — Al IAdo PV / Fase 5
Modelagem de normalidade com Autoencoder para detecção de anomalias
no lado CA do inversor fotovoltaico.

Fundamentação:
  O Autoencoder aprende a reconstruir o comportamento SAUDÁVEL do inversor
  a partir do dataset de Paderborn. Em operação real, sinais anômalos
  (falhas) produzem erro de reconstrução alto — acima do limiar operacional
  (percentil 99 do erro saudável). μ + 3σ é mantido apenas como referência
  teórica comparativa, não como limiar operacional.

  Esta abordagem é adequada porque dados de falha raramente estão
  disponíveis em manutenção preditiva real (Ibrahim, 2022; Ahirwar, 2025).

Arquitetura:
  Entrada : n_features normalizadas (RobustScaler)
  Encoder : n_features → 64 → 32 → 16  (ReLU + Dropout 0.2)
  Latente : 16 dimensões
  Decoder : 16 → 32 → 64 → n_features  (ReLU + saída Linear)
  Loss    : MSE — erro de reconstrução por janela
  Limiar  : percentil 99 do erro de reconstrução saudável (operacional);
            μ + 3σ é referência comparativa, não o limiar em uso

Entrada : dados/processados/features_paderborn.parquet
Saída   : resultados/autoencoder/
            modelo_autoencoder.pt   ← pesos do modelo
            scaler.pkl              ← RobustScaler ajustado
            limiar.json             ← limiar de anomalia + metadados
            curva_treino.png        ← loss por época
            distribuicao_erro.png   ← distribuição do erro + limiar

Uso:
  python src/ml/autoencoder.py
  python src/ml/autoencoder.py --epochs 200 --latente 8

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

_logger = _get_logger("autoencoder")


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



import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ml.estilo_graficos import PALETA, TAM, aplicar_estilo

aplicar_estilo()
import matplotlib
matplotlib.use("Agg")   # sem display — salva direto em arquivo

from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# ── Caminhos ─────────────────────────────────────────────────
RAIZ           = Path(__file__).parent.parent.parent
ARQUIVO_FEAT   = RAIZ / "dados" / "processados" / "features_paderborn.parquet"
PASTA_SAIDA    = RAIZ / "resultados" / "autoencoder"

# ── Hiperparâmetros padrão ────────────────────────────────────
LATENTE_DIM    = 16     # dimensão do espaço latente
EPOCHS         = 150    # épocas de treinamento
BATCH_SIZE     = 32     # amostras por batch
LR             = 1e-3   # taxa de aprendizado (Adam)
DROPOUT        = 0.2    # regularização
VAL_FRAC       = 0.2    # fração de validação
PACIENCIA      = 20     # early stopping: épocas sem melhora
SIGMA          = 3.0    # fator k da REFERÊNCIA μ+kσ (comparativa); o limiar
                        # operacional é o percentil 99, não μ+kσ
THRESHOLD_METHOD = "p99"
SEED           = 42

# Colunas de metadado (não entram no modelo)
META_COLS = ["janela_idx", "amostra_inicio", "tempo_s"]


# ============================================================
# ARQUITETURA DO AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):
    """
    Autoencoder simétrico com Dropout para regularização.

    O encoder comprime as features em um espaço latente de baixa
    dimensão que captura o padrão normal do inversor.
    O decoder reconstrói as features originais a partir do latente.
    Janelas de falha terão alto erro de reconstrução.
    """

    def __init__(self, n_features: int, latente_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latente_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latente_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_features),
            # Saída linear — sem ativação, features normalizadas
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna representação latente (útil para visualização)."""
        return self.encoder(x)


# ============================================================
# TREINO
# ============================================================

def treinar(modelo, loader_treino, loader_val,
            epochs: int, lr: float, paciencia: int,
            device: torch.device) -> tuple:
    """
    Loop de treinamento com early stopping.
    Retorna (historico_treino, historico_val, epoca_melhor).
    """
    criterio  = nn.MSELoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, patience=10, factor=0.5
    )

    hist_treino, hist_val = [], []
    melhor_val  = float("inf")
    melhor_pesos = None
    sem_melhora  = 0
    epoca_melhor = 0

    for epoca in range(1, epochs + 1):

        # ── Treino ───────────────────────────────────────────
        modelo.train()
        loss_treino = 0.0
        for batch in loader_treino:
            x = batch[0].to(device)
            otimizador.zero_grad()
            x_rec = modelo(x)
            loss  = criterio(x_rec, x)
            loss.backward()
            otimizador.step()
            loss_treino += loss.item() * len(x)
        loss_treino /= len(loader_treino.dataset)

        # ── Validação ─────────────────────────────────────────
        modelo.eval()
        loss_val = 0.0
        with torch.no_grad():
            for batch in loader_val:
                x     = batch[0].to(device)
                x_rec = modelo(x)
                loss_val += criterio(x_rec, x).item() * len(x)
        loss_val /= len(loader_val.dataset)

        hist_treino.append(loss_treino)
        hist_val.append(loss_val)
        scheduler.step(loss_val)

        # Early stopping
        if loss_val < melhor_val - 1e-6:
            melhor_val   = loss_val
            melhor_pesos = {k: v.clone() for k, v in modelo.state_dict().items()}
            sem_melhora  = 0
            epoca_melhor = epoca
        else:
            sem_melhora += 1

        if epoca % 10 == 0 or epoca == 1:
            _log(f"   Época {epoca:>4}/{epochs} | "
                  f"treino: {loss_treino:.6f} | "
                  f"val: {loss_val:.6f}"
                  + (" ✓" if sem_melhora == 0 else ""))

        if sem_melhora >= paciencia:
            _log(f"\n   ⏹️  Early stopping na época {epoca} "
                  f"(melhor val={melhor_val:.6f} na época {epoca_melhor})")
            break

    # Restaura melhores pesos
    if melhor_pesos:
        modelo.load_state_dict(melhor_pesos)

    return hist_treino, hist_val, epoca_melhor


# ============================================================
# CÁLCULO DO LIMIAR
# ============================================================

def calcular_erros(modelo, X_tensor: torch.Tensor,
                   device: torch.device) -> np.ndarray:
    """
    Calcula o erro de reconstrução (MSE) por janela.
    Retorna array de shape (n_janelas,).
    """
    modelo.eval()
    erros = []
    with torch.no_grad():
        # Processa em batches para não estourar memória
        for i in range(0, len(X_tensor), 64):
            batch = X_tensor[i:i+64].to(device)
            rec   = modelo(batch)
            mse   = ((batch - rec) ** 2).mean(dim=1)
            erros.extend(mse.cpu().numpy())
    return np.array(erros)


def calcular_limiar(erros_treino: np.ndarray,
                    sigma: float = SIGMA) -> dict:
    """
    Define o limiar de anomalia do Autoencoder.

    DEFINIÇÃO OFICIAL (não confundir):
    - Limiar OPERACIONAL = percentil 99 do erro de reconstrução saudável.
      Controla diretamente a taxa de falso positivo (~1%) e é robusto a
      distribuições assimétricas com poucas janelas.
    - Referência COMPARATIVA = μ + 3σ (assume normalidade; só para comparação
      teórica, NUNCA usado como limiar operacional).
    - Referência ADICIONAL = percentil 95.

    O campo `threshold_method` registra explicitamente o método em uso.
    """
    mu      = float(erros_treino.mean())
    sig     = float(erros_treino.std())
    p99     = float(np.percentile(erros_treino, 99))
    p95     = float(np.percentile(erros_treino, 95))
    mu_3sig = mu + sigma * sig

    return {
        "threshold_method"  : "p99",        # método operacional em uso
        "limiar"            : p99,          # operacional (chave de compat. retroativa)
        "limiar_operacional": p99,          # operacional explícito = percentil 99
        "mu"                : mu,
        "sigma"             : sig,
        "k"                 : sigma,
        "limiar_p99"        : p99,          # operacional: percentil 99
        "limiar_p95"        : p95,          # referência adicional
        "limiar_mu3sigma"   : mu_3sig,      # referência teórica comparativa
        "limiar_mu3s"       : mu_3sig,      # alias de compat. retroativa
    }


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plotar_curvas(hist_treino: list, hist_val: list,
                  epoca_melhor: int, pasta: Path):
    """Curvas de loss por época."""
    fig, ax = plt.subplots(figsize=TAM["unico"])
    epocas = range(1, len(hist_treino) + 1)
    ax.plot(epocas, hist_treino, label="Treino",     color=PALETA[0])
    ax.plot(epocas, hist_val,   label="Validação",   color=PALETA[1])
    ax.axvline(epoca_melhor, color="green", linestyle="--",
               alpha=0.7, label=f"Melhor época ({epoca_melhor})")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder — Curva de Treinamento")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    caminho = pasta / "curva_treino.png"
    fig.savefig(caminho)
    plt.close(fig)
    _log(f"   📊 {caminho.name}")


def plotar_distribuicao(erros_treino: np.ndarray,
                        erros_val: np.ndarray,
                        info_limiar: dict, pasta: Path):
    """
    CALIBRAÇÃO DO DETECTOR (não é análise de falha): histograma do erro de
    reconstrução (MSE) do Autoencoder em dados SAUDÁVEIS (treino + validação),
    usado para fixar o limiar operacional p99. Uma anomalia real cairia à
    DIREITA do limiar; a fração da validação saudável acima do limiar é a taxa
    de falsos positivos. Não representa nenhum componente/modo da FMECA.
    """
    fig, ax = plt.subplots(figsize=TAM["unico"])

    ax.hist(erros_treino, bins=30, alpha=0.6,
            color=PALETA[0], label="Treino (saudável)")
    ax.hist(erros_val, bins=20, alpha=0.6,
            color=PALETA[1], label="Validação (saudável)")

    limiar = info_limiar["limiar"]
    ax.axvline(limiar, color="#d03b3b", linewidth=2, linestyle="--",
               label=f"Limiar operacional p99 = {limiar:.4f}")

    # μ+kσ entra apenas como REFERÊNCIA comparativa (não é o limiar em uso).
    mu3s = info_limiar.get("limiar_mu3sigma", info_limiar.get("limiar_mu3s"))
    if mu3s is not None:
        ax.axvline(mu3s, color="#898781", linewidth=1.5, linestyle=":",
                   label=f"Referência μ+{info_limiar['k']:.0f}σ = {mu3s:.4f}")

    fp = info_limiar.get("fp_val_pct")
    subt = "anomalias cairiam à direita do limiar"
    if isinstance(fp, (int, float)):
        subt += f" · FP validação = {fp:.2f}%"

    ax.set_xlabel("Erro de reconstrução (MSE) — dados SAUDÁVEIS")
    ax.set_ylabel("Frequência (nº de janelas)")
    ax.set_title("Calibração do detector — distribuição do erro saudável\n"
                 f"({subt})", fontsize=11)
    ax.legend()
    fig.tight_layout()
    caminho = pasta / "distribuicao_erro.png"
    fig.savefig(caminho)
    plt.close(fig)
    _log(f"   📊 {caminho.name}")


def plotar_erro_temporal(erros: np.ndarray,
                         tempos: np.ndarray,
                         info_limiar: dict, pasta: Path):
    """Erro de reconstrução ao longo do tempo."""
    fig, ax = plt.subplots(figsize=TAM["unico"])
    ax.plot(tempos, erros, color=PALETA[0], alpha=0.8, linewidth=0.8)
    ax.axhline(info_limiar["limiar"], color="red", linestyle="--",
               linewidth=1.5, label=f"Limiar = {info_limiar['limiar']:.4f}")
    ax.fill_between(tempos, erros, info_limiar["limiar"],
                    where=erros > info_limiar["limiar"],
                    color="red", alpha=0.3, label="Anomalia detectada")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Erro de Reconstrução (MSE)")
    ax.set_title("Erro Temporal — Dataset de Paderborn (saudável)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    caminho = pasta / "erro_temporal.png"
    fig.savefig(caminho)
    plt.close(fig)
    _log(f"   📊 {caminho.name}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_autoencoder(
    arquivo_feat : Path  = ARQUIVO_FEAT,
    pasta_saida  : Path  = PASTA_SAIDA,
    latente_dim  : int   = LATENTE_DIM,
    epochs       : int   = EPOCHS,
    batch_size   : int   = BATCH_SIZE,
    lr           : float = LR,
    paciencia    : int   = PACIENCIA,
    sigma        : float = SIGMA,
    seed         : int   = SEED,
) -> bool:
    """Pipeline completo de treinamento do Autoencoder."""

    _log("=" * 60)
    _log("  AL IADO PV — AUTOENCODER (Paderborn)")
    _log("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Dispositivo ──────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"\n🖥️  Dispositivo: {device}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if device.type == "cuda" else ""))

    # ── 1. Carrega features ──────────────────────────────────
    _log(f"\n📂 Carregando features...")
    if not arquivo_feat.exists():
        _log(f"   ❌ Não encontrado: {arquivo_feat}")
        _log("   Execute primeiro: python src/ml/features_ca.py")
        return False

    df = pd.read_parquet(arquivo_feat)
    tempos = df["tempo_s"].values if "tempo_s" in df.columns else None

    colunas_feat = [c for c in df.columns if c not in META_COLS]
    X = df[colunas_feat].values.astype(np.float32)
    n_janelas, n_features = X.shape
    _log(f"   ✅ {n_janelas} janelas × {n_features} features")

    # ── 2. Normalização com RobustScaler ─────────────────────
    # RobustScaler usa mediana e IQR — resistente a outliers
    # (THD alto em transientes não distorce a escala geral)
    _log(f"\n⚖️  Normalizando com RobustScaler...")
    # Divisão TEMPORAL com purga (NÃO aleatória): janelas com 50% de
    # sobreposição não podem vazar entre treino e validação (item 3.4).
    from src.ml.split_temporal import split_treino_val

    idx_tr, idx_val = split_treino_val(len(X), val_frac=VAL_FRAC, purge_janelas=2)
    X_treino_raw, X_val_raw = X[idx_tr], X[idx_val]
    scaler = RobustScaler()
    X_treino = scaler.fit_transform(X_treino_raw).astype(np.float32)
    X_val    = scaler.transform(X_val_raw).astype(np.float32)
    X_all    = scaler.transform(X).astype(np.float32)
    _log(f"   Treino : {len(X_treino)} janelas")
    _log(f"   Val    : {len(X_val)} janelas")

    # ── 3. DataLoaders ───────────────────────────────────────
    ds_treino = TensorDataset(torch.from_numpy(X_treino))
    ds_val    = TensorDataset(torch.from_numpy(X_val))
    loader_treino = DataLoader(ds_treino, batch_size=batch_size, shuffle=True)
    loader_val    = DataLoader(ds_val,    batch_size=batch_size, shuffle=False)

    # ── 4. Modelo ────────────────────────────────────────────
    modelo = Autoencoder(n_features, latente_dim, DROPOUT).to(device)
    n_params = sum(p.numel() for p in modelo.parameters())

    _log(f"\n🧠 Arquitetura:")
    _log(f"   Entrada  : {n_features}")
    _log(f"   Encoder  : {n_features} → 64 → 32 → {latente_dim}")
    _log(f"   Latente  : {latente_dim} dimensões")
    _log(f"   Decoder  : {latente_dim} → 32 → 64 → {n_features}")
    _log(f"   Parâmetros: {n_params:,}")

    # ── 5. Treinamento ───────────────────────────────────────
    _log(f"\n🏋️  Treinando ({epochs} épocas, early stopping={paciencia})...")
    hist_t, hist_v, ep_melhor = treinar(
        modelo, loader_treino, loader_val,
        epochs, lr, paciencia, device
    )

    # ── 6. Erros de reconstrução ─────────────────────────────
    _log(f"\n📐 Calculando erros de reconstrução...")
    T_treino = torch.from_numpy(X_treino)
    T_val    = torch.from_numpy(X_val)
    T_all    = torch.from_numpy(X_all)

    erros_treino = calcular_erros(modelo, T_treino, device)
    erros_val    = calcular_erros(modelo, T_val,    device)
    erros_all    = calcular_erros(modelo, T_all,    device)

    # ── 7. Limiar de anomalia ────────────────────────────────
    info_limiar = calcular_limiar(erros_treino, sigma)
    limiar      = info_limiar["limiar"]

    _log(f"\n🎯 Limiares de anomalia:")
    _log(f"   μ (treino)     = {info_limiar['mu']:.6f}")
    _log(f"   σ (treino)     = {info_limiar['sigma']:.6f}")
    _log(f"   Percentil 99   = {info_limiar['limiar_p99']:.6f}  ← operacional")
    _log(f"   Percentil 95   = {info_limiar['limiar_p95']:.6f}")
    _log(f"   μ + {sigma}σ        = {info_limiar['limiar_mu3s']:.6f}  ← referência teórica")

    # Taxa de falso positivo no conjunto de validação
    fp_val = (erros_val > limiar).mean() * 100
    fp_all = (erros_all > limiar).mean() * 100
    _log(f"\n   Falsos positivos (val): {fp_val:.1f}%")
    _log(f"   Falsos positivos (all): {fp_all:.1f}%")
    _log(f"   (limiar p99 alveja FP ≈ 1%; μ+3σ daria ≈ 0,3% se o erro fosse normal)")

    # ── 8. Salva artefatos ───────────────────────────────────
    _log(f"\n💾 Salvando artefatos...")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Modelo PyTorch
    arq_modelo = pasta_saida / "modelo_autoencoder.pt"
    torch.save({
        "state_dict"  : modelo.state_dict(),
        "n_features"  : n_features,
        "latente_dim" : latente_dim,
        "colunas_feat": colunas_feat,
        "data_treino" : datetime.now().isoformat(),
    }, arq_modelo)
    _log(f"   ✅ {arq_modelo.name}")

    # Scaler (+ sidecar SHA-256 para carga verificada nas etapas seguintes)
    arq_scaler = pasta_saida / "scaler.pkl"
    with open(arq_scaler, "wb") as f:
        pickle.dump(scaler, f)
    from src.core.seguranca import gravar_sidecar_sha256

    gravar_sidecar_sha256(arq_scaler)
    _log(f"   ✅ {arq_scaler.name}")

    # Limiar e metadados
    arq_limiar = pasta_saida / "limiar.json"
    metadados  = {
        **info_limiar,
        "n_janelas_treino"  : len(X_treino),
        "n_features"        : n_features,
        "latente_dim"       : latente_dim,
        "epochs_treinadas"  : len(hist_t),
        "epoca_melhor"      : ep_melhor,
        "loss_val_melhor"   : float(min(hist_v)),
        "fp_val_pct"        : float(fp_val),
        "data_treino"       : datetime.now().isoformat(),
        "device"            : str(device),
    }
    with open(arq_limiar, "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_limiar.name}")

    # ── 9. Visualizações ─────────────────────────────────────
    _log(f"\n📊 Gerando gráficos...")
    plotar_curvas(hist_t, hist_v, ep_melhor, pasta_saida)
    plotar_distribuicao(erros_treino, erros_val, info_limiar, pasta_saida)
    if tempos is not None:
        plotar_erro_temporal(erros_all, tempos, info_limiar, pasta_saida)

    # ── 10. Resumo final ─────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  AUTOENCODER CONCLUÍDO!")
    _log(f"  Janelas treino  : {len(X_treino)}")
    _log(f"  Features        : {n_features}")
    _log(f"  Épocas treinadas: {len(hist_t)}")
    _log(f"  Loss val melhor : {min(hist_v):.6f}")
    _log(f"  Limiar anomalia : {limiar:.6f}")
    _log(f"  Falsos positivos: {fp_val:.1f}% (val)")
    _log(f"  Artefatos em    : resultados/autoencoder/")
    _log(f"\n  Próximo passo: injeção de falhas sintéticas")
    _log(f"  (src/ml/injecao_falhas.py)")
    _log(f"{'='*60}")
    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    parser = argparse.ArgumentParser(
        description="Treina Autoencoder para detecção de anomalias no lado CA"
    )
    parser.add_argument("--epochs",  type=int,   default=EPOCHS,
                        help=f"Épocas de treinamento (padrão: {EPOCHS})")
    parser.add_argument("--latente", type=int,   default=LATENTE_DIM,
                        help=f"Dimensão do espaço latente (padrão: {LATENTE_DIM})")
    parser.add_argument("--lr",      type=float, default=LR,
                        help=f"Taxa de aprendizado (padrão: {LR})")
    parser.add_argument("--sigma",   type=float, default=SIGMA,
                        help=f"Fator do limiar μ+k*σ (padrão: {SIGMA})")
    args = parser.parse_args()

    executar_autoencoder(
        latente_dim = args.latente,
        epochs      = args.epochs,
        lr          = args.lr,
        sigma       = args.sigma,
    )
