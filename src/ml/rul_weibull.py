"""
rul_weibull.py — Al IAdo PV / Fase 5
Estimativa de Vida Útil Remanescente (RUL) com Análise de Weibull.

Fundamentação metodológica:
  O dataset de Paderborn contém apenas dados saudáveis (sem falhas reais).
  A estratégia adotada — definida na metodologia da dissertação — é gerar
  dados de tempo até a falha (TTF) por meio de trajetórias de degradação
  sintética progressiva, fundamentadas no FMEA do TCC (Torres, 2024).

  Cada trajetória simula um inversor que inicia saudável e degrada
  gradualmente (severidade 0→1,0 em N_STEPS janelas). O TTF é o passo
  em que o Autoencoder detecta a anomalia (erro > limiar).
  Com N_TRAJ trajetórias por tipo de falha, ajusta-se a distribuição
  de Weibull de 2 parâmetros aos TTF obtidos.

Distribuição de Weibull de 2 parâmetros:
  PDF : f(t) = (β/η) × (t/η)^(β−1) × exp(−(t/η)^β)
  CDF : F(t) = 1 − exp(−(t/η)^β)
  R(t): R(t) = exp(−(t/η)^β)         ← Função de Confiabilidade
  h(t): h(t) = (β/η) × (t/η)^(β−1)  ← Taxa de Falha

  Parâmetros:
    β (shape)  — inclinação de Weibull
                 β < 1: mortalidade infantil
                 β = 1: falhas aleatórias (exponencial)
                 β > 1: desgaste (esperado para degradação gradual)
    η (scale)  — vida característica: t em que R(t) = e^−1 ≈ 36,8%
    MTTF = η × Γ(1 + 1/β)

RUL estimado:
  Para um inversor com degradação observada em t_atual, a RUL esperada é:
    RUL = MTTF_condicional − t_atual
  onde MTTF_condicional usa a distribuição Weibull truncada em t_atual.

Entrada:
  resultados/autoencoder/modelo_autoencoder.pt
  resultados/autoencoder/scaler.pkl
  resultados/autoencoder/limiar.json
  dados/brutos/Inverter_Data_Set.csv

Saída:
  resultados/autoencoder/weibull_ttf.png
  resultados/autoencoder/weibull_confiabilidade.png
  resultados/autoencoder/weibull_rul.png
  resultados/autoencoder/weibull_results.json

Uso:
  python src/ml/rul_weibull.py

Autor: Rodolfo Torres (UTFPR)
"""

from src.core.logs import get_logger as _get_logger

_logger = _get_logger("rul_weibull")


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from scipy.stats import weibull_min
from scipy.special import gamma as gamma_func, gammaincc

import torch
from src.ml.features_ca   import extrair_janela, JANELA, FS
from src.ml.autoencoder   import Autoencoder
from src.ml.injecao_falhas import (
    FUNCOES_FALHA, FALHAS,
    T_INICIO_ESTAVEL, T_FIM_ESTAVEL
)

# ── Caminhos ─────────────────────────────────────────────────
RAIZ        = Path(__file__).parent.parent.parent
ARQUIVO_CSV = RAIZ / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_AE    = RAIZ / "resultados" / "autoencoder"

# ── Parâmetros de simulação ───────────────────────────────────
N_TRAJ  = 100    # trajetórias de degradação por tipo de falha
N_STEPS = 120    # passos de degradação por trajetória (sev 0→1,0)
# Cada passo representa um intervalo de monitoramento
# Em deployment real: ajustar conforme a frequência de aquisição
BATCH_INFERENCIA = 16


def calcular_erros_batch(vetores: np.ndarray,
                         modelo: Autoencoder,
                         scaler,
                         device: torch.device) -> np.ndarray:
    """Normaliza um lote de features e retorna o MSE por amostra."""
    vnorm = scaler.transform(vetores).astype(np.float32)
    with torch.inference_mode():
        x     = torch.from_numpy(vnorm).to(device)
        x_rec = modelo(x)
        erros = ((x - x_rec) ** 2).mean(dim=1).detach().cpu().numpy()
    return erros


# ============================================================
# TRAJETÓRIA DE DEGRADAÇÃO PROGRESSIVA
# ============================================================

def gerar_ttf(df_estavel: pd.DataFrame,
              modelo: Autoencoder,
              scaler,
              device: torch.device,
              colunas_feat: list,
              limiar: float,
              tipo_falha: str,
              n_steps: int,
              seed: int,
              batch_size: int = BATCH_INFERENCIA) -> int:
    """
    Simula uma trajetória de degradação progressiva e retorna o TTF.

    A severidade aumenta linearmente de 0 a 1,0 em n_steps passos.
    O TTF é o passo em que o erro de reconstrução cruza o limiar.
    Se não cruzar, retorna n_steps (censurado à direita).

    Parâmetros:
        seed : garante variabilidade entre trajetórias (diferentes
               janelas são selecionadas do período estável)
    """
    fn          = FUNCOES_FALHA[tipo_falha]
    rng         = np.random.default_rng(seed)
    severidades = np.linspace(0.0, 1.0, n_steps)
    n_disp      = len(df_estavel) - JANELA
    if n_disp <= 0:
        raise ValueError("Periodo estavel menor que a janela de extracao.")

    modelo.eval()

    for inicio_batch in range(0, n_steps, batch_size):
        fim_batch = min(inicio_batch + batch_size, n_steps)
        vetores = []

        for step in range(inicio_batch, fim_batch):
            sev = severidades[step]

            # Seleciona janela aleatória do período estável
            inicio = int(rng.integers(0, n_disp))
            janela = df_estavel.iloc[inicio:inicio + JANELA]

            if sev > 0.01:
                janela = fn(janela, float(sev))

            feats = extrair_janela(janela)
            vetores.append([feats.get(c, 0.0) for c in colunas_feat])

        erros = calcular_erros_batch(
            np.asarray(vetores, dtype=np.float32),
            modelo, scaler, device
        )
        cruzamentos = np.flatnonzero(erros > limiar)
        if len(cruzamentos) > 0:
            return inicio_batch + int(cruzamentos[0])

    return n_steps  # censurado: não detectado no horizonte


# ============================================================
# AJUSTE DE WEIBULL
# ============================================================

def ajustar_weibull(ttfs: np.ndarray) -> dict:
    """
    Ajusta distribuição de Weibull de 2 parâmetros (MLE).
    Retorna parâmetros e métricas de ajuste.

    floc=0 fixa a localização em zero (origem) — padrão para
    análise de confiabilidade de componentes sem período de garantia.
    """
    # MLE com localização fixada em 0
    shape, loc, scale = weibull_min.fit(ttfs, floc=0)
    beta = shape   # parâmetro de forma
    eta  = scale   # parâmetro de escala (vida característica)

    # MTTF = η × Γ(1 + 1/β)
    mttf = eta * gamma_func(1 + 1 / beta)

    # B10: tempo em que 10% das unidades falham
    b10  = eta * (-np.log(0.90)) ** (1 / beta)

    # Bondade de ajuste: KS test
    from scipy.stats import kstest
    ks_stat, ks_pval = kstest(ttfs, "weibull_min",
                               args=(shape, loc, scale))

    return {
        "beta"    : float(beta),
        "eta"     : float(eta),
        "mttf"    : float(mttf),
        "b10"     : float(b10),
        "ks_stat" : float(ks_stat),
        "ks_pval" : float(ks_pval),
        "n_traj"  : len(ttfs),
        "ttf_mean": float(np.mean(ttfs)),
        "ttf_std" : float(np.std(ttfs)),
        "ttf_min" : float(np.min(ttfs)),
        "ttf_max" : float(np.max(ttfs)),
    }


# ============================================================
# ESTIMATIVA DE RUL CONDICIONAL
# ============================================================

def rul_condicional(t_atual: float, beta: float, eta: float) -> float:
    """
    RUL esperado dado que o componente sobreviveu até t_atual.

    Pela propriedade de memória da Weibull:
      E[T - t | T > t] = integral_t^∞ R(s)/R(t) ds
                       = eta * exp(z) * Γ(1 + 1/beta, z) - t
      onde z = (t/eta)^beta e Γ(.,.) é a gama incompleta superior.
    """
    if beta <= 0 or eta <= 0:
        return float("nan")

    if t_atual <= 0:
        return eta * gamma_func(1 + 1 / beta)  # MTTF completo

    z = (t_atual / eta) ** beta
    s = 1 + 1 / beta

    if z > 700:
        # Aproximação assintótica evita overflow numérico para tempos extremos.
        return float((eta / beta) * (t_atual / eta) ** (1 - beta))

    gama_sup = gamma_func(s) * gammaincc(s, z)
    media_condicional = eta * np.exp(z) * gama_sup
    return float(max(media_condicional - t_atual, 0.0))


# ============================================================
# VISUALIZAÇÕES
# ============================================================

def plotar_ttf_histogramas(ttfs_dict: dict, params: dict, pasta: Path):
    """Histogramas de TTF com curva de Weibull ajustada."""
    n_falhas = len(FALHAS)
    fig, axes = plt.subplots(1, n_falhas, figsize=(15, 5))
    fig.suptitle("Distribuição do Tempo até Falha (TTF) — Ajuste de Weibull",
                 fontsize=13, fontweight="bold")

    for ax, falha in zip(axes, FALHAS):
        fid  = falha["id"]
        nome = falha["nome"]
        ttfs = ttfs_dict[fid]
        p    = params[fid]

        ax.hist(ttfs, bins=20, density=True, alpha=0.6,
                color=falha["cor"], label="TTF simulados")

        t_lin = np.linspace(0.1, max(ttfs) * 1.1, 200)
        pdf   = weibull_min.pdf(t_lin, p["beta"], loc=0, scale=p["eta"])
        ax.plot(t_lin, pdf, "k-", linewidth=2, label="Weibull ajustada")
        ax.axvline(p["mttf"], color="red", linestyle="--",
                   label=f"MTTF={p['mttf']:.1f}")
        ax.axvline(p["b10"],  color="blue", linestyle=":",
                   linewidth=1.5, label=f"B10={p['b10']:.1f}")

        npm_str = f"NPR={falha['npr']}" if falha['npr'] else "D=10"
        ax.set_title(f"{nome} ({npm_str})\n"
                     f"β={p['beta']:.2f}  η={p['eta']:.1f}  "
                     f"KS p={p['ks_pval']:.3f}", fontsize=9)
        ax.set_xlabel("TTF (passos de degradação)")
        ax.set_ylabel("Densidade de Probabilidade")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    arq = pasta / "weibull_ttf.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"   📊 {arq.name}")


def plotar_confiabilidade(ttfs_dict: dict, params: dict, pasta: Path):
    """Funções de confiabilidade R(t) e taxa de falha h(t)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Análise de Confiabilidade — Funções de Weibull",
                 fontsize=13, fontweight="bold")

    for col, falha in enumerate(FALHAS):
        fid  = falha["id"]
        p    = params[fid]
        ttfs = ttfs_dict[fid]
        t    = np.linspace(0.1, max(ttfs) * 1.2, 300)

        # Confiabilidade R(t)
        ax_r = axes[0][col]
        R    = weibull_min.sf(t, p["beta"], loc=0, scale=p["eta"])
        ax_r.plot(t, R, color=falha["cor"], linewidth=2)
        ax_r.axhline(0.368, color="gray", linestyle="--",
                     alpha=0.7, label=f"R=36,8% → η={p['eta']:.1f}")
        ax_r.axhline(0.90, color="blue", linestyle=":",
                     alpha=0.7, label=f"R=90% → B10={p['b10']:.1f}")
        ax_r.fill_between(t, R, alpha=0.15, color=falha["cor"])
        ax_r.set_ylim([0, 1.05])
        ax_r.set_xlabel("t (passos)")
        ax_r.set_ylabel("R(t) = P(T > t)")
        npm_str = f"NPR={falha['npr']}" if falha['npr'] else "D=10"
        ax_r.set_title(f"{falha['nome']}\nβ={p['beta']:.2f}, η={p['eta']:.1f} ({npm_str})",
                        fontsize=9)
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.3)

        # Taxa de falha h(t)
        ax_h = axes[1][col]
        H    = weibull_min.pdf(t, p["beta"], loc=0, scale=p["eta"]) / \
               np.maximum(weibull_min.sf(t, p["beta"], loc=0, scale=p["eta"]), 1e-10)
        ax_h.plot(t, H, color=falha["cor"], linewidth=2)
        beta_desc = ("crescente ↑" if p["beta"] > 1.1
                     else "constante →" if p["beta"] > 0.9
                     else "decrescente ↓")
        ax_h.set_title(f"Taxa de Falha h(t)\n(β={p['beta']:.2f} — {beta_desc})",
                        fontsize=9)
        ax_h.set_xlabel("t (passos)")
        ax_h.set_ylabel("h(t)")
        ax_h.grid(True, alpha=0.3)

    plt.tight_layout()
    arq = pasta / "weibull_confiabilidade.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"   📊 {arq.name}")


def plotar_rul(params: dict, pasta: Path):
    """RUL esperado em função do tempo atual t."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for falha in FALHAS:
        fid  = falha["id"]
        p    = params[fid]
        mttf = p["mttf"]

        # Avalia RUL em 20 pontos de t_atual (0 a 80% do MTTF)
        t_pontos = np.linspace(0, mttf * 0.8, 20)
        ruls     = [rul_condicional(t, p["beta"], p["eta"])
                    for t in t_pontos]

        npm_str = f"NPR={falha['npr']}" if falha['npr'] else "D=10"
        ax.plot(t_pontos / mttf * 100, ruls,
                color=falha["cor"], linewidth=2.5,
                label=f"{falha['nome']} ({npm_str}) — MTTF={mttf:.1f}")

    ax.set_xlabel("Tempo atual (% do MTTF)")
    ax.set_ylabel("RUL esperado (passos de degradação)")
    ax.set_title("Vida Útil Remanescente (RUL) Condicional\n"
                 "E[T − t | T > t] estimado pela Distribuição de Weibull",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 80])

    plt.tight_layout()
    arq = pasta / "weibull_rul.png"
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"   📊 {arq.name}")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_rul_weibull() -> bool:
    _log("=" * 60)
    _log("  AL IADO PV — RUL COM WEIBULL")
    _log("=" * 60)
    _log(f"\n  Trajetórias por falha: {N_TRAJ}")
    _log(f"  Passos de degradação : {N_STEPS} (sev 0→1,0)")

    # ── 1. Carrega artefatos ─────────────────────────────────
    _log(f"\n📂 Carregando artefatos...")
    for arq in [PASTA_AE/"modelo_autoencoder.pt",
                PASTA_AE/"scaler.pkl",
                PASTA_AE/"limiar.json"]:
        if not arq.exists():
            _log(f"   ❌ {arq.name} não encontrado")
            return False

    checkpoint = torch.load(PASTA_AE/"modelo_autoencoder.pt",
                            map_location="cpu", weights_only=False)
    from src.core.seguranca import carregar_pickle_com_sidecar

    scaler = carregar_pickle_com_sidecar(PASTA_AE / "scaler.pkl")
    with open(PASTA_AE/"limiar.json", "r") as f:
        info_limiar = json.load(f)

    n_features   = checkpoint["n_features"]
    latente_dim  = checkpoint["latente_dim"]
    colunas_feat = checkpoint["colunas_feat"]
    limiar       = info_limiar["limiar"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()
    _log(f"   ✅ Limiar={limiar:.4f} | device={device}")

    # ── 2. Dataset estável ───────────────────────────────────
    _log(f"\n📂 Carregando dataset...")
    df = pd.read_csv(ARQUIVO_CSV)
    df_estavel = df.iloc[int(T_INICIO_ESTAVEL*FS):
                          int(T_FIM_ESTAVEL*FS)].reset_index(drop=True)
    _log(f"   ✅ {len(df_estavel):,} amostras estáveis")

    # ── 3. Gera TTFs por tipo de falha ───────────────────────
    _log(f"\n⚙️  Gerando trajetórias de degradação...")
    ttfs_dict = {}

    for falha in FALHAS:
        fid  = falha["id"]
        nome = falha["nome"]
        _log(f"\n   🔴 {nome} ({N_TRAJ} trajetórias × {N_STEPS} passos)...")

        ttfs = []
        for i in range(N_TRAJ):
            ttf = gerar_ttf(
                df_estavel, modelo, scaler, device,
                colunas_feat, limiar, fid, N_STEPS, seed=i
            )
            ttfs.append(ttf)
            if (i + 1) % 20 == 0:
                _log(f"      [{i+1:>3}/{N_TRAJ}] TTF médio até agora: "
                      f"{np.mean(ttfs):.1f} passos", end="\r")

        ttfs = np.array(ttfs, dtype=float)
        # Adiciona pequeno jitter nos censurados para evitar spike no max
        censurados = ttfs == N_STEPS
        if censurados.sum() > 0:
            ttfs[censurados] += np.random.uniform(0, 5, censurados.sum())

        ttfs_dict[fid] = ttfs
        pct_cens = censurados.mean() * 100
        _log(f"\n      TTF: μ={ttfs.mean():.1f} ± {ttfs.std():.1f} | "
              f"min={ttfs.min():.0f} | max={ttfs.max():.0f} | "
              f"censurados={pct_cens:.0f}%")

    # ── 4. Ajuste de Weibull ─────────────────────────────────
    _log(f"\n📐 Ajustando distribuição de Weibull...")
    params = {}
    for falha in FALHAS:
        fid = falha["id"]
        p   = ajustar_weibull(ttfs_dict[fid])
        params[fid] = p
        npm_str = f"NPR={falha['npr']}" if falha['npr'] else "  D=10"
        _log(f"\n   {falha['nome']} ({npm_str})")
        _log(f"      β={p['beta']:.3f}  η={p['eta']:.1f}  "
              f"MTTF={p['mttf']:.1f}  B10={p['b10']:.1f}")
        _log(f"      KS p-value={p['ks_pval']:.3f} "
              + ("✅ ajuste adequado" if p['ks_pval'] > 0.05
                 else "⚠️  ajuste pode ser melhorado"))
        tipo_beta = ("crescente (desgaste)" if p["beta"] > 1.1
                     else "constante (aleatório)" if p["beta"] > 0.9
                     else "decrescente (mortalidade infantil)")
        _log(f"      Taxa de falha: {tipo_beta}")

    # ── 5. Visualizações ─────────────────────────────────────
    _log(f"\n📊 Gerando gráficos...")
    plotar_ttf_histogramas(ttfs_dict, params, PASTA_AE)
    plotar_confiabilidade(ttfs_dict, params, PASTA_AE)
    plotar_rul(params, PASTA_AE)

    # ── 6. Salva resultados ──────────────────────────────────
    arq_json = PASTA_AE / "weibull_results.json"
    relatorio = {
        "parametros_simulacao": {
            "n_trajetorias": N_TRAJ,
            "n_steps"      : N_STEPS,
            "limiar"       : float(limiar),
        },
        "falhas": {}
    }
    for falha in FALHAS:
        fid = falha["id"]
        relatorio["falhas"][fid] = {
            "nome"  : falha["nome"],
            "npr"   : falha["npr"],
            "weibull": params[fid],
            "ttfs"  : ttfs_dict[fid].tolist(),
        }
    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_json.name}")

    # ── 7. Resumo final ──────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  ANÁLISE DE WEIBULL E RUL CONCLUÍDA!")
    _log(f"\n  {'Falha':<28} {'β':>6} {'η':>7} {'MTTF':>8} {'B10':>8}")
    _log(f"  {'-'*58}")
    for falha in FALHAS:
        fid = falha["id"]
        p   = params[fid]
        _log(f"  {falha['nome']:<28} "
              f"{p['beta']:>6.3f} {p['eta']:>7.1f} "
              f"{p['mttf']:>8.1f} {p['b10']:>8.1f}")

    _log(f"\n  Interpretação do β:")
    _log(f"  β > 1 → taxa de falha crescente (desgaste) — esperado")
    _log(f"  β < 1 → mortalidade infantil")
    _log(f"  β = 1 → falhas aleatórias (exponencial)")
    _log(f"\n  Fase 5 do pipeline de ML concluída!")
    _log(f"  Próximo passo: integração no orquestrador")
    _log(f"{'='*60}")
    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    executar_rul_weibull()
