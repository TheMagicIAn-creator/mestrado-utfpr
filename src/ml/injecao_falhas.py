"""
injecao_falhas.py — Al IAdo PV / Fase 5
Injeção de falhas sintéticas fundamentada no FMEA do TCC (Torres, 2024).

Fundamentação metodológica:
  O FMEA aplicado ao sistema fotovoltaico do CEAMAZON identificou:
    NPR=210 → Inversor: "Problema de conexão com a rede" (S=3, O=7, D=10)
    NPR=150 → Subsistema CA: "Curto-circuito em dispositivos de proteção"
                              (S=5, O=3, D=10)

  A prioridade NPR define a ordem de injeção: primeiro as falhas de maior
  criticidade. Cada falha é modelada pela sua assinatura elétrica esperada
  nos sinais de corrente e tensão CA (Francisti, 2025; Ibrahim, 2022).

Falhas implementadas (em ordem de NPR):
  FALHA 1 — Degradação do Filtro LCL (NPR=210, relacionada ao inversor)
    Assinatura: aumento de THD, elevação dos harmônicos 5° e 7°
    Modelagem: injeção aditiva de componentes harmônicas nas correntes CA
    Física: redução da indutância ou capacitância do filtro LCL diminui
            a atenuação dos harmônicos de chaveamento

  FALHA 2 — Desbalanceamento de Fase (NPR=150, subsistema CA)
    Assinatura: assimetria de corrente entre fases (desbalanceamento > 5%)
    Modelagem: redução proporcional da amplitude de uma fase
    Física: curto-circuito em proteção ou falha de IGBT em uma fase

  FALHA 3 — Ruído de Sensor de Corrente (D=10 — alta dificuldade de detecção)
    Assinatura: ruído gaussiano elevado em uma fase
    Modelagem: sobreposição de ruído branco com std proporcional ao sinal
    Física: degradação do sensor Hall ou circuito de condicionamento

Estratégia de severidade:
  Cada falha é injetada em 7 níveis de severidade [0.05→1.0] para
  identificar a severidade mínima detectável (SMD) — ponto onde o erro
  de reconstrução cruza o limiar do Autoencoder.
  Isso conecta diretamente com o índice D (detecção) do FMEA:
  D=10 → falha muito difícil de detectar → SMD esperada mais alta.

Entrada:
  dados/brutos/Inverter_Data_Set.csv
  resultados/autoencoder/modelo_autoencoder.pt
  resultados/autoencoder/scaler.pkl
  resultados/autoencoder/limiar.json

Saída:
  resultados/autoencoder/injecao_falhas_resultados.png
  resultados/autoencoder/injecao_falhas_severidade.png
  resultados/autoencoder/injecao_falhas_report.json

Uso:
  python src/ml/injecao_falhas.py

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

_logger = _get_logger("injecao_falhas")


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
from src.ml.estilo_graficos import TAM, aplicar_estilo

aplicar_estilo()
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

import torch
from src.ml.features_ca import (
    extrair_janela, JANELA, FS, F0,
    COLUNAS_CORRENTE, COLUNAS_TENSAO, COLUNA_DC, FASES
)
from src.ml.autoencoder import Autoencoder

# ── Caminhos ─────────────────────────────────────────────────
RAIZ          = Path(__file__).parent.parent.parent
ARQUIVO_CSV   = RAIZ / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_AE      = RAIZ / "resultados" / "autoencoder"

# ── Parâmetros de injeção ────────────────────────────────────
# Janela de análise: t=10-20s (período estável do Paderborn, pós-transiente)
T_INICIO_ESTAVEL = 10.0   # segundos
T_FIM_ESTAVEL    = 20.0   # segundos

# Severidades: de muito leve a severa
SEVERIDADES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Definição das falhas (nome, descrição, cor para gráfico)
FALHAS = [
    {
        "id"      : "lcl",
        "nome"    : "Degradação Filtro LCL",
        "npr"     : 210,
        "s"       : 3, "o": 7, "d": 10,
        "cor"     : "#E53935",
        "descricao": "Injeção de harmônicos 5°, 7° e 11° nas correntes CA",
        # Schema de proveniência da falha sintética (item 4.4)
        "evidence_level"     : "E2",
        "hipotese_fisica"    : (
            "Capacitor com ESR elevado / indutor degradado no filtro LCL reduz a "
            "atenuação do chaveamento, elevando harmônicos ímpares nas correntes."
        ),
        "sinais"             : ["i_a", "i_b", "i_c"],
        "formula"            : "i += sev·(0,30·h5 + 0,20·h7 + h11)·amplitude",
        "severity_definition": "fração [0..1] da amplitude harmônica injetada",
        "source"             : "Torres (2024) FMECA CEAMAZON (NPR=210)",
        "limitations"        : [
            "amplitudes harmônicas são plausíveis, não medidas em bancada",
            "não modela envelhecimento térmico real do capacitor",
        ],
    },
    {
        "id"      : "desbalanceamento",
        "nome"    : "Desbalanceamento de Fase",
        "npr"     : 150,
        "s"       : 5, "o": 3, "d": 10,
        "cor"     : "#FB8C00",
        "descricao": "Redução de amplitude da fase A",
        "evidence_level"     : "E2",
        "hipotese_fisica"    : (
            "Assimetria entre fases (perda parcial de fase ou carga "
            "desbalanceada) reduz a amplitude de uma das correntes."
        ),
        "sinais"             : ["i_a"],
        "formula"            : "i_a ·= (1 − sev·0,12)  (reduz amplitude da fase A; máx 12%)",
        "severity_definition": "fração de redução da amplitude da fase A (calibrada: máx 12%)",
        "source"             : "Torres (2024) FMECA, subsistema CA (NPR=150)",
        "limitations"        : [
            "modelo simplificado; desbalanceamento real afeta fase E amplitude",
        ],
    },
    {
        "id"      : "sensor",
        "nome"    : "Falha de Sensor CA",
        "npr"     : None,
        "s"       : None, "o": None, "d": 10,
        "cor"     : "#8E24AA",
        "descricao": "Ruído gaussiano na corrente da fase A",
        "evidence_level"     : "E2",
        "hipotese_fisica"    : (
            "Degradação do sensor Hall / circuito de condicionamento introduz "
            "ruído de medição na corrente."
        ),
        "sinais"             : ["i_a"],
        "formula"            : "i_a += N(0, sev·σ_sinal)  (ruído gaussiano; ×1.0, calibrado p/ D=10)",
        "severity_definition": "desvio do ruído ≈ sev·σ_sinal (calibrado: a falha MAIS difícil, D=10)",
        "source"             : "FMEA (D=10 — alta dificuldade de detecção)",
        "limitations"        : [
            "RUÍDO GAUSSIANO É UM PROXY — exige CALIBRAÇÃO FÍSICA do sensor real",
            "a alta sensibilidade observada é E2 (sintética): NÃO afirmar alta "
            "sensibilidade da falha de sensor sem esta ressalva",
        ],
    },
]


def smd_probabilistico(deteccoes_por_severidade: dict, alvo: float = 0.95) -> dict:
    """
    SMD probabilística (item 4.3) a partir de detecções REPETIDAS (múltiplas
    sementes/janelas), em vez da primeira média acima do limiar.

    `deteccoes_por_severidade`: {severidade: lista[bool]} — cada bool é uma
    detecção numa repetição independente.

    Retorna taxa de detecção por severidade, SMD pontual (menor severidade com
    qualquer detecção) e SMD_95 (menor severidade com taxa de detecção ≥ alvo).
    SMD_95 = None quando nenhuma severidade alcança o alvo.
    """
    import numpy as np

    taxas, n_rep = {}, {}
    for sev, dets in deteccoes_por_severidade.items():
        arr = np.asarray(list(dets), dtype=float)
        taxas[float(sev)] = float(arr.mean()) if arr.size else 0.0
        n_rep[float(sev)] = int(arr.size)

    sevs = sorted(taxas)
    smd_pontual = next((s for s in sevs if taxas[s] > 0.0), None)
    smd_95 = next((s for s in sevs if taxas[s] >= alvo), None)
    return {
        "taxa_deteccao": taxas,
        "smd_pontual": smd_pontual,
        "smd_95": smd_95,
        "alvo": alvo,
        "n_repeticoes": n_rep,
    }


# ============================================================
# MODELOS DE FALHA (assinaturas elétricas)
# ============================================================

def falha_degradacao_lcl(janela_df: pd.DataFrame,
                          severidade: float,
                          f0: float = F0,
                          fs: int   = FS) -> pd.DataFrame:
    """
    FALHA 1 — Degradação do Filtro LCL (NPR=210)

    Um filtro LCL degradado (capacitor com ESR elevado ou indutor
    com indutância reduzida) atenua menos os harmônicos de chaveamento.
    O resultado é um aumento do THD e elevação específica dos harmônicos
    de ordem 5, 7 e 11 (dominantes em inversores VSI trifásicos).

    Modelagem aditiva: sinal_falha = sinal_saudável + Σ Ak·sin(k·ω₀·t + φk)

    Parâmetro severidade:
      0.05 → leve: THD aumenta ~2%
      0.30 → moderada: THD aumenta ~15%
      1.00 → severa: THD aumenta ~50%
    """
    janela_falha = janela_df.copy()
    n = JANELA
    t = np.arange(n) / fs

    for col in COLUNAS_CORRENTE:
        sinal     = janela_falha[col].values
        amplitude = np.std(sinal)  # referência de amplitude do sinal

        # Harmônicos característicos de inversores VSI com filtro LCL degradado
        # Amplitudes relativas baseadas em Francisti (2025) e Smith (1999)
        h5  = severidade * 0.30 * amplitude * np.sin(2 * np.pi * 5  * f0 * t)
        h7  = severidade * 0.20 * amplitude * np.sin(2 * np.pi * 7  * f0 * t)
        h11 = severidade * 0.10 * amplitude * np.sin(2 * np.pi * 11 * f0 * t)
        h13 = severidade * 0.05 * amplitude * np.sin(2 * np.pi * 13 * f0 * t)

        janela_falha[col] = sinal + h5 + h7 + h11 + h13

    return janela_falha


def falha_desbalanceamento_fase(janela_df: pd.DataFrame,
                                 severidade: float) -> pd.DataFrame:
    """
    FALHA 2 — Desbalanceamento de Fase (NPR=150)

    Curto-circuito em dispositivo de proteção ou falha de IGBT em uma fase
    causa assimetria nas correntes trifásicas. A fase afetada tem amplitude
    reduzida. O desbalanceamento é medido pela feature inter-fase:
      desbalanceamento = (max_rms - min_rms) / media_rms

    Fator de redução CALIBRADO p/ curva severidade↔detecção realista
    (fator = 1 - severidade × 0.12):
      severidade=0.3 → ~3,6% de redução (incipiente, ~limiar FMEA de 5%, DIFÍCIL)
      severidade=0.5 → 6% de redução (perceptível)
      severidade=1.0 → 12% de redução (desbalanceamento severo, mas plausível)

    Antes usava ×0.7 (até 70% de redução) → o detector separava com erro ZERO
    em qualquer severidade (validacao_report perfeito = artificial). Um
    desbalanceamento real raramente passa de ~10–15% sem desarme de proteção.

    A fase A é escolhida por ser a referência do FMEA (Id.1 do Apêndice E).
    """
    janela_falha = janela_df.copy()
    fator        = 1.0 - severidade * 0.12

    # Afeta corrente e tensão da fase A
    janela_falha["i_a_k"] = janela_df["i_a_k"].values * fator
    if "u_a_k-1" in janela_falha.columns:
        janela_falha["u_a_k-1"] = janela_df["u_a_k-1"].values * fator

    return janela_falha


def falha_sensor_corrente(janela_df: pd.DataFrame,
                           severidade: float,
                           seed: int = 0) -> pd.DataFrame:
    """
    FALHA 3 — Falha de Sensor de Corrente (D=10)

    Degradação do sensor Hall ou do circuito de condicionamento
    introduz ruído gaussiano no sinal medido. É a falha com maior
    índice D (dificuldade de detecção = 10) — o ruído se confunde
    com variações normais do sinal.

    Modelagem: sinal_falha = sinal + N(0, σ_ruído)
    onde σ_ruído = severidade × std(sinal) × 0.3 (CALIBRADO para D=10).

    Como o FMEA atribui D=10 (a MAIOR dificuldade de detecção), esta deve ser a
    falha MAIS difícil — não a mais fácil. O multiplicador caiu de ×3 para ×0.3,
    o que produz uma curva severidade↔detecção real (varredura empírica):
      severidade=0.30 → σ_ruído ≈ 0,09·σ_sinal (SNR ~21 dB) — NÃO detectada (difícil)
      severidade=0.50 → σ_ruído ≈ 0,15·σ_sinal — limítrofe
      severidade=1.00 → σ_ruído ≈ 0,30·σ_sinal (SNR ~10 dB) — detectada

    Antes (×3) o ruído era enorme já em sev=0.3 → detecção trivial PERFEITA em
    todas as severidades, contradizendo o índice D=10 do próprio FMEA.
    Nota: features espectrais (THD/harmônicos/kurtosis) são naturalmente
    sensíveis a ruído branco — por isso o multiplicador precisa ser pequeno.
    """
    rng          = np.random.default_rng(seed)
    janela_falha = janela_df.copy()

    sinal   = janela_df["i_a_k"].values
    std_sig = np.std(sinal)
    ruido   = rng.normal(0, severidade * std_sig * 0.3, size=len(sinal))

    janela_falha["i_a_k"] = sinal + ruido
    return janela_falha


# Mapa de funções de falha
FUNCOES_FALHA = {
    "lcl"             : falha_degradacao_lcl,
    "desbalanceamento": falha_desbalanceamento_fase,
    "sensor"          : falha_sensor_corrente,
}


# ============================================================
# INFERÊNCIA COM O AUTOENCODER
# ============================================================

def calcular_erro_reconstrucao(janela_df: pd.DataFrame,
                                modelo: Autoencoder,
                                scaler,
                                device: torch.device,
                                colunas_feat: list) -> float:
    """
    Extrai features de uma janela, normaliza e calcula o erro
    de reconstrução do Autoencoder.
    """
    # Extrai features
    feats = extrair_janela(janela_df)

    # Monta vetor na ordem correta (igual ao treino)
    vetor = np.array([feats.get(c, 0.0) for c in colunas_feat],
                     dtype=np.float32)

    # Normaliza com o scaler ajustado no treino
    vetor_norm = scaler.transform(vetor.reshape(1, -1)).astype(np.float32)

    # Inferência
    modelo.eval()
    with torch.no_grad():
        x     = torch.from_numpy(vetor_norm).to(device)
        x_rec = modelo(x)
        erro  = float(((x - x_rec) ** 2).mean().cpu())

    return erro


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_injecao_falhas() -> bool:
    """
    Pipeline completo de injeção de falhas sintéticas.
    """
    _log("=" * 60)
    _log("  AL IADO PV — INJEÇÃO DE FALHAS SINTÉTICAS")
    _log("=" * 60)
    _log("\n  Fundamentação: FMEA Torres (2024)")
    _log("  NPR=210 → Degradação Filtro LCL")
    _log("  NPR=150 → Desbalanceamento de Fase")
    _log("  D=10    → Falha de Sensor CA")

    # ── 1. Carrega artefatos do Autoencoder ──────────────────
    _log(f"\n📂 Carregando Autoencoder...")

    arq_modelo = PASTA_AE / "modelo_autoencoder.pt"
    arq_scaler = PASTA_AE / "scaler.pkl"
    arq_limiar = PASTA_AE / "limiar.json"

    for arq in [arq_modelo, arq_scaler, arq_limiar]:
        if not arq.exists():
            _log(f"   ❌ Não encontrado: {arq.name}")
            _log("   Execute primeiro: python src/ml/autoencoder.py")
            return False

    checkpoint = torch.load(arq_modelo, map_location="cpu",
                            weights_only=False)
    from src.core.seguranca import carregar_pickle_com_sidecar

    scaler = carregar_pickle_com_sidecar(arq_scaler)
    with open(arq_limiar, "r") as f:
        info_limiar = json.load(f)

    n_features   = checkpoint["n_features"]
    latente_dim  = checkpoint["latente_dim"]
    colunas_feat = checkpoint["colunas_feat"]
    limiar       = info_limiar["limiar"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    _log(f"   ✅ Modelo: {n_features} features → latente {latente_dim}")
    _log(f"   ✅ Limiar de anomalia: {limiar:.4f}")

    # ── 2. Carrega dataset e seleciona janelas estáveis ──────
    _log(f"\n📂 Carregando dataset de Paderborn...")
    df = pd.read_csv(ARQUIVO_CSV)

    # Período estável: t=10-20s (pós-transiente de velocidade)
    idx_inicio = int(T_INICIO_ESTAVEL * FS)
    idx_fim    = int(T_FIM_ESTAVEL * FS)
    df_estavel = df.iloc[idx_inicio:idx_fim].reset_index(drop=True)
    _log(f"   ✅ Período estável: t={T_INICIO_ESTAVEL}-{T_FIM_ESTAVEL}s "
          f"({len(df_estavel):,} amostras)")

    # ── 3. Erro baseline (comportamento saudável) ─────────────
    _log(f"\n⚕️  Calculando erro baseline (saudável)...")
    n_janelas_baseline = 20
    erros_baseline = []
    for i in range(n_janelas_baseline):
        inicio = i * (JANELA // 2)
        if inicio + JANELA > len(df_estavel):
            break
        janela = df_estavel.iloc[inicio:inicio + JANELA]
        erro   = calcular_erro_reconstrucao(
            janela, modelo, scaler, device, colunas_feat
        )
        erros_baseline.append(erro)

    baseline_mean = np.mean(erros_baseline)
    baseline_std  = np.std(erros_baseline)
    _log(f"   Baseline: μ={baseline_mean:.4f} ± {baseline_std:.4f}")
    _log(f"   Limiar  : {limiar:.4f} "
          f"({limiar/baseline_mean:.1f}× acima do baseline)")

    # ── 4. Injeção de falhas por severidade ──────────────────
    _log(f"\n💉 Injetando falhas (3 tipos × {len(SEVERIDADES)} severidades)...")

    resultados = {}   # {id_falha: {severidade: erro_medio}}

    # Janela de referência para injeção
    janela_ref = df_estavel.iloc[0:JANELA].copy()

    for falha in FALHAS:
        fid  = falha["id"]
        fn   = FUNCOES_FALHA[fid]
        erros_por_sev = {}

        _log(f"\n   🔴 {falha['nome']} (NPR={falha['npr']})")

        for sev in SEVERIDADES:
            # Injeta falha em 5 janelas diferentes e faz média
            erros_sev = []
            for j in range(5):
                inicio  = j * (JANELA // 4)
                if inicio + JANELA > len(df_estavel):
                    inicio = 0
                janela_base  = df_estavel.iloc[inicio:inicio + JANELA].copy()
                janela_falha = fn(janela_base, sev)
                erro = calcular_erro_reconstrucao(
                    janela_falha, modelo, scaler, device, colunas_feat
                )
                erros_sev.append(erro)

            erro_medio    = np.mean(erros_sev)
            detectado     = erro_medio > limiar
            margem        = erro_medio / limiar

            erros_por_sev[sev] = {
                "erro"      : erro_medio,
                "detectado" : detectado,
                "margem"    : margem,
            }

            status = "✅ DETECTADA" if detectado else "⬜ não detectada"
            _log(f"      sev={sev:.2f} | erro={erro_medio:.4f} | "
                  f"margem={margem:.2f}× | {status}")

        resultados[fid] = erros_por_sev

    # ── 5. Severidade mínima detectável (SMD) ────────────────
    _log(f"\n🎯 Severidade Mínima Detectável (SMD):")
    smd_report = {}
    for falha in FALHAS:
        fid = falha["id"]
        smd = None
        for sev in SEVERIDADES:
            if resultados[fid][sev]["detectado"]:
                smd = sev
                break
        smd_report[fid] = smd
        if smd:
            _log(f"   {falha['nome']:<30}: SMD = {smd:.2f} "
                  f"(erro = {resultados[fid][smd]['erro']:.4f})")
        else:
            _log(f"   {falha['nome']:<30}: não detectada em nenhuma severidade")

    # ── 6. Visualizações ─────────────────────────────────────
    _log(f"\n📊 Gerando gráficos...")
    PASTA_AE.mkdir(parents=True, exist_ok=True)

    # Gráfico 1: Erro vs Severidade por tipo de falha
    fig, axes = plt.subplots(1, 3, figsize=TAM["painel_3"], sharey=False)
    fig.suptitle("Injeção de Falhas Sintéticas — Erro de Reconstrução vs Severidade",
                 fontsize=13, fontweight="bold")

    for ax, falha in zip(axes, FALHAS):
        fid   = falha["id"]
        sevs  = SEVERIDADES
        erros = [resultados[fid][s]["erro"] for s in sevs]

        cores = [falha["cor"] if resultados[fid][s]["detectado"]
                 else "#BDBDBD" for s in sevs]

        ax.bar([str(s) for s in sevs], erros, color=cores, alpha=0.85,
               edgecolor="white", linewidth=0.5)
        ax.axhline(limiar, color="red", linestyle="--", linewidth=1.5,
                   label=f"Limiar = {limiar:.2f}")
        ax.axhline(baseline_mean, color="green", linestyle=":",
                   linewidth=1.2, label=f"Baseline = {baseline_mean:.4f}")

        npm_str = f"NPR={falha['npr']}" if falha['npr'] else "D=10"
        ax.set_title(f"{falha['nome']}\n({npm_str})", fontsize=10)
        ax.set_xlabel("Severidade")
        ax.set_ylabel("Erro de Reconstrução (MSE)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Marca SMD
        smd = smd_report[fid]
        if smd:
            idx_smd = sevs.index(smd)
            ax.bar([str(smd)], [resultados[fid][smd]["erro"]],
                   color=falha["cor"], edgecolor="black", linewidth=2,
                   alpha=1.0)
            ax.annotate(f"SMD={smd}",
                        xy=(idx_smd, resultados[fid][smd]["erro"]),
                        xytext=(idx_smd, resultados[fid][smd]["erro"] * 1.05),
                        fontsize=8, ha="center", fontweight="bold")

    plt.tight_layout()
    arq_g1 = PASTA_AE / "injecao_falhas_resultados.png"
    fig.savefig(arq_g1)
    plt.close(fig)
    _log(f"   📊 {arq_g1.name}")

    # Gráfico 2: Comparação consolidada em escala log
    fig, ax = plt.subplots(figsize=TAM["unico"])

    x      = np.arange(len(SEVERIDADES))
    largura = 0.25
    offsets = [-largura, 0, largura]

    for i, falha in enumerate(FALHAS):
        fid   = falha["id"]
        erros = [resultados[fid][s]["erro"] for s in SEVERIDADES]
        ax.bar(x + offsets[i], erros, largura, label=falha["nome"],
               color=falha["cor"], alpha=0.85, edgecolor="white")

    ax.axhline(limiar, color="red", linestyle="--", linewidth=2,
               label=f"Limiar anomalia = {limiar:.4f}")
    ax.axhline(baseline_mean, color="green", linestyle=":",
               linewidth=1.5, label=f"Baseline saudável = {baseline_mean:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEVERIDADES])
    ax.set_xlabel("Severidade da Falha")
    ax.set_ylabel("Erro de Reconstrução (MSE)")
    ax.set_title("Comparação das Falhas Sintéticas\n"
                 "(barras acima da linha vermelha = anomalia detectada)",
                 fontsize=12)
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    arq_g2 = PASTA_AE / "injecao_falhas_comparacao.png"
    fig.savefig(arq_g2)
    plt.close(fig)
    _log(f"   📊 {arq_g2.name}")

    # ── 7. Salva relatório JSON ───────────────────────────────
    relatorio = {
        "evidence_level": "E2",
        "evidence_note": (
            "Falhas sintéticas orientadas pelo FMEA (ground truth para validar o "
            "detector). Não é validação experimental externa (E3). A injeção de "
            "ruído do sensor é um PROXY e exige calibração física."
        ),
        "threshold_method": "p99",
        "limiar": float(limiar),
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "smd": {k: float(v) if v is not None else None
                for k, v in smd_report.items()},
        "falhas": {}
    }
    for falha in FALHAS:
        fid = falha["id"]
        relatorio["falhas"][fid] = {
            "nome": falha["nome"],
            "npr": falha["npr"],
            "descricao": falha["descricao"],
            # Schema de proveniência da falha sintética (item 4.4)
            "evidence_level": falha.get("evidence_level", "E2"),
            "hipotese_fisica": falha.get("hipotese_fisica"),
            "sinais": falha.get("sinais"),
            "formula": falha.get("formula"),
            "severity_definition": falha.get("severity_definition"),
            "source": falha.get("source"),
            "limitations": falha.get("limitations"),
            "resultados": {
                str(s): {
                    "erro": float(resultados[fid][s]["erro"]),
                    "detectado": bool(resultados[fid][s]["detectado"]),
                    "margem": float(resultados[fid][s]["margem"]),
                }
                for s in SEVERIDADES
            }
        }

    arq_report = PASTA_AE / "injecao_falhas_report.json"
    with open(arq_report, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_report.name}")

    # ── 8. Resumo final ───────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  INJEÇÃO DE FALHAS CONCLUÍDA!")
    _log(f"  Baseline saudável : {baseline_mean:.4f} ± {baseline_std:.4f}")
    _log(f"  Limiar de anomalia: {limiar:.4f}")
    _log()
    for falha in FALHAS:
        fid = falha["id"]
        smd = smd_report[fid]
        if smd:
            erro_smd = resultados[fid][smd]["erro"]
            margem   = erro_smd / limiar
            _log(f"  {falha['nome']:<30}")
            _log(f"    SMD = {smd:.2f} | erro = {erro_smd:.4f} | "
                  f"margem = {margem:.1f}× acima do limiar")
        else:
            _log(f"  {falha['nome']:<30}")
            _log(f"    Não detectada — severidade insuficiente")
    _log()
    _log(f"  Próximo passo: validação cruzada + métricas finais")
    _log(f"{'='*60}")

    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    executar_injecao_falhas()