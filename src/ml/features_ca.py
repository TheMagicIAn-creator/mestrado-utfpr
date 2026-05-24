"""
features_ca.py — Al IAdo PV / Fase 5
Extração de features do lado CA a partir do dataset de Paderborn.

Dataset: Inverter_Data_Set.csv (Stender, Wallscheid & Böcker, 2020)
  - ~235 mil amostras, taxa de amostragem: 10 kHz
  - Inversor IGBT trifásico em operação SAUDÁVEL
  - Frequência fundamental: 60 Hz (Brasil)

Sinais utilizados:
  - i_a_k, i_b_k, i_c_k  → correntes CA trifásicas (instante atual)
  - u_a_k-1, u_b_k-1, u_c_k-1 → tensões CA (instante anterior)
  - u_dc_k → tensão do barramento CC

Estratégia de janelamento:
  - Janela: 1024 amostras = 0,1024 s ≈ 6 ciclos a 60 Hz
  - Sobreposição: 512 amostras (50%)
  - Resolução espectral: ~9,77 Hz por bin

Features extraídas (por janela):
  DOMÍNIO DO TEMPO (por fase):
    RMS, média, desvio padrão, kurtosis, skewness,
    fator de crista, pico a pico

  DOMÍNIO DA FREQUÊNCIA (por fase):
    THD, amplitudes dos harmônicos 3°, 5°, 7°, 11°, 13°,
    centróide espectral, largura de banda espectral,
    energia nas bandas: baixa (0-120 Hz), média (120-1kHz),
    chaveamento (1k-5kHz)

  INTER-FASE:
    Desbalanceamento de corrente, desbalanceamento de tensão,
    potência ativa estimada por fase

Saída: dados/processados/features_paderborn.parquet
       dados/processados/features_paderborn_stats.csv

Uso:
  python src/ml/features_ca.py

Autor: Rodolfo Torres (UTFPR)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import windows
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Configuração ──────────────────────────────────────────────
FS            = 10_000          # Hz — taxa de amostragem
F0            = 60              # Hz — frequência fundamental (Brasil)
JANELA        = 1024            # amostras por janela (~6 ciclos)
SOBREPOSICAO  = 512             # amostras de overlap (50%)
PASSO         = JANELA - SOBREPOSICAO

# Harmônicos a extrair (ímpares não-múltiplos de 3 são dominantes em inversores)
HARMONICOS    = [3, 5, 7, 11, 13]

# Bandas de frequência para análise de energia
BANDAS = {
    "baixa"      : (0,     120),    # fundamental + 2° harmônico
    "media"      : (120,   1000),   # harmônicos de ordem média
    "chaveamento": (1000,  5000),   # harmônicos de chaveamento (limite Nyquist)
}

# Caminhos
RAIZ          = Path(__file__).parent.parent.parent
ARQUIVO_CSV   = RAIZ / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_SAIDA   = RAIZ / "dados" / "processados"

# Colunas por tipo
COLUNAS_CORRENTE = ["i_a_k", "i_b_k", "i_c_k"]
COLUNAS_TENSAO   = ["u_a_k-1", "u_b_k-1", "u_c_k-1"]
COLUNA_DC        = "u_dc_k"
FASES            = ["a", "b", "c"]


# ============================================================
# FEATURES DE DOMÍNIO DO TEMPO
# ============================================================

def features_tempo(sinal: np.ndarray, prefixo: str) -> dict:
    """
    Extrai 7 features estatísticas do domínio do tempo.

    Parâmetros:
        sinal   : array de amostras da janela
        prefixo : ex. 'i_a' para corrente fase A

    Retorna dicionário com as features nomeadas.
    """
    rms         = np.sqrt(np.mean(sinal ** 2))
    media       = np.mean(sinal)
    desvio      = np.std(sinal)
    kurt        = stats.kurtosis(sinal)          # achatamento (>3 = outliers)
    skew        = stats.skew(sinal)              # assimetria
    pico        = np.max(np.abs(sinal))
    fator_crista = pico / rms if rms > 1e-10 else 0.0  # sensível a impactos
    pico_a_pico  = np.max(sinal) - np.min(sinal)

    return {
        f"{prefixo}_rms"          : rms,
        f"{prefixo}_media"        : media,
        f"{prefixo}_desvio"       : desvio,
        f"{prefixo}_kurtosis"     : kurt,
        f"{prefixo}_skewness"     : skew,
        f"{prefixo}_fator_crista" : fator_crista,
        f"{prefixo}_pico_a_pico"  : pico_a_pico,
    }


# ============================================================
# FEATURES DE DOMÍNIO DA FREQUÊNCIA
# ============================================================

def calcular_espectro(sinal: np.ndarray, fs: int) -> tuple:
    """
    Aplica janela de Hann + FFT e retorna frequências e amplitudes.
    A janela de Hann reduz o vazamento espectral (Smith, 1999).

    Retorna: (frequencias_hz, amplitudes_normalizadas)
    """
    n         = len(sinal)
    jan       = windows.hann(n)
    sinal_jan = sinal * jan

    espectro  = np.fft.rfft(sinal_jan)
    freqs     = np.fft.rfftfreq(n, d=1.0 / fs)

    # Amplitude real (fator 2 compensa espelho do espectro)
    amps = (2.0 / n) * np.abs(espectro)

    return freqs, amps


def amplitude_harmonica(freqs: np.ndarray, amps: np.ndarray,
                        f0: float, ordem: int,
                        tolerancia_hz: float = 15.0) -> float:
    """
    Retorna a amplitude do harmônico de ordem N.
    Busca na janela [N*f0 - tol, N*f0 + tol] Hz.
    """
    alvo   = ordem * f0
    mascara = np.abs(freqs - alvo) <= tolerancia_hz
    if not np.any(mascara):
        return 0.0
    return float(np.max(amps[mascara]))


def calcular_thd(freqs: np.ndarray, amps: np.ndarray,
                 f0: float, n_harmonicos: int = 13) -> float:
    """
    THD (Total Harmonic Distortion) — razão entre distorção harmônica
    total e a componente fundamental.

    THD = sqrt(sum(V_n^2, n=2..N)) / V_1

    Um filtro LCL degradado eleva o THD de corrente.
    """
    v1 = amplitude_harmonica(freqs, amps, f0, 1)
    if v1 < 1e-10:
        return 0.0

    soma_quadrados = 0.0
    for n in range(2, n_harmonicos + 1):
        vn = amplitude_harmonica(freqs, amps, f0, n)
        soma_quadrados += vn ** 2

    return float(np.sqrt(soma_quadrados) / v1)


def centroide_espectral(freqs: np.ndarray, amps: np.ndarray) -> float:
    """
    Centróide espectral — "centro de gravidade" do espectro.
    Deslocamento indica mudança na distribuição de energia.
    """
    soma_amps = np.sum(amps)
    if soma_amps < 1e-10:
        return 0.0
    return float(np.sum(freqs * amps) / soma_amps)


def largura_banda_espectral(freqs: np.ndarray, amps: np.ndarray,
                             centroide: float) -> float:
    """
    Largura de banda espectral — dispersão em torno do centróide.
    Aumenta quando surgem novos componentes de frequência (falha).
    """
    soma_amps = np.sum(amps)
    if soma_amps < 1e-10:
        return 0.0
    return float(np.sqrt(np.sum(((freqs - centroide) ** 2) * amps) / soma_amps))


def energia_banda(freqs: np.ndarray, amps: np.ndarray,
                  f_min: float, f_max: float) -> float:
    """Energia (soma de amplitudes²) numa banda de frequência."""
    mascara = (freqs >= f_min) & (freqs <= f_max)
    return float(np.sum(amps[mascara] ** 2))


def features_frequencia(sinal: np.ndarray, prefixo: str,
                         fs: int = FS, f0: float = F0) -> dict:
    """
    Extrai features espectrais de um sinal numa janela.
    Total: 5 harmônicos + THD + centróide + largura_banda + 3 bandas = 11 features
    """
    freqs, amps = calcular_espectro(sinal, fs)

    thd        = calcular_thd(freqs, amps, f0)
    centroide  = centroide_espectral(freqs, amps)
    largura    = largura_banda_espectral(freqs, amps, centroide)

    resultado = {
        f"{prefixo}_thd"               : thd,
        f"{prefixo}_centroide"         : centroide,
        f"{prefixo}_largura_banda"     : largura,
    }

    # Amplitude de cada harmônico
    for h in HARMONICOS:
        resultado[f"{prefixo}_harm_{h}"] = amplitude_harmonica(
            freqs, amps, f0, h
        )

    # Energia por banda
    for nome_banda, (fmin, fmax) in BANDAS.items():
        resultado[f"{prefixo}_energia_{nome_banda}"] = energia_banda(
            freqs, amps, fmin, fmax
        )

    return resultado


# ============================================================
# FEATURES INTER-FASE
# ============================================================

def features_interfase(correntes: dict, tensoes: dict,
                        dc: float) -> dict:
    """
    Features que envolvem mais de uma fase:
      - Desbalanceamento de corrente (variação entre fases)
      - Desbalanceamento de tensão
      - Potência ativa estimada por fase (i × u)
      - Tensão CC média da janela
    """
    rms_i = np.array([correntes[f"i_{f}_rms"] for f in FASES])
    rms_u = np.array([tensoes[f"u_{f}_rms"] for f in FASES])

    media_i    = np.mean(rms_i)
    media_u    = np.mean(rms_u)
    desbal_i   = (np.max(rms_i) - np.min(rms_i)) / media_i if media_i > 1e-10 else 0.0
    desbal_u   = (np.max(rms_u) - np.min(rms_u)) / media_u if media_u > 1e-10 else 0.0

    return {
        "desbalanceamento_corrente" : desbal_i,
        "desbalanceamento_tensao"   : desbal_u,
        "potencia_a"                : rms_i[0] * rms_u[0],
        "potencia_b"                : rms_i[1] * rms_u[1],
        "potencia_c"                : rms_i[2] * rms_u[2],
        "tensao_dc_media"           : dc,
    }


# ============================================================
# EXTRAÇÃO POR JANELA
# ============================================================

def extrair_janela(df_janela: pd.DataFrame) -> dict:
    """
    Recebe um slice do DataFrame (tamanho = JANELA) e
    retorna um dicionário com todas as features da janela.
    """
    features = {}

    # ── Correntes CA ─────────────────────────────────────────
    rms_correntes = {}
    for col, fase in zip(COLUNAS_CORRENTE, FASES):
        sinal   = df_janela[col].values
        prefixo = f"i_{fase}"

        ft = features_tempo(sinal, prefixo)
        ff = features_frequencia(sinal, prefixo)

        features.update(ft)
        features.update(ff)
        rms_correntes[f"i_{fase}_rms"] = ft[f"{prefixo}_rms"]

    # ── Tensões CA ───────────────────────────────────────────
    rms_tensoes = {}
    for col, fase in zip(COLUNAS_TENSAO, FASES):
        sinal   = df_janela[col].values
        prefixo = f"u_{fase}"

        ft = features_tempo(sinal, prefixo)
        ff = features_frequencia(sinal, prefixo)

        features.update(ft)
        features.update(ff)
        rms_tensoes[f"u_{fase}_rms"] = ft[f"{prefixo}_rms"]

    # ── DC ───────────────────────────────────────────────────
    dc_media = float(df_janela[COLUNA_DC].mean())

    # ── Inter-fase ───────────────────────────────────────────
    fi = features_interfase(rms_correntes, rms_tensoes, dc_media)
    features.update(fi)

    return features


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_features_ca(
    arquivo_csv : Path = ARQUIVO_CSV,
    pasta_saida : Path = PASTA_SAIDA,
    limite_linhas: int = None          # None = usa tudo
) -> bool:
    """
    Pipeline completo de extração de features CA.

    1. Carrega o dataset de Paderborn
    2. Aplica janelamento com sobreposição
    3. Extrai features de tempo + frequência por janela
    4. Salva em Parquet e gera estatísticas descritivas

    Retorna True se concluiu com sucesso.
    """
    print("=" * 60)
    print("  AL IADO PV — FEATURES CA (Paderborn)")
    print("=" * 60)

    # ── 1. Carrega dados ─────────────────────────────────────
    print(f"\n📂 Carregando dataset...")
    print(f"   Arquivo: {arquivo_csv.name}")

    if not arquivo_csv.exists():
        print(f"   ❌ Arquivo não encontrado: {arquivo_csv}")
        return False

    df = pd.read_csv(arquivo_csv, nrows=limite_linhas)
    print(f"   ✅ {len(df):,} amostras carregadas | {df.shape[1]} colunas")

    # Verifica colunas necessárias
    colunas_necessarias = COLUNAS_CORRENTE + COLUNAS_TENSAO + [COLUNA_DC]
    faltando = [c for c in colunas_necessarias if c not in df.columns]
    if faltando:
        print(f"   ❌ Colunas ausentes: {faltando}")
        return False

    # ── 2. Janelamento ───────────────────────────────────────
    n_total    = len(df)
    n_janelas  = (n_total - JANELA) // PASSO + 1
    duracao_s  = n_total / FS

    print(f"\n🪟  Janelamento:")
    print(f"   Tamanho da janela : {JANELA} amostras = {JANELA/FS*1000:.1f} ms")
    print(f"   Sobreposição      : {SOBREPOSICAO} amostras (50%)")
    print(f"   Passo             : {PASSO} amostras = {PASSO/FS*1000:.1f} ms")
    print(f"   Duração total     : {duracao_s:.1f} s")
    print(f"   Janelas geradas   : {n_janelas:,}")

    # ── 3. Extrai features ───────────────────────────────────
    print(f"\n⚙️  Extraindo features...")

    registros = []
    for i in range(n_janelas):
        inicio = i * PASSO
        fim    = inicio + JANELA
        slice_ = df.iloc[inicio:fim]

        feats = extrair_janela(slice_)
        feats["janela_idx"]    = i
        feats["amostra_inicio"] = inicio
        feats["tempo_s"]       = inicio / FS
        registros.append(feats)

        # Progresso a cada 500 janelas
        if (i + 1) % 500 == 0 or i == n_janelas - 1:
            pct = (i + 1) / n_janelas * 100
            print(f"   [{pct:5.1f}%] {i+1:,}/{n_janelas:,} janelas", end="\r")

    print()  # quebra linha após o progresso
    df_features = pd.DataFrame(registros)

    n_features = len([c for c in df_features.columns
                      if c not in ["janela_idx", "amostra_inicio", "tempo_s"]])
    print(f"   ✅ {len(df_features):,} janelas × {n_features} features extraídas")

    # ── 4. Salva resultados ──────────────────────────────────
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Parquet (formato eficiente para ML)
    arq_parquet = pasta_saida / "features_paderborn.parquet"
    df_features.to_parquet(arq_parquet, index=False)
    print(f"\n💾 Salvo: {arq_parquet.name} ({arq_parquet.stat().st_size / 1024:.0f} KB)")

    # Estatísticas descritivas (CSV para inspeção rápida)
    arq_stats = pasta_saida / "features_paderborn_stats.csv"
    df_features.describe().T.to_csv(arq_stats)
    print(f"   Salvo: {arq_stats.name}")

    # ── 5. Resumo das features ───────────────────────────────
    print(f"\n📊 Resumo das features extraídas:")
    colunas_feat = [c for c in df_features.columns
                    if c not in ["janela_idx", "amostra_inicio", "tempo_s"]]

    grupos = {
        "Tempo — corrente" : [c for c in colunas_feat if c.startswith("i_") and
                              any(s in c for s in ["rms","media","desvio",
                                                    "kurtosis","skewness",
                                                    "fator","pico"])],
        "Tempo — tensão"   : [c for c in colunas_feat if c.startswith("u_") and
                              any(s in c for s in ["rms","media","desvio",
                                                    "kurtosis","skewness",
                                                    "fator","pico"])],
        "Freq — corrente"  : [c for c in colunas_feat if c.startswith("i_") and
                              any(s in c for s in ["thd","harm","centroide",
                                                    "largura","energia"])],
        "Freq — tensão"    : [c for c in colunas_feat if c.startswith("u_") and
                              any(s in c for s in ["thd","harm","centroide",
                                                    "largura","energia"])],
        "Inter-fase"       : [c for c in colunas_feat if
                              any(s in c for s in ["desbalanceamento",
                                                    "potencia","dc"])],
    }

    total = 0
    for grupo, cols in grupos.items():
        print(f"   {grupo:<22}: {len(cols):>3} features")
        total += len(cols)
    print(f"   {'TOTAL':<22}: {total:>3} features por janela")

    # Verifica NaN
    n_nan = df_features[colunas_feat].isna().sum().sum()
    if n_nan > 0:
        print(f"\n   ⚠️  {n_nan} valores NaN detectados — verificar sinais muito próximos de zero")
    else:
        print(f"\n   ✅ Nenhum valor NaN — dados prontos para o Autoencoder")

    print(f"\n{'='*60}")
    print(f"  EXTRAÇÃO CONCLUÍDA!")
    print(f"  Próximo passo: EDA das features + treinar Autoencoder")
    print(f"{'='*60}")

    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrai features CA do dataset de Paderborn"
    )
    parser.add_argument(
        "--limite", type=int, default=None,
        help="Limita o número de linhas (ex: 50000 para teste rápido)"
    )
    args = parser.parse_args()

    executar_features_ca(limite_linhas=args.limite)