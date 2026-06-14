"""
features_ca.py — Al IAdo PV / Fase 5
Extração de features do lado CA a partir do dataset de Paderborn.

Dataset: Inverter_Data_Set.csv (Stender, Wallscheid & Böcker, 2020)
  - ~235 mil amostras, taxa de amostragem: 10 kHz
  - Inversor IGBT trifásico em operação SAUDÁVEL
  - Frequência fundamental nominal: 60 Hz (F0 estimado adaptativamente)

Sinais utilizados:
  - i_a_k, i_b_k, i_c_k  → correntes CA trifásicas (instante atual)
  - u_a_k-1, u_b_k-1, u_c_k-1 → tensões CA (instante anterior)
  - u_dc_k → tensão do barramento CC

Estratégia de janelamento:
  - Janela: 1024 amostras = 102,4 ms ≈ 6 ciclos a 60 Hz
  - Sobreposição: 512 amostras (50%)
  - Resolução espectral: ~9,77 Hz por bin

Features extraídas (por janela):
  DOMÍNIO DO TEMPO (por fase — exceto média AC):
    RMS, desvio padrão, kurtosis, skewness, fator de crista, pico a pico

  DOMÍNIO DA FREQUÊNCIA (por fase):
    THD, harmônicos 3°/5°/7°/11°/13°, centróide espectral,
    largura de banda, energia bandas baixa/média/chaveamento

  INTER-FASE:
    Desbalanceamento de corrente/tensão, potência por fase, DC média

  F0 estimado (1 feature): frequência fundamental real da janela

Saída: dados/processados/features_paderborn.parquet
       dados/processados/features_paderborn_stats.csv

Uso:
  python src/ml/features_ca.py
  python src/ml/features_ca.py --limite 50000   # teste rápido

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

_logger = _get_logger("features_ca")


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



import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import windows
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Configuração ──────────────────────────────────────────────
FS           = 10_000      # Hz — taxa de amostragem
F0           = 60          # Hz — frequência fundamental nominal (Brasil)
JANELA       = 1024        # amostras por janela (~6 ciclos a 60 Hz)
SOBREPOSICAO = 512         # amostras de overlap (50%)
PASSO        = JANELA - SOBREPOSICAO

# Harmônicos a extrair
HARMONICOS = [3, 5, 7, 11, 13]

# Bandas de frequência para análise de energia
BANDAS = {
    "baixa"      : (0,    120),   # fundamental + 2° harmônico
    "media"      : (120, 1000),   # harmônicos de ordem média
    "chaveamento": (1000, 5000),  # harmônicos de chaveamento (≤ Nyquist)
}

# Features a excluir do vetor final (média de sinais AC ≈ 0, CV enganoso)
FEATURES_EXCLUIR = [
    "i_a_media", "i_b_media", "i_c_media",
    "u_a_media", "u_b_media", "u_c_media",
]

# Caminhos
RAIZ        = Path(__file__).parent.parent.parent
ARQUIVO_CSV = RAIZ / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_SAIDA = RAIZ / "dados" / "processados"

# Colunas por tipo
COLUNAS_CORRENTE = ["i_a_k", "i_b_k", "i_c_k"]
COLUNAS_TENSAO   = ["u_a_k-1", "u_b_k-1", "u_c_k-1"]
COLUNA_DC        = "u_dc_k"
FASES            = ["a", "b", "c"]

# Caches pequenos e determinísticos para evitar recriar os mesmos vetores
# em cada janela analisada.
_CACHE_HANN = {}
_CACHE_FREQS = {}


def _janela_hann(n: int) -> np.ndarray:
    if n not in _CACHE_HANN:
        _CACHE_HANN[n] = windows.hann(n)
    return _CACHE_HANN[n]


def _freqs_rfft(n: int, fs: int) -> np.ndarray:
    chave = (n, fs)
    if chave not in _CACHE_FREQS:
        _CACHE_FREQS[chave] = np.fft.rfftfreq(n, d=1.0 / fs)
    return _CACHE_FREQS[chave]


# ============================================================
# ESPECTRO
# ============================================================

def calcular_espectro(sinal: np.ndarray, fs: int) -> tuple:
    """
    Aplica janela de Hann + FFT.
    Retorna (frequencias_hz, amplitudes_normalizadas).
    Janela de Hann reduz vazamento espectral (Smith, 1999).
    """
    n         = len(sinal)
    jan       = _janela_hann(n)
    espectro  = np.fft.rfft(sinal * jan)
    freqs     = _freqs_rfft(n, fs)
    amps      = (2.0 / n) * np.abs(espectro)  # fator 2 compensa espelho
    return freqs, amps


def estimar_f0(freqs: np.ndarray, amps: np.ndarray,
               f0_nominal: float = F0,
               faixa_hz: float = 40.0) -> float:
    """
    Estima a frequência fundamental real buscando o pico máximo
    na faixa [f0_nominal - faixa, f0_nominal + faixa] Hz.

    Necessário para datasets de acionamento de motor (Paderborn)
    onde F0 varia com a velocidade do motor.
    """
    f_min   = max(5.0, f0_nominal - faixa_hz)
    f_max   = f0_nominal + faixa_hz
    mascara = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(mascara):
        return f0_nominal
    return float(freqs[mascara][np.argmax(amps[mascara])])


# ============================================================
# FEATURES DE DOMÍNIO DO TEMPO
# ============================================================

def features_tempo(sinal: np.ndarray, prefixo: str) -> dict:
    """
    Extrai 7 features do domínio do tempo.
    Nota: *_media é calculada mas marcada para exclusão posterior
    (média de sinal AC ≈ 0, coeficiente de variação enganoso).
    """
    rms          = np.sqrt(np.mean(sinal ** 2))
    media        = np.mean(sinal)
    desvio       = np.std(sinal)
    kurt         = stats.kurtosis(sinal)
    skew         = stats.skew(sinal)
    pico         = np.max(np.abs(sinal))
    fator_crista = pico / rms if rms > 1e-10 else 0.0
    pico_a_pico  = np.max(sinal) - np.min(sinal)

    return {
        f"{prefixo}_rms"          : rms,
        f"{prefixo}_media"        : media,       # excluída no vetor final
        f"{prefixo}_desvio"       : desvio,
        f"{prefixo}_kurtosis"     : kurt,
        f"{prefixo}_skewness"     : skew,
        f"{prefixo}_fator_crista" : fator_crista,
        f"{prefixo}_pico_a_pico"  : pico_a_pico,
    }


# ============================================================
# FEATURES DE DOMÍNIO DA FREQUÊNCIA
# ============================================================

def amplitude_harmonica(freqs: np.ndarray, amps: np.ndarray,
                         f0: float, ordem: int,
                         tolerancia_hz: float = 15.0) -> float:
    """Amplitude do harmônico de ordem N em torno de N*f0 ± tolerância."""
    alvo    = ordem * f0
    mascara = np.abs(freqs - alvo) <= tolerancia_hz
    if not np.any(mascara):
        return 0.0
    return float(np.max(amps[mascara]))


def calcular_thd(freqs: np.ndarray, amps: np.ndarray,
                 f0: float, n_harmonicos: int = 13) -> float:
    """
    THD = sqrt(sum(V_n², n=2..N)) / V_1
    Usa o F0 estimado adaptativamente para evitar THD inflado
    por referência errada de frequência.
    """
    v1 = amplitude_harmonica(freqs, amps, f0, 1)
    if v1 < 1e-10:
        return 0.0
    soma = sum(
        amplitude_harmonica(freqs, amps, f0, n) ** 2
        for n in range(2, n_harmonicos + 1)
    )
    return float(np.sqrt(soma) / v1)


def centroide_espectral(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Centro de massa do espectro. Desloca com mudança de distribuição."""
    s = np.sum(amps)
    return float(np.sum(freqs * amps) / s) if s > 1e-10 else 0.0


def largura_banda_espectral(freqs: np.ndarray, amps: np.ndarray,
                             centroide: float) -> float:
    """Dispersão espectral em torno do centróide."""
    s = np.sum(amps)
    if s < 1e-10:
        return 0.0
    return float(np.sqrt(np.sum(((freqs - centroide) ** 2) * amps) / s))


def energia_banda(freqs: np.ndarray, amps: np.ndarray,
                  f_min: float, f_max: float) -> float:
    """Energia (∑ amplitude²) numa banda de frequência."""
    mascara = (freqs >= f_min) & (freqs <= f_max)
    return float(np.sum(amps[mascara] ** 2))


def features_frequencia(sinal: np.ndarray, prefixo: str,
                         f0_real: float,
                         fs: int = FS) -> dict:
    """
    Extrai 11 features espectrais usando F0 estimado adaptativamente.
    Recebe f0_real calculado externamente (em extrair_janela)
    para garantir consistência entre todas as fases.
    """
    freqs, amps = calcular_espectro(sinal, fs)

    thd      = calcular_thd(freqs, amps, f0_real)
    centroid = centroide_espectral(freqs, amps)
    largura  = largura_banda_espectral(freqs, amps, centroid)

    resultado = {
        f"{prefixo}_thd"          : thd,
        f"{prefixo}_centroide"    : centroid,
        f"{prefixo}_largura_banda": largura,
    }

    for h in HARMONICOS:
        resultado[f"{prefixo}_harm_{h}"] = amplitude_harmonica(
            freqs, amps, f0_real, h
        )

    for nome_banda, (fmin, fmax) in BANDAS.items():
        resultado[f"{prefixo}_energia_{nome_banda}"] = energia_banda(
            freqs, amps, fmin, fmax
        )

    return resultado


# ============================================================
# FEATURES INTER-FASE
# ============================================================

def features_interfase(rms_correntes: dict, rms_tensoes: dict,
                        dc: float) -> dict:
    """
    Features que envolvem mais de uma fase:
    desbalanceamento, potência estimada e tensão CC.
    """
    rms_i = np.array([rms_correntes[f"i_{f}_rms"] for f in FASES])
    rms_u = np.array([rms_tensoes[f"u_{f}_rms"]   for f in FASES])

    media_i  = np.mean(rms_i)
    media_u  = np.mean(rms_u)
    desbal_i = (np.max(rms_i) - np.min(rms_i)) / media_i if media_i > 1e-10 else 0.0
    desbal_u = (np.max(rms_u) - np.min(rms_u)) / media_u if media_u > 1e-10 else 0.0

    return {
        "desbalanceamento_corrente": desbal_i,
        "desbalanceamento_tensao"  : desbal_u,
        "potencia_a"               : rms_i[0] * rms_u[0],
        "potencia_b"               : rms_i[1] * rms_u[1],
        "potencia_c"               : rms_i[2] * rms_u[2],
        "tensao_dc_media"          : dc,
    }


# ============================================================
# EXTRAÇÃO POR JANELA
# ============================================================

def extrair_janela(df_janela: pd.DataFrame) -> dict:
    """
    Extrai todas as features de uma janela de 1024 amostras.
    F0 é estimado uma vez a partir da fase A e usado em todas as fases.
    """
    features = {}

    # Estima F0 uma única vez — fase A corrente como referência
    sinal_ref   = df_janela["i_a_k"].values
    freqs_r, amps_r = calcular_espectro(sinal_ref, FS)
    f0_real     = estimar_f0(freqs_r, amps_r, F0)
    features["f0_estimado"] = f0_real

    # ── Correntes CA ─────────────────────────────────────────
    rms_correntes = {}
    for col, fase in zip(COLUNAS_CORRENTE, FASES):
        sinal   = df_janela[col].values
        prefixo = f"i_{fase}"
        ft = features_tempo(sinal, prefixo)
        ff = features_frequencia(sinal, prefixo, f0_real)
        features.update(ft)
        features.update(ff)
        rms_correntes[f"i_{fase}_rms"] = ft[f"{prefixo}_rms"]

    # ── Tensões CA ───────────────────────────────────────────
    rms_tensoes = {}
    for col, fase in zip(COLUNAS_TENSAO, FASES):
        sinal   = df_janela[col].values
        prefixo = f"u_{fase}"
        ft = features_tempo(sinal, prefixo)
        ff = features_frequencia(sinal, prefixo, f0_real)
        features.update(ft)
        features.update(ff)
        rms_tensoes[f"u_{fase}_rms"] = ft[f"{prefixo}_rms"]

    # ── DC ───────────────────────────────────────────────────
    dc_media = float(df_janela[COLUNA_DC].mean())

    # ── Inter-fase ───────────────────────────────────────────
    features.update(features_interfase(rms_correntes, rms_tensoes, dc_media))

    return features


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def executar_features_ca(
    arquivo_csv  : Path = ARQUIVO_CSV,
    pasta_saida  : Path = PASTA_SAIDA,
    limite_linhas: int  = None
) -> bool:
    """
    Pipeline completo de extração de features CA.
    Retorna True se concluiu com sucesso.
    """
    _log("=" * 60)
    _log("  AL IADO PV — FEATURES CA (Paderborn)")
    _log("=" * 60)

    # 1. Carrega dados
    _log(f"\n📂 Carregando dataset...")
    if not arquivo_csv.exists():
        _log(f"   ❌ Não encontrado: {arquivo_csv}")
        return False

    df = pd.read_csv(arquivo_csv, nrows=limite_linhas)
    _log(f"   ✅ {len(df):,} amostras | {df.shape[1]} colunas")

    faltando = [c for c in COLUNAS_CORRENTE + COLUNAS_TENSAO + [COLUNA_DC]
                if c not in df.columns]
    if faltando:
        _log(f"   ❌ Colunas ausentes: {faltando}")
        return False

    # 2. Janelamento
    n_total   = len(df)
    n_janelas = (n_total - JANELA) // PASSO + 1
    _log(f"\n🪟  Janelamento:")
    _log(f"   Janela       : {JANELA} amostras = {JANELA/FS*1000:.1f} ms")
    _log(f"   Sobreposição : {SOBREPOSICAO} amostras (50%)")
    _log(f"   Duração total: {n_total/FS:.1f} s")
    _log(f"   Janelas      : {n_janelas:,}")

    # 3. Extração
    _log(f"\n⚙️  Extraindo features...")
    registros = []
    for i in range(n_janelas):
        inicio = i * PASSO
        fim    = inicio + JANELA
        feats  = extrair_janela(df.iloc[inicio:fim])
        feats["janela_idx"]     = i
        feats["amostra_inicio"] = inicio
        feats["tempo_s"]        = inicio / FS
        registros.append(feats)

        if (i + 1) % 500 == 0 or i == n_janelas - 1:
            _log(f"   [{(i+1)/n_janelas*100:5.1f}%] {i+1:,}/{n_janelas:,}", end="\r")

    _log()
    df_feat = pd.DataFrame(registros)

    # 4. Remove features com média AC ≈ 0
    colunas_remover = [c for c in FEATURES_EXCLUIR if c in df_feat.columns]
    df_feat.drop(columns=colunas_remover, inplace=True)
    _log(f"   ✅ {len(df_feat):,} janelas × "
          f"{len(df_feat.columns)-3} features "
          f"(removidas: {len(colunas_remover)} médias AC)")

    # 5. Salva
    pasta_saida.mkdir(parents=True, exist_ok=True)

    arq_parquet = pasta_saida / "features_paderborn.parquet"
    df_feat.to_parquet(arq_parquet, index=False)
    _log(f"\n💾 {arq_parquet.name} ({arq_parquet.stat().st_size/1024:.0f} KB)")

    arq_stats = pasta_saida / "features_paderborn_stats.csv"
    df_feat.describe().T.to_csv(arq_stats)
    _log(f"   {arq_stats.name}")

    # 6. Resumo
    meta     = ["janela_idx", "amostra_inicio", "tempo_s"]
    col_feat = [c for c in df_feat.columns if c not in meta]

    grupos = {
        "Tempo — corrente" : [c for c in col_feat if c.startswith("i_") and
                              any(s in c for s in ["rms","desvio","kurtosis",
                                                    "skewness","fator","pico"])],
        "Tempo — tensão"   : [c for c in col_feat if c.startswith("u_") and
                              any(s in c for s in ["rms","desvio","kurtosis",
                                                    "skewness","fator","pico"])],
        "Freq — corrente"  : [c for c in col_feat if c.startswith("i_") and
                              any(s in c for s in ["thd","harm","centroide",
                                                    "largura","energia"])],
        "Freq — tensão"    : [c for c in col_feat if c.startswith("u_") and
                              any(s in c for s in ["thd","harm","centroide",
                                                    "largura","energia"])],
        "Inter-fase + F0"  : [c for c in col_feat if
                              any(s in c for s in ["desbalanceamento",
                                                    "potencia","dc","f0"])],
    }

    _log(f"\n📊 Features por grupo:")
    total = 0
    for grupo, cols in grupos.items():
        _log(f"   {grupo:<22}: {len(cols):>3}")
        total += len(cols)
    _log(f"   {'TOTAL':<22}: {total:>3}")

    # THD resumo
    _log(f"\n📈 THD das correntes (F0 adaptativo):")
    for fase in FASES:
        col = f"i_{fase}_thd"
        if col in df_feat.columns:
            _log(f"   {col}: mean={df_feat[col].mean():.4f}  "
                  f"std={df_feat[col].std():.4f}  "
                  f"max={df_feat[col].max():.4f}")

    _log(f"\n📈 F0 estimado:")
    _log(f"   mean={df_feat['f0_estimado'].mean():.2f} Hz  "
          f"std={df_feat['f0_estimado'].std():.2f} Hz  "
          f"min={df_feat['f0_estimado'].min():.2f}  "
          f"max={df_feat['f0_estimado'].max():.2f}")

    n_nan = df_feat[col_feat].isna().sum().sum()
    if n_nan > 0:
        _log(f"\n   ⚠️  {n_nan} valores NaN detectados")
    else:
        _log(f"\n   ✅ Nenhum NaN — dados prontos para o Autoencoder")

    _log(f"\n{'='*60}")
    _log(f"  EXTRAÇÃO CONCLUÍDA!")
    _log(f"  Próximo passo: Autoencoder em src/ml/autoencoder.py")
    _log(f"{'='*60}")
    return True


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    from src.core.logs import habilitar_console
    habilitar_console()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limite", type=int, default=None,
                        help="Limita linhas (ex: 50000 para teste)")
    args = parser.parse_args()
    executar_features_ca(limite_linhas=args.limite)
