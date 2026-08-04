"""
injecao_falhas.py — Al IAdo PV / Fase 5
Injeção de falhas sintéticas fundamentada na FMECA do TCC (Torres, 2024).
FONTE ÚNICA dos componentes/modos/índices: docs/fmeca.md.

Fundamentação metodológica:
  A FMECA (FMEA + Criticidade; NPR = S×O×D) aponta o inversor como o
  componente mais crítico do SFV. Pela Tab. 3.3 do TCC (Cristaldi et al.,
  2017), os componentes CA-elétricos do inversor que mais falham — e são
  detectáveis no sinal — são Contator AC, IGBT e Fusível AC.

  A prioridade NPR define a ordem de injeção: primeiro as falhas de maior
  criticidade. Cada falha é modelada pela sua assinatura elétrica esperada
  nos sinais de corrente CA.

Falhas implementadas (em ordem de NPR; índices S/O/D e NPR em docs/fmeca.md):
  Id.1 — Contator AC (NPR=315, S=5·O=7·D=9 — mais crítico)
    Assinatura: transiente/ruído de comutação na corrente CA (proxy)
    Física: contatos desgastados/soldados (chattering) → comutação deficiente

  Id.2 — IGBT (NPR=90, S=5·O=6·D=3)
    Assinatura: elevação de THD e harmônicos 5°/7°/11°/13°
    Física: IGBT envelhecido (bond wire, Vce↑) → chaveamento imperfeito

  Id.3 — Fusível AC (NPR=30, S=5·O=3·D=2)
    Assinatura: redução de amplitude de uma fase (desbalanceamento)
    Física: fusível degradado/rompido → perda parcial de fase

Estratégia de severidade:
  Cada falha é injetada em 7 níveis de severidade [0.05→1.0] para
  identificar a severidade mínima detectável (SMD) — ponto onde o erro
  de reconstrução cruza o limiar do Autoencoder.
  ATENÇÃO: o índice D da FMECA (detecção EM CAMPO) e a detectabilidade
  empírica do Autoencoder são conceitos distintos (ver docs/fmeca.md).

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
from src.ml.estilo_graficos import (
    COR_ALERTA, TAM, aplicar_estilo, salvar_figura,
)

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
from src.ml.dados_avaliacao import carregar_paderborn_compacto, preparar_janelas_holdout
from src.ml.estatistica import intervalo_wilson

# ── Caminhos ─────────────────────────────────────────────────
RAIZ          = Path(__file__).parent.parent.parent
ARQUIVO_CSV   = RAIZ / "dados" / "brutos" / "Inverter_Data_Set.csv"
PASTA_AE      = RAIZ / "resultados" / "autoencoder"

# ── Parâmetros de injeção ────────────────────────────────────
# Severidades: de muito leve a severa
SEVERIDADES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
ALVO_SMD = 0.95
N_JANELAS_SMD = 100  # limitado pelo numero de janelas nao sobrepostas do holdout

# ── Falhas FMECA — FONTE ÚNICA: docs/fmeca.md ────────────────
# Componentes CA-elétricos do inversor que mais falham (Tab. 3.3 do TCC,
# Cristaldi et al. 2017). NPR = S×O×D (índice da FMECA; D NUNCA é o NPR).
# Índices S/O/D estipulados pelo pesquisador (Torres, 2024) — ver docs/fmeca.md.
# Modo de falha / efeito / causa: reservados para preenchimento por Rodolfo.
# Ordenadas por criticidade (NPR): Contator AC > IGBT > Fusível AC.
FALHAS = [
    {
        "id"      : "contator_ac",
        "nome"    : "Contator AC",
        "componente": "Contator AC",
        "s"       : 5, "o": 7, "d": 9, "npr": 5 * 7 * 9,   # = 315
        "criticidade": 5 + 7,                               # C = S+O = 12
        # Modo/efeito/causa: FMECA preenchida por Rodolfo (docs/fmeca.md).
        "modo_falha": ("Fuga de corrente; injeção contínua de energia ainda "
                       "com a falta de energia da concessionária ou oscilação "
                       "severa"),
        "efeito"    : ("Paralisação do sistema por falha no isolamento; riscos "
                       "de eletrocussão em técnicos operando na rede externa"),
        "causa"     : ("Arco elétrico e desgaste mecânico; degradação da bobina "
                       "e do isolamento"),
        "cor"     : "#2a78d6",  # PALETA[0] — ver estilo_graficos.CORES_FALHAS
        "descricao": "Transiente/ruído de comutação na corrente CA",
        # Schema de proveniência da falha sintética (item 4.4)
        "evidence_level"     : "E2",
        "hipotese_fisica"    : (
            "Contatos do contator CA desgastados/soldados (chattering) causam "
            "comutação deficiente, introduzindo transientes e conteúdo de alta "
            "frequência na corrente CA — modelado como ruído no sinal medido."
        ),
        "sinais"             : ["i_a"],
        "formula"            : "i_a += N(0, sev·σ_sinal·0,3)  (proxy do transiente de comutação)",
        "severity_definition": "desvio do ruído ≈ sev·σ_sinal·0,3 (componente mais crítico, NPR=315)",
        "source"             : (
            "Torres (2024) TCC — Tab. 3.3 (Cristaldi et al., 2017: Contator AC "
            "= 12% dos tickets); Golnas (2012); Voss et al. (2009). "
            "S/O/D estipulados pelo pesquisador; NPR=S×O×D=315."
        ),
        "limitations"        : [
            "RUÍDO GAUSSIANO É UM PROXY do transiente — exige CALIBRAÇÃO FÍSICA "
            "do contator real",
            "a detectabilidade observada é E2 (sintética); o índice D=9 da FMECA "
            "refere-se à detecção EM CAMPO, não à do Autoencoder (ver docs/fmeca.md)",
        ],
    },
    {
        "id"      : "igbt",
        "nome"    : "IGBT",
        "componente": "IGBT",
        "s"       : 5, "o": 6, "d": 3, "npr": 5 * 6 * 3,   # = 90
        "criticidade": 5 + 6,                               # C = 11
        # Modo/efeito/causa: FMECA preenchida por Rodolfo (docs/fmeca.md).
        "modo_falha": ("Não comutação CC→CA; curto-circuito permanente entre "
                       "os terminais"),
        "efeito"    : ("Interrupção imediata no fornecimento de energia e "
                       "(possível) disparo de alarme de hardware no display"),
        "causa"     : "Estresse termodinâmico e surtos de sobretensão",
        "cor"     : "#1baf7a",  # PALETA[1]
        "descricao": "Injeção de harmônicos 5°, 7°, 11° e 13° nas correntes CA",
        "evidence_level"     : "E2",
        "hipotese_fisica"    : (
            "IGBT envelhecido (lift-off de bond wire, Vce(sat) elevado) comuta de "
            "forma imperfeita, elevando os harmônicos ímpares e o THD das correntes."
        ),
        "sinais"             : ["i_a", "i_b", "i_c"],
        "formula"            : "i += sev·(0,30·h5 + 0,20·h7 + 0,10·h11 + 0,05·h13)·amplitude",
        "severity_definition": "fração [0..1] da amplitude harmônica injetada",
        "source"             : (
            "Torres (2024) TCC — Tab. 3.3 (IGBT = 6% dos tickets); harmônicos "
            "característicos de VSI (Francisti, 2025; Smith, 1999). "
            "S/O/D estipulados pelo pesquisador; NPR=S×O×D=90."
        ),
        "limitations"        : [
            "amplitudes harmônicas são plausíveis, não medidas em bancada",
            "não modela envelhecimento térmico real do IGBT",
        ],
    },
    {
        "id"      : "fusivel_ac",
        "nome"    : "Fusível AC",
        "componente": "Fusível AC",
        "s"       : 5, "o": 3, "d": 2, "npr": 5 * 3 * 2,   # = 30
        "criticidade": 5 + 3,                               # C = 8
        # Modo/efeito/causa: FMECA preenchida por Rodolfo (docs/fmeca.md).
        "modo_falha": "Interrupção da condução de corrente (abertura do elo fusível)",
        "efeito"    : ("Isolamento de uma ou mais fases da saída CA; interrupção "
                       "no fornecimento de energia; desarme do inversor por "
                       "desbalanceamento de fases"),
        "causa"     : "Fadiga térmica por ciclos de carga; surtos de rede",
        "cor"     : "#eda100",  # PALETA[2] — baixo contraste: exige rótulo direto
        "descricao": "Redução de amplitude de uma fase (perda parcial)",
        "evidence_level"     : "E2",
        "hipotese_fisica"    : (
            "Fusível CA degradado/rompido causa perda parcial de uma fase, "
            "reduzindo a amplitude da corrente dessa fase (desbalanceamento)."
        ),
        "sinais"             : ["i_a"],
        "formula"            : "i_a ·= (1 − sev·0,12)  (reduz amplitude da fase A; máx 12%)",
        "severity_definition": "fração de redução da amplitude da fase A (calibrada: máx 12%)",
        "source"             : (
            "Torres (2024) TCC — Tab. 3.3 (Fusíveis AC = 4% dos tickets, 12% "
            "da energia perdida). S/O/D estipulados pelo pesquisador; NPR=S×O×D=30."
        ),
        "limitations"        : [
            "modelo simplificado; perda de fase real pode desarmar a proteção",
            "redução máx de 12% p/ manter a curva severidade↔detecção plausível",
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

    taxas, n_rep, intervalos = {}, {}, {}
    for sev, dets in deteccoes_por_severidade.items():
        arr = np.asarray(list(dets), dtype=float)
        taxas[float(sev)] = float(arr.mean()) if arr.size else 0.0
        n_rep[float(sev)] = int(arr.size)
        low, high = intervalo_wilson(int(arr.sum()), int(arr.size))
        intervalos[float(sev)] = {"low": low, "high": high}

    sevs = sorted(taxas)
    smd_pontual = next((s for s in sevs if taxas[s] > 0.0), None)
    smd_95 = next((s for s in sevs if taxas[s] >= alvo), None)
    smd_95_conservadora = next(
        (s for s in sevs if intervalos[s]["low"] >= alvo), None
    )
    return {
        "taxa_deteccao": taxas,
        "intervalo_wilson_95": intervalos,
        "smd_pontual": smd_pontual,
        "smd_95": smd_95,
        "smd_95_conservadora": smd_95_conservadora,
        "alvo": alvo,
        "n_repeticoes": n_rep,
    }


# ============================================================
# MODELOS DE FALHA (assinaturas elétricas)
# ============================================================

def falha_harmonicos_igbt(janela_df: pd.DataFrame,
                          severidade: float,
                          f0: float = F0,
                          fs: int   = FS) -> pd.DataFrame:
    """
    FALHA — IGBT (NPR=90)

    Um IGBT envelhecido (lift-off de bond wire, Vce(sat) elevado) comuta de
    forma imperfeita. O resultado é um aumento do THD e elevação específica
    dos harmônicos de ordem 5, 7 e 11 (dominantes em inversores VSI trifásicos).

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

        # Harmônicos característicos de inversores VSI com IGBT degradado
        # Amplitudes relativas baseadas em Francisti (2025) e Smith (1999)
        h5  = severidade * 0.30 * amplitude * np.sin(2 * np.pi * 5  * f0 * t)
        h7  = severidade * 0.20 * amplitude * np.sin(2 * np.pi * 7  * f0 * t)
        h11 = severidade * 0.10 * amplitude * np.sin(2 * np.pi * 11 * f0 * t)
        h13 = severidade * 0.05 * amplitude * np.sin(2 * np.pi * 13 * f0 * t)

        janela_falha[col] = sinal + h5 + h7 + h11 + h13

    return janela_falha


def falha_perda_fase_fusivel(janela_df: pd.DataFrame,
                             severidade: float) -> pd.DataFrame:
    """
    FALHA — Fusível AC (NPR=30)

    Um fusível CA degradado/rompido causa perda parcial de uma fase, reduzindo
    a amplitude da corrente dessa fase (desbalanceamento). Medido pela feature
    inter-fase:
      desbalanceamento = (max_rms - min_rms) / media_rms

    Fator de redução CALIBRADO p/ curva severidade↔detecção realista
    (fator = 1 - severidade × 0.12):
      severidade=0.3 → ~3,6% de redução (incipiente, ~limiar FMEA de 5%, DIFÍCIL)
      severidade=0.5 → 6% de redução (perceptível)
      severidade=1.0 → 12% de redução (perda severa, mas plausível)

    Antes usava ×0.7 (até 70% de redução) → o detector separava com erro ZERO
    em qualquer severidade (validacao_report perfeito = artificial). Uma perda
    parcial real raramente passa de ~10–15% sem desarme de proteção.

    A fase A é a referência (Id.3 da FMECA consolidada — docs/fmeca.md).
    """
    janela_falha = janela_df.copy()
    fator        = 1.0 - severidade * 0.12

    # Afeta corrente e tensão da fase A
    janela_falha["i_a_k"] = janela_df["i_a_k"].values * fator
    if "u_a_k-1" in janela_falha.columns:
        janela_falha["u_a_k-1"] = janela_df["u_a_k-1"].values * fator

    return janela_falha


def falha_transiente_contator(janela_df: pd.DataFrame,
                              severidade: float,
                              seed: int = 0) -> pd.DataFrame:
    """
    FALHA — Contator AC (NPR=315, componente mais crítico)

    Contatos do contator CA desgastados/soldados (chattering) causam comutação
    deficiente, introduzindo transientes e conteúdo de alta frequência na
    corrente CA — modelado aqui como ruído gaussiano no sinal medido (proxy).

    Modelagem: sinal_falha = sinal + N(0, σ_ruído)
    onde σ_ruído = severidade × std(sinal) × 0.3.

    O multiplicador ×0.3 produz uma curva severidade↔detecção realista
    (varredura empírica):
      severidade=0.30 → σ_ruído ≈ 0,09·σ_sinal (SNR ~21 dB) — difícil
      severidade=0.50 → σ_ruído ≈ 0,15·σ_sinal — limítrofe
      severidade=1.00 → σ_ruído ≈ 0,30·σ_sinal (SNR ~10 dB) — detectada

    Nota: features espectrais (THD/harmônicos/kurtosis) são naturalmente
    sensíveis a ruído branco — por isso o multiplicador precisa ser pequeno.
    O índice D=9 da FMECA (docs/fmeca.md) refere-se à detecção EM CAMPO,
    distinta da detectabilidade empírica do Autoencoder.
    """
    rng          = np.random.default_rng(seed)
    janela_falha = janela_df.copy()

    sinal   = janela_df["i_a_k"].values
    std_sig = np.std(sinal)
    ruido   = rng.normal(0, severidade * std_sig * 0.3, size=len(sinal))

    janela_falha["i_a_k"] = sinal + ruido
    return janela_falha


# Mapa de funções de falha (id da FMECA → assinatura elétrica)
FUNCOES_FALHA = {
    "contator_ac": falha_transiente_contator,
    "igbt"       : falha_harmonicos_igbt,
    "fusivel_ac" : falha_perda_fase_fusivel,
}


# ============================================================
# INFERÊNCIA COM O AUTOENCODER
# ============================================================

def calcular_erro_reconstrucao(janela_df: pd.DataFrame,
                                modelo: Autoencoder,
                                scaler,
                                device: torch.device,
                                colunas_feat: list,
                                estat_residuo: dict | None = None,
                                metodo: str = "mse") -> float:
    """
    Extrai features de uma janela, normaliza e calcula o ESCORE de anomalia
    do Autoencoder (fonte única: src/ml/escore_anomalia.py).

    Por padrão (`metodo="mse"`, sem régua) devolve o MSE médio — comportamento
    histórico. Com `metodo="localizado"` e a régua `estat_residuo` (μ/σ por
    feature), devolve o escore localizado (top-k dos resíduos padronizados),
    sensível a falha concentrada em poucas features.
    """
    from src.ml import escore_anomalia as ea

    feats = extrair_janela(janela_df)
    vetor = np.array([feats.get(c, 0.0) for c in colunas_feat],
                     dtype=np.float32)
    vetor_norm = scaler.transform(vetor.reshape(1, -1)).astype(np.float32)
    residuo = ea.residuo_de_vetor(modelo, vetor_norm, device)
    return float(ea.pontuar(residuo, estat_residuo, metodo)[0])


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
    _log("\n  Fundamentação: FMECA Torres (2024) — docs/fmeca.md")
    _log("  NPR=315 → Contator AC (S=5·O=7·D=9)")
    _log("  NPR=90  → IGBT (S=5·O=6·D=3)")
    _log("  NPR=30  → Fusível AC (S=5·O=3·D=2)")

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
    limiar       = info_limiar["limiar"]   # OPERACIONAL (método escolhido)

    # Escore operacional (fonte única). O método e a régua por-feature vêm do
    # limiar.json/estatistica_residuo.npz gravados pelo autoencoder. Sem eles
    # (artefato antigo), cai para MSE — o comportamento histórico.
    from src.ml import escore_anomalia as ea

    metodo_escore = info_limiar.get("metodo_escore", "mse")
    estat_residuo = ea.carregar_estatistica(PASTA_AE)
    _log(f"   ✅ Escore: {ea.descricao_metodo(metodo_escore, info_limiar.get('k_localizado', 5))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Autoencoder(n_features, latente_dim).to(device)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    _log(f"   ✅ Modelo: {n_features} features → latente {latente_dim}")
    _log(f"   ✅ Limiar de anomalia: {limiar:.4f}")

    # ── 2. Carrega dataset e seleciona janelas estáveis ──────
    _log(f"\n📂 Carregando dataset de Paderborn...")
    df = carregar_paderborn_compacto(ARQUIVO_CSV)
    janelas_holdout, meta_holdout = preparar_janelas_holdout(
        df, n_max=N_JANELAS_SMD
    )
    del df
    _log(
        f"   ✅ {len(janelas_holdout)} janelas não sobrepostas do bloco de teste "
        "(treino/calibração excluídos)"
    )

    # ── 3. Erro baseline (comportamento saudável) ─────────────
    _log(f"\n⚕️  Calculando erro baseline (saudável)...")
    erros_baseline = []
    for janela in janelas_holdout:
        erro   = calcular_erro_reconstrucao(
            janela, modelo, scaler, device, colunas_feat,
            estat_residuo, metodo_escore,
        )
        erros_baseline.append(erro)

    baseline_mean = np.mean(erros_baseline)
    baseline_std  = np.std(erros_baseline)
    _log(f"   Baseline: μ={baseline_mean:.4f} ± {baseline_std:.4f}")
    _log(f"   Limiar  : {limiar:.4f} "
          f"({limiar/baseline_mean:.1f}× acima do baseline)")

    # ── 4. Injeção de falhas por severidade ──────────────────
    _log(f"\n💉 Injetando falhas (3 tipos × {len(SEVERIDADES)} severidades)...")

    resultados = {}
    smd_detalhado = {}

    for falha in FALHAS:
        fid  = falha["id"]
        fn   = FUNCOES_FALHA[fid]
        erros_por_sev = {}
        deteccoes_por_sev = {}

        _log(f"\n   🔴 {falha['nome']} (NPR={falha['npr']})")

        for sev in SEVERIDADES:
            erros_sev = []
            for j, janela_base in enumerate(janelas_holdout):
                if fid == "contator_ac":
                    janela_falha = fn(janela_base, sev, seed=10_000 + j)
                else:
                    janela_falha = fn(janela_base, sev)
                erro = calcular_erro_reconstrucao(
                    janela_falha, modelo, scaler, device, colunas_feat,
                    estat_residuo, metodo_escore,
                )
                erros_sev.append(erro)

            erros_arr = np.asarray(erros_sev, dtype=float)
            deteccoes = erros_arr > limiar
            taxa = float(deteccoes.mean())
            ci_low, ci_high = intervalo_wilson(
                int(deteccoes.sum()), len(deteccoes)
            )
            erro_medio = float(erros_arr.mean())
            erro_mediano = float(np.median(erros_arr))
            detectado = taxa >= ALVO_SMD
            margem = erro_mediano / limiar
            deteccoes_por_sev[sev] = deteccoes.tolist()

            erros_por_sev[sev] = {
                "erro"      : erro_medio,
                "erro_mediano": erro_mediano,
                "erro_q25"  : float(np.percentile(erros_arr, 25)),
                "erro_q75"  : float(np.percentile(erros_arr, 75)),
                "detectado" : detectado,
                "margem"    : margem,
                "taxa_deteccao": taxa,
                "taxa_ci_low": ci_low,
                "taxa_ci_high": ci_high,
                "n": len(erros_arr),
            }

            status = "✅ alvo atingido" if detectado else "⬜ abaixo do alvo"
            _log(
                f"      sev={sev:.2f} | erro mediano={erro_mediano:.4f} | "
                f"detecção={taxa:.1%} (IC95% {ci_low:.1%}–{ci_high:.1%}) | {status}"
            )

        resultados[fid] = erros_por_sev
        smd_detalhado[fid] = smd_probabilistico(
            deteccoes_por_sev, alvo=ALVO_SMD
        )

    # ── 5. Severidade mínima detectável (SMD) ────────────────
    _log(f"\n🎯 Severidade Mínima Detectável (SMD):")
    smd_report = {}
    for falha in FALHAS:
        fid = falha["id"]
        smd = smd_detalhado[fid]["smd_95"]
        smd_report[fid] = smd
        if smd:
            _log(f"   {falha['nome']:<30}: SMD95 = {smd:.2f} "
                  f"(taxa = {resultados[fid][smd]['taxa_deteccao']:.1%})")
        else:
            _log(f"   {falha['nome']:<30}: nenhuma severidade atingiu {ALVO_SMD:.0%}")

    # ── 6. Visualizações ─────────────────────────────────────
    _log(f"\n📊 Gerando gráficos...")
    PASTA_AE.mkdir(parents=True, exist_ok=True)

    # Gráfico 1: mediana e intervalo interquartil do erro por falha.
    fig, axes = plt.subplots(
        1, 3, figsize=TAM["painel_3"], sharey=True, layout="constrained"
    )
    fig.suptitle("Injeção sintética — distribuição do erro no holdout temporal")

    for ax, falha in zip(axes, FALHAS):
        fid   = falha["id"]
        sevs = np.asarray(SEVERIDADES, dtype=float)
        medianas = np.asarray([resultados[fid][s]["erro_mediano"] for s in SEVERIDADES])
        q25 = np.asarray([resultados[fid][s]["erro_q25"] for s in SEVERIDADES])
        q75 = np.asarray([resultados[fid][s]["erro_q75"] for s in SEVERIDADES])

        ax.plot(sevs, medianas, marker="o", color=falha["cor"], label="Mediana")
        ax.fill_between(sevs, q25, q75, color=falha["cor"], alpha=0.18,
                        label="Intervalo interquartil")
        ax.axhline(limiar, color=COR_ALERTA, linestyle="--", linewidth=1.5,
                   label=f"Limiar = {limiar:.2f}")
        ax.axhline(baseline_mean, color="#147a3d", linestyle=":",
                   linewidth=1.2, label=f"Baseline = {baseline_mean:.4f}")

        ax.set_title(f"{falha['nome']}\n(NPR={falha['npr']})", fontsize=10)
        ax.set_xlabel("Severidade")
        ax.set_ylabel("Erro de reconstrução (MSE, escala log)")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

        smd = smd_report[fid]
        if smd:
            y_smd = resultados[fid][smd]["erro_mediano"]
            ax.scatter([smd], [y_smd], s=90, facecolors="none",
                       edgecolors="black", linewidths=1.5, zorder=4)
            ax.annotate(f"SMD95={smd}", xy=(smd, y_smd), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=8)

    arq_g1 = PASTA_AE / "injecao_falhas_resultados.png"
    salvar_figura(
        fig,
        arq_g1,
        "E2 sintético. Cada ponto resume janelas não sobrepostas do teste; a faixa mostra Q25–Q75.",
    )
    _log(f"   📊 {arq_g1.name}")

    # Gráfico 2: probabilidade de detecção e IC de Wilson.
    fig, ax = plt.subplots(figsize=TAM["unico"], layout="constrained")
    for falha in FALHAS:
        fid = falha["id"]
        taxas = np.asarray([resultados[fid][s]["taxa_deteccao"] for s in SEVERIDADES])
        lows = np.asarray([resultados[fid][s]["taxa_ci_low"] for s in SEVERIDADES])
        highs = np.asarray([resultados[fid][s]["taxa_ci_high"] for s in SEVERIDADES])
        erros_y = np.vstack([taxas - lows, highs - taxas])
        ax.errorbar(
            SEVERIDADES, taxas, yerr=erros_y, color=falha["cor"], marker="o",
            capsize=3, label=f"{falha['nome']} (NPR={falha['npr']})",
        )

    ax.axhline(ALVO_SMD, color=COR_ALERTA, linestyle="--",
               label=f"Alvo SMD = {ALVO_SMD:.0%}")
    ax.set_xlabel("Severidade da falha")
    ax.set_ylabel("Taxa de detecção no limiar operacional")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0, 1, 6), labels=[f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_title("Detectabilidade por severidade\nIntervalos de Wilson de 95%")
    ax.legend(loc="best")

    arq_g2 = PASTA_AE / "injecao_falhas_comparacao.png"
    salvar_figura(
        fig,
        arq_g2,
        "SMD95 é a menor severidade cuja taxa pontual atinge 95%; consulte o IC antes de concluir detectabilidade.",
    )
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
        "score_method": info_limiar.get(
            "score_method", info_limiar.get("metodo_escore")
        ),
        "score_threshold": info_limiar.get(
            "score_threshold", info_limiar.get("limiar")
        ),
        "mse_p99": info_limiar.get("mse_p99", info_limiar.get("limiar_p99")),
        "sigma_multiplier": info_limiar.get(
            "sigma_multiplier", info_limiar.get("k")
        ),
        "top_k": info_limiar.get("top_k", info_limiar.get("k_localizado")),
        "threshold_fallback_percentile": info_limiar.get(
            "threshold_fallback_percentile"
        ),
        "threshold_effective_percentile": info_limiar.get(
            "threshold_effective_percentile",
            info_limiar.get("percentil_limiar"),
        ),
        "threshold_source": info_limiar.get(
            "threshold_source", "bloco_calibracao_temporal"
        ),
        "limiar": float(limiar),
        "baseline_mean": float(baseline_mean),
        "baseline_std": float(baseline_std),
        "protocolo_avaliacao": meta_holdout,
        "alvo_smd": ALVO_SMD,
        "smd": {k: float(v) if v is not None else None
                for k, v in smd_report.items()},
        "smd_probabilistica": smd_detalhado,
        "falhas": {}
    }
    for falha in FALHAS:
        fid = falha["id"]
        relatorio["falhas"][fid] = {
            "nome": falha["nome"],
            "componente": falha.get("componente", falha["nome"]),
            # Índices FMECA (fonte única: docs/fmeca.md). NPR = S×O×D.
            "s": falha["s"], "o": falha["o"], "d": falha["d"],
            "npr": falha["npr"],
            "criticidade": falha.get("criticidade"),
            "modo_falha": falha.get("modo_falha", ""),
            "efeito": falha.get("efeito", ""),
            "causa": falha.get("causa", ""),
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
                    "erro_mediano": float(resultados[fid][s]["erro_mediano"]),
                    "erro_q25": float(resultados[fid][s]["erro_q25"]),
                    "erro_q75": float(resultados[fid][s]["erro_q75"]),
                    "detectado": bool(resultados[fid][s]["detectado"]),
                    "margem": float(resultados[fid][s]["margem"]),
                    "taxa_deteccao": float(resultados[fid][s]["taxa_deteccao"]),
                    "taxa_ci_low": float(resultados[fid][s]["taxa_ci_low"]),
                    "taxa_ci_high": float(resultados[fid][s]["taxa_ci_high"]),
                    "n": int(resultados[fid][s]["n"]),
                }
                for s in SEVERIDADES
            }
        }

    arq_report = PASTA_AE / "injecao_falhas_report.json"
    with open(arq_report, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_report.name}")

    linhas_smd = []
    for falha in FALHAS:
        fid = falha["id"]
        for sev in SEVERIDADES:
            res = resultados[fid][sev]
            linhas_smd.append({
                "falha": falha["nome"],
                "falha_id": fid,
                "npr": falha["npr"],
                "severidade": sev,
                "n": res["n"],
                "erro_mediano": res["erro_mediano"],
                "erro_q25": res["erro_q25"],
                "erro_q75": res["erro_q75"],
                "taxa_deteccao": res["taxa_deteccao"],
                "taxa_ci_low": res["taxa_ci_low"],
                "taxa_ci_high": res["taxa_ci_high"],
                "atinge_alvo_smd": res["detectado"],
                "evidence_level": "E2",
            })
    arq_tabela = PASTA_AE / "injecao_smd_tabela.csv"
    pd.DataFrame(linhas_smd).to_csv(arq_tabela, index=False)
    _log(f"   📋 {arq_tabela.name}")

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
