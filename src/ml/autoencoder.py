"""
autoencoder.py — Al IAdo PV / Fase 5
Modelagem de normalidade com Autoencoder para detecção de anomalias
no lado CA do inversor fotovoltaico.

Fundamentação:
  O Autoencoder aprende a reconstruir o comportamento SAUDÁVEL do inversor
  a partir dos ensaios saudáveis F0L/F0M do conjunto experimental GPVS-Faults,
  obtido em microrede fotovoltaica conectada à rede. Em operação, sinais anômalos
  (falhas) produzem erro de reconstrução alto — acima do limiar operacional
  (percentil 99 do erro saudável). μ + 3σ é mantido apenas como referência
  teórica comparativa, não como limiar operacional.

  Esta abordagem é adequada porque dados de falha raramente estão
  disponíveis em manutenção preditiva real; Ibrahim (2022) usa o mesmo
  princípio de modelagem de normalidade por erro de reconstrução.

Arquitetura:
  Entrada : n_features normalizadas (RobustScaler)
  Encoder : n_features → 16 → latente (ReLU + Dropout 0.2; gargalo SEM ativação)
  Latente : 8 dimensões por padrão
  Decoder : latente → 16 → n_features (ReLU + saída Linear)
  Loss    : MSE — erro de reconstrução por janela
  Limiar  : percentil 99 do erro saudável no bloco de calibração (operacional);
            μ + 3σ é referência comparativa, não o limiar em uso

Entrada : dados/processados/features_gpvs.parquet
Saída   : resultados/autoencoder/
            modelo_autoencoder.pt   ← pesos do modelo
            scaler.pkl              ← RobustScaler ajustado
            limiar.json             ← limiar de anomalia + metadados
            calibracao_autoencoder.csv/md
                                    ← auditoria tabular da calibração
            curva_treino.png        ← loss por época
            distribuicao_erro.png   ← distribuição do erro + limiar

Uso:
  python src/ml/autoencoder.py
  python src/ml/autoencoder.py --epochs 200 --latente 8

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta: python src/ml/<arquivo>.py
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("autoencoder")
_log = _adaptar_log(_logger)


import json
import os
import pickle
import argparse
import numpy as np
import pandas as pd
# O estilo e as figuras vivem em src/ml/graficos_autoencoder.py (sem torch).
from src.ml.estilo_graficos import aplicar_estilo

aplicar_estilo()

from pathlib import Path
from src.core.tempo import agora_local

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
    DataLoader = TensorDataset = None

    class _NNIndisponivel:
        Module = object

        def __getattr__(self, nome):
            raise ModuleNotFoundError(
                "O PyTorch e necessario para treinar ou executar o autoencoder."
            )

    nn = _NNIndisponivel()
from sklearn.preprocessing import RobustScaler

from src.ml import escore_anomalia as ea


def _exigir_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "O PyTorch e necessario para treinar ou executar o autoencoder."
        )

# ── Caminhos ─────────────────────────────────────────────────
RAIZ           = Path(__file__).parent.parent.parent
ARQUIVO_FEAT   = RAIZ / "dados" / "processados" / "features_gpvs.parquet"
PASTA_SAIDA    = RAIZ / "resultados" / "autoencoder"

# ── Hiperparâmetros padrão ────────────────────────────────────
LATENTE_DIM    = 8      # dimensão do espaço latente (varredura: 4/8/16)
EPOCHS         = 150    # épocas de treinamento
BATCH_SIZE     = 32     # amostras por batch
LR             = 1e-3   # taxa de aprendizado (Adam)
DROPOUT        = 0.2    # regularização
# Split temporal. MANTIDO COMO LITERAL de propósito: src/ml/pipeline.py lê
# estas constantes por AST, SEM importar o módulo, para registrar a proveniência
# num ambiente sem torch (ver _parametros_do_fonte). Trocar por os.getenv()
# quebra essa leitura — e "consertar" o leitor para entender getenv seria pior,
# porque o manifesto passaria a gravar o DEFAULT em vez do valor efetivamente
# usado, que é exatamente o oposto do que proveniência serve.
#
# Para varrer o split num experimento, edite aqui e registre a rodada; não há
# atalho por variável de ambiente, e isso é deliberado.
#
# F0L e F0M são divididos separadamente. Cada papel é disjunto e a purga fica
# em src/ml/gpvs.py; estes literais são registrados no manifesto por AST.
TRAIN_RATIO      = 0.50  # ajuste do scaler e dos pesos
VALIDATION_RATIO = 0.15  # early stopping
CALIBRATION_RATIO = 0.15 # escolha do limiar
TEST_RATIO       = 0.20  # falso positivo saudável final
# Alias histórico para leitores antigos; já representa só a calibração.
CALIB_RATIO      = CALIBRATION_RATIO
PACIENCIA      = 20     # early stopping: épocas sem melhora
SIGMA          = 3.0    # fator k da REFERÊNCIA μ+kσ (comparativa); o limiar
                        # operacional é o percentil 99, não μ+kσ
THRESHOLD_METHOD = "p99"
SEED           = 42

# Colunas de metadado (não entram no modelo)
from src.ml.gpvs_principal import META_COLS


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

    def __init__(self, n_features: int, latente_dim: int = LATENTE_DIM,
                 dropout: float = DROPOUT):
        _exigir_torch()
        super().__init__()

        # Arquitetura compacta n→16→latente→16→n, adequada ao contrato de 24
        # features e aos dois ensaios saudáveis do GPVS.
        #
        # A saída do gargalo NÃO tem ReLU. Com ReLU o latente é não negativo por
        # construção, e unidades podem morrer em zero permanente — o problema
        # registrado em docs/auditoria_pipeline_ml.md §23, que ficou adiado
        # justamente até uma rodada de re-treino. Esta é a rodada.
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, latente_dim),
            # sem ativação: o latente pode assumir valores negativos
        )

        self.decoder = nn.Sequential(
            nn.Linear(latente_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, n_features),
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
    _exigir_torch()
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
    _exigir_torch()
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


def calcular_limiar(erros_calibracao: np.ndarray,
                    sigma: float = SIGMA) -> dict:
    """
    Define o limiar de anomalia do Autoencoder.

    DEFINIÇÃO OFICIAL (não confundir):
    - Limiar OPERACIONAL = percentil 99 nominal do erro de reconstrução
      saudável no bloco temporal de calibração, com interpolação linear do
      NumPy. Ele define o ponto de operação, mas não garante FPR empírico de
      1%, sobretudo quando a calibração tem poucas janelas.
    - Referência COMPARATIVA = μ + 3σ (assume normalidade; só para comparação
      teórica, NUNCA usado como limiar operacional).
    - Referência ADICIONAL = percentil 95.

    O campo `threshold_method` registra explicitamente o método em uso.
    """
    mu      = float(erros_calibracao.mean())
    sig     = float(erros_calibracao.std())
    p99     = float(np.percentile(erros_calibracao, 99))
    p95     = float(np.percentile(erros_calibracao, 95))
    mu_3sig = mu + sigma * sig

    return {
        "threshold_method"  : "p99",        # método operacional em uso
        "limiar"            : p99,          # operacional (chave de compat. retroativa)
        "limiar_operacional": p99,          # operacional explícito = percentil 99
        "score_method"      : "mse",
        "score_threshold"   : p99,
        "mu"                : mu,
        "sigma"             : sig,
        "k"                 : sigma,        # legado: multiplicador de sigma
        "sigma_multiplier"  : sigma,
        "limiar_p99"        : p99,          # operacional: percentil 99
        "mse_p99"           : p99,
        "limiar_p95"        : p95,          # referência adicional
        "limiar_mu3sigma"   : mu_3sig,      # referência teórica comparativa
        "limiar_mu3s"       : mu_3sig,      # alias de compat. retroativa
        "threshold_source"  : "bloco_calibracao_temporal",
        "top_k"             : None,
        "threshold_fallback_percentile": 99.0,
        "threshold_effective_percentile": 99.0,
        "threshold_quantile_method": "linear",
        "threshold_nominal_tail_probability_pct": 1.0,
    }


# ============================================================
# VISUALIZAÇÕES
# ============================================================
# Movidas para src/ml/graficos_autoencoder.py: aquele módulo não importa
# torch, então regenerar figuras a partir de artefatos salvos deixa de exigir
# a stack de ML (vale também para o Cloud em modo consulta). Reexportadas aqui
# para os chamadores existentes não mudarem.
from src.ml.graficos_autoencoder import (  # noqa: E402,F401
    _info_em_escala_mse,
    plotar_curvas,
    plotar_distribuicao,
    plotar_erro_temporal,
    regenerar_graficos_autoencoder,
    resumo_excedencia,
    salvar_resumo_calibracao,
)

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

    _exigir_torch()

    _log("=" * 60)
    _log("  AL IADO PV — AUTOENCODER PRINCIPAL (GPVS-Faults F0)")
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
        _log("   Execute primeiro a etapa features_gpvs do pipeline")
        return False

    df = pd.read_parquet(arquivo_feat)
    tempos = df["tempo_centro_s"].values if "tempo_centro_s" in df.columns else None

    colunas_feat = [c for c in df.columns if c not in META_COLS]
    X_bruto = df[colunas_feat].values.astype(np.float32)
    n_janelas, n_features = X_bruto.shape
    _log(f"   ✅ {n_janelas} janelas × {n_features} features")

    # ── 2. Normalização com RobustScaler ─────────────────────
    # RobustScaler usa mediana e IQR — resistente a outliers
    # (THD alto em transientes não distorce a escala geral)
    _log(f"\n⚖️  Normalizando com RobustScaler...")
    # F0L e F0M são divididos separadamente em quatro papéis temporais. Assim,
    # early stopping, escolha do limiar e estimativa final de FP não reutilizam
    # as mesmas janelas.
    from src.ml.gpvs_principal import (
        ajustar_normalizacao_f0,
        split_features_gpvs,
    )

    split = split_features_gpvs(df)
    X, normalizacao_baseline = ajustar_normalizacao_f0(df, split)
    idx_tr = split["treino"]
    idx_val = split["validacao"]
    idx_calib = split["calibracao"]
    idx_teste = split["teste"]
    X_treino_raw = X[idx_tr]
    X_val_raw = X[idx_val]
    X_calib_raw = X[idx_calib]
    X_teste_raw = X[idx_teste]
    scaler = RobustScaler()
    X_treino = scaler.fit_transform(X_treino_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_calib = scaler.transform(X_calib_raw).astype(np.float32)
    X_teste = scaler.transform(X_teste_raw).astype(np.float32)
    X_all    = scaler.transform(X).astype(np.float32)
    _log(f"   Treino : {len(X_treino)} janelas")
    _log(f"   Val.   : {len(X_val)} janelas (early stopping)")
    _log(f"   Calib. : {len(X_calib)} janelas (limiar)")
    _log(f"   Teste  : {len(X_teste)} janelas (isolado)")

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
    _log(f"   Encoder  : {n_features} → 16 → {latente_dim}")
    _log(f"   Latente  : {latente_dim} dimensões (gargalo sem ativação)")
    _log(f"   Decoder  : {latente_dim} → 16 → {n_features}")
    # A razão parâmetros/amostra é o número que motivou encolher a rede — vale
    # mais impresso a cada execução do que enterrado num comentário.
    _log(f"   Parâmetros: {n_params:,} "
         f"({n_params / max(len(X_treino), 1):.1f} por janela de treino)")

    # ── 5. Treinamento ───────────────────────────────────────
    _log(f"\n🏋️  Treinando ({epochs} épocas, early stopping={paciencia})...")
    hist_t, hist_v, ep_melhor = treinar(
        modelo, loader_treino, loader_val,
        epochs, lr, paciencia, device
    )

    # ── 6. Erros de reconstrução (MSE) e resíduos por feature ─
    _log(f"\n📐 Calculando erros de reconstrução...")
    erros_treino = calcular_erros(modelo, torch.from_numpy(X_treino), device)
    erros_val = calcular_erros(modelo, torch.from_numpy(X_val), device)
    erros_calib = calcular_erros(modelo, torch.from_numpy(X_calib), device)
    erros_teste = calcular_erros(modelo, torch.from_numpy(X_teste), device)
    erros_all    = calcular_erros(modelo, torch.from_numpy(X_all),    device)

    # Resíduo por feature → régua saudável (μ/σ) do escore LOCALIZADO. A régua
    # é uma transformação aprendida no TREINO; o bloco de calibração fica
    # reservado para escolher o limiar. Usar a calibração para ajustar a régua
    # e o limiar sobreajustava duas vezes o mesmo bloco: na rodada 50/20/30 isso
    # produziu FP=2,38% na calibração e 15% no teste. Com a régua no treino, o
    # mesmo modelo e split deram 1,67% no teste antes da nova execução completa.
    R_treino = ea.residuo_por_feature(modelo, X_treino, device)
    R_val = ea.residuo_por_feature(modelo, X_val, device)
    R_calib = ea.residuo_por_feature(modelo, X_calib, device)
    estat_residuo = ea.ajustar_estatistica_residuo(R_treino)
    sc_loc_treino = ea.escore_localizado(R_treino, estat_residuo)
    sc_loc_val = ea.escore_localizado(R_val, estat_residuo)
    sc_loc_calib = ea.escore_localizado(R_calib, estat_residuo)
    auto_percentil_elegivel = bool(
        ea.AUTO_PERCENTIL and ea.pode_autocalibrar_percentil(len(R_calib))
    )
    corte_cal = int(len(R_calib) * 0.8)
    n_auto_validacao = len(R_calib) - corte_cal
    if auto_percentil_elegivel:
        sc_fit = sc_loc_calib[:corte_cal]
        sc_val = sc_loc_calib[corte_cal:]
        limiar_loc, percentil_usado = ea.limiar_por_fp_alvo(sc_fit, sc_val, ea.FP_ALVO)
    else:
        percentil_usado = ea.PERCENTIL_LIMIAR
        limiar_loc = float(np.percentile(sc_loc_calib, percentil_usado))
    sc_loc_teste = ea.escore_localizado(
        ea.residuo_por_feature(modelo, X_teste, device), estat_residuo)
    sc_loc_all = ea.escore_localizado(
        ea.residuo_por_feature(modelo, X_all, device), estat_residuo)

    # ── 7. Limiar de anomalia (MSE e localizado; operacional = escolhido) ─
    info_mse = calcular_limiar(erros_calib, sigma)      # p99/p95/μ+3σ do MSE
    limiar_mse = float(info_mse["limiar"])

    metodo = ea.METODO_ESCORE            # 'mse' (padrão) ou 'localizado'
    k_loc  = ea.K_LOCALIZADO
    if metodo == "localizado":
        limiar_op = limiar_loc
        sc_op_treino = sc_loc_treino
        sc_op_calib, sc_op_teste, sc_op_all = sc_loc_calib, sc_loc_teste, sc_loc_all
    else:
        limiar_op = limiar_mse
        sc_op_treino = erros_treino
        sc_op_calib, sc_op_teste, sc_op_all = erros_calib, erros_teste, erros_all

    _log(f"\n🎯 Escore operacional: {ea.descricao_metodo(metodo, k_loc)}")
    _log(f"   Limiar MSE (p99)        = {limiar_mse:.6f}")
    _log(f"   Limiar localizado (p99) = {limiar_loc:.6f}")
    _log(f"   Limiar OPERACIONAL      = {limiar_op:.6f}  ← método '{metodo}'")

    # A calibração fixa o limiar; o teste fornece a estimativa final de FP.
    fp_calib = float((sc_op_calib > limiar_op).mean() * 100)
    fp_teste = float((sc_op_teste > limiar_op).mean() * 100)
    fp_all = float((sc_op_all > limiar_op).mean() * 100)
    fp_mse_calib = resumo_excedencia(erros_calib, limiar_mse)
    fp_mse_teste = resumo_excedencia(erros_teste, limiar_mse)
    fp_score_calib = resumo_excedencia(sc_op_calib, limiar_op)
    fp_score_teste = resumo_excedencia(sc_op_teste, limiar_op)
    _log(f"\n   Falsos positivos (calibração): {fp_calib:.1f}%")
    _log(f"   Falsos positivos (teste isolado): {fp_teste:.1f}%")
    _log(f"   Falsos positivos (all): {fp_all:.1f}%")

    # info_limiar carrega o OPERACIONAL em 'limiar' (o que injeção/validação/RUL
    # leem) + os dois limiares e o método, para auditoria e reversão.
    info_limiar = dict(info_mse)
    info_limiar["limiar"] = limiar_op
    info_limiar["limiar_operacional"] = limiar_op
    info_limiar["score_method"] = metodo
    info_limiar["score_threshold"] = limiar_op
    info_limiar["metodo_escore"] = metodo
    info_limiar["limiar_mse"] = limiar_mse
    info_limiar["mse_p99"] = limiar_mse
    info_limiar["limiar_localizado"] = limiar_loc
    info_limiar["k_localizado"] = k_loc
    info_limiar["top_k"] = k_loc if metodo == "localizado" else None
    info_limiar["percentil_limiar"] = percentil_usado
    info_limiar["threshold_fallback_percentile"] = ea.PERCENTIL_LIMIAR
    info_limiar["threshold_effective_percentile"] = percentil_usado
    info_limiar["score_standardization_source"] = "bloco_treino_modelo"
    info_limiar["percentil_auto"] = auto_percentil_elegivel
    info_limiar["threshold_auto_validation_n"] = n_auto_validacao
    info_limiar["threshold_auto_min_validation_n"] = ea.minimo_validacao_fp()
    info_limiar["threshold_auto_reason"] = (
        "meta de FP avaliada em subbloco com resolução empírica suficiente"
        if auto_percentil_elegivel
        else "fallback p99: subbloco não resolve empiricamente a meta de FP"
    )
    limiar = limiar_op

    # ── 8. Salva artefatos ───────────────────────────────────
    _log(f"\n💾 Salvando artefatos...")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    from src.ml.gpvs_principal import salvar_normalizacao_baseline

    arq_normalizacao = salvar_normalizacao_baseline(
        normalizacao_baseline, pasta_saida
    )
    _log(f"   ✅ {arq_normalizacao.name}")

    # Modelo PyTorch
    arq_modelo = pasta_saida / "modelo_autoencoder.pt"
    torch.save({
        "state_dict"  : modelo.state_dict(),
        "n_features"  : n_features,
        "latente_dim" : latente_dim,
        "colunas_feat": colunas_feat,
        "data_treino" : agora_local().isoformat(),
    }, arq_modelo)
    _log(f"   ✅ {arq_modelo.name}")

    # Scaler (+ sidecar SHA-256 para carga verificada nas etapas seguintes)
    arq_scaler = pasta_saida / "scaler.pkl"
    with open(arq_scaler, "wb") as f:
        pickle.dump(scaler, f)
    from src.core.seguranca import gravar_sidecar_sha256

    gravar_sidecar_sha256(arq_scaler)
    _log(f"   ✅ {arq_scaler.name}")

    # Régua por-feature (μ/σ do |resíduo| saudável) do escore localizado —
    # consumida por injeção/validação/RUL para computar o mesmo escore.
    arq_estat = ea.salvar_estatistica(estat_residuo, pasta_saida)
    _log(f"   ✅ {arq_estat.name}")

    # Limiar e metadados
    arq_limiar = pasta_saida / "limiar.json"
    metadados  = {
        **info_limiar,
        "n_janelas_treino"  : len(X_treino),
        "n_janelas_validacao": len(X_val),
        "n_janelas_calibracao": len(X_calib),
        "n_janelas_teste"   : len(X_teste),
        "n_features"        : n_features,
        "latente_dim"       : latente_dim,
        "epochs_treinadas"  : len(hist_t),
        "epoca_melhor"      : ep_melhor,
        "loss_val_melhor"   : float(min(hist_v)),
        "fp_val_pct"        : float(fp_calib),
        "fp_calib_pct"      : float(fp_calib),
        "fp_test_pct"       : float(fp_teste),
        "fp_mse_p99"        : {
            "calibracao": fp_mse_calib,
            "teste": fp_mse_teste,
        },
        "fp_score_operacional": {
            "calibracao": fp_score_calib,
            "teste": fp_score_teste,
        },
        "threshold_calibration_resolution_pct": float(100.0 / len(X_calib)),
        "window_overlap_fraction": 0.0,
        "window_step_for_no_shared_samples": 1,
        "baseline_normalization": {
            "method": "mediana_e_iqr_por_ensaio",
            "source_f0": "bloco_treino_de_cada_ensaio",
            "commissioning_fraction_pre_fault": normalizacao_baseline[
                "baseline_fraction"
            ],
            "minimum_windows": normalizacao_baseline["baseline_min_windows"],
            "iqr_floor_fraction_of_global_f0_train": normalizacao_baseline[
                "iqr_floor_fraction"
            ],
            "artifact": arq_normalizacao.name,
            "note": (
                "Pesos e limiar permanecem globais; em novos ensaios, somente "
                "o baseline inicial saudável define mediana e IQR locais."
            ),
        },
        "dataset": {
            "name": "GPVS-Faults",
            "doi": "10.17632/n76t439f65.1",
            "healthy_experiments": ["F0L", "F0M"],
            "fault_experiments_reserved": "F1L-F7M",
        },
        "split_temporal"    : {
            "protocolo": "GPVS_F0L_F0M_50_15_15_20_com_purga",
            "estrategia": split["estrategia"],
            "n_blocos": split["n_blocos"],
            "limites": split["limites"],
            "purge_janelas": split["purge_janelas"],
            "distancia_sem_compartilhamento": 1,
            "papeis_por_ensaio": {
                ensaio: {
                    papel: {
                        "inicio": int(indices[0]),
                        "fim_exclusivo": int(indices[-1]) + 1,
                        "n": len(indices),
                    }
                    for papel, indices in papeis.items()
                }
                for ensaio, papeis in split["por_ensaio"].items()
            },
            "nota": (
                "F0L e F0M são separados temporalmente em treino, validação, "
                "calibração e teste. F1-F7 não participam do scaler, do "
                "treinamento nem do limiar."
            ),
        },
        "data_treino"       : agora_local().isoformat(),
        "device"            : str(device),
    }
    with open(arq_limiar, "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    _log(f"   ✅ {arq_limiar.name}")

    arq_diag = pasta_saida / "diagnostico_autoencoder.npz"
    np.savez_compressed(
        arq_diag,
        historico_treino=np.asarray(hist_t, dtype=np.float32),
        historico_validacao=np.asarray(hist_v, dtype=np.float32),
        # Alias mantido para leitores de artefatos anteriores ao split em 4 papéis.
        historico_calibracao=np.asarray(hist_v, dtype=np.float32),
        erros_treino=erros_treino.astype(np.float32),
        erros_validacao=erros_val.astype(np.float32),
        erros_calibracao=erros_calib.astype(np.float32),
        erros_teste=erros_teste.astype(np.float32),
        erros_todos=erros_all.astype(np.float32),
        scores_operacionais_treino=sc_op_treino.astype(np.float32),
        scores_operacionais_validacao=(
            sc_loc_val if metodo == "localizado" else erros_val
        ).astype(np.float32),
        scores_operacionais_calibracao=sc_op_calib.astype(np.float32),
        scores_operacionais_teste=sc_op_teste.astype(np.float32),
        scores_operacionais_todos=sc_op_all.astype(np.float32),
        tempos=np.asarray(tempos if tempos is not None else [], dtype=np.float32),
        ensaios=df["ensaio"].astype(str).to_numpy()
        if "ensaio" in df.columns else np.asarray([], dtype=str),
        indices_teste=idx_teste.astype(np.int32),
        epoca_melhor=np.asarray([ep_melhor], dtype=np.int32),
    )
    _log(f"   ✅ {arq_diag.name}")

    # ── 9. Visualizações ─────────────────────────────────────
    # Estes gráficos DOCUMENTAM a distribuição do MSE — usam o limiar de MSE
    # (não o operacional), para o eixo e a linha de limiar ficarem na mesma
    # escala do que é plotado. A comparação MSE × localizado vive em
    # src/ml/diagnostico_escore.py.
    _log(f"\n📊 Gerando gráficos...")
    plotar_curvas(hist_t, hist_v, ep_melhor, pasta_saida)
    info_mse["fp_test_pct"] = float((erros_teste > limiar_mse).mean() * 100)
    info_mse["split_temporal"] = metadados["split_temporal"]
    plotar_distribuicao(
        erros_treino, erros_calib, erros_teste, info_mse, pasta_saida
    )
    if tempos is not None:
        plotar_erro_temporal(
            erros_all, tempos, info_mse, pasta_saida, indices_teste=idx_teste,
            grupos=df["ensaio"].astype(str).to_numpy()
            if "ensaio" in df.columns else None,
        )
    salvar_resumo_calibracao(
        erros_treino, erros_calib, erros_teste, metadados, pasta_saida,
        erros_validacao=erros_val,
        scores_treino=sc_op_treino,
        scores_validacao=(sc_loc_val if metodo == "localizado" else erros_val),
        scores_calibracao=sc_op_calib,
        scores_teste=sc_op_teste,
    )

    # ── 10. Resumo final ─────────────────────────────────────
    _log(f"\n{'='*60}")
    _log(f"  AUTOENCODER CONCLUÍDO!")
    _log(f"  Janelas treino  : {len(X_treino)}")
    _log(f"  Features        : {n_features}")
    _log(f"  Épocas treinadas: {len(hist_t)}")
    _log(f"  Loss val melhor : {min(hist_v):.6f}")
    _log(f"  Limiar anomalia : {limiar:.6f}")
    _log(f"  Falsos positivos: {fp_teste:.1f}% (teste isolado)")
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
