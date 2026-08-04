"""
escore_anomalia.py — Al IAdo PV / fonte ÚNICA do escore de anomalia do pipeline CA.

Dois escores, ambos calibrados para ~1% de falso positivo no bloco saudável:

- 'mse'        : média do erro de reconstrução sobre TODAS as ~109 features
                 (escore histórico). Dilui falhas localizadas no espectro.
- 'localizado' : média dos top-k |resíduos| PADRONIZADOS por feature — z do
                 |resíduo| contra a régua saudável, agregado pelos k maiores.
                 Sensível a falha concentrada em poucas features (harmônicos do
                 IGBT, perda de fase do Fusível).

Fundamentação (docs/auditoria_pipeline_ml.md §13): erro de reconstrução como
sinal de anomalia (Ibrahim, 2022, eq. 3); padronização por-feature do resíduo
(Francisti, 2025 — Z-score/Shewhart, aqui sobre o resíduo do AE); agregação
top-k como generalização robusta da regra de Shewhart / SPC multivariável e da
contribuição por feature (Narayanan, 2023).

Método OPERACIONAL padrão: 'localizado'. Para reproduzir EXATAMENTE o pipeline
antigo, defina a variável de ambiente:
    AL_IADO_ESCORE_ANOMALIA=mse
O k do top-k é configurável por AL_IADO_ESCORE_K (padrão 5; justificar por
varredura, ver §13). Este módulo é uma FOLHA: depende só de numpy/torch — nunca
de autoencoder/injecao/validacao/rul (evita ciclo de import).

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ── Configuração operacional (env, com padrões seguros) ─────────────────────
METODO_ESCORE = os.getenv("AL_IADO_ESCORE_ANOMALIA", "localizado").strip().lower()
K_LOCALIZADO = int(os.getenv("AL_IADO_ESCORE_K", "5"))
# Limiar operacional por taxa máxima de falso positivo EMPÍRICA no bloco de
# calibração. O corte usa uma ordem estatística: permite no máximo
# floor(n_calib * FP_ALVO / 100) excedências. Com 91 janelas e alvo de 1%, isso
# significa zero excedências, sem fingir que a amostra consegue resolver 1%
# (sua menor taxa não nula é 1/91 = 1,10%).
#
# Definir AL_IADO_ESCORE_PERCENTIL fixa um percentil manual (desliga a política
# automática) para reprodução de rodadas antigas.
_PERCENTIL_ENV = os.getenv("AL_IADO_ESCORE_PERCENTIL")
PERCENTIL_LIMIAR = float(_PERCENTIL_ENV) if _PERCENTIL_ENV else 99.0
AUTO_PERCENTIL = _PERCENTIL_ENV is None
FP_ALVO = float(os.getenv("AL_IADO_ESCORE_FP_ALVO", "1.0"))  # % de FP alvo (auto)
POLITICA_LIMIAR = "fpr_empirico_maximo"
# Semente do bootstrap usado para MEDIR a incerteza do limiar (não para
# alterá-lo — ver incerteza_do_limiar).
SEED_BOOTSTRAP = int(os.getenv("AL_IADO_ESCORE_BOOTSTRAP_SEED", "42"))


def incerteza_do_limiar(amostra, p: float, n_boot: int = 500,
                        seed: int = SEED_BOOTSTRAP,
                        confianca: float = 95.0) -> dict:
    """Incerteza AMOSTRAL do limiar, por bootstrap — não é uma correção dele.

    Devolve ``{limiar, ic_low, ic_high, n_boot, largura_relativa}``. Serve para
    responder "quanto do FP medido é ruído de estimativa?", que é pergunta
    legítima com 73 janelas de calibração e um quantil de cauda.

    IMPORTANTE, para não se repetir o erro que motivou esta função: bootstrap
    **não reduz** a variância do limiar entre amostras diferentes. Medido neste
    projeto, a mediana das réplicas melhora a dispersão em ~2% — nada. É
    esperado: o bootstrap estima a distribuição amostral a partir de UMA
    amostra; não acrescenta informação que a amostra não tem.

    Também foram testados e REJEITADOS como substitutos do percentil empírico:

    - **ajuste paramétrico** (lognormal): ótimo quando a distribuição está
      certa (erro −37%), **catastrófico quando errada** (erro até 5× maior em
      gama, Weibull e mistura bimodal). Como a distribuição real do escore
      localizado não é conhecida nem verificável, o risco é inaceitável;
    - **EVT / Pareto generalizada na cauda**: pior que o empírico neste regime
      — com 73 pontos sobram ~18 excedências acima do 75º percentil, poucas
      para ajustar a GPD de forma estável.

    O percentil empírico permanece a escolha certa: é o estimador robusto, e o
    limite é o tamanho da amostra, não o estimador.
    """
    a = np.asarray(amostra, dtype=float)
    pontual = float(np.percentile(a, p))
    if n_boot <= 0 or len(a) < 8:
        return {"limiar": pontual, "ic_low": float("nan"),
                "ic_high": float("nan"), "n_boot": 0,
                "largura_relativa": float("nan")}
    rng = np.random.default_rng(seed)
    replicas = np.array([
        np.percentile(rng.choice(a, size=len(a), replace=True), p)
        for _ in range(int(n_boot))
    ], dtype=float)
    margem = (100.0 - confianca) / 2.0
    lo = float(np.percentile(replicas, margem))
    hi = float(np.percentile(replicas, 100.0 - margem))
    return {
        "limiar": pontual, "ic_low": lo, "ic_high": hi, "n_boot": int(n_boot),
        # Quanto o limiar "balança" em relação ao próprio valor. Acima de ~0,3
        # o número não sustenta comparação fina de FP entre configurações.
        "largura_relativa": float((hi - lo) / pontual) if pontual else float("nan"),
    }


def calibrar_limiar_fpr(scores_calibracao,
                        fp_alvo_pct: float | None = None) -> dict:
    """Fixa um limiar que respeita o FPR empírico máximo na calibração.

    A decisão de anomalia do pipeline é ``score > limiar``. Para ``n`` escores
    saudáveis, permitimos no máximo ``floor(n * alvo / 100)`` valores acima do
    corte e escolhemos a ordem estatística correspondente. O teste final não é
    recebido por esta função e, portanto, não pode contaminar o limiar.

    O retorno distingue duas ideias que não podem ser confundidas:

    - ``fpr_observado_pct``: restrição empírica, garantida no bloco usado;
    - ``resolucao_amostral_pct``: menor FPR não nulo que a amostra consegue
      medir. Se ela for maior que o alvo, o resultado não certifica 1% fora da
      amostra, mesmo quando a calibração observada tem zero excedências.
    """
    alvo = FP_ALVO if fp_alvo_pct is None else float(fp_alvo_pct)
    if not 0.0 <= alvo < 100.0:
        raise ValueError("fp_alvo_pct deve estar em [0, 100)")

    scores = np.asarray(scores_calibracao, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("scores_calibracao não pode ser vazio")
    if not np.isfinite(scores).all():
        raise ValueError("scores_calibracao deve conter apenas valores finitos")

    n = int(scores.size)
    max_excedencias = int(np.floor(n * alvo / 100.0 + 1e-12))
    max_excedencias = min(max_excedencias, n - 1)
    ordem = np.sort(scores)
    indice_corte = n - max_excedencias - 1
    limiar = float(ordem[indice_corte])
    excedencias = int(np.count_nonzero(scores > limiar))
    fpr_observado = float(excedencias / n * 100.0)
    resolucao = float(100.0 / n)

    return {
        "limiar": limiar,
        "politica": POLITICA_LIMIAR,
        "fp_alvo_pct": alvo,
        "n_calibracao": n,
        "max_excedencias": max_excedencias,
        "excedencias_observadas": excedencias,
        "fpr_observado_pct": fpr_observado,
        "percentil_efetivo": float((indice_corte + 1) / n * 100.0),
        "resolucao_amostral_pct": resolucao,
        "alvo_resolvivel_na_amostra": bool(resolucao <= alvo),
        "restricao_satisfeita": bool(fpr_observado <= alvo + 1e-12),
    }


def limiar_por_fp_alvo(scores_ajuste, scores_val, fp_alvo_pct: float | None = None,
                       percentis=(99.0, 99.3, 99.5, 99.7, 99.9)) -> tuple[float, float]:
    """Auto-calibra o limiar visando o FP alvo, SEM ajuste manual.

    Retorna (limiar, percentil): o MENOR percentil de ``scores_ajuste`` cujo
    falso positivo em ``scores_val`` (bloco saudável NÃO visto) fica <=
    ``fp_alvo_pct``. Se nenhum candidato atinge o alvo, sobe o corte até a ordem
    estatística conservadora de ``scores_val``; não retorna silenciosamente um
    limiar que viola a própria meta.

    ``fp_alvo_pct=None`` usa ``FP_ALVO`` (env), em vez de um literal fixado no
    chamador — era o que `macro_comum` fazia, ignorando a configuração.

    O estimador é o percentil EMPÍRICO, deliberadamente. Ver
    `incerteza_do_limiar` para o registro de por que bootstrap e ajuste
    paramétrico foram testados e rejeitados como substitutos.
    """
    alvo = FP_ALVO if fp_alvo_pct is None else float(fp_alvo_pct)
    a = np.asarray(scores_ajuste, dtype=float)
    v = np.asarray(scores_val, dtype=float)
    escolhido = float(percentis[-1])
    for p in percentis:
        lim = float(np.percentile(a, p))
        if float((v > lim).mean() * 100.0) <= alvo:
            escolhido = float(p)
            return lim, escolhido

    guarda = calibrar_limiar_fpr(v, alvo)
    limiar = max(float(np.percentile(a, escolhido)), float(guarda["limiar"]))
    return limiar, 100.0

# Nome canônico do artefato com a régua por-feature (μ/σ do |resíduo| saudável).
ARQUIVO_ESTATISTICA = "estatistica_residuo.npz"


# ============================================================
# NÚCLEO — funções puras (testáveis sem dataset)
# ============================================================

def ajustar_estatistica_residuo(residuos_saudaveis: np.ndarray) -> dict:
    """μ/σ por-feature do |resíduo| no bloco saudável (a régua de padronização).

    residuos_saudaveis: (n_janelas, n_features) do resíduo (x − x_rec) em
    espaço normalizado. Retorna {"mu": (F,), "sigma": (F,)}.
    """
    abs_r = np.abs(np.asarray(residuos_saudaveis, dtype=float))
    mu = abs_r.mean(axis=0)
    sigma = abs_r.std(axis=0) + 1e-9
    return {"mu": mu, "sigma": sigma}


def escore_mse_medio(residuos: np.ndarray) -> np.ndarray:
    """Escore histórico: MSE médio sobre todas as features. (n,)→(n,)."""
    r = np.atleast_2d(np.asarray(residuos, dtype=float))
    return (r ** 2).mean(axis=1)


def escore_localizado(residuos: np.ndarray, stats: dict,
                      k: int = K_LOCALIZADO) -> np.ndarray:
    """Escore sensível a falha LOCALIZADA: média dos top-k |resíduos| z-padronizados.

    Não dilui um desvio concentrado em poucas features. residuos: (n, F).
    """
    r = np.abs(np.atleast_2d(np.asarray(residuos, dtype=float)))
    z = (r - stats["mu"]) / stats["sigma"]
    k = int(max(1, min(k, z.shape[1])))
    topk = np.partition(z, -k, axis=1)[:, -k:]
    return topk.mean(axis=1)


def pontuar(residuos: np.ndarray, stats: dict | None = None,
            metodo: str = METODO_ESCORE, k: int = K_LOCALIZADO) -> np.ndarray:
    """Escore operacional segundo `metodo`. Cai para 'mse' se faltar a régua.

    Fallback seguro: sem `stats` (régua ausente), retorna o MSE médio — o
    comportamento histórico —, nunca quebra por artefato faltando.
    """
    if metodo == "mse" or stats is None:
        return escore_mse_medio(residuos)
    return escore_localizado(residuos, stats, k=k)


# ============================================================
# INFERÊNCIA — resíduo por feature a partir do modelo
# ============================================================

def residuo_por_feature(modelo, X: np.ndarray, device) -> np.ndarray:
    """Resíduo (x − x_rec) por feature, em espaço normalizado. X: (n, F)→(n, F)."""
    import torch

    modelo.eval()
    with torch.no_grad():
        t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        rec = modelo(t)
        return (t - rec).cpu().numpy()


def residuo_de_vetor(modelo, vetor_norm: np.ndarray, device) -> np.ndarray:
    """Resíduo por feature de UMA janela já normalizada. (1,F)/(F,)→(F,)."""
    v = np.asarray(vetor_norm, dtype=np.float32).reshape(1, -1)
    return residuo_por_feature(modelo, v, device).ravel()


# ============================================================
# PERSISTÊNCIA da régua por-feature (μ/σ)
# ============================================================

def salvar_estatistica(stats: dict, pasta: Path,
                       nome: str = ARQUIVO_ESTATISTICA) -> Path:
    caminho = Path(pasta) / nome
    np.savez(caminho, mu=np.asarray(stats["mu"], dtype=np.float64),
             sigma=np.asarray(stats["sigma"], dtype=np.float64))
    return caminho


def carregar_estatistica(pasta: Path,
                         nome: str = ARQUIVO_ESTATISTICA) -> dict | None:
    """Carrega a régua por-feature; None se ausente (→ fallback para MSE)."""
    caminho = Path(pasta) / nome
    if not caminho.exists():
        return None
    with np.load(caminho) as d:
        return {"mu": d["mu"], "sigma": d["sigma"]}


def descricao_metodo(metodo: str = METODO_ESCORE, k: int = K_LOCALIZADO) -> str:
    if metodo == "mse":
        return "MSE médio sobre todas as features (histórico)"
    return f"localizado: média dos top-{k} |resíduos| padronizados por feature"
