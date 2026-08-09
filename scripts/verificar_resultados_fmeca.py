# -*- coding: utf-8 -*-
"""Audita tabelas e artefatos do pipeline FMECA.

Uso:
    python scripts/verificar_resultados_fmeca.py

O verificador cruza os formatos publicados (JSON/CSV/PNG) e valida a
metodologia vigente: holdout temporal com purga, limiar operacional congelado,
SMD probabilística com Wilson, validação sintética E2 e Weibull com censura à
direita. Ausência de ajuste por poucos eventos é um resultado válido, não uma
falha do verificador; nesses casos, a RUL restrita por Kaplan-Meier deve
permanecer disponível até o horizonte observado.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

PASTA_AE = RAIZ / "resultados" / "autoencoder"
PASTA_EXP = RAIZ / "resultados" / "experimentos"
PASTA_CMP = RAIZ / "resultados" / "comparacao"
PASTA_MACRO = RAIZ / "resultados" / "macro"

ESPERADO = {
    "contator_ac": {"nome": "Contator AC", "s": 5, "o": 7, "d": 9, "npr": 315},
    "igbt": {"nome": "IGBT", "s": 5, "o": 6, "d": 3, "npr": 90},
    "fusivel_ac": {"nome": "Fusível AC", "s": 5, "o": 3, "d": 2, "npr": 30},
}
IDS_ANTIGOS = {"lcl", "desbalanceamento", "sensor"}
SEVERIDADES_VALIDACAO = {0.3, 0.5, 1.0}

PNGS_OBRIGATORIOS = (
    "curva_treino.png",
    "distribuicao_erro.png",
    "erro_temporal.png",
    "injecao_falhas_resultados.png",
    "injecao_falhas_comparacao.png",
    "validacao_roc.png",
    "validacao_pr.png",
    "validacao_matriz.png",
    "validacao_matrizes_severidades.png",
    "validacao_metricas.png",
    "weibull_ttf.png",
    "weibull_confiabilidade.png",
    "weibull_rul.png",
)


class Auditoria:
    def __init__(self) -> None:
        self.erros: list[str] = []
        self.avisos: list[str] = []

    def erro(self, mensagem: str) -> None:
        self.erros.append(mensagem)

    def aviso(self, mensagem: str) -> None:
        self.avisos.append(mensagem)

    def exigir(self, condicao: bool, mensagem: str) -> None:
        if not condicao:
            self.erro(mensagem)

    def json(self, caminho: Path) -> dict | None:
        if not caminho.is_file():
            self.erro(f"{caminho.name}: ausente")
            return None
        try:
            return json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.erro(f"{caminho.name}: JSON ilegível ({exc})")
            return None

    def csv(self, caminho: Path) -> list[dict[str, str]]:
        if not caminho.is_file():
            self.erro(f"{caminho.name}: ausente")
            return []
        try:
            with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
                return list(csv.DictReader(arquivo))
        except (OSError, csv.Error) as exc:
            self.erro(f"{caminho.name}: CSV ilegível ({exc})")
            return []


def _numero(valor, padrao: float = math.nan) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def _proximo(a: float, b: float, tolerancia: float = 1e-8) -> bool:
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=tolerancia, abs_tol=tolerancia
    )


def _texto_ponto_operacao(dados: dict) -> str:
    metodo = dados.get("score_method") or dados.get("metodo_escore")
    if not metodo:
        metodo = f"método {dados.get('threshold_method', 'desconhecido')}"
    percentil = _numero(
        dados.get("threshold_effective_percentile", dados.get("percentil_limiar"))
    )
    if math.isfinite(percentil):
        return f"{metodo} / percentil efetivo {percentil:.1f}"
    return str(metodo)


def _smd_calculada(valores: dict, alvo: float, *, conservadora: bool) -> float | None:
    for severidade in sorted(float(v) for v in valores):
        item = valores[str(severidade)] if str(severidade) in valores else valores[str(severidade).rstrip("0").rstrip(".")]
        taxa = item["low"] if conservadora else item
        if float(taxa) >= alvo:
            return severidade
    return None


def _validar_split_temporal(split: dict) -> tuple[bool, str]:
    """Valida os contratos contíguo e intercalado, ambos com intervalos [a, b)."""
    limites = split.get("limites") or {}
    purga = int(split.get("purge_janelas", 0))
    n_janelas = int(split.get("n_janelas", 0) or 0)
    estrategia = split.get("estrategia")

    if purga < 1:
        return False, "purga ausente"

    if estrategia == "blocos_intercalados":
        intervalos: list[tuple[int, int, str]] = []
        for nome in ("treino", "val", "teste"):
            partes = limites.get(nome)
            if not isinstance(partes, list) or not partes:
                return False, f"intervalos de {nome} ausentes"
            for parte in partes:
                if not isinstance(parte, (list, tuple)) or len(parte) != 2:
                    return False, f"intervalo inválido em {nome}"
                inicio, fim = parte
                if not isinstance(inicio, int) or not isinstance(fim, int):
                    return False, f"limites não inteiros em {nome}"
                if inicio < 0 or inicio >= fim:
                    return False, f"intervalo vazio ou invertido em {nome}"
                if n_janelas and fim > n_janelas:
                    return False, f"intervalo de {nome} excede n_janelas"
                intervalos.append((inicio, fim, nome))

        intervalos.sort()
        for anterior, atual in zip(intervalos, intervalos[1:]):
            ini_atual, _, nome_atual = atual
            _, fim_anterior, nome_anterior = anterior
            if ini_atual < fim_anterior:
                return False, "intervalos se sobrepõem"
            if nome_atual != nome_anterior and ini_atual - fim_anterior < purga:
                return False, "fronteira entre conjuntos viola a purga"
        return True, "ok"

    partes = []
    for nome in ("treino", "val", "teste"):
        parte = limites.get(nome)
        if not isinstance(parte, (list, tuple)) or len(parte) != 2:
            return False, f"limite contíguo de {nome} inválido"
        inicio, fim = parte
        if not isinstance(inicio, int) or not isinstance(fim, int):
            return False, f"limites não inteiros em {nome}"
        if inicio < 0 or inicio >= fim:
            return False, f"intervalo vazio ou invertido em {nome}"
        if n_janelas and fim > n_janelas:
            return False, f"intervalo de {nome} excede n_janelas"
        partes.append((inicio, fim, nome))

    for anterior, atual in zip(partes, partes[1:]):
        if atual[0] - anterior[1] < purga:
            return False, "blocos contíguos se sobrepõem ou violam a purga"
    return True, "ok"


def checar_limiar(aud: Auditoria) -> dict | None:
    dados = aud.json(PASTA_AE / "limiar.json")
    if not dados:
        return None

    aud.exigir(
        dados.get("threshold_method") == "p99",
        "limiar: threshold_method legado deve permanecer p99",
    )
    aud.exigir(
        dados.get("threshold_source") == "bloco_calibracao_temporal",
        "limiar: origem deve ser o bloco de calibração temporal",
    )
    score_method = dados.get("score_method") or dados.get("metodo_escore")
    aud.exigir(score_method in {"mse", "localizado"}, "limiar: score_method inválido")
    if score_method == "localizado":
        aud.exigir(
            dados.get("score_standardization_source") == "bloco_treino_modelo",
            "limiar: régua localizada deve ser ajustada no bloco de treino",
        )
    aud.exigir(
        _proximo(_numero(dados.get("mse_p99")), _numero(dados.get("limiar_p99"))),
        "limiar: mse_p99 difere do limiar_p99 legado",
    )
    aud.exigir(
        _proximo(_numero(dados.get("limiar")), _numero(dados.get("score_threshold"))),
        "limiar: valor operacional difere do score_threshold",
    )
    aud.exigir(
        _proximo(
            _numero(dados.get("limiar_operacional")),
            _numero(dados.get("score_threshold")),
        ),
        "limiar: limiar_operacional difere do score_threshold",
    )
    percentil_efetivo = _numero(
        dados.get("threshold_effective_percentile", dados.get("percentil_limiar"))
    )
    percentil_fallback = _numero(dados.get("threshold_fallback_percentile"), 99.0)
    if score_method == "localizado":
        aud.exigir(
            _proximo(
                _numero(dados.get("limiar")),
                _numero(dados.get("limiar_localizado")),
            ),
            "limiar: escore localizado difere de limiar_localizado",
        )
        aud.exigir(
            _proximo(_numero(dados.get("top_k")), _numero(dados.get("k_localizado"))),
            "limiar: top_k difere de k_localizado",
        )
        aud.exigir(
            _proximo(percentil_efetivo, _numero(dados.get("percentil_limiar"))),
            "limiar: percentil efetivo difere do percentil_limiar",
        )
        aud.exigir(
            percentil_efetivo >= percentil_fallback,
            "limiar: percentil efetivo abaixo do fallback declarado",
        )
    else:
        aud.exigir(
            _proximo(_numero(dados.get("limiar")), _numero(dados.get("limiar_p99"))),
            "limiar: score mse deve usar o p99 registrado",
        )
        aud.exigir(
            _proximo(percentil_efetivo, percentil_fallback),
            "limiar: percentil efetivo do score mse difere do fallback",
        )
    for campo in ("n_janelas_treino", "n_janelas_calibracao", "n_janelas_teste"):
        aud.exigir(int(dados.get(campo, 0)) > 0, f"limiar: {campo} deve ser positivo")

    split = dados.get("split_temporal") or {}
    split_valido, motivo_split = _validar_split_temporal(split)
    aud.exigir(split_valido, f"limiar: split temporal inválido ({motivo_split})")

    fp_score = dados.get("fp_score_operacional") or {}
    fp_mse = dados.get("fp_mse_p99") or {}
    for nome, bloco in (("score operacional", fp_score), ("MSE p99", fp_mse)):
        for parte in ("calibracao", "teste"):
            item = bloco.get(parte) or {}
            aud.exigir(int(item.get("n", 0)) > 0, f"limiar: FP {nome}/{parte} sem n")
            aud.exigir(
                0 <= _numero(item.get("rate_pct")) <= 100,
                f"limiar: FP {nome}/{parte} fora de [0,100]",
            )
            aud.exigir(
                0 <= _numero(item.get("ci95_low_pct")) <= _numero(item.get("ci95_high_pct")) <= 100,
                f"limiar: IC95 {nome}/{parte} inválido",
            )

    linhas_cal = aud.csv(PASTA_AE / "calibracao_autoencoder.csv")
    aud.exigir(len(linhas_cal) == 3, "calibracao_autoencoder.csv: deve ter treino/calibracao/teste")
    blocos_cal = {linha.get("bloco") for linha in linhas_cal}
    aud.exigir(
        blocos_cal == {"treino", "calibracao", "teste_isolado"},
        f"calibracao_autoencoder.csv: blocos {sorted(blocos_cal)}",
    )
    aud.exigir(
        (PASTA_AE / "calibracao_autoencoder.md").is_file(),
        "calibracao_autoencoder.md: ausente",
    )

    print(
        f"• autoencoder: treino={dados['n_janelas_treino']}, "
        f"calibração={dados['n_janelas_calibracao']}, "
        f"teste={dados['n_janelas_teste']}, "
        f"ponto={_texto_ponto_operacao(dados)}, "
        f"FP teste={_numero(dados.get('fp_test_pct')):.2f}%"
    )
    return dados


def checar_injecao(aud: Auditoria, limiar: dict | None) -> dict | None:
    dados = aud.json(PASTA_AE / "injecao_falhas_report.json")
    if not dados:
        return None

    aud.exigir(dados.get("evidence_level") == "E2", "injeção: evidence_level deve ser E2")
    aud.exigir(
        dados.get("threshold_source") == "bloco_calibracao_temporal",
        "injeção: origem do limiar incorreta",
    )
    if limiar:
        aud.exigir(
            _proximo(_numero(dados.get("limiar")), _numero(limiar.get("limiar"))),
            "injeção: limiar diverge de limiar.json",
        )
        aud.exigir(
            dados.get("score_method") == limiar.get("score_method"),
            "injeção: score_method diverge de limiar.json",
        )
        aud.exigir(
            _proximo(
                _numero(dados.get("score_threshold")),
                _numero(limiar.get("score_threshold")),
            ),
            "injeção: score_threshold diverge de limiar.json",
        )
        aud.exigir(
            _proximo(
                _numero(dados.get("threshold_effective_percentile")),
                _numero(limiar.get("threshold_effective_percentile")),
            ),
            "injeção: percentil efetivo diverge de limiar.json",
        )

    protocolo = dados.get("protocolo_avaliacao") or {}
    aud.exigir(protocolo.get("sem_sobreposicao") is True, "injeção: janelas devem ser não sobrepostas")
    aud.exigir(int(protocolo.get("n_janelas_usadas", 0)) >= 30, "injeção: amostra insuficiente")

    familias = dados.get("falhas") or {}
    aud.exigir(set(familias) == set(ESPERADO), f"injeção: ids {sorted(familias)}")
    aud.exigir(not (set(familias) & IDS_ANTIGOS), "injeção: taxonomia antiga presente")
    for falha_id, esperado in ESPERADO.items():
        falha = familias.get(falha_id) or {}
        for campo in ("s", "o", "d", "npr"):
            aud.exigir(
                falha.get(campo) == esperado[campo],
                f"injeção[{falha_id}].{campo}={falha.get(campo)!r}; esperado {esperado[campo]}",
            )
        aud.exigir(
            falha.get("npr") == falha.get("s", 0) * falha.get("o", 0) * falha.get("d", 0),
            f"injeção[{falha_id}]: NPR != S×O×D",
        )
        for campo in ("modo_falha", "efeito", "causa", "hipotese_fisica", "limitations"):
            aud.exigir(bool(str(falha.get(campo, "")).strip()), f"injeção[{falha_id}].{campo} vazio")

    alvo = _numero(dados.get("alvo_smd"), 0.95)
    probabilistica = dados.get("smd_probabilistica") or {}
    for falha_id in ESPERADO:
        bloco = probabilistica.get(falha_id) or {}
        taxas = bloco.get("taxa_deteccao") or {}
        intervalos = bloco.get("intervalo_wilson_95") or {}
        repeticoes = bloco.get("n_repeticoes") or {}
        aud.exigir(bool(taxas), f"injeção[{falha_id}]: taxas de detecção ausentes")
        aud.exigir(set(taxas) == set(intervalos) == set(repeticoes), f"injeção[{falha_id}]: severidades inconsistentes")
        for severidade, taxa in taxas.items():
            intervalo = intervalos[severidade]
            baixo, alto = _numero(intervalo.get("low")), _numero(intervalo.get("high"))
            taxa = _numero(taxa)
            aud.exigir(0 <= baixo <= taxa <= alto <= 1, f"injeção[{falha_id}/{severidade}]: IC de Wilson inválido")
            aud.exigir(int(repeticoes[severidade]) >= 30, f"injeção[{falha_id}/{severidade}]: n<30")

        smd95 = _smd_calculada(taxas, alvo, conservadora=False) if taxas else None
        smd95_cons = _smd_calculada(intervalos, alvo, conservadora=True) if intervalos else None
        aud.exigir(bloco.get("smd_95") == smd95, f"injeção[{falha_id}]: SMD95 inconsistente")
        aud.exigir(bloco.get("smd_95_conservadora") == smd95_cons, f"injeção[{falha_id}]: SMD95 conservadora inconsistente")

    linhas = aud.csv(PASTA_AE / "injecao_smd_tabela.csv")
    n_esperado = sum(
        len((probabilistica.get(fid) or {}).get("taxa_deteccao") or {})
        for fid in ESPERADO
    )
    aud.exigir(len(linhas) == n_esperado, f"injecao_smd_tabela.csv: {len(linhas)}/{n_esperado} linhas")
    for linha in linhas:
        fid, sev = linha.get("falha_id"), linha.get("severidade")
        taxa_json = ((probabilistica.get(fid) or {}).get("taxa_deteccao") or {}).get(str(float(sev)))
        if taxa_json is None:
            taxa_json = ((probabilistica.get(fid) or {}).get("taxa_deteccao") or {}).get(str(float(sev)).rstrip("0").rstrip("."))
        aud.exigir(_proximo(_numero(linha.get("taxa_deteccao")), _numero(taxa_json)), f"injeção CSV/JSON divergem em {fid}/sev={sev}")

    print(f"• injeção: FMECA, SMD95, Wilson e CSV cruzados ({_texto_ponto_operacao(dados)})")
    for fid in ESPERADO:
        bloco = probabilistica.get(fid) or {}
        print(
            f"    {ESPERADO[fid]['nome']:12s} SMD95="
            f"{bloco.get('smd_95') if bloco.get('smd_95') is not None else '—'} | "
            f"conservadora={bloco.get('smd_95_conservadora') if bloco.get('smd_95_conservadora') is not None else '—'}"
        )
    return dados


def checar_validacao(aud: Auditoria, limiar: dict | None) -> dict | None:
    dados = aud.json(PASTA_AE / "validacao_report.json")
    if not dados:
        return None
    meta = dados.get("__meta__") or {}
    aud.exigir(meta.get("evidence_level") == "E2", "validação: evidence_level deve ser E2")
    aud.exigir(meta.get("threshold_source") == "bloco_calibracao_temporal", "validação: origem do limiar incorreta")
    if limiar:
        aud.exigir(
            _proximo(_numero(meta.get("limiar_operacional")), _numero(limiar.get("limiar"))),
            "validação: limiar diverge de limiar.json",
        )
        aud.exigir(
            meta.get("score_method") == limiar.get("score_method"),
            "validação: score_method diverge de limiar.json",
        )
        aud.exigir(
            _proximo(
                _numero(meta.get("score_threshold")),
                _numero(limiar.get("score_threshold")),
            ),
            "validação: score_threshold diverge de limiar.json",
        )
        aud.exigir(
            _proximo(
                _numero(meta.get("threshold_effective_percentile")),
                _numero(limiar.get("threshold_effective_percentile")),
            ),
            "validação: percentil efetivo diverge de limiar.json",
        )

    casos = {k: v for k, v in dados.items() if k != "__meta__" and isinstance(v, dict)}
    esperados = {
        f"{fid}_sev{sev}" for fid in ESPERADO for sev in ("0.3", "0.5", "1.0")
    }
    aud.exigir(set(casos) == esperados, f"validação: casos divergentes ({sorted(set(casos) ^ esperados)})")

    for chave, caso in casos.items():
        for metrica in ("precision", "recall", "f1", "accuracy", "auc_roc", "auc_pr", "specificity", "fnr"):
            valor = _numero(caso.get(metrica))
            aud.exigir(0 <= valor <= 1, f"validação[{chave}].{metrica} fora de [0,1]")
        aud.exigir(_proximo(_numero(caso.get("fnr")), 1 - _numero(caso.get("recall"))), f"validação[{chave}]: FNR != 1-recall")
        for metrica in ("recall", "specificity", "auc_roc", "auc_pr"):
            valor = _numero(caso.get(metrica))
            baixo = _numero(caso.get(f"{metrica}_ci_low"))
            alto = _numero(caso.get(f"{metrica}_ci_high"))
            aud.exigir(0 <= baixo <= valor <= alto <= 1, f"validação[{chave}]: IC de {metrica} inválido")

        matriz = caso.get("confusion") or []
        aud.exigir(len(matriz) == 2 and all(len(linha) == 2 for linha in matriz), f"validação[{chave}]: matriz 2×2 ausente")
        if len(matriz) == 2 and all(len(linha) == 2 for linha in matriz):
            tn, fp = matriz[0]
            fn, tp = matriz[1]
            aud.exigir(tn + fp == caso.get("n_neg"), f"validação[{chave}]: n_neg inconsistente")
            aud.exigir(fn + tp == caso.get("n_pos"), f"validação[{chave}]: n_pos inconsistente")

    linhas = aud.csv(PASTA_AE / "validacao_tabela.csv")
    aud.exigir(len(linhas) == len(esperados), f"validacao_tabela.csv: {len(linhas)}/{len(esperados)} linhas")
    for linha in linhas:
        chave = f"{linha.get('falha_id')}_sev{float(linha.get('severidade'))}"
        caso = casos.get(chave) or {}
        for metrica in ("f1", "auc_roc", "recall", "specificity", "fnr"):
            aud.exigir(_proximo(_numero(linha.get(metrica)), _numero(caso.get(metrica))), f"validação CSV/JSON divergem em {chave}/{metrica}")
        if limiar:
            aud.exigir(
                linha.get("score_method") == limiar.get("score_method"),
                f"validação CSV/limiar divergem em {chave}/score_method",
            )
            aud.exigir(
                _proximo(
                    _numero(linha.get("score_threshold")),
                    _numero(limiar.get("score_threshold")),
                ),
                f"validação CSV/limiar divergem em {chave}/score_threshold",
            )
            aud.exigir(
                _proximo(
                    _numero(linha.get("threshold_effective_percentile")),
                    _numero(limiar.get("threshold_effective_percentile")),
                ),
                f"validação CSV/limiar divergem em {chave}/percentil efetivo",
            )

    print(f"• validação: 9 cenários, matrizes, ICs e CSV cruzados ({_texto_ponto_operacao(meta)})")
    for fid in ESPERADO:
        caso = casos.get(f"{fid}_sev1.0") or {}
        print(
            f"    {ESPERADO[fid]['nome']:12s} sev=1.0 | "
            f"AUC={_numero(caso.get('auc_roc')):.3f} | "
            f"recall={_numero(caso.get('recall')):.3f} | "
            f"FNR={_numero(caso.get('fnr')):.3f}"
        )
    return dados


def _ci_valido(parametro: float, intervalo) -> bool:
    return (
        isinstance(intervalo, list)
        and len(intervalo) == 2
        and 0 < _numero(intervalo[0]) <= parametro <= _numero(intervalo[1])
    )


def checar_weibull(aud: Auditoria) -> dict | None:
    dados = aud.json(PASTA_AE / "weibull_results.json")
    if not dados:
        return None
    meta = dados.get("__meta__") or {}
    parametros = dados.get("parametros_simulacao") or {}
    familias = dados.get("falhas") or {}
    aud.exigir(meta.get("evidence_level") == "E2", "Weibull: evidence_level deve ser E2")
    aud.exigir("sint" in str(meta.get("evidence_note", "")).lower(), "Weibull: ressalva sintética ausente")
    aud.exigir(set(familias) == set(ESPERADO), f"Weibull: ids {sorted(familias)}")
    min_eventos = int(parametros.get("min_eventos_weibull", 10))
    max_censura = _numero(parametros.get("max_censura_rul_pct"), 50.0)

    for fid, esperado in ESPERADO.items():
        falha = familias.get(fid) or {}
        ajuste = falha.get("weibull") or {}
        ttfs = falha.get("ttfs") or []
        eventos = falha.get("eventos_observados") or []
        n_traj = int(ajuste.get("n_traj", len(ttfs)))
        n_eventos = int(ajuste.get("n_eventos", sum(bool(v) for v in eventos)))
        n_cens = int(ajuste.get("n_censurados", n_traj - n_eventos))
        censura = _numero(ajuste.get("censura_pct"), 100 * n_cens / n_traj if n_traj else math.nan)
        aud.exigir(falha.get("npr") == esperado["npr"], f"Weibull[{fid}]: NPR incorreto")
        aud.exigir(len(ttfs) == len(eventos) == n_traj, f"Weibull[{fid}]: trajetórias/eventos inconsistentes")
        aud.exigir(sum(bool(v) for v in eventos) == n_eventos, f"Weibull[{fid}]: contagem de eventos incorreta")
        aud.exigir(n_traj - n_eventos == n_cens, f"Weibull[{fid}]: censura incorreta")
        aud.exigir(_proximo(censura, 100 * n_cens / n_traj), f"Weibull[{fid}]: percentual de censura incorreto")
        aud.exigir(bool(str(falha.get("ressalva_ajuste", "")).strip()), f"Weibull[{fid}]: ressalva ausente")
        horizonte = _numero(ajuste.get("rul_restrita_horizonte"))
        rul_restrita = _numero(ajuste.get("rul_restrita_inicial"))
        aud.exigir(
            ajuste.get("rul_restrita_disponivel") is True,
            f"Weibull[{fid}]: RUL restrita KM deve estar disponível",
        )
        aud.exigir(
            horizonte > 0 and 0 <= rul_restrita <= horizonte,
            f"Weibull[{fid}]: RUL restrita fora do horizonte observado",
        )

        if n_eventos < min_eventos:
            aud.exigir(
                falha.get("status_ajuste") == "nao_estimavel_parametrico_rul_restrita",
                f"Weibull[{fid}]: poucos eventos devem manter apenas RUL restrita",
            )
            aud.exigir(ajuste.get("beta") is None and ajuste.get("eta") is None, f"Weibull[{fid}]: parâmetros não devem existir com poucos eventos")
            aud.exigir(ajuste.get("rul_parametrica_disponivel") is False, f"Weibull[{fid}]: RUL paramétrica não pode ser reportada")
        else:
            for nome in ("beta", "eta", "mttf", "b10"):
                valor = _numero(ajuste.get(nome))
                aud.exigir(valor > 0, f"Weibull[{fid}].{nome} deve ser positivo")
                aud.exigir(_ci_valido(valor, ajuste.get(f"{nome}_ci95")), f"Weibull[{fid}]: IC95 de {nome} inválido")
            aud.exigir(ajuste.get("fit_converged") is True, f"Weibull[{fid}]: ajuste não convergiu")
            aud.exigir(ajuste.get("rul_parametrica_disponivel") is True, f"Weibull[{fid}]: RUL paramétrica deveria estar disponível")
            if censura > max_censura:
                aud.exigir(
                    ajuste.get("rul_parametrica_alta_incerteza") is True,
                    f"Weibull[{fid}]: alta censura deve sinalizar incerteza paramétrica",
                )
                aud.exigir(
                    falha.get("status_ajuste") == "exploratorio_alta_censura",
                    f"Weibull[{fid}]: status de alta censura ausente",
                )

    linhas = aud.csv(PASTA_AE / "weibull_tabela.csv")
    aud.exigir(len(linhas) == len(ESPERADO), f"weibull_tabela.csv: {len(linhas)}/{len(ESPERADO)} linhas")
    por_npr = {int(linha["npr"]): linha for linha in linhas if linha.get("npr")}
    for fid, esperado in ESPERADO.items():
        falha = familias.get(fid) or {}
        ajuste = falha.get("weibull") or {}
        linha = por_npr.get(esperado["npr"]) or {}
        aud.exigir(linha.get("status_ajuste") == falha.get("status_ajuste"), f"Weibull CSV/JSON divergem em {fid}/status")
        if ajuste.get("beta") is not None:
            aud.exigir(_proximo(_numero(linha.get("beta")), _numero(ajuste.get("beta"))), f"Weibull CSV/JSON divergem em {fid}/beta")

    print("• Weibull: eventos, censura, ajustes, ICs e CSV cruzados")
    for fid in ESPERADO:
        falha = familias.get(fid) or {}
        w = falha.get("weibull") or {}
        print(
            f"    {ESPERADO[fid]['nome']:12s} eventos={w.get('n_eventos')}/"
            f"{w.get('n_traj')} | censura={_numero(w.get('censura_pct')):.1f}% | "
            f"{falha.get('status_ajuste')}"
        )
    return dados


def checar_imagens(aud: Auditoria) -> None:
    try:
        from PIL import Image
    except ImportError:
        aud.aviso("Pillow indisponível; integridade visual dos PNGs não foi testada")
        return

    for nome in PNGS_OBRIGATORIOS:
        caminho = PASTA_AE / nome
        if not caminho.is_file():
            aud.erro(f"imagem ausente: {nome}")
            continue
        try:
            with Image.open(caminho) as imagem:
                imagem.verify()
            with Image.open(caminho) as imagem:
                largura, altura = imagem.size
            aud.exigir(largura >= 800 and altura >= 400, f"{nome}: resolução baixa ({largura}×{altura})")
            aud.exigir(caminho.stat().st_size >= 20_000, f"{nome}: arquivo suspeito ({caminho.stat().st_size} bytes)")
        except (OSError, ValueError) as exc:
            aud.erro(f"{nome}: PNG inválido ({exc})")
    print(f"• gráficos: {len(PNGS_OBRIGATORIOS)} PNGs verificados (integridade e resolução)")


def checar_experimentos(aud: Auditoria) -> None:
    presentes = 0
    for chave in ("ibrahim",):
        caminho = PASTA_EXP / chave / "resultado.json"
        if not caminho.is_file():
            aud.aviso(f"experimento {chave}: resultado ausente")
            continue
        dados = aud.json(caminho) or {}
        presentes += 1
        modelos = dados.get("modelos") or {}
        for nome, modelo in modelos.items():
            deteccao = (modelo or {}).get("deteccao_por_falha") or {}
            aud.exigir(
                not (set(deteccao) & IDS_ANTIGOS),
                f"experimento {chave}/{nome}: ids antigos presentes",
            )
        if chave == "ibrahim":
            aud.exigir(
                set(modelos).issubset({"AE-LSTM"}),
                "experimento Ibrahim: modelos auxiliares removidos ainda ativos",
            )
    if presentes:
        print(f"• experimentos: {presentes}/1 artefato comparativo presente")

    legado = PASTA_CMP / "comparacao_literatura.json"
    if legado.is_file():
        aud.aviso(
            "comparação E1 legada presente em resultados/comparacao; "
            "a fonte vigente é resultados/macro"
        )

    macro = PASTA_MACRO / "comparacao_resultado.json"
    tabela = PASTA_MACRO / "comparacao_tabela.md"
    if not macro.is_file():
        aud.aviso("macrocomparação vigente: artefato ausente")
        return
    dados_macro = aud.json(macro)
    if not isinstance(dados_macro, list):
        aud.erro("macrocomparação: comparacao_resultado.json deve ser lista")
        return
    aud.exigir(tabela.is_file(), "macrocomparação: comparacao_tabela.md ausente")

    nomes = {str(item.get("nome", "")) for item in dados_macro if isinstance(item, dict)}
    aud.exigir(any("Proposto" in nome for nome in nomes), "macrocomparação: método proposto ausente")
    aud.exigir(any("Ibrahim" in nome for nome in nomes), "macrocomparação: Ibrahim ausente")
    for item in dados_macro:
        if not isinstance(item, dict):
            aud.erro("macrocomparação: item inválido")
            continue
        nome = str(item.get("nome", "sem_nome"))
        familias = item.get("falhas") or {}
        ids = set(familias)
        aud.exigir(not (ids & IDS_ANTIGOS), f"macrocomparação[{nome}]: ids antigos presentes")
        aud.exigir(ids == set(ESPERADO), f"macrocomparação[{nome}]: famílias {sorted(ids)}")
        aud.exigir(0 <= _numero(item.get("fp_pct")) <= 100, f"macrocomparação[{nome}]: fp_pct inválido")
        aud.exigir(int(item.get("n_calib", 0)) > 0, f"macrocomparação[{nome}]: n_calib inválido")
        aud.exigir(int(item.get("n_aval", 0)) > 0, f"macrocomparação[{nome}]: n_aval inválido")
        for fid, esperado in ESPERADO.items():
            falha = familias.get(fid) or {}
            aud.exigir(falha.get("npr") == esperado["npr"], f"macrocomparação[{nome}/{fid}]: NPR incorreto")
            aud.exigir(0 <= _numero(falha.get("auc")) <= 1, f"macrocomparação[{nome}/{fid}]: AUC inválido")
            aud.exigir(0 <= _numero(falha.get("tpr_fpr10")) <= 1, f"macrocomparação[{nome}/{fid}]: TPR inválido")
            por_sev = falha.get("por_sev") or {}
            aud.exigir(bool(por_sev), f"macrocomparação[{nome}/{fid}]: severidades ausentes")
            for sev, metricas in por_sev.items():
                taxa = _numero((metricas or {}).get("taxa"))
                aud.exigir(0 <= taxa <= 1, f"macrocomparação[{nome}/{fid}/sev={sev}]: taxa inválida")
    try:
        from src.ml.macro_comparar import estado_proveniencia

        motivos = estado_proveniencia()
        aud.exigir(
            not motivos,
            "macrocomparação: manifesto stale (" + "; ".join(motivos) + ")",
        )
    except Exception as exc:
        aud.erro(f"macrocomparação: falha ao verificar proveniência ({exc})")
    print(f"• macrocomparação: {len(dados_macro)} métodos, famílias FMECA e tabela publicados")


def main() -> int:
    print("=" * 72)
    print(" AUDITORIA DOS RESULTADOS — FMECA consolidada e evidência E2")
    print("=" * 72)
    auditoria = Auditoria()
    limiar = checar_limiar(auditoria)
    checar_injecao(auditoria, limiar)
    checar_validacao(auditoria, limiar)
    checar_weibull(auditoria)
    checar_imagens(auditoria)
    checar_experimentos(auditoria)

    print("-" * 72)
    for aviso in auditoria.avisos:
        print(f"  AVISO: {aviso}")
    if auditoria.erros:
        print("\nREPROVADO — inconsistências encontradas:")
        for erro in auditoria.erros:
            print(f"  - {erro}")
        return 1
    complemento = (
        "Avisos representam artefatos comparativos opcionais ainda pendentes."
        if auditoria.avisos
        else "Nenhuma pendência de artefato foi encontrada."
    )
    print(
        "\nAPROVADO — JSON, CSV, gráficos e critérios metodológicos estão "
        f"consistentes.\n{complemento}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
