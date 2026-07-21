# -*- coding: utf-8 -*-
"""Audita tabelas e artefatos do pipeline FMECA.

Uso:
    python scripts/verificar_resultados_fmeca.py

O verificador cruza os formatos publicados (JSON/CSV/PNG) e valida a
metodologia vigente: holdout temporal com purga, limiar p99 de calibração,
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
PASTA_AE = RAIZ / "resultados" / "autoencoder"
PASTA_EXP = RAIZ / "resultados" / "experimentos"
PASTA_CMP = RAIZ / "resultados" / "comparacao"

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


def _smd_calculada(valores: dict, alvo: float, *, conservadora: bool) -> float | None:
    for severidade in sorted(float(v) for v in valores):
        item = valores[str(severidade)] if str(severidade) in valores else valores[str(severidade).rstrip("0").rstrip(".")]
        taxa = item["low"] if conservadora else item
        if float(taxa) >= alvo:
            return severidade
    return None


def checar_limiar(aud: Auditoria) -> dict | None:
    dados = aud.json(PASTA_AE / "limiar.json")
    if not dados:
        return None

    aud.exigir(dados.get("threshold_method") == "p99", "limiar: método deve ser p99")
    aud.exigir(
        dados.get("threshold_source") == "bloco_calibracao_temporal",
        "limiar: origem deve ser o bloco de calibração temporal",
    )
    aud.exigir(
        _proximo(_numero(dados.get("limiar")), _numero(dados.get("limiar_p99"))),
        "limiar: valor operacional difere do p99 registrado",
    )
    for campo in ("n_janelas_treino", "n_janelas_calibracao", "n_janelas_teste"):
        aud.exigir(int(dados.get(campo, 0)) > 0, f"limiar: {campo} deve ser positivo")

    split = dados.get("split_temporal") or {}
    limites = split.get("limites") or {}
    treino = limites.get("treino") or []
    calibracao = limites.get("val") or []
    teste = limites.get("teste") or []
    aud.exigir(
        all(len(parte) == 2 for parte in (treino, calibracao, teste)),
        "limiar: limites do split temporal incompletos",
    )
    if all(len(parte) == 2 for parte in (treino, calibracao, teste)):
        aud.exigir(
            treino[1] < calibracao[0] < calibracao[1] < teste[0] < teste[1],
            "limiar: blocos temporais se sobrepõem ou estão fora de ordem",
        )
        aud.exigir(int(split.get("purge_janelas", 0)) >= 1, "limiar: purga ausente")

    print(
        f"• autoencoder: treino={dados['n_janelas_treino']}, "
        f"calibração={dados['n_janelas_calibracao']}, "
        f"teste={dados['n_janelas_teste']}, "
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

    print("• injeção: FMECA, SMD95, Wilson e CSV cruzados")
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

    print("• validação: 9 cenários, matrizes, ICs e CSV cruzados")
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
    for chave in ("francisti", "ibrahim"):
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
                not any("prophet" in nome.lower() for nome in modelos),
                "experimento Ibrahim: Prophet ainda ativo",
            )
    if presentes:
        print(f"• experimentos: {presentes}/2 artefatos comparativos presentes")

    comparacao = PASTA_CMP / "comparacao_literatura.json"
    if not comparacao.is_file():
        aud.aviso("comparação com a literatura: artefato ausente")
    else:
        dados = aud.json(comparacao) or {}
        familias = set((dados.get("auc_por_falha_metodo") or {}).keys())
        aud.exigir(not (familias & IDS_ANTIGOS), "comparação: ids antigos presentes")
        aud.exigir(familias == set(ESPERADO), f"comparação: famílias {sorted(familias)}")
        print("• comparação com a literatura: artefato presente")


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
