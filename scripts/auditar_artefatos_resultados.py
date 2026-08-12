"""Inventaria e verifica os artefatos acadêmicos versionados do projeto."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]
PASTA_MANIFESTOS = RAIZ / "resultados" / "manifestos"
PASTA_AUDITORIA = RAIZ / "resultados" / "auditoria"
ARQUIVO_CSV = PASTA_AUDITORIA / "inventario_artefatos.csv"
ARQUIVO_MD = PASTA_AUDITORIA / "relatorio_auditoria_artefatos.md"

ETAPAS_CANONICAS = {
    "features_gpvs", "autoencoder", "injecao_falhas", "validacao",
    "rul_weibull", "validacao_gpvs_e3",
}
ETAPAS_LEGADAS = {"features_ca", "macro_comparacao"}
SUFIXOS_TEXTO_PORTAVEL = {
    ".csv", ".json", ".md", ".toml", ".txt", ".yaml", ".yml",
}


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def sha256_manifesto(caminho: Path) -> str:
    """Replica o hash v2: LF/UTF-8 para texto e bytes para binários."""
    if caminho.suffix.lower() not in SUFIXOS_TEXTO_PORTAVEL:
        return sha256_arquivo(caminho)
    digest = hashlib.sha256()
    with caminho.open("r", encoding="utf-8", newline=None) as arquivo:
        for bloco in iter(lambda: arquivo.read(64 * 1024), ""):
            digest.update(bloco.encode("utf-8"))
    return digest.hexdigest()


def _finito(valor) -> bool:
    if isinstance(valor, dict):
        return all(_finito(item) for item in valor.values())
    if isinstance(valor, list):
        return all(_finito(item) for item in valor)
    if isinstance(valor, float):
        return math.isfinite(valor)
    return True


def qualidade_estrutural(caminho: Path) -> str:
    try:
        if caminho.suffix.lower() == ".json":
            conteudo = json.loads(caminho.read_text(encoding="utf-8"))
            return "ok" if _finito(conteudo) else "nao_finito"
        if caminho.suffix.lower() == ".csv":
            with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
                leitor = csv.reader(arquivo)
                cabecalho = next(leitor, None)
                if not cabecalho or len(set(cabecalho)) != len(cabecalho):
                    return "cabecalho_invalido"
                for linha in leitor:
                    if len(linha) != len(cabecalho):
                        return "linha_irregular"
            return "ok"
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return "invalido"
    return "nao_aplicavel"


def _arquivos_rastreados() -> list[str]:
    resultado = subprocess.run(
        ["git", "ls-files", "-z", "--", "resultados", "dados/processados"],
        cwd=RAIZ, check=True, capture_output=True,
    )
    return sorted(
        item.decode("utf-8")
        for item in resultado.stdout.split(b"\0") if item
        if not item.decode("utf-8").startswith("resultados/auditoria/")
    )


def _manifestos() -> tuple[dict[str, dict], dict[str, list[str]]]:
    manifestos: dict[str, dict] = {}
    donos: dict[str, list[str]] = defaultdict(list)
    for caminho in sorted(PASTA_MANIFESTOS.glob("*.json")):
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        etapa = str(dados.get("stage") or caminho.stem)
        manifestos[etapa] = dados
        for artefato in dados.get("outputs", []):
            donos[str(artefato).replace("\\", "/")].append(etapa)
    return manifestos, donos


def construir_inventario() -> list[dict]:
    manifestos, donos = _manifestos()
    inventario = []
    for relativo in _arquivos_rastreados():
        caminho = RAIZ / relativo
        etapas = donos.get(relativo, [])
        hashes_esperados = {
            etapa: manifestos[etapa].get("output_artifacts", {}).get(relativo)
            for etapa in etapas
        }
        hash_atual = sha256_arquivo(caminho)
        hash_modo_manifesto = sha256_manifesto(caminho)
        comparacoes = [
            esperado == hash_modo_manifesto
            for esperado in hashes_esperados.values()
            if esperado
        ]
        if comparacoes:
            hash_manifesto = "ok" if all(comparacoes) else "divergente"
        else:
            hash_manifesto = "sem_hash_manifesto"

        if any(etapa in ETAPAS_CANONICAS for etapa in etapas):
            papel = "canonico"
        elif any(etapa in ETAPAS_LEGADAS for etapa in etapas):
            papel = "legado_comparativo"
        elif relativo.startswith("resultados/manifestos/"):
            papel = "manifesto"
        elif relativo.startswith("resultados/auditoria/"):
            papel = "auditoria"
        else:
            papel = "suplementar"

        inventario.append({
            "arquivo": relativo,
            "papel": papel,
            "etapas": ";".join(etapas),
            "bytes": caminho.stat().st_size,
            "sha256": hash_atual,
            "hash_manifesto": hash_manifesto,
            "qualidade_estrutural": qualidade_estrutural(caminho),
        })
    return inventario


def _resumo_weibull() -> list[dict]:
    caminho = RAIZ / "resultados" / "autoencoder" / "weibull_results.json"
    if not caminho.exists():
        return []
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    linhas = []
    for falha in dados.get("falhas", {}).values():
        ajuste = falha.get("weibull", {})
        diagnostico = ajuste.get("diagnostico_papel_weibull") or {}
        eventos_observados = falha.get("eventos_observados", [])
        a_dets = falha.get("a_dets", [])
        niveis_derivados = len({
            float(a) for a, evento in zip(a_dets, eventos_observados) if evento
        })
        n_eventos = ajuste.get("n_eventos")
        niveis = ajuste.get("n_niveis_distintos", niveis_derivados)
        empates = ajuste.get("taxa_empates")
        if empates is None and n_eventos:
            empates = 1.0 - niveis / n_eventos
        linhas.append({
            "falha": falha.get("nome"),
            "eventos": n_eventos,
            "trajetorias": ajuste.get("n_traj"),
            "niveis": niveis,
            "empates": empates,
            "beta": ajuste.get("beta"),
            "eta": ajuste.get("eta"),
            "r2": diagnostico.get("r2"),
            "recomendado": ajuste.get("resumo_parametrico_recomendado"),
        })
    return linhas


def escrever_relatorios(inventario: list[dict]) -> None:
    PASTA_AUDITORIA.mkdir(parents=True, exist_ok=True)
    campos = list(inventario[0]) if inventario else []
    with ARQUIVO_CSV.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(inventario)

    divergentes = [item for item in inventario if item["hash_manifesto"] == "divergente"]
    invalidos = [
        item for item in inventario
        if item["qualidade_estrutural"] not in {"ok", "nao_aplicavel"}
    ]
    por_papel: dict[str, int] = defaultdict(int)
    for item in inventario:
        por_papel[item["papel"]] += 1

    linhas = [
        "# Auditoria dos artefatos acadêmicos\n\n",
        "## Escopo e fonte dos dados\n\n",
        "O pipeline canônico usa **GPVS-Faults** (DOI `10.17632/n76t439f65.1`). "
        "F0L/F0M formam o domínio saudável; F1L-F7M são os 14 ensaios de falha "
        "da validação externa E3. A injeção orientada pela FMECA e a análise "
        "Weibull são validações sintéticas E2 construídas sobre janelas saudáveis "
        "GPVS; não mesclam outro dataset no treinamento principal.\n\n",
        "## Cobertura versionada\n\n",
        f"Foram inventariados **{len(inventario)} artefatos científicos "
        "rastreados** em `resultados/` e `dados/processados/`, excluindo os "
        "dois arquivos produzidos por esta própria auditoria. ",
        ", ".join(f"{papel}: {quantidade}" for papel, quantidade in sorted(por_papel.items())),
        ". O CSV anexo registra tamanho, SHA-256, etapa proprietária, papel e "
        "validação estrutural de cada saída.\n\n",
        f"Hashes divergentes de manifestos: **{len(divergentes)}**. "
        f"JSON/CSV estruturalmente inválidos: **{len(invalidos)}**.\n\n",
        "Modelos (`*.pt`, `*.pkl`), dados brutos, estado local do Obsidian e "
        "figuras opcionais do benchmark Ibrahim permanecem deliberadamente "
        "ignorados; são regeneráveis ou locais e não constituem resultados "
        "canônicos publicáveis.\n\n",
        "## Auditoria paramétrica e dos eixos\n\n",
        "- `a_det` é magnitude de assinatura em `[0, 1]`, não tempo, vida útil "
        "nem probabilidade de falha física.\n",
        "- Os cruzamentos são observados em 120 pontos (`delta_a = 1/119`). O "
        "MLE Weibull 2P usa censura por intervalo em cada célula da grade; "
        "não detecções no teto usam censura à direita sob hipótese declarada.\n",
        "- O papel Weibull agrupa empates da grade. `R2pp` permanece triagem "
        "descritiva, não teste formal de aderência.\n",
        "- As matrizes de confusão preservam contagens, normalizam a cor por "
        "classe real e exibem `n` mais percentual da linha.\n",
        "- ECDF, densidades, ROC/PR e séries temporais mantêm grandeza e unidade "
        "explícitas; painéis com limites distintos incluem aviso para comparação "
        "pelos valores dos eixos.\n\n",
        "## Diagnóstico Weibull 2P\n\n",
        "| Componente | Eventos | Níveis distintos | Empates | beta | eta | R2pp | Síntese |\n",
        "|---|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for item in _resumo_weibull():
        linhas.append(
            f"| {item['falha']} | {item['eventos']}/{item['trajetorias']} | "
            f"{item['niveis']} | {item['empates']:.1%} | {item['beta']:.3f} | "
            f"{item['eta']:.3f} | {item['r2']:.3f} | "
            f"{'recomendada' if item['recomendado'] else 'não recomendada'} |\n"
        )
    linhas.extend([
        "\nO fusível não tem poucos eventos: há 100 cruzamentos. A limitação é "
        "resolução, pois eles se concentram em poucos níveis da grade. IGBT e "
        "fusível só permanecem não recomendados quando o desvio do modelo 2P "
        "continua após corrigir quantização e empates.\n\n",
        "## Arquivos\n\n",
        "- `inventario_artefatos.csv`: relação completa e verificável.\n",
        "- `relatorio_auditoria_artefatos.md`: síntese metodológica desta auditoria.\n",
    ])
    ARQUIVO_MD.write_text("".join(linhas), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="falha se houver hash divergente ou JSON/CSV inválido",
    )
    args = parser.parse_args()
    inventario = construir_inventario()
    escrever_relatorios(inventario)
    problemas = [
        item for item in inventario
        if item["hash_manifesto"] == "divergente"
        or item["qualidade_estrutural"] not in {"ok", "nao_aplicavel"}
    ]
    return 1 if args.check and problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
