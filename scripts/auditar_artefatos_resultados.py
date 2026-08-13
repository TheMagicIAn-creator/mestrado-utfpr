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

from PIL import Image, ImageStat


RAIZ = Path(__file__).resolve().parents[1]
PASTA_MANIFESTOS = RAIZ / "resultados" / "manifestos"
PASTA_AUDITORIA = RAIZ / "resultados" / "auditoria"
ARQUIVO_CSV = PASTA_AUDITORIA / "inventario_artefatos.csv"
ARQUIVO_MD = PASTA_AUDITORIA / "relatorio_auditoria_artefatos.md"
ARQUIVO_FIGURAS = PASTA_AUDITORIA / "catalogo_figuras.csv"

ETAPAS_CANONICAS = {
    "features_gpvs", "autoencoder", "injecao_falhas", "validacao",
    "rul_weibull", "validacao_gpvs_e3",
}
ETAPAS_LEGADAS = {"features_ca", "macro_comparacao"}
SUFIXOS_TEXTO_PORTAVEL = {
    ".csv", ".json", ".md", ".toml", ".txt", ".yaml", ".yml",
}

FIGURAS = {
    "resultados/qualidade/features_gpvs_qualidade.png": (
        "Qualidade das features saudáveis", "GPVS F0L/F0M", "E1",
        "src.ml.gpvs_principal", "tempo do ensaio / desbalanceamento",
        "THD / densidade", True, "canônica",
    ),
    "resultados/autoencoder/curva_treino.png": (
        "Convergência do autoencoder", "GPVS F0L/F0M", "E1",
        "src.ml.autoencoder", "época", "função de perda", False, "canônica",
    ),
    "resultados/autoencoder/diagnostico_escore.png": (
        "Diagnóstico do escore", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.diagnostico_escore", "magnitude injetada a_inj", "taxa de detecção",
        False, "canônica",
    ),
    "resultados/autoencoder/distribuicao_erro.png": (
        "Distribuição do erro saudável", "GPVS F0L/F0M", "E1",
        "src.ml.autoencoder", "erro de reconstrução", "densidade / ECDF",
        False, "canônica",
    ),
    "resultados/autoencoder/erro_temporal.png": (
        "Erro por posição temporal", "GPVS F0L/F0M", "E1",
        "src.ml.autoencoder", "tempo do ensaio", "erro de reconstrução",
        True, "canônica",
    ),
    "resultados/autoencoder/injecao_falhas_comparacao.png": (
        "Detecção por magnitude", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.injecao_falhas", "magnitude injetada a_inj", "taxa de detecção",
        False, "canônica",
    ),
    "resultados/autoencoder/injecao_falhas_resultados.png": (
        "Escore por magnitude", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.injecao_falhas", "magnitude injetada a_inj", "escore / limiar",
        False, "canônica",
    ),
    "resultados/autoencoder/validacao_matriz.png": (
        "Matriz de confusão agregada", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.validacao", "classe predita", "classe real", False, "canônica",
    ),
    "resultados/autoencoder/validacao_matrizes_severidades.png": (
        "Matrizes por severidade", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.validacao", "classe predita", "classe real", False, "canônica",
    ),
    "resultados/autoencoder/validacao_metricas.png": (
        "Métricas por magnitude", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.validacao", "magnitude injetada a_inj", "métrica de classificação",
        False, "canônica",
    ),
    "resultados/autoencoder/validacao_pr.png": (
        "Curvas precisão-revocação", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.validacao", "revocação", "precisão", False, "canônica",
    ),
    "resultados/autoencoder/validacao_roc.png": (
        "Curvas ROC", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.validacao", "taxa de falso positivo", "taxa de verdadeiro positivo",
        False, "canônica",
    ),
    "resultados/autoencoder/weibull_ttf.png": (
        "Primeiro cruzamento do detector", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.graficos_rul", "magnitude a_det", "número de trajetórias",
        False, "canônica",
    ),
    "resultados/autoencoder/weibull_confiabilidade.png": (
        "Sobrevivência empírica do detector", "GPVS F0 + injeção FMECA",
        "E2", "src.ml.graficos_rul", "magnitude a_det",
        "S_D(a)", False, "canônica",
    ),
    "resultados/autoencoder/weibull_intensidade_deteccao.png": (
        "Intensidade paramétrica do primeiro cruzamento",
        "GPVS F0 + injeção FMECA", "E2", "src.ml.graficos_rul",
        "magnitude a_det", "h_D(a), não taxa de falha física",
        False, "canônica",
    ),
    "resultados/autoencoder/weibull_funcoes_distribuicao.png": (
        "Funções da magnitude de detecção", "GPVS F0 + injeção FMECA",
        "E2", "src.ml.graficos_rul", "magnitude a_det",
        "densidade / CDF", False, "canônica",
    ),
    "resultados/autoencoder/weibull_distribuicao.png": (
        "Papel de probabilidade Weibull", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.graficos_rul", "ln(a_det)", "ln[-ln(1-F_D)]",
        False, "canônica",
    ),
    "resultados/autoencoder/weibull_rul.png": (
        "Margem residual de magnitude", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.graficos_rul", "magnitude já aplicada", "margem em a_det",
        False, "canônica",
    ),
    "resultados/autoencoder/weibull_sensibilidade_grade.png": (
        "Sensibilidade à resolução", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.graficos_rul", "magnitude a_det", "ECDF por resolução",
        False, "canônica",
    ),
    "resultados/autoencoder/weibull_modos_operacao.png": (
        "Estratificação F0L/F0M", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.graficos_rul", "magnitude a_det", "ECDF por modo",
        False, "canônica",
    ),
    "resultados/gpvs/gpvs_series_temporais.png": (
        "Séries temporais GPVS", "GPVS F1L-F7M", "E3",
        "src.ml.validacao_gpvs_principal", "tempo do ensaio", "sinal / escore",
        True, "canônica",
    ),
    "resultados/gpvs/gpvs_metricas_por_cenario.png": (
        "Métricas por cenário real", "GPVS F1L-F7M", "E3",
        "src.ml.validacao_gpvs_principal", "ensaio", "métrica de classificação",
        False, "canônica",
    ),
    "resultados/gpvs/gpvs_transferencia_estrita.png": (
        "Transferência estrita", "GPVS F1L-F7M", "E3",
        "src.ml.validacao_gpvs_principal", "ensaio", "métrica de classificação",
        False, "canônica",
    ),
    "resultados/gpvs/gpvs_macro_comparacao.png": (
        "Resumo macro GPVS", "GPVS F1L-F7M", "E3",
        "src.ml.validacao_gpvs_principal", "métrica", "estimativa e IC95%",
        False, "canônica",
    ),
    "resultados/macro/comparacao_deteccao_severidade.png": (
        "Comparação macro", "GPVS e benchmarks documentais", "E2/E4",
        "src.ml.macro_comparacao", "severidade", "detecção", False,
        "comparativa",
    ),
    "resultados/macro/ibrahim_deteccao_severidade.png": (
        "Benchmark Ibrahim", "benchmark documental", "E4",
        "src.ml.macro_comparacao", "severidade", "detecção", False,
        "comparativa",
    ),
    "resultados/macro/proposto_deteccao_severidade.png": (
        "Método proposto", "GPVS F0 + injeção FMECA", "E2",
        "src.ml.macro_comparacao", "severidade", "detecção", False,
        "comparativa",
    ),
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
        [
            "git", "ls-files", "-z", "--cached", "--others",
            "--exclude-standard", "--", "resultados", "dados/processados",
        ],
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


def qualidade_visual_png(caminho: Path) -> dict:
    """Mede integridade básica sem confundir fundo branco com painel vazio."""
    if not caminho.exists():
        return {
            "largura_px": None, "altura_px": None,
            "fracao_nao_branca": None, "variancia_luminancia": None,
            "status_visual": "ausente",
        }
    try:
        with Image.open(caminho) as imagem:
            imagem.verify()
        with Image.open(caminho) as imagem:
            cinza = imagem.convert("L")
            histograma = cinza.histogram()
            total = max(sum(histograma), 1)
            nao_brancos = sum(histograma[:250]) / total
            variancia = float(ImageStat.Stat(cinza).var[0])
            largura, altura = imagem.size
        status = (
            "dimensao_insuficiente" if largura < 800 or altura < 450 else
            "quase_vazio" if nao_brancos < 0.005 or variancia < 2.0 else
            "ok"
        )
        return {
            "largura_px": largura,
            "altura_px": altura,
            "fracao_nao_branca": nao_brancos,
            "variancia_luminancia": variancia,
            "status_visual": status,
        }
    except (OSError, ValueError):
        return {
            "largura_px": None, "altura_px": None,
            "fracao_nao_branca": None, "variancia_luminancia": None,
            "status_visual": "png_invalido",
        }


def construir_catalogo_figuras() -> list[dict]:
    catalogo = []
    for relativo, metadados in FIGURAS.items():
        (
            titulo, dataset, evidencia, gerador, eixo_x, eixo_y,
            eixo_temporal, papel,
        ) = metadados
        catalogo.append({
            "arquivo": relativo,
            "titulo": titulo,
            "dataset": dataset,
            "nivel_evidencia": evidencia,
            "gerador": gerador,
            "eixo_x": eixo_x,
            "eixo_y": eixo_y,
            "eixo_temporal": eixo_temporal,
            "papel": papel,
            **qualidade_visual_png(RAIZ / relativo),
        })
    return catalogo


def escrever_catalogo_figuras(catalogo: list[dict]) -> None:
    PASTA_AUDITORIA.mkdir(parents=True, exist_ok=True)
    campos = list(catalogo[0]) if catalogo else []
    with ARQUIVO_FIGURAS.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(catalogo)


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
            "p_aderencia": (
                ajuste.get("teste_aderencia_quantizada") or {}
            ).get("p_value"),
            "grade_estavel": (
                ajuste.get("sensibilidade_grade") or {}
            ).get("estavel"),
            "modos": ajuste.get("ajustes_por_modo") or {},
            "recomendado": ajuste.get("resumo_parametrico_recomendado"),
        })
    return linhas


def escrever_relatorios(inventario: list[dict], catalogo: list[dict]) -> None:
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
    problemas_visuais = [
        item for item in catalogo if item["status_visual"] != "ok"
    ]
    temporais = sum(bool(item["eixo_temporal"]) for item in catalogo)

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
        "arquivos produzidos por esta própria auditoria. ",
        ", ".join(f"{papel}: {quantidade}" for papel, quantidade in sorted(por_papel.items())),
        ". O CSV anexo registra tamanho, SHA-256, etapa proprietária, papel e "
        "validação estrutural de cada saída.\n\n",
        f"Hashes divergentes de manifestos: **{len(divergentes)}**. "
        f"JSON/CSV estruturalmente inválidos: **{len(invalidos)}**.\n\n",
        f"O catálogo cobre **{len(catalogo)} figuras**: {temporais} com eixo "
        f"temporal e {len(catalogo) - temporais} sem eixo temporal. Problemas "
        f"automáticos de integridade visual: **{len(problemas_visuais)}**.\n\n",
        "Modelos (`*.pt`, `*.pkl`), dados brutos, estado local do Obsidian e "
        "figuras opcionais do benchmark Ibrahim permanecem deliberadamente "
        "ignorados; são regeneráveis ou locais e não constituem resultados "
        "canônicos publicáveis.\n\n",
        "## Auditoria paramétrica e dos eixos\n\n",
        "- `a_det` é magnitude de assinatura em `[0, 1]`, não tempo, vida útil "
        "nem probabilidade de falha física.\n",
        "- Os cruzamentos canônicos são observados em 501 pontos "
        "(`delta_a = 0,002`), com sensibilidade em 101 e 251 pontos. A "
        "persistência ocupa `delta_a = 0,02` em todas as grades. O "
        "MLE Weibull 2P usa censura por intervalo em cada célula da grade; "
        "não detecções no teto usam censura à direita sob hipótese declarada.\n",
        "- O papel Weibull agrupa empates da grade. `R2pp` permanece diagnóstico "
        "visual; a decisão usa bootstrap paramétrico com a mesma quantização e "
        "estabilidade entre as duas grades mais finas.\n",
        "- Papel de probabilidade, PDF/CDF, sobrevivência e intensidade são "
        "figuras separadas. A intensidade `h_D(a)` não contém marcas empíricas "
        "sobre o eixo e é rotulada como não física.\n",
        "- F0L e F0M são estratificados. Ambos pertencem ao GPVS; nenhum "
        "Paderborn ou PMSM entra nos gráficos canônicos.\n",
        "- As matrizes de confusão preservam contagens, normalizam a cor por "
        "classe real e exibem `n` mais percentual da linha.\n",
        "- ECDF, densidades, ROC/PR e séries temporais mantêm grandeza e unidade "
        "explícitas; painéis com limites distintos incluem aviso para comparação "
        "pelos valores dos eixos.\n\n",
        "## Diagnóstico Weibull 2P\n\n",
        "| Componente | Eventos | Níveis distintos | Empates | beta | eta | R2pp | p aderência | Grade estável | Síntese |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n",
    ]
    for item in _resumo_weibull():
        linhas.append(
            f"| {item['falha']} | {item['eventos']}/{item['trajetorias']} | "
            f"{item['niveis']} | {item['empates']:.1%} | {item['beta']:.3f} | "
            f"{item['eta']:.3f} | {item['r2']:.3f} | "
            f"{item['p_aderencia']:.3f} | "
            f"{'sim' if item['grade_estavel'] else 'não'} | "
            f"{'recomendada' if item['recomendado'] else 'não recomendada'} |\n"
        )
    linhas.extend([
        "\nO número de eventos e o número de níveis distintos são grandezas "
        "diferentes: empates não removem trajetórias. A interpretação principal "
        "é a distribuição empírica global; a Weibull 2P só é adotada quando "
        "aderência, resolução e estabilidade permitem.\n\n",
        "## Arquivos\n\n",
        "- `inventario_artefatos.csv`: relação completa e verificável.\n",
        "- `catalogo_figuras.csv`: dataset, gerador, eixos, evidência e QA visual.\n",
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
    catalogo = construir_catalogo_figuras()
    escrever_catalogo_figuras(catalogo)
    escrever_relatorios(inventario, catalogo)
    problemas = [
        item for item in inventario
        if item["hash_manifesto"] == "divergente"
        or item["qualidade_estrutural"] not in {"ok", "nao_aplicavel"}
    ]
    problemas.extend(
        item for item in catalogo if item["status_visual"] != "ok"
    )
    return 1 if args.check and problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
