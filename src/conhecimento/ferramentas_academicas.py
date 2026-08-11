"""Adaptadores de literatura, experimentos, datasets e classificacao."""

from __future__ import annotations

import re
import shutil

from src.conhecimento.intencoes_ferramentas import _experimentos_alvo
from src.core.config import RAIZ_PROJETO
from src.core.texto import normalizar_sem_acentos as _normalizar
from src.ml.pipeline import capacidade_recalculo_pipeline
from src.ml.resultados import resumir_resultados

def buscar_na_web(progresso=None, pergunta: str = "") -> dict:
    """Adapta src.conhecimento.web_search.buscar_web para o formato de ferramenta."""
    from src.conhecimento.web_search import buscar_web

    if progresso:
        progresso(f"Pesquisando na web: '{pergunta[:60]}'...")

    termo = (pergunta or "").strip()
    # Remove gatilhos de comando para deixar só o termo
    for gat in (
        "buscar na web", "pesquisar na web", "pesquise na web", "busque na web",
        "buscar online", "pesquisar online", "procure na internet",
        "procure online", "na internet", "na web", "buscar", "pesquisar",
        "procurar", "google", "googlar",
    ):
        termo = re.sub(rf"\b{gat}\b", "", termo, flags=re.IGNORECASE)
    termo = termo.strip(" ,.;?!:")

    if not termo:
        return {
            "ok": False,
            "etapa": "Busca na web",
            "mensagem": "Me diga o que quer pesquisar (ex.: 'pesquise na web sobre IEC 61724').",
            "imagens": [],
            "resposta_pronta": True,
        }

    out = buscar_web(termo)
    return {
        "ok": bool(out["ok"]),
        "etapa": "Busca na web",
        "mensagem": out["mensagem"],
        "imagens": [],
        "resposta_pronta": False,  # passa pelo LLM para integrar com o contexto
    }


def listar_base_bibliografica(progresso=None, pergunta: str = "") -> dict:
    """
    Devolve o catálogo COMPLETO da literatura indexada (todos os documentos,
    agrupados por tema). Lê os metadados do ChromaDB diretamente — NÃO usa RAG
    — então a lista é determinística e nunca trunca nem inventa referências.
    """
    if progresso:
        progresso("Lendo o catálogo completo da base de conhecimento...")

    try:
        import chromadb

        from src.conhecimento.agente import catalogo_literatura
        from src.core.config import NOME_COLECAO, PASTA_CHROMADB

        cliente = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        colecao = cliente.get_collection(NOME_COLECAO)
        texto = catalogo_literatura(colecao)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "etapa": "Base bibliográfica",
            "mensagem": (
                "Não consegui ler o catálogo da base de conhecimento agora "
                f"({exc}). Verifique se o ChromaDB foi construído."
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    return {
        "ok": True,
        "etapa": "Base bibliográfica",
        "mensagem": texto,
        "imagens": [],
        "resposta_pronta": True,  # texto determinístico — não passa pelo LLM
    }


def _comparacao_desatualizada(pasta_macro) -> str:
    """Devolve o motivo se a comparação macro for mais velha que o detector.

    Vazio significa "pode citar". A checagem é por MANIFESTO, não por mtime de
    arquivo: `git checkout` reescreve mtime e mentiria.
    """
    import json
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO

    manifestos = Path(RAIZ_PROJETO) / "resultados" / "manifestos"

    def _lido(nome: str) -> dict:
        arq = manifestos / f"{nome}.json"
        if not arq.is_file():
            return {}
        try:
            return json.loads(arq.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    macro, ae = _lido("macro_comparacao"), _lido("autoencoder")
    t_macro, t_ae = macro.get("created_at"), ae.get("created_at")
    if not t_macro or not t_ae:
        return ""            # sem manifesto dos dois lados, não afirmo nada
    if t_macro >= t_ae:
        return ""

    commit_macro = str(macro.get("git_commit", ""))[:8] or "?"
    commit_ae = str(ae.get("git_commit", ""))[:8] or "?"
    return (
        f"Ela foi gerada em `{t_macro[:19]}` (commit `{commit_macro}`), mas o "
        f"Autoencoder vigente foi treinado depois, em `{t_ae[:19]}` "
        f"(commit `{commit_ae}`). Os números descrevem um detector que não é "
        f"mais o do pipeline."
    )


def consultar_comparacao_macro(progresso=None, pergunta: str = "") -> dict:
    """Comparação vigente: método proposto × AE-LSTM do Ibrahim, por AUC e SMD.

    Lê `resultados/macro/` — a FONTE ÚNICA de resultado de anomalia desde que os
    macro-códigos substituíram o framework por artigo. Não treina nem recalcula:
    só apresenta o que está publicado, então funciona também na nuvem.

    Existe porque essa pasta era inalcançável pelo chat: `resultados/experimentos/`
    foi deletada em `9fe0322` e nenhuma ferramenta lia `resultados/macro/`. Pedir
    "compare meu método com a literatura" caía num caminho morto.
    """
    import json
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO

    if progresso:
        progresso("Lendo a comparação publicada (proposto × Ibrahim)...")

    pasta = Path(RAIZ_PROJETO) / "resultados" / "macro"
    tabela = pasta / "comparacao_tabela.md"
    dados = pasta / "comparacao_resultado.json"
    if not tabela.is_file() or not dados.is_file():
        return {
            "ok": False, "etapa": "Comparação com a literatura",
            "mensagem": (
                "Ainda não há comparação publicada em `resultados/macro/`. "
                "Rode no PC: `python -m src.ml.macro_comparar`."
            ),
            "imagens": [], "resposta_pronta": True, "forcar_resposta_direta": True,
        }

    try:
        metodos = json.loads(dados.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "ok": False, "etapa": "Comparação com a literatura",
            "mensagem": f"A comparação publicada está ilegível: {exc}",
            "imagens": [], "resposta_pronta": True, "forcar_resposta_direta": True,
        }

    # A comparação pode ter sido medida sobre um modelo que já foi substituído.
    # Aconteceu: em 09/08/2026 ela saiu às 14:59 e o Autoencoder foi retreinado
    # às 23:37, na migração para o GPVS-Faults. O artefato não carrega aviso de
    # obsolescência, então quem perguntasse "sou melhor que o AE-LSTM?" receberia
    # números de um detector aposentado com cara de vigentes.
    obsoleta = _comparacao_desatualizada(pasta)
    if obsoleta:
        return {
            "ok": False, "etapa": "Comparação com a literatura",
            "mensagem": (
                "⚠️ **A comparação publicada está desatualizada e não deve ser "
                "citada.**\n\n" + obsoleta
                + "\n\nRode no PC para atualizá-la: "
                  "`python -m src.ml.macro_comparar`."
            ),
            "imagens": [], "resposta_pronta": True, "forcar_resposta_direta": True,
        }

    # Protocolo: sai do próprio artefato, não de constante escrita aqui.
    protocolo = [
        f"- **{m.get('nome', '?')}** — limiar no percentil "
        f"{m.get('percentil', '?')}, FP {m.get('fp_pct', 0):.1f}%, "
        f"{m.get('n_calib', '?')} janelas de calibração e "
        f"{m.get('n_aval', '?')} de avaliação"
        for m in metodos
    ]

    msg = (
        "## Método proposto × literatura\n\n"
        + tabela.read_text(encoding="utf-8").strip()
        + "\n\n**Protocolo de cada método**\n"
        + "\n".join(protocolo)
        + "\n\n**Como ler**\n"
        "- **SMD** é a menor severidade em que o método detecta a falha em ≥95% "
        "das janelas, com o falso positivo travado em 10%. **Menor é melhor** — "
        "é o *pickup* do detector.\n"
        "- Em severidade 1,0 todos saturam em 100%: é o SMD que discrimina.\n\n"
        "**Ressalvas** — evidência **E2** (falha sintética injetada no sinal, "
        "fundamentada na FMECA): mostra que o detector responde à assinatura "
        "elétrica esperada, **não** desempenho em campo. Amostra pequena "
        f"({metodos[0].get('n_aval', '?')} janelas), então os valores são "
        "consistentes, não precisos. A grade de severidade é discreta "
        "(0,05…1,0): um SMD de 0,50 significa \"falhou em 0,3, passou em 0,5\"."
    )

    imagens = []
    for arquivo, legenda in (
        ("comparacao_deteccao_severidade.png", "Detecção por severidade — proposto × Ibrahim"),
        ("proposto_deteccao_severidade.png", "Método proposto — detecção por severidade"),
        ("ibrahim_deteccao_severidade.png", "Ibrahim (AE-LSTM) — detecção por severidade"),
    ):
        caminho = pasta / arquivo
        if caminho.is_file():
            imagens.append({
                "path": str(caminho), "caption": legenda,
                "group": "Comparação com a literatura", "inline": False,
            })

    return {
        "ok": True, "etapa": "Comparação com a literatura",
        "mensagem": msg, "imagens": imagens, "resposta_pronta": True,
    }


def listar_experimentos_artigos(progresso=None, pergunta: str = "") -> dict:
    """Catálogo dos experimentos de ML por artigo-base + status dos modelos."""
    if progresso:
        progresso("Lendo o catálogo de experimentos por artigo...")
    try:
        from src.ml.experimentos_artigos import catalogo_experimentos_md

        msg = catalogo_experimentos_md()
        msg += (
            "\n\nPara rodar, peça por exemplo: \"rode o experimento do Ibrahim\" "
            "ou use a barra lateral (🧪 Experimentos por artigo)."
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "etapa": "Experimentos por artigo",
            "mensagem": f"Não consegui ler o catálogo de experimentos: {exc}",
            "imagens": [], "resposta_pronta": True,
        }
    return {
        "ok": True, "etapa": "Experimentos por artigo",
        "mensagem": msg, "imagens": [], "resposta_pronta": True,
    }


def limpar_experimentos_artigos(progresso=None, pergunta: str = "") -> dict:
    """Apaga artefatos dos experimentos por artigo mediante confirmacao."""
    from pathlib import Path

    from src.ml.experimentos_artigos import ORDEM_EXPERIMENTOS, PASTA_EXPERIMENTOS

    alvos = _experimentos_alvo(pergunta) or list(ORDEM_EXPERIMENTOS)
    alvos = list(dict.fromkeys(alvos))
    rotulo = (
        "TODOS"
        if len(alvos) >= len(ORDEM_EXPERIMENTOS)
        else " ".join(k.upper() for k in alvos)
    )
    token = (
        "CONFIRMAR LIMPEZA EXPERIMENTOS"
        if rotulo == "TODOS" else
        f"CONFIRMAR LIMPEZA EXPERIMENTOS {rotulo}"
    )

    pasta_base = Path(PASTA_EXPERIMENTOS).resolve()
    pastas = []
    for key in alvos:
        p = (pasta_base / key).resolve()
        if pasta_base in p.parents:
            pastas.append(p)

    existentes = [p for p in pastas if p.exists()]
    n_arquivos = sum(
        1 for pasta in existentes for item in pasta.rglob("*") if item.is_file()
    )

    if _normalizar(token) not in _normalizar(pergunta):
        nomes = ", ".join(alvos)
        return {
            "ok": True,
            "etapa": "Limpeza de experimentos",
            "mensagem": (
                f"Isso vai apagar os artefatos dos experimentos: **{nomes}**.\n\n"
                f"Diretorios encontrados: {len(existentes)} | arquivos: {n_arquivos}.\n"
                "A acao e irreversivel e nao apaga dados brutos nem literatura.\n\n"
                f"Para confirmar, escreva exatamente:\n\n`{token}`"
            ),
            "imagens": [],
            "resposta_pronta": True,
        }

    if progresso:
        progresso("Apagando artefatos dos experimentos por artigo...")

    removidos = []
    for pasta in existentes:
        shutil.rmtree(pasta)
        removidos.append(pasta)

    if removidos:
        detalhe = "\n".join(f"- {p.relative_to(RAIZ_PROJETO)}" for p in removidos)
        detalhe = f"\n\nDiretorios removidos:\n{detalhe}"
    else:
        detalhe = "\n\nNao havia diretorios de experimento para remover."

    return {
        "ok": True,
        "etapa": "Limpeza de experimentos",
        "mensagem": (
            "Experimentos por artigo apagados. Os dados brutos permanecem "
            "intactos; quando quiser comparar novamente, peca para rodar o "
            "experimento do autor desejado ou todos os experimentos."
            f"{detalhe}"
        ),
        "imagens": [],
        "resposta_pronta": True,
    }


def _md_experimento_legacy(res: dict) -> tuple[str, list[dict]]:
    """Markdown + imagens de um resultado de experimento."""
    if not res.get("ok"):
        ref = res.get("referencia", res.get("experimento", "experimento"))
        return f"### {ref}\nNão executado — {res.get('mensagem', 'sem modelos disponíveis')}.", []

    mp = res["metrica_principal"]
    linhas = [
        f"### {res['referencia']} — {res['dataset']} ({res['tarefa']})",
        f"| Modelo | {mp} | demais |",
        "|---|---:|---|",
    ]
    for nome, m in res["modelos"].items():
        if not m.get("disponivel", True):
            linhas.append(f"| {nome} | — | _{m.get('motivo', 'indisponível')}_ |")
            continue
        principal = m.get(mp)
        outras = ", ".join(
            f"{k}={v:.3f}" for k, v in m.items()
            if isinstance(v, (int, float)) and k not in (mp, "disponivel")
        )
        linhas.append(f"| {nome} | {principal:.4f} | {outras} |")
    linhas.append(
        f"\n**Melhor: {res['melhor_modelo']}** ({mp}={res['melhor_valor']:.4f}). "
        f"Salvo em `resultados/experimentos/{res['experimento']}/`."
    )
    imagens = []
    graf = res.get("grafico")
    if graf:
        from src.core.utils import resolve_project_path
        graf_abs = resolve_project_path(graf)  # relativo→absoluto (na interface)
        if graf_abs.exists():
            imagens.append({"path": str(graf_abs), "caption": f"{res['referencia']} — comparação"})
    return "\n".join(linhas), imagens


def _md_experimento(res: dict) -> tuple[str, list[dict]]:
    """Markdown + imagens no schema padronizado dos experimentos."""
    if not res.get("ok"):
        ref = res.get("referencia", res.get("experimento", "experimento"))
        return f"### {ref}\nNao executado - {res.get('mensagem', 'sem modelos disponiveis')}.", []

    mp = res["metrica_principal"]
    linhas = [
        f"### {res['referencia']} - {res['dataset']} ({res['tarefa']})",
        "| Modelo | Accuracy | Precision | Recall | F1 | AUC | Specificity | Anomalias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nome, m in res["modelos"].items():
        if not m.get("disponivel", True):
            linhas.append(f"| {nome} (_{m.get('motivo', 'indisponivel')}_) | - | - | - | - | - | - | - |")
            continue
        valores = []
        for chave in ("accuracy", "precision", "recall", "f1", "auc", "specificity"):
            valor = m.get(chave)
            valores.append(f"{valor:.3f}" if isinstance(valor, (int, float)) else "-")
        anomalias = m.get("anomalias_detectadas", "-")
        linhas.append(
            f"| {nome} | {valores[0]} | {valores[1]} | {valores[2]} | "
            f"{valores[3]} | {valores[4]} | {valores[5]} | {anomalias} |"
        )
    linhas.append(
        f"\n**Melhor: {res['melhor_modelo']}** ({mp}={res['melhor_valor']:.4f}). "
        f"Salvo em `resultados/experimentos/{res['experimento']}/`."
    )

    # Bloco de METODOLOGIA do protocolo por artigo (split temporal, injeção
    # FMECA e a regra de decisão de cada modelo) — rastreabilidade na resposta.
    met = res.get("metodologia")
    if met:
        linhas.append(f"\n**Protocolo do artigo** (`{met.get('protocolo', '?')}`):")
        sp = met.get("split", {})
        if sp:
            partes = [f"treino={sp.get('treino')}", f"teste={sp.get('teste')}"]
            if sp.get("val"):
                partes.insert(1, f"val={sp.get('val')}")
            linhas.append(
                f"- Split {sp.get('tipo', '?')} (purga={sp.get('purga_janelas')}): "
                f"{', '.join(partes)} janelas.")
        inj = met.get("injecao", {})
        if inj:
            linhas.append(
                f"- Injeção: {inj.get('tipo', '?')} — famílias FMECA "
                f"{', '.join(inj.get('falhas', []))} (severidade {inj.get('severidade')}).")
        for modelo, regra in (met.get("decisoes") or {}).items():
            linhas.append(f"- Decisão {modelo}: {regra}.")
        for nota in met.get("fidelidade", []):
            linhas.append(f"- _{nota}_")

    # Detecção por família de falha FMECA (quando o protocolo reporta)
    com_falhas = {
        nome: m["deteccao_por_falha"]
        for nome, m in res["modelos"].items()
        if isinstance(m, dict) and m.get("deteccao_por_falha")
    }
    if com_falhas:
        linhas.append("\n**Detecção por família de falha (recall):**")
        linhas.append("| Modelo | Contator AC (NPR 315) | IGBT (NPR 90) | Fusível AC (NPR 30) |")
        linhas.append("|---|---:|---:|---:|")
        for nome, det in com_falhas.items():
            def _pct(v):
                return f"{v:.0%}" if isinstance(v, (int, float)) else "—"
            linhas.append(
                f"| {nome} | {_pct(det.get('contator_ac'))} | "
                f"{_pct(det.get('igbt'))} | {_pct(det.get('fusivel_ac'))} |")

    from src.core.utils import resolve_project_path

    imagens = []
    for graf in res.get("graficos", []) or [res.get("grafico")]:
        if not graf:
            continue
        graf_abs = resolve_project_path(graf)  # relativo→absoluto (na interface)
        if graf_abs.exists():
            imagens.append({"path": str(graf_abs), "caption": f"{res['referencia']} - experimento"})
    return "\n".join(linhas), imagens


def rodar_experimento_artigo(progresso=None, pergunta: str = "") -> dict:
    """Roda um ou mais experimentos por artigo e devolve a comparação."""
    if not capacidade_recalculo_pipeline()["disponivel"]:
        resumo = resumir_resultados(pergunta)
        return {
            "ok": True,
            "etapa": "Experimentos por artigo",
            "mensagem": (
                "## Experimento indisponível neste ambiente\n\n"
                "Os experimentos exigem os dados locais de Paderborn. O site "
                "não os recalcula, mas pode consultar os resultados publicados.\n\n"
                + resumo["mensagem"]
            ),
            "imagens": resumo.get("imagens", []),
            "resposta_pronta": True,
        }

    from src.ml.experimentos_artigos import catalogo_experimentos_md
    # 10.4 — isola cargas pesadas (torch) em subprocesso para
    # que um segfault/conflito de OpenMP não derrube o app. Cai para in-process
    # se o subprocesso não puder ser lançado.
    from src.ml.exec_experimento_isolado import (
        executar_experimento_isolado as executar_experimento,
    )

    alvos = _experimentos_alvo(pergunta)
    if not alvos:
        return {
            "ok": True, "etapa": "Experimentos por artigo",
            "mensagem": (
                "Diga qual experimento rodar (por autor). Ex.: \"rode o "
                "experimento do Ibrahim\" ou \"compare os experimentos de "
                "anomalia\".\n\n" + catalogo_experimentos_md()
            ),
            "imagens": [], "resposta_pronta": True,
        }

    blocos, imagens = [], []
    for key in alvos:
        if progresso:
            progresso(f"Rodando experimento: {key}...")
        try:
            res = executar_experimento(key, progresso=progresso)
        except Exception as exc:  # noqa: BLE001
            res = {"experimento": key, "ok": False, "mensagem": str(exc)}
        md, imgs = _md_experimento(res)
        blocos.append(md)
        imagens.extend(imgs)

    cabecalho = (
        "## Experimentos por artigo — resultados\n"
        if len(alvos) > 1 else ""
    )
    return {
        "ok": True, "etapa": "Experimentos por artigo",
        "mensagem": cabecalho + "\n\n".join(blocos),
        "imagens": imagens, "resposta_pronta": True,
    }


def _contar_linhas(caminho) -> int:
    """Conta linhas de um arquivo grande sem carregá-lo na memória."""
    total = 0
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            total += bloco.count(b"\n")
    return total


def consultar_datasets(progresso=None, pergunta: str = "") -> dict:
    """Explica o dataset principal e separa explicitamente os legados."""
    if progresso:
        progresso("Lendo metadados dos datasets...")
    import json
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO

    raiz = Path(RAIZ_PROJETO)
    gpvs_resultado = raiz / "resultados" / "gpvs" / "validacao_gpvs_e3.json"
    gpvs_bruto = raiz / "dados" / "brutos" / "gpvs" / "csv" / "CSV_Files"
    arquivos_gpvs = [gpvs_bruto / f"F{i}{modo}.csv" for i in range(8) for modo in "LM"]
    n_locais = sum(caminho.exists() for caminho in arquivos_gpvs)
    linhas = [
        "## Dataset do pipeline principal\n",
        "### GPVS-Faults — microrede fotovoltaica experimental\n",
        "- Única fonte de dados dos resultados canônicos novos.\n",
        f"- Arquivos locais: {n_locais}/16 (`F0L.csv` a `F7M.csv`).\n",
        "- F0L/F0M: treino, validação, calibração e teste saudável.\n",
        "- F1L-F7M: validação E3 de bancada de sete falhas em IPPT/MPPT.\n",
        "- E2: assinaturas sintéticas FMECA aplicadas somente ao holdout F0.\n",
        "- DOI: https://doi.org/10.17632/n76t439f65.1.\n",
    ]
    if gpvs_resultado.exists():
        try:
            gpvs = json.loads(gpvs_resultado.read_text(encoding="utf-8"))
            resumo = gpvs["macro_summary"]["canonical_ae"]["all"]
            linhas.append(
                f"- Resultado E3: AUC macro {resumo['auc']['mean']:.3f}, "
                f"sensibilidade {resumo['sensitivity']['mean']:.3f} e "
                f"especificidade {resumo['specificity']['mean']:.3f}.\n"
                "- Um único Autoencoder: sem retreino nem recalibração do limiar "
                "por ensaio; há baseline de comissionamento pré-falha.\n"
            )
        except Exception as exc:  # noqa: BLE001
            linhas.append(
                f"- Artefato presente, mas não foi possível ler as métricas ({exc})."
            )
    else:
        linhas.append(
            "- Resultado E3 ainda não publicado; a existência dos CSVs, sozinha, "
            "não constitui validação."
        )

    linhas.append(
        "\n### Fora do resultado canônico\n"
        "- **Stender**, **PMSM**, **PV Farms**, telemetria residencial e Bearing "
        "DataCenter permanecem apenas como literatura, auditoria ou experimentos "
        "legados. Nenhuma linha ou métrica deles é fundida ao GPVS.\n"
        "- **Stender/Paderborn University** e **PV Farms** são preservados pelo "
        "nome para consulta histórica, nunca como fonte do resultado vigente.\n"
        "- Stender não é o Paderborn Bearing Dataset; PV Farms é simulado no domínio CC e não "
        "diagnostica falhas CA do inversor.\n"
        "\n**Separação de domínio:** GPVS é o único dataset canônico; os demais "
        "não fornecem amostras nem métricas ao pipeline principal.\n"
        "\n**Weibull físico:** nenhum desses conjuntos fornece, tal como está, "
        "tempos de vida/falha de unidades independentes com censura. A Weibull "
        "atual usa `a_det` sintético (E2), não tempo físico nem RUL de campo."
    )
    return {
        "ok": True, "etapa": "Datasets do projeto",
        "mensagem": "\n".join(linhas), "imagens": [], "resposta_pronta": True,
    }


def comparar_abordagens_ml(progresso=None, pergunta: str = "") -> dict:
    """Compara supervisionado x não supervisionado x sintético (FMECA), com rigor."""
    msg = (
        "## Abordagens de ML na dissertação\n\n"
        "| Abordagem | O que faz | Rótulos? | No projeto |\n"
        "|---|---|---|---|\n"
        "| **Não supervisionada** | aprende a normalidade | não | GPVS F0L/F0M: Autoencoder denso canônico |\n"
        "| **Sintética FMECA (E2)** | testa assinaturas CA modeladas | ground truth sintético | holdout saudável GPVS F0 |\n"
        "| **Experimental (E3)** | testa falhas reais de bancada | fase pré/pós-falha | GPVS F1L-F7M, mesmo detector e limiar |\n"
        "| **Legada/comparativa** | contextualiza métodos | varia | Stender/Paderborn, PMSM, PV Farms e AE-LSTM; fora do resultado canônico |\n\n"
        "**Rigor:**\n"
        "- O não supervisionado DETECTA anomalia, mas NÃO garante diagnóstico "
        "causal da falha.\n"
        "- A validação sintética (E2) depende de calibração física (ex.: o ruído "
        "de sensor é um proxy).\n"
        "- Separação de domínio: nenhum dataset externo é fundido ao GPVS ou transfere métricas ao "
        "pipeline principal.\n"
        "- PV Farms não diagnostica falhas CA do inversor.\n"
        "- Na E3 não há retreino nem recalibração do limiar; a primeira metade "
        "pré-falha serve apenas ao baseline de comissionamento.\n"
        "- Nenhum dataset atual autoriza Weibull em tempo físico; `a_det` permanece "
        "uma intensidade sintética E2.\n"
        "- Comparações históricas por artigo mantêm protocolos próprios e não "
        "devem ser somadas às métricas GPVS."
    )
    return {
        "ok": True, "etapa": "Abordagens de ML",
        "mensagem": msg, "imagens": [],
        "resposta_pronta": True, "forcar_resposta_direta": True,
    }


def treinar_classificador_pv(progresso=None, pergunta: str = "") -> dict:
    """Treina e salva o classificador supervisionado PV Farms (CC)."""
    if progresso:
        progresso("Treinando o classificador PV Farms (CC)...")
    try:
        from src.ml.classificador_pv_infer import AVISO_DOMINIO, treinar_e_salvar

        r = treinar_e_salvar()
        m = r["metricas"]
        msg = (
            f"Classificador PV Farms (**CC**) treinado e salvo. "
            f"{r['n_features']} features, classes {r['classes']}.\n\n"
            f"F1={m.get('f1', 0):.3f} · MCC={m.get('mcc', 0):.3f} · "
            f"balanced_acc={m.get('balanced_accuracy', 0):.3f}. Evidência **E1**.\n\n"
            f"{AVISO_DOMINIO}"
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "etapa": "Classificador PV Farms",
                "mensagem": f"Não consegui treinar: {exc}", "imagens": [],
                "resposta_pronta": True}
    return {"ok": True, "etapa": "Classificador PV Farms", "mensagem": msg,
            "imagens": [], "resposta_pronta": True}


def avaliar_classificador_pv(progresso=None, pergunta: str = "") -> dict:
    """Mostra métricas + limitações do classificador PV Farms já treinado."""
    import json
    from pathlib import Path

    from src.core.config import RAIZ_PROJETO
    from src.ml.classificador_pv_infer import AVISO_DOMINIO

    arq = Path(RAIZ_PROJETO) / "resultados" / "classificacao_pv" / "metricas.json"
    if not arq.exists():
        return {"ok": True, "etapa": "Classificador PV Farms",
                "mensagem": "Classificador ainda não treinado. Peça: \"treine o "
                "classificador PV Farms\".", "imagens": [], "resposta_pronta": True}
    m = json.loads(arq.read_text(encoding="utf-8"))
    msg = (
        "## Classificador PV Farms (CC)\n"
        f"- Modelo: {m.get('modelo', 'Random Forest')} · evidência **E1**\n"
        f"- Acurácia: {m.get('accuracy', 0):.3f} · F1: {m.get('f1', 0):.3f} · "
        f"MCC: {m.get('mcc', 0):.3f} · balanced_acc: {m.get('balanced_accuracy', 0):.3f}\n"
        f"- Specificity ({m.get('specificity_tipo', '-')}): {m.get('specificity', 0):.3f}\n\n"
        f"{AVISO_DOMINIO}"
    )
    return {"ok": True, "etapa": "Classificador PV Farms", "mensagem": msg,
            "imagens": [], "resposta_pronta": True}


def classificar_amostra_pv(progresso=None, pergunta: str = "") -> dict:
    """Classifica uma amostra PV Farms enviada como JSON na mensagem."""
    import json
    import re

    from src.ml.classificador_pv_infer import AVISO_DOMINIO, classificar

    achado = re.search(r"\{.*\}", pergunta or "", re.S)
    if not achado:
        return {"ok": True, "etapa": "Classificação PV Farms",
                "mensagem": "Envie a amostra como JSON, ex.: "
                "`classifique a amostra {\"feature_0\": 1.2, ...}`.\n\n" + AVISO_DOMINIO,
                "imagens": [], "resposta_pronta": True}
    try:
        amostra = json.loads(achado.group(0))
    except Exception:
        return {"ok": True, "etapa": "Classificação PV Farms",
                "mensagem": "JSON inválido. Use {\"coluna\": valor, ...}.\n\n" + AVISO_DOMINIO,
                "imagens": [], "resposta_pronta": True}
    r = classificar(amostra)
    if not r.get("ok"):
        msg = f"Não classifiquei: {r.get('erro')}\n\n{r.get('aviso', AVISO_DOMINIO)}"
    else:
        msg = (f"Classe prevista: **{r['classe_nome']}** "
               f"(probabilidade {r['probabilidade']:.2f}) — domínio CC.\n\n"
               f"{r['aviso']} (importância de feature ≠ causalidade.)")
    return {"ok": True, "etapa": "Classificação PV Farms", "mensagem": msg,
            "imagens": [], "resposta_pronta": True}
