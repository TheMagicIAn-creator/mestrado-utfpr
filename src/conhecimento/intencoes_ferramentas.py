"""Predicados lexicais pequenos usados pelo roteador de ferramentas."""

from __future__ import annotations

from src.core.texto import normalizar_sem_acentos


_TERMOS_BIBLIO = (
    "bibliografia",
    "bibliografica",
    "referencias",
    "artigos",
    "papers",
    "literatura",
)
_TERMOS_TOTALIDADE = ("todas", "todos", "completa", "completo", "inteira", "inteiro")
_GATILHOS_CATALOGO_FORTE = (
    "base bibliografica",
    "catalogo bibliografico",
    "inventario da literatura",
    "o que voce tem indexado",
)
_QUALIFICADORES_TOPICO = ("sobre", "acerca", "relacionado", "tema")
_TERMOS_PIPELINE_IMPLICITO = (
    "recalcule",
    "recalcular",
    "retreine",
    "retreinar",
    "refaca",
    "refazer",
    "rode tudo",
    "execute tudo",
)


def _text(value: str) -> str:
    return normalizar_sem_acentos(value or "").lower()


def _deve_forcar(pergunta: str) -> bool:
    text = _text(pergunta)
    return any(term in text for term in _TERMOS_PIPELINE_IMPLICITO + ("do zero", "force"))


def _quer_adicionar_anexo_biblioteca(
    pergunta: str,
    *,
    tem_anexos: bool = False,
) -> bool:
    text = _text(pergunta)
    action = any(
        term in text
        for term in (
            "adicione",
            "adicionar",
            "importe",
            "importar",
            "indexe",
            "indexar",
            "inclua",
            "incluir",
            "salve",
            "persistir",
        )
    )
    target = any(
        term in text
        for term in ("anexo", "arquivo", "pdf", "biblioteca", "literatura", "fonte")
    )
    return action and (target or tem_anexos)


def _quer_leitura_efemera_anexo(pergunta: str) -> bool:
    text = _text(pergunta)
    action = any(
        term in text
        for term in (
            "leia",
            "ler",
            "resuma",
            "resumir",
            "analise",
            "analisar",
            "explique",
            "o que diz",
        )
    )
    target = any(term in text for term in ("anexo", "arquivo", "pdf", "documento", "texto"))
    return action and target


def _quer_status(pergunta: str) -> bool:
    text = _text(pergunta)
    return "status" in text and any(
        term in text for term in ("pipeline", "resultado", "publicacao", "artefato")
    )


def _quer_catalogo(pergunta: str) -> bool:
    text = _text(pergunta)
    if any(trigger in text for trigger in _GATILHOS_CATALOGO_FORTE):
        return True
    return any(term in text for term in _TERMOS_BIBLIO) and any(
        term in text for term in _TERMOS_TOTALIDADE
    ) and not any(term in text for term in _QUALIFICADORES_TOPICO)


def _quer_literatura_tematica(pergunta: str) -> bool:
    text = _text(pergunta)
    return any(term in text for term in _TERMOS_BIBLIO) and any(
        term in text for term in _QUALIFICADORES_TOPICO
    )


def _quer_consultar_datasets(pergunta: str) -> bool:
    text = _text(pergunta)
    return any(
        term in text
        for term in (
            "dataset", "data set", "gpvs", "conjunto de dados",
            "paderborn", "pv farms", "pmsm",
        )
    )


def _quer_comparar_abordagens(pergunta: str) -> bool:
    text = _text(pergunta)
    return "compar" in text and any(
        term in text
        for term in ("abordagem", "metodo", "denso", "lstm", "e3")
    )


def _quer_registrar_no_cerebro(pergunta: str) -> bool:
    text = _text(pergunta)
    action = any(term in text for term in ("registre", "registrar", "guarde", "guardar", "anote"))
    target = any(term in text for term in ("cerebro", "vault", "obsidian", "memoria", "decisao", "resultado"))
    return action and target


def _quer_limpar(pergunta: str) -> bool:
    text = _text(pergunta)
    if any(
        term in text
        for term in ("painel", "interface", "layout", "aba", "menu", "codigo", "função", "funcao")
    ):
        return False
    return any(
        term in text
        for term in ("limpar", "limpe", "apagar", "apague", "remover", "remova", "excluir", "exclua")
    ) and any(
        term in text for term in ("resultado", "artefato", "comparacao", "confiabilidade")
    )


def _quer_resposta_autoral(pergunta: str) -> bool:
    text = _text(pergunta)
    return any(
        term in text
        for term in (
            "escreva", "redija", "explique", "analise", "interprete", "resuma",
            "opiniao", "o que isso significa", "reforca", "qual o melhor",
            "minha proposta",
            "como ta",
            "como esta",
        )
    )


def _parece_pedido_de_ferramenta(pergunta: str) -> bool:
    text = _text(pergunta)
    return any(
        term in text
        for term in (
            "execute",
            "rode",
            "gere",
            "recalcule",
            "retreine",
            "mostre os resultados",
            "status do pipeline",
            "buscar na web",
            "adicione",
            "importe",
            "indexe",
            "apague",
            "exclua",
            "limpe",
            "confirmar",
            "cancelar",
        )
    )


__all__ = [name for name in globals() if name.startswith("_") and not name.startswith("__")]
