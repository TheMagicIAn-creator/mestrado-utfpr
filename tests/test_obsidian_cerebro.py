from __future__ import annotations

import json
from pathlib import Path

from src.conhecimento.memoria_persistente import MemoriaPersistente
from src.conhecimento.obsidian import (
    buscar_notas_obsidian,
    contar_notas_indexadas,
    espelhar_memoria_validada,
    ler_nota,
    sincronizar_obsidian,
)


class _Vetores(list):
    def tolist(self):
        return list(self)


class _Embeddings:
    def __init__(self):
        self.chamadas = 0

    def encode(self, textos, **kwargs):
        self.chamadas += 1
        return _Vetores([[float(len(str(texto))), 1.0] for texto in textos])


class _Colecao:
    def __init__(self):
        self.itens = {}

    def count(self):
        return len(self.itens)

    def get(self, include=None):
        ids = list(self.itens)
        return {
            "ids": ids,
            "metadatas": [self.itens[item]["metadata"] for item in ids],
        }

    def upsert(self, *, ids, embeddings, documents, metadatas):
        for item_id, vetor, doc, meta in zip(ids, embeddings, documents, metadatas):
            self.itens[item_id] = {
                "embedding": vetor,
                "document": doc,
                "metadata": meta,
            }

    def delete(self, *, ids):
        for item_id in ids:
            self.itens.pop(item_id, None)

    def query(self, *, query_embeddings, n_results, include=None):
        del query_embeddings, include
        itens = list(self.itens.values())[:n_results]
        return {
            "documents": [[item["document"] for item in itens]],
            "metadatas": [[item["metadata"] for item in itens]],
            "distances": [[0.1 + i * 0.1 for i in range(len(itens))]],
        }


def _escrever_nota(
    raiz: Path,
    *,
    nome: str = "Decisao.md",
    al_iado: str = "true",
    status: str = "ativo",
    corpo: str = "# Decisão\n\n## Método\n\nUsar validação temporal com purga.",
) -> Path:
    raiz.mkdir(parents=True, exist_ok=True)
    caminho = raiz / nome
    caminho.write_text(
        f"""---
al_iado: {al_iado}
titulo: "Decisão temporal"
tipo: decisao
status: {status}
confianca: alta
nivel_evidencia: projeto
tags: [metodologia, temporal]
---

{corpo}

## Conexões

- [[Níveis de evidência]]
""",
        encoding="utf-8",
    )
    return caminho


def test_nota_exige_opt_in_e_schema_curado(tmp_path):
    raiz = tmp_path / "Cerebro"
    ativa = _escrever_nota(raiz)
    rascunho = _escrever_nota(raiz, nome="Rascunho.md", status="rascunho")
    privada = _escrever_nota(raiz, nome="Privada.md", al_iado="false")

    nota = ler_nota(ativa, raiz)

    assert nota is not None
    assert nota.titulo == "Decisão temporal"
    assert nota.wikilinks == ("Níveis de evidência",)
    assert ler_nota(rascunho, raiz) is None
    assert ler_nota(privada, raiz) is None


def test_sincronizacao_e_incremental_e_remove_nota_desativada(tmp_path):
    raiz = tmp_path / "Cerebro"
    caminho = _escrever_nota(raiz)
    colecao = _Colecao()
    modelo = _Embeddings()

    primeiro = sincronizar_obsidian(colecao, modelo, raiz=raiz)
    chamadas_primeiro = modelo.chamadas
    segundo = sincronizar_obsidian(colecao, modelo, raiz=raiz)

    assert primeiro["notas_ativas"] == 1
    assert primeiro["chunks_ativos"] >= 1
    assert contar_notas_indexadas(colecao) == 1
    assert segundo["notas_atualizadas"] == 0
    assert modelo.chamadas == chamadas_primeiro

    texto = caminho.read_text(encoding="utf-8").replace("status: ativo", "status: rascunho")
    caminho.write_text(texto, encoding="utf-8")
    removido = sincronizar_obsidian(colecao, modelo, raiz=raiz)

    assert removido["notas_ativas"] == 0
    assert removido["notas_removidas"] == 1
    assert colecao.count() == 0


def test_busca_rotula_obsidian_como_contexto_nao_bibliografico(tmp_path):
    raiz = tmp_path / "Cerebro"
    _escrever_nota(raiz)
    colecao = _Colecao()
    modelo = _Embeddings()
    sincronizar_obsidian(colecao, modelo, raiz=raiz)

    contexto = buscar_notas_obsidian(
        "Qual foi a decisão sobre validação temporal?",
        modelo,
        colecao,
    )

    assert "CÉREBRO OBSIDIAN" in contexto
    assert "não é evidência bibliográfica" in contexto
    assert "validação temporal" in contexto
    assert "arquivo=Decisao.md" in contexto


def test_memoria_validada_ganha_espelho_markdown_auditavel(tmp_path):
    memoria_json = tmp_path / "memoria.json"
    cerebro = tmp_path / "Cerebro"
    memoria = MemoriaPersistente(memoria_json, pasta_obsidian=cerebro)

    resultado = memoria.registrar(
        {
            "tipo": "preferencia",
            "escopo": "conversa",
            "conteudo": "Prefere respostas objetivas em português.",
            "evidencia_usuario": "Prefiro respostas objetivas em português.",
        },
        origem="teste",
        validado_por="Groq",
        confianca=0.94,
    )

    nota = cerebro / "Memorias validadas" / f"{resultado.item['id']}.md"
    assert nota.is_file()
    texto = nota.read_text(encoding="utf-8")
    assert "al_iado: true" in texto
    assert "nivel_evidencia: usuario" in texto
    assert "fonte de verdade" in texto
    assert "Prefere respostas objetivas" in texto

    nota.unlink()
    repetido = memoria.registrar(
        {
            "tipo": "preferencia",
            "escopo": "conversa",
            "conteudo": "Prefere respostas objetivas em português.",
            "evidencia_usuario": "Prefiro respostas objetivas em português.",
        },
        origem="teste",
        validado_por="Groq",
        confianca=0.94,
    )
    assert repetido.criado is False
    assert nota.is_file()


def test_espelho_reconstroi_json_versionado(tmp_path):
    caminho = tmp_path / "memoria.json"
    raiz = tmp_path / "Cerebro"
    caminho.write_text(
        json.dumps({
            "schema_version": 1,
            "itens": [{
                "id": "abc123",
                "tipo": "contexto_projeto",
                "escopo": "compartilhado",
                "conteudo": "O Obsidian é o mapa navegável do projeto.",
                "evidencia_usuario": "Integre ao cérebro Obsidian.",
                "status": "ativo",
                "confianca": 0.88,
                "validado_por": "Groq",
                "origem": "chat",
            }],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    relatorio = espelhar_memoria_validada(caminho, raiz=raiz)

    assert relatorio["escritas"] == 1
    assert (raiz / "Memorias validadas" / "abc123.md").is_file()


def test_sincroniza_e_recupera_em_chromadb_real(tmp_path):
    import chromadb

    raiz = tmp_path / "Cerebro"
    _escrever_nota(raiz)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    colecao = client.get_or_create_collection(
        "obsidian_teste",
        metadata={"hnsw:space": "cosine"},
    )
    modelo = _Embeddings()

    estado = sincronizar_obsidian(colecao, modelo, raiz=raiz)
    contexto = buscar_notas_obsidian(
        "decisão sobre validação temporal",
        modelo,
        colecao,
    )

    assert estado["notas_ativas"] == 1
    assert colecao.count() == estado["chunks_ativos"]
    assert "Decisão temporal" in contexto
