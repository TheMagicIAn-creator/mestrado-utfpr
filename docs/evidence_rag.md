# Evidence RAG do ALIAdo

## Fluxo canônico

```text
PDF
 -> chunking page-aware
 -> JSONL v2 (raw_text separado de retrieval_text)
 -> embedding multilíngue
 -> ChromaDB + BM25
 -> Reciprocal Rank Fusion
 -> filtro consultivo de metadados
 -> reranking e expansão de vizinhança
 -> Evidence Package
 -> Evidence Guard
 -> Router LLM
 -> resposta citável ou abstenção
```

O PDF e o `raw_text` são a fonte documental. O `retrieval_text` contextual serve
somente à recuperação e nunca pode ser apresentado como citação literal.

## Perfil promovido

O perfil padrão é `r4_hybrid`. Ele preserva busca semântica e BM25, usa RRF com
constante 60, consulta filtros explícitos de metadados sem torná-los restrições
duras, limita a busca filtrada aos três melhores trechos e expande vizinhos
somente depois do ranking.

O Evidence Guard exige a cadeia:

```text
claim -> evidence_id -> chunk_id -> document_id -> página -> PDF
```

Autor, ano, página e quote são conferidos contra o pacote recuperado. Memória,
Obsidian e sessões não têm valor de fonte científica. Sem suporte suficiente, o
comportamento correto é abster-se.

## Rollback

Para voltar temporariamente ao ranking anterior, sem alterar arquivos:

```powershell
$env:AL_IADO_RETRIEVAL_PROFILE = "baseline"
python -m src.webapp
```

Para retornar ao perfil promovido:

```powershell
Remove-Item Env:AL_IADO_RETRIEVAL_PROFILE -ErrorAction SilentlyContinue
python -m src.webapp
```

O snapshot anterior permanece recuperável pelo histórico Git imediatamente
anterior à promoção R6. Os manifestos R0-R5 mantêm hashes, métricas e decisões.
