# Fechamento da auditoria semantica das PRs #99 e #102

Data da revisao: 31/08/2026.

## Escopo

Esta revisao fecha a ressalva registrada no commit `35f36b2`, segundo a qual as
PRs #99 e #102 nao deveriam ser tratadas como auditadas. A avaliacao considerou:

- os diffs dos merges historicos `ab54deb` (#99) e `fbcc007` (#102);
- os contratos publicos preservados no codigo atual;
- os descendentes ativos das funcoes extraidas;
- a remocao posterior de modulos substituidos;
- testes comportamentais e arquiteturais no `main` de 31/08/2026.

O objetivo nao foi declarar equivalencia byte a byte com codigo que deixou de
existir, mas verificar se alguma mudanca semantica silenciosa dessas duas
refatoracoes continua ativa no produto ou nas publicacoes cientificas.

## PR #99 - utilitarios de texto e log

### Alteracao historica

A PR centralizou tres semanticas textuais em `src/core/texto.py`:

- `normalizar_sem_acentos`: converte para texto, remove acentos e usa minusculas;
- `normalizar_espacos`: acrescenta compactacao de espacos e preserva pontuacao;
- `normalizar_busca`: mantem apenas termos alfanumericos para busca.

Tambem consolidou o adaptador de chamadas semelhantes a `print` em
`adaptar_logger_como_print`.

### Verificacao atual

- Os chamadores de busca usam `normalizar_busca` ou a normalizacao sem acentos,
  conforme o contrato lexical esperado.
- Exportacao e snippets usam a variante que preserva pontuacao.
- `None`, zero, `False`, acentos, pontuacao, espacos e progresso com `end="\r"`
  possuem testes diretos.
- O teste estrutural impede o retorno dos antigos utilitarios duplicados.

**Conclusao:** nenhuma regressao semantica ativa foi encontrada. A PR #99 passa
a ser tratada como auditada no estado atual.

## PR #102 - modularizacao dos pontos criticos

### Alteracao historica

A PR separou responsabilidades antes concentradas em `agente.py`,
`ferramentas.py`, `obsidian.py`, na interface Streamlit e nos antigos modulos de
experimentos/Weibull. Os tres primeiros continuam ativos; Streamlit e os
modulos cientificos antigos foram substituidos por implementacoes canonicas e
nao pertencem mais a arvore.

### Verificacao atual

- `agente.py` permanece como fachada e resolve tardiamente os contratos movidos
  para `agente_interacao.py`, `agente_recuperacao.py` e `agente_contexto.py`.
- `obsidian.py` preserva por importacao tardia as consultas movidas para
  `consultas_obsidian.py`.
- `ferramentas.py` reexporta os contratos movidos para os modulos academicos e
  de intencao, enquanto mantem o despacho cientifico atual.
- O mapa de `src/` documenta as responsabilidades vigentes.
- Os testes de dependencia impedem a inversao `ml -> conhecimento`, verificam o
  lock de escrita no ChromaDB e rejeitam excecao ampla descartada em silencio.
- O limite de 1.000 linhas e a matriz de responsabilidade dos modulos criticos
  permanecem testados.
- Foi acrescentada uma regressao explicita para as fachadas de agente,
  Obsidian e ferramentas.

**Conclusao:** nenhuma regressao semantica ativa atribuivel a PR #102 foi
encontrada. A PR #102 passa a ser tratada como auditada no estado atual.

## Codigo historico substituido

Nao foram restaurados nem validados como interfaces atuais:

- `src/interface/` (Streamlit);
- `src/ml/rul_weibull.py` e `src/ml/graficos_rul.py`;
- `src/ml/experimentos_artigos.py` e `src/ml/graficos_experimentos.py`;
- demais pipelines E2/Paderborn removidos em campanhas posteriores.

Esses caminhos nao sao fonte da publicacao vigente. A comparacao GPVS-Faults
Denso versus AE-LSTM e a confiabilidade bibliografica possuem contratos
independentes. Restaurar codigo superado criaria uma segunda arquitetura
canonica e contrariaria a especificacao ativa.

## Resultado

O alerta de auditoria incompleta do commit `35f36b2` esta encerrado para o
codigo vigente. Nenhum artefato cientifico precisou ser regenerado, porque as
PRs avaliadas nao alteram hoje os calculos, as particoes, os escores ou as
metricas das duas publicacoes canonicas.

## Limitacoes

- A conclusao vale para o estado atual e seus contratos testados; nao certifica
  artefatos historicos ja removidos.
- Mudancas futuras nas fachadas devem manter os testes de compatibilidade ou
  registrar explicitamente uma quebra de API.

