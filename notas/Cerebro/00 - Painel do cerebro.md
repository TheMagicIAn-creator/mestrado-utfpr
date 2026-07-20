---
al_iado: true
titulo: "Painel do cérebro Al IAdo PV"
tipo: contexto
status: ativo
confianca: alta
nivel_evidencia: projeto
tags: [al-iado, cerebro, moc, mestrado]
---

# Painel do cérebro Al IAdo PV

Este é o mapa de conteúdo curado que conecta decisões, conceitos e memórias do projeto. A presença de uma nota neste mapa não a transforma em evidência científica: artigos são citados a partir dos PDFs indexados e resultados são lidos dos artefatos atuais do pipeline.

## Fundamentos

- [[Níveis de evidência]]
- [[Separação dos domínios de dados]]
- [[Arquitetura Gemini e Groq]]

## Memória auditável

- As memórias aprovadas pelo Groq aparecem em `Memorias validadas/`.
- O JSON `notas/memorias/agentes/memoria_validada.json` permanece como fonte de verdade.
- Uma memória superada continua auditável, mas deixa de entrar no contexto do agente.

## Regra de precedência

1. Artefato recalculado e manifesto vigente.
2. Evidência primária recuperada dos PDFs.
3. Memória validada e nota curada do projeto.
4. Sessão conversacional antiga.

Em caso de conflito, prevalece a camada superior e a divergência deve ser explicitada.
