# Estado pós-migração para o GPVS — quem mexeu em quê

**Data:** 2026-08-10 · **Autor:** Claude · **Destinatário:** Codex e Rodolfo

Documento de coordenação entre os dois agentes, no formato que
`briefing_code_al_iado.md` estabelece. Serve para não tocarmos a mesma coisa e
para deixar registradas as decisões que **nenhum dos dois deve tomar sozinho**.

---

## 1. O que o Codex fez (PRs #111–#119) — não mexi em nada disso

A migração de Stender → **GPVS-Faults** e a elevação a **E3 de bancada** são
mudanças grandes e bem executadas. Conferi o protocolo em
`resultados/gpvs/validacao_gpvs_e3.json` e ele se sustenta:

- fonte única de dados, sem fusão com Stender/PV Farms;
- pesos e limiar **congelados** em F1–F7 (`adaptation_per_experiment: false`);
- **bootstrap com o ensaio como unidade**, não a janela — o erro clássico
  evitado explicitamente;
- Wilson por cenário marcado como *apenas descritivo*, pela autocorrelação;
- declaração explícita de que não houve seleção de cenários por desempenho;
- limitações escritas dentro do próprio artefato.

Também conferi que o trabalho anterior foi preservado e reparametrizado, não
descartado: `split_blocos_intercalados` (15 → 14 blocos, 60/20/20 → 50/20/30),
`a_det`, `motivo_nao_estimavel`. **Nada disso foi revertido por mim.**

## 2. O que eu corrigi — só coerência documental

Nenhuma destas mudanças toca modelo, limiar, dado ou artefato de resultado.
**Não invalidam manifesto e não exigem rodar o pipeline.**

| # | Arquivo | O quê |
|---|---|---|
| 1 | `CLAUDE.md` | passou a declarar GPVS-Faults como dataset principal, F0L/F0M como treino, F1–F7 como E3 de bancada, Stender como referência histórica, e o MSE médio como escore vigente |
| 2 | `docs/glossario.md` | verbete novo **"Símbolos que colidem"**, separando os dois sentidos de `F0` |
| 3 | `src/ml/features_ca.py` | comentário avisando que ali `F0` é frequência |
| 4 | `notas/Cerebro/Resultados/Validação experimental GPVS-Faults.md` | a nota dizia que **o limiar** era ajustado por ensaio; o artefato diz o contrário |
| 5 | `tests/test_consistencia_docs.py` | quatro guardas — todas verificadas como reprovando na versão anterior |

### Por que o `CLAUDE.md` era urgente

Ele não foi tocado em nenhuma das nove PRs, e **"GPVS" aparecia zero vezes**.
Como o `PERFIL_COMPACTO` do agente deriva dele, o Al IAdo estava respondendo
sobre o Paderborn enquanto o pipeline rodava GPVS. Não é cosmético: é o agente
dando informação errada ao pesquisador a cada consulta.

A guarda `test_claude_md_declara_o_dataset_que_o_pipeline_realmente_usa` existe
para isso não se repetir — ela lê `pipeline.py` e falha se o `CLAUDE.md`
divergir. **Codex: se você migrar o pipeline de novo, essa guarda vai te avisar
antes do merge.**

### A colisão de `F0`, e por que NÃO renomeei

Dois sentidos vivos, ambos legítimos:

| escrita | significa | onde |
|---|---|---|
| `F0`, `F0_MIN`, `F0_MAX` | frequência fundamental (Hz) | `features_ca.py`, `gpvs_principal.py` (`F0 = GRID_FREQUENCY_HZ`) |
| `F0L`, `F0M`, "ensaio F0" | condição **saudável** do GPVS | `gpvs*.py`, `docs/`, `resultados/gpvs/` |

Convivem no mesmo módulo: `gpvs_principal.py` usa `F0` como frequência e
`JANELA = FS/F0`, enquanto lê `F0L.csv`/`F0M.csv`.

**Não renomeei nenhum dos dois de propósito.** `F0` como frequência é convenção
de eletrotécnica; `F0L`/`F0M` são os nomes publicados por Bakdi et al. (2020) —
renomear quebraria a correspondência com a fonte externa. A separação é de
**escrita**, e está no glossário. Se você discordar, é conversa antes de mexer.

---

## 3. ❓ DECISÕES — nenhum de nós deve tomar sozinho

Seguindo a regra 2 do briefing: *"Onde houver ❓ DECISÃO, PARE e PERGUNTE."*

### ❓ D12 — features do lado CC no vetor E3

Das 24 features do protocolo GPVS, **8 são do lado CC**: `Ipv_median`,
`Ipv_iqr`, `Vpv_median`, `Vpv_iqr`, `Vdc_median`, `Vdc_iqr`, `p_dc_median`,
`p_dc_iqr`.

O `CLAUDE.md` declara "Componentes foco: **Lado CA** do inversor", e a PR #107
removeu `tensao_dc_media` exatamente por escopo, com três argumentos: fora do
escopo, inerte sob injeção, e alavanca desproporcional no escore top-k.

**Não encontrei justificativa escrita** para reintroduzir o lado CC. Pode haver
uma boa — as falhas F1/F2 do GPVS são de origem CC, e um detector cego ao CC
não as veria. Mas ela precisa estar no texto, não implícita no código.

Opções: (a) manter e justificar no Capítulo 3, redefinindo o escopo como
"inversor, com ênfase no lado CA"; (b) manter e publicar uma **ablação só-CA**
ao lado, mostrando o que o CC acrescenta; (c) restringir a 16 features CA.

*Minha recomendação: (a) + (b). A ablação responde a pergunta antes que a banca
faça. Mas isso mexe no escopo declarado da dissertação — é decisão do Rodolfo.*

### ❓ D13 — escore localizado rebaixado a ablação

O `docs/glossario.md` passou a registrar que o escore operacional vigente é o
**MSE médio**, com o localizado top-k como "ablação diagnóstica".

O escore localizado não era um detalhe: é o que permite atribuir o desvio a
"harmônico 5 da fase A" e ligar a detecção ao modo de falha da FMECA — o elo
com a RCM, que é o eixo do trabalho. E a comparação publicada em
`resultados/macro/` foi medida **sobre ele**.

Se a troca tem evidência sob o protocolo E3, ótimo — mas ela precisa estar
escrita, e a comparação macro precisa ser refeita sob o escore novo ou
aposentada explicitamente. Deixei uma ressalva no `CLAUDE.md` para o agente não
apresentar a comparação sem dizer sob qual escore ela foi medida.

*Não audito isso sem o Rodolfo mandar — é reversão de método, não bug.*

---

## 4. O que continua com o Rodolfo, independente de nós dois

- **Escopar o plugin Obsidian Git a `notas/`.** Ele ainda commita o repositório
  inteiro; já apagou 24 artefatos em `e5518db`.
- **A sensibilidade de 0,406 no E3.** É o número que a banca vai atacar. Não é
  bug — é achado, e merece investigação própria.

## 5. Verificação desta entrega

```
python3 -m pytest -q -W ignore -m "not pesado" --continue-on-collection-errors
python3 -m ruff check --select F821,F822,F823 src tests scripts
```

Linha de base medida em `66cdf1e` **antes** de qualquer alteração minha:
**902 passando, 2 pulados**.
