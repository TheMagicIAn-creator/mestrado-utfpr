# Registro de decisão — o alvo de FPR de 1%

**Decidido por:** Rodolfo Torres · **Origem:** PR #94 (rascunho, não mesclada)
**Status:** **o corte estrito de 1% NÃO é adotado**; o achado que o motivou é
mantido e instrumentado.
**Verificável por:** `python scripts/diagnostico_limiar.py`

---

## O que a PR #94 propunha

Impor `FPR ≤ 1%` no bloco de calibração por ordem estatística: permitir no
máximo `floor(n_calib × 1%)` excedências. Com 91 janelas de calibração, isso é
**zero** excedências.

A PR mediu o efeito e o declarou com honestidade — é por isso que ela ficou em
rascunho em vez de ser mesclada:

| Métrica | Vigente | Sob corte estrito |
|---|---:|---:|
| FPR na calibração | 4,40% (4/91) | 0,00% (0/91) |
| FPR no teste | 10,23% (9/88) | 0,00% (0/88) |
| Recall contator, sev. 1,0 | 1,000 | 0,825 |
| Recall IGBT, sev. 1,0 | 0,850 | **0,025** |
| Recall fusível, sev. 1,0 | 1,000 | **0,025** |

## Por que não foi adotado

**Custo em cascata, não localizado.** Recall de 0,025 não é "menos sensível": é
o detector deixando de detectar. Três consequências que a PR não enumera:

1. **A Weibull colapsa.** O TTF de cada trajetória é o passo em que o escore
   cruza o limiar operacional. Com recall de 0,025, quase nenhuma trajetória
   cruza — censura próxima de 100%, e não há ajuste a fazer.
2. **A retroalimentação da FMECA desaparece.** `POD_mon = 0,025` → 97,5% de não
   detecção → `D_mon = 10` → `min(D_campo, 10) = D_campo` → o NPR projetado
   volta a ser **idêntico** ao oficial. O resultado que fecha o ciclo do RCM
   deixa de existir.
3. **A SMD fica nula** para IGBT e fusível — o pior desfecho possível para o
   capítulo de detecção.

**Troca-se resultado por um ganho que a amostra não consegue medir.** Com 91
janelas, a menor taxa não nula é `1/91 = 1,10%`. Logo "no máximo 1%" só pode
significar "zero eventos observados", e mesmo assim o limite superior do IC95
de Wilson fica em torno de 4%. Isto é o mesmo teto amostral já registrado em
`docs/auditoria_pipeline_ml.md` §22, onde bootstrap, ajuste paramétrico e EVT
foram testados e rejeitados: **o limite é o tamanho da amostra, não o
estimador** — e não é o limiar.

---

## O achado que MOTIVOU a PR, e que é o verdadeiro

O título da #94 fala de FPR. O corpo dela traz outra coisa, mais importante:

| Evidência | Calibração | Teste |
|---|---:|---:|
| Janelas | 91 | 88 |
| Mediana de F0 | 51,25 Hz | 100,19 Hz |
| Mediana do escore localizado | 1,384 | 4,266 |
| Falsos positivos | 4 (4,40%) | 9 (10,23%) |

A mediana de F0 do teste está a dezenas de IQRs da mediana da calibração. **Os
dois blocos não estão no mesmo regime operacional.**

Isso reenquadra o problema inteiro. O FPR de 10% no teste **não é limiar mal
calibrado — é o detector vendo um regime que a calibração não cobriu.** Um
inversor a 100 Hz reconstruído por um modelo calibrado a 51 Hz produz erro
maior, e o escore sobe. Subir o corte até zerar esses alarmes trata um problema
de **cobertura de dados** com um instrumento de **limiar**: por isso o alarme
some junto com a detecção real.

A resposta certa é sobre o dado, não sobre o corte:

- calibrar em bloco que cubra os regimes de F0 presentes no teste;
- ou declarar o escopo do detector como um regime de F0 e avaliá-lo nele;
- ou estratificar a avaliação por regime, reportando FPR e recall em cada um.

Nenhuma dessas é uma mudança de constante, e nenhuma foi feita ainda. Ficam
registradas como caminho, não como pendência silenciosa.

---

## O segundo achado: o alvo declarado não é imposto

`AL_IADO_ESCORE_FP_ALVO = 1.0` **não** restringe o limiar. Ele escolhe entre
cinco percentis candidatos (99,0 a 99,9) o menor cujo FP num sub-bloco fique
abaixo do alvo — e, **se nenhum atinge o alvo, aceita o mais conservador
assim mesmo**. É o que ocorre hoje: p99,9 escolhido, FPR de 10,2% no teste.

O número na configuração não descreve o que o sistema faz. Isso é corrigido por
**transparência, não por coerção**:

- `.env.example` passa a dizer explicitamente que o alvo é uma preferência
  entre candidatos, que pode não ser atingido, e que hoje não é;
- `scripts/diagnostico_limiar.py` reporta `alvo_atingido` e, quando falso,
  avisa em vez de deixar o leitor supor;
- o mesmo script calcula o corte estrito **contrafactual** e o recall que ele
  custaria, para que esta decisão seja reproduzível em vez de lembrada.

---

## O que foi aproveitado da #94

| Elemento | Destino |
|---|---|
| Ordem estatística para FPR máximo | `scripts/diagnostico_limiar.py::limiar_fpr_maximo` — como diagnóstico, não como política |
| Distinção alvo × resolução amostral | preservada e testada; é o melhor argumento da PR |
| Diagnóstico de drift de F0 | `scripts/diagnostico_limiar.py`, com o critério em IQRs |
| Custo do corte estrito | calculado no script, não decorado |
| Propagação de proveniência da política | **não** aplicada: sem política nova, seriam campos descrevendo algo que não existe |
| Régua μ/σ do bloco de treino | **não** aplicada: muda o escore e exige revalidar tudo; ver abaixo |
| Corte estrito como padrão | **não** aplicado |

### Por que a régua do treino também ficou de fora

A #94 move a estimativa de `μ/σ` por feature da calibração para o bloco de
treino. O argumento é bom (o treino cobre mais regimes de F0), mas a mudança
**altera o escore de toda janela** e portanto invalida limiar, SMD, validação,
Weibull e a retroalimentação de uma vez — misturada, na mesma PR, com o corte
que destrói o recall. Não dá para saber qual dos dois causou o quê.

Se for adotada, deve ser sozinha, medida contra os artefatos vigentes, e com a
justificativa correta: **cobertura de regime**, que é o achado real da #94 —
não falso positivo.

---

## Por que o diagnóstico vive em `scripts/` e não em `src/ml/`

`src/ml/escore_anomalia.py` é dependência de proveniência das etapas de
autoencoder, injeção e validação. Acrescentar código lá marcaria as três como
`stale` e pediria retreino — por um diagnóstico que não muda resultado nenhum.
O custo não se justifica; é o mesmo critério aplicado em
`src/ml/graficos_autoencoder.py`.

---

## Referências internas

- `scripts/diagnostico_limiar.py` — a medição, executável
- `docs/auditoria_pipeline_ml.md` §22 — o teto amostral do estimador
- `docs/nomenclatura_deteccao.md` — `POD_mon` e a ponte com a FMECA
- `docs/metodologia_ml.md` §3 — definição do limiar operacional
