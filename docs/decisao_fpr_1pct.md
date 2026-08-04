# Decisão técnica — FPR operacional máximo de 1%

## Resumo

O pipeline passa a tratar 1% como uma **restrição empírica verificável** no
bloco de calibração, não como sinônimo informal de p99. O limiar permite no
máximo `floor(n_calib × 1%)` excedências saudáveis. Com 91 janelas de
calibração, isso significa zero excedências e percentil efetivo 100.

A mudança reduz o falso positivo observado de 4/91 para 0/91 na calibração e
de 9/88 para 0/88 no teste temporal usado nesta verificação. Ela também expõe
um custo importante: no ponto operacional mais conservador, o recall das
falhas localizadas cai. A PR deve permanecer em rascunho até esse compromisso
entre alarmes e sensibilidade ser aceito para a dissertação.

## Causa verificada

O código anterior declarava `FP_ALVO=1.0`, mas procurava o limiar em apenas
cinco percentis candidatos (`99,0` a `99,9`) e aceitava o maior mesmo quando o
alvo não era atingido. Além disso, a régua por feature era estimada em cerca de
80% da calibração, um trecho que não representava o regime do teste.

Na execução de referência de 4 de agosto de 2026:

| Evidência | Calibração | Teste temporal |
|---|---:|---:|
| Janelas | 91 | 88 |
| Mediana de F0 | 51,25 Hz | 100,19 Hz |
| Mediana do escore localizado | 1,384 | 4,266 |
| Falsos positivos | 4 (4,40%) | 9 (10,23%) |

A mediana de F0 do teste está aproximadamente 32 IQRs da mediana da
calibração. Isso caracteriza deslocamento de regime operacional, não simples
flutuação de cauda. O MSE era mais estável, mas sua resolução observada também
era 1/91 e 1/88 — acima de 1% quando havia uma única excedência.

## Mudança

1. A régua do escore localizado (`mu`/`sigma` do resíduo por feature) passa a
   ser estimada no bloco de treino saudável, que contém regimes de F0 baixo e
   alto.
2. Toda a calibração fixa o corte por ordem estatística e também continua
   guiando o early stopping. Nenhuma pontuação do teste entra no cálculo.
3. `limiar.json` registra alvo, orçamento e número observado de excedências,
   resolução amostral, fonte da régua e se a restrição foi satisfeita.
4. Injeção, validação, RUL e auditoria propagam e conferem a mesma política.
5. O resumo de F0 por bloco e o alerta de drift ficam no artefato para evitar
   que mudanças de regime sejam novamente confundidas com precisão do limiar.

O override `AL_IADO_ESCORE_PERCENTIL` continua disponível apenas para
reprodução de rodadas antigas. Sem esse override, a política padrão é
`fpr_empirico_maximo` com `AL_IADO_ESCORE_FP_ALVO=1.0`.

## Verificação e compromisso de desempenho

A execução limpa do autoencoder com o dataset local produziu:

| Métrica | Antes | Política FPR ≤1% |
|---|---:|---:|
| FPR na calibração | 4,40% (4/91) | 0,00% (0/91) |
| FPR no teste do autoencoder | 10,23% (9/88) | 0,00% (0/88) |
| FPR saudável na validação E2 | 12,50% (5/40) | 0,00% (0/40) |
| Recall contator, severidade 1,0 | 1,000 | 0,825 |
| Recall IGBT, severidade 1,0 | 0,850 | 0,025 |
| Recall fusível, severidade 1,0 | 1,000 | 0,025 |

A AUC em severidade 1,0 permaneceu alta (0,996 para contator, 0,916 para IGBT
e 0,936 para fusível), mostrando que a capacidade de ordenação não desapareceu.
O que mudou foi o ponto operacional: exigir praticamente zero alarmes numa
amostra pequena coloca o corte numa região extrema e reduz o recall.

## Limitação estatística

Com 91 janelas, a menor taxa empírica não nula é `1/91 = 1,10%`; com 88,
`1/88 = 1,14%`. Logo, “no máximo 1%” equivale a observar zero falsos positivos.
Mesmo com zero eventos, o limite superior do IC95% de Wilson é aproximadamente
4,05% na calibração e 4,18% no teste. O resultado satisfaz a restrição
**observada**, mas não certifica uma taxa de campo de 1%.

Para demonstrar 1% com precisão útil sem sacrificar tanto o recall, será
necessário coletar um conjunto saudável maior e representativo de cada regime
de F0. Ajustar o corte no teste atual ou omitir janelas difíceis produziria uma
estimativa otimista e não é permitido pelo pipeline.

## Como reproduzir

Após atualizar o código, executar na ordem:

```powershell
python src/ml/features_ca.py
python src/ml/autoencoder.py
python src/ml/injecao_falhas.py
python src/ml/validacao.py
python src/ml/rul_weibull.py
python scripts/verificar_resultados_fmeca.py
```

Os resultados regenerados devem ser revisados antes de commit, principalmente
`limiar.json`, `validacao_report.json`, as SMDs e a censura do RUL.
