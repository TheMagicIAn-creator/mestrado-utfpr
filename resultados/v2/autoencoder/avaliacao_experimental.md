# Avaliação experimental do autoencoder denso V2

Gerado em `2026-08-13T21:50:02.278695-03:00` com o GPVS-Faults (DOI 10.17632/n76t439f65.1).

## Resultado principal

O autoencoder obteve AUC-ROC macro 0.778 (IC95% 0.695-0.859), sensibilidade 0.455 e especificidade 0.953 no limiar congelado.
O baseline PCA obteve AUC-ROC macro 0.789, sensibilidade 0.416 e especificidade 0.984.
Na comparação pareada, a diferença AE-PCA foi -0.011 para AUC-ROC (IC95% -0.022 a -0.002) e +0.004 para acurácia balanceada. O PCA teve AUC e especificidade maiores; o autoencoder teve sensibilidade maior. Para acurácia balanceada e MCC, os intervalos incluem zero, sem superioridade global.

## Protocolo

- Arquitetura, semente, scaler e limiar foram congelados usando apenas F0L/F0M.
- F1L-F7M foram abertos somente após o congelamento do detector.
- A fonte situa a introdução manual da falha na metade do registro.
- A linha de 50% é nominal; os CSVs não contêm um canal instrumentado de disparo.
- A primeira metade do trecho anterior à fronteira ajusta apenas o baseline local.
- A segunda metade anterior mede especificidade; o trecho posterior mede sensibilidade.
- IC95% macro: bootstrap dos 14 ensaios, que são a unidade de inferência.

## Interpretação

O GPVS valida detecção de mudanças de regime em bancada. Ele não contém tempos até falha, censura ou reparo e, portanto, não estima confiabilidade física, taxa de falha, Weibull temporal ou RUL.
Somente F1 representa diretamente falha total em IGBT. Os demais modos não devem ser renomeados como falha de contator ou fusível da FMECA.
