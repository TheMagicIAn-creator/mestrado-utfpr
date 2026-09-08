# Detectabilidade por magnitude (E2)

## Escopo

A varredura injeta severidade crescente nos três itens da FMECA vigente sobre as janelas saudáveis de holdout F0 e registra `a_det`, o menor `a` em que cada modelo congelado confirma a detecção. `a` é fração da assinatura nominal, não tempo: `a_det`, a POD e a Weibull descrevem a dispersão da MAGNITUDE de detecção, nunca confiabilidade, taxa de falha ou RUL. O detector é o mesmo congelado na etapa `comparacao` e avaliado na E3; a campanha não treina nem recalibra.

## Método por item

| Item | Escopo FMECA | Ensaios reais | Método | Simulação física |
|---|---|---|---|---|
| IGBT | igbt | F1L, F1M | electrical_signature | sim |
| Sistema de sensor/realimentação | sensor_feedback_system | F2L, F2M | electrical_signature | sim |
| Sistema/circuito de controle do inversor | inverter_control_system | F6L, F6M, F7L, F7M | measured_state_interpolation | não |

## a_det empírico e âncora

| Item | Modelo | Fração detectada | a10 | a50 | Weibull adotada | Âncora (IQR) |
|---|---|---:|---:|---:|---|---:|
| igbt | ae_denso | 1.00 | 0.080 | 0.120 | não | 34.63 |
| igbt | ae_lstm | 1.00 | 0.140 | 0.220 | não | 34.63 |
| sensor_realimentacao | ae_denso | 0.91 | 0.580 | 0.780 | não | 15.06 |
| sensor_realimentacao | ae_lstm | 0.06 | — | — | sim | 15.06 |
| controle | ae_denso | 1.00 | 0.020 | 0.560 | não | 2.22 |
| controle | ae_lstm | 0.60 | 0.020 | 0.760 | não | 2.22 |

## Leitura da âncora

Em `a=1` a distância entre a janela injetada e o ensaio real sai em unidades do IQR saudável. Nas assinaturas elétricas (IGBT, sensor), distância grande diz que a injeção é caricatura da falha medida — é informação, não defeito. Na interpolação de controle, `a=1` É um estado medido, então a distância próxima de zero é esperada e confirma que a injeção não simula física.

Trajetórias por item: 281. Confirmação: 3 pontos consecutivos.
