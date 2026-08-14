# Confiabilidade física V2 — cenários bibliográficos

## Veredito

O GPVS-Faults sustenta a avaliação experimental do detector, mas não contém tempos de vida por ativo, censura, exposição de frota ou histórico de reparos. Portanto, ele **não estima confiabilidade física, taxa de falha, Weibull temporal, MTTF, MTBF ou RUL**.

As curvas deste pacote são análises de sensibilidade: quantidades de fontes identificadas foram normalizadas dimensionalmente e avaliadas sob o mesmo modelo exponencial de taxa constante. Os cenários diferem em escopo e natureza da evidência e não devem ser tratados como réplicas.

## Cenários normalizados

| Cenário | Quantidade original | λ (ano⁻¹) | 1/λ (anos) | B10 (anos) | Natureza |
|---|---:|---:|---:|---:|---|
| Torres/Colli: taxa transcrita | 0,000175 falha/h | 1,533000 | 0,652 | 0,069 | secondary bibliographic rate |
| Cristaldi: 1 falha em 8 anos | 0,125000 falha/ano | 0,125000 | 8,000 | 0,843 | literature assumption |
| Obeidat: alta qualidade | 8,069000 falhas/10⁶ h | 0,070684 | 14,147 | 1,491 | mil hdbk 217f prediction |
| Obeidat: baixa qualidade | 50,760000 falhas/10⁶ h | 0,444658 | 2,249 | 0,237 | mil hdbk 217f prediction |
| Dhople: exemplo Markov | 10,000000 anos (MTTF de entrada) | 0,100000 | 10,000 | 1,054 | illustrative markov parameter |

A conversão adota `1 ano = 8.760 h`. O valor `1/λ` é MTTF somente sob o modelo não reparável exponencial; quando a fonte usa MTBF ou um modelo reparável, a semântica original permanece registrada no JSON.

## Funções publicadas

Para `t` em anos e `λ` em falhas por ano:

- `R(t) = exp(-λt)`: confiabilidade ou sobrevivência;
- `F(t) = 1 - exp(-λt)`: probabilidade acumulada de falha;
- `f(t) = λ exp(-λt)`: densidade de probabilidade, em ano⁻¹;
- `h(t) = λ`: taxa instantânea de falha constante, em ano⁻¹;
- `B_p = -ln(1-p)/λ`: tempo em que a fração acumulada `p` falhou.

A densidade `f(t)` é uma **curva analítica suave**. Pontos dispersos pertencem a um gráfico de probabilidade construído com tempos de falha observados. Como essa amostra de vida não existe no GPVS, nenhum papel de Weibull físico é produzido nesta etapa.

## Auditoria dimensional

A Tabela 3.4 de Torres (2024) transcreve `λ = 1,750 × 10^-4 falha/h` para o inversor. Na seção posterior de disponibilidade, `1/(1,8 × 10^-4)` é apresentado como `5.555,55 anos`; dimensionalmente, o resultado é `5.555,55 horas`, aproximadamente `0,634 ano`. A V2 não altera a fonte: usa a taxa exata da tabela e registra a inconsistência.

Cristaldi et al. (2017) informam `0,125 falha/ano` para o inversor, mas o MTTF próximo de seis anos citado no artigo é do sistema string-BoS completo. O recíproco da taxa isolada do inversor é oito anos; são escopos diferentes.

Obeidat e Shuttleworth (2015) publicam predições MIL-HDBK-217F N2, não observações de frota. Dhople e Dominguez-Garcia (2012) usam dez anos como parâmetro ilustrativo em um caso Markov reparável.

## Limites de inferência

- Nenhum cenário foi ajustado aos ensaios GPVS-Faults.
- Não há intervalos de confiança porque as fontes não fornecem a amostra primária necessária para reamostragem.
- As curvas não estimam taxas específicas de contator AC, IGBT ou fusível AC.
- A priorização FMECA permanece julgamento de risco separado do detector e dos cenários de confiabilidade.
- Um Weibull físico exigirá tempos de vida/exposição, modo de falha, censura e unidade observacional definidos antes do ajuste.

## Artefatos

- `cenarios.csv`: valores originais, conversões e ressalvas;
- `curvas.csv`: funções amostradas em uma grade temporal comum;
- `marcos.csv`: B1, B10, mediana, 1/λ e probabilidades em 1, 5 e 10 anos;
- `resultado.json`: contrato consolidado para a aplicação web;
- figuras em PNG de 300 dpi e PDF vetorial;
- `manifesto_v2.json`: hashes das fontes, código e saídas.
