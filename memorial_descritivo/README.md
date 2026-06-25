# Memorial Descritivo Elétrico — Modelo Responsivo

Modelo **enxuto e corporativo** de memorial descritivo para **projetos
elétricos residenciais**, com **base de cálculo responsiva**: você edita os
dados do projeto em **um único arquivo** e o documento é recalculado e
reemitido em **`.docx` e `.pdf`** — previsão de cargas, divisão de circuitos,
condutores, proteção, queda de tensão, demanda, ramal de entrada, aterramento
e SPDA.

Layout com **logo no cabeçalho de todas as páginas** e **marca d'água de
fundo** (logo substituível). Embasamento nas **ABNT NBR 5410** (instalações de
baixa tensão), **NBR 5419** (SPDA) e correlatas.

---

## Como usar

```bash
pip install -r requirements.txt
python gerar_memorial.py
```

Saída em `saida/`:
- `Memorial_Descritivo_Eletrico.docx`
- `Memorial_Descritivo_Eletrico.pdf`

> O PDF é gerado **direto** (biblioteca `reportlab`), sem precisar de Word ou
> LibreOffice instalados.

---

## O que editar — `dados_projeto.py` (a "base de cálculo")

Este é o único arquivo que você precisa mexer. Tudo nele é **entrada**:

| Bloco | O que controla |
|-------|----------------|
| `EMPRESA` | Nome, contato, **logo** (`assets/logo.png`), cor corporativa, intensidade da marca d'água |
| `PROJETO` | Identificação da obra, cliente, código do documento, responsável técnico, ART |
| `INSTALACAO` | Tensões (127/220 V), sistema (mono/bi/trifásico), esquema de aterramento, fator de potência |
| `AMBIENTES` | Área, perímetro e categoria (seca/úmida) de cada cômodo → **carga calculada pela NBR** |
| `CIRCUITOS_*` | Divisão de circuitos (iluminação, TUG, TUE), comprimentos, método de instalação, DR |
| `DEMANDA` | Fatores de demanda (ajuste pela norma da **concessionária local**) |
| `SPDA` | Dimensões da edificação, densidade de descargas `Ng`, fator `Cd` |

Troque a logo: substitua `assets/logo.png` pelo PNG da empresa (de preferência
com fundo transparente) e rode de novo — a marca d'água é re-derivada
automaticamente. Se não houver logo, o script cria uma de exemplo.

---

## Estrutura

```
memorial_descritivo/
├── dados_projeto.py   ← ENTRADAS (edite aqui)
├── calculos_nbr.py    ← motor de cálculo (NBR 5410 / 5419)
├── conteudo.py        ← texto e estrutura do documento (fonte única)
├── render_docx.py     ← renderiza o .docx (python-docx)
├── render_pdf.py      ← renderiza o .pdf (reportlab)
├── marca.py           ← logo de exemplo + marca d'água
├── gerar_memorial.py  ← ponto de entrada (roda tudo)
├── assets/            ← logo.png (substituível)
└── saida/             ← .docx e .pdf gerados
```

O conteúdo é montado **uma vez** em `conteudo.py` e consumido pelos dois
renderizadores, de modo que o DOCX e o PDF **nunca divergem**.

---

## Base normativa do dimensionamento

| Etapa | Cláusula NBR 5410:2004 |
|-------|------------------------|
| Carga de iluminação | 9.5.2.1 (100 VA até 6 m²; +60 VA/4 m²) |
| Tomadas (qtd. e potência) | 9.5.2.2 |
| Seção mínima dos condutores | 6.2.6.1.1 (1,5 mm² iluminação; 2,5 mm² tomadas) |
| Ampacidade | Tabela 36 (Cu/PVC), com FCT e FCA |
| Coordenação proteção × condutor | 5.3.4 (IB ≤ IN ≤ IZ) |
| Condutor de proteção PE | Tabela 58 |
| Queda de tensão | 6.2.7 (4 % terminal; 7 % global) |
| Proteção DR (30 mA) | 5.1.3.2 |
| DPS | 6.3.5 |
| Aterramento | 6.4 |
| SPDA | ABNT NBR 5419-1/-2 (estimativa de `Nd`; risco `R1` na -2) |

> **Atenção (responsabilidade técnica):** este é um **modelo**. Os fatores de
> demanda e o padrão de entrada dependem da **concessionária local**; a decisão
> de SPDA exige o **gerenciamento de risco completo da NBR 5419-2**. Revise e
> assuma o conteúdo antes de emitir como peça de projeto (CREA/ART).
