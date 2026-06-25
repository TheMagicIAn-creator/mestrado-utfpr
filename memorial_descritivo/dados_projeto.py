# -*- coding: utf-8 -*-
"""
============================================================================
 BASE DE CÁLCULO RESPONSIVA  —  Memorial Descritivo de Instalação Elétrica
============================================================================

Este é o ÚNICO arquivo que você precisa editar.

Tudo o que está aqui são as ENTRADAS do projeto. Ao alterar qualquer valor
(área de um cômodo, comprimento de um circuito, potência de um chuveiro,
fator de demanda, dados do SPDA, identificação da obra, logotipo...) e
rodar de novo `gerar_memorial.py`, o memorial inteiro é RECALCULADO e
reemitido em .docx e .pdf — previsão de cargas, divisão de circuitos,
seção dos condutores, disjuntores, DR, queda de tensão, demanda, ramal de
entrada, aterramento e SPDA. Nada é digitado "na mão" no documento.

Base normativa: ABNT NBR 5410:2004 (instalações de BT), ABNT NBR 5419:2015
(SPDA), ABNT NBR IEC 60898 (disjuntores) e normas correlatas.
============================================================================
"""

# ───────────────────────────────────────────────────────────────────────────
# 1) IDENTIFICAÇÃO / CABEÇALHO CORPORATIVO
# ───────────────────────────────────────────────────────────────────────────
EMPRESA = {
    "nome":        "NOME DA EMPRESA LTDA.",
    "slogan":      "Engenharia Elétrica · Projetos e Consultoria",
    "cnpj":        "00.000.000/0001-00",
    "contato":     "contato@empresa.com.br  ·  (00) 0000-0000",
    "cidade_uf":   "Cidade/UF",
    # Logotipo usado no cabeçalho e como marca d'água de fundo.
    # Troque por um PNG (de preferência com fundo transparente). Se o arquivo
    # não existir, o gerador cria automaticamente uma logo de exemplo.
    "logo":        "assets/logo.png",
    # Intensidade da marca d'água de fundo (0 = invisível, 1 = logo cheia).
    "marca_dagua_intensidade": 0.10,
    # Cor de destaque corporativa (hex, sem '#').
    "cor_destaque": "1F3864",   # azul corporativo
}

PROJETO = {
    "titulo":        "MEMORIAL DESCRITIVO – PROJETO ELÉTRICO RESIDENCIAL",
    "codigo_doc":    "MD-ELE-001",
    "revisao":       "00",
    "data":          "25/06/2026",
    "obra":          "Residência Unifamiliar",
    "endereco":      "Rua Exemplo, nº 000 – Bairro – Cidade/UF – CEP 00000-000",
    "cliente":       "NOME DO PROPRIETÁRIO / CONTRATANTE",
    "area_construida_m2": 96.0,           # apenas informativo na capa
    "responsavel":   "Eng. Eletricista Fulano de Tal",
    "titulo_prof":   "Engenheiro Eletricista",
    "crea":          "CREA-UF 000000000-0",
    "art":           "ART nº 0000000000000",
}

# ───────────────────────────────────────────────────────────────────────────
# 2) CARACTERÍSTICAS GERAIS DA INSTALAÇÃO
# ───────────────────────────────────────────────────────────────────────────
INSTALACAO = {
    "concessionaria":  "Concessionária local (ex.: COPEL / CEMIG / CPFL / Enel)",
    "fornecimento":    "Bifásico (2F+N) – 127/220 V",   # texto exibido
    "sistema":         "bifasico",   # 'monofasico' | 'bifasico' | 'trifasico'
    "tensao_fn":       127.0,        # tensão fase-neutro (V)
    "tensao_ff":       220.0,        # tensão fase-fase (V)
    "frequencia_hz":   60,
    "esquema_aterramento": "TN-S",   # TN-S | TN-C-S | TT
    "fator_potencia":  0.92,         # fp médio de referência p/ corrente
    "temperatura_ambiente_C": 30,    # referência das tabelas da NBR 5410
}

# ───────────────────────────────────────────────────────────────────────────
# 3) AMBIENTES (a previsão de carga é calculada automaticamente pela NBR 5410)
#    - 'area'      : área em m²            (define a carga de iluminação)
#    - 'perimetro' : perímetro em m        (define a quantidade de tomadas)
#    - 'categoria' : 'seca' ou 'umida'     (define a potência por tomada e DR)
#         'umida'  = cozinha, copa, área de serviço, lavanderia, banheiro,
#                    varanda e áreas externas (600 VA nas 3 primeiras + 100 VA;
#                    exige DR 30 mA nas tomadas — NBR 5410 9.5.2.2 / 5.1.3.2.5)
#         'seca'   = salas, dormitórios, halls, garagem (100 VA por tomada)
#    - 'min_tomadas': piso mínimo de tomadas no ambiente (opcional)
# ───────────────────────────────────────────────────────────────────────────
AMBIENTES = [
    {"nome": "Sala de estar/jantar", "area": 18.0, "perimetro": 17.0, "categoria": "seca"},
    {"nome": "Cozinha",              "area": 10.0, "perimetro": 13.0, "categoria": "umida"},
    {"nome": "Área de serviço",      "area":  4.0, "perimetro":  8.0, "categoria": "umida"},
    {"nome": "Dormitório 1 (suíte)", "area": 14.0, "perimetro": 15.0, "categoria": "seca"},
    {"nome": "Dormitório 2",         "area": 11.0, "perimetro": 13.5, "categoria": "seca"},
    {"nome": "Banheiro social",      "area":  4.0, "perimetro":  8.0, "categoria": "umida"},
    {"nome": "Banheiro da suíte",    "area":  3.5, "perimetro":  7.5, "categoria": "umida"},
    {"nome": "Circulação / Hall",    "area":  5.0, "perimetro":  9.0, "categoria": "seca"},
    {"nome": "Varanda",              "area":  6.0, "perimetro": 10.0, "categoria": "umida"},
    {"nome": "Garagem",              "area": 15.0, "perimetro": 16.0, "categoria": "seca"},
]

# ───────────────────────────────────────────────────────────────────────────
# 4) CIRCUITOS TERMINAIS
#    Os circuitos de ILUMINAÇÃO e TOMADAS (TUG) referenciam os ambientes pelo
#    nome — a potência do circuito é SOMADA automaticamente da previsão de
#    carga calculada para cada ambiente. Os de uso específico (TUE) recebem a
#    potência nominal do equipamento.
#
#    Campos comuns:
#      'id'         : identificador (C1, C2, ...)
#      'descricao'  : texto no quadro de cargas
#      'sistema'    : 'FN' (fase-neutro 127 V) | 'FF' (fase-fase 220 V) | '3F'
#      'comprimento': comprimento estimado do circuito, em metros (ida)
#      'metodo'     : método de referência da NBR 5410 (B1, B2, C ou D)
#      'fct'/'fca'  : fatores de correção de temperatura e de agrupamento
#      'dr_30ma'    : True se exige proteção diferencial de 30 mA
# ───────────────────────────────────────────────────────────────────────────
CIRCUITOS_ILUMINACAO = [
    {"id": "C1", "descricao": "Iluminação – setor social",
     "ambientes": ["Sala de estar/jantar", "Circulação / Hall", "Varanda", "Garagem"],
     "sistema": "FN", "comprimento": 22, "metodo": "B1", "fct": 1.0, "fca": 1.0,
     "dr_30ma": False},
    {"id": "C2", "descricao": "Iluminação – setor íntimo/serviço",
     "ambientes": ["Cozinha", "Área de serviço", "Dormitório 1 (suíte)",
                   "Dormitório 2", "Banheiro social", "Banheiro da suíte"],
     "sistema": "FN", "comprimento": 20, "metodo": "B1", "fct": 1.0, "fca": 1.0,
     "dr_30ma": False},
]

CIRCUITOS_TUG = [
    {"id": "C3", "descricao": "Tomadas – salas e dormitórios",
     "ambientes": ["Sala de estar/jantar", "Dormitório 1 (suíte)",
                   "Dormitório 2", "Circulação / Hall"],
     "sistema": "FN", "comprimento": 24, "metodo": "B1", "fct": 1.0, "fca": 1.0,
     "dr_30ma": False},
    {"id": "C4", "descricao": "Tomadas – cozinha (bancada/copa)",
     "ambientes": ["Cozinha"],
     "sistema": "FN", "comprimento": 14, "metodo": "B1", "fct": 1.0, "fca": 1.0,
     "dr_30ma": True},
    {"id": "C5", "descricao": "Tomadas – área de serviço e varanda",
     "ambientes": ["Área de serviço", "Varanda"],
     "sistema": "FN", "comprimento": 16, "metodo": "B1", "fct": 1.0, "fca": 1.0,
     "dr_30ma": True},
    {"id": "C6", "descricao": "Tomadas – banheiros e garagem",
     "ambientes": ["Banheiro social", "Banheiro da suíte", "Garagem"],
     "sistema": "FN", "comprimento": 18, "metodo": "B1", "fct": 1.0, "fca": 1.0,
     "dr_30ma": True},
]

CIRCUITOS_TUE = [
    {"id": "C7", "descricao": "Chuveiro elétrico – banheiro social",
     "potencia_W": 5500, "fp": 1.00, "sistema": "FF", "comprimento": 12,
     "metodo": "B1", "fct": 1.0, "fca": 1.0, "dr_30ma": True},
    {"id": "C8", "descricao": "Chuveiro elétrico – suíte",
     "potencia_W": 5500, "fp": 1.00, "sistema": "FF", "comprimento": 14,
     "metodo": "B1", "fct": 1.0, "fca": 1.0, "dr_30ma": True},
    {"id": "C9", "descricao": "Torneira elétrica – cozinha",
     "potencia_W": 4500, "fp": 1.00, "sistema": "FF", "comprimento": 13,
     "metodo": "B1", "fct": 1.0, "fca": 1.0, "dr_30ma": True},
    {"id": "C10", "descricao": "Ar-condicionado – dormitório 1 (12.000 BTU)",
     "potencia_W": 1400, "fp": 0.95, "sistema": "FF", "comprimento": 15,
     "metodo": "B1", "fct": 1.0, "fca": 1.0, "dr_30ma": False},
]

# ───────────────────────────────────────────────────────────────────────────
# 5) DEMANDA E RAMAL DE ENTRADA
#    Fatores de demanda EXEMPLIFICATIVOS. Substitua pelos valores da norma de
#    fornecimento da concessionária local (ex.: COPEL NTC 901100, CEMIG ND-5.1,
#    CPFL/Enel). O gerador aplica estes fatores sobre as cargas calculadas.
# ───────────────────────────────────────────────────────────────────────────
DEMANDA = {
    "fd_iluminacao_tug": 0.65,   # fator de demanda p/ iluminação + TUG
    "fd_chuveiro":       0.75,   # fator p/ chuveiros/aquecedores (>1 unidade)
    "fd_outros_tue":     1.00,   # fator p/ demais TUE (ar-cond., etc.)
    "metodo_ramal":      "B1",   # método de referência do ramal de entrada
    "fct_ramal":         1.0,
    "fca_ramal":         1.0,
}

# ───────────────────────────────────────────────────────────────────────────
# 6) SPDA — PROTEÇÃO CONTRA DESCARGAS ATMOSFÉRICAS (NBR 5419:2015)
#    Estimativa simplificada da frequência de eventos perigosos (Nd). A decisão
#    definitiva deve vir do gerenciamento de risco completo da NBR 5419-2
#    (risco R1 ≤ risco tolerável RT = 1×10⁻⁵).
# ───────────────────────────────────────────────────────────────────────────
SPDA = {
    "comprimento_m": 12.0,   # L da edificação
    "largura_m":      8.0,   # W da edificação
    "altura_m":       6.0,   # H da edificação (mais alta)
    "Ng": 6.0,               # densidade de descargas (raios/km²/ano) – mapa NBR 5419-2
    "Cd": 0.5,               # fator de localização (0,25 cercada / 0,5 vizinhança / 1 isolada)
    # Limiar de referência para a frase de recomendação automática.
    # (não substitui o gerenciamento de risco da NBR 5419-2)
    "limiar_Nd": 1.0e-3,
}
