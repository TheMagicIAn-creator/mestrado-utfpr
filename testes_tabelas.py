from pathlib import Path
from src.core.config import PASTA_LITERATURA
from src.conhecimento.indexador import extrair_tabelas_pdf
from src.core.utils import parsear_nome_arquivo

pdfs = list(PASTA_LITERATURA.rglob("*torres*aplicacao*"))
if not pdfs:
    print("TCC não encontrado")
else:
    pdf = pdfs[0]
    print(f"Testando: {pdf.name}\n")
    info    = parsear_nome_arquivo(pdf.name)
    tabelas = extrair_tabelas_pdf(pdf, info)
    print(f"Total de tabelas extraídas: {len(tabelas)}\n")

    for i, t in enumerate(tabelas, 1):
        print(f"=== TABELA {i} ===")
        print(t[:300])
        print()