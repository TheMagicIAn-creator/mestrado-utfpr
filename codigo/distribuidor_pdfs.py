import shutil
from pathlib import Path
from pypdf import PdfReader

PASTA_PDFS = r"C:\Users\Rodolfo Torres\Documents\Renomeados_v2"
PASTA_LITERATURA = r"C:\Users\Rodolfo Torres\Documents\mestrado-utfpr\literatura"

MAPA_TEMAS = {
    "ml-preditivo":   "literatura/ml-preditivo",
    "inversores-pv":  "literatura/inversores-pv",
    "manutencao":     "literatura/manutencao",
    "confiabilidade": "literatura/confiabilidade",
    "sinais-ca":      "literatura/sinais-ca",
}


def extrair_texto_pdf(caminho_pdf, num_paginas=2):
    try:
        reader = PdfReader(caminho_pdf)
        texto = ""
        for i in range(min(num_paginas, len(reader.pages))):
            texto += reader.pages[i].extract_text() or ""
        return texto
    except Exception:
        return ""


def classificar_tema(nome_arquivo, texto):
    nome = nome_arquivo.lower()
    texto = texto.lower()

    if any(p in nome or p in texto for p in [
        "machine learning", "deep learning", "random forest",
        "neural", "lstm"
    ]):
        return "ml-preditivo"

    if any(p in nome or p in texto for p in [
        "inverter", "inversor", "pv", "photovoltaic", "solar"
    ]):
        return "inversores-pv"

    if any(p in nome or p in texto for p in [
        "reliability centered", "fmea", "fmeca",
        "failure mode", "preventive maintenance"
    ]):
        return "manutencao"

    if any(p in nome or p in texto for p in [
        "reliability", "mtbf", "failure rate", "taxa de falha", "risk"
    ]):
        return "confiabilidade"

    if any(p in nome or p in texto for p in [
        "signal", "sinal", "harmonic", "harmonico",
        "thd", "current", "corrente", "voltage",
        "tensao", "fft", "spectrum", "power quality"
    ]):
        return "sinais-ca"

    return "ml-preditivo"


def distribuir_pdfs():
    pdfs = list(Path(PASTA_PDFS).glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError("Nenhum PDF encontrado em: " + PASTA_PDFS)

    log = []
    log.append("=" * 55)
    log.append("DISTRIBUIDOR DE PDFs")
    log.append("Total de PDFs: " + str(len(pdfs)))
    log.append("=" * 55)
    log.append("")

    sucesso = 0
    falha = 0
    pulado = 0

    for i, pdf in enumerate(pdfs, 1):
        log.append("[" + str(i) + "/" + str(len(pdfs)) + "] " + pdf.name)

        try:
            texto = extrair_texto_pdf(str(pdf))
            tema_key = classificar_tema(pdf.name, texto)

            pasta_destino = Path(PASTA_LITERATURA) / tema_key
            pasta_destino.mkdir(parents=True, exist_ok=True)

            destino = pasta_destino / pdf.name

            if destino.exists():
                log.append("  Ja existe: " + pdf.name)
                log.append("  Status   : PULADO\n")
                pulado += 1
                continue

            shutil.copy2(str(pdf), str(destino))

            log.append("  Tema  : " + tema_key)
            log.append("  Dest  : " + str(pasta_destino))
            log.append("  Status: OK\n")
            sucesso += 1

        except Exception as e:
            log.append("  Status: ERRO - " + str(e) + "\n")
            falha += 1

    log.append("=" * 55)
    log.append("RESUMO FINAL")
    log.append("Distribuidos com sucesso : " + str(sucesso))
    log.append("Pulados                  : " + str(pulado))
    log.append("Com erro                 : " + str(falha))
    log.append("=" * 55)

    caminho_log = Path(PASTA_PDFS) / "log_distribuicao.txt"
    with open(str(caminho_log), "w", encoding="utf-8") as f:
        f.write("\n".join(log))


if __name__ == "__main__":
    distribuir_pdfs()