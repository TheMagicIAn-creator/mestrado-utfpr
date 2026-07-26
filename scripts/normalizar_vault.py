"""normalizar_vault.py — Al IAdo PV / manutencao do vault Obsidian.

Garante, em TODA nota de `notas/`:
  1. frontmatter YAML valido (repara o corrompido; cria o ausente);
  2. pelo menos UMA tag — estrutural pela pasta (cerebro/literatura/sessao/
     memoria) mais tags de TOPICO inferidas do conteudo;
  3. sinalizacao de metadado bibliografico problematico em `Literatura/`
     (autor-invalido / ano-invalido / titulo-invalido) via `revisar: true`
     + `motivo_revisao`, SEM inventar valor — a correcao real exige o PDF.

As tags de topico sao os pontos principais da dissertacao (ver TOPICOS): fmea,
fmeca, rcm, manutencao, confiabilidade, weibull-rul, inversor-pv, contator-ac,
igbt, fusivel-ac, autoencoder, deteccao-anomalia, machine-learning,
sinais-eletricos, paderborn, escore-localizado, evidencia-e2.

Uso:
  python scripts/normalizar_vault.py            # SIMULA (nao escreve)
  python scripts/normalizar_vault.py --aplicar  # aplica

Idempotente: rodar de novo em vault ja normalizado nao altera nada.
Autor: Rodolfo Torres (UTFPR)
"""
from pathlib import Path
import re, sys, yaml, collections

RAIZ = Path("notas")

# ── Taxonomia: os pontos PRINCIPAIS da dissertacao (tag <- palavras-chave) ──
TOPICOS = {
    "fmeca":            [r"\bfmeca\b", r"criticidade.{0,20}fmea", r"\bnpr\b"],
    "fmea":             [r"\bfmea\b", r"modo.{0,3}de.{0,3}falha", r"failure mode"],
    "rcm":              [r"\brcm\b", r"reliability.centred", r"reliability.centered",
                         r"manuten..o centrada"],
    "manutencao":       [r"manuten..o", r"maintenance", r"mantenabilidade",
                         r"preditiv", r"predictive maintenance"],
    "confiabilidade":   [r"confiabilidad", r"reliability", r"disponibilidade",
                         r"\bmtbf\b", r"\bmttf\b"],
    "weibull-rul":      [r"weibull", r"\brul\b", r"vida .til", r"remaining useful",
                         r"kaplan.meier", r"censur"],
    "inversor-pv":      [r"inversor", r"inverter", r"fotovoltaic", r"photovoltaic",
                         r"\bpv\b", r"solar"],
    "contator-ac":      [r"contator", r"contactor"],
    "igbt":             [r"\bigbt\b"],
    "fusivel-ac":       [r"fus.vel", r"\bfuse\b"],
    "autoencoder":      [r"autoencoder", r"auto.encoder", r"\bae-lstm\b"],
    "deteccao-anomalia":[r"anomal", r"detec..o de falha", r"fault detection",
                         r"outlier"],
    "machine-learning": [r"machine learning", r"aprendizado de m.quina",
                         r"random forest", r"\bxgboost\b", r"\bsvm\b",
                         r"rede neural", r"neural network", r"\blstm\b"],
    "sinais-eletricos": [r"\bthd\b", r"harm.nic", r"\bfft\b", r"espectr",
                         r"processamento de sinal", r"signal processing",
                         r"\brms\b", r"forma de onda"],
    "paderborn":        [r"paderborn", r"inverter_data_set", r"stender"],
    "escore-localizado":[r"escore localizado", r"top-k", r"res.duo padronizado"],
    "evidencia-e2":     [r"\be2\b.{0,30}sint", r"inje..o sint", r"ground truth"],
}
COMPILADO = {tag: [re.compile(p, re.I) for p in pats] for tag, pats in TOPICOS.items()}

# tags estruturais por pasta (toda nota tem PELO MENOS uma)
def tag_estrutural(f: Path) -> str:
    s = str(f)
    if "sessoes_arquivadas" in s: return "sessao-arquivada"
    if "/sessoes" in s:           return "sessao"
    if "memorias" in s:           return "memoria"
    if "Literatura" in s:         return "literatura"
    if "Cerebro" in s:            return "cerebro"
    return "nota"

RUIDO = {"al-iado-pv", "al-iado", "mestrado", "mestrado-utfpr", "sessao-web",
         "streamlit", "obsidian", "-", "data"}


def topicos_do_texto(txt: str, limite: int = 6) -> list[str]:
    achados = []
    for tag, pats in COMPILADO.items():
        n = sum(len(p.findall(txt)) for p in pats)
        if n:
            achados.append((n, tag))
    achados.sort(reverse=True)
    return [t for _, t in achados[:limite]]


def analisar_metadado(d: dict) -> list[str]:
    """Sinaliza problemas de metadado bibliografico (sem inventar valor)."""
    marcas = []
    autor = str(d.get("autor", "") or "").strip()
    ano = str(d.get("ano", "") or "").strip()
    titulo = str(d.get("titulo", "") or "").strip()
    if re.search(r"desconhecid|unknown|^n/?a$", autor, re.I) or (autor and len(autor) < 4):
        marcas.append("autor-invalido")
    if ano and not re.fullmatch(r"(19|20)\d{2}", ano):
        marcas.append("ano-invalido")
    if titulo and (len(titulo) < 12 or re.match(
            r"^(therefore|and |the |of |in |for |prof|www|universidade federal|"
            r"vice president|library)", titulo, re.I)):
        marcas.append("titulo-invalido")
    return marcas


def montar_frontmatter(d: dict) -> str:
    ordem = ["titulo", "autor", "ano", "tema", "tipo", "status", "confianca",
             "nivel_evidencia", "al_iado", "revisar", "motivo_revisao", "tags"]
    linhas = []
    for k in ordem:
        if k not in d or d[k] in (None, ""):
            continue
        v = d[k]
        if k == "tags":
            linhas.append(f"tags: [{', '.join(v)}]")
        elif isinstance(v, bool):
            linhas.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, str) and (":" in v or '"' in v or "#" in v):
            linhas.append(f'{k}: "{v.replace(chr(34), chr(39))}"')
        else:
            linhas.append(f"{k}: {v}")
    for k, v in d.items():                       # preserva campos extras
        if k in ordem or v in (None, ""):
            continue
        if isinstance(v, (dict, list)):
            continue
        linhas.append(f"{k}: {v}" if not isinstance(v, str) or ":" not in v
                      else f'{k}: "{v}"')
    return "---\n" + "\n".join(linhas) + "\n---\n"


def processar(aplicar: bool):
    stats = collections.Counter()
    for f in sorted(RAIZ.rglob("*.md")):
        if ".obsidian" in str(f) or "Templates" in str(f):
            continue
        txt = f.read_text(encoding="utf-8", errors="ignore")
        d, corpo = {}, txt
        if txt.startswith("---") and txt.count("---") >= 2:
            fim = txt.index("---", 3)
            bruto, corpo = txt[3:fim], txt[fim + 3:].lstrip("\n")
            try:
                carregado = yaml.safe_load(bruto)
                d = carregado if isinstance(carregado, dict) else {}
                if not isinstance(carregado, dict):
                    stats["frontmatter reconstruido"] += 1
            except Exception:
                # frontmatter corrompido → resgata pares chave: valor simples
                stats["frontmatter reparado"] += 1
                for linha in bruto.splitlines():
                    m = re.match(r"^([a-z_]+):\s*(.*)$", linha.strip(), re.I)
                    if m and m.group(2):
                        d[m.group(1)] = m.group(2).strip().strip('"')
        else:
            stats["frontmatter criado"] += 1

        if not d.get("titulo"):
            d["titulo"] = f.stem.replace("_", " ").replace("-", " ")

        # ── tags: estrutural + topicos, sem ruido, sem duplicata ──
        atuais = d.get("tags") or []
        if isinstance(atuais, str):
            atuais = re.findall(r"[A-Za-zÀ-ÿ0-9_\-]+", atuais)
        atuais = [str(t).strip().lstrip("#") for t in atuais if str(t).strip()]
        base = [tag_estrutural(f)]
        novos = topicos_do_texto(f"{d.get('titulo','')} {corpo[:6000]}")
        final, vistos = [], set()
        for t in base + atuais + novos:
            tl = t.lower()
            if tl and tl not in RUIDO and tl not in vistos:
                vistos.add(tl)
                final.append(tl)
        if len(final) > len(atuais) + 1:
            stats["tags de topico adicionadas"] += 1
        d["tags"] = final

        # ── metadado bibliografico problematico ──
        marcas = analisar_metadado(d) if "Literatura" in str(f) else []
        if marcas:
            d["revisar"] = True
            d["motivo_revisao"] = ",".join(marcas)
            for m in marcas:
                if m not in d["tags"]:
                    d["tags"].append(m)
            stats["metadado sinalizado"] += 1
        else:
            d.pop("revisar", None)
            d.pop("motivo_revisao", None)

        novo = montar_frontmatter(d) + "\n" + corpo
        if aplicar and novo != txt:
            f.write_text(novo, encoding="utf-8")
        stats["notas processadas"] += 1
    return stats


if __name__ == "__main__":
    aplicar = "--aplicar" in sys.argv
    st = processar(aplicar)
    print("MODO:", "APLICADO" if aplicar else "SIMULACAO")
    for k, v in st.most_common():
        print(f"  {k:<28}: {v}")
