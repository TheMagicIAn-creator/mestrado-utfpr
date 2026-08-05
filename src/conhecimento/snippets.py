"""
snippets.py — Al IAdo PV

Cofre de TRECHOS VERBATIM (scripts/código) que o pesquisador manda o agente
guardar. É diferente, por design, da memória validada:

- Memória validada guarda DECISÕES/preferências curadas e REJEITA código.
- Aqui o conteúdo é armazenado LITERALMENTE e recuperado IDÊNTICO, SEM passar
  pelo LLM — para "me manda o script que salvei" devolver exatamente o que foi
  salvo, byte a byte, e sobreviver a consolidação + reboot (persiste no Git).

Fonte de verdade: notas/snippets/snippets.json (versionado; commitado pela
persistência na nuvem, alvo "snippet"). Lógica de detecção/extração/formatação
é pura e testável; o I/O de disco e a persistência ficam isolados.

Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

import hashlib
import json
import re

from src.core.config import PASTA_NOTAS
from src.core.texto import normalizar_espacos as _normalizar

PASTA_SNIPPETS = PASTA_NOTAS / "snippets"
ARQUIVO_SNIPPETS = PASTA_SNIPPETS / "snippets.json"
SCHEMA_VERSION = 1

_FENCE = re.compile(r"```([\w+-]*)\r?\n(.*?)```", re.S)


# ── Detecção de intenção ─────────────────────────────────────────────────────

_ALVOS_CODIGO = (
    "codigo", "script", "trecho", "snippet", "code", "funcao", "programa",
)
_VERBOS_SALVAR = (
    "grave", "gravar", "guarde", "guardar", "salve", "salvar", "memorize",
    "registre", "registrar", "anote este", "guarda esse", "salva esse",
)
_VERBOS_RECUPERAR = (
    "me manda", "me mande", "me envie", "me mostra", "me mostre", "recupera",
    "recupere", "resgata", "resgate", "qual era", "qual foi", "cade", "traz",
    "traga", "devolve", "manda de novo", "manda aquele",
)
_MARCAS_SALVO = (
    "salvei", "guardei", "gravei", "salvo", "guardado", "gravado",
    "que te pedi para guardar", "que te pedi pra guardar",
    "que mandei guardar", "que voce guardou", "que voce salvou",
)


def quer_salvar_snippet(pergunta: str) -> bool:
    """'grave este código', 'guarde esse script', 'salve este trecho'."""
    txt = _normalizar(pergunta)
    tem_verbo = any(v in txt for v in _VERBOS_SALVAR)
    tem_alvo = any(a in txt for a in _ALVOS_CODIGO)
    return tem_verbo and tem_alvo


def quer_recuperar_snippet(pergunta: str) -> bool:
    """'me manda o script que salvei', 'qual era o código que guardei'.

    Sinal decisivo: fala de código/script E de algo JÁ salvo (salvei/guardei/
    salvo...). O marcador em passado é o que distingue recuperar de salvar —
    por isso este predicado deve ser checado ANTES de quer_salvar_snippet
    (afinal 'guardei' contém 'guarde'). O verbo de recuperar reforça, mas o
    par alvo+marca já basta.
    """
    txt = _normalizar(pergunta)
    tem_alvo = any(a in txt for a in _ALVOS_CODIGO)
    tem_marca = any(m in txt for m in _MARCAS_SALVO)
    return tem_alvo and tem_marca


def quer_listar_snippets(pergunta: str) -> bool:
    txt = _normalizar(pergunta)
    gatilhos = ("meus snippets", "meus trechos", "meus scripts salvos",
                "trechos salvos", "codigos salvos", "scripts guardados",
                "quais scripts", "quais codigos", "liste os snippets",
                "listar snippets", "o que voce guardou")
    return any(g in txt for g in gatilhos)


# ── Extração de código ───────────────────────────────────────────────────────

def extrair_blocos_codigo(texto: str) -> list[dict]:
    """Retorna [{linguagem, codigo}] de todos os blocos ``` do texto."""
    blocos = []
    for lang, corpo in _FENCE.findall(str(texto or "")):
        corpo = corpo.rstrip("\n")
        if corpo.strip():
            blocos.append({"linguagem": (lang or "").strip().lower(), "codigo": corpo})
    return blocos


def ultimo_bloco_codigo(pergunta: str, historico: list[dict] | None) -> dict | None:
    """Acha o código a salvar: primeiro no texto atual; senão, o bloco mais
    recente do histórico (normalmente a resposta anterior do agente com o
    script). `historico` é a lista de {role, content} de st.session_state."""
    blocos = extrair_blocos_codigo(pergunta)
    if blocos:
        return blocos[-1]
    for msg in reversed(historico or []):
        blocos = extrair_blocos_codigo(str(msg.get("content", "")))
        if blocos:
            return blocos[-1]
    return None


# ── Persistência (fonte de verdade em JSON) ──────────────────────────────────

def _agora_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _carregar_bruto() -> dict:
    if not ARQUIVO_SNIPPETS.is_file():
        return {"schema_version": SCHEMA_VERSION, "itens": []}
    try:
        dados = json.loads(ARQUIVO_SNIPPETS.read_text(encoding="utf-8"))
        if not isinstance(dados.get("itens"), list):
            raise ValueError
        return dados
    except Exception:
        return {"schema_version": SCHEMA_VERSION, "itens": []}


def carregar_snippets() -> list[dict]:
    return list(_carregar_bruto().get("itens", []))


def _rotulo_padrao(codigo: str, linguagem: str) -> str:
    """Nome legível derivado do código (primeira definição/import útil)."""
    for linha in codigo.splitlines():
        m = re.search(r"\b(?:def|class)\s+([A-Za-z_]\w*)", linha)
        if m:
            return m.group(1)
    for linha in codigo.splitlines():
        alvo = re.search(r"title\(\s*['\"]([^'\"]{3,40})", linha)
        if alvo:
            return _slug(alvo.group(1))
    return f"{linguagem or 'trecho'}"


def _slug(texto: str) -> str:
    s = _normalizar(texto)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:48] or "trecho"


def salvar_snippet(codigo: str, *, rotulo: str = "", linguagem: str = "") -> dict:
    """Grava o trecho VERBATIM. Dedup por hash do código: salvar o mesmo código
    de novo não duplica. Retorna o registro."""
    codigo = str(codigo).rstrip("\n")
    if not codigo.strip():
        raise ValueError("Trecho vazio; nada a salvar.")
    rotulo = _slug(rotulo) if rotulo else _rotulo_padrao(codigo, linguagem)
    item_id = hashlib.sha256(codigo.encode("utf-8")).hexdigest()[:16]

    dados = _carregar_bruto()
    for item in dados["itens"]:
        if item.get("id") == item_id:
            return item  # já existe idêntico
    registro = {
        "id": item_id,
        "rotulo": rotulo,
        "linguagem": (linguagem or "").strip().lower(),
        "codigo": codigo,
        "criado_em_utc": _agora_utc(),
    }
    dados["itens"].append(registro)
    PASTA_SNIPPETS.mkdir(parents=True, exist_ok=True)
    tmp = ARQUIVO_SNIPPETS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(ARQUIVO_SNIPPETS)
    return registro


def recuperar_snippet(pergunta: str) -> dict | None:
    """Encontra o trecho pedido: por rótulo mencionado, senão o mais recente."""
    itens = carregar_snippets()
    if not itens:
        return None
    txt = _normalizar(pergunta)
    # Tenta casar por rótulo (nome) mencionado na pergunta.
    for item in reversed(itens):
        rot = _normalizar(item.get("rotulo", ""))
        if rot and rot in txt:
            return item
    return itens[-1]  # mais recente


def formatar_snippet_para_chat(registro: dict, *, total: int = 1) -> str:
    """Bloco de resposta VERBATIM (não passa pelo LLM)."""
    lang = registro.get("linguagem") or ""
    codigo = registro.get("codigo", "")
    rotulo = registro.get("rotulo", "trecho")
    cabecalho = f"📌 Trecho salvo **{rotulo}** (idêntico ao que você guardou):"
    if total > 1:
        cabecalho += f"\n\n_Você tem {total} trechos salvos; peça pelo nome para escolher outro._"
    return f"{cabecalho}\n\n```{lang}\n{codigo}\n```"


def formatar_lista_snippets(itens: list[dict]) -> str:
    if not itens:
        return "Você ainda não salvou nenhum trecho de código. Diga *'guarde este script'* após um bloco de código para começar."
    linhas = ["📂 **Trechos salvos** (peça *'me manda o script \\<nome\\>'* para recuperar idêntico):"]
    for item in itens:
        n = len(item.get("codigo", "").splitlines())
        linhas.append(f"- **{item.get('rotulo','trecho')}** ({item.get('linguagem') or 'texto'}, {n} linhas)")
    return "\n".join(linhas)
