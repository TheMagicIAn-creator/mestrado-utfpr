"""
Consistência documento ↔ código.

A auditoria documental de 2026-07 encontrou o CLAUDE.md descrevendo modelos,
tamanhos de chunk e automações que não existiam mais no código. Estes testes
travam as classes de desalinhamento encontradas — só leem TEXTO dos arquivos,
sem importar módulos pesados, para rodarem no CI leve.
"""

import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CLAUDE = (RAIZ / "CLAUDE.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (RAIZ / ".env.example").read_text(encoding="utf-8")
PROVEDORES = (RAIZ / "src/conhecimento/provedores.py").read_text(encoding="utf-8")
INDEXADOR = (RAIZ / "src/conhecimento/indexador.py").read_text(encoding="utf-8")


def test_env_example_e_arquivo_env_valido():
    """Cada linha é comentário, vazia ou CHAVE=valor (sem resíduo de shell)."""
    padrao = re.compile(r"^(#.*|\s*|[A-Z][A-Z0-9_]*=.*)$")
    for i, linha in enumerate(ENV_EXAMPLE.splitlines(), 1):
        assert padrao.match(linha), f".env.example linha {i} inválida: {linha!r}"


def test_env_example_cobre_variaveis_lidas_pelo_codigo():
    """Toda os.getenv("X") em src/ aparece no .env.example (docum. mínima)."""
    ignoradas = {"KMP_DUPLICATE_LIB_OK"}  # setada em código, não configuração
    lidas = set()
    for py in (RAIZ / "src").rglob("*.py"):
        for m in re.finditer(r'os\.getenv\(\s*["\']([A-Z][A-Z0-9_]*)["\']',
                             py.read_text(encoding="utf-8")):
            lidas.add(m.group(1))
    faltando = lidas - ignoradas - {
        m.group(1)
        for m in re.finditer(r"^#?\s*([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE, re.M)
    }
    assert not faltando, f"variáveis lidas no código sem doc no .env.example: {faltando}"


def test_modelos_citados_existem_no_codigo():
    """Modelos nomeados no CLAUDE.md/.env.example existem em provedores.py.

    A equipe agora é 100% Gemini (Groq/LLaMA foram removidos): os modelos
    da família LLaMA e o próprio Groq não podem reaparecer nos docs.
    """
    proibidos = ("3.1 8B", "LLaMA 3.1", "llama-3.1", "llama-3.3",
                 "Gemma", "gemma")
    for texto, nome in ((CLAUDE, "CLAUDE.md"), (ENV_EXAMPLE, ".env.example")):
        for p in proibidos:
            assert p not in texto, f"{nome} cita modelo inexistente no código: {p!r}"
    # Aliases -latest (a família 2.5 foi aposentada em 2026; versões explícitas
    # giram rápido). Nível 1 = pro (conversa); 2 = flash (auditor); 3 = flash-lite.
    for modelo in ("gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"):
        assert modelo in PROVEDORES, f"provedores.py não define {modelo}"
        assert modelo in CLAUDE, f"CLAUDE.md não documenta o modelo real {modelo}"


def test_chunk_de_literatura_documentado_bate_com_o_codigo():
    m = re.search(r'TAMANHO_CHUNK_LITERATURA",\s*"(\d+)"', INDEXADOR)
    assert m, "default de TAMANHO_CHUNK_LITERATURA não encontrado no indexador"
    assert m.group(1) in CLAUDE, (
        f"CLAUDE.md não cita o tamanho real de chunk ({m.group(1)})"
    )


def test_claude_md_sem_metricas_de_pipeline_hardcoded():
    """Regra do próprio CLAUDE.md: métricas vêm dos artefatos, não do perfil.

    NPR é permitido (valor estático do FMECA do TCC, é literatura, não
    resultado do pipeline).
    """
    padrao = re.compile(r"\b(F1|AUC|MTTF|SMD|limiar|recall)\s*[=:]\s*\d", re.I)
    achados = [m.group(0) for m in padrao.finditer(CLAUDE)]
    assert not achados, f"CLAUDE.md contém métrica de pipeline fixada: {achados}"


def test_requirements_referenciados_existem():
    """Nenhum doc aponta para um arquivo requirements-* removido."""
    for doc in (RAIZ / "docs").glob("*.md"):
        for m in re.finditer(r"requirements[\w-]*\.txt",
                             doc.read_text(encoding="utf-8")):
            assert (RAIZ / m.group(0)).exists(), (
                f"{doc.name} referencia {m.group(0)}, que não existe"
            )


def test_experimentos_do_claude_md_existem_no_registry():
    """Experimentos listados no CLAUDE.md batem com o REGISTRO do código."""
    codigo = (RAIZ / "src/ml/experimentos_artigos.py").read_text(encoding="utf-8")
    bloco = re.search(r"REGISTRO[^=]*=\s*\{(.*?)\n\}", codigo, re.S)
    assert bloco, "dict REGISTRO não encontrado"
    chaves = set(re.findall(r'"(\w+)":\s*ExperimentoArtigo', bloco.group(1)))
    assert chaves == {"francisti", "ibrahim"}, (
        f"registry divergente do documentado: {chaves}"
    )
    for cortado in ("ghoneim", "sharma", "ahirwar", "stender"):
        assert cortado not in chaves, f"experimento cortado voltou: {cortado}"


def test_obsidian_documentado_com_governanca_e_sem_status_bibliografico():
    assert "Todo Markdown útil do vault" in CLAUDE
    assert "obsidian_pv" in CLAUDE
    assert "nunca vira citação bibliográfica" in CLAUDE
    assert "al_iado: false" in CLAUDE
    assert "sessão atual/arquivada" in CLAUDE
