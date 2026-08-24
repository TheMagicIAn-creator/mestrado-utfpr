"""
Consistência documento ↔ código.

A auditoria documental de 2026-07 encontrou o CLAUDE.md descrevendo modelos,
tamanhos de chunk e automações que não existiam mais no código. Estes testes
travam as classes de desalinhamento encontradas — só leem TEXTO dos arquivos,
sem importar módulos pesados, para rodarem no CI leve.
"""

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
CLAUDE = (RAIZ / "CLAUDE.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (RAIZ / ".env.example").read_text(encoding="utf-8")
PROVEDORES = (RAIZ / "src/conhecimento/provedores.py").read_text(encoding="utf-8")
INDEXADOR = (RAIZ / "src/conhecimento/indexador.py").read_text(encoding="utf-8")
README_SRC = (RAIZ / "src/README.md").read_text(encoding="utf-8")


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
    # Modelos GA explicitos verificados na API; Pro continua opt-in.
    for modelo in ("gemini-pro-latest", "gemini-3.6-flash", "gemini-3.5-flash-lite"):
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


def test_readme_src_nao_reintroduz_pipeline_aposentado():
    for retired in (
        "experimentos_artigos.py",
        "classificador_pv.py",
        "autoencoder_v2",
        "macro_comparar.py",
        "rul_weibull.py",
    ):
        assert retired not in README_SRC


def test_obsidian_documentado_com_governanca_e_sem_status_bibliografico():
    assert "Todo Markdown útil do vault" in CLAUDE
    assert "obsidian_pv" in CLAUDE
    assert "nunca vira citação bibliográfica" in CLAUDE
    assert "al_iado: false" in CLAUDE
    assert "sessão atual/arquivada" in CLAUDE


def test_readme_src_inventaria_todos_os_modulos():
    """Novo módulo não pode ficar invisível no mapa arquitetural."""
    faltando = []
    for pasta in ("core", "conhecimento", "ml"):
        for arquivo in sorted((RAIZ / "src" / pasta).glob("*.py")):
            if arquivo.name == "__init__.py":
                continue
            if f"`{arquivo.name}`" not in README_SRC:
                faltando.append(f"{pasta}/{arquivo.name}")
    assert not faltando, f"módulos ausentes de src/README.md: {faltando}"


def test_config_nao_reintroduz_modelo_gemini_aposentado():
    config = (RAIZ / "src/core/config.py").read_text(encoding="utf-8")
    assert "gemini-2.5-pro" not in config
    assert "gemini-2.5-flash" not in config


def test_descricoes_operacionais_usam_fmeca():
    """FMEA segue válido como conceito; a origem das falhas do projeto é FMECA."""
    padrao = re.compile(
        r"(?:orientad[ao]s?|fundamentad[ao]s?|assinaturas?|inje[cç][aã]o)"
        r".{0,80}\bFMEA\b",
        re.I | re.S,
    )
    achados = []
    for py in (RAIZ / "src").rglob("*.py"):
        texto = py.read_text(encoding="utf-8")
        if padrao.search(texto):
            achados.append(py.relative_to(RAIZ).as_posix())
    assert not achados, f"descrição operacional ainda usa FMEA: {achados}"


# ── O CLAUDE.md é a constituição do agente; ele não pode envelhecer sozinho ──

def test_claude_md_declara_o_dataset_que_o_pipeline_realmente_usa():
    """Guarda contra a dessincronia que aconteceu de fato.

    Entre 09 e 10/08/2026 o pipeline canônico migrou de Stender para
    GPVS-Faults em nove PRs — e o `CLAUDE.md` não foi tocado em nenhuma delas.
    "GPVS" aparecia zero vezes nele enquanto `pipeline.py` já apontava para
    `features_gpvs`. Como o `PERFIL_COMPACTO` do agente deriva do CLAUDE.md, o
    Al IAdo passou a descrever um projeto que não existia mais.

    Nada aqui verifica NÚMERO — números vivem nos artefatos. O que se verifica é
    que o documento nomeia o mesmo dataset que o código executa.
    """
    pipeline = (RAIZ / "src/ml/pipeline.py").read_text(encoding="utf-8")
    claude = (RAIZ / "CLAUDE.md").read_text(encoding="utf-8")

    if "features_gpvs" not in pipeline:
        pytest.skip("pipeline não usa a etapa GPVS; guarda não se aplica")

    assert "GPVS" in claude, (
        "o pipeline canônico roda GPVS-Faults, mas o CLAUDE.md não o menciona. "
        "O agente responderá sobre o dataset errado."
    )


def test_claude_md_nao_chama_stender_de_dataset_principal():
    """O conjunto Stender virou referência histórica, não a base de treino."""
    claude = (RAIZ / "CLAUDE.md").read_text(encoding="utf-8")
    pipeline = (RAIZ / "src/ml/pipeline.py").read_text(encoding="utf-8")
    if "features_gpvs" not in pipeline:
        pytest.skip("pipeline não usa a etapa GPVS; guarda não se aplica")

    proibidas = [
        "no dataset de operação normal (Paderborn)",
        "Datasets: Paderborn (inversor saudável) e PV Farms",
    ]
    achados = [f for f in proibidas if f in claude]
    assert not achados, (
        f"CLAUDE.md ainda aponta Stender/Paderborn como base principal: {achados}"
    )


def test_glossario_desambigua_F0():
    """`F0` tem dois sentidos vivos: frequência fundamental e ensaio saudável.

    É o mesmo tipo de homonímia que o `D` já causou. O glossário é o árbitro
    declarado de conflito entre documentos, então a separação mora nele.
    """
    glossario = (RAIZ / "docs/glossario.md").read_text(encoding="utf-8")
    assert "Símbolos que colidem" in glossario
    for marca in ("F0L", "frequência fundamental", "GRID_FREQUENCY_HZ"):
        assert marca in glossario, f"o verbete de F0 perdeu a marca {marca!r}"


def test_contrato_gpvs_congela_selecao_antes_da_e3():
    artefato = RAIZ / "resultados/comparacao/comparacao_autoencoders.json"
    dados = json.loads(artefato.read_text(encoding="utf-8"))
    protocolo = dados["protocol"]
    assert protocolo["model_selection_uses_fault_data"] is False
    assert "congelados" in protocolo["e3"]


def test_mapa_de_resultados_publica_somente_comparacao_e_confiabilidade():
    mapa = RAIZ / "docs/mapa_de_resultados.md"
    assert mapa.exists(), "docs/mapa_de_resultados.md sumiu"
    texto = mapa.read_text(encoding="utf-8")

    for marca in ("Autoencoder Denso", "AE-LSTM", "R(t)", "F(t)", "f(t)", "h(t)"):
        assert marca in texto, f"o mapa perdeu a marca {marca!r}"
    assert "S_D(a)" not in texto
    assert "a_det" not in texto
    assert "anos" in texto

    # O mapa não pode virar tabela de métricas: valor citado em documento
    # envelhece em silêncio, e a regra do projeto é ler o artefato vigente.
    assert "não repete valores" in texto or "não** repete valores" in texto


def test_artefato_canonico_de_confiabilidade_declara_que_nao_estima_do_dataset():
    """A ressalva tem de viajar com o dado, não só com o texto.

    Se um dia o artefato perder essa declaração, o mapa passa a afirmar por
    conta própria — e nós voltamos a ter duas fontes para a mesma ressalva.
    """
    artefato = RAIZ / "resultados/confiabilidade/metodologia.json"
    dados = json.loads(artefato.read_text(encoding="utf-8"))
    assert dados.get("status") == "bibliographic_component_sensitivity"
    assert dados.get("evidence_scope") == "bibliographic_reliability_only"
    assert dados.get("time_unit_primary") == "hour", (
        "o eixo da confiabilidade física é TEMPO; se virar magnitude, ela "
        "deixou de representar tempo de operação"
    )
