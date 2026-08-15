"""A comparação com a literatura vem dos macro-códigos, e é alcançável pelo chat.

Duas coisas quebradas ao mesmo tempo, encontradas em 30/07:

1. `resultados/experimentos/` foi deletada em `9fe0322`, quando os macro-códigos
   substituíram o framework por artigo — mas as ferramentas do agente ainda
   apontavam para lá. Pedir "compare meu método com a literatura" caía num
   caminho morto.
2. **Nenhuma** ferramenta lia `resultados/macro/`. O resultado que sustenta a
   dissertação — AUC 0,978 × 0,909 e SMD 0,50 × 1,00 no IGBT — era inalcançável
   pelo chat, embora estivesse versionado e publicado no site.

O framework E1 foi aposentado do roteador (os módulos seguem no repositório,
preservando o histórico) e `consultar_comparacao_macro` ocupou o lugar.

Protege também a separação E1 × E2: 0,588 e 0,909 medem coisas diferentes e
nunca vão na mesma tabela.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.conhecimento.ferramentas as fr


_E1_APOSENTADAS = {
    "rodar_experimento_artigo",
    "listar_experimentos_artigos",
    "limpar_experimentos_artigos",
    "comparar_experimentos_auc",
}


# ── o caminho morto não existe mais ──────────────────────────────────────────

def test_ferramentas_do_framework_e1_saem_do_despacho():
    assert not (_E1_APOSENTADAS & set(fr._DESPACHO)), (
        "o framework E1 recriaria resultados/experimentos/ com números "
        "conflitantes com os macros"
    )


def test_ferramentas_do_framework_e1_saem_do_catalogo_do_llm():
    nomes = {e["name"] for e in fr.ESPEC_FERRAMENTAS}
    assert not (_E1_APOSENTADAS & nomes)


def test_comparacao_e1_concorrente_foi_removida():
    raiz = Path(__file__).resolve().parents[1]
    assert not (raiz / "src/ml/comparacao_literatura.py").exists()
    fonte = (raiz / "src/conhecimento/ferramentas.py").read_text(encoding="utf-8")
    assert "def comparar_experimentos_auc" not in fonte


def test_a_comparacao_macro_esta_registrada():
    assert "consultar_comparacao_macro" in fr._DESPACHO
    nomes = {e["name"] for e in fr.ESPEC_FERRAMENTAS}
    assert "consultar_comparacao_macro" in nomes


# ── roteamento: toda forma do pedido chega ao lugar certo ────────────────────

@pytest.mark.parametrize("pergunta", [
    "compare meu método com a literatura",
    "meu detector é melhor que o AE-LSTM?",
    "como estou frente ao Ibrahim?",
    "o AE LSTM ganha do meu?",
    "compare os experimentos de anomalia por AUC",
    "rode o experimento do Ibrahim",      # não há mais o que rodar: lê o publicado
    "quais experimentos existem?",
])
def test_roteia_para_a_comparacao_publicada(pergunta):
    decisao = fr._decisao_rapida(pergunta) or {}
    assert decisao.get("ferramenta") == "consultar_comparacao_macro", pergunta


@pytest.mark.parametrize("pergunta,esperado", [
    ("rode o pipeline completo", "rodar_pipeline_completo"),
    ("apague os resultados do weibull", "limpar_resultados_ml"),
    ("compare as abordagens de ML", "comparar_abordagens_ml"),
])
def test_nao_sequestra_outros_pedidos(pergunta, esperado):
    assert (fr._decisao_rapida(pergunta) or {}).get("ferramenta") == esperado


# ── conteúdo da resposta ─────────────────────────────────────────────────────

def _macro(tmp_path, monkeypatch, tabela="| Método | AUC |\n|---|---|\n| Proposto | 0.978 |"):
    pasta = tmp_path / "resultados" / "macro"
    pasta.mkdir(parents=True)
    (pasta / "comparacao_tabela.md").write_text(tabela, encoding="utf-8")
    (pasta / "comparacao_resultado.json").write_text(json.dumps([
        {"nome": "Proposto", "percentil": 99.0, "fp_pct": 0.0,
         "n_calib": 17, "n_aval": 25, "falhas": {}},
        {"nome": "Ibrahim 2022 (AE-LSTM temporal)", "percentil": 99.0, "fp_pct": 0.0,
         "n_calib": 17, "n_aval": 25, "falhas": {}},
    ]), encoding="utf-8")
    monkeypatch.setattr("src.core.config.RAIZ_PROJETO", tmp_path)
    return pasta


def test_traz_a_tabela_publicada(tmp_path, monkeypatch):
    _macro(tmp_path, monkeypatch)
    r = fr.consultar_comparacao_macro()
    assert r["ok"]
    assert "0.978" in r["mensagem"]
    assert "Proposto" in r["mensagem"] and "Ibrahim" in r["mensagem"]


def test_declara_o_protocolo_lido_do_artefato(tmp_path, monkeypatch):
    """O protocolo não pode ser constante escrita na ferramenta."""
    _macro(tmp_path, monkeypatch)
    msg = fr.consultar_comparacao_macro()["mensagem"]
    assert "17" in msg and "25" in msg          # janelas de calibração/avaliação
    assert "99.0" in msg or "99" in msg


def test_sempre_carrega_as_ressalvas(tmp_path, monkeypatch):
    """Resultado E2 nunca pode ser apresentado como desempenho de campo."""
    _macro(tmp_path, monkeypatch)
    msg = fr.consultar_comparacao_macro()["mensagem"]
    assert "E2" in msg
    assert "campo" in msg.lower()
    assert "smd" in msg.lower() and "menor é melhor" in msg.lower()


def test_explica_por_que_severidade_1_nao_discrimina(tmp_path, monkeypatch):
    _macro(tmp_path, monkeypatch)
    assert "satur" in fr.consultar_comparacao_macro()["mensagem"].lower()


def test_nunca_treina(tmp_path, monkeypatch):
    """A ferramenta só lê artefato — não pode importar o caminho de treino."""
    _macro(tmp_path, monkeypatch)
    import inspect

    fonte = inspect.getsource(fr.consultar_comparacao_macro)
    for proibido in ("executar(", "treinar", "macro_proposto", "macro_ibrahim"):
        assert proibido not in fonte, f"a ferramenta não pode {proibido}"


def test_sem_comparacao_publicada_orienta(tmp_path, monkeypatch):
    (tmp_path / "resultados" / "macro").mkdir(parents=True)
    monkeypatch.setattr("src.core.config.RAIZ_PROJETO", tmp_path)
    r = fr.consultar_comparacao_macro()
    assert not r["ok"]
    assert "macro_comparar" in r["mensagem"]


def test_artefato_corrompido_nao_derruba(tmp_path, monkeypatch):
    pasta = _macro(tmp_path, monkeypatch)
    (pasta / "comparacao_resultado.json").write_text("{ nao é json", encoding="utf-8")
    r = fr.consultar_comparacao_macro()
    assert not r["ok"] and "ileg" in r["mensagem"].lower()


# ── A comparação pode envelhecer sem avisar ────────────────────────────────

def test_comparacao_mais_velha_que_o_autoencoder_e_recusada(tmp_path, monkeypatch):
    """Números de um detector aposentado não podem sair com cara de vigentes.

    Aconteceu: a comparação macro foi gerada em 09/08/2026 às 14:59 e o
    Autoencoder foi retreinado às 23:37, na migração para o GPVS-Faults. O
    artefato não carrega aviso de obsolescência, então "sou melhor que o
    AE-LSTM?" devolveria AUC e SMD de um modelo que não existe mais.

    A checagem é por MANIFESTO e não por mtime: `git checkout` reescreve mtime
    e mentiria sobre a idade.
    """
    import json

    from src.conhecimento import ferramentas_academicas as fa

    manifestos = tmp_path / "resultados" / "manifestos"
    manifestos.mkdir(parents=True)
    (manifestos / "macro_comparacao.json").write_text(json.dumps(
        {"created_at": "2026-08-09T14:59:09", "git_commit": "7f91ff06"}))
    (manifestos / "autoencoder.json").write_text(json.dumps(
        {"created_at": "2026-08-09T23:37:03", "git_commit": "a4aa42c2"}))
    monkeypatch.setattr(fa, "RAIZ_PROJETO", tmp_path, raising=False)
    monkeypatch.setattr("src.core.config.RAIZ_PROJETO", tmp_path)

    motivo = fa._comparacao_desatualizada(tmp_path / "resultados" / "macro")
    assert motivo, "comparação mais velha que o detector tem de ser recusada"
    assert "14:59:09" in motivo and "23:37:03" in motivo
    assert "7f91ff06" in motivo and "a4aa42c2" in motivo


def test_comparacao_mais_nova_que_o_autoencoder_e_aceita(tmp_path, monkeypatch):
    """Contraprova: a guarda não pode recusar comparação legítima."""
    import json

    from src.conhecimento import ferramentas_academicas as fa

    manifestos = tmp_path / "resultados" / "manifestos"
    manifestos.mkdir(parents=True)
    (manifestos / "macro_comparacao.json").write_text(json.dumps(
        {"created_at": "2026-08-10T02:00:00", "git_commit": "ffffffff"}))
    (manifestos / "autoencoder.json").write_text(json.dumps(
        {"created_at": "2026-08-09T23:37:03", "git_commit": "a4aa42c2"}))
    monkeypatch.setattr("src.core.config.RAIZ_PROJETO", tmp_path)

    assert fa._comparacao_desatualizada(tmp_path / "resultados" / "macro") == ""


def test_sem_manifesto_dos_dois_lados_nao_afirma_obsolescencia(tmp_path, monkeypatch):
    """Ausência de manifesto não é prova de nada — não inventar veredito."""
    from src.conhecimento import ferramentas_academicas as fa

    (tmp_path / "resultados" / "manifestos").mkdir(parents=True)
    monkeypatch.setattr("src.core.config.RAIZ_PROJETO", tmp_path)

    assert fa._comparacao_desatualizada(tmp_path / "resultados" / "macro") == ""


# ── A comparação tem de ver o MESMO vetor que o detector foi treinado a ver ──

def test_macros_usam_o_extrator_do_dataset_canonico():
    """O bug mais caro desta sessão foi mudo, e este teste é a rede contra ele.

    Depois da migração para o GPVS, `macro_proposto` e `macro_ibrahim`
    continuaram importando `features_ca.extrair_janela` — o extrator do Stender,
    que devolve ~108 features com nomes como `i_a_rms`. Mas `det["colunas"]` são
    as 24 features do GPVS (`Ipv_median`, `ia_thd`, …). Nenhum dos 24 nomes
    existia no dicionário, e o acesso era `.get(c, 0.0)`.

    Resultado: vetor de 24 ZEROS, sem erro de shape, sem aviso. O autoencoder
    reconstruía o nada e a comparação publicava esse número como AUC.

    Duas coisas travadas aqui: o extrator tem de ser o do dataset canônico, e o
    acesso NÃO pode ter default silencioso.
    """
    import ast

    from src.core.config import RAIZ_PROJETO

    for nome in ("macro_proposto", "macro_ibrahim"):
        fonte = (RAIZ_PROJETO / f"src/ml/{nome}.py").read_text(encoding="utf-8")
        arvore = ast.parse(fonte)

        origens = {
            no.module
            for no in ast.walk(arvore)
            if isinstance(no, ast.ImportFrom) and no.module
            and any(a.name == "extrair_janela" for a in no.names)
        }
        assert origens == {"src.ml.gpvs_principal"}, (
            f"{nome} importa extrair_janela de {origens or 'lugar nenhum'}; "
            "tem de vir de gpvs_principal, o mesmo extrator que gerou as "
            "features de treino do detector"
        )

        assert "extrair_janela(j).get(" not in fonte, (
            f"{nome} usa `.get(coluna, default)` ao montar o vetor de features. "
            "Feature que falta é defeito e tem de estourar: foi o default "
            "silencioso que transformou incompatibilidade de dataset em "
            "resultado plausível."
        )


def test_macros_nao_carregam_mais_o_dataset_stender():
    """GPVS é o único dataset de estimativa. O Stender saiu da cadeia."""
    from src.core.config import RAIZ_PROJETO

    for nome in ("macro_proposto", "macro_ibrahim", "macro_comparar"):
        fonte = (RAIZ_PROJETO / f"src/ml/{nome}.py").read_text(encoding="utf-8")
        for proibido in ("carregar_paderborn_compacto", "features_paderborn"):
            assert proibido not in fonte, (
                f"{nome} ainda referencia {proibido} — o comparativo precisa "
                "rodar sobre o GPVS para ser pertinente"
            )
