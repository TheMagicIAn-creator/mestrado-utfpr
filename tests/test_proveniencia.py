"""
Sprint 1 — rastreabilidade: manifesto de proveniência e estado stale.

Verifica que:
- alteração de parâmetro invalida a etapa (stale);
- alteração do código da etapa invalida (stale);
- regeneração de artefato upstream invalida downstream (stale);
- artefato ausente / sem manifesto → pending;
- manifesto compatível → ready.
"""

from pathlib import Path

from src.ml import proveniencia as P


def _escreve(p: Path, txt: str = "x") -> Path:
    p.write_text(txt, encoding="utf-8")
    return p


def test_hash_muda_com_conteudo(tmp_path):
    a = _escreve(tmp_path / "a.txt", "um")
    h1 = P.sha256_arquivo(a)
    _escreve(a, "dois")
    assert h1 and h1 != P.sha256_arquivo(a)
    assert P.sha256_arquivo(tmp_path / "naoexiste") is None


def test_comparar_detecta_parametros(tmp_path):
    code = _escreve(tmp_path / "code.py", "print(1)")
    out = _escreve(tmp_path / "out.json", "{}")
    m1 = P.gerar_manifesto("s", code, {"epochs": 100}, {}, [out])
    m2 = P.gerar_manifesto("s", code, {"epochs": 200}, {}, [out])
    assert any("parâmetro" in x for x in P.comparar(m1, m2))


def test_comparar_detecta_upstream(tmp_path):
    code = _escreve(tmp_path / "code.py")
    up = _escreve(tmp_path / "feat.parquet", "v1")
    out = _escreve(tmp_path / "out.json")
    m1 = P.gerar_manifesto("s", code, {}, {"features": up}, [out])
    _escreve(up, "v2")  # upstream regenerado
    m2 = P.gerar_manifesto("s", code, {}, {"features": up}, [out])
    assert any("upstream" in x for x in P.comparar(m1, m2))


def test_estado_pending_sem_artefato(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PASTA_MANIFESTOS", tmp_path / "man")
    code = _escreve(tmp_path / "code.py")
    r = P.estado_etapa("s", [tmp_path / "naoexiste.json"], code)
    assert r["estado"] == P.PENDING


def test_estado_pending_sem_manifesto(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PASTA_MANIFESTOS", tmp_path / "man")
    code = _escreve(tmp_path / "code.py")
    out = _escreve(tmp_path / "out.json")
    r = P.estado_etapa("s", [out], code)  # artefato existe, mas sem manifesto
    assert r["estado"] == P.PENDING


def test_estado_ready_depois_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PASTA_MANIFESTOS", tmp_path / "man")
    code = _escreve(tmp_path / "code.py", "v1")
    out = _escreve(tmp_path / "out.json", "{}")

    P.salvar_manifesto(P.gerar_manifesto("s", code, {"epochs": 100}, {}, [out]))
    assert P.estado_etapa("s", [out], code, {"epochs": 100}, {})["estado"] == P.READY

    # parâmetro mudou → stale
    assert P.estado_etapa("s", [out], code, {"epochs": 200}, {})["estado"] == P.STALE

    # código mudou → stale
    _escreve(code, "v2")
    assert P.estado_etapa("s", [out], code, {"epochs": 100}, {})["estado"] == P.STALE


def test_estado_pipeline_valores_validos():
    """O pipeline real reporta apenas ready/stale/pending para cada etapa."""
    from src.ml.pipeline import estado_pipeline

    estados = estado_pipeline()
    assert set(estados) == {
        "features_ca", "autoencoder", "injecao_falhas", "validacao", "rul_weibull",
    }
    for info in estados.values():
        assert info["estado"] in {P.READY, P.STALE, P.PENDING}


def test_pipeline_captura_parametros_das_etapas():
    from src.ml.pipeline import get_stage

    auto = get_stage("autoencoder").parameters()
    assert auto["epochs"] > 0
    assert auto["latente_dim"] > 0
    assert auto["threshold_method"] == "p99"

    validacao = get_stage("validacao").parameters()
    assert validacao["n_janelas_saudavel"] > 0
    assert validacao["sevs_validacao"]


def test_pipeline_registra_todos_artefatos_upstream():
    from src.ml.pipeline import _inputs_da_etapa, get_stage

    inputs = _inputs_da_etapa(get_stage("autoencoder"))
    assert any("features_paderborn.parquet" in key for key in inputs)
    assert any("features_paderborn_stats.csv" in key for key in inputs)


def test_status_markdown_usa_estado_trivalorado():
    from src.ml.pipeline import status_markdown

    md = status_markdown().lower()
    assert "status do pipeline" in md
    assert any(t in md for t in ("pronto", "pendente", "stale", "desatualizado"))
