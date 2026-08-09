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


def test_hash_textual_normalizado_ignora_crlf(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_bytes(b"print(1)\nprint(2)\n")
    b.write_bytes(b"print(1)\r\nprint(2)\r\n")
    assert P.sha256_arquivo(a) != P.sha256_arquivo(b)
    assert P.sha256_arquivo_texto_normalizado(a) == P.sha256_arquivo_texto_normalizado(b)


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


def test_manifesto_v2_registra_dependencias_e_outputs(tmp_path):
    code = _escreve(tmp_path / "code.py")
    dep = _escreve(tmp_path / "dep.py")
    out = _escreve(tmp_path / "out.json", "{}")
    m = P.gerar_manifesto("s", code, {}, {}, [out], code_dependencies={"dep": dep})
    assert m["manifest_version"] == 2
    assert m["code_hash_mode"] == "text_lf_utf8"
    assert m["output_hash_mode"] == "text_lf_utf8_by_suffix_else_binary"
    assert m["code_dependencies"]["dep"] == P.sha256_arquivo_texto_normalizado(dep)
    assert list(m["output_artifacts"].values()) == [
        P.sha256_arquivo_texto_normalizado(out)
    ]


def test_hash_de_saida_textual_e_portavel_entre_lf_e_crlf(tmp_path):
    code = _escreve(tmp_path / "code.py")
    out = tmp_path / "out.json"
    out.write_bytes(b'{"ok": true}\r\n')
    crlf = P.gerar_manifesto("s", code, {}, {}, [out])
    out.write_bytes(b'{"ok": true}\n')
    lf = P.gerar_manifesto("s", code, {}, {}, [out])

    assert crlf["output_artifacts"] == lf["output_artifacts"]


def test_comparar_pode_declarar_input_ausente_sem_ocultar_mudanca(tmp_path):
    code = _escreve(tmp_path / "code.py")
    up = _escreve(tmp_path / "up.bin", "original")
    out = _escreve(tmp_path / "out.json", "{}")
    salvo = P.gerar_manifesto("s", code, {}, {"up": up}, [out])

    up.unlink()
    ausente = P.gerar_manifesto("s", code, {}, {"up": up}, [out])
    assert any("upstream" in x for x in P.comparar(salvo, ausente))
    assert not P.comparar(salvo, ausente, permitir_inputs_ausentes=True)

    _escreve(up, "alterado")
    alterado = P.gerar_manifesto("s", code, {}, {"up": up}, [out])
    assert any(
        "upstream" in x
        for x in P.comparar(salvo, alterado, permitir_inputs_ausentes=True)
    )


def test_estado_stale_quando_dependencia_cientifica_muda(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PASTA_MANIFESTOS", tmp_path / "man")
    code = _escreve(tmp_path / "code.py", "v1")
    dep = _escreve(tmp_path / "dep.py", "v1")
    out = _escreve(tmp_path / "out.json", "{}")
    P.salvar_manifesto(
        P.gerar_manifesto("s", code, {}, {}, [out], code_dependencies={"dep": dep})
    )
    assert P.estado_etapa("s", [out], code, {}, {}, {"dep": dep})["estado"] == P.READY
    _escreve(dep, "v2")
    res = P.estado_etapa("s", [out], code, {}, {}, {"dep": dep})
    assert res["estado"] == P.STALE
    assert any("dependência" in motivo for motivo in res["motivos"])


def test_manifesto_v1_existente_vira_stale_sem_quebrar_leitura(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PASTA_MANIFESTOS", tmp_path / "man")
    code = _escreve(tmp_path / "code.py", "v1")
    out = _escreve(tmp_path / "out.json", "{}")
    P.salvar_manifesto({
        "stage": "s",
        "code_sha256": P.sha256_arquivo(code),
        "parameters": {},
        "input_artifacts": {},
        "outputs": [str(out)],
    })
    res = P.estado_etapa("s", [out], code, {}, {})
    assert res["estado"] == P.STALE
    assert "manifesto v2 ausente" in res["motivos"]


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
    assert auto["dropout"] == 0.2
    assert auto["paciencia"] > 0
    assert auto["threshold_method"] == "p99"

    validacao = get_stage("validacao").parameters()
    assert validacao["n_janelas_saudavel"] > 0
    assert validacao["prevalencia_rara"] == 0.05
    assert validacao["sevs_validacao"]

    rul = get_stage("rul_weibull").parameters()
    assert rul["a_det_unidade"] == "a_det_fracao_da_assinatura_nominal"
    assert rul["ttf_unidade"] == rul["a_det_unidade"]  # alias
    assert rul["tempo_fisico_calibrado"] is False
    assert rul["persistencia_cruzamento"] > 0


def test_pipeline_le_parametros_sem_importar_modulo_pesado(monkeypatch):
    import src.ml.pipeline as pipeline

    def falha_import(module):
        if module == "src.ml.autoencoder":
            raise ModuleNotFoundError("torch")
        return __import__(module, fromlist=["*"])

    monkeypatch.setattr(pipeline, "import_module", falha_import)
    auto = pipeline.get_stage("autoencoder").parameters()

    assert auto["epochs"] > 0
    assert auto["threshold_method"] == "p99"


def test_pipeline_registra_todos_artefatos_upstream():
    from src.ml.pipeline import _inputs_da_etapa, get_stage

    inputs = _inputs_da_etapa(get_stage("autoencoder"))
    assert any("features_paderborn.parquet" in key for key in inputs)
    assert any("features_paderborn_stats.csv" in key for key in inputs)


def test_pipeline_registra_dependencias_cientificas():
    from src.ml.pipeline import _code_dependencies, get_stage

    deps = _code_dependencies(get_stage("autoencoder"))
    assert "src.ml.escore_anomalia" in deps
    assert "src.ml.split_temporal" in deps


def test_status_markdown_usa_estado_trivalorado():
    from src.ml.pipeline import status_markdown

    md = status_markdown().lower()
    assert "status do pipeline" in md
    assert any(t in md for t in ("pronto", "pendente", "stale", "desatualizado"))
