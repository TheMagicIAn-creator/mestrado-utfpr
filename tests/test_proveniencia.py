from __future__ import annotations

import json

from src.ml import proveniencia as provenance


def test_code_hash_is_stable_between_lf_and_crlf(tmp_path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert provenance.sha256_arquivo_texto_normalizado(lf) == provenance.sha256_arquivo_texto_normalizado(crlf)


def test_manifest_v2_hashes_code_dependencies_inputs_and_outputs(tmp_path):
    code = tmp_path / "stage.py"
    dependency = tmp_path / "dependency.py"
    source = tmp_path / "source.csv"
    output = tmp_path / "output.json"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    dependency.write_text("OTHER = 2\n", encoding="utf-8")
    source.write_text("x\n1\n", encoding="utf-8")
    output.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")

    manifest = provenance.gerar_manifesto(
        "stage",
        code,
        {"alpha": 1},
        {"source": source},
        [output],
        code_dependencies={"dependency": dependency},
        evidence_level="E3_bench",
    )

    assert manifest["manifest_version"] == 2
    assert manifest["code_hash_mode"] == "text_lf_utf8"
    assert manifest["code_dependencies"]["dependency"]
    assert manifest["input_artifacts"]["source"]
    assert manifest["output_artifacts"][provenance.to_project_relative_path(output)]
    assert manifest["evidence_level"] == "E3_bench"


def test_v1_manifest_is_marked_stale_against_v2(tmp_path):
    code = tmp_path / "stage.py"
    output = tmp_path / "output.csv"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    output.write_text("x\n1\n", encoding="utf-8")
    current = provenance.gerar_manifesto("stage", code, {}, {}, [output])
    legacy = {
        "stage": "stage",
        "code_sha256": current["code_sha256"],
        "parameters": {},
        "input_artifacts": {},
    }
    assert "manifesto v2 ausente" in provenance.comparar(legacy, current)


def test_missing_local_input_can_be_unverified_in_query_mode(tmp_path):
    code = tmp_path / "stage.py"
    output = tmp_path / "output.csv"
    present = tmp_path / "source.csv"
    missing = tmp_path / "missing.csv"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    output.write_text("x\n1\n", encoding="utf-8")
    present.write_text("x\n1\n", encoding="utf-8")
    saved = provenance.gerar_manifesto("stage", code, {}, {"source": present}, [output])
    current = provenance.gerar_manifesto("stage", code, {}, {"source": missing}, [output])
    assert provenance.comparar(saved, current, permitir_inputs_ausentes=True) == []
