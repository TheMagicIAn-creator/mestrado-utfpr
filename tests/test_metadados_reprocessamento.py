from __future__ import annotations

import hashlib
import json

import src.conhecimento.processador_pdf as proc
import src.core.config as config
import src.orquestrador as orquestrador


def _sha256(conteudo: bytes) -> str:
    return hashlib.sha256(conteudo).hexdigest()


def _configurar_raiz(tmp_path, monkeypatch):
    literatura = tmp_path / "literatura"
    cromadb = tmp_path / "base_conhecimento"
    pendentes = tmp_path / "metadados_pendentes.json"
    revisados = tmp_path / "metadados_revisados.json"
    snapshot = tmp_path / "artefatos" / "literatura.jsonl.gz"
    literatura.mkdir()

    monkeypatch.setattr(proc, "RAIZ_PROJETO", tmp_path)
    monkeypatch.setattr(proc, "PASTA_LITERATURA", literatura)
    monkeypatch.setattr(orquestrador, "RAIZ_PROJETO", tmp_path)
    monkeypatch.setattr(orquestrador, "PASTA_LITERATURA", literatura)
    monkeypatch.setattr(orquestrador, "PASTA_CHROMADB", cromadb)
    monkeypatch.setattr(config, "RAIZ_PROJETO", tmp_path)
    monkeypatch.setattr(config, "ARQUIVO_INDICE_LITERATURA", snapshot)
    return literatura, pendentes, revisados


def test_metadados_curados_por_hash_dispensam_nova_inferencia(tmp_path, monkeypatch):
    literatura, _pendentes, revisados = _configurar_raiz(tmp_path, monkeypatch)
    pdf = literatura / "documento_0000.pdf"
    pdf.write_bytes(b"pdf-identidade-estavel")
    revisados.write_text(json.dumps({
        "documentos": {
            _sha256(pdf.read_bytes()): {
                "autor": "Autor Revisado",
                "titulo": "Título Revisado",
                "ano": "2001",
            }
        }
    }), encoding="utf-8")

    meta = proc.extrair_metadados_pdf(pdf)

    assert meta["autor"] == "Autor Revisado"
    assert meta["ano"] == "2001"
    assert meta["fonte_metadados"] == "curadoria"
    assert proc.metadados_resolvidos(meta)


def test_reprocessamento_resolve_ano_conhecido_sd_e_registro_orfao(
    tmp_path, monkeypatch
):
    literatura, pendentes, revisados = _configurar_raiz(tmp_path, monkeypatch)
    pasta = literatura / "tema"
    pasta.mkdir()
    conhecido = pasta / "autor-desconhecido_documento_0000.pdf"
    sem_data = pasta / "instituicao_apostila-tecnica_0000.pdf"
    conhecido.write_bytes(b"documento-com-ano")
    sem_data.write_bytes(b"documento-sem-data")

    revisados.write_text(json.dumps({
        "documentos": {
            _sha256(conhecido.read_bytes()): {
                "autor": "Autor Correto",
                "titulo": "Documento Técnico",
                "ano": "2001",
            },
            _sha256(sem_data.read_bytes()): {
                "autor": "Instituição",
                "titulo": "Apostila Técnica",
                "ano": "0000",
                "ano_confirmado_ausente": True,
            },
        }
    }), encoding="utf-8")
    pendentes.write_text(json.dumps({
        conhecido.name: {
            "arquivo": f"literatura/tema/{conhecido.name}",
            "resolvido": False,
        },
        sem_data.name: {
            "arquivo": f"literatura/tema/{sem_data.name}",
            "resolvido": False,
        },
        "arquivo-antigo_0000.pdf": {
            "arquivo": "literatura/tema/arquivo-antigo_0000.pdf",
            "resolvido": False,
        },
    }), encoding="utf-8")

    mensagem = orquestrador.reprocessar_metadados_ruins()
    resultado = json.loads(pendentes.read_text(encoding="utf-8"))

    assert "1 arquivo(s) renomeado(s)" in mensagem
    assert "1 fonte(s) confirmada(s) como s.d." in mensagem
    assert "1 registro(s) órfão(s)" in mensagem
    assert (pasta / "correto_documento-tecnico_2001.pdf").is_file()
    assert (pasta / "instituicao_apostila-tecnica_0000.pdf").is_file()
    assert all(item["resolvido"] for item in resultado.values())
    assert resultado[sem_data.name]["ano_confirmado_ausente"] is True


def test_registro_de_pendencia_atualiza_item_existente(tmp_path, monkeypatch):
    literatura, pendentes, _revisados = _configurar_raiz(tmp_path, monkeypatch)
    pdf = literatura / "documento.pdf"
    pdf.write_bytes(b"pdf")
    pendentes.write_text(json.dumps({
        pdf.name: {
            "arquivo": "caminho/antigo.pdf",
            "autor_atual": "",
            "registrado": "2026-01-01T10:00",
            "resolvido": False,
        }
    }), encoding="utf-8")

    proc._registrar_pendencia(pdf, "Autor", "Título", "0000")
    item = json.loads(pendentes.read_text(encoding="utf-8"))[pdf.name]

    assert item["arquivo"] == "literatura/documento.pdf"
    assert item["autor_atual"] == "Autor"
    assert item["titulo_atual"] == "Título"
    assert item["registrado"] == "2026-01-01T10:00"
    assert item["arquivo_hash"] == _sha256(b"pdf")
