from __future__ import annotations

import csv
import json

from scripts.auditar_artefatos_resultados import (
    construir_inventario,
    qualidade_estrutural,
    sha256_arquivo,
    sha256_manifesto,
)


def test_hash_de_manifesto_normaliza_quebras_de_linha(tmp_path):
    crlf = tmp_path / "crlf.md"
    lf = tmp_path / "lf.md"
    crlf.write_bytes(b"linha 1\r\nlinha 2\r\n")
    lf.write_bytes(b"linha 1\nlinha 2\n")

    assert sha256_arquivo(crlf) != sha256_arquivo(lf)
    assert sha256_manifesto(crlf) == sha256_manifesto(lf)


def test_qualidade_estrutural_detecta_json_nao_finito(tmp_path):
    caminho = tmp_path / "resultado.json"
    caminho.write_text(json.dumps({"metrica": float("nan")}), encoding="utf-8")
    assert qualidade_estrutural(caminho) == "nao_finito"


def test_qualidade_estrutural_detecta_csv_irregular(tmp_path):
    caminho = tmp_path / "resultado.csv"
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["a", "b"])
        escritor.writerow([1])
    assert qualidade_estrutural(caminho) == "linha_irregular"


def test_inventario_atual_nao_tem_hash_divergente():
    inventario = construir_inventario()
    assert inventario
    assert not [
        item for item in inventario
        if item["hash_manifesto"] == "divergente"
    ]
