"""Leitura e montagem segura de anexos sem rede."""

from src.conhecimento import leitor_anexos as la


def test_ler_texto_utf8_e_arquivo_vazio():
    texto = la.ler_anexo("nota.md", "Tensão média".encode())
    vazio = la.ler_anexo("vazio.txt", b"")

    assert texto["tipo"] == "texto"
    assert texto["texto"] == "Tensão média"
    assert vazio["tipo"] == "erro"
    assert vazio["erro"] == "arquivo vazio"


def test_binario_desconhecido_nao_e_injetado_no_prompt():
    resultado = la.ler_anexo("modelo.bin", b"abc\x00def")
    bloco = la.montar_bloco_texto_anexos([resultado])

    assert resultado["tipo"] == "erro"
    assert "tipo nao suportado" in resultado["erro"]
    assert "Anexo nao lido" in bloco


def test_limite_global_trunca_anexos(monkeypatch):
    monkeypatch.setattr(la, "MAX_CHARS_TOTAL", 8)
    anexos = la.ler_anexos([
        ("a.txt", b"123456"),
        ("b.txt", b"abcdef"),
        ("c.txt", b"xyz"),
    ])

    assert anexos[0]["texto"] == "123456"
    assert "truncado" in anexos[1]["texto"]
    assert "omitido" in anexos[2]["texto"]


def test_imagem_so_e_marcada_quando_tem_conteudo():
    anexos = [
        {"tipo": "imagem", "imagem_b64": "YWJj", "nome": "x.png", "resumo": "1x1"},
        {"tipo": "erro", "imagem_b64": "", "nome": "y.png"},
    ]

    assert la.tem_imagem(anexos) is True
    assert "segue anexada" in la.montar_bloco_texto_anexos(anexos, suporta_imagem=True)
    assert "nao le imagens" in la.montar_bloco_texto_anexos(anexos, suporta_imagem=False)
