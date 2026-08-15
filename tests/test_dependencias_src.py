"""Restrições arquiteturais mínimas entre os pacotes de src/."""

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_ml_nao_importa_conhecimento():
    achados = []
    for arquivo in (RAIZ / "src/ml").glob("*.py"):
        fonte = arquivo.read_text(encoding="utf-8")
        if "src.conhecimento" in fonte:
            achados.append(arquivo.name)
    assert not achados, f"dependência reversa ml -> conhecimento: {achados}"


def test_escritas_chromadb_usam_lock_canonico():
    contratos = {
        "src/conhecimento/indexador.py": ("indexar_pdf_unico", "indexar_sessao"),
        "src/conhecimento/indice_portatil.py": ("importar_colecao",),
        "src/conhecimento/obsidian.py": ("sincronizar_obsidian",),
        "src/conhecimento/consolidar_memoria.py": ("atualizar_chromadb",),
    }
    for relativo, funcoes in contratos.items():
        fonte = (RAIZ / relativo).read_text(encoding="utf-8")
        assert "from src.conhecimento.index_lock import lock_indexacao" in fonte
        for funcao in funcoes:
            assert f"def {funcao}(" in fonte


def _captura_excecao_ampla(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    tipos = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(tipo, ast.Name) and tipo.id in {"Exception", "BaseException"}
               for tipo in tipos)


def test_excecao_ampla_nao_pode_ser_descartada_com_pass():
    silenciosos = []
    for arquivo in (RAIZ / "src").rglob("*.py"):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        for handler in (n for n in ast.walk(arvore) if isinstance(n, ast.ExceptHandler)):
            if (_captura_excecao_ampla(handler)
                    and len(handler.body) == 1
                    and isinstance(handler.body[0], ast.Pass)):
                silenciosos.append(
                    f"{arquivo.relative_to(RAIZ).as_posix()}:{handler.lineno}"
                )
    assert not silenciosos, f"falhas amplas descartadas silenciosamente: {silenciosos}"


