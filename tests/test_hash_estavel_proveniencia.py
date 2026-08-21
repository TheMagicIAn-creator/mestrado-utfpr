"""
Rerun determinístico não pode invalidar a proveniência.

POR QUE ESTE TESTE EXISTE
=========================
Em 15/08/2026 o pesquisador re-rodou `exec_etapa_isolada autoencoder`. O treino
é determinístico por semente: o limiar saiu bit a bit idêntico,
`0.8577015399932861`. A ÚNICA diferença no arquivo foi ``data_treino``, de 12/08
para 15/08 — um relógio.

Mesmo assim o SHA-256 mudou e **dois testes passaram a reprovar no `main`**:
o inventário de artefatos e o verificador da FMECA, ambos acusando "hash
divergente" para o `limiar.json`.

O dano real não é o CI vermelho. É o hábito: uma cadeia de proveniência que
acusa divergência a cada rerun ensina o leitor a ignorar o alarme — justamente
o alarme que precisa ser levado a sério quando um artefato muda de verdade.

Além disso, a regra "que hash usar para qual arquivo" chegou a existir em
quatro cópias. Cópia de regra deriva: quando a fonte passou a hashear JSON sem
campos de data, os verificadores antigos continuaram nos bytes e acusaram
divergência inexistente. O auditor canônico delega a `funcao_de_hash_para`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.ml.proveniencia import (
    CAMPOS_VOLATEIS,
    funcao_de_hash_para,
    sha256_arquivo,
    sha256_arquivo_texto_normalizado,
    sha256_json_estavel,
)

RAIZ = Path(__file__).resolve().parents[1]


# ── o comportamento que faltava ────────────────────────────────────────────

def test_mudar_so_a_data_nao_muda_o_hash(tmp_path):
    """O caso exato: mesmo limiar, relógio diferente."""
    base = {"limiar": 0.8577015399932861, "metodo_escore": "mse"}

    antes = tmp_path / "antes.json"
    antes.write_text(json.dumps({**base, "data_treino": "2026-08-12T09:13:47"}),
                     encoding="utf-8")
    depois = tmp_path / "depois.json"
    depois.write_text(json.dumps({**base, "data_treino": "2026-08-15T14:22:12"}),
                      encoding="utf-8")

    assert sha256_json_estavel(antes) == sha256_json_estavel(depois)
    # E o hash antigo mudaria — é por isso que a troca era necessária.
    assert sha256_arquivo_texto_normalizado(antes) != \
        sha256_arquivo_texto_normalizado(depois)


def test_mudar_o_valor_cientifico_MUDA_o_hash(tmp_path):
    """A guarda não pode virar cegueira: conteúdo diferente tem de divergir."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"limiar": 0.8577, "data_treino": "x"}), encoding="utf-8")
    b.write_text(json.dumps({"limiar": 0.9123, "data_treino": "x"}), encoding="utf-8")

    assert sha256_json_estavel(a) != sha256_json_estavel(b)


def test_campo_volatil_aninhado_tambem_sai(tmp_path):
    """Os carimbos moram em blocos internos, não só na raiz."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"bloco": {"v": 1, "created_at": "2026-01-01"}}),
                 encoding="utf-8")
    b.write_text(json.dumps({"bloco": {"v": 1, "created_at": "2026-12-31"}}),
                 encoding="utf-8")

    assert sha256_json_estavel(a) == sha256_json_estavel(b)


def test_ordem_das_chaves_nao_altera_o_hash(tmp_path):
    """Reescrever o JSON com outra ordem não é mudança científica."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"x": 1, "y": 2}), encoding="utf-8")
    b.write_text(json.dumps({"y": 2, "x": 1}), encoding="utf-8")

    assert sha256_json_estavel(a) == sha256_json_estavel(b)


def test_json_invalido_cai_no_hash_textual(tmp_path):
    """Nunca devolver algo silenciosamente errado."""
    quebrado = tmp_path / "quebrado.json"
    quebrado.write_text("{ isto não é json", encoding="utf-8")

    assert sha256_json_estavel(quebrado) == \
        sha256_arquivo_texto_normalizado(quebrado)


def test_arquivo_ausente_devolve_none(tmp_path):
    assert sha256_json_estavel(tmp_path / "nao_existe.json") is None


# ── o roteamento é fonte única ─────────────────────────────────────────────

@pytest.mark.parametrize("nome,esperada", [
    ("limiar.json", sha256_json_estavel),
    ("tabela.csv", sha256_arquivo_texto_normalizado),
    ("relatorio.md", sha256_arquivo_texto_normalizado),
    ("modelo_autoencoder.pt", sha256_arquivo),
    ("normalizacao_baseline_gpvs.npz", sha256_arquivo),
    ("figura.png", sha256_arquivo),
])
def test_roteamento_por_natureza_do_arquivo(nome, esperada):
    assert funcao_de_hash_para(Path("qualquer") / nome) is esperada


def test_binario_cientifico_nunca_vai_pelo_caminho_textual():
    """Pesos e matrizes são bytes — normalizar EOL neles corromperia o hash."""
    for nome in ("modelo.pt", "scaler.pkl", "features.parquet", "dados.npz"):
        assert funcao_de_hash_para(Path(nome)) is sha256_arquivo


# ── a regra não pode ser copiada de novo ───────────────────────────────────

CONSUMIDORES = (
    "scripts/auditar_resultados.py",
)


@pytest.mark.parametrize("caminho", CONSUMIDORES)
def test_ninguem_reimplementa_a_escolha_de_hash(caminho):
    """Detecta a assinatura da cópia: um ternário sobre SUFIXOS_TEXTO_PORTAVEL.

    As três cópias existiam e as três derivaram juntas. Quem precisar do hash
    tem de chamar `funcao_de_hash_para`, não redecidir.
    """
    arquivo = RAIZ / caminho
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))

    for no in ast.walk(arvore):
        if not isinstance(no, ast.IfExp):
            continue
        nomes = {
            n.id for n in ast.walk(no)
            if isinstance(n, ast.Name)
        }
        assert "SUFIXOS_TEXTO_PORTAVEL" not in nomes, (
            f"{caminho}:{no.lineno} voltou a decidir o hash por conta própria. "
            "Use proveniencia.funcao_de_hash_para — três cópias desta regra já "
            "derivaram e acusaram divergência inexistente."
        )


def test_os_campos_volateis_estao_declarados_e_cobrem_o_caso_real():
    assert "data_treino" in CAMPOS_VOLATEIS, (
        "é o campo que quebrou a cadeia em 15/08/2026"
    )
    assert "created_at" in CAMPOS_VOLATEIS, "os manifestos usam created_at"


# ── os artefatos vigentes conferem ─────────────────────────────────────────

def test_manifestos_versionados_conferem_com_os_artefatos_no_disco():
    """Guarda de regressão do reparo: nenhum hash de JSON pode divergir.

    Se este teste reprovar depois de um rerun, a divergência é de CONTEÚDO —
    não mais de relógio — e merece investigação, não regravação.
    """
    divergentes = []
    for manifesto in sorted((RAIZ / "resultados/manifestos").glob("*.json")):
        dados = json.loads(manifesto.read_text(encoding="utf-8"))
        for bloco in ("input_artifacts", "output_artifacts", "code_dependencies"):
            for nome, esperado in (dados.get(bloco) or {}).items():
                if not esperado or not str(nome).endswith(".json"):
                    continue
                alvo = RAIZ / nome
                if not alvo.is_file():
                    continue
                if funcao_de_hash_para(alvo)(alvo) != esperado:
                    divergentes.append(f"{manifesto.name} -> {nome}")

    assert not divergentes, f"hashes de JSON divergentes: {divergentes}"
