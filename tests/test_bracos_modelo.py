"""
Os dois braços do cerne, separados — e a separação sendo estrutural.

POR QUE ESTE TESTE EXISTE
=========================
O pesquisador foi explícito em 15/08/2026: *"Dividir o pipeline em Autoencoder
Denso (e seus resultados) e o Autoencoder LSTM (e seus resultados). Não quero
mistura entre eles, exceto que eu peça."*

Havia mistura em três níveis, e cada um precisa de guarda própria:

1. **Pasta.** Os dois gravavam em `resultados/macro/`, distinguidos por prefixo
   de arquivo. Pedir "os resultados do LSTM" obrigava a filtrar por nome.
2. **Identidade.** O nome de cada modelo era literal repetido em
   `macro_proposto.NOME`, `macro_ibrahim.NOME`, no dicionário `SLUG` de
   `macro_weibull` e nos rótulos de `contracts.py`. Quatro cópias.
3. **Execução.** Não dava para rodar só um: `montar_detectores` treinava o
   AE-LSTM sempre, mesmo quando só o denso interessava.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.ml.bracos_modelo import (
    BRACOS,
    DENSO,
    LSTM,
    PASTA_COMPARACAO,
    BracoModelo,
    identificar,
    por_id,
    preparar_pastas,
    resumo_markdown,
)

RAIZ = Path(__file__).resolve().parents[1]


# ── as pastas não se cruzam ────────────────────────────────────────────────

def test_cada_braco_tem_pasta_propria_e_disjunta():
    assert DENSO.pasta != LSTM.pasta
    assert DENSO.pasta.name == "ae_denso"
    assert LSTM.pasta.name == "ae_lstm"
    # Nenhum é ancestral do outro: um não pode enxergar os artefatos do outro
    # por engano ao varrer a própria pasta.
    assert LSTM.pasta not in DENSO.pasta.parents
    assert DENSO.pasta not in LSTM.pasta.parents


def test_a_comparacao_mora_fora_das_pastas_dos_bracos():
    """Resultado de braço é de UM modelo. Cruzamento é outra coisa."""
    for braco in BRACOS:
        assert PASTA_COMPARACAO != braco.pasta
        assert braco.pasta not in PASTA_COMPARACAO.parents


def test_preparar_pastas_e_idempotente(tmp_path, monkeypatch):
    import src.ml.bracos_modelo as bm

    monkeypatch.setattr(bm, "PASTA_MODELOS", tmp_path / "modelos")
    monkeypatch.setattr(bm, "PASTA_COMPARACAO", tmp_path / "comparacao")
    monkeypatch.setattr(
        bm, "DENSO",
        BracoModelo(id="ae_denso", nome="x", cor="#000", arquitetura="a", papel="p"),
    )
    caminhos = bm.preparar_pastas()
    assert all(Path(c).is_dir() for c in caminhos.values())
    bm.preparar_pastas()   # segunda vez não pode estourar


# ── a identidade é única ───────────────────────────────────────────────────

def test_os_nomes_historicos_continuam_reconhecidos():
    """Artefato já publicado grava o nome longo; ele não pode virar órfão."""
    assert por_id("Proposto (AE denso + MSE p99)") is DENSO
    assert por_id("Ibrahim 2022 (AE-LSTM temporal)") is LSTM
    assert por_id("proposto") is DENSO
    assert por_id("ibrahim") is LSTM


def test_id_desconhecido_estoura_dizendo_os_validos():
    with pytest.raises(KeyError) as erro:
        por_id("ae_transformer")
    assert "ae_denso" in str(erro.value) and "ae_lstm" in str(erro.value)


def test_identificar_nao_estoura_e_devolve_none():
    """Usado ao ler artefatos de outra natureza — ausência não é erro."""
    assert identificar("PCA linear") is None
    assert identificar("ae_denso") is DENSO


def test_ids_e_cores_sao_distintos():
    assert len({b.id for b in BRACOS}) == len(BRACOS)
    assert len({b.cor for b in BRACOS}) == len(BRACOS)
    assert len({b.nome for b in BRACOS}) == len(BRACOS)


def test_o_braco_e_imutavel():
    """Identidade de modelo não pode ser remendada em tempo de execução."""
    with pytest.raises(Exception):
        DENSO.id = "outro"


# ── rodar um braço não pode acionar o outro ────────────────────────────────

def test_montar_detectores_respeita_a_selecao(monkeypatch):
    """O pedido literal: não misturar, exceto quando pedido.

    Se `montar_detectores` treinasse o AE-LSTM mesmo quando só o denso foi
    pedido, o custo e os artefatos do outro braço apareceriam sem ninguém ter
    pedido.
    """
    from src.ml import bracos_modelo, macro_weibull

    construidos = []

    def falso_scorer(braco, janelas):
        construidos.append(braco.id)
        return lambda janelas_: []

    monkeypatch.setattr(bracos_modelo, "construir_scorer", falso_scorer)

    macro_weibull.montar_detectores([], [DENSO])
    assert construidos == ["ae_denso"], "o LSTM não podia ter sido construído"

    construidos.clear()
    macro_weibull.montar_detectores([], None)
    assert construidos == ["ae_denso", "ae_lstm"], "sem seleção, rodam os dois"


def test_o_bloco_carrega_o_id_do_braco():
    """Sem o id no artefato, quem lê depois volta a adivinhar pelo nome."""
    fonte = (RAIZ / "src/ml/macro_weibull.py").read_text(encoding="utf-8")
    assert 'bloco["braco_id"] = detector["braco"].id' in fonte


def test_a_cli_permite_rodar_um_braco_so():
    fonte = (RAIZ / "src/ml/macro_weibull.py").read_text(encoding="utf-8")
    assert '"--braco"' in fonte
    assert "ae_denso | ae_lstm" in fonte


# ── ninguém pode redeclarar a identidade ───────────────────────────────────

CONSUMIDORES = (
    "src/ml/macro_weibull.py",
    "src/ml/macro_proposto.py",
    "src/ml/macro_ibrahim.py",
)


@pytest.mark.parametrize("caminho", CONSUMIDORES)
def test_ninguem_remonta_o_mapa_de_modelos(caminho):
    """Detecta um dicionário literal mapeando nome de modelo para pasta/slug.

    Era exatamente o `SLUG = {...}` de `macro_weibull`: a quarta cópia da
    identidade. Quem precisa da pasta pergunta ao braço.
    """
    arvore = ast.parse((RAIZ / caminho).read_text(encoding="utf-8"))
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Dict):
            continue
        chaves = [
            k.value for k in no.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
        suspeitas = [c for c in chaves if "AE " in c or "AE-LSTM" in c]
        assert len(suspeitas) < 2, (
            f"{caminho}:{no.lineno} remonta o mapa de modelos "
            f"({suspeitas}). Use bracos_modelo — quatro cópias desta "
            "identidade já divergiram."
        )


def test_o_resumo_explica_a_separacao_ao_pesquisador():
    texto = resumo_markdown()
    assert "resultados/modelos/ae_denso/" in texto
    assert "resultados/modelos/ae_lstm/" in texto
    assert "só são geradas quando pedidas" in texto


def test_pastas_reais_sao_criadas_sob_resultados():
    caminhos = preparar_pastas()
    assert set(caminhos) == {"ae_denso", "ae_lstm", "comparacao"}
    for caminho in caminhos.values():
        assert caminho.is_dir()
        assert caminho.is_relative_to(RAIZ / "resultados")
