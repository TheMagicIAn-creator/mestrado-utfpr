"""Cofre de trechos verbatim (memória literal de código)."""

from __future__ import annotations

import src.conhecimento.snippets as sn

_CDF = """import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

media = 10.0
desvio_padrao = 2.0
x = np.linspace(media - 4 * desvio_padrao, media + 4 * desvio_padrao, 1000)
p = norm.cdf(x, loc=media, scale=desvio_padrao)
plt.plot(x, p)
plt.show()"""


def test_deteccao_salvar_vs_recuperar_nao_colidem():
    assert sn.quer_salvar_snippet("guarde este script para mim")
    assert sn.quer_salvar_snippet("salve esse código, por favor")
    # 'guardei'/'salvei' (passado) => recuperar, NAO salvar
    assert sn.quer_recuperar_snippet("me manda o script que guardei")
    assert sn.quer_recuperar_snippet("qual era o código que salvei?")
    # frases neutras nao disparam
    assert not sn.quer_salvar_snippet("escreva um script de CDF")
    assert not sn.quer_recuperar_snippet("escreva um script de CDF")


def test_extrai_bloco_de_codigo_com_e_sem_linguagem():
    texto = f"Aqui está:\n```python\n{_CDF}\n```\nPronto."
    blocos = sn.extrair_blocos_codigo(texto)
    assert len(blocos) == 1
    assert blocos[0]["linguagem"] == "python"
    assert blocos[0]["codigo"] == _CDF


def test_ultimo_bloco_vem_do_historico_quando_pergunta_nao_tem():
    historico = [
        {"role": "user", "content": "escreva um CDF"},
        {"role": "assistant", "content": f"```python\n{_CDF}\n```"},
    ]
    bloco = sn.ultimo_bloco_codigo("guarde este script", historico)
    assert bloco is not None and bloco["codigo"] == _CDF


def test_salvar_e_recuperar_verbatim(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "PASTA_SNIPPETS", tmp_path / "snippets")
    monkeypatch.setattr(sn, "ARQUIVO_SNIPPETS", tmp_path / "snippets" / "snippets.json")

    reg = sn.salvar_snippet(_CDF, linguagem="python")
    assert reg["codigo"] == _CDF

    recuperado = sn.recuperar_snippet("me manda o script que salvei")
    # BYTE A BYTE identico ao salvo
    assert recuperado["codigo"] == _CDF
    saida = sn.formatar_snippet_para_chat(recuperado)
    assert _CDF in saida  # aparece integral no bloco de resposta


def test_dedup_por_hash_nao_duplica(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "PASTA_SNIPPETS", tmp_path / "snippets")
    monkeypatch.setattr(sn, "ARQUIVO_SNIPPETS", tmp_path / "snippets" / "snippets.json")
    sn.salvar_snippet(_CDF, linguagem="python")
    sn.salvar_snippet(_CDF, linguagem="python")
    assert len(sn.carregar_snippets()) == 1


def test_dois_scripts_distintos_coexistem_e_recupera_por_nome(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "PASTA_SNIPPETS", tmp_path / "snippets")
    monkeypatch.setattr(sn, "ARQUIVO_SNIPPETS", tmp_path / "snippets" / "snippets.json")
    sn.salvar_snippet("def alpha():\n    return 1", linguagem="python", rotulo="alpha")
    sn.salvar_snippet("def beta():\n    return 2", linguagem="python", rotulo="beta")
    assert len(sn.carregar_snippets()) == 2
    r = sn.recuperar_snippet("me manda o codigo alpha que salvei")
    assert "alpha" in r["codigo"]


def test_recuperar_sem_nada_salvo_retorna_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sn, "PASTA_SNIPPETS", tmp_path / "snippets")
    monkeypatch.setattr(sn, "ARQUIVO_SNIPPETS", tmp_path / "snippets" / "snippets.json")
    assert sn.recuperar_snippet("me manda o script que salvei") is None
