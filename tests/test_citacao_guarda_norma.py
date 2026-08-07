"""
Regressões da guarda de citação — o caso que ela existia para pegar e não pegava.

Achado da auditoria (docs/auditoria_total_src.md §11): a guarda comparava a
norma citada contra o TEXTO DOS TRECHOS recuperados. Como artigos de FMEA e RCM
mencionam IEC 60812 e ISO 14224 o tempo todo, bastava um trecho citar a norma
para que uma citação inventada — com cláusula e página inverificáveis — passasse
sem alerta. É o cenário de risco acadêmico direto, e o docstring do próprio
módulo o dá como exemplo do que deveria capturar.
"""

from __future__ import annotations

import pytest

from src.core.citacao_guarda import alerta_citacao_infundada

# Fonte real do acervo cujo TRECHO menciona uma norma que NÃO está indexada.
_FONTE_QUE_MENCIONA_NORMA = {
    "c1": ('Carpinetti L (2016) — p. 44 — trecho: "A analise FMEA segue a '
           'IEC 60812 para definir severidade e ocorrencia."'),
}


def test_norma_fabricada_dispara_mesmo_com_a_norma_citada_dentro_do_trecho():
    """O caso central: a norma aparece no trecho, mas o PDF dela não está na base.

    Mencionada dentro de outro artigo não é lastro — é hearsay. A cláusula e a
    página seguem inverificáveis.
    """
    aviso = alerta_citacao_infundada(
        'Conforme a IEC 60812:2018, Clausula 7.3.3, p. 27: "o NPR deve ser..."',
        _FONTE_QUE_MENCIONA_NORMA,
    )
    assert "IEC 60812" in aviso
    assert "NAO verificadas" in aviso or "NÃO verificadas" in aviso


def test_norma_realmente_indexada_nao_dispara():
    """Se o documento normativo É a fonte, a citação está lastreada."""
    fontes = {"c1": "IEC 60812 (2018) — p. 27 — trecho: \"...\""}
    assert alerta_citacao_infundada(
        "Conforme a IEC 60812:2018, p. 27, o NPR...", fontes) == ""


# ── falso alarme que ensinava a ignorar o aviso verdadeiro ──────────────────

@pytest.mark.parametrize("frase", [
    "Ibrahim et al. proposent un autoencodeur en 2022 pour la detection.",
    "Segun Ghoneim en 2021, el modelo alcanza AUC alta.",
    "Le modele a ete entraine en 2024 sur le jeu de Paderborn.",
])
def test_preposicao_en_seguida_de_ano_nao_e_norma(frase):
    """FR/ES são idiomas que o CLAUDE.md manda o agente usar.

    Com `EN` sob `re.I`, "en 2022" virava "norma técnica fabricada". Aviso que
    grita em texto legítimo ensina a ignorar o aviso verdadeiro.
    """
    assert alerta_citacao_infundada(frase, _FONTE_QUE_MENCIONA_NORMA) == ""


def test_EN_em_caixa_alta_continua_sendo_norma():
    """A norma europeia real não pode deixar de ser detectada."""
    aviso = alerta_citacao_infundada(
        "Segundo a EN 50438, p. 12, o inversor deve...",
        _FONTE_QUE_MENCIONA_NORMA)
    assert "EN 50438" in aviso


# ── truncamento que escondia evidência ─────────────────────────────────────

def test_mais_de_tres_normas_fabricadas_anuncia_quantas_ficaram_de_fora():
    """Truncar em 3 sem dizer que há mais escondia parte da fabricação."""
    resposta = ("Ver IEC 60812, ISO 14224, IEEE 1547, ABNT NBR 16274 e "
                "IEC 61727 para os limites.")
    aviso = alerta_citacao_infundada(resposta, _FONTE_QUE_MENCIONA_NORMA)
    assert "e mais 2" in aviso


def test_ate_tres_normas_nao_anuncia_resto():
    aviso = alerta_citacao_infundada(
        "Ver ISO 14224 e IEEE 1547.", _FONTE_QUE_MENCIONA_NORMA)
    assert "e mais" not in aviso
