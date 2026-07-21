from scripts.verificar_resultados_fmeca import _smd_calculada


def test_smd_calculada_distingue_pontual_de_conservadora():
    taxas = {"0.1": 0.80, "0.2": 0.95, "0.3": 1.0}
    intervalos = {
        "0.1": {"low": 0.60, "high": 0.90},
        "0.2": {"low": 0.88, "high": 0.99},
        "0.3": {"low": 0.95, "high": 1.0},
    }
    assert _smd_calculada(taxas, 0.95, conservadora=False) == 0.2
    assert _smd_calculada(intervalos, 0.95, conservadora=True) == 0.3


def test_smd_calculada_retorna_none_quando_alvo_nao_e_atingido():
    taxas = {"0.1": 0.10, "0.5": 0.70, "1.0": 0.94}
    assert _smd_calculada(taxas, 0.95, conservadora=False) is None
