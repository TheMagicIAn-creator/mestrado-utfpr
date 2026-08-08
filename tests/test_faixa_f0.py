"""
A faixa de busca de F0 tem de cobrir o dataset — e não mais que isso.

Achado da auditoria (docs/auditoria_parametros.md §1): a busca era [20, 100] Hz,
dimensionada para a rede brasileira de 60 Hz. O Paderborn é bancada de
acionamento de motor de indução com velocidade variável, e Stender, Wallscheid
& Böcker (2020) registram p = 2 pares de polos e n ∈ [404; 3232] 1/min — ou
seja, fundamental elétrica em [13,5; 107,7] Hz.

A faixa antiga cortava AS DUAS PONTAS. A mediana de F0 medida no bloco de teste
foi 100,19 Hz, encostada no teto — assinatura de estimador saturado.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.features_ca import F0_MAX, F0_MIN, FS, estimar_f0

# Da Tab. de parâmetros de Stender et al. (2020).
P_PARES_POLOS = 2
ROTACAO_MIN, ROTACAO_MAX = 404, 3232


def _f_eletrica(rpm: float) -> float:
    return rpm / 60.0 * P_PARES_POLOS


def _espectro(f0: float, n: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / FS
    x = (np.sin(2 * np.pi * f0 * t)
         + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
         + 0.3 * np.sin(2 * np.pi * 3 * f0 * t))
    return np.fft.rfftfreq(n, 1 / FS), np.abs(np.fft.rfft(x * np.hanning(n)))


def test_faixa_cobre_a_fisica_do_dataset():
    """[13,5; 107,7] Hz precisa caber dentro de [F0_MIN, F0_MAX]."""
    assert F0_MIN < _f_eletrica(ROTACAO_MIN), "piso corta a rotação mínima"
    assert F0_MAX > _f_eletrica(ROTACAO_MAX), "teto corta a rotação máxima"


def test_faixa_nao_e_larga_o_bastante_para_travar_no_segundo_harmonico():
    """Faixa folgada demais deixa o estimador confundir 2·f0 com f0.

    O briefing de 06/08 sugeria [20, 384] Hz. 384 Hz é 3,6× a fundamental
    máxima da máquina — e como as features harmônicas são ancoradas em F0, um
    F0 dobrado corrompe o vetor inteiro.
    """
    assert F0_MAX < 2 * _f_eletrica(ROTACAO_MAX), (
        "teto acima de 2× a fundamental máxima admite o 2º harmônico")


@pytest.mark.parametrize("rpm", [ROTACAO_MIN, 1000, 1500, 3000, ROTACAO_MAX])
def test_estimador_acerta_em_toda_a_faixa_de_rotacao(rpm):
    f0 = _f_eletrica(rpm)
    est = estimar_f0(*_espectro(f0))
    assert abs(est - f0) < 0.10 * f0, (
        f"n={rpm} 1/min (f0={f0:.1f} Hz): estimou {est:.2f} Hz")


def test_nao_satura_no_teto_da_faixa():
    """O modo de falha original: F0 real acima do teto vira o próprio teto."""
    f0 = _f_eletrica(ROTACAO_MAX)          # 107,7 Hz
    est = estimar_f0(*_espectro(f0))
    assert est < F0_MAX - 1.0, "estimativa encostada no teto"


def test_faixa_antiga_falharia_nas_duas_pontas():
    """Documenta o defeito, para ninguém restaurar [20, 100] por engano."""
    assert _f_eletrica(ROTACAO_MIN) < 20.0    # 13,5 Hz — abaixo do piso antigo
    assert _f_eletrica(ROTACAO_MAX) > 100.0   # 107,7 Hz — acima do teto antigo


def test_faixa_hz_explicito_preserva_o_comportamento_antigo():
    """Reprodução de rodadas anteriores continua possível."""
    freqs, amps = _espectro(50.0)
    assert estimar_f0(freqs, amps, f0_nominal=60.0, faixa_hz=40.0) > 0
