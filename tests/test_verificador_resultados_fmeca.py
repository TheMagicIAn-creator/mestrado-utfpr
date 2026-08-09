import subprocess
import sys
from pathlib import Path

from scripts.verificar_resultados_fmeca import _smd_calculada, _validar_split_temporal


def test_verificador_executa_diretamente_fora_da_raiz(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    processo = subprocess.run(
        [sys.executable, str(raiz / "scripts/verificar_resultados_fmeca.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert processo.returncode == 0, processo.stdout + processo.stderr
    assert "APROVADO" in processo.stdout


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


def test_split_contiguo_com_purga_e_valido():
    split = {
        "n_janelas": 100,
        "purge_janelas": 2,
        "limites": {"treino": [0, 60], "val": [62, 80], "teste": [82, 100]},
    }

    assert _validar_split_temporal(split) == (True, "ok")


def test_split_intercalado_com_purga_e_valido():
    split = {
        "estrategia": "blocos_intercalados",
        "n_janelas": 60,
        "purge_janelas": 2,
        "limites": {
            "treino": [[0, 10], [24, 34], [34, 44]],
            "val": [[12, 22]],
            "teste": [[46, 60]],
        },
    }

    assert _validar_split_temporal(split) == (True, "ok")


def test_split_intercalado_rejeita_sobreposicao():
    split = {
        "estrategia": "blocos_intercalados",
        "purge_janelas": 2,
        "limites": {
            "treino": [[0, 12]],
            "val": [[10, 20]],
            "teste": [[22, 30]],
        },
    }

    valido, motivo = _validar_split_temporal(split)
    assert valido is False
    assert "sobrepõem" in motivo


def test_split_intercalado_rejeita_fronteira_sem_purga():
    split = {
        "estrategia": "blocos_intercalados",
        "purge_janelas": 2,
        "limites": {
            "treino": [[0, 10]],
            "val": [[11, 20]],
            "teste": [[22, 30]],
        },
    }

    valido, motivo = _validar_split_temporal(split)
    assert valido is False
    assert "purga" in motivo
