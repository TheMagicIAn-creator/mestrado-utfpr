"""
Sprint 2 — schema e calibração das falhas sintéticas (item 4.4).

Cada falha injetada deve declarar proveniência física (hipótese, sinais,
fórmula, severidade, fonte, limitações) e nível de evidência E2. A falha de
sensor deve trazer a ressalva de CALIBRAÇÃO FÍSICA (proxy de ruído).
"""

from src.ml.injecao_falhas import FALHAS

CAMPOS = (
    "evidence_level", "hipotese_fisica", "sinais", "formula",
    "severity_definition", "source", "limitations",
)


def test_toda_falha_tem_schema_e2():
    assert len(FALHAS) >= 3
    for f in FALHAS:
        for campo in CAMPOS:
            assert f.get(campo), f"falha '{f['id']}' sem campo '{campo}'"
        assert f["evidence_level"] == "E2"
        assert isinstance(f["sinais"], list) and f["sinais"]
        assert isinstance(f["limitations"], list) and f["limitations"]


def test_sensor_exige_calibracao_fisica():
    sensor = next(f for f in FALHAS if f["id"] == "sensor")
    txt = " ".join(sensor["limitations"]).lower()
    assert "calibra" in txt           # ressalva de calibração física
    assert "proxy" in txt or "sensibilidade" in txt
