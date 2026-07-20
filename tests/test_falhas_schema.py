"""
Sprint 2 — schema e calibração das falhas sintéticas (item 4.4).

Cada falha injetada deve declarar proveniência física (hipótese, sinais,
fórmula, severidade, fonte, limitações) e nível de evidência E2, além dos
índices FMECA (S, O, D, NPR=S×O×D). A falha do Contator AC (proxy de ruído)
deve trazer a ressalva de CALIBRAÇÃO FÍSICA.
"""

from src.ml.injecao_falhas import FALHAS

CAMPOS = (
    "evidence_level", "hipotese_fisica", "sinais", "formula",
    "severity_definition", "source", "limitations",
)
COMPONENTES = {"contator_ac", "igbt", "fusivel_ac"}


def test_toda_falha_tem_schema_e2():
    assert len(FALHAS) >= 3
    for f in FALHAS:
        for campo in CAMPOS:
            assert f.get(campo), f"falha '{f['id']}' sem campo '{campo}'"
        assert f["evidence_level"] == "E2"
        assert isinstance(f["sinais"], list) and f["sinais"]
        assert isinstance(f["limitations"], list) and f["limitations"]


def test_ids_sao_os_componentes_fmeca():
    assert {f["id"] for f in FALHAS} == COMPONENTES


def test_npr_e_produto_sod_nunca_d_isolado():
    """NPR = S×O×D (FMECA). Nenhum índice pode ser None nem NPR = D."""
    for f in FALHAS:
        s, o, d, npr = f["s"], f["o"], f["d"], f["npr"]
        assert None not in (s, o, d, npr), f"'{f['id']}' com índice None"
        assert npr == s * o * d, f"'{f['id']}': NPR {npr} != S×O×D {s*o*d}"
        assert npr != d, f"'{f['id']}': NPR não pode ser o D isolado"


def test_contator_exige_calibracao_fisica():
    contator = next(f for f in FALHAS if f["id"] == "contator_ac")
    txt = " ".join(contator["limitations"]).lower()
    assert "calibra" in txt           # ressalva de calibração física
    assert "proxy" in txt or "sensibilidade" in txt or "campo" in txt
