"""Guardas das decisões metodológicas aprovadas em 2026-09-01."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.config import RAIZ_PROJETO
from src.ml.confiabilidade_componentes import FMECA_COMPONENTS, methodology
from src.ml.dados_gpvs import FAULT_CONTRACTS
from src.ml.sensibilidade_escore import (
    SENSITIVITY_PERCENTILES,
    SENSITIVITY_TOP_K,
)


ROOT = Path(RAIZ_PROJETO)


def test_sensitivity_grid_is_exactly_three_by_three():
    assert SENSITIVITY_TOP_K == (5, 10, 20)
    assert SENSITIVITY_PERCENTILES == (99.0, 99.5, 99.9)
    assert len(SENSITIVITY_TOP_K) * len(SENSITIVITY_PERCENTILES) == 9


def test_native_gpvs_mapping_preserves_functional_control_scope():
    assert FAULT_CONTRACTS[1]["fmeca_scope"] == "igbt"
    assert FAULT_CONTRACTS[2]["fmeca_scope"] == "sensor_feedback_system"
    assert FAULT_CONTRACTS[6]["fmeca_scope"] == "inverter_control_system"
    assert FAULT_CONTRACTS[7]["fmeca_scope"] == "inverter_control_system"
    assert FAULT_CONTRACTS[6]["scope_relation"] == "functional_control_anomaly"
    assert FAULT_CONTRACTS[7]["physical_component_failure"] is False


def test_current_fmeca_has_no_fabricated_scores_or_legacy_components():
    component_ids = {component.component_id for component in FMECA_COMPONENTS}
    assert component_ids == {
        "igbt",
        "sensor_feedback_system",
        "inverter_control_system",
    }
    for component in FMECA_COMPONENTS:
        assert component.status == "awaiting_user_fmeca"
        assert component.severity is None
        assert component.occurrence is None
        assert component.detectability is None
        assert component.npr is None


def test_current_publication_does_not_depend_on_revoked_projection_fields():
    serialized = json.dumps(methodology(), ensure_ascii=False).lower()
    for revoked in ("pod_mon", "d_mon", "d_proj", "npr_proj"):
        assert revoked not in serialized


def test_current_scientific_docs_do_not_restore_revoked_projection():
    paths = (
        ROOT / "docs" / "fmeca.md",
        ROOT / "docs" / "metodologia_ml.md",
        ROOT / "docs" / "confiabilidade_fisica.md",
        ROOT / "docs" / "mapa_de_resultados.md",
        ROOT / "docs" / "reproducibilidade.md",
        ROOT / "docs" / "glossario.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    for revoked in ("pod_mon", "d_mon", "d_proj", "npr_proj"):
        assert revoked not in text
