"""Helpers leves compartilhados pelos modulos da interface Streamlit."""

from __future__ import annotations

from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.interface.streamlit_proxy import st


_logger = get_logger("interface.streamlit")

_CORES_ESTADO = {
    "ok": "#1baf7a",
    "alerta": "#eda100",
    "erro": "#e34948",
    "neutro": "#898781",
}


def _falha_recuperavel(
    operacao: str,
    exc: Exception,
    *,
    notificar: bool = False,
) -> None:
    """Registra fallback operacional e, quando necessario, avisa no app."""
    detalhe = mascarar_segredos(str(exc))
    _logger.warning("%s: %s", operacao, detalhe)
    if notificar and hasattr(st, "toast"):
        st.toast(f"{operacao}. Detalhes registrados no log.", icon="⚠️")


def _html_pensando(rotulo: str = "Pensando") -> str:
    """Monta o marcador HTML seguro exibido antes do primeiro token."""
    return f'<span class="alp-pensando">{rotulo}…</span>'


def _estado(rotulo: str, nivel: str = "ok") -> str:
    """Monta um indicador compacto de estado para a barra lateral."""
    cor = _CORES_ESTADO.get(nivel, _CORES_ESTADO["neutro"])
    return (
        f'<div class="alp-estado">'
        f'<span class="alp-ponto" style="background:{cor}"></span>{rotulo}</div>'
    )
