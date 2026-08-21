from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src/webapp/templates/index.html").read_text(encoding="utf-8")
CSS = (ROOT / "src/webapp/static/styles.css").read_text(encoding="utf-8")
JS = (ROOT / "src/webapp/static/app.js").read_text(encoding="utf-8")


def test_primeira_tela_e_chat_utilizavel_sem_dashboard_bloqueante():
    assert 'id="view-chat"' in HTML
    assert 'class="view chat-view is-active"' in HTML
    assert 'id="conversation"' in HTML
    assert 'id="chat-form"' in HTML
    assert 'id="send-button"' in HTML
    assert "loading-state" not in HTML
    assert "/vendor/plotly" not in HTML


def test_navegacao_cientifica_cobre_as_quatro_areas_canonicas():
    assert 'data-view="e3"' in HTML
    assert 'data-view="e2"' in HTML
    assert 'data-view="reliability"' in HTML
    assert 'data-view="sources"' in HTML
    assert "/api/results/e3" in JS
    assert "/api/results/e2" in JS
    assert "/api/reliability" in JS
    assert "/api/sources" in JS


def test_figuras_sao_sob_demanda_com_zoom_pdf_e_lazy_loading():
    assert 'loading="lazy"' in JS
    assert 'data-open-figure' in JS
    assert 'id="image-dialog"' in HTML
    assert 'id="zoom-in"' in HTML
    assert 'id="zoom-out"' in HTML
    assert 'id="dialog-pdf"' in HTML
    assert "figureCards(data.figures)" in JS


def test_chat_usa_streaming_cancelamento_retentativa_e_exportacao():
    assert 'fetch("/api/chat/stream"' in JS
    assert "AbortController" in JS
    assert 'event === "status"' in JS
    assert 'event === "delta"' in JS
    assert 'event === "done"' in JS
    assert "data-retry-index" in JS
    assert "exportConversation" in JS
    assert "URL.createObjectURL" in JS


def test_layout_tem_dimensoes_estaveis_e_compositor_sem_sobreposicao():
    assert "height: 100dvh" in CSS
    assert ".chat-view" in CSS and "flex-direction: column" in CSS
    assert ".conversation" in CSS and "flex: 1" in CSS
    assert ".composer-region" in CSS and "flex: 0 0 auto" in CSS
    assert "grid-template-columns: 40px minmax(0, 1fr) 42px" in CSS
    assert "max-height: 170px" in CSS
    assert "position: fixed" not in CSS[CSS.index(".composer-region") : CSS.index(".stream-status")]


def test_mobile_usa_drawer_e_respeita_area_segura():
    assert "@media (max-width: 760px)" in CSS
    assert "body.sidebar-open .sidebar" in CSS
    assert "translateX(-102%)" in CSS
    assert "env(safe-area-inset-bottom)" in CSS
    assert 'id="sidebar-open"' in HTML
    assert 'id="sidebar-scrim"' in HTML


def test_tema_reducao_de_movimento_e_acessibilidade_estao_presentes():
    assert 'data-theme="light"' in HTML
    assert "prefers-reduced-motion: reduce" in CSS
    assert 'class="skip-link"' in HTML
    assert 'aria-live="polite"' in HTML
    assert "focus-visible" in CSS
    assert "window.lucide.createIcons" in JS


def test_tipografia_nao_escala_com_viewport_e_letter_spacing_e_neutro():
    assert not re.search(r"font-size\s*:\s*[^;]*(?:vw|vh|vmin|vmax)", CSS)
    assert "letter-spacing: 0" in CSS
    assert "letter-spacing: -" not in CSS


def test_interface_nao_expoe_nomes_ou_resultados_legados():
    source = f"{HTML}\n{CSS}\n{JS}".casefold()
    for legacy in (
        "webapp_v2",
        "autoencoder_v2",
        "autoencoder v2",
        "resultados/v2",
        "resultados/macro",
        "ae × pca",
        "plotly",
    ):
        assert legacy not in source


def test_cartoes_repetidos_mantem_raio_maximo_de_oito_pixels():
    radii = [int(value) for value in re.findall(r"border-radius:\s*(\d+)px", CSS)]
    assert radii
    assert max(radii) <= 8
