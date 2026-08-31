from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "src/webapp/templates/index.html").read_text(encoding="utf-8")
CSS = (ROOT / "src/webapp/static/styles.css").read_text(encoding="utf-8")
JS = (ROOT / "src/webapp/static/app.js").read_text(encoding="utf-8")


def test_primeira_tela_e_chat_first_sem_dashboard():
    assert 'id="view-chat"' in HTML
    assert 'class="view chat-view is-active"' in HTML
    assert 'id="conversation"' in HTML
    assert 'id="chat-form"' in HTML
    assert 'id="send-button"' in HTML
    assert 'data-view-panel="library"' in HTML
    assert 'data-view-panel="results"' not in HTML
    assert 'data-result-tab' not in HTML
    assert "Confiabilidade e manutenção</h1>" not in HTML
    assert "Denso × AE-LSTM</" not in HTML


def test_historico_tem_novo_chat_busca_renomeacao_arquivo_e_exclusao():
    for element_id in (
        "new-chat",
        "history-query",
        "rename-chat",
        "archive-chat",
        "delete-chat",
        "export-chat",
        "conversation-rename-dialog",
        "conversation-delete-dialog",
    ):
        assert f'id="{element_id}"' in HTML
    assert "/api/conversations" in JS
    assert 'data-conversation-action="${action}"' in JS
    assert 'historyActionButton("rename"' in JS
    assert 'historyActionButton(archived ? "restore" : "archive"' in JS
    assert 'historyActionButton("delete"' in JS
    assert "syncConversations" in JS


def test_referencias_e_unica_area_secundaria_e_permanece_editavel():
    assert "Referências" in HTML
    assert 'id="library-add"' in HTML
    assert 'id="library-add-dialog"' in HTML
    assert 'id="library-edit-dialog"' in HTML
    assert "/api/library" in JS
    assert "data-edit-source" in JS
    assert "data-reindex-source" in JS
    assert "/api/results/e3" not in JS
    assert "/api/reliability" not in JS
    assert "results-charts.js" not in JS
    assert "d3.min.js" not in JS


def test_chat_usa_streaming_cancelamento_retentativa_exportacao_e_imagens():
    assert 'fetch("/api/chat/stream"' in JS
    assert "AbortController" in JS
    assert 'event === "status"' in JS
    assert 'event === "delta"' in JS
    assert 'event === "done"' in JS
    assert "data-retry-index" in JS
    assert "exportConversation" in JS
    assert "URL.createObjectURL" in JS
    assert "appendResponseAssets" in JS
    assert 'id="media-dialog"' in HTML


def test_matematica_e_local_e_nao_interfere_em_codigo():
    assert "/static/vendor/katex/katex.min.css" in HTML
    assert "/static/vendor/katex/katex.min.js" in HTML
    assert "/static/vendor/katex/auto-render.min.js" in HTML
    assert "window.renderMathInElement" in JS
    assert '"pre", "code"' in JS
    assert ".message-content .katex-display" in CSS


def test_layout_tem_dimensoes_estaveis_e_compositor_sem_sobreposicao():
    assert "height: 100dvh" in CSS
    assert ".chat-view" in CSS and "flex-direction: column" in CSS
    assert ".conversation" in CSS and "flex: 1" in CSS
    assert ".composer-shell" in CSS and "flex: 0 0 auto" in CSS
    assert "grid-template-columns: 40px minmax(0, 1fr) 40px" in CSS
    assert "max-height: 180px" in CSS


def test_mobile_usa_drawer_e_respeita_area_segura():
    assert "@media (max-width: 760px)" in CSS
    assert "body.sidebar-open .sidebar" in CSS
    assert "translateX(-102%)" in CSS
    assert "env(safe-area-inset-bottom)" in CSS
    assert 'id="sidebar-open"' in HTML
    assert 'id="sidebar-scrim"' in HTML


def test_tema_movimento_reduzido_e_acessibilidade():
    assert 'data-theme="light"' in HTML
    assert "prefers-reduced-motion: reduce" in CSS
    assert 'class="skip-link"' in HTML
    assert 'aria-live="polite"' in HTML
    assert "focus-visible" in CSS
    assert "window.lucide.createIcons" in JS


def test_interface_nao_tem_prompts_que_forcam_assunto_ou_resultados():
    assert "data-prompt" not in HTML
    assert "prompt-grid" not in HTML
    assert "data-prompt" not in JS
    source = f"{HTML}\n{CSS}\n{JS}".casefold()
    for legacy in (
        "webapp_v2",
        "autoencoder_v2",
        "resultados/v2",
        "resultados/macro",
        "plotly",
    ):
        assert legacy not in source


def test_tipografia_nao_escala_com_viewport_e_raios_sao_contidos():
    assert not re.search(r"font-size\s*:\s*[^;]*(?:vw|vh|vmin|vmax)", CSS)
    assert "letter-spacing: 0" in CSS
    assert "letter-spacing: -" not in CSS
    radii = [int(value) for value in re.findall(r"border-radius:\s*(\d+)px", CSS)]
    assert radii and max(radii) <= 8
