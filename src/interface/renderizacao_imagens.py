"""Agrupamento, antevisao e download de imagens no Streamlit."""

from __future__ import annotations

from pathlib import Path

from src.interface.streamlit_proxy import st

def _grupo_imagem(img: dict) -> str:
    grupo = img.get("group")
    if grupo:
        return str(grupo)
    legenda = str(img.get("caption", "Resultados"))
    return legenda.split(" - ", 1)[0] if " - " in legenda else "Resultados"


# Exibição PROPORCIONAL: todos os gráficos são gerados a DPI fixo
# (src/ml/estilo_graficos.DPI), então largura_px/DPI = polegadas físicas.
# Cada polegada vira um nº fixo de pixels na tela — fontes e elementos
# aparecem do MESMO tamanho em todos os gráficos, independente do tipo.
_DPI_GERACAO = 150        # deve casar com src.ml.estilo_graficos.DPI
_PX_POR_POLEGADA = 72     # escala de exibição (12 pol → 864 px)
_TETO_EXIBICAO = 1080     # nunca estoura a largura útil do chat
_LARGURA_PAREAVEL = 560   # só exibe lado a lado o que cabe em meia coluna


def _dimensoes_imagem(path: str) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def _polegadas_imagem(img: dict) -> float | None:
    largura_px, _ = _dimensoes_imagem(img["path"])
    if largura_px:
        return largura_px / _DPI_GERACAO
    return None


def _largura_exibicao_imagem(img: dict) -> int:
    pol = _polegadas_imagem(img)
    if pol is None:
        return 860  # sem PIL/arquivo: largura neutra
    return min(_TETO_EXIBICAO, round(pol * _PX_POR_POLEGADA))


def _imagem_larga(img: dict) -> bool:
    """Painéis (>= 13 pol de largura física) sempre sozinhos, em linha cheia."""
    pol = _polegadas_imagem(img)
    if pol is not None:
        return pol >= 13
    # fallback (imagem ilegível): heurística antiga por legenda
    tipo = str(img.get("kind", "")).lower()
    legenda = str(img.get("caption", "")).lower()
    return (
        tipo in {"comparacao", "wide"}
        or "comparacao" in legenda
        or "anomalias detectadas" in legenda
        or "curvas" in legenda
        or "heatmap" in legenda
    )


def _ordem_imagem(img: dict, indice: int) -> tuple:
    try:
        valor_grupo = img.get("group_order", 0)
        ordem_grupo = int(0 if valor_grupo is None else valor_grupo)
    except Exception:
        ordem_grupo = 0
    try:
        valor_ordem = img.get("order", indice)
        ordem = int(indice if valor_ordem is None else valor_ordem)
    except Exception:
        ordem = indice
    return ordem_grupo, ordem, indice


# Contador monotônico p/ chaves únicas dos download_button (Streamlit exige
# key única por widget; monotônico garante unicidade dentro e entre reruns).
_DL_KEY = [0]


def _botao_download(img: dict, alvo=None, *, compacto: bool = False) -> None:
    """Botão de download da figura (PNG). Não renderiza a imagem, só o botão."""
    destino = alvo if alvo is not None else st
    p = Path(img["path"])
    if not p.is_file():
        return
    try:
        dados = p.read_bytes()
    except OSError:
        return
    _DL_KEY[0] += 1
    legenda = img.get("caption") or p.name
    destino.download_button(
        label="PNG" if compacto else "Baixar",
        data=dados,
        file_name=p.name,
        mime="image/png",
        key=f"dl_{_DL_KEY[0]}",
        icon=":material/download:",
        help=f"Salvar {legenda} ({p.name})",
        on_click="ignore",
        width="stretch" if compacto else "content",
    )


def _botao_download_texto(texto: str, nome: str, alvo=None) -> None:
    """Botão de download de um texto puro (ex.: transcrito da conversa em .txt)."""
    destino = alvo if alvo is not None else st
    _DL_KEY[0] += 1
    destino.download_button(
        label="Baixar .txt",
        data=(texto or "").encode("utf-8"),
        file_name=nome,
        mime="text/plain",
        key=f"dl_txt_{_DL_KEY[0]}",
        icon=":material/download:",
        help=f"Salvar {nome}",
        on_click="ignore",
        width="content",
    )


def _controles_antevisao(img: dict, alvo=None) -> None:
    """Antevisão sob demanda: a figura não ocupa o fluxo normal do chat."""
    destino = alvo if alvo is not None else st
    p = Path(img["path"])
    if not p.is_file():
        return

    legenda = img.get("caption") or p.name
    destino.caption(legenda)
    col_ver, col_baixar = destino.columns(2, gap="small")
    _DL_KEY[0] += 1
    with col_ver.popover(
        "Ver",
        icon=":material/visibility:",
        help="Abrir antevisão responsiva sem baixar o arquivo",
        width="stretch",
        key=f"preview_{_DL_KEY[0]}",
    ):
        st.image(
            str(p),
            caption=legenda,
            width="stretch",
        )
        largura, altura = _dimensoes_imagem(str(p))
        tamanho_kb = p.stat().st_size / 1024
        dimensoes = f"{largura} × {altura} px" if largura and altura else "dimensões indisponíveis"
        st.caption(f"{dimensoes} · {tamanho_kb:.0f} KB · PNG")
    _botao_download(img, col_baixar, compacto=True)


def _renderizar_imagem_unica(img: dict, coluna=None) -> None:
    alvo = coluna if coluna is not None else st
    # width="stretch": a figura se ajusta à largura da tela/coluna — nunca
    # estoura nem fica minúscula (substitui a largura fixa em px).
    alvo.image(
        img["path"],
        caption=img.get("caption", ""),
        width="stretch",
    )
    _botao_download(img, alvo)


def _renderizar_lote_regular(lote: list[dict]) -> None:
    """
    Exibe imagens não-panorâmicas. Pareia lado a lado APENAS quando as duas
    cabem em meia coluna (largura de exibição <= _LARGURA_PAREAVEL) — antes,
    gráficos de 12 pol eram espremidos em colunas de ~430 px e o tamanho
    final dependia da paridade do lote.
    """
    fila = list(lote)
    while fila:
        img = fila.pop(0)
        cabe_par = (
            _largura_exibicao_imagem(img) <= _LARGURA_PAREAVEL
            and fila
            and _largura_exibicao_imagem(fila[0]) <= _LARGURA_PAREAVEL
        )
        if cabe_par:
            par = [img, fila.pop(0)]
            cols = st.columns(2, gap="small")
            for col, item in zip(cols, par):
                _renderizar_imagem_unica(item, col)
        else:
            _renderizar_imagem_unica(img)


def _renderizar_grupo_imagens(imagens: list[dict]) -> None:
    pendentes_regulares: list[dict] = []
    for img in imagens:
        if _imagem_larga(img):
            _renderizar_lote_regular(pendentes_regulares)
            pendentes_regulares = []
            _renderizar_imagem_unica(img)
        else:
            pendentes_regulares.append(img)
    _renderizar_lote_regular(pendentes_regulares)


def renderizar_imagens(imagens: list[dict]) -> None:
    """
    Renderiza imagens, ignorando paths que não existem mais no disco.
    Cenário comum: o usuário apagou ou recalculou artefatos pelo chat e ainda
    há mensagens antigas com paths inválidos no histórico.
    """
    if not imagens:
        return

    validas = []
    invalidas = 0
    for idx, img in enumerate(imagens):
        caminho = img.get("path", "")
        if caminho and Path(caminho).is_file():
            img = dict(img)
            img["_idx"] = idx
            validas.append(img)
        else:
            invalidas += 1

    if not validas:
        if invalidas:
            st.caption(
                f"_({invalidas} imagem(ns) referenciada(s) já não existe(m) no disco — "
                "rode o pipeline novamente para regenerá-las.)_"
            )
        return

    validas.sort(key=lambda img: _ordem_imagem(img, int(img.get("_idx", 0))))

    # inline=True → renderiza na tela (com botão de download embaixo);
    # inline=False → só botão de download (não ocupa a tela com a figura).
    inline = [img for img in validas if img.get("inline", True)]
    download_only = [img for img in validas if not img.get("inline", True)]

    grupos: dict[str, list[dict]] = {}
    for img in inline:
        grupos.setdefault(_grupo_imagem(img), []).append(img)

    mostrar_titulos = len(grupos) > 1
    for grupo, itens in grupos.items():
        if mostrar_titulos:
            st.markdown(f"**{grupo}**")
        _renderizar_grupo_imagens(itens)

    if download_only:
        st.caption(
            "Gráficos disponíveis. Abra a antevisão para inspecionar antes de baixar."
        )
        for inicio in range(0, len(download_only), 2):
            par = download_only[inicio:inicio + 2]
            cols = st.columns(len(par), gap="small")
            for col, img in zip(cols, par):
                _controles_antevisao(img, col)

    if invalidas:
        st.caption(
            f"_({invalidas} imagem(ns) adicional(is) referenciada(s) não está(ão) "
            "mais no disco.)_"
        )
