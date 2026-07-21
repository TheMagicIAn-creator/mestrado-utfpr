"""Exportação do histórico de conversa (lógica pura, sem Streamlit)."""

from __future__ import annotations

from src.core.conversa_export import (
    montar_transcricao,
    nome_arquivo_conversa,
    quer_exportar_conversa,
)


def test_detecta_pedido_de_exportar_conversa():
    positivos = [
        "Gere um arquivo .txt do histórico dessa conversa.",
        "exporte a conversa",
        "baixar o histórico dessa sessão",
        "salvar essa conversa em txt",
        "me dá um documento com a transcrição do chat",
        "gere um arquivo do nosso bate-papo",
    ]
    for p in positivos:
        assert quer_exportar_conversa(p), p


def test_nao_dispara_sem_os_dois_sinais():
    negativos = [
        "qual o histórico do inversor?",          # só alvo, sem arquivo
        "gere um arquivo de resultados",          # só arquivo, sem conversa
        "rode o pipeline",                        # nada a ver
        "explique a conversa sobre FMEA",         # alvo, mas não pede arquivo
        "mostre os gráficos",
    ]
    for n in negativos:
        assert not quer_exportar_conversa(n), n


def test_transcricao_inclui_conteudo_completo_de_cada_mensagem():
    mensagens = [
        {"role": "user", "content": "e ai, tudo certo?"},
        {"role": "assistant", "content": "Tudo certo, Rodolfo. Vamos ao pipeline."},
        {"role": "user", "content": "Sim, faça."},
        {"role": "assistant", "content": "Processamento concluído. Segue a tabela..."},
    ]
    txt = montar_transcricao(mensagens, exportado_em="21/07/2026 às 14:30")

    assert "HISTÓRICO DA CONVERSA" in txt
    assert "21/07/2026 às 14:30" in txt
    assert "Total de trocas (perguntas do pesquisador): 2" in txt
    # conteúdo integral, com rótulos legíveis
    assert "Rodolfo:" in txt and "Al IAdo PV:" in txt
    for m in mensagens:
        assert m["content"] in txt


def test_transcricao_vazia_nao_quebra():
    txt = montar_transcricao([], exportado_em="x")
    assert "Ainda não há mensagens" in txt


def test_transcricao_registra_imagens_exibidas():
    mensagens = [{"role": "assistant", "content": "veja a curva",
                  "imagens": [{"caption": "ROC do autoencoder", "path": "/x/roc.png"}]}]
    txt = montar_transcricao(mensagens)
    assert "ROC do autoencoder" in txt


def test_nome_arquivo_seguro():
    assert nome_arquivo_conversa("2026-07-21_14-30") == "conversa_al_iado_2026-07-21_14-30.txt"
    # sanitiza caracteres estranhos
    assert "/" not in nome_arquivo_conversa("2026/07/21 14:30")
    assert nome_arquivo_conversa("").endswith(".txt")
