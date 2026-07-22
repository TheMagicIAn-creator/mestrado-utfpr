"""
Regressoes de intencao do chat.

Esses casos protegem contra o erro mais perigoso do agente: transformar uma
pergunta discursiva/conceitual em execucao pesada de pipeline ou experimento.
"""

from src.conhecimento.agente import deve_consultar_literatura
from src.conhecimento.ferramentas import (
    _decisao_rapida,
    decidir_acao,
    limpar_experimentos_artigos,
)


def _ferramenta(pergunta: str):
    decisao = _decisao_rapida(pergunta) or {}
    return decisao.get("ferramenta")


def test_revisao_bibliografica_de_rul_nao_roda_weibull():
    decisao = _decisao_rapida(
        "Faça uma revisão bibliográfica curta sobre RUL em eletrônica de potência."
    )
    assert decisao == {"usar_ferramenta": False, "ferramenta": None}


def test_citar_artigos_sobre_anomalias_vai_para_rag_nao_experimentos():
    pergunta = "Cite artigos sobre detecção de anomalias em inversores fotovoltaicos."
    esperado = {"usar_ferramenta": False, "ferramenta": None}

    assert _decisao_rapida(pergunta) == esperado
    assert deve_consultar_literatura(pergunta)

    class RoteadorRuim:
        def invoke(self, *_args, **_kwargs):  # pragma: no cover - nao deve chamar
            raise AssertionError("o LLM roteador nao deve ser chamado")

    assert decidir_acao(pergunta, RoteadorRuim()) == esperado


def test_explicacao_de_fmea_com_base_no_projeto_nao_aciona_pipeline():
    decisao = _decisao_rapida("Não use literatura. Explique FMEA com base no projeto.")
    assert decisao == {"usar_ferramenta": False, "ferramenta": None}
    assert not deve_consultar_literatura(
        "Não use literatura. Explique FMEA com base no projeto."
    )


def test_stender_sobre_paderborn_vai_para_rag_nao_catalogo_dataset():
    decisao = _decisao_rapida("O que o Stender diz sobre o dataset de Paderborn?")
    assert decisao == {"usar_ferramenta": False, "ferramenta": None}
    assert deve_consultar_literatura(
        "O que o Stender diz sobre o dataset de Paderborn?"
    )


def test_citar_fonte_sem_inventar_nao_desliga_literatura():
    pergunta = (
        "O que Stender diz sobre Paderborn? Cite a fonte e a página exata, "
        "sem inventar."
    )
    assert deve_consultar_literatura(pergunta)


def test_compare_autores_com_artefatos_consulta_resultados_sem_treinar():
    pergunta = (
        "Compare Sharma, Ibrahim e Ahirwar usando somente os artefatos "
        "recalculados do repositório. Mostre gráficos e matrizes."
    )
    assert _ferramenta(pergunta) == "consultar_resultados"


def test_resultados_locais_vs_replicacoes_consulta_proveniencia():
    pergunta = (
        "Quais resultados são dos datasets locais e quais são apenas "
        "replicações dos artigos?"
    )
    assert _ferramenta(pergunta) == "consultar_resultados"


def test_apagar_experimentos_tem_rota_propria_e_confirmacao():
    assert _ferramenta("Quero que apague os experimentos.") == "limpar_experimentos_artigos"
    res = limpar_experimentos_artigos(pergunta="Quero que apague os experimentos.")
    assert res["ok"]
    assert "CONFIRMAR LIMPEZA EXPERIMENTOS" in res["mensagem"]
    assert "dados brutos" in res["mensagem"].lower()


# ── comparar_experimentos_auc ─────────────────────────────────────────────────

def test_compare_experimentos_de_anomalia_vai_para_auc():
    """Sessão 14/06 bug: 'compare os experimentos de anomalia' devolvia narrativa LLM."""
    assert _ferramenta("compare os experimentos de anomalia") == "comparar_experimentos_auc"


def test_compare_por_auc_vai_para_auc():
    assert _ferramenta("compare os experimentos por AUC") == "comparar_experimentos_auc"


def test_comparar_modelos_de_anomalia_vai_para_auc():
    assert _ferramenta("comparar os modelos de anomalia") == "comparar_experimentos_auc"


def test_qual_melhor_modelo_anomalia_vai_para_auc():
    assert _ferramenta("qual o melhor modelo de anomalia") == "comparar_experimentos_auc"


def test_rode_experimento_nao_confunde_com_auc():
    """Rodar um experimento NÃO deve ir para comparar_experimentos_auc."""
    assert _ferramenta("rode o experimento do francisti") == "rodar_experimento_artigo"


def test_compare_autores_sem_experimento_vai_para_consultar_resultados():
    """Teste de regressão: autores nomeados + artefatos = consultar_resultados."""
    pergunta = (
        "Compare Sharma, Ibrahim e Ahirwar usando somente os artefatos "
        "recalculados do repositório. Mostre gráficos e matrizes."
    )
    assert _ferramenta(pergunta) == "consultar_resultados"


def test_compare_abordagens_ml_nao_vai_para_auc():
    """'compare as abordagens de ML' deve ir para comparar_abordagens_ml."""
    assert _ferramenta("compare as abordagens de ML") == "comparar_abordagens_ml"


def test_forcar_resposta_direta_bypass_llm():
    """Ferramenta com forcar_resposta_direta deve retornar mensagem sem LLM."""
    from src.conhecimento.ferramentas import comentar_resultado

    class LLMQueNuncaDeveSerChamado:
        def invoke(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("o LLM nao deve ser chamado com forcar_resposta_direta")

    resultado = {
        "ok": True,
        "mensagem": "tabela AUC aqui",
        "resposta_pronta": True,
        "forcar_resposta_direta": True,
    }
    resp = comentar_resultado(
        "compare os experimentos de anomalia",
        resultado,
        "perfil",
        LLMQueNuncaDeveSerChamado(),
    )
    assert resp == "tabela AUC aqui"


def test_pedido_de_codigo_de_grafico_vai_para_llm_nao_devolve_artefato():
    """Bug do usuario: "gere um codigo de um grafico da TTF" trazia as figuras
    ja criadas, em vez de o LLM ESCREVER o codigo. Deve ir para o LLM."""
    esperado = {"usar_ferramenta": False, "ferramenta": None}
    for pergunta in (
        "gere um código para plotar a distribuição do erro do autoencoder",
        "me dê o script em Python do gráfico da TTF",
        "como plotar a curva de Weibull no matplotlib?",
        "escreva a função que desenha o histograma do erro de reconstrução",
        "quero o código do gráfico da TTF",
    ):
        assert _decisao_rapida(pergunta) == esperado, pergunta


def test_pedido_de_execucao_ou_consulta_nao_e_confundido_com_codigo():
    """A correcao de codigo NAO pode roubar pedidos legitimos de rodar/mostrar."""
    from src.conhecimento.ferramentas import _quer_codigo_snippet

    assert _ferramenta("rode o pipeline completo") == "rodar_pipeline_completo"
    assert _ferramenta("mostre os gráficos da validação") == "consultar_resultados"
    # "autoencoder"/"decodifique" contem a substring "code" — nao pode disparar.
    assert not _quer_codigo_snippet("qual o AUC do autoencoder?")
    assert not _quer_codigo_snippet("decodifique o sinal")
    assert not _quer_codigo_snippet("escreva um resumo da distribuição do erro")


def test_gerar_grafico_sem_palavra_codigo_nao_despeja_resultados():
    """'gera um gráfico da ttf' nao pode cair em consultar_resultados so por
    conter 'gráfico' — verbo de geracao bloqueia o despejo (vai ao LLM)."""
    for pergunta in (
        "gera um gráfico da ttf",
        "plota a curva de weibull pra mim",
        "desenha a distribuição do erro",
    ):
        assert _ferramenta(pergunta) != "consultar_resultados", pergunta


def test_consulta_legitima_de_resultados_preservada():
    assert _ferramenta("mostre a matriz de confusão") == "consultar_resultados"
    assert _ferramenta("mostre os resultados do weibull") == "consultar_resultados"
    assert _ferramenta("cadê as imagens da roc?") == "consultar_resultados"
    # 'geral' NAO pode ser confundido com verbo de geracao
    assert _ferramenta("de modo geral, mostre a matriz") == "consultar_resultados"


def test_declaracao_de_memoria_nao_vira_comando_de_pipeline():
    """'Lembre-se: decidimos... injetada no pipeline...' e uma DECLARACAO para
    memorizar, nao um comando para rodar o pipeline."""
    d = _decisao_rapida(
        "Lembre-se: decidimos que a primeira falha a ser injetada no pipeline "
        "e o Contator AC, por causa do NPR mais alto."
    )
    assert d == {"usar_ferramenta": False, "ferramenta": None}


def test_pergunta_de_recall_nao_executa_pipeline():
    """'Qual falha decidimos injetar primeiro?' e recall, nao 'injete a falha'."""
    assert _ferramenta(
        "Qual falha a gente decidiu injetar primeiro no pipeline, o Contator AC?"
    ) != "rodar_pipeline_completo"


def test_tabelas_conceituais_nao_despejam_resultados():
    """'quais as tabelas de S/O/D da FMECA' e conceitual, nao o artefato."""
    assert _ferramenta("quais as tabelas para cada uma das variaveis (S O e D)?") \
        != "consultar_resultados"


def test_comandos_e_consultas_legitimos_preservados():
    assert _ferramenta("rode o pipeline completo") == "rodar_pipeline_completo"
    assert _ferramenta("gere os resultados") == "rodar_pipeline_completo"
    assert _ferramenta("mostre os resultados do weibull") == "consultar_resultados"


def test_pergunta_conceitual_sem_artefato_vai_ao_llm_nao_despeja_grafico():
    """'por onde começar o TCC?' / 'os NPR foram demais?' sao conceituais — vao
    ao LLM definitivamente, sem o fallback-LLM despejar consultar_resultados."""
    for q in (
        "por onde acha que devo começar a tratar as coisas do meu tcc?",
        "achas que os valores que adotei no npr foram demais ou fora da realidade?",
        "Qual falha a gente decidiu injetar primeiro no pipeline, o Contator AC?",
    ):
        assert _decisao_rapida(q) == {"usar_ferramenta": False, "ferramenta": None}, q


def test_pergunta_com_artefato_segue_fluxo_normal():
    """Pergunta que cita artefato ('cadê as imagens da roc?') ainda consulta."""
    assert _ferramenta("cadê as imagens da roc?") == "consultar_resultados"
