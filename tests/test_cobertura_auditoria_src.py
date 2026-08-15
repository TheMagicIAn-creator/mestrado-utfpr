"""Matriz de responsabilidade dos módulos destacados na auditoria geral."""

from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]

ALVOS = {
    "src/core/importacao.py": "tests/test_imports_modulos_extraidos.py",
    "src/conhecimento/agente_contexto.py": "tests/test_inteligencia_agente.py",
    "src/conhecimento/agente_interacao.py": "tests/test_inteligencia_agente.py",
    "src/conhecimento/agente_recuperacao.py": "tests/test_inteligencia_agente.py",
    "src/conhecimento/consultas_obsidian.py": "tests/test_inventario_vault.py",
    "src/conhecimento/ferramentas_academicas.py": "tests/test_roteamento_intencao.py",
    "src/conhecimento/indice_portatil.py": "tests/test_indice_portatil.py",
    "src/conhecimento/intencoes_ferramentas.py": "tests/test_roteamento_intencao.py",
    "src/conhecimento/leitor_anexos.py": "tests/test_leitor_anexos.py",
    "src/conhecimento/resultados_ml.py": "tests/test_resultados_ml.py",
    "src/conhecimento/roteamento_ferramentas.py": "tests/test_roteamento_intencao.py",
    "src/ml/classificador_pv.py": "tests/test_classificador_pv.py",
    "src/ml/eda.py": "tests/test_eda.py",
    "src/ml/exec_etapa_isolada.py": "tests/test_exec_etapa_isolada.py",
    "src/ml/features_ca.py": "tests/test_metodologia_estatistica.py",
    "src/ml/rul_weibull.py": "tests/test_metodologia_estatistica.py",
    "src/ml/retroalimentacao_fmeca.py": "tests/test_retroalimentacao_fmeca.py",
    "src/ml/diagnostico_escore.py": "tests/test_limiar.py",
    "src/ml/graficos_autoencoder.py": "tests/test_graficos_autoencoder.py",
    "src/ml/graficos_experimentos.py": "tests/test_protocolos_artigos.py",
    "src/ml/graficos_rul.py": "tests/test_metodologia_estatistica.py",
    "src/ml/dados_avaliacao.py": "tests/test_metodologia_estatistica.py",
    "src/ml/estatistica.py": "tests/test_metodologia_estatistica.py",
    "src/ml/macro_proposto.py": "tests/test_macros_diretas.py",
    "src/ml/macro_ibrahim.py": "tests/test_macros_diretas.py",
    "src/ml/macro_comparar.py": "tests/test_macros_diretas.py",
}


def test_modulos_criticos_tem_teste_direto_nomeado():
    ausentes = []
    for modulo, teste in ALVOS.items():
        caminho_modulo = RAIZ / modulo
        caminho_teste = RAIZ / teste
        if not caminho_modulo.exists() or not caminho_teste.exists():
            ausentes.append(f"{modulo} -> {teste}")

    assert not ausentes, f"módulos críticos sem teste direto: {ausentes}"
