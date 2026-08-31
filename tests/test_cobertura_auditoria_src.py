"""Matriz de responsabilidade dos módulos destacados na auditoria geral."""

from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]

ALVOS = {
    "src/conhecimento/agente_contexto.py": "tests/test_inteligencia_agente.py",
    "src/conhecimento/agente_interacao.py": "tests/test_inteligencia_agente.py",
    "src/conhecimento/agente_recuperacao.py": "tests/test_inteligencia_agente.py",
    "src/conhecimento/consultas_obsidian.py": "tests/test_inventario_vault.py",
    "src/conhecimento/ferramentas_academicas.py": "tests/test_roteamento_canonico.py",
    "src/conhecimento/indice_portatil.py": "tests/test_indice_portatil.py",
    "src/conhecimento/intencoes_ferramentas.py": "tests/test_roteamento_canonico.py",
    "src/conhecimento/leitor_anexos.py": "tests/test_leitor_anexos.py",
    "src/conhecimento/resultados_ml.py": "tests/test_resultados_ml.py",
    "src/conhecimento/roteamento_ferramentas.py": "tests/test_roteamento_canonico.py",
    "src/ml/avaliacao_comparativa.py": "tests/test_publicacao_comparacao.py",
    "src/ml/confiabilidade_componentes.py": "tests/test_confiabilidade_componentes.py",
    "src/ml/dados_gpvs.py": "tests/test_dados_gpvs_canonico.py",
    "src/ml/modelos_autoencoder.py": "tests/test_modelos_autoencoder_canonicos.py",
    "src/ml/pipeline.py": "tests/test_pipeline_canonico.py",
    "src/ml/proveniencia.py": "tests/test_proveniencia.py",
    "src/ml/publicacao_comparacao.py": "tests/test_publicacao_comparacao.py",
    "src/ml/publicacao_confiabilidade.py": "tests/test_confiabilidade_componentes.py",
    "src/ml/resultados.py": "tests/test_resultados_canonicos.py",
    "src/ml/sensibilidade_escore.py": "tests/test_sensibilidade_escore.py",
}


def test_modulos_criticos_tem_teste_direto_nomeado():
    ausentes = []
    for modulo, teste in ALVOS.items():
        caminho_modulo = RAIZ / modulo
        caminho_teste = RAIZ / teste
        if not caminho_modulo.exists() or not caminho_teste.exists():
            ausentes.append(f"{modulo} -> {teste}")

    assert not ausentes, f"módulos críticos sem teste direto: {ausentes}"
