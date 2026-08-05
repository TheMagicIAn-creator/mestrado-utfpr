"""Matriz de responsabilidade dos módulos destacados na auditoria geral."""

from pathlib import Path


RAIZ = Path(__file__).resolve().parents[1]

ALVOS = {
    "src/interface/streamlit_app.py": "tests/test_interface_minimalista.py",
    "src/conhecimento/indice_portatil.py": "tests/test_indice_portatil.py",
    "src/conhecimento/leitor_anexos.py": "tests/test_leitor_anexos.py",
    "src/conhecimento/resultados_ml.py": "tests/test_resultados_ml.py",
    "src/ml/classificador_pv.py": "tests/test_classificador_pv.py",
    "src/ml/eda.py": "tests/test_eda.py",
    "src/ml/exec_etapa_isolada.py": "tests/test_exec_etapa_isolada.py",
    "src/ml/features_ca.py": "tests/test_metodologia_estatistica.py",
    "src/ml/rul_weibull.py": "tests/test_metodologia_estatistica.py",
    "src/ml/retroalimentacao_fmeca.py": "tests/test_retroalimentacao_fmeca.py",
    "src/ml/diagnostico_escore.py": "tests/test_limiar.py",
    "src/ml/graficos_autoencoder.py": "tests/test_graficos_autoencoder.py",
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
