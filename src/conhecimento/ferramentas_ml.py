"""
ferramentas_ml.py — compatibilidade retroativa.
Todo o conteúdo foi consolidado em ferramentas.py.
"""
from src.conhecimento.ferramentas import (  # noqa: F401
    ESPEC_FERRAMENTAS,
    executar_ferramenta,
    rodar_features_ca,
    rodar_autoencoder,
    rodar_injecao_falhas,
    rodar_validacao,
    rodar_weibull,
    rodar_pipeline_completo,
    consultar_resultados,
    consultar_status_pipeline,
    limpar_resultados_ml,
    buscar_na_web,
)
