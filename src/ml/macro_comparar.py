"""
macro_comparar.py — Al IAdo PV / comparativo dos dois macro-códigos

Roda (ou reaproveita) os dois métodos sobre o MESMO holdout e a MESMA injeção
FMECA, e emite UMA tabela enxuta + UM gráfico sobreposto:

  - Proposto  (src/ml/macro_proposto.py) — AE denso + MSE p99
  - Ibrahim   (src/ml/macro_ibrahim.py)  — AE-LSTM temporal

Substitui o antigo framework de experimentos por artigo (que produzia tabelas
de 33 colunas e matrizes de confusão enganosas sob limiar de prevalência rara).

Uso:
  python src/ml/macro_comparar.py

Saídas: resultados/macro/comparacao_*.{json,md,csv,png}
Autor: Rodolfo Torres (UTFPR)
"""

from __future__ import annotations

try:
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger
except ModuleNotFoundError:  # execução direta
    import sys as _sys
    from pathlib import Path as _Path
    _raiz = str(_Path(__file__).resolve().parents[2])
    if _raiz not in _sys.path:
        _sys.path.insert(0, _raiz)
    from src.core.logs import adaptar_logger_como_print as _adaptar_log
    from src.core.logs import get_logger as _get_logger

_logger = _get_logger("macro_comparar")
_log = _adaptar_log(_logger)


from pathlib import Path

RAIZ = Path(__file__).parent.parent.parent
PASTA_SAIDA = RAIZ / "resultados" / "macro"


def _saidas_macro() -> list[Path]:
    sufixos = (
        "tabela.md",
        "tabela.csv",
        "resultado.json",
        "deteccao_severidade.png",
    )
    return [
        PASTA_SAIDA / f"{prefixo}_{sufixo}"
        for prefixo in ("proposto", "ibrahim", "comparacao")
        for sufixo in sufixos
    ]


def manifesto_atual(n_janelas: int | None = None) -> dict:
    """Descreve integralmente o comparativo sem depender do pipeline principal."""
    from src.ml.injecao_falhas import ARQUIVO_CSV, N_JANELAS_SMD, SEVERIDADES
    from src.ml.macro_comum import FRACAO_AJUSTE_LIMIAR, PURGA
    from src.ml.macro_ibrahim import EPOCHS, SEQ_LEN
    from src.ml.proveniencia import gerar_manifesto

    return gerar_manifesto(
        "macro_comparacao",
        Path(__file__),
        {
            "n_janelas": int(n_janelas or N_JANELAS_SMD),
            "severidades": list(SEVERIDADES),
            "purga": PURGA,
            "fracao_ajuste_limiar": FRACAO_AJUSTE_LIMIAR,
            "aelstm_seq_len": SEQ_LEN,
            "aelstm_epochs": EPOCHS,
        },
        {
            "dataset_stender": ARQUIVO_CSV,
            "features": RAIZ / "dados/processados/features_paderborn.parquet",
            "modelo_autoencoder": RAIZ / "resultados/autoencoder/modelo_autoencoder.pt",
            "limiar_autoencoder": RAIZ / "resultados/autoencoder/limiar.json",
            "scaler_autoencoder": RAIZ / "resultados/autoencoder/scaler.pkl",
            "hash_scaler_autoencoder": RAIZ / "resultados/autoencoder/scaler.pkl.sha256",
        },
        _saidas_macro(),
        code_dependencies={
            nome: RAIZ / caminho
            for nome, caminho in {
                "macro_proposto": "src/ml/macro_proposto.py",
                "macro_ibrahim": "src/ml/macro_ibrahim.py",
                "macro_comum": "src/ml/macro_comum.py",
                "modelos_anomalia": "src/ml/modelos_anomalia.py",
                "dados_avaliacao": "src/ml/dados_avaliacao.py",
                "injecao_falhas": "src/ml/injecao_falhas.py",
                "escore_anomalia": "src/ml/escore_anomalia.py",
                "features_ca": "src/ml/features_ca.py",
            }.items()
        },
        evidence_level="E2",
    )


def registrar_manifesto(n_janelas: int | None = None) -> Path:
    from src.ml.proveniencia import salvar_manifesto

    return salvar_manifesto(manifesto_atual(n_janelas))


def estado_proveniencia(n_janelas: int | None = None) -> list[str]:
    from src.ml.proveniencia import carregar_manifesto, comparar

    return comparar(
        carregar_manifesto("macro_comparacao"),
        manifesto_atual(n_janelas),
        permitir_inputs_ausentes=True,
    )


def entradas_proveniencia_indisponiveis(n_janelas: int | None = None) -> list[str]:
    """Entradas locais que o ambiente atual não consegue revalidar por hash."""
    atual = manifesto_atual(n_janelas)
    return [
        nome
        for nome, arquivo_hash in atual.get("input_artifacts", {}).items()
        if arquivo_hash is None
    ]


def executar(n_janelas: int | None = None) -> list[dict]:
    from src.ml import macro_ibrahim, macro_proposto
    from src.ml.macro_comum import salvar_saidas, tabela_enxuta

    _log("=" * 60)
    _log("  COMPARATIVO — método proposto × Ibrahim (2022)")
    _log("=" * 60)

    _log("\n[1/2] Método proposto...")
    r_prop = macro_proposto.executar(n_janelas)

    _log("\n[2/2] Método do Ibrahim...")
    r_ibra = macro_ibrahim.executar(n_janelas)

    resultados = [r_prop, r_ibra]
    saidas = salvar_saidas(resultados, PASTA_SAIDA, prefixo="comparacao")

    _log("\n" + "=" * 60)
    _log("  TABELA COMPARATIVA")
    _log("=" * 60)
    _log("\n" + tabela_enxuta(resultados))
    _log(f"\n  Artefatos em {PASTA_SAIDA}:")
    for k, v in saidas.items():
        _log(f"    {k}: {Path(v).name}")
    caminho_manifesto = registrar_manifesto(n_janelas)
    _log(f"    manifesto: {caminho_manifesto.name}")
    _log("\n  Leitura: compare por AUC (independe do limiar). A detecção por")
    _log("  severidade mostra a partir de que intensidade cada método enxerga")
    _log("  cada falha da FMECA. Ambos calibram o limiar em dados saudáveis.")
    _log("=" * 60)
    return resultados


if __name__ == "__main__":
    from src.core.logs import habilitar_console

    habilitar_console()
    executar()
