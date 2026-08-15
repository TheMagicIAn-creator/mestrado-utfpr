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
except ModuleNotFoundError as _erro:  # execução direta
    # Só trata a ausência do PACOTE `src` (rodar o arquivo direto, sem a raiz no
    # sys.path). Qualquer outra dependência faltando é repassada: reimportar não
    # a faria aparecer, e o retry produzia um traceback DUPLO com a causa real
    # ("No module named 'dotenv'") enterrada no meio — foi o que aconteceu com o
    # venv desativado em 15/08/2026.
    if (_erro.name or "").split(".")[0] != "src":
        raise
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
    from src.ml.gpvs_principal import ARQUIVO_FEATURES, PASTA_GPVS
    from src.ml.injecao_falhas import N_JANELAS_SMD, SEVERIDADES
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
            # Fonte unica desde 15/08/2026. Era o Stender; a comparacao ficou
            # para tras na migracao do pipeline e passou a pontuar vetores de
            # zeros em silencio. Ver a nota em macro_proposto.construir_scorer.
            "dataset_gpvs_f0l": PASTA_GPVS / "F0L.csv",
            "dataset_gpvs_f0m": PASTA_GPVS / "F0M.csv",
            "features": ARQUIVO_FEATURES,
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
                "gpvs": "src/ml/gpvs.py",
                "gpvs_principal": "src/ml/gpvs_principal.py",
                "injecao_falhas": "src/ml/injecao_falhas.py",
                "escore_anomalia": "src/ml/escore_anomalia.py",
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


def executar(n_janelas: int | None = None, com_weibull: bool = False,
             n_steps: int | None = None) -> list[dict]:
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

    # As curvas de confiabilidade por modelo NÃO saem daqui por padrão. A
    # varredura de magnitude custa até N_STEPS inferências por trajetória, por
    # falha, por modelo — ordens de grandeza acima do resto deste script. Quem
    # quer as curvas pede por elas.
    if com_weibull:
        from src.ml import macro_weibull

        _log("\n[3/3] Detectabilidade por modelo (varredura de magnitude)...")
        macro_weibull.executar(n_janelas, n_steps)
    else:
        _log("\n  As curvas por modelo (papel de Weibull, S_D, f_D/F_D, h_D)")
        _log("  não entram aqui: a varredura de magnitude é cara. Rode")
        _log("  `python -m src.ml.macro_weibull` — ou repita este comando com")
        _log("  `--weibull`.")
    return resultados


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Comparativo proposto × Ibrahim")
    p.add_argument("--n-janelas", type=int, default=None,
                   help="teto de janelas do holdout (padrão: todas)")
    p.add_argument("--weibull", action="store_true",
                   help="também gera as curvas de detectabilidade por modelo")
    p.add_argument("--n-steps", type=int, default=None,
                   help="passos da grade de magnitude quando --weibull")
    args = p.parse_args(argv)

    from src.core.logs import habilitar_console

    habilitar_console()
    executar(args.n_janelas, com_weibull=args.weibull, n_steps=args.n_steps)


if __name__ == "__main__":
    main()
