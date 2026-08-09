"""
verificar_datasets.py — Al IAdo PV / Sprint 4-5 (reprodutibilidade)

Valida os datasets locais (não versionados): presença, SHA-256, nº de linhas,
colunas, classes, duplicatas e domínio. Gera/atualiza
dados/dataset_manifest.json com os metadados, sem versionar os dados brutos.

Uso:
    python scripts/verificar_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import RAIZ_PROJETO  # noqa: E402
from src.ml.proveniencia import sha256_arquivo  # noqa: E402

BASE = Path(RAIZ_PROJETO) / "dados" / "brutos"
MANIFESTO = Path(RAIZ_PROJETO) / "dados" / "dataset_manifest.json"

# Mínimo de linhas para o arquivo ser UTILIZÁVEL, não só existir. A validação
# também exige as quatro classes esperadas e um mínimo por classe. Isso separa
# a prévia de 100 linhas do Kaggle do arquivo de treino completo de 600 linhas
# sem rejeitar o arquivo de teste completo, que tem 100 linhas (25 por classe).
#
# Para o conjunto Stender o piso vem dos parâmetros reais de features_ca
# (JANELA e PASSO): abaixo de dezenas de janelas o pipeline não é utilizável.
try:
    from src.ml.features_ca import JANELA as _JANELA, PASSO as _PASSO
except Exception:  # noqa: BLE001 - o diagnóstico não pode depender do pipeline
    _JANELA, _PASSO = 1024, 512

# Uma janela só passa no piso e quebra adiante: o holdout precisa de DEZENAS de
# janelas não sobrepostas para o split temporal, a calibração e o Weibull. Um
# piso de 50 janelas (~26 mil linhas) separa "prévia do Kaggle" de "dataset
# utilizável" sem exigir o tamanho exato do arquivo de referência.
_MIN_JANELAS_UTEIS = 50
_MIN_LINHAS_PADERBORN = (_MIN_JANELAS_UTEIS - 1) * _PASSO + _JANELA

DATASETS = [
    {
        "nome": "PV Farms simulado (treino)", "arquivo": "train_data.csv",
        "sep": ";", "rotulo": "class", "dominio": "CC",
        "uso": "benchmark supervisionado de falhas CC simuladas",
        "min_linhas": 600, "classes_esperadas": [0, 1, 2, 3],
        "min_por_classe": 100,
    },
    {
        "nome": "PV Farms simulado (teste)", "arquivo": "test_data.csv",
        "sep": ";", "rotulo": "class", "dominio": "CC",
        "uso": "avaliação supervisionada de falhas CC simuladas",
        "min_linhas": 100, "classes_esperadas": [0, 1, 2, 3],
        "min_por_classe": 25,
    },
    {
        "nome": "Stender inverter (Paderborn)",
        "arquivo": "Inverter_Data_Set.csv",
        "sep": ",", "rotulo": None, "dominio": "CA",
        "uso": "modelagem de normalidade em inversor de acionamento saudável",
        "min_linhas": _MIN_LINHAS_PADERBORN,
        "colunas_obrigatorias": [
            "i_a_k", "i_b_k", "i_c_k", "u_a_k-1", "u_b_k-1", "u_c_k-1",
        ],
    },
]

_ARQUIVOS_GPVS = [f"F{i}{modo}.csv" for i in range(8) for modo in "LM"]
_COLUNAS_GPVS = [
    "Time", "Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc",
    "Iabc", "If", "Vabc", "Vf",
]
_MIN_LINHAS_GPVS = 50_000
_SHA256_ZIP_GPVS = "88cd20c848fee86752870cf9b198eab45568c31355685328dd75aba982bf1a63"


def _contar_linhas(caminho: Path) -> int:
    total = 0
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            total += bloco.count(b"\n")
    return total


def _verificar_gpvs(base: Path) -> dict:
    """Valida os 16 ensaios GPVS sem carregar os CSVs completos na memoria."""
    raiz = base / "gpvs"
    candidatos = [raiz / "csv" / "CSV_Files", raiz / "csv", raiz]
    pasta = next((p for p in candidatos if (p / "F0L.csv").exists()), candidatos[0])
    ausentes = [nome for nome in _ARQUIVOS_GPVS if not (pasta / nome).exists()]
    info = {
        "arquivo": "gpvs/csv/CSV_Files/F0L.csv ... F7M.csv",
        "dominio": "PV conectado a rede",
        "uso": "validacao experimental externa E3 por ensaio",
        "presente": len(ausentes) < len(_ARQUIVOS_GPVS),
        "arquivos_esperados": len(_ARQUIVOS_GPVS),
        "arquivos_presentes": len(_ARQUIVOS_GPVS) - len(ausentes),
    }
    if not info["presente"]:
        return info

    avisos = []
    if ausentes:
        avisos.append(f"ensaios ausentes: {ausentes}")
    hashes = {}
    linhas = {}
    periodos_us = []
    try:
        import numpy as np
        import pandas as pd

        for nome in _ARQUIVOS_GPVS:
            caminho = pasta / nome
            if not caminho.exists():
                continue
            n_linhas = max(0, _contar_linhas(caminho) - 1)
            linhas[nome] = n_linhas
            hashes[nome] = sha256_arquivo(caminho)
            if n_linhas < _MIN_LINHAS_GPVS:
                avisos.append(
                    f"{nome} tem {n_linhas} linhas; minimo {_MIN_LINHAS_GPVS}"
                )
            amostra = pd.read_csv(caminho, nrows=2_000)
            faltantes = [c for c in _COLUNAS_GPVS if c not in amostra.columns]
            if faltantes:
                avisos.append(f"{nome}: colunas ausentes {faltantes}")
                continue
            valores = amostra[_COLUNAS_GPVS].to_numpy(dtype=float)
            if not np.isfinite(valores).all():
                avisos.append(f"{nome}: NaN ou infinito na amostra inicial")
            dt = np.diff(amostra["Time"].to_numpy(dtype=float))
            if np.any(dt <= 0):
                avisos.append(f"{nome}: Time nao estritamente crescente")
            else:
                periodo_us = float(np.median(dt) * 1e6)
                periodos_us.append(periodo_us)
                if not 90.0 <= periodo_us <= 110.0:
                    avisos.append(
                        f"{nome}: periodo observado {periodo_us:.4f} us fora de ~100 us"
                    )
    except Exception as exc:  # noqa: BLE001
        avisos.append(f"falha ao validar GPVS: {exc}")

    zip_path = raiz / "CSV_Files.zip"
    zip_hash = sha256_arquivo(zip_path) if zip_path.exists() else None
    if zip_hash and zip_hash != _SHA256_ZIP_GPVS:
        avisos.append("CSV_Files.zip nao coincide com o SHA-256 oficial")

    info.update({
        "linhas_por_ensaio": linhas,
        "linhas_total": int(sum(linhas.values())),
        "sha256_por_ensaio": hashes,
        "zip_sha256": zip_hash,
        "zip_sha256_oficial": _SHA256_ZIP_GPVS,
        "sampling_period_us_min": min(periodos_us) if periodos_us else None,
        "sampling_period_us_max": max(periodos_us) if periodos_us else None,
        "readme_sampling_period_us": 9.9989,
        "sampling_note": (
            "CSV observado ~100 us (~10 kHz), dez vezes o valor declarado no ReadMe"
        ),
        "utilizavel": not avisos,
    })
    if avisos:
        info["avisos"] = avisos
        info["aviso"] = "; ".join(avisos)
    return info


def verificar(silencioso: bool = False) -> dict:
    registros = {}
    for ds in DATASETS:
        caminho = BASE / ds["arquivo"]
        existe = caminho.exists()
        info = {
            "arquivo": ds["arquivo"],
            "dominio": ds["dominio"],
            "uso": ds["uso"],
            "presente": existe,
        }
        if existe:
            n_linhas = max(0, _contar_linhas(caminho) - 1)  # menos cabeçalho
            minimo = int(ds.get("min_linhas", 0))
            truncado = bool(minimo and n_linhas < minimo)
            avisos = []
            info.update({
                "linhas": n_linhas,
                "sha256": sha256_arquivo(caminho),
                "tamanho_bytes": caminho.stat().st_size,
                "min_linhas": minimo,
                "truncado": truncado,
            })
            if truncado:
                avisos.append(
                    f"apenas {n_linhas} linhas, mínimo {minimo} — provável "
                    "prévia do Kaggle; baixe pelo botão Download do dataset"
                )
            try:
                import pandas as pd

                if ds["rotulo"]:
                    df = pd.read_csv(caminho, sep=ds["sep"])
                    rotulo = ds["rotulo"]
                    if rotulo not in df.columns:
                        avisos.append(f"coluna de rótulo ausente: {rotulo}")
                    else:
                        dist = df[rotulo].value_counts().sort_index()
                        classes = sorted(dist.index.tolist())
                        esperadas = list(ds.get("classes_esperadas", []))
                        minimo_classe = int(ds.get("min_por_classe", 0))
                        info.update({
                            "n_features": int(df.shape[1] - 1),
                            "classes": classes,
                            "distribuicao_classes": {
                                str(k): int(v) for k, v in dist.items()
                            },
                            "linhas_duplicadas": int(df.duplicated().sum()),
                        })
                        if esperadas and classes != esperadas:
                            avisos.append(
                                f"classes {classes}; esperado {esperadas}"
                            )
                        abaixo = {
                            str(k): int(v) for k, v in dist.items()
                            if minimo_classe and int(v) < minimo_classe
                        }
                        if abaixo:
                            avisos.append(
                                f"classes abaixo de {minimo_classe} amostras: {abaixo}"
                            )
                        duplicadas = int(info["linhas_duplicadas"])
                        if duplicadas:
                            avisos.append(
                                f"{duplicadas} linhas exatamente duplicadas; use grupos "
                                "na validação cruzada"
                            )
                else:
                    cabecalho = pd.read_csv(caminho, sep=ds["sep"], nrows=0)
                    obrigatorias = list(ds.get("colunas_obrigatorias", []))
                    ausentes = [c for c in obrigatorias if c not in cabecalho.columns]
                    info["n_features"] = int(len(cabecalho.columns))
                    if ausentes:
                        avisos.append(f"colunas obrigatórias ausentes: {ausentes}")
            except Exception as exc:  # noqa: BLE001
                avisos.append(f"colunas não lidas: {exc}")

            erros_estruturais = [
                a for a in avisos
                if "linhas exatamente duplicadas" not in a
            ]
            info["utilizavel"] = not truncado and not erros_estruturais
            if avisos:
                info["avisos"] = avisos
                info["aviso"] = "; ".join(avisos)
        registros[ds["nome"]] = info

        if not silencioso:
            if not existe:
                marca, extra = "❌", "AUSENTE (baixe localmente)"
            elif not info.get("utilizavel", True):
                marca = "❌"
                if info.get("truncado"):
                    extra = (
                        f"{info['linhas']} linhas — TRUNCADO "
                        f"(mínimo {info['min_linhas']})"
                    )
                else:
                    extra = f"{info['linhas']} linhas — NÃO UTILIZÁVEL"
            else:
                marca = "✅"
                extra = (f"{info['linhas']} linhas | "
                         f"sha={str(info.get('sha256'))[:12]}…")
            print(f"  {marca} {ds['nome']:20s} [{ds['dominio']}] {extra}")
            for aviso in info.get("avisos", []):
                print(f"     → {aviso}")

    gpvs = _verificar_gpvs(BASE)
    registros["GPVS-Faults experimental"] = gpvs
    if not silencioso:
        if not gpvs["presente"]:
            print("  ❌ GPVS-Faults experimental [PV/rede] AUSENTE (baixe localmente)")
        elif gpvs.get("utilizavel"):
            print(
                "  ✅ GPVS-Faults experimental [PV/rede] "
                f"{gpvs['arquivos_presentes']}/16 ensaios | "
                f"{gpvs['linhas_total']} linhas"
            )
        else:
            print("  ❌ GPVS-Faults experimental [PV/rede] NAO UTILIZAVEL")
            for aviso in gpvs.get("avisos", []):
                print(f"     → {aviso}")

    if any(r["presente"] for r in registros.values()):
        MANIFESTO.write_text(
            json.dumps({"datasets": registros}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not silencioso:
            print(f"\n  Manifesto atualizado: {MANIFESTO.relative_to(RAIZ_PROJETO)}")
    return registros


if __name__ == "__main__":
    from src.core.utils import configurar_saida_utf8

    configurar_saida_utf8()
    print("AL IADO PV — verificação de datasets\n")
    verificar(silencioso=False)
