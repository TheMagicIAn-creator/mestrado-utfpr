"""
verificar_datasets.py — Al IAdo PV / Sprint 4-5 (reprodutibilidade)

Valida os datasets locais (não versionados): presença, SHA-256, nº de linhas,
colunas e domínio (CA/CC). Gera/atualiza dados/dataset_manifest.json com os
metadados — sem versionar os dados brutos.

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

DATASETS = [
    {
        "nome": "PV Farms (treino)", "arquivo": "train_data.csv",
        "sep": ";", "rotulo": "class", "dominio": "CC",
        "uso": "classificação supervisionada de falhas CC",
    },
    {
        "nome": "PV Farms (teste)", "arquivo": "test_data.csv",
        "sep": ";", "rotulo": "class", "dominio": "CC",
        "uso": "avaliação supervisionada",
    },
    {
        "nome": "Paderborn", "arquivo": "Inverter_Data_Set.csv",
        "sep": ",", "rotulo": None, "dominio": "CA",
        "uso": "modelagem de normalidade (inversor saudável)",
    },
]


def _contar_linhas(caminho: Path) -> int:
    total = 0
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            total += bloco.count(b"\n")
    return total


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
            info.update({
                "linhas": n_linhas,
                "sha256": sha256_arquivo(caminho),
                "tamanho_bytes": caminho.stat().st_size,
            })
            # colunas + classes (apenas para os pequenos rotulados)
            if ds["rotulo"]:
                try:
                    import pandas as pd

                    df = pd.read_csv(caminho, sep=ds["sep"], nrows=5)
                    info["n_features"] = df.shape[1] - 1
                    df_full = pd.read_csv(caminho, sep=ds["sep"], usecols=[ds["rotulo"]])
                    info["classes"] = sorted(df_full[ds["rotulo"]].unique().tolist())
                except Exception as exc:  # noqa: BLE001
                    info["aviso"] = f"colunas não lidas: {exc}"
        registros[ds["nome"]] = info

        if not silencioso:
            marca = "✅" if existe else "❌"
            extra = (f"{info.get('linhas', '?')} linhas | sha={str(info.get('sha256'))[:12]}…"
                     if existe else "AUSENTE (baixe localmente)")
            print(f"  {marca} {ds['nome']:20s} [{ds['dominio']}] {extra}")

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
