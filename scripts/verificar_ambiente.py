"""
verificar_ambiente.py — Al IAdo PV / Sprint 4-5 (reprodutibilidade)

Diagnóstico do ambiente local: imports, versões, chaves de API, datasets
(presença + linhas + SHA-256), coleções do ChromaDB, estado das etapas do
pipeline (ready/stale/pending) e bibliotecas opcionais (degradação honesta).

NÃO imprime valores de chaves nem dados sensíveis. Sai com código 0 sempre
(é diagnóstico, não gate); o relatório indica o que está pendente.

Uso:
    python scripts/verificar_ambiente.py
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.utils import configurar_saida_utf8  # noqa: E402

configurar_saida_utf8()

OK, FALTA, AVISO = "✅", "❌", "⚠️"


def _versao(dist: str) -> str | None:
    try:
        return md.version(dist)
    except Exception:
        return None


def secao(titulo: str) -> None:
    print(f"\n{'='*60}\n  {titulo}\n{'='*60}")


# O projeto exige 3.12+ na PRÁTICA, não por convenção: src/conhecimento/
# agente.py usa f-string com barra invertida, sintaxe aceita só a partir do
# 3.12. Em 3.11 o módulo nem compila — o agente inteiro cai no import, e o
# erro que aparece ("f-string expression part cannot include a backslash")
# não sugere "versão errada de Python" para quem está começando.
PYTHON_MINIMO = (3, 12)
PYTHON_ALVO = (3, 13)


def checar_python() -> None:
    secao("Interpretador Python")
    v = sys.version_info
    atual = f"{v.major}.{v.minor}.{v.micro}"
    if v[:2] < PYTHON_MINIMO:
        marca, nota = FALTA, (
            f"  → o agente NÃO carrega abaixo de "
            f"{PYTHON_MINIMO[0]}.{PYTHON_MINIMO[1]} (sintaxe de f-string). "
            f"Instale o {PYTHON_ALVO[0]}.{PYTHON_ALVO[1]} e recrie o .venv."
        )
    elif v[:2] < PYTHON_ALVO:
        marca, nota = AVISO, (
            f"  → funciona, mas o alvo do projeto e do CI é "
            f"{PYTHON_ALVO[0]}.{PYTHON_ALVO[1]}."
        )
    else:
        marca, nota = OK, ""
    print(f"  {marca} versão                  {atual}")
    if nota:
        print(nota)

    # Rodar com o Python do sistema em vez do .venv é a origem silenciosa de
    # "instalei o pacote e mesmo assim não acha".
    dentro_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  {OK if dentro_venv else AVISO} ambiente virtual        "
          f"{'ativo' if dentro_venv else 'NÃO ativo — está usando o Python do sistema'}")
    if not dentro_venv:
        print("  → ative com: .\\.venv\\Scripts\\Activate.ps1 (Windows) "
              "ou source .venv/bin/activate")
    print(f"  ℹ️  executável              {sys.executable}")


def checar_git() -> None:
    """Estado do Git que costuma travar `pull`/`checkout` sem explicar por quê."""
    import subprocess

    secao("Git (o que trava atualização)")
    raiz = Path(__file__).resolve().parents[1]

    def _git(*args) -> str | None:
        try:
            r = subprocess.run(("git", *args), cwd=raiz, capture_output=True,
                               text=True, timeout=15)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:  # noqa: BLE001 - git ausente não derruba o diagnóstico
            return None

    if _git("rev-parse", "--git-dir") is None:
        print(f"  {AVISO} git indisponível ou pasta não é um repositório")
        return

    ramo = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    print(f"  ℹ️  branch                  {ramo}")

    # merge.ours.driver: sem ele, recalcular o pipeline localmente vira
    # conflito contra os artefatos já commitados (.gitattributes depende dele,
    # e o driver NÃO viaja no commit — é configuração por máquina).
    tem_driver = (_git("config", "merge.ours.driver") or "").lower() in {"true", "1"}
    print(f"  {OK if tem_driver else AVISO} merge.ours.driver       "
          f"{'configurado' if tem_driver else 'AUSENTE'}")
    if not tem_driver:
        print("  → configure uma vez: git config merge.ours.driver true")

    # Arquivos sujos que bloqueiam checkout/pull. Estado de aplicativo
    # (Obsidian, plugins) é o caso recorrente e pode ser descartado sem perda.
    # Corte por espaço, não por posição fixa: `_git` faz strip() e come o
    # espaço inicial da PRIMEIRA linha do --porcelain, o que deslocava o nome
    # do arquivo em um caractere ("cripts/..." em vez de "scripts/...").
    sujos = [ln.split(maxsplit=1)[-1]
             for ln in (_git("status", "--porcelain") or "").splitlines()
             if ln.strip() and not ln.strip().startswith("??")]
    if not sujos:
        print(f"  {OK} árvore limpa            nenhum arquivo bloqueia o pull")
    else:
        estado_app = [f for f in sujos
                      if "/.obsidian/" in f or "/.smart-env/" in f]
        seus = [f for f in sujos if f not in estado_app]
        print(f"  {AVISO} {len(sujos)} arquivo(s) modificado(s) — podem bloquear o pull")
        if estado_app:
            print(f"     • {len(estado_app)} de estado do Obsidian/plugins "
                  "(descartável: git restore notas/)")
        if seus:
            print(f"     • {len(seus)} FORA de notas/ — confira antes de descartar:")
            for f in seus[:5]:
                print(f"         {f}")


def checar_nucleo() -> None:
    secao("Núcleo (obrigatório)")
    for dist in ("numpy", "pandas", "scipy", "scikit-learn", "torch",
                 "chromadb", "streamlit", "sentence-transformers", "pytest"):
        v = _versao(dist)
        print(f"  {OK if v else FALTA} {dist:24s} {v or '(ausente)'}")


def checar_opcionais() -> None:
    secao("Opcionais (degradação honesta)")
    mapa = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
    }
    for mod, desc in mapa.items():
        try:
            disponivel = importlib.util.find_spec(mod) is not None
        except Exception:
            disponivel = False
        print(f"  {OK if disponivel else AVISO} {mod:20s} {desc}"
              f"{'' if disponivel else '  (recurso fica indisponível)'}")


def checar_chaves() -> None:
    secao("Chaves de API (.env) — só presença, nunca o valor")
    from dotenv import load_dotenv

    load_dotenv()
    # Equipe 100% Gemini: uma única chave atende conversa, auditoria e fundo.
    for chave in ("GOOGLE_API_KEY",):
        tem = bool(os.getenv(chave))
        print(f"  {OK if tem else AVISO} {chave:18s} {'configurada' if tem else 'ausente'}")


def checar_datasets() -> None:
    secao("Datasets (presença + SHA-256 + linhas)")
    try:
        from scripts.verificar_datasets import verificar as vd

        vd(silencioso=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  {AVISO} não foi possível verificar datasets: {exc}")


def checar_chromadb() -> None:
    secao("ChromaDB")
    try:
        import chromadb

        from src.core.config import NOME_COLECAO, PASTA_CHROMADB

        cli = chromadb.PersistentClient(path=str(PASTA_CHROMADB))
        nomes = [c.name for c in cli.list_collections()]
        print(f"  {OK} coleções: {nomes}")
        if NOME_COLECAO in nomes:
            print(f"  {OK} {NOME_COLECAO}: {cli.get_collection(NOME_COLECAO).count()} chunks")
    except Exception as exc:  # noqa: BLE001
        print(f"  {AVISO} ChromaDB indisponível: {exc}")


def checar_pipeline() -> None:
    secao("Pipeline de ML (ready / stale / pending)")
    try:
        from src.ml.pipeline import NOMES_ETAPAS, estado_pipeline

        rotulo = {"ready": OK, "stale": AVISO, "pending": "⬜"}
        for key, info in estado_pipeline().items():
            est = info["estado"]
            extra = f" — {', '.join(info.get('motivos', []))}" if est != "ready" else ""
            print(f"  {rotulo.get(est, '?')} {NOMES_ETAPAS[key]:22s} {est}{extra}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {AVISO} pipeline indisponível: {exc}")


def main() -> int:
    print("AL IADO PV — verificação de ambiente")
    # Interpretador e Git primeiro: são as causas que fazem TODO o resto
    # falhar de forma confusa (agente que não importa, pull que não passa).
    checar_python()
    checar_git()
    checar_nucleo()
    checar_opcionais()
    checar_chaves()
    checar_datasets()
    checar_chromadb()
    checar_pipeline()
    print("\nDiagnóstico concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
