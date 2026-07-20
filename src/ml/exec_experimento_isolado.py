"""
exec_experimento_isolado.py — Sprint 5 (10.4): isolamento de cargas pesadas.

Roda um experimento por artigo em um SUBPROCESSO separado, de modo que uma
falha catastrófica de biblioteca pesada (torch,
torch — segfault, conflito de OpenMP, estouro de memória) NÃO derrube o app
Streamlit nem o terminal do agente.

Contrato:
  - O pai monta o comando `python -m src.ml.exec_experimento_isolado <key> <out>`,
    captura o stdout do filho linha-a-linha (encaminhado como progresso) e lê o
    resultado de um arquivo JSON temporário.
  - O filho roda o `executar_experimento` IN-PROCESS (marcado por
    AL_IADO_EXP_CHILD=1 para evitar recursão), serializa o resultado e o grava.

Degradação honesta: se o subprocesso não puder ser lançado, cai de volta para a
execução in-process. Windows-friendly (usa subprocess, sem os.fork).

Chaves de ambiente:
  - AL_IADO_SEM_ISOLAMENTO=1  → força execução in-process (debug/CI).
  - AL_IADO_EXP_CHILD=1       → marcador interno do filho (não definir à mão).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Raiz do repositório (…/src/ml/exec_experimento_isolado.py → parents[2]).
_RAIZ = Path(__file__).resolve().parents[2]

TIMEOUT_PADRAO_S = 3600  # 1 h — cobre treino de RL/torch em CPU.


def _slug(texto: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(texto).lower()).strip("_")[:40] or "exp"


def _rodar_inproc(key: str, progresso=None) -> dict:
    """Caminho in-process (sem isolamento)."""
    from src.ml.experimentos_artigos import executar_experimento

    return executar_experimento(key, progresso=progresso)


def executar_experimento_isolado(
    key: str, progresso=None, timeout_s: int = TIMEOUT_PADRAO_S
) -> dict:
    """
    Executa o experimento `key` em subprocesso isolado e devolve o dict de
    resultado (mesmo formato de `executar_experimento`). Nunca levanta: em
    qualquer falha de isolamento devolve um dict com ``ok=False`` e mensagem,
    ou cai para in-process quando o subprocesso sequer pôde ser lançado.
    """
    # Opt-out global ou já estamos dentro do filho → in-process direto.
    if (
        os.environ.get("AL_IADO_SEM_ISOLAMENTO") == "1"
        or os.environ.get("AL_IADO_EXP_CHILD") == "1"
    ):
        return _rodar_inproc(key, progresso=progresso)

    fd, caminho_out = tempfile.mkstemp(prefix=f"exp_{_slug(key)}_", suffix=".json")
    os.close(fd)
    out = Path(caminho_out)

    cmd = [sys.executable, "-m", "src.ml.exec_experimento_isolado", str(key), str(out)]
    # Menor privilégio: o filho treina modelos LOCAIS e não precisa de chaves
    # de API — env_minimo_subprocesso remove GROQ/GOOGLE/etc. e já define
    # KMP/UTF-8 com defaults seguros.
    from src.core.seguranca import env_minimo_subprocesso

    env = env_minimo_subprocesso(extras={"AL_IADO_EXP_CHILD": "1"})

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_RAIZ),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except Exception as exc:  # noqa: BLE001 — não conseguiu lançar → fallback
        if progresso:
            progresso(f"[isolamento indisponível: {exc}; rodando in-process]")
        try:
            out.unlink()
        except OSError:
            pass
        return _rodar_inproc(key, progresso=progresso)

    ultimas: list[str] = []
    try:
        assert proc.stdout is not None
        for linha in proc.stdout:
            linha = linha.rstrip("\n")
            if not linha:
                continue
            ultimas.append(linha)
            if len(ultimas) > 60:
                ultimas.pop(0)
            if progresso:
                try:
                    progresso(linha)
                except Exception:  # noqa: BLE001 — callback do app não derruba
                    pass
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        _limpar(out)
        return {
            "experimento": key,
            "ok": False,
            "mensagem": (
                f"Experimento '{key}' excedeu {timeout_s}s e foi interrompido "
                f"(execução isolada)."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        _limpar(out)
        return {
            "experimento": key,
            "ok": False,
            "mensagem": f"Falha ao executar '{key}' isolado: {exc}",
        }

    # O filho escreve o JSON (com ok True/False) e SÓ ENTÃO encerra com 0. Como
    # mkstemp já criou o arquivo VAZIO, o sinal confiável de catástrofe
    # (segfault/OOM/erro de import — que nem chega a gravar) é o conteúdo VAZIO
    # ou ilegível. Se há JSON válido, ele prevalece — mesmo com returncode != 0
    # (um ok=False legítimo: "cartão de dataset", "lib faltando").
    try:
        conteudo = out.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        conteudo = ""
    finally:
        _limpar(out)

    if not conteudo.strip():
        tail = "\n".join(ultimas[-15:])
        return {
            "experimento": key,
            "ok": False,
            "mensagem": (
                f"Experimento '{key}' falhou no subprocesso isolado "
                f"(código {proc.returncode}; sem resultado gravado — provável "
                f"crash de biblioteca).\nÚltimas linhas:\n{tail}"
            ),
        }

    try:
        return json.loads(conteudo)
    except Exception as exc:  # noqa: BLE001
        return {
            "experimento": key,
            "ok": False,
            "mensagem": f"Resultado isolado ilegível para '{key}': {exc}",
        }


def _limpar(caminho: Path) -> None:
    try:
        caminho.unlink()
    except OSError:
        pass


def _main(argv: list[str]) -> int:
    """Entrada do FILHO: roda o experimento e grava o JSON serializável."""
    if len(argv) < 3:
        print("uso: python -m src.ml.exec_experimento_isolado <key> <out.json>")
        return 2

    key, caminho_out = argv[1], argv[2]

    # Garante a raiz no sys.path mesmo se o cwd não for o esperado.
    if str(_RAIZ) not in sys.path:
        sys.path.insert(0, str(_RAIZ))

    from src.core.utils import configurar_saida_utf8

    configurar_saida_utf8()
    os.environ["AL_IADO_EXP_CHILD"] = "1"  # impede recursão de isolamento

    from src.ml.experimentos_artigos import (
        executar_experimento,
        _resultado_serializavel,
    )

    def _prog(msg):
        print(msg, flush=True)

    try:
        res = executar_experimento(key, progresso=_prog)
    except Exception as exc:  # noqa: BLE001 — o pai detecta via returncode/arquivo
        res = {"experimento": key, "ok": False, "mensagem": f"erro no experimento: {exc}"}

    try:
        serial = _resultado_serializavel(res)
    except Exception:  # noqa: BLE001
        serial = {
            "experimento": key,
            "ok": bool(res.get("ok")),
            "mensagem": str(res.get("mensagem", "")),
        }

    Path(caminho_out).write_text(
        json.dumps(serial, ensure_ascii=False), encoding="utf-8"
    )
    # Sempre 0 após gravar: o `ok` viaja DENTRO do JSON. Reservar exit != 0 só
    # para crash real (que nem chega aqui) evita o pai descartar um ok=False
    # legítimo (ex.: "cartão de dataset" ou "lib faltando").
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
