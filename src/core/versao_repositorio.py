"""Detecta clone defasado ANTES de uma execução cara gastar o GPVS.

POR QUE EXISTE
==============
Duas vezes a campanha E2 rodou inteira num clone que estava num commit anterior
ao código vigente, e o sintoma só apareceu no fim, no `output_count` do JSON de
saída: a etapa publicou menos artefatos do que a versão corrente publica. Entre
carregar os 16 CSVs, reextrair features e varrer a severidade dos três itens,
isso custa a execução completa para descobrir uma coisa que `git` responde em
um segundo.

COMO A COMPARAÇÃO É FEITA
=========================
Sem `git fetch`. `ls-remote` pergunta ao remoto qual é o SHA da branch atual e
não escreve nada em `.git`; o veredito sai de comparar aquele SHA com o HEAD
local:

- SHA igual              -> em dia;
- objeto remoto ausente  -> o remoto avançou e este clone não puxou: ATRASADO;
- remoto é ancestral     -> há commit local ainda não enviado: adiantado, ok;
- nenhum dos dois        -> as histórias divergiram.

NADA AQUI LEVANTA POR CONTA DO AMBIENTE
=======================================
Sem rede, sem `git`, fora de um repositório, em HEAD destacado ou numa branch
sem correspondente no remoto, o estado sai como indeterminado e a execução
segue. Uma guarda de conveniência que impeça de trabalhar offline seria pior do
que o problema que ela evita.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.core.config import RAIZ_PROJETO

TEMPO_LIMITE_S = 15

EM_DIA = "em_dia"
ATRASADO = "atrasado"
ADIANTADO = "adiantado"
DIVERGENTE = "divergente"
INDETERMINADO = "indeterminado"


@dataclass(frozen=True)
class EstadoDoClone:
    estado: str
    motivo: str
    branch: str | None = None
    local: str | None = None
    remoto: str | None = None

    @property
    def precisa_de_pull(self) -> bool:
        return self.estado in {ATRASADO, DIVERGENTE}

    def mensagem(self) -> str:
        """Texto pronto para log ou erro, já com o comando a executar."""
        if not self.precisa_de_pull:
            return self.motivo
        alvo = self.branch or "<branch>"
        return (
            f"{self.motivo} Rode `git pull origin {alvo}` antes de executar; "
            "publicar a partir de código defasado gera artefato que não "
            "corresponde à etapa vigente."
        )


def _git(*argumentos: str, cwd: Path) -> str | None:
    """Executa git e devolve a saída, ou None em qualquer condição adversa."""
    try:
        processo = subprocess.run(
            ["git", *argumentos],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TEMPO_LIMITE_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if processo.returncode != 0:
        return None
    return processo.stdout.strip()


def estado_do_clone(raiz: Path | str = RAIZ_PROJETO) -> EstadoDoClone:
    """Compara o HEAD local com a branch correspondente no remoto."""
    raiz = Path(raiz)

    local = _git("rev-parse", "HEAD", cwd=raiz)
    if not local:
        return EstadoDoClone(INDETERMINADO, "Não é um repositório git utilizável.")

    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=raiz)
    if not branch or branch == "HEAD":
        return EstadoDoClone(
            INDETERMINADO,
            "HEAD destacado: não há branch para comparar com o remoto.",
            local=local,
        )

    saida = _git("ls-remote", "origin", f"refs/heads/{branch}", cwd=raiz)
    if saida is None:
        return EstadoDoClone(
            INDETERMINADO,
            "Remoto inacessível; a verificação de versão foi ignorada.",
            branch=branch,
            local=local,
        )
    if not saida:
        return EstadoDoClone(
            INDETERMINADO,
            f"A branch `{branch}` não existe no remoto.",
            branch=branch,
            local=local,
        )

    remoto = saida.split()[0]
    comum = {"branch": branch, "local": local, "remoto": remoto}
    if remoto == local:
        return EstadoDoClone(EM_DIA, f"Clone em dia com origin/{branch}.", **comum)

    if _git("cat-file", "-e", f"{remoto}^{{commit}}", cwd=raiz) is None:
        return EstadoDoClone(
            ATRASADO,
            f"origin/{branch} está em {remoto[:7]} e este clone, em {local[:7]}.",
            **comum,
        )

    if _git("merge-base", "--is-ancestor", remoto, local, cwd=raiz) is not None:
        return EstadoDoClone(
            ADIANTADO,
            f"Há commit local ainda não enviado para origin/{branch}.",
            **comum,
        )

    if _git("merge-base", "--is-ancestor", local, remoto, cwd=raiz) is not None:
        return EstadoDoClone(
            ATRASADO,
            f"origin/{branch} tem commit que este clone não possui.",
            **comum,
        )

    return EstadoDoClone(
        DIVERGENTE,
        f"As histórias local e origin/{branch} divergiram.",
        **comum,
    )


class CloneDefasado(RuntimeError):
    """O clone está atrás do remoto e a execução publicaria código velho."""


def exigir_clone_atualizado(
    *,
    raiz: Path | str = RAIZ_PROJETO,
    progresso=None,
) -> EstadoDoClone:
    """Levanta `CloneDefasado` quando o clone está atrás; caso contrário, segue.

    Estado indeterminado nunca bloqueia: offline, sem git ou em HEAD destacado
    a execução continua normalmente.
    """
    estado = estado_do_clone(raiz)
    if progresso and estado.estado != EM_DIA:
        progresso(estado.mensagem())
    if estado.precisa_de_pull:
        raise CloneDefasado(estado.mensagem())
    return estado


__all__ = [
    "ADIANTADO",
    "ATRASADO",
    "DIVERGENTE",
    "EM_DIA",
    "INDETERMINADO",
    "CloneDefasado",
    "EstadoDoClone",
    "estado_do_clone",
    "exigir_clone_atualizado",
]
