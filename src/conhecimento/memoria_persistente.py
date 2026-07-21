"""Memoria estruturada, validada e reutilizavel entre sessoes.

As sessoes em Markdown continuam sendo o registro conversacional completo.
Este modulo guarda somente fatos operacionais duraveis aprovados pelo agente
auditor: preferencias, decisoes, correcoes e contexto estavel do projeto.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.core.config import ARQUIVO_MEMORIA_VALIDADA, PASTA_CEREBRO_OBSIDIAN


SCHEMA_VERSION = 1
TIPOS_PERMITIDOS = {
    "preferencia",
    "decisao_metodologica",
    "correcao",
    "contexto_projeto",
}
ESCOPOS_PERMITIDOS = {"conversa", "literatura", "ml", "compartilhado"}
STATUS_ATIVO = "ativo"
STATUS_SUPERADO = "superado"

_SEGREDOS = (
    re.compile(r"AIza[A-Za-z0-9_-]{25,}"),
    re.compile(r"gsk_[A-Za-z0-9_-]{15,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"(?i)(api[_ -]?key|token|senha|password)\s*[:=]\s*\S{8,}"),
)
_METRICAS_VOLATEIS = re.compile(
    r"(?i)\b(auc|f1|mttf|b10|rmse|mae|acuracia|accuracy|limiar)\b.{0,24}\d"
)
_LOCK = threading.RLock()


class MemoriaInvalida(ValueError):
    """Candidato de memoria reprovado pelas regras locais."""


class MemoriaCorrompida(ValueError):
    """O arquivo persistente existe, mas nao segue o schema esperado."""


@dataclass(frozen=True)
class ResultadoRegistro:
    item: dict
    criado: bool


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalizar(texto: str) -> str:
    base = unicodedata.normalize("NFKD", str(texto).lower())
    sem_acentos = "".join(c for c in base if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", sem_acentos)).strip()


def _tokens(texto: str) -> set[str]:
    tokens = {t for t in _normalizar(texto).split() if len(t) >= 3}
    # Flexao nominal simples cobre o caso mais frequente da memoria curta
    # (resposta/respostas, objetiva/objetivas) sem introduzir dependencias.
    tokens.update(t[:-1] for t in list(tokens) if len(t) > 4 and t.endswith("s"))
    return tokens


def _vazio() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "atualizado_em_utc": None,
        "itens": [],
    }


class MemoriaPersistente:
    """Repositorio JSON atomico de memorias aprovadas pelo auditor."""

    def __init__(
        self,
        caminho: str | Path | None = None,
        *,
        pasta_obsidian: str | Path | None = None,
    ) -> None:
        caminho_padrao = caminho is None
        self.caminho = Path(caminho or ARQUIVO_MEMORIA_VALIDADA)
        self.pasta_obsidian = Path(
            pasta_obsidian or PASTA_CEREBRO_OBSIDIAN
        ) if (caminho_padrao or pasta_obsidian is not None) else None

    def _espelhar_obsidian(self) -> None:
        if self.pasta_obsidian is None:
            return
        try:
            from src.conhecimento.obsidian import espelhar_memoria_validada

            espelhar_memoria_validada(
                self.caminho,
                raiz=self.pasta_obsidian,
            )
        except Exception:
            # O Markdown e uma visao derivada. Falha no espelho nunca pode
            # invalidar o JSON atomico que acabou de ser aprovado.
            pass

    def _persistir_nuvem(self) -> None:
        """Commita o JSON de volta ao GitHub quando na nuvem (Streamlit Cloud).

        Best-effort e desligado por padrao: so faz algo com o master switch
        AL_IADO_PERSISTIR_NUVEM e um token presentes. Falha aqui nunca invalida
        a gravacao local ja concluida.
        """
        try:
            from src.conhecimento.persistencia_nuvem import (
                persistencia_ativa,
                persistir_memoria_validada,
            )

            if persistencia_ativa():
                persistir_memoria_validada(self.caminho)
        except Exception:
            pass

    def _ler(self, *, estrito: bool = False) -> dict:
        if not self.caminho.is_file():
            return _vazio()
        try:
            dados = json.loads(self.caminho.read_text(encoding="utf-8"))
            if dados.get("schema_version") != SCHEMA_VERSION:
                raise MemoriaCorrompida("Versao de schema incompativel.")
            if not isinstance(dados.get("itens"), list):
                raise MemoriaCorrompida("Campo 'itens' invalido.")
            return dados
        except (OSError, json.JSONDecodeError, AttributeError, MemoriaCorrompida) as exc:
            if estrito:
                raise MemoriaCorrompida(
                    f"Memoria persistente ilegivel em {self.caminho}: {exc}"
                ) from exc
            return _vazio()

    def _salvar(self, dados: dict) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        temporario = self.caminho.with_name(self.caminho.name + ".tmp")
        try:
            temporario.write_text(
                json.dumps(dados, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporario, self.caminho)
        finally:
            temporario.unlink(missing_ok=True)

    @staticmethod
    def _validar_candidato(candidato: dict) -> dict:
        tipo = str(candidato.get("tipo", "")).strip().lower()
        escopo = str(candidato.get("escopo", "compartilhado")).strip().lower()
        conteudo = re.sub(r"\s+", " ", str(candidato.get("conteudo", ""))).strip()
        evidencia = re.sub(
            r"\s+", " ", str(candidato.get("evidencia_usuario", ""))
        ).strip()

        if tipo not in TIPOS_PERMITIDOS:
            raise MemoriaInvalida(f"Tipo de memoria nao permitido: {tipo}")
        if escopo not in ESCOPOS_PERMITIDOS:
            raise MemoriaInvalida(f"Escopo de memoria nao permitido: {escopo}")
        if not (5 <= len(conteudo) <= 700):
            raise MemoriaInvalida("Conteudo deve ter entre 5 e 700 caracteres.")
        if not (3 <= len(evidencia) <= 700):
            raise MemoriaInvalida("Memoria sem evidencia direta do pesquisador.")
        if any(p.search(conteudo) or p.search(evidencia) for p in _SEGREDOS):
            raise MemoriaInvalida("Segredos nunca podem ser persistidos.")
        if _METRICAS_VOLATEIS.search(conteudo):
            raise MemoriaInvalida(
                "Metricas recalculaveis pertencem aos artefatos, nao a memoria."
            )

        termos_conteudo = _tokens(conteudo)
        termos_evidencia = _tokens(evidencia)
        if termos_conteudo and not (termos_conteudo & termos_evidencia):
            raise MemoriaInvalida(
                "O candidato nao esta ancorado no texto do pesquisador."
            )

        substitui_id = str(candidato.get("substitui_id", "")).strip() or None
        return {
            "tipo": tipo,
            "escopo": escopo,
            "conteudo": conteudo,
            "evidencia_usuario": evidencia,
            "substitui_id": substitui_id,
        }

    def registrar(
        self,
        candidato: dict,
        *,
        origem: str,
        validado_por: str,
        confianca: float,
    ) -> ResultadoRegistro:
        validado = self._validar_candidato(candidato)
        confianca = max(0.0, min(1.0, float(confianca)))
        if confianca < 0.70:
            raise MemoriaInvalida("Confianca insuficiente para persistir memoria.")

        assinatura = "|".join(
            [validado["tipo"], validado["escopo"], _normalizar(validado["conteudo"])]
        )
        item_id = hashlib.sha256(assinatura.encode("utf-8")).hexdigest()[:20]

        resultado = None
        with _LOCK:
            dados = self._ler(estrito=True)
            for item in dados["itens"]:
                if item.get("id") == item_id and item.get("status") == STATUS_ATIVO:
                    resultado = ResultadoRegistro(item=dict(item), criado=False)
                    break

            if resultado is None:
                agora = _agora()
                substitui_id = validado.pop("substitui_id")
                if substitui_id:
                    for item in dados["itens"]:
                        if item.get("id") == substitui_id and item.get("status") == STATUS_ATIVO:
                            item["status"] = STATUS_SUPERADO
                            item["superado_em_utc"] = agora

                item = {
                    "id": item_id,
                    **validado,
                    "origem": str(origem)[:160],
                    "validado_por": str(validado_por)[:120],
                    "confianca": round(confianca, 3),
                    "status": STATUS_ATIVO,
                    "versao": 1,
                    "criado_em_utc": agora,
                    "substitui_id": substitui_id,
                }
                dados["itens"].append(item)
                dados["atualizado_em_utc"] = agora
                self._salvar(dados)
                resultado = ResultadoRegistro(item=dict(item), criado=True)
        self._espelhar_obsidian()
        if resultado.criado:
            # Só toca o GitHub quando o arquivo REALMENTE mudou.
            self._persistir_nuvem()
        assert resultado is not None
        return resultado

    def listar(self, *, somente_ativas: bool = True) -> list[dict]:
        with _LOCK:
            itens = [dict(i) for i in self._ler().get("itens", [])]
        if somente_ativas:
            itens = [i for i in itens if i.get("status") == STATUS_ATIVO]
        return itens

    def contar(self) -> int:
        return len(self.listar(somente_ativas=True))

    def recuperar(self, consulta: str, *, limite: int = 6) -> list[dict]:
        itens = self.listar(somente_ativas=True)
        if not itens:
            return []
        termos_consulta = _tokens(consulta)
        if not termos_consulta:
            return sorted(
                itens, key=lambda i: i.get("criado_em_utc", ""), reverse=True
            )[:limite]

        documentos = [
            _tokens(" ".join([i.get("conteudo", ""), i.get("escopo", "")]))
            for i in itens
        ]
        frequencia = {
            termo: sum(1 for doc in documentos if termo in doc)
            for termo in termos_consulta
        }
        pontuados = []
        total = len(itens)
        consulta_norm = _normalizar(consulta)
        for item, doc in zip(itens, documentos):
            score = sum(
                math.log((total + 1) / (frequencia[t] + 0.5))
                for t in termos_consulta & doc
            )
            if _normalizar(item.get("conteudo", "")) in consulta_norm:
                score += 2.0
            if item.get("escopo") == "compartilhado":
                score += 0.15
            if score > 0:
                pontuados.append((score, item.get("criado_em_utc", ""), item))
        pontuados.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [dict(item) for _, _, item in pontuados[:limite]]

    def formatar_para_prompt(
        self,
        consulta: str,
        *,
        limite: int = 6,
        max_chars: int = 2400,
    ) -> str:
        linhas = []
        usados = 0
        for item in self.recuperar(consulta, limite=limite):
            linha = (
                f"- [{item['tipo']} | {item['escopo']} | id={item['id']}] "
                f"{item['conteudo']}"
            )
            if usados + len(linha) > max_chars:
                break
            linhas.append(linha)
            usados += len(linha)
        return "\n".join(linhas)
