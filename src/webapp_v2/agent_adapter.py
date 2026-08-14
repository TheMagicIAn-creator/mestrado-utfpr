"""Adaptador do agente para HTTP, sem dependência de Streamlit."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.core.config import RAIZ_PROJETO
from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.webapp_v2.scientific_context import scientific_context_for
from src.webapp_v2.session_journal import SessionJournal

MAX_MESSAGE_CHARS = 12_000
MAX_HISTORY_ITEMS = 16
MAX_ATTACHMENTS = 4
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024
_logger = get_logger("webapp.agent_adapter")
AGENT_ENGINE = "src.conhecimento.agente.perguntar"


class AgenteIndisponivel(RuntimeError):
    """O agente não pôde ser inicializado ou responder."""


@dataclass
class _Componentes:
    perfil: str
    modelo_embeddings: object
    literatura: object
    sessoes: object
    obsidian: object
    indice_lexical: object
    llm: object
    auditor: object
    modo_consulta: bool


def _historico_normalizado(historico) -> list[dict[str, str]]:
    if historico is None:
        return []
    if not isinstance(historico, list):
        raise ValueError("history deve ser uma lista")
    saida = []
    for item in historico[-MAX_HISTORY_ITEMS:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise ValueError("Cada item do histórico deve ter role user ou assistant")
        conteudo = str(item.get("content") or "").strip()
        if conteudo:
            saida.append({"role": item["role"], "content": conteudo[:MAX_MESSAGE_CHARS]})
    return saida


def _validar_anexos(anexos) -> list[tuple[str, bytes]]:
    if not anexos:
        return []
    if len(anexos) > MAX_ATTACHMENTS:
        raise ValueError(f"No máximo {MAX_ATTACHMENTS} anexos por mensagem")
    saida = []
    for nome, dados in anexos:
        nome_seguro = Path(str(nome)).name[:180] or "anexo"
        if not isinstance(dados, bytes):
            raise ValueError("Conteúdo de anexo inválido")
        if len(dados) > MAX_ATTACHMENT_BYTES:
            raise ValueError(f"{nome_seguro}: limite de 15 MB excedido")
        saida.append((nome_seguro, dados))
    return saida


class AgentAdapter:
    """Inicializa a base somente no primeiro turno e serializa esse carregamento."""

    def __init__(
        self,
        answerer: Callable | None = None,
        session_journal: SessionJournal | None = None,
    ):
        self._answerer = answerer
        self._session_journal = session_journal or SessionJournal()
        self._lock = threading.RLock()
        self._components: _Componentes | None = None
        self._state = "ready" if answerer is not None else "idle"
        self._error: str | None = None

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "provider": "Google Gemini",
                "engine": AGENT_ENGINE,
                "retrieval": ["semantic", "bm25", "sessions", "obsidian"],
                "evidence_audit": True,
                "detail": self._error,
                "lazy_initialization": True,
            }

    def initialize(self) -> dict:
        """Aquece o mesmo runtime usado pela conversa e devolve seu estado."""
        if self._answerer is None:
            self._initialize()
        return self.status()

    def reset(self) -> None:
        with self._lock:
            self._components = None
            self._error = None
            self._state = "ready" if self._answerer is not None else "idle"

    def _initialize(self) -> _Componentes:
        with self._lock:
            if self._components is not None:
                return self._components
            self._state = "loading"
            self._error = None
            try:
                from src.conhecimento.base_runtime import carregar_base_conhecimento
                from src.conhecimento.multiagente import criar_equipe_agentes

                equipe = criar_equipe_agentes()
                base = carregar_base_conhecimento()
                self._components = _Componentes(
                    perfil=base.perfil,
                    modelo_embeddings=base.modelo_embeddings,
                    literatura=base.literatura,
                    sessoes=base.sessoes,
                    obsidian=base.obsidian,
                    indice_lexical=base.indice_lexical,
                    llm=equipe.conversa,
                    auditor=equipe.auditoria,
                    modo_consulta=base.modo_consulta,
                )
                self._state = "ready"
                return self._components
            except Exception as exc:
                self._state = "error"
                self._error = mascarar_segredos(str(exc))
                raise AgenteIndisponivel(self._error) from exc

    @staticmethod
    def _contexto(historico: list[dict[str, str]]) -> str:
        nomes = {"user": "Rodolfo", "assistant": "ALIAdo PV"}
        return "\n\n".join(
            f"{nomes[item['role']]}: {item['content']}" for item in historico[-8:]
        )

    @staticmethod
    def _imagem_publica(item: dict) -> dict | None:
        bruto = item.get("path") or item.get("caminho")
        if not bruto:
            return None
        path = Path(bruto)
        if not path.is_absolute():
            path = Path(RAIZ_PROJETO) / path
        try:
            relativo = path.resolve().relative_to(
                (Path(RAIZ_PROJETO) / "resultados" / "v2").resolve()
            )
        except (OSError, ValueError):
            return None
        partes = relativo.parts
        if not partes or partes[0] not in {"autoencoder", "confiabilidade"}:
            return None
        return {
            "url": "/artifacts/" + relativo.as_posix(),
            "caption": str(item.get("caption") or item.get("legenda") or path.stem),
        }

    def _answer_real(
        self,
        mensagem: str,
        historico: list[dict[str, str]],
        anexos_bytes: list[tuple[str, bytes]],
    ) -> dict:
        componentes = self._initialize()
        contexto_cientifico = scientific_context_for(mensagem)
        anexos = []
        if anexos_bytes:
            from src.conhecimento.leitor_anexos import ler_anexos

            anexos = ler_anexos(anexos_bytes)

        if not anexos:
            try:
                from src.conhecimento.ferramentas import (
                    decidir_acao,
                    processar_com_ferramentas,
                )

                decisao = decidir_acao(mensagem, componentes.llm)
                if decisao.get("usar_ferramenta"):
                    saida = processar_com_ferramentas(
                        pergunta=mensagem,
                        perfil=componentes.perfil,
                        llm=componentes.llm,
                        progresso=None,
                        decisao=decisao,
                        contexto="\n\n".join(
                            item
                            for item in (
                                self._contexto(historico),
                                contexto_cientifico,
                            )
                            if item
                        ),
                    )
                    imagens = []
                    if saida.get("resultado"):
                        for item in saida["resultado"].get("imagens", []):
                            publica = self._imagem_publica(item)
                            if publica:
                                imagens.append(publica)
                    resposta_ferramenta = saida.get("resposta") or "Sem resposta."
                    if "Verificação de citações" not in resposta_ferramenta:
                        from src.core.citacao_guarda import alerta_citacao_infundada

                        aviso = alerta_citacao_infundada(resposta_ferramenta, {})
                        if aviso:
                            resposta_ferramenta = aviso.strip() + "\n\n" + resposta_ferramenta
                    return {
                        "answer": resposta_ferramenta,
                        "images": imagens,
                        "route": "tool",
                        "scientific_contract": "v2" if contexto_cientifico else None,
                    }
            except AgenteIndisponivel:
                raise
            except Exception as exc:
                # O RAG permanece disponível quando o classificador de ferramenta falha.
                _logger.warning("roteamento por ferramenta indisponivel; usando RAG: %s", exc)

        if not componentes.modo_consulta:
            try:
                from src.conhecimento.obsidian import sincronizar_obsidian

                sincronizar_obsidian(
                    componentes.obsidian,
                    componentes.modelo_embeddings,
                )
            except Exception as exc:
                _logger.warning("sincronizacao incremental do Obsidian falhou: %s", exc)

        from src.conhecimento.agente import perguntar

        resposta = perguntar(
            pergunta=mensagem,
            perfil=componentes.perfil,
            modelo_embeddings=componentes.modelo_embeddings,
            colecao=componentes.literatura,
            llm=componentes.llm,
            historico=historico,
            streaming=False,
            colecao_sessoes=componentes.sessoes,
            nome_provedor="Google Gemini",
            anexos=anexos,
            colecao_obsidian=componentes.obsidian,
            indice_lexical=componentes.indice_lexical,
            auditor=componentes.auditor,
            contexto_autoritativo=contexto_cientifico,
        )
        return {
            "answer": resposta,
            "images": [],
            "route": "rag",
            "scientific_contract": "v2" if contexto_cientifico else None,
        }

    def answer(
        self,
        message: str,
        history=None,
        attachments=None,
        session_id: str | None = None,
    ) -> dict:
        mensagem = str(message or "").strip()
        if not mensagem:
            raise ValueError("A mensagem não pode estar vazia")
        if len(mensagem) > MAX_MESSAGE_CHARS:
            raise ValueError(f"A mensagem excede {MAX_MESSAGE_CHARS} caracteres")
        historico = _historico_normalizado(history)
        anexos = _validar_anexos(attachments)
        session_id = self._session_journal.validar_session_id(session_id)

        if self._answerer is None and not anexos:
            from src.conhecimento.agente_interacao import resposta_interacao_simples

            resposta_simples = resposta_interacao_simples(mensagem)
            if resposta_simples:
                return {
                    "answer": resposta_simples,
                    "images": [],
                    "route": "local",
                    "memories_saved": 0,
                }

        try:
            if self._answerer is not None:
                resposta = self._answerer(mensagem, historico, anexos)
            else:
                resposta = self._answer_real(mensagem, historico, anexos)
        except (ValueError, AgenteIndisponivel):
            raise
        except Exception as exc:
            erro = mascarar_segredos(str(exc))
            with self._lock:
                self._error = erro
                self._state = "error"
            raise AgenteIndisponivel(erro) from exc

        if isinstance(resposta, str):
            resposta = {"answer": resposta, "images": [], "route": "adapter"}
        if not isinstance(resposta, dict) or not str(resposta.get("answer") or "").strip():
            raise AgenteIndisponivel("O agente retornou uma resposta vazia")
        resposta.setdefault("images", [])
        resposta.setdefault("route", "adapter")
        resposta.setdefault("memories_saved", 0)
        if self._answerer is None and self._components is not None:
            aprender = getattr(self._components.auditor, "aprender_da_interacao", None)
            if callable(aprender):
                try:
                    aprendizado = aprender(mensagem, str(resposta["answer"]))
                    resposta["memories_saved"] = int(
                        getattr(aprendizado, "salvas", 0) or 0
                    )
                except Exception as exc:
                    _logger.warning("aprendizado do turno indisponivel: %s", exc)
            resposta["session"] = self._session_journal.record(
                session_id,
                mensagem,
                str(resposta["answer"]),
                resposta["images"],
                self._components.modelo_embeddings,
            )
        with self._lock:
            self._state = "ready"
            self._error = None
        return resposta
