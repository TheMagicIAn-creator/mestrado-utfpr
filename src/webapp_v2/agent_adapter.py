"""Adaptador do agente para HTTP, sem dependência de Streamlit."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
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
DEFAULT_OBSIDIAN_SYNC_INTERVAL_S = 300.0


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
        self._initialization_lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self._last_obsidian_sync: float | None = None
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

        # A carga da base pode levar dezenas de segundos. Um lock separado
        # serializa esse trabalho sem bloquear status e respostas locais.
        with self._initialization_lock:
            with self._lock:
                if self._components is not None:
                    return self._components
                self._state = "loading"
                self._error = None
            try:
                from src.conhecimento.base_runtime import carregar_base_conhecimento
                from src.conhecimento.multiagente import criar_equipe_agentes

                equipe = criar_equipe_agentes()
                base = carregar_base_conhecimento(
                    sincronizar_obsidian_local=False,
                    embeddings_baixo_consumo=True,
                )
                componentes = _Componentes(
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
                with self._lock:
                    self._components = componentes
                    self._state = "ready"
                    self._error = None
                return componentes
            except Exception as exc:
                erro = mascarar_segredos(str(exc))
                with self._lock:
                    self._state = "error"
                    self._error = erro
                raise AgenteIndisponivel(erro) from exc

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
                from src.conhecimento.roteamento_ferramentas import _decisao_rapida

                decisao_local = (
                    _decisao_rapida(mensagem) if contexto_cientifico else None
                )
                # Perguntas discursivas inequivocas sobre os contratos V2 nao
                # precisam gastar uma chamada ao Gemini apenas para concluir
                # que nenhuma ferramenta deve ser executada.
                if decisao_local and not decisao_local.get("usar_ferramenta"):
                    decisao = decisao_local
                else:
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

    def _postprocess_interaction(
        self,
        mensagem: str,
        resposta: dict,
        session_id: str | None,
        componentes: _Componentes,
    ) -> None:
        """Persiste sessao e memoria sem atrasar a resposta HTTP."""
        with self._maintenance_lock:
            try:
                self._session_journal.record(
                    session_id,
                    mensagem,
                    str(resposta["answer"]),
                    resposta["images"],
                    componentes.modelo_embeddings,
                )
            except Exception as exc:
                _logger.warning("registro assincrono da sessao falhou: %s", exc)

            aprender = getattr(componentes.auditor, "aprender_da_interacao", None)
            if callable(aprender):
                try:
                    aprender(mensagem, str(resposta["answer"]))
                except Exception as exc:
                    _logger.warning("aprendizado assincrono indisponivel: %s", exc)

            self._sync_obsidian_if_due(componentes)

    def _sync_obsidian_if_due(self, componentes: _Componentes) -> None:
        """Atualiza notas locais fora do caminho critico e com limitacao temporal."""
        if componentes.modo_consulta:
            return
        try:
            intervalo = max(
                0.0,
                float(
                    os.getenv(
                        "AL_IADO_OBSIDIAN_SYNC_INTERVAL_S",
                        str(DEFAULT_OBSIDIAN_SYNC_INTERVAL_S),
                    )
                ),
            )
        except ValueError:
            intervalo = DEFAULT_OBSIDIAN_SYNC_INTERVAL_S

        agora = monotonic()
        if (
            self._last_obsidian_sync is not None
            and agora - self._last_obsidian_sync < intervalo
        ):
            return
        self._last_obsidian_sync = agora
        try:
            from src.conhecimento.obsidian import sincronizar_obsidian

            sincronizar_obsidian(
                componentes.obsidian,
                componentes.modelo_embeddings,
            )
        except Exception as exc:
            _logger.warning("sincronizacao assincrona do Obsidian falhou: %s", exc)

    def _schedule_postprocessing(
        self,
        mensagem: str,
        resposta: dict,
        session_id: str | None,
        componentes: _Componentes,
    ) -> None:
        threading.Thread(
            target=self._postprocess_interaction,
            args=(mensagem, resposta, session_id, componentes),
            name="aliado-v2-maintenance",
            daemon=True,
        ).start()

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
            componentes = self._components
            self._schedule_postprocessing(
                mensagem,
                resposta,
                session_id,
                componentes,
            )
            resposta["maintenance_scheduled"] = True
        with self._lock:
            self._state = "ready"
            self._error = None
        return resposta
