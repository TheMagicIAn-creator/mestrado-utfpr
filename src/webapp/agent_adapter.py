"""Adaptador HTTP do agente ALIAdo."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable

from src.core.config import RAIZ_PROJETO
from src.core.identidade import nome_pesquisador
from src.core.logs import get_logger
from src.core.seguranca import mascarar_segredos
from src.webapp.scientific_context import scientific_context_for
from src.webapp.session_journal import SessionJournal

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


@dataclass(frozen=True)
class _PendingAction:
    action: str
    phase: str
    stages: tuple[str, ...]
    created_at: float


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
        library_service=None,
    ):
        self._answerer = answerer
        self._session_journal = session_journal or SessionJournal()
        self._lock = threading.RLock()
        self._initialization_lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self._pending_lock = threading.RLock()
        self._pending_actions: dict[str, _PendingAction] = {}
        self._library_service = library_service
        self._warm_thread: threading.Thread | None = None
        self._last_obsidian_sync: float | None = None
        self._components: _Componentes | None = None
        self._state = "pronto" if answerer is not None else "iniciando"
        self._error: str | None = None

    @property
    def session_journal(self) -> SessionJournal:
        """Catálogo persistente compartilhado pelas APIs de conversa."""

        return self._session_journal

    def configure_library_service(self, library_service) -> None:
        """Liga o adaptador ao mesmo serviço protegido usado pela API da biblioteca."""

        self._library_service = library_service

    @staticmethod
    def _local_action_response(answer: str, *, route: str = "local") -> dict:
        return {
            "answer": answer,
            "images": [],
            "route": route,
            "memories_saved": 0,
        }

    def _set_pending(self, session_id: str, pending: _PendingAction) -> None:
        with self._pending_lock:
            if len(self._pending_actions) >= 256 and session_id not in self._pending_actions:
                oldest = min(
                    self._pending_actions,
                    key=lambda key: self._pending_actions[key].created_at,
                )
                self._pending_actions.pop(oldest, None)
            self._pending_actions[session_id] = pending

    def _get_pending(self, session_id: str | None) -> _PendingAction | None:
        if not session_id:
            return None
        with self._pending_lock:
            pending = self._pending_actions.get(session_id)
            if pending and monotonic() - pending.created_at > 1800:
                self._pending_actions.pop(session_id, None)
                return None
            return pending

    def _handle_pending_action(
        self,
        mensagem: str,
        session_id: str | None,
    ) -> dict | None:
        from src.conhecimento.ferramentas import _selected_stages, limpar_resultados_ml
        from src.conhecimento.intencoes_ferramentas import _quer_limpar
        from src.core.texto import normalizar_sem_acentos

        text = " ".join(normalizar_sem_acentos(mensagem).lower().split())
        cancellation = text in {"cancelar", "cancele", "nao", "não"} or text.startswith(
            ("cancelar ", "cancele ")
        )
        confirmation = text in {
            "confirmar",
            "confirmo",
            "sim",
            "pode excluir",
            "pode apagar",
            "confirmar exclusao",
        }
        pending = self._get_pending(session_id)

        if pending is None:
            if cancellation or confirmation:
                return self._local_action_response(
                    "Não há uma exclusão pendente nesta conversa."
                )
            if not _quer_limpar(mensagem):
                return None
            if not session_id:
                return self._local_action_response(
                    "Para excluir resultados com segurança, use uma conversa com `session_id` ativo."
                )
            stages = _selected_stages(mensagem)
            if not stages:
                self._set_pending(
                    session_id,
                    _PendingAction("delete_results", "select_stages", (), monotonic()),
                )
                return self._local_action_response(
                    "Quais resultados deseja excluir: **comparação**, **confiabilidade** ou ambos?"
                )
            self._set_pending(
                session_id,
                _PendingAction("delete_results", "confirm", stages, monotonic()),
            )
            result = limpar_resultados_ml(pergunta=mensagem, etapas=stages)
            return self._local_action_response(result["mensagem"], route="tool")

        if cancellation:
            with self._pending_lock:
                self._pending_actions.pop(session_id, None)
            return self._local_action_response("A exclusão pendente foi cancelada.")

        stages = _selected_stages(mensagem)
        if not stages and any(term in text.split() for term in ("ambos", "ambas")):
            stages = ("comparacao", "confiabilidade")

        if pending.phase == "select_stages":
            if not stages:
                return self._local_action_response(
                    "Ainda preciso saber quais resultados: **comparação**, **confiabilidade** ou ambos."
                )
            self._set_pending(
                session_id,
                _PendingAction("delete_results", "confirm", stages, monotonic()),
            )
            result = limpar_resultados_ml(pergunta=mensagem, etapas=stages)
            return self._local_action_response(result["mensagem"], route="tool")

        if stages and not confirmation:
            self._set_pending(
                session_id,
                _PendingAction("delete_results", "confirm", stages, monotonic()),
            )
            result = limpar_resultados_ml(pergunta=mensagem, etapas=stages)
            return self._local_action_response(result["mensagem"], route="tool")

        if not confirmation:
            return self._local_action_response(
                "A exclusão ainda não foi executada. Responda **confirmar** ou **cancelar**."
            )

        with self._pending_lock:
            current = self._pending_actions.get(session_id)
            if current != pending:
                return self._local_action_response(
                    "A ação pendente mudou; revise a seleção antes de confirmar."
                )
            self._pending_actions.pop(session_id, None)
        result = limpar_resultados_ml(
            pergunta=mensagem,
            etapas=pending.stages,
            confirmado=True,
        )
        return self._local_action_response(result["mensagem"], route="tool")

    def _handle_library_action(
        self,
        mensagem: str,
        anexos: list[tuple[str, bytes]],
        *,
        library_write_allowed: bool,
        library_write_reason: str | None,
    ) -> dict | None:
        from src.conhecimento.intencoes_ferramentas import (
            _quer_adicionar_anexo_biblioteca,
        )

        if not _quer_adicionar_anexo_biblioteca(
            mensagem,
            tem_anexos=bool(anexos),
        ):
            return None
        from src.conhecimento.ferramentas import adicionar_anexo_biblioteca

        result = adicionar_anexo_biblioteca(
            pergunta=mensagem,
            anexos=anexos,
            library_service=self._library_service,
            library_write_allowed=library_write_allowed,
            library_write_reason=library_write_reason,
        )
        response = self._local_action_response(result["mensagem"], route="tool")
        response["library_jobs"] = result.get("jobs", [])
        return response

    def status(self) -> dict:
        with self._lock:
            route = self._route_status(self._components)
            return {
                "state": self._state,
                "provider": route.get("provider") or "automatic",
                "model": route.get("model"),
                "routing": route,
                "engine": AGENT_ENGINE,
                "retrieval": ["semantic", "bm25", "sessions", "obsidian"],
                "evidence_audit": True,
                "detail": self._error,
                "background_warmup": True,
            }

    @staticmethod
    def _route_status(componentes: _Componentes | None) -> dict:
        llm = getattr(componentes, "llm", None)
        getter = getattr(llm, "route_status", None)
        if callable(getter):
            try:
                status = getter()
                if isinstance(status, dict):
                    return status
            except Exception:
                _logger.warning("status seguro do Router indisponível")
        return {
            "provider": None,
            "model": None,
            "task_type": None,
            "route_reason": "not_invoked",
            "fallback_used": False,
            "validation_used": False,
        }

    def initialize(self) -> dict:
        """Aquece o mesmo runtime usado pela conversa e devolve seu estado."""
        if self._answerer is None:
            self._initialize()
        return self.status()

    def warm_background(self) -> threading.Thread | None:
        """Inicia o runtime pesado sem atrasar a primeira resposta HTTP."""
        if self._answerer is not None:
            return None
        with self._lock:
            if self._components is not None:
                return None
            if self._warm_thread is not None and self._warm_thread.is_alive():
                return self._warm_thread
            self._state = "iniciando"

            def warm() -> None:
                try:
                    self._initialize()
                except AgenteIndisponivel:
                    return

            self._warm_thread = threading.Thread(
                target=warm,
                name="aliado-runtime-warmup",
                daemon=True,
            )
            self._warm_thread.start()
            return self._warm_thread

    def reset(self) -> None:
        with self._lock:
            self._components = None
            self._error = None
            self._state = "pronto" if self._answerer is not None else "iniciando"

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
                self._state = "iniciando"
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
                    self._state = "pronto"
                    self._error = None
                return componentes
            except Exception as exc:
                erro = mascarar_segredos(str(exc))
                with self._lock:
                    self._state = "degradado"
                    self._error = erro
                raise AgenteIndisponivel(erro) from exc

    @staticmethod
    def _contexto(historico: list[dict[str, str]]) -> str:
        nomes = {"user": nome_pesquisador(), "assistant": "ALIAdo"}
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
                (Path(RAIZ_PROJETO) / "resultados").resolve()
            )
        except (OSError, ValueError):
            return None
        partes = relativo.parts
        if not partes or partes[0] not in {"comparacao", "confiabilidade"}:
            return None
        return {
            "url": "/artifacts/" + (
                "comparison/" if partes[0] == "comparacao" else "reliability/"
            ) + Path(*partes[1:]).as_posix(),
            "caption": str(item.get("caption") or item.get("legenda") or path.stem),
        }

    def _answer_real(
        self,
        mensagem: str,
        historico: list[dict[str, str]],
        anexos_bytes: list[tuple[str, bytes]],
        on_chunk: Callable[[str], None] | None = None,
        library_write_allowed: bool = False,
        library_write_reason: str | None = None,
    ) -> dict:
        componentes = self._initialize()
        contexto_cientifico = scientific_context_for(mensagem)
        try:
            from src.conhecimento.ferramentas import (
                decidir_acao,
                processar_com_ferramentas,
            )

            decisao = (
                decidir_acao(mensagem, componentes.llm, tem_anexos=True)
                if anexos_bytes
                else decidir_acao(mensagem, componentes.llm)
            )
            if decisao.get("esclarecimento"):
                return self._local_action_response(str(decisao["esclarecimento"]))
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
                    anexos=anexos_bytes,
                    library_service=self._library_service,
                    library_write_allowed=library_write_allowed,
                    library_write_reason=library_write_reason,
                )
                imagens = []
                if saida.get("resultado"):
                    for item in saida["resultado"].get("imagens", []):
                        publica = self._imagem_publica(item)
                        if publica:
                            imagens.append(publica)
                resposta_ferramenta = saida.get("resposta") or "Sem resposta."
                if decisao.get("ferramenta") in {
                    "consultar_resultados",
                    "consultar_comparacao_autoencoders",
                    "executar_comparacao_autoencoders",
                    "gerar_confiabilidade",
                    "executar_pipeline_cientifico",
                } and "Verificação de citações" not in resposta_ferramenta:
                    from src.core.citacao_guarda import alerta_citacao_infundada

                    aviso = alerta_citacao_infundada(resposta_ferramenta, {})
                    if aviso:
                        resposta_ferramenta = aviso.strip() + "\n\n" + resposta_ferramenta
                return {
                    "answer": resposta_ferramenta,
                    "images": imagens,
                    "route": "tool",
                    "scientific_contract": "canonical" if contexto_cientifico else None,
                    "library_jobs": (saida.get("resultado") or {}).get("jobs", []),
                }
        except AgenteIndisponivel:
            raise
        except Exception as exc:
            from src.conhecimento.intencoes_ferramentas import (
                _parece_pedido_de_ferramenta,
            )

            _logger.warning("roteamento por ferramenta indisponivel: %s", exc)
            if _parece_pedido_de_ferramenta(mensagem):
                return self._local_action_response(
                    "Não consegui concluir a operação. Especifique se deseja consultar, "
                    "executar, recalcular, importar ou excluir."
                )

        anexos = []
        if anexos_bytes:
            from src.conhecimento.leitor_anexos import ler_anexos

            anexos = ler_anexos(anexos_bytes)

        from src.conhecimento.agente import perguntar

        resposta = perguntar(
            pergunta=mensagem,
            perfil=componentes.perfil,
            modelo_embeddings=componentes.modelo_embeddings,
            colecao=componentes.literatura,
            llm=componentes.llm,
            historico=historico,
            streaming=on_chunk is not None,
            colecao_sessoes=componentes.sessoes,
            nome_provedor="Router LLM",
            anexos=anexos,
            colecao_obsidian=componentes.obsidian,
            indice_lexical=componentes.indice_lexical,
            auditor=componentes.auditor,
            contexto_autoritativo=contexto_cientifico,
            on_chunk=on_chunk,
        )
        return {
            "answer": resposta,
            "images": [],
            "route": "rag",
            "inference": self._route_status(componentes),
            "scientific_contract": "canonical" if contexto_cientifico else None,
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
            name="aliado-maintenance",
            daemon=True,
        ).start()

    def answer(
        self,
        message: str,
        history=None,
        attachments=None,
        session_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        library_write_allowed: bool = False,
        library_write_reason: str | None = None,
    ) -> dict:
        mensagem = str(message or "").strip()
        if not mensagem:
            raise ValueError("A mensagem não pode estar vazia")
        if len(mensagem) > MAX_MESSAGE_CHARS:
            raise ValueError(f"A mensagem excede {MAX_MESSAGE_CHARS} caracteres")
        historico = _historico_normalizado(history)
        anexos = _validar_anexos(attachments)
        session_id = self._session_journal.validar_session_id(session_id)

        pending_response = self._handle_pending_action(mensagem, session_id)
        if pending_response is not None:
            return pending_response

        library_response = self._handle_library_action(
            mensagem,
            anexos,
            library_write_allowed=library_write_allowed,
            library_write_reason=library_write_reason,
        )
        if library_response is not None:
            return library_response

        if self._answerer is None and not anexos:
            from src.conhecimento.agente_interacao import resposta_interacao_simples

            resposta_simples = resposta_interacao_simples(mensagem, historico)
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
                resposta = self._answer_real(
                    mensagem,
                    historico,
                    anexos,
                    on_chunk=on_chunk,
                    library_write_allowed=library_write_allowed,
                    library_write_reason=library_write_reason,
                )
        except (ValueError, AgenteIndisponivel):
            raise
        except Exception as exc:
            erro = mascarar_segredos(str(exc))
            with self._lock:
                self._error = erro
                self._state = "degradado"
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
            self._state = "pronto"
            self._error = None
        return resposta
