"use strict";

const API = {
  conversations: "/api/conversations",
  library: "/api/library",
  render: "/api/render",
};

const STORAGE_KEY = "aliado:sessions:canonical";
const THEME_KEY = "aliado:theme";
const MAX_LOCAL_SESSIONS = 30;
let fallbackSessionSequence = 0;

const state = {
  currentView: "chat",
  sessions: [],
  currentSessionId: null,
  historyMode: "active",
  historyQuery: "",
  pendingDeleteId: null,
  files: [],
  attachmentPolicy: "conversation",
  controller: null,
  identity: {
    displayName: "Rodolfo",
    greeting: greetingForHour(new Date().getHours()),
    timezone: "America/Sao_Paulo",
  },
  libraryData: null,
  libraryFilters: { query: "", category: "", language: "" },
  libraryJobs: new Map(),
  toastTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function greetingForHour(hour) {
  if (hour >= 5 && hour < 12) return "Bom dia";
  if (hour >= 12 && hour < 18) return "Boa tarde";
  return "Boa noite";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeSearch(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLocaleLowerCase();
}

function icons() {
  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "aria-hidden": "true" } });
  }
}

function typesetMath(container) {
  if (!container || typeof window.renderMathInElement !== "function") return;
  try {
    window.renderMathInElement(container, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: String.raw`\(`, right: String.raw`\)`, display: false },
        { left: String.raw`\[`, right: String.raw`\]`, display: true },
        { left: "$", right: "$", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      throwOnError: false,
      strict: "warn",
    });
  } catch (_error) {
    // O Markdown continua legível quando uma expressão isolada é inválida.
  }
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("is-visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => node.classList.remove("is-visible"), 2800);
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { Accept: "application/json", ...options.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail || `Falha HTTP ${response.status}`);
    error.code = payload.error;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function sessionId() {
  if (window.crypto?.randomUUID) {
    return `sessao_${crypto.randomUUID().replaceAll("-", "")}`;
  }
  fallbackSessionSequence += 1;
  return `sessao_${Date.now()}_${fallbackSessionSequence}`;
}

function normalizeSession(session) {
  return {
    id: String(session.id || sessionId()),
    title: String(session.title || "Novo chat"),
    updatedAt: Number(session.updatedAt || Date.parse(session.updated_at || "") || Date.now()),
    status: session.status || "active",
    messages: Array.isArray(session.messages) ? session.messages : [],
    messagesLoaded: session.messagesLoaded !== false,
    serverBacked: Boolean(session.serverBacked),
  };
}

function loadSessions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    state.sessions = Array.isArray(parsed)
      ? parsed.slice(0, MAX_LOCAL_SESSIONS).map(normalizeSession)
      : [];
  } catch (error) {
    console.warn("Histórico local inválido; iniciando um novo chat.", error);
    state.sessions = [];
  }
  if (state.sessions.length) {
    state.currentSessionId = state.sessions[0].id;
  } else {
    createSession(false);
  }
}

function currentSession() {
  return state.sessions.find((session) => session.id === state.currentSessionId);
}

function saveSessions() {
  state.sessions.sort((a, b) => b.updatedAt - a.updatedAt);
  const localSnapshot = state.sessions
    .filter((session) => session.status !== "deleted" && session.messagesLoaded !== false)
    .slice(0, MAX_LOCAL_SESSIONS);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(localSnapshot));
  renderHistory();
  updateConversationChrome();
}

function createSession(render = true) {
  const session = normalizeSession({
    id: sessionId(),
    title: "Novo chat",
    updatedAt: Date.now(),
    status: "active",
    messages: [],
  });
  state.sessions.unshift(session);
  state.currentSessionId = session.id;
  state.historyMode = "active";
  saveSessions();
  if (render) {
    activateView("chat");
    renderConversation();
    $("#chat-input").focus();
  }
  return session;
}

function historyGroup(timestamp) {
  const now = new Date();
  const date = new Date(timestamp);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const itemDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const days = Math.floor((today - itemDay) / 86400000);
  if (days <= 0) return "Hoje";
  if (days === 1) return "Ontem";
  if (days < 7) return "Últimos 7 dias";
  if (days < 30) return "Últimos 30 dias";
  return "Anteriores";
}

function historyActionButton(action, sessionIdValue, label, icon, danger = false) {
  return `<button type="button" class="${danger ? "is-danger" : ""}" data-conversation-action="${action}" data-session-id="${escapeHtml(sessionIdValue)}"><i data-lucide="${icon}"></i><span>${label}</span></button>`;
}

function renderHistory() {
  const list = $("#history-list");
  const query = normalizeSearch(state.historyQuery);
  const sessions = state.sessions
    .filter((session) => (session.status || "active") === state.historyMode)
    .filter((session) => !query || normalizeSearch(session.title).includes(query))
    .sort((a, b) => b.updatedAt - a.updatedAt);

  $$('[data-history-mode]').forEach((button) => {
    const active = button.dataset.historyMode === state.historyMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });

  if (!sessions.length) {
    const label = state.historyMode === "archived"
      ? "Nenhum chat arquivado"
      : (query ? "Nenhum chat encontrado" : "Nenhum chat recente");
    list.innerHTML = `<div class="history-empty">${label}</div>`;
    return;
  }

  const groups = new Map();
  sessions.forEach((session) => {
    const label = historyGroup(session.updatedAt);
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(session);
  });

  list.innerHTML = [...groups.entries()].map(([label, items]) => `
    <section class="history-group">
      <span class="history-group-label">${label}</span>
      ${items.map((session) => {
        const archived = session.status === "archived";
        return `<div class="history-entry ${session.id === state.currentSessionId ? "is-current" : ""}">
          <button class="history-item" type="button" data-select-session="${escapeHtml(session.id)}" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</button>
          <details class="conversation-menu">
            <summary aria-label="Ações de ${escapeHtml(session.title)}" title="Mais ações"><i data-lucide="ellipsis"></i></summary>
            <div class="conversation-menu-popover">
              ${historyActionButton("rename", session.id, "Renomear", "pencil")}
              ${historyActionButton(archived ? "restore" : "archive", session.id, archived ? "Restaurar" : "Arquivar", archived ? "archive-restore" : "archive")}
              ${historyActionButton("delete", session.id, "Excluir", "trash-2", true)}
            </div>
          </details>
        </div>`;
      }).join("")}
    </section>`).join("");
  icons();
}

async function syncConversations() {
  const results = await Promise.allSettled([
    apiJson(`${API.conversations}?status=active`),
    apiJson(`${API.conversations}?status=archived`),
  ]);
  let changed = false;
  results.forEach((result) => {
    if (result.status !== "fulfilled") return;
    result.value.conversations.forEach((item) => {
      let session = state.sessions.find((candidate) => candidate.id === item.id);
      const updatedAt = Date.parse(item.updated_at || "") || Date.now();
      if (session) {
        session.title = item.title || session.title;
        session.status = item.status || session.status;
        session.updatedAt = Math.max(session.updatedAt || 0, updatedAt);
        session.serverBacked = true;
      } else {
        session = normalizeSession({
          ...item,
          updatedAt,
          messages: [],
          messagesLoaded: false,
          serverBacked: true,
        });
        state.sessions.push(session);
      }
      changed = true;
    });
  });
  if (changed) {
    saveSessions();
  } else {
    renderHistory();
  }
}

async function loadConversation(session) {
  if (!session || session.messagesLoaded !== false) return session;
  const payload = await apiJson(`${API.conversations}/${encodeURIComponent(session.id)}`);
  session.messages = payload.conversation.messages || [];
  session.title = payload.conversation.title || session.title;
  session.status = payload.conversation.status || session.status;
  session.messagesLoaded = true;
  session.serverBacked = true;
  saveSessions();
  return session;
}

async function selectConversation(sessionIdValue) {
  const session = state.sessions.find((candidate) => candidate.id === sessionIdValue);
  if (!session) return;
  state.currentSessionId = session.id;
  activateView("chat");
  try {
    await loadConversation(session);
  } catch (error) {
    toast(error.message);
  }
  renderConversation();
}

function updateConversationChrome() {
  const session = currentSession();
  $("#conversation-title").textContent = session?.title || "Novo chat";
  $("#conversation-actions").hidden = state.currentView !== "chat";
  $("#library-actions").hidden = state.currentView !== "library";
  const archived = session?.status === "archived";
  $("#archived-notice").hidden = !archived || state.currentView !== "chat";
  $("#chat-input").disabled = archived;
  $("#chat-files").disabled = archived;
  $("#send-button").disabled = archived;
  $("#archive-chat").hidden = archived || !session?.messages.length;
  $("#rename-chat").disabled = !session;
  $("#export-chat").disabled = !session?.messages.length;
  $("#delete-chat").disabled = !session;
}

function activateView(view, updateHash = true) {
  const normalized = view === "references" ? "library" : view;
  state.currentView = normalized === "library" ? "library" : "chat";
  $$('[data-view-panel]').forEach((panel) => {
    const active = panel.dataset.viewPanel === state.currentView;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  $$('[data-view]').forEach((button) => {
    button.classList.toggle("is-active", button.dataset.view === state.currentView);
  });
  if (updateHash) {
    const hash = state.currentView === "library" ? "#references" : "#chat";
    if (location.hash !== hash) history.replaceState(null, "", hash);
  }
  if (state.currentView === "library") void loadLibrary();
  updateConversationChrome();
  closeSidebar();
}

function updateWelcomeIdentity() {
  const title = $("#welcome-title");
  if (title) title.textContent = `${state.identity.greeting}, ${state.identity.displayName}.`;
  $("#profile-name").textContent = state.identity.displayName;
}

function renderUserMessage(message) {
  const article = document.createElement("article");
  article.className = "message message-user";
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = message.content;
  article.appendChild(bubble);
  return article;
}

function safeAssetUrl(value) {
  const url = String(value || "");
  return url.startsWith("/artifacts/") ? url : "";
}

function appendCitations(container, citations) {
  if (!Array.isArray(citations) || !citations.length) return;
  const list = document.createElement("div");
  list.className = "citation-list";
  citations.forEach((citation) => {
    const href = String(citation.url || "");
    if (!href.startsWith("http://") && !href.startsWith("https://") && !href.startsWith("/")) return;
    const link = document.createElement("a");
    link.href = href;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = citation.label || href;
    list.appendChild(link);
  });
  if (list.childElementCount) container.appendChild(list);
}

function appendResponseAssets(container, images) {
  if (!Array.isArray(images) || !images.length) return;
  const valid = images
    .map((item) => ({ ...item, url: safeAssetUrl(item.url) }))
    .filter((item) => item.url);
  if (!valid.length) return;
  const gallery = document.createElement("div");
  gallery.className = "response-assets";
  valid.forEach((item) => {
    const figure = document.createElement("figure");
    figure.className = "response-asset";
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.openMedia = item.url;
    button.dataset.mediaTitle = item.caption || "Figura";
    button.setAttribute("aria-label", `Ampliar ${item.caption || "figura"}`);
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = item.caption || "Figura produzida pelo ALIAdo";
    image.loading = "lazy";
    image.decoding = "async";
    const caption = document.createElement("figcaption");
    caption.textContent = item.caption || "Figura";
    button.appendChild(image);
    figure.append(button, caption);
    gallery.appendChild(figure);
  });
  container.appendChild(gallery);
}

function renderAssistantMessage(message, index) {
  const article = document.createElement("article");
  article.className = "message message-assistant";
  article.innerHTML = `
    <div class="assistant-layout">
      <div class="assistant-avatar" aria-hidden="true">A</div>
      <div>
        <div class="message-author">ALIAdo</div>
        <div class="message-content" data-message-index="${index}"></div>
        <div class="message-meta">
          <span>${message.response_ms ? `${Math.round(message.response_ms)} ms` : "Resposta do ALIAdo"}</span>
          <div class="message-actions">
            <button class="message-action" type="button" data-copy-index="${index}" aria-label="Copiar resposta" title="Copiar"><i data-lucide="copy"></i></button>
            <button class="message-action" type="button" data-retry-index="${index}" aria-label="Tentar novamente" title="Tentar novamente"><i data-lucide="refresh-cw"></i></button>
          </div>
        </div>
      </div>
    </div>`;
  const content = $(".message-content", article);
  content.textContent = message.content;
  appendCitations(content, message.citations);
  appendResponseAssets(content, message.images);
  return article;
}

async function hydrateAssistantMessages(session) {
  if (!session) return;
  const sessionIdAtStart = session.id;
  const messages = session.messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => message.role === "assistant")
    .map(({ message, index }) => ({ id: String(index), content: message.content }));
  if (!messages.length) return;
  try {
    const payload = await apiJson(API.render, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    if (state.currentSessionId !== sessionIdAtStart) return;
    payload.messages.forEach((rendered) => {
      const index = Number(rendered.id);
      const node = $(`.message-content[data-message-index="${rendered.id}"]`);
      const message = session.messages[index];
      if (!node || !message) return;
      node.innerHTML = rendered.html;
      appendCitations(node, message.citations);
      appendResponseAssets(node, message.images);
      typesetMath(node);
    });
  } catch (error) {
    console.warn("Não foi possível restaurar a renderização enriquecida.", error);
  }
}

function buildWelcomeState() {
  const welcome = document.createElement("section");
  welcome.className = "welcome-state";
  welcome.id = "welcome-state";
  welcome.innerHTML = `
    <div class="welcome-mark" aria-hidden="true">A</div>
    <h1 id="welcome-title">${escapeHtml(state.identity.greeting)}, ${escapeHtml(state.identity.displayName)}.</h1>
    <p>Converse, pesquise suas referências, analise arquivos ou peça a execução do pipeline.</p>`;
  return welcome;
}

function renderConversation() {
  const conversation = $("#conversation");
  const session = currentSession();
  conversation.innerHTML = "";
  if (!session?.messages.length) {
    conversation.appendChild(buildWelcomeState());
  } else {
    session.messages.forEach((message, index) => {
      conversation.appendChild(
        message.role === "user"
          ? renderUserMessage(message)
          : renderAssistantMessage(message, index),
      );
    });
  }
  updateWelcomeIdentity();
  updateConversationChrome();
  icons();
  void hydrateAssistantMessages(session);
  scrollConversation(false);
}

function scrollConversation(smooth = true) {
  const conversation = $("#conversation");
  conversation.scrollTo({
    top: conversation.scrollHeight,
    behavior: smooth && !matchMedia("(prefers-reduced-motion: reduce)").matches ? "smooth" : "auto",
  });
}

function setStreaming(active) {
  const button = $("#send-button");
  button.classList.toggle("is-cancel", active);
  button.type = active ? "button" : "submit";
  button.setAttribute("aria-label", active ? "Cancelar resposta" : "Enviar mensagem");
  button.title = active ? "Cancelar resposta" : "Enviar mensagem";
  button.innerHTML = active
    ? '<i data-lucide="square" aria-hidden="true"></i>'
    : '<i data-lucide="arrow-up" aria-hidden="true"></i>';
  icons();
}

function updateStreamStatus(message = "") {
  $("#stream-status").textContent = message;
}

function addPendingAssistant() {
  const article = document.createElement("article");
  article.className = "message message-assistant";
  article.innerHTML = `
    <div class="assistant-layout">
      <div class="assistant-avatar" aria-hidden="true">A</div>
      <div>
        <div class="message-author">ALIAdo</div>
        <div class="message-content pending-response">
          <div class="response-loader" role="status" aria-label="ALIAdo está preparando a resposta"><span></span><span></span><span></span><em>Organizando contexto</em></div>
        </div>
        <div class="message-meta"><span>Preparando resposta</span></div>
      </div>
    </div>`;
  $("#conversation").appendChild(article);
  return article;
}

async function consumeSse(response, onEvent) {
  if (!response.ok) {
    let detail = `Falha HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_error) {
      // Mantém a mensagem HTTP.
    }
    throw new Error(detail);
  }
  if (!response.body) throw new Error("O navegador não disponibilizou o fluxo de resposta.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replaceAll("\r\n", "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const dataLines = [];
      block.split("\n").forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      });
      if (dataLines.length) onEvent(event, JSON.parse(dataLines.join("\n")));
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
}

function previousHistory(session) {
  return session.messages.slice(-16).map((item) => ({
    role: item.role,
    content: item.content,
  }));
}

const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function queuePdfFile(file, metadata = {}) {
  const form = new FormData();
  form.append("file", file, file.name);
  Object.entries(metadata).forEach(([key, value]) => {
    if (value !== "" && value != null) form.append(key, value);
  });
  const payload = await apiJson(API.library, { method: "POST", body: form });
  void trackLibraryJob(payload.job);
  return payload.job;
}

async function queueAttachedPdfs() {
  const pdfs = state.files.filter((file) => file.name.toLocaleLowerCase().endsWith(".pdf"));
  if (!pdfs.length || state.attachmentPolicy !== "library") return;
  updateStreamStatus(`Adicionando ${pdfs.length} PDF${pdfs.length > 1 ? "s" : ""} às referências`);
  for (const file of pdfs) {
    try {
      await queuePdfFile(file);
    } catch (error) {
      toast(error.code === "duplicate_pdf"
        ? `${file.name} já está nas referências.`
        : `Não foi possível indexar ${file.name}: ${error.message}`);
    }
  }
}

function updateRouteLabel(payload) {
  const route = payload?.agent?.routing || payload?.inference || {};
  const rawProvider = route.provider || payload?.agent?.provider;
  const provider = rawProvider === "automatic" ? "" : rawProvider;
  const model = route.model || payload?.agent?.model;
  const label = [provider, model].filter(Boolean).join(" · ") || "Seleção automática";
  $("#model-route").lastChild.textContent = label;
  $("#model-route").classList.toggle("is-ready", Boolean(provider || model));
}

async function sendMessage(rawMessage) {
  if (state.controller) return;
  const input = $("#chat-input");
  const message = String(rawMessage ?? input.value).trim();
  if (!message) return;
  let session = currentSession();
  if (!session) session = createSession(false);
  if (session.status === "archived") {
    toast("Restaure o chat para continuar.");
    return;
  }

  const historyItems = previousHistory(session);
  session.messages.push({ role: "user", content: message });
  if (session.title === "Novo chat" || session.title === "Nova conversa") {
    session.title = message.replace(/\s+/g, " ").slice(0, 58);
  }
  session.updatedAt = Date.now();
  saveSessions();

  const welcome = $("#welcome-state");
  if (welcome) welcome.remove();
  $("#conversation").appendChild(renderUserMessage({ content: message }));
  const pending = addPendingAssistant();
  const pendingContent = $(".message-content", pending);
  const pendingMeta = $(".message-meta span", pending);
  input.value = "";
  resizeInput();
  scrollConversation();

  const controller = new AbortController();
  state.controller = controller;
  setStreaming(true);
  updateStreamStatus("Preparando contexto");
  await queueAttachedPdfs();

  let body;
  let headers = {};
  if (state.files.length) {
    const form = new FormData();
    form.append("message", message);
    form.append("history", JSON.stringify(historyItems));
    form.append("session_id", session.id);
    state.files.forEach((file) => form.append("files", file, file.name));
    body = form;
  } else {
    headers = { "Content-Type": "application/json" };
    body = JSON.stringify({ message, history: historyItems, session_id: session.id });
  }

  let complete = "";
  let completed = false;
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers,
      body,
      signal: controller.signal,
    });
    await consumeSse(response, (event, payload) => {
      if (event === "status") {
        const phase = payload.message || "Consultando contexto";
        updateStreamStatus(phase);
        pendingMeta.textContent = phase;
        const loaderLabel = $(".response-loader em", pendingContent);
        if (loaderLabel) loaderLabel.textContent = phase;
      } else if (event === "delta") {
        complete += payload.text || "";
        pendingContent.classList.remove("pending-response");
        pendingContent.classList.add("streaming");
        pendingContent.textContent = complete;
        scrollConversation(false);
      } else if (event === "done") {
        completed = true;
        complete = payload.answer || complete;
        pendingContent.classList.remove("streaming", "pending-response");
        pendingContent.innerHTML = payload.answer_html || escapeHtml(complete);
        appendCitations(pendingContent, payload.citations);
        appendResponseAssets(pendingContent, payload.images);
        typesetMath(pendingContent);
        pendingMeta.textContent = `${Math.round(payload.response_ms || 0)} ms · ${payload.route || "agente"}`;
        const index = session.messages.length;
        const actions = document.createElement("div");
        actions.className = "message-actions";
        actions.innerHTML = `
          <button class="message-action" type="button" data-copy-index="${index}" aria-label="Copiar resposta" title="Copiar"><i data-lucide="copy"></i></button>
          <button class="message-action" type="button" data-retry-index="${index}" aria-label="Tentar novamente" title="Tentar novamente"><i data-lucide="refresh-cw"></i></button>`;
        $(".message-meta", pending).appendChild(actions);
        session.messages.push({
          role: "assistant",
          content: complete,
          citations: payload.citations || [],
          images: payload.images || [],
          response_ms: payload.response_ms,
        });
        session.updatedAt = Date.now();
        saveSessions();
        updateRouteLabel(payload);
        updateStreamStatus("");
        icons();
        setTimeout(() => void syncConversations(), 1000);
      } else if (event === "error") {
        throw new Error(payload.detail || "O ALIAdo não conseguiu responder.");
      }
    });
    if (!completed && !complete) throw new Error("A resposta terminou sem conteúdo.");
  } catch (error) {
    pendingContent.classList.remove("streaming", "pending-response");
    if (error.name === "AbortError") {
      pendingContent.textContent = complete || "Resposta cancelada.";
      pendingMeta.textContent = "Cancelada";
    } else {
      pendingContent.textContent = `Não foi possível concluir: ${error.message}`;
      pendingMeta.textContent = "Falha na resposta";
      toast("A resposta não pôde ser concluída.");
    }
    updateStreamStatus("");
  } finally {
    state.controller = null;
    state.files = [];
    state.attachmentPolicy = "conversation";
    const defaultPolicy = $('input[name="attachment-policy"][value="conversation"]');
    if (defaultPolicy) defaultPolicy.checked = true;
    renderAttachments();
    setStreaming(false);
    input.focus();
  }
}

function renderAttachments() {
  const list = $("#attachment-list");
  const policy = $("#attachment-policy");
  list.innerHTML = "";
  state.files.forEach((file, index) => {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    chip.innerHTML = `<i data-lucide="file" aria-hidden="true"></i><span></span><button type="button" data-remove-file="${index}" aria-label="Remover anexo" title="Remover"><i data-lucide="x"></i></button>`;
    $("span", chip).textContent = file.name;
    list.appendChild(chip);
  });
  policy.hidden = !state.files.some((file) => file.name.toLocaleLowerCase().endsWith(".pdf"));
  icons();
}

function resizeInput() {
  const input = $("#chat-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function exportConversation() {
  const session = currentSession();
  if (!session?.messages.length) {
    toast("Este chat ainda está vazio.");
    return;
  }
  const content = [
    `# ${session.title}`,
    "",
    ...session.messages.flatMap((message) => [
      `## ${message.role === "user" ? state.identity.displayName : "ALIAdo"}`,
      "",
      message.content,
      "",
    ]),
  ].join("\n");
  const url = URL.createObjectURL(new Blob([content], { type: "text/markdown;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${session.id}_chat.md`;
  link.click();
  URL.revokeObjectURL(url);
}

function retryMessage(index) {
  const session = currentSession();
  if (!session) return;
  for (let cursor = Number(index) - 1; cursor >= 0; cursor -= 1) {
    if (session.messages[cursor]?.role === "user") {
      void sendMessage(session.messages[cursor].content);
      return;
    }
  }
}

async function copyMessage(index) {
  const message = currentSession()?.messages[Number(index)];
  if (!message) return;
  await navigator.clipboard.writeText(message.content);
  toast("Resposta copiada.");
}

function openRenameDialog(sessionIdValue) {
  const session = state.sessions.find((item) => item.id === sessionIdValue);
  if (!session) return;
  state.currentSessionId = session.id;
  const form = $("#conversation-rename-form");
  form.elements.title.value = session.title;
  $("#conversation-rename-dialog").showModal();
  requestAnimationFrame(() => form.elements.title.select());
}

function openDeleteDialog(sessionIdValue) {
  const session = state.sessions.find((item) => item.id === sessionIdValue);
  if (!session) return;
  state.pendingDeleteId = session.id;
  $("#conversation-delete-dialog").showModal();
}

async function performConversationAction(action, sessionIdValue) {
  const session = state.sessions.find((item) => item.id === sessionIdValue);
  if (!session) return;
  if (action === "rename") {
    openRenameDialog(session.id);
    return;
  }
  if (action === "delete") {
    openDeleteDialog(session.id);
    return;
  }

  const status = action === "archive" ? "archived" : "active";
  if (session.serverBacked) {
    const endpoint = `${API.conversations}/${encodeURIComponent(session.id)}/${action}`;
    try {
      await apiJson(endpoint, { method: "POST" });
    } catch (error) {
      if (error.code !== "conversation_not_found") throw error;
    }
  }
  session.status = status;
  session.updatedAt = Date.now();
  state.historyMode = status;
  saveSessions();
  renderConversation();
  toast(status === "archived" ? "Chat arquivado." : "Chat restaurado.");
}

async function confirmDeleteConversation() {
  const sessionIdValue = state.pendingDeleteId;
  const session = state.sessions.find((item) => item.id === sessionIdValue);
  $("#conversation-delete-dialog").close();
  state.pendingDeleteId = null;
  if (!session) return;
  if (session.serverBacked) {
    try {
      await apiJson(`${API.conversations}/${encodeURIComponent(session.id)}`, { method: "DELETE" });
    } catch (error) {
      if (error.code !== "conversation_not_found") {
        toast(error.message);
        return;
      }
    }
  }
  session.status = "deleted";
  if (state.currentSessionId === session.id) createSession(false);
  saveSessions();
  renderConversation();
  toast("Chat removido da interface.");
}

const CATEGORY_LABELS = {
  confiabilidade: "Confiabilidade",
  "inversores-pv": "Inversores PV",
  manutencao: "Manutenção",
  "ml-preditivo": "ML preditivo",
  "sinais-eletricos": "Sinais elétricos",
};

const LANGUAGE_LABELS = {
  pt: "Português",
  en: "Inglês",
  es: "Espanhol",
  fr: "Francês",
  desconhecido: "Não identificado",
};

const INDEX_LABELS = {
  indexed: "Indexada",
  indexing: "Indexando",
  metadata_stale: "Metadados pendentes",
  indexed_snapshot_stale: "Snapshot pendente",
  index_failed: "Falha no índice",
  not_indexed: "Não indexada",
};

function downloadRows(items) {
  return `<div class="download-list">${items.map((item) => `
    <div class="download-row">
      <div><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.sha256 || "")}</small></div>
      <a class="download-link" href="${escapeHtml(item.url)}" download><i data-lucide="download"></i><span>Baixar</span></a>
    </div>`).join("")}</div>`;
}

function renderLibrary(data) {
  state.libraryData = data;
  $("#library-count").textContent = data.summary.documents;
  const writeNote = data.writable
    ? "Alterações ficam locais até revisão e commit."
    : (data.write_policy.reason || "Referências disponíveis somente para leitura.");
  return `
    <section class="library-summary" aria-label="Resumo das referências">
      <div class="library-stat"><span>Documentos</span><strong>${data.summary.documents}</strong><small>PDFs identificados por SHA-256</small></div>
      <div class="library-stat"><span>Trechos indexados</span><strong>${Number(data.summary.indexed_chunks).toLocaleString("pt-BR")}</strong><small>Disponíveis para busca híbrida</small></div>
      <div class="library-stat"><span>Alertas de metadados</span><strong>${data.summary.metadata_warnings}</strong><small>Sem alterar os PDFs originais</small></div>
    </section>
    <section class="library-control-band">
      <div class="library-toolbar">
        <label class="library-search"><span class="sr-only">Pesquisar referências</span><i data-lucide="search"></i><input id="library-query" type="search" value="${escapeHtml(state.libraryFilters.query)}" placeholder="Título, autor, ano ou arquivo"></label>
        <label><span class="sr-only">Categoria</span><select id="library-category"><option value="">Todas as categorias</option>${data.categories.map((value) => `<option value="${value}"${state.libraryFilters.category === value ? " selected" : ""}>${CATEGORY_LABELS[value] || escapeHtml(value)}</option>`).join("")}</select></label>
        <label><span class="sr-only">Idioma</span><select id="library-language"><option value="">Todos os idiomas</option>${data.languages.map((value) => `<option value="${value}"${state.libraryFilters.language === value ? " selected" : ""}>${LANGUAGE_LABELS[value] || escapeHtml(value)}</option>`).join("")}</select></label>
        <output id="library-result-count" aria-live="polite"></output>
      </div>
      <p class="library-policy"><i data-lucide="shield-check"></i><span>${escapeHtml(writeNote)}</span></p>
      <div class="library-jobs" id="library-jobs" aria-live="polite"></div>
      <div class="library-list" id="library-list"></div>
    </section>
    <details class="library-provenance">
      <summary>Proveniência e arquivos auditáveis</summary>
      <div class="provenance-body">
        <p>A Biblioteca registra hash, categoria, idioma e quantidade de trechos de cada PDF. O índice vetorial é reconstruível.</p>
        ${downloadRows([...(data.provenance?.manifests || []), ...(data.provenance?.reports || [])])}
      </div>
    </details>`;
}

function renderLibraryDocuments() {
  const container = $("#library-list");
  if (!container || !state.libraryData) return;
  const query = normalizeSearch(state.libraryFilters.query);
  const documents = state.libraryData.documents.filter((document) => {
    const searchable = normalizeSearch([
      document.title,
      document.authors.join(" "),
      document.year || "s.d.",
      document.file_name,
    ].join(" "));
    return (!query || searchable.includes(query))
      && (!state.libraryFilters.category || document.category === state.libraryFilters.category)
      && (!state.libraryFilters.language || document.language === state.libraryFilters.language);
  });
  $("#library-result-count").textContent = `${documents.length} de ${state.libraryData.documents.length}`;
  if (!documents.length) {
    container.innerHTML = '<div class="empty-library"><i data-lucide="search-x"></i><p>Nenhuma referência corresponde aos filtros.</p></div>';
    icons();
    return;
  }
  container.innerHTML = documents.map((document) => {
    const warnings = document.extraction_warnings?.length
      ? `<span class="metadata-warning" title="${escapeHtml(document.extraction_warnings.join("; "))}">Revisar extração</span>`
      : "";
    const writeActions = state.libraryData.writable ? `
      <button class="icon-button" type="button" data-edit-source="${document.source_id}" aria-label="Editar metadados" title="Editar"><i data-lucide="pencil"></i></button>
      <button class="icon-button" type="button" data-reindex-source="${document.source_id}" aria-label="Reindexar fonte" title="Reindexar"><i data-lucide="refresh-cw"></i></button>` : "";
    return `<article class="library-row" data-source-id="${document.source_id}">
      <div class="library-document-main">
        <h2>${escapeHtml(document.title)}</h2>
        <p>${escapeHtml(document.authors.join("; "))} · ${document.year || "s.d."}</p>
        <div class="library-document-meta"><span>${CATEGORY_LABELS[document.category] || escapeHtml(document.category)}</span><span>${LANGUAGE_LABELS[document.language] || escapeHtml(document.language)}</span><span>${Number(document.chunk_count).toLocaleString("pt-BR")} trechos</span><span data-index-status="${escapeHtml(document.index_status)}">${INDEX_LABELS[document.index_status] || escapeHtml(document.index_status)}</span>${warnings}</div>
      </div>
      <div class="library-row-actions">${writeActions}<a class="icon-button" href="${escapeHtml(document.url)}" target="_blank" rel="noreferrer" aria-label="Abrir PDF" title="Abrir PDF"><i data-lucide="external-link"></i></a></div>
    </article>`;
  }).join("");
  icons();
}

function renderLibraryJobs() {
  const container = $("#library-jobs");
  if (!container) return;
  const jobs = [...state.libraryJobs.values()].slice(-4).reverse();
  container.innerHTML = jobs.map((job) => {
    const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
    return `<div class="library-job" data-job-state="${escapeHtml(job.state)}"><div><strong>${job.kind === "add" ? "Nova referência" : "Reindexação"}</strong><span>${escapeHtml(job.message)}</span></div><div class="job-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress}"><span style="width:${progress}%"></span></div></div>`;
  }).join("");
}

async function trackLibraryJob(initialJob) {
  let job = initialJob;
  state.libraryJobs.set(job.job_id, job);
  renderLibraryJobs();
  while (["queued", "running"].includes(job.state)) {
    await wait(1200);
    try {
      const payload = await apiJson(`/api/library/jobs/${encodeURIComponent(job.job_id)}`);
      job = payload.job;
      state.libraryJobs.set(job.job_id, job);
      renderLibraryJobs();
    } catch (error) {
      toast(error.message);
      return;
    }
  }
  toast(job.state === "completed" ? "Referência indexada." : `Falha na indexação: ${job.message}`);
  state.libraryData = null;
  if (state.currentView === "library") await loadLibrary(true);
}

async function loadLibrary(force = false) {
  const content = $("#library-content");
  const loading = $('[data-loading="library"]');
  if (state.libraryData && !force) return;
  loading.hidden = false;
  content.innerHTML = "";
  try {
    const data = await apiJson(API.library);
    state.libraryData = data;
    content.innerHTML = renderLibrary(data);
    $("#library-add").hidden = !data.writable;
    renderLibraryDocuments();
    renderLibraryJobs();
    loading.hidden = true;
    icons();
  } catch (error) {
    loading.hidden = true;
    content.innerHTML = `<div class="error-state"><i data-lucide="circle-alert"></i><strong>Não foi possível carregar as referências.</strong><p>${escapeHtml(error.message)}</p><button class="secondary-button" type="button" data-reload-library><i data-lucide="refresh-cw"></i><span>Tentar novamente</span></button></div>`;
    icons();
  }
}

function openLibraryEdit(sourceId) {
  const document = state.libraryData?.documents.find((item) => item.source_id === sourceId);
  if (!document) return;
  const form = $("#library-edit-form");
  form.elements.source_id.value = document.source_id;
  form.elements.title.value = document.title;
  form.elements.authors.value = document.authors.join("; ");
  form.elements.year.value = document.year || "";
  form.elements.category.value = document.category;
  form.elements.language.value = document.language;
  $("#library-edit-dialog").showModal();
}

async function reindexLibrarySource(sourceId) {
  try {
    const payload = await apiJson(`/api/library/${encodeURIComponent(sourceId)}/reindex`, { method: "POST" });
    void trackLibraryJob(payload.job);
    toast("Reindexação iniciada.");
  } catch (error) {
    toast(error.message);
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
  $("#theme-toggle").innerHTML = theme === "dark"
    ? '<i data-lucide="sun"></i>'
    : '<i data-lucide="moon"></i>';
  icons();
}

async function pollStatus() {
  let delay = 4000;
  try {
    const payload = await apiJson("/api/status");
    if (payload.identity) {
      state.identity = {
        displayName: payload.identity.display_name || state.identity.displayName,
        greeting: payload.identity.greeting || state.identity.greeting,
        timezone: payload.identity.timezone || state.identity.timezone,
      };
      updateWelcomeIdentity();
    }
    const status = $("#runtime-status");
    status.dataset.state = payload.state;
    status.setAttribute("aria-label", payload.state);
    $("#runtime-status-text").textContent = {
      pronto: "Pronto para conversar",
      iniciando: "Aquecendo conhecimento",
      degradado: "Modo degradado",
    }[payload.state] || "Verificando";
    updateRouteLabel({ agent: payload.agent });
    delay = payload.state === "pronto" ? 45000 : 5000;
  } catch (error) {
    console.warn("Status do runtime indisponível.", error);
    $("#runtime-status").dataset.state = "degradado";
    $("#runtime-status-text").textContent = "Sem conexão";
  }
  setTimeout(pollStatus, document.hidden ? Math.max(delay, 60000) : delay);
}

function openMedia(url, title) {
  const safeUrl = safeAssetUrl(url);
  if (!safeUrl) return;
  $("#media-title").textContent = title || "Figura";
  $("#media-image").src = safeUrl;
  $("#media-image").alt = title || "Figura";
  $("#media-download").href = safeUrl;
  $("#media-dialog").showModal();
}

async function handleDocumentClick(event) {
  const view = event.target.closest("[data-view]");
  if (view) {
    activateView(view.dataset.view);
    return;
  }

  const action = event.target.closest("[data-conversation-action]");
  if (action) {
    event.preventDefault();
    await performConversationAction(action.dataset.conversationAction, action.dataset.sessionId);
    return;
  }

  const session = event.target.closest("[data-select-session]");
  if (session) {
    await selectConversation(session.dataset.selectSession);
    return;
  }

  const removeFile = event.target.closest("[data-remove-file]");
  if (removeFile) {
    state.files.splice(Number(removeFile.dataset.removeFile), 1);
    renderAttachments();
    return;
  }

  const copy = event.target.closest("[data-copy-index]");
  if (copy) {
    await copyMessage(copy.dataset.copyIndex);
    return;
  }

  const retry = event.target.closest("[data-retry-index]");
  if (retry) {
    retryMessage(retry.dataset.retryIndex);
    return;
  }

  const media = event.target.closest("[data-open-media]");
  if (media) {
    openMedia(media.dataset.openMedia, media.dataset.mediaTitle);
    return;
  }

  const editSource = event.target.closest("[data-edit-source]");
  if (editSource) {
    openLibraryEdit(editSource.dataset.editSource);
    return;
  }

  const reindexSource = event.target.closest("[data-reindex-source]");
  if (reindexSource) {
    await reindexLibrarySource(reindexSource.dataset.reindexSource);
    return;
  }

  if (event.target.closest("[data-reload-library]")) {
    await loadLibrary(true);
    return;
  }

  const closeDialog = event.target.closest("[data-close-dialog]");
  if (closeDialog) $("#" + closeDialog.dataset.closeDialog).close();
}

function bindEvents() {
  document.addEventListener("click", handleDocumentClick);

  $("#new-chat").addEventListener("click", () => createSession());
  $("#sidebar-open").addEventListener("click", () => document.body.classList.add("sidebar-open"));
  $("#sidebar-close").addEventListener("click", closeSidebar);
  $("#sidebar-scrim").addEventListener("click", closeSidebar);

  $("#history-query").addEventListener("input", (event) => {
    state.historyQuery = event.target.value;
    renderHistory();
  });
  $$('[data-history-mode]').forEach((button) => {
    button.addEventListener("click", () => {
      state.historyMode = button.dataset.historyMode;
      renderHistory();
      void syncConversations();
    });
  });

  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void sendMessage();
  });
  $("#send-button").addEventListener("click", () => {
    if (state.controller) state.controller.abort();
  });
  $("#chat-input").addEventListener("input", resizeInput);
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void sendMessage();
    }
  });
  $("#chat-files").addEventListener("change", (event) => {
    state.files = [...event.target.files].slice(0, 4);
    renderAttachments();
    event.target.value = "";
  });
  $$('input[name="attachment-policy"]').forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.attachmentPolicy = input.value;
    });
  });

  $("#rename-chat").addEventListener("click", () => openRenameDialog(state.currentSessionId));
  $("#export-chat").addEventListener("click", exportConversation);
  $("#archive-chat").addEventListener("click", () => void performConversationAction("archive", state.currentSessionId));
  $("#delete-chat").addEventListener("click", () => openDeleteDialog(state.currentSessionId));
  $("#restore-current-chat").addEventListener("click", () => void performConversationAction("restore", state.currentSessionId));
  $("#cancel-delete-chat").addEventListener("click", () => $("#conversation-delete-dialog").close());
  $("#confirm-delete-chat").addEventListener("click", () => void confirmDeleteConversation());

  $("#conversation-rename-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const session = currentSession();
    const title = event.currentTarget.elements.title.value.trim();
    if (!session || !title) return;
    if (session.serverBacked) {
      try {
        await apiJson(`${API.conversations}/${encodeURIComponent(session.id)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        });
      } catch (error) {
        if (error.code !== "conversation_not_found") {
          toast(error.message);
          return;
        }
      }
    }
    session.title = title;
    session.updatedAt = Date.now();
    saveSessions();
    $("#conversation-rename-dialog").close();
    toast("Chat renomeado.");
  });

  $("#theme-toggle").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });

  $("#library-add").addEventListener("click", () => $("#library-add-dialog").showModal());
  document.addEventListener("input", (event) => {
    if (event.target.id === "library-query") {
      state.libraryFilters.query = event.target.value;
      renderLibraryDocuments();
    }
  });
  document.addEventListener("change", (event) => {
    if (event.target.id === "library-category") {
      state.libraryFilters.category = event.target.value;
      renderLibraryDocuments();
    }
    if (event.target.id === "library-language") {
      state.libraryFilters.language = event.target.value;
      renderLibraryDocuments();
    }
  });

  $("#library-add-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const values = new FormData(form);
    const file = values.get("file");
    if (!(file instanceof File) || !file.size) {
      toast("Selecione um PDF.");
      return;
    }
    const submit = $('button[type="submit"]', form);
    submit.disabled = true;
    try {
      await queuePdfFile(file, {
        title: values.get("title"),
        authors: values.get("authors"),
        year: values.get("year"),
        category: values.get("category"),
        language: values.get("language"),
      });
      form.reset();
      $("#library-add-dialog").close();
      toast("Referência recebida; indexação em andamento.");
    } catch (error) {
      toast(error.code === "duplicate_pdf" ? "Este PDF já está nas referências." : error.message);
    } finally {
      submit.disabled = false;
    }
  });

  $("#library-edit-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const values = new FormData(form);
    const sourceId = values.get("source_id");
    const submit = $('button[type="submit"]', form);
    submit.disabled = true;
    try {
      await apiJson(`/api/library/${encodeURIComponent(sourceId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: values.get("title"),
          authors: values.get("authors"),
          year: values.get("year") || null,
          category: values.get("category"),
          language: values.get("language"),
        }),
      });
      $("#library-edit-dialog").close();
      state.libraryData = null;
      await loadLibrary(true);
      toast("Metadados salvos. Reindexe para atualizar a busca.");
    } catch (error) {
      toast(error.message);
    } finally {
      submit.disabled = false;
    }
  });

  $("#media-close").addEventListener("click", () => $("#media-dialog").close());
  $("#media-dialog").addEventListener("click", (event) => {
    if (event.target === $("#media-dialog")) $("#media-dialog").close();
  });

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
      event.preventDefault();
      createSession();
    }
    if (event.key === "Escape") closeSidebar();
  });

  window.addEventListener("hashchange", () => {
    activateView(location.hash === "#references" ? "library" : "chat", false);
  });
}

function initialize() {
  applyTheme(localStorage.getItem(THEME_KEY) || "light");
  loadSessions();
  bindEvents();
  renderHistory();
  renderConversation();
  activateView(location.hash === "#references" ? "library" : "chat", false);
  void syncConversations();
  void pollStatus();
  $("#lucide-runtime")?.addEventListener("load", icons);
  icons();
}

initialize();
