"use strict";

const API = {
  e3: "/api/results/e3",
  e2: "/api/results/e2",
  reliability: "/api/reliability",
  sources: "/api/sources",
  render: "/api/render",
};

const VIEW_NAMES = new Set(["chat", "e3", "e2", "reliability", "sources"]);
const STORAGE_KEY = "aliado:sessions:canonical";
const THEME_KEY = "aliado:theme";
const MAX_SESSIONS = 12;

const state = {
  currentView: "chat",
  sessions: [],
  currentSessionId: null,
  files: [],
  controller: null,
  cache: {},
  figures: new Map(),
  zoom: 1,
  toastTimer: null,
  identity: {
    displayName: "Rodolfo",
    greeting: greetingForHour(new Date().getHours()),
    timezone: "America/Sao_Paulo",
  },
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

function icons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "aria-hidden": "true" } });
}

function typesetMath(container) {
  if (!container || typeof window.renderMathInElement !== "function") return;
  try {
    window.renderMathInElement(container, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      throwOnError: false,
      strict: "warn",
    });
  } catch (_error) {
    // Mantem o LaTeX legivel quando a biblioteca nao consegue interpretar uma expressao.
  }
}

function updateWelcomeIdentity() {
  const title = $("#welcome-title");
  if (!title) return;
  title.textContent = `${state.identity.greeting}, ${state.identity.displayName}.`;
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function pct(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  return `${fmt(Number(value) * 100, digits)}%`;
}

function sci(value) {
  if (value === null || value === undefined) return "—";
  const [mantissa, exponent] = Number(value).toExponential(2).split("e");
  const superscript = String(Number(exponent))
    .split("")
    .map((character) => ({ "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" })[character])
    .join("");
  return `${mantissa.replace(".", ",")} × 10${superscript}`;
}

function bytes(value) {
  const number = Number(value || 0);
  if (number < 1024) return `${number} B`;
  if (number < 1024 * 1024) return `${fmt(number / 1024, 1)} KB`;
  return `${fmt(number / (1024 * 1024), 1)} MB`;
}

function ci(metric, digits = 3) {
  return `${fmt(metric.ci95_low, digits)}–${fmt(metric.ci95_high, digits)}`;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("is-visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => node.classList.remove("is-visible"), 2400);
}

function sessionId() {
  if (window.crypto?.randomUUID) return `sessao_${crypto.randomUUID().replaceAll("-", "")}`;
  return `sessao_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function loadSessions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    state.sessions = Array.isArray(parsed) ? parsed.slice(0, MAX_SESSIONS) : [];
  } catch (_error) {
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
  state.sessions = state.sessions.slice(0, MAX_SESSIONS);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
  renderHistory();
}

function createSession(render = true) {
  const session = {
    id: sessionId(),
    title: "Nova conversa",
    updatedAt: Date.now(),
    messages: [],
  };
  state.sessions.unshift(session);
  state.currentSessionId = session.id;
  saveSessions();
  if (render) {
    activateView("chat");
    renderConversation();
    $("#chat-input").focus();
  }
  return session;
}

function renderHistory() {
  const list = $("#history-list");
  if (!state.sessions.length) {
    list.innerHTML = '<div class="history-empty">Nenhuma conversa</div>';
    return;
  }
  list.innerHTML = state.sessions
    .map(
      (session) => `
        <button class="history-item ${session.id === state.currentSessionId ? "is-current" : ""}"
          type="button" data-session-id="${escapeHtml(session.id)}" title="${escapeHtml(session.title)}">
          ${escapeHtml(session.title)}
        </button>`,
    )
    .join("");
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

function activateView(view, updateHash = true) {
  if (!VIEW_NAMES.has(view)) view = "chat";
  state.currentView = view;
  $$('[data-view-panel]').forEach((panel) => {
    const active = panel.dataset.viewPanel === view;
    panel.hidden = !active;
    panel.classList.toggle("is-active", active);
  });
  $$(".nav-item").forEach((item) => {
    const active = item.dataset.view === view;
    item.classList.toggle("is-active", active);
    if (active) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
  if (updateHash && location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
  if (view !== "chat") loadPanel(view);
  closeSidebar();
  $("#workspace-main").focus({ preventScroll: true });
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

function appendCitations(container, citations) {
  if (!Array.isArray(citations) || !citations.length) return;
  const list = document.createElement("div");
  list.className = "citation-list";
  citations.forEach((citation) => {
    const link = document.createElement("a");
    link.href = citation.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = citation.label || citation.url;
    list.appendChild(link);
  });
  container.appendChild(list);
}

function renderAssistantMessage(message, index) {
  const article = document.createElement("article");
  article.className = "message message-assistant";
  article.innerHTML = `
    <div class="assistant-layout">
      <div class="assistant-avatar" aria-hidden="true">A</div>
      <div>
        <div class="message-author">ALIAdo</div>
        <div class="message-content"></div>
        <div class="message-meta">
          <span>${message.response_ms ? `${fmt(message.response_ms, 0)} ms` : "Resposta acadêmica"}</span>
          <div class="message-actions">
            <button class="message-action" type="button" data-copy-index="${index}" aria-label="Copiar resposta" title="Copiar resposta"><i data-lucide="copy"></i></button>
            <button class="message-action" type="button" data-retry-index="${index}" aria-label="Tentar novamente" title="Tentar novamente"><i data-lucide="refresh-cw"></i></button>
          </div>
        </div>
      </div>
    </div>`;
  const content = $(".message-content", article);
  content.textContent = message.content;
  content.dataset.messageIndex = String(index);
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

  const applyCitationsFallback = () => {
    messages.forEach(({ id }) => {
      const node = $(`.message-content[data-message-index="${id}"]`);
      const message = session.messages[Number(id)];
      if (node && !$(".citation-list", node)) appendCitations(node, message?.citations);
    });
  };

  try {
    const response = await fetch(API.render, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ messages }),
    });
    if (!response.ok) throw new Error(`Falha HTTP ${response.status}`);
    const payload = await response.json();
    if (state.currentSessionId !== sessionIdAtStart) return;
    payload.messages.forEach((rendered) => {
      const node = $(`.message-content[data-message-index="${rendered.id}"]`);
      const message = session.messages[Number(rendered.id)];
      if (!node || !message) return;
      node.innerHTML = rendered.html;
      appendCitations(node, message.citations);
      typesetMath(node);
    });
  } catch (_error) {
    if (state.currentSessionId === sessionIdAtStart) applyCitationsFallback();
  }
}

function renderConversation() {
  const conversation = $("#conversation");
  const session = currentSession();
  conversation.innerHTML = "";
  if (!session || !session.messages.length) {
    conversation.appendChild($("#welcome-state-template") || buildWelcomeState());
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
  icons();
  void hydrateAssistantMessages(session);
  scrollConversation(false);
}

function buildWelcomeState() {
  const source = $("#welcome-state");
  if (source) return source;
  const welcome = document.createElement("div");
  welcome.className = "welcome-state";
  welcome.id = "welcome-state";
  welcome.innerHTML = `
    <div class="welcome-mark" aria-hidden="true">A</div>
    <h2 id="welcome-title">${escapeHtml(state.identity.greeting)}, ${escapeHtml(state.identity.displayName)}.</h2>
    <p>Em que parte da dissertação trabalhamos agora?</p>
    <div class="prompt-grid" id="prompt-grid">
      <button type="button" data-prompt="Compare o Autoencoder Denso com o AE-LSTM nos resultados E3."><i data-lucide="git-compare-arrows"></i><span>Comparar Denso e AE-LSTM</span></button>
      <button type="button" data-prompt="Interprete os limites de detectabilidade SMD95 da validação FMECA E2."><i data-lucide="scan-search"></i><span>Interpretar SMD95</span></button>
      <button type="button" data-prompt="Explique as curvas R(t), F(t), f(t) e h(t) dos componentes."><i data-lucide="activity"></i><span>Analisar confiabilidade</span></button>
      <button type="button" data-prompt="Prepare um resumo acadêmico dos resultados atuais para minha orientadora."><i data-lucide="file-text"></i><span>Resumo para orientadora</span></button>
    </div>`;
  return welcome;
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
  const node = $("#stream-status");
  node.textContent = message;
  node.classList.toggle("is-active", Boolean(message));
}

function addPendingAssistant() {
  const article = document.createElement("article");
  article.className = "message message-assistant";
  article.innerHTML = `
    <div class="assistant-layout">
      <div class="assistant-avatar" aria-hidden="true">A</div>
      <div>
        <div class="message-author">ALIAdo</div>
        <div class="message-content streaming"></div>
        <div class="message-meta"><span>Gerando resposta</span></div>
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

async function sendMessage(rawMessage) {
  if (state.controller) return;
  const input = $("#chat-input");
  const message = String(rawMessage ?? input.value).trim();
  if (!message) return;
  let session = currentSession();
  if (!session) session = createSession(false);
  const history = previousHistory(session);
  session.messages.push({ role: "user", content: message });
  if (session.title === "Nova conversa") session.title = message.slice(0, 54);
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

  let body;
  let headers = {};
  if (state.files.length) {
    const form = new FormData();
    form.append("message", message);
    form.append("history", JSON.stringify(history));
    form.append("session_id", session.id);
    state.files.forEach((file) => form.append("files", file, file.name));
    body = form;
  } else {
    headers = { "Content-Type": "application/json" };
    body = JSON.stringify({ message, history, session_id: session.id });
  }

  let complete = "";
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers,
      body,
      signal: controller.signal,
    });
    await consumeSse(response, (event, payload) => {
      if (event === "status") {
        updateStreamStatus(payload.message || "Consultando base acadêmica");
      } else if (event === "delta") {
        complete += payload.text || "";
        pendingContent.textContent = complete;
        scrollConversation(false);
      } else if (event === "done") {
        complete = payload.answer || complete;
        pendingContent.classList.remove("streaming");
        pendingContent.innerHTML = payload.answer_html || escapeHtml(complete);
        appendCitations(pendingContent, payload.citations);
        typesetMath(pendingContent);
        pendingMeta.textContent = `${fmt(payload.response_ms, 0)} ms · ${payload.route || "agente"}`;
        const index = session.messages.length;
        const actions = document.createElement("div");
        actions.className = "message-actions";
        actions.innerHTML = `
          <button class="message-action" type="button" data-copy-index="${index}" aria-label="Copiar resposta" title="Copiar resposta"><i data-lucide="copy"></i></button>
          <button class="message-action" type="button" data-retry-index="${index}" aria-label="Tentar novamente" title="Tentar novamente"><i data-lucide="refresh-cw"></i></button>`;
        $(".message-meta", pending).appendChild(actions);
        session.messages.push({
          role: "assistant",
          content: complete,
          citations: payload.citations || [],
          response_ms: payload.response_ms,
        });
        session.updatedAt = Date.now();
        saveSessions();
        updateStreamStatus("");
        icons();
      } else if (event === "error") {
        throw new Error(payload.detail || "O agente não conseguiu responder.");
      }
    });
  } catch (error) {
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
    renderAttachments();
    setStreaming(false);
    input.focus();
  }
}

function renderAttachments() {
  const list = $("#attachment-list");
  list.innerHTML = "";
  state.files.forEach((file, index) => {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    chip.innerHTML = `<i data-lucide="file" aria-hidden="true"></i><span></span><button type="button" data-remove-file="${index}" aria-label="Remover anexo" title="Remover anexo"><i data-lucide="x"></i></button>`;
    $("span", chip).textContent = file.name;
    list.appendChild(chip);
  });
  icons();
}

function resizeInput() {
  const input = $("#chat-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 170)}px`;
}

function figureCards(figures) {
  figures.forEach((figure) => state.figures.set(figure.url, figure));
  return `<div class="figure-grid">${figures
    .map(
      (figure) => `
      <article class="figure-card">
        <button class="figure-open" type="button" data-open-figure="${escapeHtml(figure.url)}" aria-label="Ampliar ${escapeHtml(figure.title)}">
          <img src="${escapeHtml(figure.url)}" alt="${escapeHtml(figure.title)}" loading="lazy" decoding="async">
        </button>
        <div class="figure-caption">
          <div><strong>${escapeHtml(figure.title)}</strong><p>${escapeHtml(figure.note)}</p></div>
          <div class="figure-links">
            <button class="icon-button" type="button" data-open-figure="${escapeHtml(figure.url)}" aria-label="Ampliar" title="Ampliar"><i data-lucide="maximize-2"></i></button>
            <a class="icon-button" href="${escapeHtml(figure.pdf_url)}" target="_blank" rel="noreferrer" aria-label="Abrir PDF" title="Abrir PDF"><i data-lucide="file-down"></i></a>
          </div>
        </div>
      </article>`,
    )
    .join("")}</div>`;
}

function downloadRows(items) {
  return `<div class="download-list">${items
    .map(
      (item) => `
      <div class="download-row">
        <div><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.sha256)} · ${bytes(item.size_bytes)}</small></div>
        <a class="download-link" href="${escapeHtml(item.url)}" download><i data-lucide="download"></i><span>Baixar</span></a>
      </div>`,
    )
    .join("")}</div>`;
}

function metricRow(label, dense, lstm) {
  return `<tr><td>${escapeHtml(label)}</td><td class="numeric">${fmt(dense.estimate)}<br><small>IC95% ${ci(dense)}</small></td><td class="numeric">${fmt(lstm.estimate)}<br><small>IC95% ${ci(lstm)}</small></td></tr>`;
}

function renderE3(data) {
  const dense = data.metrics.ae_denso;
  const lstm = data.metrics.ae_lstm;
  const diff = data.paired_differences.find((item) => item.metric === "auc_pr");
  const rows = [
    ["AUC-PR", dense.auc_pr, lstm.auc_pr],
    ["AUC-ROC", dense.auc_roc, lstm.auc_roc],
    ["Sensibilidade", dense.sensitivity, lstm.sensitivity],
    ["Especificidade", dense.specificity, lstm.specificity],
    ["Acurácia balanceada", dense.balanced_accuracy, lstm.balanced_accuracy],
    ["MCC", dense.mcc, lstm.mcc],
    ["F1", dense.f1, lstm.f1],
  ];
  return `
    <section class="summary-band">
      <h2>Comparação congelada nos 14 ensaios de falha</h2>
      <p>A semente 42 foi pré-fixada; seleção, escalonamento e limiares não acessaram F1L–F7M. AUC-PR é a métrica primária.</p>
      <div class="boundary-note"><i data-lucide="info"></i><span>${escapeHtml(data.dataset.fault_boundary.caveat)}</span></div>
    </section>
    <section class="metric-strip" aria-label="Síntese E3">
      <div class="metric-item is-blue"><span>AUC-PR Denso</span><strong>${fmt(dense.auc_pr.estimate)}</strong><small>IC95% ${ci(dense.auc_pr)}</small></div>
      <div class="metric-item is-amber"><span>AUC-PR AE-LSTM</span><strong>${fmt(lstm.auc_pr.estimate)}</strong><small>IC95% ${ci(lstm.auc_pr)}</small></div>
      <div class="metric-item is-accent"><span>Diferença pareada</span><strong>${fmt(diff.difference_dense_minus_lstm)}</strong><small>IC95% ${fmt(diff.ci95_low)}–${fmt(diff.ci95_high)}</small></div>
      <div class="metric-item is-rose"><span>Ensaios independentes</span><strong>14</strong><small>Bootstrap pareado por ensaio</small></div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Métricas macro</h2><p>Estimativas na semente de referência, com IC95% por 20.000 reamostragens.</p></div><div class="model-key"><span><i></i>Denso</span><span><i></i>AE-LSTM</span></div></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Métrica</th><th class="numeric">Autoencoder Denso</th><th class="numeric">AE-LSTM</th></tr></thead><tbody>${rows.map((row) => metricRow(...row)).join("")}</tbody></table></div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Arquiteturas e ponto operacional</h2><p>Mesmo pré-processamento, orçamento de treino e conjunto de sementes.</p></div></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Modelo</th><th>Arquitetura</th><th class="numeric">Parâmetros</th><th class="numeric">Limiar p99</th><th class="numeric">FP saudável</th></tr></thead><tbody>
        ${Object.entries(data.models).map(([id, model]) => `<tr><td>${escapeHtml(model.name)}</td><td>${escapeHtml(model.architecture)}</td><td class="numeric">${Number(model.n_parameters).toLocaleString("pt-BR")}</td><td class="numeric">${fmt(model.score_threshold, 4)}</td><td class="numeric">${pct(model.healthy_test_false_positive_rate, 2)}</td></tr>`).join("")}
      </tbody></table></div>
    </section>
    <section class="section-band"><div class="section-heading"><div><h2>Figuras acadêmicas</h2><p>PNG 300 dpi e PDF vetorial.</p></div></div>${figureCards(data.figures)}</section>
    <section class="section-band"><div class="section-heading"><div><h2>Dados-fonte</h2><p>Tabelas usadas na publicação das figuras.</p></div></div>${downloadRows(data.tables)}</section>`;
}

function renderE2(data) {
  return `
    <section class="summary-band">
      <h2>Primeiro cruzamento persistente no eixo de magnitude</h2>
      <p>${escapeHtml(data.smd95_definition)}. As mesmas janelas, severidades e perturbações foram compartilhadas pelos dois modelos.</p>
      <div class="boundary-note"><i data-lucide="triangle-alert"></i><span>O eixo a<sub>det</sub> é adimensional e não representa tempo, vida útil ou RUL. ${escapeHtml(data.interval_caveat)}</span></div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Limites de detectabilidade SMD95</h2><p>“Não atingido” permanece censurado no limite a<sub>det</sub>=1.</p></div><div class="model-key"><span><i></i>Denso</span><span><i></i>AE-LSTM</span></div></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Modelo</th><th>Componente</th><th class="numeric">SMD95</th><th class="numeric">NPR</th><th class="numeric">Detecção em a=1</th><th class="numeric">Sem cruzamento</th></tr></thead><tbody>
        ${data.summary.map((item) => `<tr><td>${escapeHtml(item.model_name)}</td><td>${escapeHtml(item.component_name)}</td><td class="numeric"><span class="status-text ${item.smd95_status === "not_reached" ? "not-reached" : ""}">${item.smd95 === null ? "Não atingido" : fmt(item.smd95, 2)}</span></td><td class="numeric">${item.npr}</td><td class="numeric">${pct(item.detection_at_max, 1)}</td><td class="numeric">${fmt(item.indetectable_at_max_pct, 1)}%</td></tr>`).join("")}
      </tbody></table></div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Assinaturas FMECA</h2><p>Proxies sintéticos aplicados ao holdout saudável completo.</p></div></div>
      <div class="signature-grid">${data.signatures.map((item) => `<article class="signature-item"><h3>${escapeHtml(item.component_name)} · NPR ${item.npr}</h3><p>${escapeHtml(item.physical_hypothesis)}</p><p>${escapeHtml(item.limitation)}</p><code>${escapeHtml(item.formula)}</code></article>`).join("")}</div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Figuras acadêmicas</h2><p>Funções empíricas e diagnóstico paramétrico mantidos em papéis distintos.</p></div></div>
      ${figureCards(data.figures)}
      <div class="boundary-note"><i data-lucide="badge-alert"></i><span>${escapeHtml(data.weibull_acceptance_scope)}</span></div>
    </section>
    <section class="section-band"><div class="section-heading"><div><h2>Dados-fonte</h2><p>Pontos, curvas e diagnósticos usados nas figuras.</p></div></div>${downloadRows(data.tables)}</section>`;
}

function evidenceLabel(type) {
  return type === "direct_bibliographic" ? "Bibliográfica direta" : "Sensibilidade derivada";
}

function renderReliability(data) {
  return `
    <section class="summary-band">
      <h2>Modelo exponencial com taxa de risco constante</h2>
      <p>As taxas são bibliográficas ou derivadas do TCC. O GPVS-Faults avalia os detectores e não fornece tempos de vida por ativo.</p>
      <div class="boundary-note"><i data-lucide="shield-alert"></i><span>Weibull físico não estimável: ${escapeHtml(data.physical_weibull.reason)}.</span></div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Funções físicas</h2><p>Tempo primário em horas, com conversão explícita por 8.760 h/ano.</p></div></div>
      <div class="formula-strip">
        <div class="formula-item"><span>Confiabilidade</span><code>R(t) = e<sup>−λt</sup></code></div>
        <div class="formula-item"><span>Falha acumulada</span><code>F(t) = 1 − e<sup>−λt</sup></code></div>
        <div class="formula-item"><span>Densidade</span><code>f(t) = λe<sup>−λt</sup></code></div>
        <div class="formula-item"><span>Risco</span><code>h(t) = λ</code></div>
      </div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Cenários rastreáveis</h2><p>Taxas derivadas permanecem rotuladas separadamente da transcrição direta do fusível.</p></div></div>
      <div class="table-wrap"><table class="data-table"><thead><tr><th>Componente</th><th>Natureza</th><th class="numeric">λ (h⁻¹)</th><th class="numeric">λ (ano⁻¹)</th><th class="numeric">1/λ (anos)</th><th>Origem</th></tr></thead><tbody>
        ${data.scenarios.map((item) => `<tr><td>${escapeHtml(item.component_name)}</td><td>${evidenceLabel(item.evidence_type)}</td><td class="numeric">${sci(item.lambda_per_hour)}</td><td class="numeric">${fmt(item.lambda_per_year, 5)}</td><td class="numeric">${fmt(item.reciprocal_time_years, 2)}</td><td>${escapeHtml(item.source_table)}<br><small>PDF ${item.pdf_page}, impressa ${item.printed_page}</small></td></tr>`).join("")}
      </tbody></table></div>
    </section>
    <section class="section-band"><div class="section-heading"><div><h2>Figuras acadêmicas</h2><p>Curvas temporais e comparação das taxas de componente.</p></div></div>${figureCards(data.figures)}</section>
    <section class="section-band"><div class="section-heading"><div><h2>Dados-fonte</h2><p>Metodologia, cenários e curvas publicadas.</p></div></div>${downloadRows(data.tables)}</section>`;
}

function renderSources(data) {
  const datasetHashes = Object.values(data.dataset.raw_files_sha256);
  return `
    <section class="summary-band">
      <h2>Dataset experimental único</h2>
      <p><a href="${escapeHtml(data.dataset.url)}" target="_blank" rel="noreferrer">${escapeHtml(data.dataset.name)}</a> · ${data.dataset.experiments} ensaios · DOI ${escapeHtml(data.dataset.doi)}</p>
      <div class="boundary-note"><i data-lucide="fingerprint"></i><span>16 arquivos brutos verificados por SHA-256; primeiro hash ${escapeHtml(datasetHashes[0])}.</span></div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Regras de separação</h2><p>Limites que governam a interpretação dos resultados.</p></div></div>
      <div class="rule-list">${data.separation_rules.map((rule) => `<div class="rule-row"><span>${escapeHtml(rule)}</span><i data-lucide="check-circle-2" aria-hidden="true"></i></div>`).join("")}</div>
    </section>
    <section class="section-band">
      <div class="section-heading"><div><h2>Fonte de confiabilidade</h2><p>Localização bibliográfica explícita.</p></div></div>
      <div class="source-list">${data.bibliography.map((source) => `<div class="source-row"><div><strong>${escapeHtml(source.title)}</strong><small>${escapeHtml(source.sha256)} · PDF ${source.pdf_page}, página impressa ${source.printed_page}</small></div><a class="download-link" href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer"><i data-lucide="external-link"></i><span>Abrir PDF</span></a></div>`).join("")}</div>
    </section>
    <section class="section-band"><div class="section-heading"><div><h2>Manifestos</h2><p>Proveniência normalizada e hashes dos outputs.</p></div></div>${downloadRows(data.manifests)}</section>
    <section class="section-band"><div class="section-heading"><div><h2>Relatórios</h2><p>Sínteses metodológicas versionadas.</p></div></div>${downloadRows(data.reports)}</section>`;
}

const RENDERERS = { e3: renderE3, e2: renderE2, reliability: renderReliability, sources: renderSources };

async function loadPanel(view, force = false) {
  const content = $(`#${view}-content`);
  const loading = $(`[data-loading="${view}"]`);
  if (!content || !loading) return;
  if (state.cache[view] && !force) return;
  loading.hidden = false;
  content.innerHTML = "";
  try {
    const response = await fetch(API[view], { headers: { Accept: "application/json" } });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Falha HTTP ${response.status}`);
    }
    const data = await response.json();
    state.cache[view] = data;
    content.innerHTML = RENDERERS[view](data);
    loading.hidden = true;
    icons();
  } catch (error) {
    loading.hidden = true;
    content.innerHTML = `<div class="error-state"><strong>Não foi possível carregar esta área.</strong><p>${escapeHtml(error.message)}</p><button class="download-link" type="button" data-reload-panel="${view}"><i data-lucide="refresh-cw"></i><span>Tentar novamente</span></button></div>`;
    icons();
  }
}

function openFigure(url) {
  const figure = state.figures.get(url);
  if (!figure) return;
  state.zoom = 1;
  $("#dialog-title").textContent = figure.title;
  $("#dialog-note").textContent = figure.note;
  $("#dialog-image").src = figure.url;
  $("#dialog-image").alt = figure.title;
  $("#dialog-pdf").href = figure.pdf_url;
  updateZoom();
  $("#image-dialog").showModal();
}

function updateZoom() {
  state.zoom = Math.min(2.5, Math.max(0.6, state.zoom));
  $("#dialog-image").style.width = `${state.zoom * 100}%`;
  $("#zoom-level").textContent = `${Math.round(state.zoom * 100)}%`;
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
  const button = $("#theme-toggle");
  button.innerHTML = theme === "dark" ? '<i data-lucide="sun"></i>' : '<i data-lucide="moon"></i>';
  icons();
}

async function pollStatus() {
  let delay = 4000;
  try {
    const response = await fetch("/api/status", { headers: { Accept: "application/json" } });
    const payload = await response.json();
    if (payload.identity) {
      state.identity = {
        displayName: payload.identity.display_name || state.identity.displayName,
        greeting: payload.identity.greeting || state.identity.greeting,
        timezone: payload.identity.timezone || state.identity.timezone,
      };
      updateWelcomeIdentity();
    }
    const node = $("#runtime-status");
    node.dataset.state = payload.state;
    $("#runtime-status-text").textContent = {
      pronto: "Pronto",
      iniciando: "Aquecendo base",
      degradado: "Modo degradado",
    }[payload.state] || "Verificando";
    delay = payload.state === "pronto" ? 15000 : 4000;
  } catch (_error) {
    $("#runtime-status").dataset.state = "degradado";
    $("#runtime-status-text").textContent = "Sem conexão";
  }
  setTimeout(pollStatus, delay);
}

function exportConversation() {
  const session = currentSession();
  if (!session || !session.messages.length) {
    toast("A conversa ainda está vazia.");
    return;
  }
  const content = [
    `# ${session.title}`,
    "",
    ...session.messages.flatMap((message) => [
      `## ${message.role === "user" ? "Rodolfo" : "ALIAdo"}`,
      "",
      message.content,
      "",
    ]),
  ].join("\n");
  const url = URL.createObjectURL(new Blob([content], { type: "text/markdown;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${session.id}_sessao.md`;
  link.click();
  URL.revokeObjectURL(url);
}

function retryMessage(index) {
  const session = currentSession();
  if (!session) return;
  let prompt = "";
  for (let cursor = Number(index) - 1; cursor >= 0; cursor -= 1) {
    if (session.messages[cursor]?.role === "user") {
      prompt = session.messages[cursor].content;
      break;
    }
  }
  if (prompt) sendMessage(prompt);
}

function bindEvents() {
  document.addEventListener("click", async (event) => {
    const nav = event.target.closest("[data-view]");
    if (nav) activateView(nav.dataset.view);

    const prompt = event.target.closest("[data-prompt]");
    if (prompt) sendMessage(prompt.dataset.prompt);

    const historyItem = event.target.closest("[data-session-id]");
    if (historyItem) {
      state.currentSessionId = historyItem.dataset.sessionId;
      activateView("chat");
      renderConversation();
    }

    const removeFile = event.target.closest("[data-remove-file]");
    if (removeFile) {
      state.files.splice(Number(removeFile.dataset.removeFile), 1);
      renderAttachments();
    }

    const figure = event.target.closest("[data-open-figure]");
    if (figure) openFigure(figure.dataset.openFigure);

    const reload = event.target.closest("[data-reload-panel]");
    if (reload) loadPanel(reload.dataset.reloadPanel, true);

    const copy = event.target.closest("[data-copy-index]");
    if (copy) {
      const message = currentSession()?.messages[Number(copy.dataset.copyIndex)];
      if (message) {
        await navigator.clipboard.writeText(message.content);
        toast("Resposta copiada.");
      }
    }

    const retry = event.target.closest("[data-retry-index]");
    if (retry) retryMessage(retry.dataset.retryIndex);
  });

  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  $("#send-button").addEventListener("click", () => {
    if (state.controller) state.controller.abort();
  });
  $("#chat-input").addEventListener("input", resizeInput);
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  $("#chat-files").addEventListener("change", (event) => {
    state.files = [...event.target.files].slice(0, 4);
    renderAttachments();
    event.target.value = "";
  });

  $("#new-chat").addEventListener("click", () => createSession());
  $(".mobile-new-chat").addEventListener("click", () => createSession());
  $("#clear-chat").addEventListener("click", () => createSession());
  $("#export-chat").addEventListener("click", exportConversation);
  $("#sidebar-open").addEventListener("click", () => document.body.classList.add("sidebar-open"));
  $("#sidebar-close").addEventListener("click", closeSidebar);
  $("#sidebar-scrim").addEventListener("click", closeSidebar);

  $("#theme-toggle").addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });

  $("#dialog-close").addEventListener("click", () => $("#image-dialog").close());
  $("#zoom-in").addEventListener("click", () => { state.zoom += 0.2; updateZoom(); });
  $("#zoom-out").addEventListener("click", () => { state.zoom -= 0.2; updateZoom(); });
  $("#image-dialog").addEventListener("click", (event) => {
    if (event.target === $("#image-dialog")) $("#image-dialog").close();
  });

  window.addEventListener("hashchange", () => activateView(location.hash.slice(1), false));
}

function initialize() {
  const savedTheme = localStorage.getItem(THEME_KEY);
  const preferredTheme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  applyTheme(savedTheme || preferredTheme);
  loadSessions();
  renderHistory();
  renderConversation();
  bindEvents();
  activateView(location.hash.slice(1) || "chat", false);
  pollStatus();
  $("#lucide-runtime")?.addEventListener("load", icons);
  icons();
}

initialize();
