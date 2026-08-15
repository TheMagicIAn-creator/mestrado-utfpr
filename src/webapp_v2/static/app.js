"use strict";

const CHAT_STORAGE_KEY = "aliado-v2-chat";
const SESSION_ID_KEY = "aliado-v2-session-id";

function newSessionId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `${Date.now()}_${Math.random().toString(36).slice(2, 14)}`;
}

function loadChatHistory() {
  try {
    const value = JSON.parse(sessionStorage.getItem(CHAT_STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value.slice(-16) : [];
  } catch (_error) {
    return [];
  }
}

function persistChatHistory() {
  sessionStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(state.chatHistory.slice(-16)));
}

const browserSessionId = sessionStorage.getItem(SESSION_ID_KEY) || newSessionId();
sessionStorage.setItem(SESSION_ID_KEY, browserSessionId);

const state = {
  dashboard: null,
  reliabilityCurves: null,
  health: null,
  trialMetric: "auc_roc",
  reliabilityFunction: "reliability",
  selectedScenarios: new Set(),
  chatHistory: loadChatHistory(),
  sessionId: browserSessionId,
  agentWarmupPromise: null,
  chatRequestActive: false,
};

const VIEW_META = {
  overview: ["Resultados V2", "Visão geral"],
  autoencoder: ["Modelo", "Autoencoder"],
  reliability: ["Cenários", "Confiabilidade"],
  fmeca: ["Criticidade", "FMECA"],
  evidence: ["Rastreabilidade", "Evidências"],
  agent: ["Pesquisa assistida", "Agente"],
};

const RELIABILITY_COLORS = {
  torres_colli_rate: "#c43d3d",
  cristaldi_inverter_rate: "#2675d8",
  obeidat_high_quality: "#168451",
  obeidat_low_quality: "#d88600",
  dhople_markov_example: "#5d4bb7",
};

const RELIABILITY_DASH = {
  torres_colli_rate: "dash",
  cristaldi_inverter_rate: "solid",
  obeidat_high_quality: "dashdot",
  obeidat_low_quality: "dashdot",
  dhople_markov_example: "dot",
};

const SOURCE_TYPE_LABELS = {
  secondary_bibliographic_rate: "transcrição secundária",
  literature_assumption: "hipótese da literatura",
  mil_hdbk_217f_prediction: "predição MIL-HDBK-217F",
  illustrative_markov_parameter: "parâmetro ilustrativo",
};

const FUNCTION_META = {
  reliability: ["Confiabilidade, R(t)", "Probabilidade", false, 20],
  cumulative_failure_probability: ["Probabilidade acumulada, F(t)", "Probabilidade", false, 20],
  failure_density_per_year: ["Densidade de falha, f(t) (ano⁻¹)", "Densidade (ano⁻¹)", true, 10],
  hazard_per_year: ["Taxa instantânea, h(t) (ano⁻¹)", "Taxa (ano⁻¹)", true, 20],
};

const $ = (id) => document.getElementById(id);

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function formatNumber(value, digits = 3) {
  return Number(value).toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatPercent(value, digits = 1) {
  return `${formatNumber(Number(value) * 100, digits)}%`;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${formatNumber(value / 1024, 1)} KB`;
  return `${formatNumber(value / (1024 * 1024), 1)} MB`;
}

function shortHash(hash) {
  return hash ? `${hash.slice(0, 12)}…${hash.slice(-6)}` : "—";
}

function metricInterval(metric) {
  return `IC95% ${formatNumber(metric.ci95_low, 3)}–${formatNumber(metric.ci95_high, 3)}`;
}

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function appendCell(row, text, className = "") {
  const cell = createElement("td", className, text);
  row.appendChild(cell);
  return cell;
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = { detail: `Resposta HTTP ${response.status}` };
  }
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `Erro HTTP ${response.status}`);
  }
  return payload;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("is-visible"), 3800);
}

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

function switchView(view, updateHash = true) {
  if (!VIEW_META[view]) view = "agent";
  document.querySelectorAll(".view").forEach((section) => {
    const active = section.dataset.view === view;
    section.hidden = !active;
    section.classList.toggle("is-active", active);
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    const active = button.dataset.target === view;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  // Só a vista do agente vira tela cheia: a página para de rolar e a lista de
  // mensagens passa a ser o único elemento rolável. Nas demais, a rolagem
  // normal da página é o certo — são painéis longos de leitura.
  document.body.classList.toggle("is-agent-view", view === "agent");
  $("view-kicker").textContent = VIEW_META[view][0];
  $("view-title").textContent = VIEW_META[view][1];
  if (updateHash) history.replaceState(null, "", `#${view}`);
  window.requestAnimationFrame(() => {
    renderChartsForView(view);
  });
  if (view === "agent") void warmupAgent();
}

function initializeNavigation() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.target));
  });
  const initial = location.hash.replace("#", "");
  switchView(VIEW_META[initial] ? initial : "agent", false);
}

function initializeTheme() {
  const saved = localStorage.getItem("aliado-theme");
  if (saved === "dark" || saved === "light") {
    document.documentElement.dataset.theme = saved;
  } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    document.documentElement.dataset.theme = "dark";
  }
  updateThemeIcon();
  $("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("aliado-theme", next);
    updateThemeIcon();
    const activeView = document.querySelector(".view:not([hidden])")?.dataset.view || "overview";
    renderChartsForView(activeView);
  });
}

function updateThemeIcon() {
  const button = $("theme-toggle");
  const icon = button.querySelector("svg, i");
  if (!icon) return;
  const name = document.documentElement.dataset.theme === "dark" ? "sun" : "moon";
  if (icon.tagName.toLowerCase() === "i") icon.dataset.lucide = name;
  else {
    const replacement = document.createElement("i");
    replacement.dataset.lucide = name;
    replacement.setAttribute("aria-hidden", "true");
    icon.replaceWith(replacement);
  }
  refreshIcons();
}

function plotLayout(overrides = {}) {
  return {
    autosize: true,
    margin: { l: 62, r: 18, t: 26, b: 55 },
    paper_bgcolor: css("--surface"),
    plot_bgcolor: css("--surface"),
    font: { family: "Inter, Segoe UI, sans-serif", size: 11, color: css("--ink-muted") },
    xaxis: {
      gridcolor: css("--plot-grid"),
      zerolinecolor: css("--line-strong"),
      linecolor: css("--line-strong"),
      tickfont: { color: css("--ink-muted") },
      titlefont: { color: css("--ink-muted") },
    },
    yaxis: {
      gridcolor: css("--plot-grid"),
      zerolinecolor: css("--line-strong"),
      linecolor: css("--line-strong"),
      tickfont: { color: css("--ink-muted") },
      titlefont: { color: css("--ink-muted") },
    },
    legend: {
      orientation: "h",
      x: 0,
      y: 1.12,
      bgcolor: "rgba(0,0,0,0)",
      font: { size: 10 },
    },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line-strong"), font: { color: css("--ink") } },
    ...overrides,
  };
}

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  scrollZoom: false,
  toImageButtonOptions: { format: "png", scale: 2, filename: "aliado-pv-v2" },
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
};

function hydrateOverview() {
  const overview = state.dashboard.overview;
  $("overview-verdict").textContent = overview.verdict;
  $("metric-auc").textContent = formatNumber(overview.metrics.auc_roc.mean, 3);
  $("metric-auc-ci").textContent = metricInterval(overview.metrics.auc_roc);
  $("metric-sensitivity").textContent = formatNumber(overview.metrics.sensitivity.mean, 3);
  $("metric-sensitivity-ci").textContent = metricInterval(overview.metrics.sensitivity);
  $("metric-specificity").textContent = formatNumber(overview.metrics.specificity.mean, 3);
  $("metric-specificity-ci").textContent = metricInterval(overview.metrics.specificity);
  $("metric-fp").textContent = `${formatNumber(overview.healthy_false_positive_pct, 2)}%`;

  const body = $("method-table").querySelector("tbody");
  body.replaceChildren();
  overview.method_comparison.forEach((method) => {
    const row = document.createElement("tr");
    appendCell(row, method.label);
    ["auc_roc", "sensitivity", "specificity", "balanced_accuracy", "mcc"].forEach((name) => {
      const metric = method[name];
      const cell = appendCell(row, formatNumber(metric.mean, 3));
      cell.title = metricInterval(metric);
    });
    body.appendChild(row);
  });
}

function renderTrialChart() {
  if (!state.dashboard || !window.Plotly) return;
  const metric = state.trialMetric;
  const rows = state.dashboard.autoencoder.trials;
  const order = Array.from(new Set(rows.map((row) => row.experiment)));
  const methods = [
    ["autoencoder_v2", "Autoencoder V2", css("--blue"), "circle"],
    ["pca", "PCA", css("--amber"), "square"],
  ];
  const traces = methods.map(([id, label, color, symbol]) => {
    const indexed = new Map(rows.filter((row) => row.method === id).map((row) => [row.experiment, row]));
    const values = order.map((experiment) => indexed.get(experiment)[metric]);
    const mean = values.reduce((total, value) => total + value, 0) / values.length;
    return {
      type: "scatter",
      mode: "markers",
      name: `${label} (média ${formatNumber(mean, 3)})`,
      x: order,
      y: values,
      customdata: order.map((experiment) => indexed.get(experiment).fault_type),
      marker: { color, size: 8, symbol, line: { color: css("--surface"), width: 1 } },
      hovertemplate: "%{x}<br>%{customdata}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
    };
  });
  const labels = { auc_roc: "AUC-ROC", sensitivity: "Sensibilidade", specificity: "Especificidade" };
  window.Plotly.react(
    "trial-chart",
    traces,
    plotLayout({
      margin: { l: 58, r: 16, t: 38, b: 58 },
      xaxis: { ...plotLayout().xaxis, title: "Ensaio independente", categoryorder: "array", categoryarray: order },
      yaxis: { ...plotLayout().yaxis, title: labels[metric], range: [0, 1.03], tickformat: ".0%" },
      shapes: [{ type: "line", x0: -0.5, x1: order.length - 0.5, y0: 0.5, y1: 0.5, line: { color: css("--line-strong"), dash: "dot", width: 1 } }],
    }),
    PLOT_CONFIG,
  );
}

function hydrateAutoencoder() {
  const ae = state.dashboard.autoencoder;
  $("architecture-display").textContent = ae.architecture.display;
  $("architecture-params").textContent = ae.architecture.trainable_parameters.toLocaleString("pt-BR");
  $("canonical-seed").textContent = String(ae.architecture.canonical_seed);
  $("threshold-value").textContent = formatNumber(ae.threshold.value, 4);
  $("calibration-order").textContent = `${ae.threshold.order_one_based}/${ae.threshold.n_calibration}`;

  const dl = $("sample-counts");
  dl.replaceChildren();
  const labels = { treino: "Treino", validacao: "Validação", calibracao: "Calibração", teste: "Teste saudável" };
  Object.entries(ae.sample_counts).forEach(([key, value]) => {
    const wrapper = document.createElement("div");
    wrapper.append(createElement("dt", "", labels[key] || key), createElement("dd", "", `${value} janelas`));
    dl.appendChild(wrapper);
  });
  renderFigureGrid("autoencoder-figures", ae.figures);
}

function renderSelectionChart() {
  if (!state.dashboard || !window.Plotly) return;
  const rows = state.dashboard.autoencoder.selection;
  const names = {
    compacto_12_4: "24-12-4-12-24",
    simetrico_16_8: "24-16-8-16-24",
    profundo_16_8_4: "24-16-8-4-8-16-24",
  };
  const trace = {
    type: "scatter",
    mode: "markers+text",
    x: rows.map((row) => names[row.architecture_id] || row.architecture_id),
    y: rows.map((row) => row.mean_validation_loss),
    text: rows.map((row) => `${row.trainable_parameters.toLocaleString("pt-BR")} par.`),
    textposition: "top center",
    marker: {
      size: rows.map((row) => (row.selected ? 15 : 11)),
      symbol: rows.map((row) => (row.selected ? "diamond" : "circle")),
      color: rows.map((row) => (row.selected ? css("--blue") : css("--ink-faint"))),
      line: { color: css("--surface"), width: 1.5 },
    },
    error_y: {
      type: "data",
      array: rows.map((row) => row.std_validation_loss),
      visible: true,
      color: css("--ink-faint"),
      thickness: 1.2,
      width: 4,
    },
    customdata: rows.map((row) => [row.median_validation_loss, row.n_seeds, row.selected ? "selecionada" : "candidata"]),
    hovertemplate: "%{x}<br>média %{y:.4f}<br>mediana %{customdata[0]:.4f}<br>%{customdata[1]} seeds · %{customdata[2]}<extra></extra>",
  };
  window.Plotly.react(
    "selection-chart",
    [trace],
    plotLayout({
      showlegend: false,
      margin: { l: 64, r: 28, t: 34, b: 68 },
      xaxis: { ...plotLayout().xaxis, title: "Arquitetura candidata", tickangle: 0 },
      yaxis: { ...plotLayout().yaxis, title: "Perda média de validação saudável", rangemode: "tozero" },
    }),
    PLOT_CONFIG,
  );
}

function renderFigureGrid(containerId, figures) {
  const container = $(containerId);
  container.replaceChildren();
  figures.forEach((figure) => {
    const item = createElement("article", "figure-item");
    const button = createElement("button", "figure-preview");
    button.type = "button";
    button.setAttribute("aria-label", `Ampliar ${figure.title}`);
    const image = document.createElement("img");
    image.src = figure.url;
    image.alt = figure.title;
    image.loading = "lazy";
    button.appendChild(image);
    button.addEventListener("click", () => openFigure(figure));
    const meta = createElement("div", "figure-meta");
    meta.append(createElement("strong", "", figure.title), createElement("span", "", figure.question));
    item.append(button, meta);
    container.appendChild(item);
  });
}

function openFigure(figure) {
  $("dialog-title").textContent = figure.title;
  $("dialog-question").textContent = figure.question;
  $("dialog-image").src = figure.url;
  $("dialog-image").alt = figure.title;
  $("dialog-download").href = figure.download_url;
  $("image-dialog").showModal();
  refreshIcons();
}

function initializeFigureDialog() {
  const dialog = $("image-dialog");
  $("dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
}

function hydrateReliability() {
  const reliability = state.dashboard.reliability;
  const controls = $("scenario-controls");
  controls.replaceChildren();
  state.selectedScenarios.clear();
  reliability.scenarios.forEach((scenario) => {
    state.selectedScenarios.add(scenario.scenario_id);
    const label = createElement("label", "scenario-option");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.value = scenario.scenario_id;
    input.addEventListener("change", () => {
      if (input.checked) state.selectedScenarios.add(input.value);
      else state.selectedScenarios.delete(input.value);
      renderReliabilityChart();
    });
    const swatch = createElement("span", "swatch");
    swatch.style.backgroundColor = RELIABILITY_COLORS[scenario.scenario_id];
    label.append(input, swatch, document.createTextNode(scenario.label));
    controls.appendChild(label);
  });

  const body = $("reliability-table").querySelector("tbody");
  body.replaceChildren();
  reliability.scenarios.forEach((scenario) => {
    const row = document.createElement("tr");
    appendCell(row, scenario.label);
    appendCell(row, scenario.scope, "muted-cell");
    appendCell(row, formatNumber(scenario.lambda_per_year, 4));
    appendCell(row, formatNumber(scenario.b10_years, 3));
    appendCell(row, formatNumber(scenario.reciprocal_time_years, 3));
    const typeCell = appendCell(row, SOURCE_TYPE_LABELS[scenario.source_type] || scenario.source_type, "muted-cell");
    typeCell.title = scenario.caveat;
    body.appendChild(row);
  });
  renderFigureGrid("reliability-figures", reliability.figures);
}

function renderReliabilityChart() {
  if (!state.dashboard || !state.reliabilityCurves || !window.Plotly) return;
  const field = state.reliabilityFunction;
  const meta = FUNCTION_META[field];
  const scenarios = state.dashboard.reliability.scenarios;
  const rows = state.reliabilityCurves.rows;
  const traces = scenarios
    .filter((scenario) => state.selectedScenarios.has(scenario.scenario_id))
    .map((scenario) => {
      const subset = rows.filter((row) => row.scenario_id === scenario.scenario_id && row.time_years <= meta[3]);
      return {
        type: "scatter",
        mode: "lines",
        name: scenario.label,
        x: subset.map((row) => row.time_years),
        y: subset.map((row) => row[field]),
        line: { color: RELIABILITY_COLORS[scenario.scenario_id], width: 2.4, dash: RELIABILITY_DASH[scenario.scenario_id] },
        customdata: subset.map(() => [scenario.source, SOURCE_TYPE_LABELS[scenario.source_type] || scenario.source_type]),
        hovertemplate: "%{fullData.name}<br>t = %{x:.2f} anos<br>valor = %{y:.4g}<br>%{customdata[1]}<extra></extra>",
      };
    });
  const yaxis = {
    ...plotLayout().yaxis,
    title: meta[1],
    type: meta[2] ? "log" : "linear",
  };
  if (meta[2]) {
    yaxis.dtick = 1;
    yaxis.tickformat = ".0e";
  }
  if (!meta[2]) {
    yaxis.range = [0, 1.02];
    yaxis.tickformat = ".0%";
  }
  window.Plotly.react(
    "reliability-chart",
    traces,
    plotLayout({
      showlegend: false,
      margin: { l: 72, r: 18, t: 50, b: 58 },
      title: { text: meta[0], x: 0, font: { size: 14, color: css("--ink") } },
      xaxis: { ...plotLayout().xaxis, title: "Tempo sob o cenário (anos)", range: [0, meta[3]] },
      yaxis,
    }),
    PLOT_CONFIG,
  );
}

function hydrateFmeca() {
  const fmeca = state.dashboard.fmeca;
  $("fmeca-separation").textContent = fmeca.separation_note;
  const body = $("fmeca-table").querySelector("tbody");
  body.replaceChildren();
  fmeca.components.forEach((component) => {
    const row = document.createElement("tr");
    appendCell(row, component.component);
    appendCell(row, component.function, "muted-cell");
    appendCell(row, String(component.s));
    appendCell(row, String(component.o));
    appendCell(row, String(component.d_field));
    appendCell(row, String(component.npr));
    appendCell(row, `${component.tickets_pct}%`);
    appendCell(row, `${component.energy_lost_pct}%`);
    appendCell(row, component.electrical_signature, "muted-cell");
    body.appendChild(row);
  });
}

function renderFmecaChart() {
  if (!state.dashboard || !window.Plotly) return;
  const components = [...state.dashboard.fmeca.components].sort((a, b) => a.npr - b.npr);
  const colors = [css("--amber"), css("--blue"), css("--red")];
  window.Plotly.react(
    "fmeca-chart",
    [{
      type: "bar",
      orientation: "h",
      x: components.map((item) => item.npr),
      y: components.map((item) => item.component),
      text: components.map((item) => `NPR ${item.npr}`),
      textposition: "outside",
      cliponaxis: false,
      marker: { color: colors },
      customdata: components.map((item) => [`S=${item.s}`, `O=${item.o}`, `D_campo=${item.d_field}`]),
      hovertemplate: "%{y}<br>NPR %{x}<br>%{customdata[0]} · %{customdata[1]} · %{customdata[2]}<extra></extra>",
    }],
    plotLayout({
      showlegend: false,
      margin: { l: 90, r: 54, t: 20, b: 48 },
      xaxis: { ...plotLayout().xaxis, title: "Número de Prioridade de Risco", range: [0, 355] },
      yaxis: { ...plotLayout().yaxis, title: "" },
    }),
    PLOT_CONFIG,
  );
}

function hydrateEvidence() {
  const evidence = state.dashboard.evidence;
  const rules = $("evidence-rules");
  rules.replaceChildren();
  evidence.rules.forEach((text) => {
    const item = createElement("div", "rule-item");
    const icon = document.createElement("i");
    icon.dataset.lucide = "check-circle-2";
    icon.setAttribute("aria-hidden", "true");
    item.append(icon, document.createTextNode(text));
    rules.appendChild(item);
  });

  const artifacts = $("artifact-list");
  artifacts.replaceChildren();
  evidence.artifacts.forEach((artifact) => {
    const row = createElement("div", "artifact-row");
    const name = createElement("div");
    name.append(createElement("strong", "", artifact.label), createElement("span", "", artifact.filename));
    const type = createElement("span", "", `${artifact.type} · ${formatBytes(artifact.size_bytes)}`);
    const hash = createElement("code", "", shortHash(artifact.sha256));
    hash.title = artifact.sha256;
    const link = createElement("a", "icon-button");
    link.href = artifact.url;
    link.download = artifact.filename;
    link.title = "Baixar artefato";
    link.setAttribute("aria-label", `Baixar ${artifact.label}`);
    const icon = document.createElement("i");
    icon.dataset.lucide = "download";
    icon.setAttribute("aria-hidden", "true");
    link.appendChild(icon);
    row.append(name, type, hash, link);
    artifacts.appendChild(row);
  });

  const sources = $("source-list");
  sources.replaceChildren();
  const seen = new Set();
  state.dashboard.reliability.scenarios.forEach((scenario) => {
    if (seen.has(scenario.source_artifact)) return;
    seen.add(scenario.source_artifact);
    const row = createElement("div", "source-row");
    const name = createElement("div");
    name.append(createElement("strong", "", scenario.source), createElement("span", "", scenario.source_location));
    const scope = createElement("span", "", scenario.scope);
    const hash = createElement("code", "", shortHash(scenario.source_sha256));
    hash.title = scenario.source_sha256;
    const link = createElement("a", "icon-button");
    link.href = scenario.doi ? `https://doi.org/${scenario.doi}` : "/artifacts/reliability/relatorio.md";
    if (scenario.doi) {
      link.target = "_blank";
      link.rel = "noreferrer";
    }
    link.title = scenario.doi ? "Abrir DOI" : "Abrir auditoria";
    link.setAttribute("aria-label", link.title);
    const icon = document.createElement("i");
    icon.dataset.lucide = scenario.doi ? "external-link" : "file-text";
    icon.setAttribute("aria-hidden", "true");
    link.appendChild(icon);
    row.append(name, scope, hash, link);
    sources.appendChild(row);
  });
  refreshIcons();
}

function hydrateProjectMeta() {
  const project = state.dashboard.project;
  $("dataset-link").textContent = project.dataset;
  $("dataset-link").href = `https://doi.org/${project.dataset_doi}`;
  const date = new Date(project.generated_at);
  if (!Number.isNaN(date.getTime())) {
    $("run-date").textContent = new Intl.DateTimeFormat("pt-BR", { day: "2-digit", month: "short", year: "numeric" }).format(date);
    $("run-date").dateTime = date.toISOString();
  }
}

function updateHealth(health) {
  state.health = health;
  const dot = $("system-dot");
  dot.classList.toggle("is-ok", health.status === "ok");
  dot.classList.toggle("is-error", health.status !== "ok");
  $("system-status").textContent = health.status === "ok" ? "Contratos V2 íntegros" : "Contrato indisponível";
  updateAgentStatus(health.agent);
}

function updateAgentStatus(status) {
  const label = $("agent-status");
  const names = { idle: "Em espera", loading: "Preparando pesquisa", working: "Pensando", ready: "Conectado", error: "Indisponível" };
  $("agent-status-text").textContent = names[status.state] || status.state;
  label.classList.toggle("is-ready", status.state === "ready");
  label.classList.toggle("is-loading", status.state === "loading" || status.state === "working");
  label.classList.toggle("is-error", status.state === "error");
  label.title = status.detail || `${status.provider || "Gemini"} · ${status.engine || "agente V2"}`;
}

async function warmupAgent() {
  if (state.health?.agent?.state === "ready") return state.health.agent;
  if (state.agentWarmupPromise) return state.agentWarmupPromise;
  updateAgentStatus({ state: "loading", provider: "Google Gemini" });
  state.agentWarmupPromise = fetchJSON("/api/agent/initialize", { method: "POST" })
    .then((result) => {
      state.health = { ...(state.health || {}), agent: result.agent };
      if (!state.chatRequestActive) updateAgentStatus(result.agent);
      return result.agent;
    })
    .catch((error) => {
      if (!state.chatRequestActive) updateAgentStatus({ state: "error", detail: error.message });
      state.agentWarmupPromise = null;
      throw error;
    });
  return state.agentWarmupPromise;
}

function renderChartsForView(view) {
  const renderers = {
    overview: renderTrialChart,
    autoencoder: renderSelectionChart,
    reliability: renderReliabilityChart,
    fmeca: renderFmecaChart,
  };
  if (renderers[view]) renderers[view]();
}

function initializeControls() {
  $("trial-metric-control").querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.trialMetric = button.dataset.metric;
      button.parentElement.querySelectorAll("button").forEach((item) => item.classList.toggle("is-active", item === button));
      renderTrialChart();
    });
  });
  $("reliability-function-control").querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.reliabilityFunction = button.dataset.function;
      button.parentElement.querySelectorAll("button").forEach((item) => item.classList.toggle("is-active", item === button));
      renderReliabilityChart();
    });
  });
}

function addChatMessage(role, content, images = [], renderedHtml = "") {
  const isUser = role === "user";
  const article = createElement("article", `message ${isUser ? "user-message" : "assistant-message"}`);
  article.append(createElement("div", "message-avatar", isUser ? "R" : "A"));
  const messageBody = createElement("div", "message-body");
  messageBody.append(createElement("div", "message-author", isUser ? "Rodolfo" : "ALIAdo PV"));
  const contentBody = createElement("div", "message-content");
  if (!isUser && renderedHtml) contentBody.innerHTML = renderedHtml;
  else contentBody.textContent = content;
  messageBody.append(contentBody);
  article.append(messageBody);
  if (images.length) {
    const gallery = createElement("div", "message-images");
    images.forEach((item) => {
      const image = document.createElement("img");
      image.src = item.url;
      image.alt = item.caption || "Figura do resultado";
      image.loading = "lazy";
      gallery.appendChild(image);
    });
    messageBody.appendChild(gallery);
  }
  $("chat-messages").appendChild(article);
  // Assim que Rodolfo fala, as sugestões saem de cena e a área devolvida vai
  // para a conversa. Só o turno do usuário conta: a saudação inicial do agente
  // não é conversa iniciada.
  if (isUser) document.body.classList.add("has-conversation");
  window.requestAnimationFrame(() => {
    $("chat-messages").scrollTo({ top: $("chat-messages").scrollHeight, behavior: "smooth" });
  });
  return article;
}

function addWaitingMessage(hasAttachments) {
  const article = addChatMessage("assistant", "");
  article.classList.add("is-waiting");
  const content = article.querySelector(".message-content");
  const indicator = createElement("div", "typing-indicator");
  const dots = createElement("span", "typing-dots");
  dots.setAttribute("aria-hidden", "true");
  dots.append(createElement("i"), createElement("i"), createElement("i"));
  const label = createElement("span", "typing-label", hasAttachments ? "Lendo os arquivos" : "Organizando sua pergunta");
  indicator.append(dots, label);
  content.replaceChildren(indicator);
  const phases = hasAttachments
    ? [[1800, "Relacionando os arquivos às evidências"], [7000, "Conferindo consistência e referências"]]
    : [[1400, "Consultando a base científica"], [6500, "Conferindo evidências e coerência"]];
  article.waitingTimers = phases.map(([delay, text]) => window.setTimeout(() => { label.textContent = text; }, delay));
  return article;
}

function removeWaitingMessage(article) {
  (article.waitingTimers || []).forEach((timer) => window.clearTimeout(timer));
  article.classList.add("is-leaving");
  window.setTimeout(() => article.remove(), 120);
}

function resizeChatInput(input) {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 144)}px`;
  input.style.overflowY = input.scrollHeight > 144 ? "auto" : "hidden";
}

function initializeChat() {
  const form = $("chat-form");
  const input = $("chat-input");
  const files = $("chat-files");
  resizeChatInput(input);
  if (state.chatHistory.length) {
    $("chat-messages").replaceChildren();
    state.chatHistory.forEach((item) => {
      addChatMessage(item.role, item.content, [], item.renderedHtml || "");
    });
  }
  files.addEventListener("change", () => {
    const names = Array.from(files.files).map((file) => file.name);
    $("attachment-line").textContent = names.length ? names.join(" · ") : "";
  });
  $("prompt-row").querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = button.dataset.prompt;
      form.requestSubmit();
    });
  });
  input.addEventListener("input", () => resizeChatInput(input));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;
    const selectedFiles = Array.from(files.files).slice(0, 4);
    const hasAttachments = selectedFiles.length > 0;
    const sendButton = form.querySelector("button[type='submit']");
    sendButton.disabled = true;
    input.disabled = true;
    form.classList.add("is-busy");
    state.chatRequestActive = true;
    addChatMessage("user", message);
    const historyForRequest = state.chatHistory.slice(-16).map(({ role, content }) => ({ role, content }));
    state.chatHistory.push({ role: "user", content: message });
    persistChatHistory();
    const waiting = addWaitingMessage(hasAttachments);
    updateAgentStatus({ state: "working" });
    input.value = "";
    files.value = "";
    $("attachment-line").textContent = "";
    resizeChatInput(input);
    try {
      let options;
      if (hasAttachments) {
        const data = new FormData();
        data.append("message", message);
        data.append("history", JSON.stringify(historyForRequest));
        data.append("session_id", state.sessionId);
        selectedFiles.forEach((file) => data.append("files", file));
        options = { method: "POST", body: data };
      } else {
        options = {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, history: historyForRequest, session_id: state.sessionId }),
        };
      }
      const result = await fetchJSON("/api/chat", options);
      removeWaitingMessage(waiting);
      const responseArticle = addChatMessage("assistant", result.answer, result.images || [], result.answer_html || "");
      responseArticle.dataset.route = result.route || "agent";
      state.chatHistory.push({
        role: "assistant",
        content: result.answer,
        renderedHtml: result.answer_html || "",
      });
      persistChatHistory();
      state.chatRequestActive = false;
      updateAgentStatus(result.agent || { state: "ready" });
      if (result.memories_saved) showToast(`${result.memories_saved} memória(s) validada(s)`);
    } catch (error) {
      removeWaitingMessage(waiting);
      addChatMessage("assistant", `Não foi possível concluir a resposta: ${error.message}`);
      state.chatRequestActive = false;
      updateAgentStatus({ state: "error", detail: error.message });
    } finally {
      state.chatRequestActive = false;
      sendButton.disabled = false;
      input.disabled = false;
      form.classList.remove("is-busy");
      resizeChatInput(input);
      input.focus();
    }
  });
}

async function loadApplication() {
  try {
    const [dashboard, curves, health] = await Promise.all([
      fetchJSON("/api/dashboard"),
      fetchJSON("/api/reliability/curves"),
      fetchJSON("/api/health"),
    ]);
    state.dashboard = dashboard;
    state.reliabilityCurves = curves;
    hydrateProjectMeta();
    hydrateOverview();
    hydrateAutoencoder();
    hydrateReliability();
    hydrateFmeca();
    hydrateEvidence();
    updateHealth(health);
    $("loading-state").hidden = true;
    document.querySelectorAll(".view").forEach((view) => {
      if (view.dataset.view === location.hash.replace("#", "") || (!location.hash && view.dataset.view === "agent")) {
        view.hidden = false;
      }
    });
    const activeView = document.querySelector(".view:not([hidden])")?.dataset.view || "agent";
    renderChartsForView(activeView);
    refreshIcons();
  } catch (error) {
    $("loading-state").replaceChildren();
    $("loading-state").append(createElement("span", "", `Contratos indisponíveis: ${error.message}`));
    $("system-dot").classList.add("is-error");
    $("system-status").textContent = "Falha nos contratos";
    showToast(error.message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initializeTheme();
  initializeNavigation();
  initializeFigureDialog();
  initializeControls();
  initializeChat();
  refreshIcons();
  loadApplication();
});
