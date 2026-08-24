"use strict";

(() => {
  const d3 = window.d3;
  if (!d3) return;

  const registrations = new WeakMap();
  const redrawCallbacks = new Set();
  const modelDefinitions = [
    { id: "ae_denso", name: "Autoencoder Denso", color: "--blue", dash: null, symbol: d3.symbolCircle },
    { id: "ae_lstm", name: "AE-LSTM", color: "--amber", dash: "7 4", symbol: d3.symbolDiamond },
  ];
  const scenarioColors = ["--blue", "--amber", "--accent", "--rose"];

  function cssColor(token) {
    return getComputedStyle(document.documentElement).getPropertyValue(token).trim();
  }

  function number(value, digits = 3) {
    return Number(value).toLocaleString("pt-BR", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function percent(value, digits = 1) {
    return `${number(Number(value) * 100, digits)}%`;
  }

  function scientific(value) {
    return Number(value).toExponential(2).replace("e", " × 10^");
  }

  function makeTooltip(plot) {
    let tooltip = plot.querySelector(".chart-tooltip");
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.className = "chart-tooltip";
      tooltip.setAttribute("role", "tooltip");
      tooltip.hidden = true;
      plot.appendChild(tooltip);
    }
    return tooltip;
  }

  function showTooltip(tooltip, event, plot, lines) {
    const [x, y] = d3.pointer(event, plot);
    tooltip.innerHTML = lines.map((line) => `<span>${line}</span>`).join("");
    tooltip.style.left = `${Math.min(plot.clientWidth - 180, Math.max(8, x + 14))}px`;
    tooltip.style.top = `${Math.max(8, y - 18)}px`;
    tooltip.hidden = false;
  }

  function hideTooltip(tooltip) {
    tooltip.hidden = true;
  }

  function createLegend(root, definitions, active, redraw) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    legend.setAttribute("aria-label", "Séries do gráfico");
    definitions.forEach((definition) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chart-legend-button";
      button.dataset.series = definition.id;
      button.setAttribute("aria-pressed", "true");
      const swatch = document.createElement("i");
      swatch.style.setProperty("--series-color", `var(${definition.color})`);
      const label = document.createElement("span");
      label.textContent = definition.name;
      button.append(swatch, label);
      button.addEventListener("click", () => {
        if (active.has(definition.id) && active.size === 1) return;
        if (active.has(definition.id)) active.delete(definition.id);
        else active.add(definition.id);
        button.setAttribute("aria-pressed", String(active.has(definition.id)));
        redraw();
      });
      legend.appendChild(button);
    });
    root.appendChild(legend);
  }

  function register(container, redraw) {
    const previous = registrations.get(container);
    if (previous) {
      previous.observer.disconnect();
      redrawCallbacks.delete(previous.redraw);
    }
    const observer = new ResizeObserver(() => redraw());
    observer.observe(container);
    registrations.set(container, { observer, redraw });
    redrawCallbacks.add(redraw);
  }

  function installChart(container, definitions, draw, showLegend = true) {
    if (!container) return;
    container.innerHTML = "";
    const active = new Set(definitions.map((definition) => definition.id));
    const plot = document.createElement("div");
    plot.className = "chart-canvas";
    const redraw = () => draw(plot, active, definitions);
    if (showLegend) createLegend(container, definitions, active, redraw);
    container.appendChild(plot);
    register(container, redraw);
    redraw();
  }

  function scaffold(plot, height, title, description, margins = {}) {
    plot.innerHTML = "";
    const width = Math.max(300, Math.floor(plot.clientWidth || 640));
    const margin = {
      top: margins.top ?? 18,
      right: margins.right ?? 24,
      bottom: margins.bottom ?? 58,
      left: margins.left ?? 76,
    };
    const innerWidth = Math.max(120, width - margin.left - margin.right);
    const innerHeight = Math.max(100, height - margin.top - margin.bottom);
    const svg = d3.select(plot).append("svg")
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img")
      .attr("aria-label", title);
    svg.append("title").text(title);
    svg.append("desc").text(description);
    const chart = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    chart.append("rect")
      .attr("class", "chart-frame")
      .attr("width", innerWidth)
      .attr("height", innerHeight);
    return { width, height, margin, innerWidth, innerHeight, svg, chart };
  }

  function axes(frame, xScale, yScale, xLabel, yLabel, options = {}) {
    const xTicks = options.xTicks ?? (frame.width < 430 ? 4 : 6);
    const xAxis = options.xAxis || d3.axisBottom(xScale).ticks(xTicks);
    const yAxis = options.yAxis || d3.axisLeft(yScale).ticks(5);
    frame.chart.append("g")
      .attr("class", "chart-grid")
      .call(d3.axisLeft(yScale).ticks(5).tickSize(-frame.innerWidth).tickFormat(""));
    frame.chart.append("g")
      .attr("class", "chart-axis")
      .attr("transform", `translate(0,${frame.innerHeight})`)
      .call(xAxis);
    frame.chart.append("g").attr("class", "chart-axis").call(yAxis);
    frame.svg.append("text")
      .attr("class", "chart-axis-title")
      .attr("data-axis", "x")
      .attr("x", frame.margin.left + frame.innerWidth / 2)
      .attr("y", frame.height - 10)
      .attr("text-anchor", "middle")
      .text(xLabel);
    frame.svg.append("text")
      .attr("class", "chart-axis-title")
      .attr("data-axis", "y")
      .attr("transform", "rotate(-90)")
      .attr("x", -(frame.margin.top + frame.innerHeight / 2))
      .attr("y", 16)
      .attr("text-anchor", "middle")
      .text(yLabel);
  }

  function definitionFor(definitions, id) {
    return definitions.find((definition) => definition.id === id);
  }

  function metricChart(container, data) {
    const labels = {
      auc_pr: "AUC-PR",
      auc_roc: "AUC-ROC",
      sensitivity: "Sensibilidade",
      specificity: "Especificidade",
      balanced_accuracy: "Acurácia balanceada",
      mcc: "MCC",
      f1: "F1",
    };
    const order = Object.keys(labels);
    installChart(container, modelDefinitions, (plot, active, definitions) => {
      const values = [];
      definitions.filter((item) => active.has(item.id)).forEach((definition) => {
        order.forEach((metric) => {
          const item = data.metrics[definition.id][metric];
          values.push({ model: definition.id, metric, ...item });
        });
      });
      const limits = d3.extent(values.flatMap((item) => [item.ci95_low, item.ci95_high]));
      const padding = Math.max(0.025, (limits[1] - limits[0]) * 0.08);
      const x = d3.scaleLinear().domain([Math.max(-1, limits[0] - padding), Math.min(1, limits[1] + padding)]).nice().range([0, 1]);
      const frame = scaffold(plot, 410, "Desempenho macro com IC95%", "Estimativas macro por ensaio para Autoencoder Denso e AE-LSTM.", { left: 148 });
      x.range([0, frame.innerWidth]);
      const y = d3.scaleBand().domain(order).range([0, frame.innerHeight]).padding(0.3);
      axes(frame, x, y, "Estimativa macro por ensaio", "Métrica", {
        yAxis: d3.axisLeft(y).tickFormat((value) => labels[value]),
      });
      const tooltip = makeTooltip(plot);
      values.forEach((item) => {
        const definition = definitionFor(definitions, item.model);
        const offset = item.model === "ae_denso" ? -y.bandwidth() * 0.18 : y.bandwidth() * 0.18;
        const cy = y(item.metric) + y.bandwidth() / 2 + offset;
        const color = cssColor(definition.color);
        frame.chart.append("line")
          .attr("class", "chart-whisker")
          .attr("x1", x(item.ci95_low)).attr("x2", x(item.ci95_high))
          .attr("y1", cy).attr("y2", cy).attr("stroke", color);
        const symbol = d3.symbol().type(definition.symbol).size(72)();
        frame.chart.append("path")
          .attr("class", "chart-mark")
          .attr("d", symbol)
          .attr("transform", `translate(${x(item.estimate)},${cy})`)
          .attr("fill", color)
          .attr("tabindex", 0)
          .on("pointerenter focus", (event) => showTooltip(tooltip, event, plot, [
            `<strong>${definition.name}</strong>`,
            `${labels[item.metric]}: ${number(item.estimate)}`,
            `IC95%: ${number(item.ci95_low)} a ${number(item.ci95_high)}`,
          ]))
          .on("pointerleave blur", () => hideTooltip(tooltip));
      });
    });
  }

  function trialChart(container, data) {
    installChart(container, modelDefinitions, (plot, active, definitions) => {
      const experiments = [...new Set(data.trials.map((item) => item.experiment))];
      const frame = scaffold(plot, 350, "AUC-PR por ensaio", "Comparação pareada dos 14 ensaios para os dois autoencoders.", { left: 70 });
      const x = d3.scalePoint().domain(experiments).range([0, frame.innerWidth]).padding(0.35);
      const y = d3.scaleLinear().domain([0, 1]).range([frame.innerHeight, 0]);
      const step = Math.max(1, Math.ceil(experiments.length / (frame.width < 430 ? 7 : 14)));
      axes(frame, x, y, "Ensaio experimental", "AUC-PR", {
        xAxis: d3.axisBottom(x).tickValues(experiments.filter((_item, index) => index % step === 0)),
        yAxis: d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")),
      });
      const tooltip = makeTooltip(plot);
      definitions.filter((definition) => active.has(definition.id)).forEach((definition) => {
        const values = data.trials.filter((item) => item.model === definition.id);
        const color = cssColor(definition.color);
        frame.chart.append("path")
          .datum(values)
          .attr("class", "chart-series-line")
          .attr("fill", "none").attr("stroke", color)
          .attr("stroke-dasharray", definition.dash)
          .attr("d", d3.line().x((item) => x(item.experiment)).y((item) => y(item.auc_pr)));
        frame.chart.selectAll(`.trial-${definition.id}`)
          .data(values).join("circle")
          .attr("class", "chart-mark")
          .attr("cx", (item) => x(item.experiment)).attr("cy", (item) => y(item.auc_pr))
          .attr("r", 4).attr("fill", color).attr("tabindex", 0)
          .on("pointerenter focus", (event, item) => showTooltip(tooltip, event, plot, [
            `<strong>${definition.name}</strong>`,
            `${item.experiment} · ${item.mode_name}`,
            `AUC-PR: ${number(item.auc_pr)}`,
          ]))
          .on("pointerleave blur", () => hideTooltip(tooltip));
      });
    });
  }

  function discriminationChart(container, data, type) {
    const isRoc = type === "roc";
    installChart(container, modelDefinitions, (plot, active, definitions) => {
      const frame = scaffold(
        plot,
        340,
        isRoc ? "Curva ROC" : "Curva precisão-revocação",
        "Curva agregada por janela da execução de referência.",
        { left: 66 },
      );
      const x = d3.scaleLinear().domain([0, 1]).range([0, frame.innerWidth]);
      const y = d3.scaleLinear().domain([0, 1]).range([frame.innerHeight, 0]);
      axes(
        frame,
        x,
        y,
        isRoc ? "Taxa de falso positivo" : "Revocação",
        isRoc ? "Taxa de verdadeiro positivo" : "Precisão",
        {
          xAxis: d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")),
          yAxis: d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")),
        },
      );
      frame.chart.append("line")
        .attr("class", "chart-reference")
        .attr("x1", x(0)).attr("x2", x(1))
        .attr("y1", y(isRoc ? 0 : data.prevalence))
        .attr("y2", y(isRoc ? 1 : data.prevalence));
      definitions.filter((definition) => active.has(definition.id)).forEach((definition) => {
        const model = data.models[definition.id];
        const points = isRoc ? model.roc : model.precision_recall;
        frame.chart.append("path")
          .datum(points)
          .attr("class", "chart-series-line")
          .attr("fill", "none")
          .attr("stroke", cssColor(definition.color))
          .attr("stroke-dasharray", definition.dash)
          .attr("d", d3.line().x((item) => x(item[0])).y((item) => y(item[1])));
      });
      const tooltip = makeTooltip(plot);
      const guide = frame.chart.append("line").attr("class", "chart-hover-guide").style("display", "none");
      const markers = frame.chart.append("g");
      frame.chart.append("rect")
        .attr("class", "chart-hover-overlay")
        .attr("width", frame.innerWidth).attr("height", frame.innerHeight)
        .on("pointermove", (event) => {
          const xValue = Math.max(0, Math.min(1, x.invert(d3.pointer(event)[0])));
          guide.style("display", null).attr("x1", x(xValue)).attr("x2", x(xValue)).attr("y1", 0).attr("y2", frame.innerHeight);
          markers.selectAll("*").remove();
          const lines = [`<strong>${isRoc ? "Taxa de falso positivo" : "Revocação"}: ${percent(xValue)}</strong>`];
          definitions.filter((definition) => active.has(definition.id)).forEach((definition) => {
            const points = isRoc ? data.models[definition.id].roc : data.models[definition.id].precision_recall;
            const index = d3.bisector((item) => item[0]).center(points, xValue);
            const point = points[index];
            markers.append("circle").attr("class", "chart-hover-marker")
              .attr("cx", x(point[0])).attr("cy", y(point[1])).attr("r", 4)
              .attr("fill", cssColor(definition.color));
            lines.push(`${definition.name}: ${percent(point[1])}`);
          });
          showTooltip(tooltip, event, plot, lines);
        })
        .on("pointerleave", () => {
          guide.style("display", "none");
          markers.selectAll("*").remove();
          hideTooltip(tooltip);
        });
    });
  }

  function scenarioDefinitions(data) {
    return data.curve_series.map((series, index) => ({
      id: series.scenario_id,
      name: series.scenario_name,
      color: scenarioColors[index % scenarioColors.length],
      dash: series.evidence_type === "direct_bibliographic" ? "7 4" : null,
    }));
  }

  function reliabilityLineChart(container, data, key, title, yLabel) {
    const definitions = scenarioDefinitions(data);
    installChart(container, definitions, (plot, active, defs) => {
      const series = data.curve_series.filter((item) => active.has(item.scenario_id));
      const points = series.flatMap((item) => item.points);
      const xDomain = d3.extent(points, (item) => item.time_years);
      const yValues = points.map((item) => item[key]);
      const isProbability = key === "reliability" || key === "cumulative_failure_probability";
      const yDomain = isProbability ? [0, 1] : [0, d3.max(yValues) * 1.08];
      const frame = scaffold(plot, 320, title, "Cenários exponenciais de confiabilidade física em tempo de operação.", { left: 78 });
      const x = d3.scaleLinear().domain(xDomain).range([0, frame.innerWidth]);
      const y = d3.scaleLinear().domain(yDomain).nice().range([frame.innerHeight, 0]);
      axes(frame, x, y, "Tempo de operação (anos)", yLabel, {
        yAxis: isProbability
          ? d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%"))
          : d3.axisLeft(y).ticks(5).tickFormat(d3.format(".2e")),
      });
      const tooltip = makeTooltip(plot);
      const guide = frame.chart.append("line").attr("class", "chart-hover-guide").style("display", "none");
      const markers = frame.chart.append("g");
      series.forEach((item) => {
        const definition = definitionFor(defs, item.scenario_id);
        frame.chart.append("path").datum(item.points)
          .attr("class", "chart-series-line").attr("fill", "none")
          .attr("stroke", cssColor(definition.color)).attr("stroke-dasharray", definition.dash)
          .attr("d", d3.line().x((point) => x(point.time_years)).y((point) => y(point[key])));
      });
      frame.chart.append("rect").attr("class", "chart-hover-overlay")
        .attr("width", frame.innerWidth).attr("height", frame.innerHeight)
        .on("pointermove", (event) => {
          const time = Math.max(xDomain[0], Math.min(xDomain[1], x.invert(d3.pointer(event)[0])));
          guide.style("display", null).attr("x1", x(time)).attr("x2", x(time)).attr("y1", 0).attr("y2", frame.innerHeight);
          markers.selectAll("*").remove();
          const lines = [`<strong>${number(time, 2)} anos</strong>`];
          series.forEach((item) => {
            const definition = definitionFor(defs, item.scenario_id);
            const index = d3.bisector((point) => point.time_years).center(item.points, time);
            const point = item.points[index];
            markers.append("circle").attr("class", "chart-hover-marker")
              .attr("cx", x(point.time_years)).attr("cy", y(point[key])).attr("r", 4)
              .attr("fill", cssColor(definition.color));
            lines.push(`${item.scenario_name}: ${isProbability ? percent(point[key]) : scientific(point[key])}`);
          });
          showTooltip(tooltip, event, plot, lines);
        })
        .on("pointerleave", () => {
          guide.style("display", "none");
          markers.selectAll("*").remove();
          hideTooltip(tooltip);
        });
    }, false);
  }

  function reliabilityRatesChart(container, data) {
    const definitions = scenarioDefinitions(data);
    installChart(container, definitions, (plot, active, defs) => {
      const values = data.scenarios.filter((item) => active.has(item.scenario_id));
      const maximum = d3.max(values, (item) => item.lambda_per_hour);
      const frame = scaffold(plot, 300, "Comparação das taxas de falha por componente", "Taxas bibliográficas e derivadas em escala linear.", { left: 214 });
      const x = d3.scaleLinear().domain([0, maximum * 1.08]).nice().range([0, frame.innerWidth]);
      const y = d3.scaleBand().domain(values.map((item) => item.plot_label)).range([0, frame.innerHeight]).padding(0.38);
      axes(frame, x, y, "Taxa de falha λ (h⁻¹)", "Cenário", {
        xAxis: d3.axisBottom(x).ticks(5, ".1e"),
        yAxis: d3.axisLeft(y),
      });
      const tooltip = makeTooltip(plot);
      values.forEach((item) => {
        const definition = definitionFor(defs, item.scenario_id);
        frame.chart.append("path")
          .attr("class", "chart-mark")
          .attr("d", d3.symbol().type(item.evidence_type === "direct_bibliographic" ? d3.symbolDiamond : d3.symbolCircle).size(92)())
          .attr("transform", `translate(${x(item.lambda_per_hour)},${y(item.plot_label) + y.bandwidth() / 2})`)
          .attr("fill", cssColor(definition.color)).attr("tabindex", 0)
          .on("pointerenter focus", (event) => showTooltip(tooltip, event, plot, [
            `<strong>${item.plot_label}</strong>`,
            `λ: ${scientific(item.lambda_per_hour)} h⁻¹`,
            item.evidence_type === "direct_bibliographic" ? "Bibliográfica direta" : "Cenário derivado",
          ]))
          .on("pointerleave blur", () => hideTooltip(tooltip));
      });
    }, false);
  }

  function renderE3(data) {
    metricChart(document.querySelector('[data-chart="e3-metrics"]'), data);
    trialChart(document.querySelector('[data-chart="e3-trials"]'), data);
    discriminationChart(document.querySelector('[data-chart="e3-roc"]'), data.discrimination, "roc");
    discriminationChart(document.querySelector('[data-chart="e3-pr"]'), data.discrimination, "pr");
  }

  function renderReliability(data) {
    reliabilityLineChart(document.querySelector('[data-chart="reliability-r"]'), data, "reliability", "Curva de confiabilidade R(t)", "R(t)");
    reliabilityLineChart(document.querySelector('[data-chart="reliability-f"]'), data, "cumulative_failure_probability", "Curva da probabilidade acumulada de falha F(t)", "F(t)");
    reliabilityLineChart(document.querySelector('[data-chart="reliability-density"]'), data, "failure_density_per_year", "Curva da densidade de probabilidade de falha f(t)", "f(t) (ano⁻¹)");
    reliabilityLineChart(document.querySelector('[data-chart="reliability-hazard"]'), data, "hazard_per_year", "Curva da taxa de falha h(t)", "h(t) (ano⁻¹)");
    reliabilityRatesChart(document.querySelector('[data-chart="reliability-rates"]'), data);
  }

  window.ALIAdoCharts = {
    render(view, data) {
      if (view === "e3") renderE3(data);
      if (view === "reliability") renderReliability(data);
    },
    rerenderAll() {
      redrawCallbacks.forEach((redraw) => redraw());
    },
  };
})();
