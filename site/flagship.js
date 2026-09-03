"use strict";

const FLAGSHIP_URL = "data/flagship-evidence.json";
const SVG_NS = "http://www.w3.org/2000/svg";
const ACTIONS = ["BRAKE", "EAST", "WEST", "NORTH", "SOUTH"];
const FAMILIES = ["obstacle", "speed", "tilt", "battery"];
const REASONS = {
  requested_certified: "Requested action retained",
  safest_certified_alternative: "Override to certified candidate",
  emergency_action_no_certified_candidate: "Uncertified emergency brake",
  shield_disabled: "Shield disabled",
};
const state = { publication: null, shieldStep: 0, candidate: 0, timer: null, opeVariant: 0 };

function object(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} must be an object`);
  return value;
}
function array(value, name, length = null) {
  if (!Array.isArray(value) || (length !== null && value.length !== length)) throw new Error(`${name} has invalid length`);
  return value;
}
function text(value, name) {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${name} must be a non-empty string`);
  return value;
}
function number(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} must be finite`);
  return value;
}
function integer(value, name) {
  if (!Number.isInteger(value)) throw new Error(`${name} must be an integer`);
  return value;
}
function digest(value, name) {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) throw new Error(`${name} must be a SHA-256 digest`);
  return value;
}
function close(left, right, tolerance = 1e-7) {
  return Math.abs(left - right) <= tolerance * Math.max(1, Math.abs(left), Math.abs(right));
}
function setField(name, value) {
  document.querySelectorAll(`[data-field="${name}"]`).forEach((node) => { node.textContent = String(value); });
}
function setNamed(group, name, value) {
  document.querySelectorAll(`[data-${group}="${name}"]`).forEach((node) => { node.textContent = String(value); });
}
function format(value, digits = 4) {
  return typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(value)
    : "—";
}
function svg(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function validateShield(shield) {
  object(shield, "shield");
  if (shield.schema_version !== "unseen-loop/shield-publication-v1") throw new Error("unsupported shield schema");
  if (JSON.stringify(shield.state_features) !== JSON.stringify(["x", "y", "vx", "vy", "battery", "tilt"])) throw new Error("shield state order changed");
  if (shield.horizon !== 2 || JSON.stringify(shield.margin_families) !== JSON.stringify(FAMILIES)) throw new Error("shield horizon or family order changed");
  const actions = array(shield.actions, "shield.actions", 5);
  actions.forEach((action, index) => {
    object(action, `action ${index}`);
    if (action.id !== index || action.name !== ACTIONS[index]) throw new Error("shield action order changed");
    array(action.vector, `action ${index} vector`, 2).forEach((value) => integer(value, "action vector"));
  });
  const canary = object(shield.canary, "shield.canary");
  if (canary.mode !== "REAL FHE" || canary.exact_complete_domain !== true) throw new Error("shield REAL FHE canary is not exact and complete");
  digest(canary.source_sha256, "shield canary source");
  object(canary.receipt, "shield receipt");
  if (canary.receipt.security_level !== 128 || canary.receipt.domain_points !== 15625) throw new Error("shield receipt security or domain is incomplete");
  const realCall = object(object(canary.real_call, "shield real canary").call, "shield real call");
  if (realCall.output_matches_clear !== true || realCall.server_secret_key_marker_present !== false) throw new Error("shield REAL FHE call failed exactness or secret-marker evidence");
  const run = object(shield.run, "shield.run");
  if (run.mode !== "CLEAR_REFERENCE" || run.disclosure !== "CLIENT_RELEASED_DERIVED_GEOMETRY") throw new Error("shield replay trust label is invalid");
  const decisions = array(run.decisions, "shield decisions");
  decisions.forEach((decision, stepIndex) => {
    object(decision, `shield decision ${stepIndex}`);
    if (decision.step !== stepIndex) throw new Error("shield steps are not contiguous");
    const requested = integer(decision.requested_action, "requested action");
    const selected = integer(decision.selected_action, "selected action");
    if (requested < 0 || requested > 4 || selected < 0 || selected > 4) throw new Error("shield action is outside protocol order");
    digest(decision.receipt_digest, "decision receipt");
    const candidates = array(decision.candidates, "decision candidates", 5);
    candidates.forEach((candidate, candidateIndex) => {
      object(candidate, `candidate ${candidateIndex}`);
      if (candidate.action !== candidateIndex) throw new Error("candidate order changed");
      const steps = array(candidate.steps, "candidate horizons", 2);
      let observedMinimum = Infinity;
      let observedCertified = true;
      steps.forEach((horizon, horizonIndex) => {
        if (horizon.horizon !== horizonIndex + 1) throw new Error("candidate horizons changed");
        ["raw", "buffer", "buffered"].forEach((kind) => object(horizon[kind], `${kind} margins`));
        FAMILIES.forEach((family) => {
          const raw = number(horizon.raw[family], `${family} raw`);
          const buffer = number(horizon.buffer[family], `${family} buffer`);
          const buffered = number(horizon.buffered[family], `${family} buffered`);
          if (buffer < 0 || !close(raw - buffer, buffered)) throw new Error("buffered shield margin arithmetic failed");
          observedMinimum = Math.min(observedMinimum, buffered);
          observedCertified = observedCertified && buffered > 0;
        });
      });
      if (!close(observedMinimum, candidate.minimum_buffered_margin) || observedCertified !== candidate.certified) throw new Error("candidate certificate invariant failed");
    });
    const requestedCandidate = candidates[requested];
    const certified = candidates.filter((candidate) => candidate.certified);
    let expected = 0;
    if (requestedCandidate.certified) expected = requested;
    else if (certified.length) {
      expected = certified.reduce((best, candidate) => candidate.minimum_buffered_margin > best.minimum_buffered_margin ? candidate : best).action;
    }
    if (selected !== expected) throw new Error("client shield selection does not replay");
    if (Boolean(decision.selected_certified) !== candidates[selected].certified) throw new Error("selected certificate flag disagrees");
    object(decision.visualization, "client-released visualization");
  });
  const summary = object(shield.summary, "shield.summary");
  const total = integer(summary.total_steps, "shield total");
  if (total !== decisions.length || summary.requested_retained + summary.override_to_certified + summary.emergency_brake !== total) throw new Error("shield accounting does not close");
}

function validateOPE(ope) {
  object(ope, "ope");
  if (ope.schema_version !== "unseen-loop/ope-publication-v1") throw new Error("unsupported OPE schema");
  const spec = object(object(ope.batch, "ope.batch").trajectory_spec, "trajectory spec");
  const horizon = integer(spec.horizon, "OPE horizon");
  integer(spec.trajectories, "OPE trajectories");
  ["states", "actions", "rewards", "behavior_propensities"].forEach((field) => {
    if (Object.hasOwn(ope.batch, field)) throw new Error(`private OPE field ${field} was published`);
  });
  const variant = object(ope.variant, "OPE variant");
  if (variant.mode !== "REAL FHE (approximate arithmetic)" || variant.disclosure !== "CLIENT_RELEASED_STATISTICS") throw new Error("OPE execution or disclosure label is invalid");
  if (variant.matches_exact_clear !== false) throw new Error("approximate CKKS result cannot claim exact agreement");
  const statistics = object(variant.statistics, "OPE statistics");
  const vectors = ["numerators", "denominators", "counts"].map((name) => array(statistics[name], name, horizon));
  vectors[0].forEach((value) => number(value, "OPE numerator"));
  vectors[1].forEach((value) => { if (number(value, "OPE denominator") <= 0) throw new Error("OPE denominator is not positive"); });
  vectors[2].forEach((value) => integer(value, "OPE count"));
  const estimate = vectors[0].reduce((sum, value, index) => sum + value / vectors[1][index], 0);
  if (!close(estimate, number(statistics.estimate, "OPE estimate"), 1e-6)) throw new Error("client OPE division does not replay");
  const receipt = object(variant.receipt, "OPE receipt");
  digest(receipt.source_sha256, "OPE source");
  const context = object(receipt.context, "CKKS context receipt");
  if (context.effective_security_level !== "tc128" || context.security_enforced !== true || context.server_context_is_private !== false) throw new Error("CKKS security evidence failed");
  const exact = object(ope.exact_canary, "exact OPE canary");
  if (exact.mode !== "REAL FHE" || exact.simulation_matches_real !== true) throw new Error("exact OPE canary failed");
  if (object(exact.call, "exact OPE call").output_matches_integer_reference !== true) throw new Error("exact OPE integer agreement failed");
  digest(exact.source_sha256, "exact OPE source");
  const uncertainty = object(ope.uncertainty, "OPE uncertainty");
  if (uncertainty.mode !== "CLEAR REFERENCE" || uncertainty.method !== "trajectory_percentile_bootstrap") throw new Error("bootstrap provenance is invalid");
  if (!(number(uncertainty.lower, "bootstrap lower") <= uncertainty.estimate && uncertainty.estimate <= number(uncertainty.upper, "bootstrap upper"))) throw new Error("bootstrap interval does not contain estimate");
}

function validateSmoke(smoke) {
  object(smoke, "smoke");
  if (smoke.schema_version !== "unseen-loop/flagship-smoke-publication-v1") throw new Error("unsupported smoke publication schema");
  text(smoke.run_id, "smoke run id");
  digest(smoke.evidence_index_sha256, "smoke evidence index");
  digest(smoke.analysis_sha256, "smoke analysis");
  const planned = integer(smoke.planned_jobs, "smoke planned jobs");
  const counts = object(smoke.status_counts, "smoke status counts");
  const succeeded = integer(counts.succeeded, "smoke succeeded jobs");
  const rejected = integer(counts.rejected, "smoke rejected jobs");
  if (succeeded + rejected !== planned) throw new Error("smoke terminal accounting does not close");
  const gatePass = object(smoke.gate_pass, "smoke gate status");
  Object.values(gatePass).forEach((value) => {
    if (typeof value !== "boolean") throw new Error("smoke gate status must be boolean");
  });
  const summary = object(smoke.evidence_summary, "smoke evidence summary");
  if (object(summary.closure, "smoke closure").planned_jobs !== planned) throw new Error("smoke closure denominator mismatch");
  object(summary.clear_shield, "smoke clear shield");
  object(summary.shield_fhe, "smoke shield FHE");
  object(summary.ope, "smoke OPE");
  object(summary.integration, "smoke integration");
  object(summary.systems, "smoke systems");
}

function gateObserved(section, name) {
  const gate = array(section.gates, `${name} gates`).find((item) => item.name === name);
  if (!gate) throw new Error(`missing smoke gate ${name}`);
  return number(gate.observed, `smoke gate ${name}`);
}


function validatePublication(publication) {
  object(publication, "publication");
  if (publication.schema_version !== "unseen-loop/flagship-publication-v1") throw new Error("unsupported flagship publication schema");
  text(object(publication.release, "release").release_id, "release id");
  validateShield(publication.shield);
  validateOPE(publication.ope);
  validateSmoke(publication.smoke);
  array(publication.allowed_claims, "allowed claims").forEach((claim) => text(claim, "allowed claim"));
  array(publication.forbidden_claims, "forbidden claims").forEach((claim) => text(claim, "forbidden claim"));
  return publication;
}

function warehousePoint(position, safety) {
  const xBounds = safety.x_bounds;
  const yBounds = safety.y_bounds;
  return [60 + ((position[0] - xBounds[0]) / (xBounds[1] - xBounds[0])) * 600, 420 - ((position[1] - yBounds[0]) / (yBounds[1] - yBounds[0])) * 360];
}

function renderWarehouse(decision, shield) {
  const layer = document.querySelector("[data-warehouse-layer]");
  if (!layer) return;
  layer.replaceChildren();
  const safety = shield.scenario.safety;
  layer.append(svg("rect", { x: 60, y: 60, width: 600, height: 360, class: "warehouse-bound" }));
  for (let index = 1; index < 10; index += 1) {
    layer.append(svg("line", { x1: 60 + index * 60, y1: 60, x2: 60 + index * 60, y2: 420, class: "warehouse-grid" }));
  }
  for (let index = 1; index < 6; index += 1) {
    layer.append(svg("line", { x1: 60, y1: 60 + index * 60, x2: 660, y2: 60 + index * 60, class: "warehouse-grid" }));
  }
  safety.obstacles.forEach((obstacle) => {
    const [x, y] = warehousePoint([obstacle.x, obstacle.y], safety);
    const radius = obstacle.radius / (safety.x_bounds[1] - safety.x_bounds[0]) * 600;
    layer.append(svg("circle", { cx: x, cy: y, r: radius, class: "warehouse-obstacle" }));
  });
  const current = warehousePoint(decision.visualization.current_position, safety);
  decision.visualization.candidate_paths.forEach((path) => {
    const points = [current, ...path.positions.map((position) => warehousePoint(position, safety))];
    const classes = ["candidate-path"];
    if (path.action === decision.requested_action) classes.push("requested");
    if (path.action === decision.selected_action) classes.push("selected");
    if (!decision.candidates[path.action].certified) classes.push("failed");
    layer.append(svg("polyline", { points: points.map((point) => point.join(",")).join(" "), class: classes.join(" ") }));
    points.slice(1).forEach((point, index) => {
      const marker = svg("circle", { cx: point[0], cy: point[1], r: path.action === decision.selected_action ? 7 : 4, class: classes.join(" ") });
      const title = svg("title"); title.textContent = `${ACTIONS[path.action]} horizon ${index + 1}`; marker.append(title); layer.append(marker);
    });
  });
  layer.append(svg("circle", { cx: current[0], cy: current[1], r: 9, fill: "var(--ink)" }));
}

function renderMargin(candidate) {
  const figure = document.querySelector("[data-margin-figure]");
  const rows = document.querySelector("[data-margin-rows]");
  if (!figure || !rows) return;
  figure.replaceChildren(); rows.replaceChildren();
  const values = candidate.steps.flatMap((step) => FAMILIES.map((family) => Math.abs(step.buffered[family])));
  const extent = Math.max(1, ...values);
  candidate.steps.forEach((step) => FAMILIES.forEach((family) => {
    const row = document.createElement("div"); row.className = "margin-row";
    const label = document.createElement("span"); label.textContent = `H${step.horizon} ${family}`;
    const track = document.createElement("div"); track.className = "margin-track";
    const marker = document.createElement("i"); marker.style.left = `${50 + 48 * step.buffered[family] / extent}%`; track.append(marker);
    const value = document.createElement("code"); value.textContent = format(step.buffered[family]);
    row.append(label, track, value); figure.append(row);
  }));
  FAMILIES.forEach((family) => {
    const tr = document.createElement("tr");
    const cells = [family, candidate.steps[0].raw[family], candidate.steps[0].buffer[family], candidate.steps[0].buffered[family], candidate.steps[1].raw[family], candidate.steps[1].buffer[family], candidate.steps[1].buffered[family]];
    cells.forEach((value) => { const td = document.createElement("td"); td.textContent = typeof value === "number" ? format(value) : value; tr.append(td); });
    rows.append(tr);
  });
}

function renderShield() {
  const shield = state.publication.shield;
  const decisions = shield.run.decisions;
  const decision = decisions[state.shieldStep];
  const stepInput = document.querySelector("[data-shield-step]");
  if (stepInput) { stepInput.disabled = false; stepInput.max = String(decisions.length - 1); stepInput.value = String(state.shieldStep); }
  const stepOutput = document.querySelector("[data-shield-step-output]");
  if (stepOutput) stepOutput.textContent = `${state.shieldStep + 1} / ${decisions.length}`;
  setNamed("decision", "step", `${state.shieldStep + 1} / ${decisions.length}`);
  setNamed("decision", "requested", ACTIONS[decision.requested_action]);
  setNamed("decision", "selected", ACTIONS[decision.selected_action]);
  setNamed("decision", "reason", REASONS[decision.reason] || decision.reason);
  setNamed("decision", "certified", decision.selected_certified ? "YES" : "NO");
  setNamed("decision", "minimum", format(decision.candidates[decision.selected_action].minimum_buffered_margin));
  setNamed("decision", "digest", decision.receipt_digest);
  renderWarehouse(decision, shield);
  const rows = document.querySelector("[data-candidate-rows]");
  if (rows) {
    rows.replaceChildren();
    decision.candidates.forEach((candidate) => {
      const tr = document.createElement("tr");
      const button = document.createElement("button"); button.type = "button"; button.className = "candidate-button"; button.textContent = `Inspect ${ACTIONS[candidate.action]}`; button.setAttribute("aria-pressed", String(state.candidate === candidate.action));
      button.addEventListener("click", () => { state.candidate = candidate.action; renderShield(); });
      const failed = candidate.failed_obligations.map(([horizon, family]) => `H${horizon} ${family}`).join(", ") || "None";
      const values = [button, `${ACTIONS[candidate.action]} / (${shield.actions[candidate.action].vector.join(", ")})`, candidate.action === decision.selected_action ? "SELECTED" : candidate.action === decision.requested_action ? "REQUESTED" : "—", format(candidate.minimum_buffered_margin), candidate.certified ? "CERTIFIED" : "FAILED", failed];
      values.forEach((value, index) => { const td = document.createElement("td"); if (value instanceof Node) td.append(value); else td.textContent = value; if (index === 4) td.className = candidate.certified ? "certificate-pass" : "certificate-fail"; tr.append(td); });
      rows.append(tr);
    });
  }
  const caption = document.querySelector("[data-margin-caption]");
  if (caption) caption.textContent = `${ACTIONS[state.candidate]} candidate margins`;
  renderMargin(decision.candidates[state.candidate]);
  document.querySelectorAll("[data-step-strip]").forEach((strip) => {
    strip.replaceChildren();
    decisions.forEach((item, index) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "step-button";
      if (item.emergency_fallback) button.classList.add("emergency"); else if (item.requested_action !== item.selected_action) button.classList.add("override");
      button.textContent = `${index + 1} ${ACTIONS[item.selected_action].slice(0, 2)}`; button.setAttribute("aria-label", `Step ${index + 1}: requested ${ACTIONS[item.requested_action]}, selected ${ACTIONS[item.selected_action]}`); button.setAttribute("aria-pressed", String(index === state.shieldStep));
      button.addEventListener("click", () => { state.shieldStep = index; state.candidate = item.selected_action; renderShield(); }); strip.append(button);
    });
  });
}

function renderOPEPlot(statistics, selectedHorizon) {
  const layer = document.querySelector("[data-ope-plot-layer]"); if (!layer) return;
  layer.replaceChildren();
  const contributions = statistics.numerators.map((value, index) => value / statistics.denominators[index]);
  const cumulative = []; contributions.reduce((sum, value) => { cumulative.push(sum + value); return sum + value; }, 0);
  const extent = Math.max(0.01, ...contributions.map(Math.abs), ...cumulative.map(Math.abs));
  const y = (value) => 210 - value / extent * 160;
  layer.append(svg("line", { x1: 55, y1: 210, x2: 870, y2: 210, class: "plot-axis" }));
  const points = [];
  contributions.forEach((value, index) => {
    const x = 75 + index * (780 / contributions.length); const width = 0.55 * (780 / contributions.length);
    layer.append(svg("rect", { x, y: Math.min(210, y(value)), width, height: Math.max(1, Math.abs(y(value) - 210)), class: "plot-bar", opacity: index < selectedHorizon ? 1 : .25 }));
    points.push([x + width / 2, y(cumulative[index])]);
  });
  layer.append(svg("polyline", { points: points.map((point) => point.join(",")).join(" "), class: "plot-line" }));
  points.forEach((point, index) => layer.append(svg("circle", { cx: point[0], cy: point[1], r: index < selectedHorizon ? 6 : 3, class: "plot-point" })));
}

function renderOPE() {
  const ope = state.publication.ope; const variant = ope.variant; const statistics = variant.statistics; const horizon = statistics.numerators.length;
  setNamed("ope-scope", "trajectories", ope.batch.trajectory_spec.trajectories); setNamed("ope-scope", "horizon", horizon);
  setNamed("estimate", "label", statistics.estimator); setNamed("estimate", "value", format(statistics.estimate)); setNamed("estimate", "clip", format(variant.clip_threshold)); setNamed("estimate", "mode", variant.mode);
  const range = document.querySelector("[data-ope-horizon]"); const selected = range ? Number(range.value) : horizon;
  if (range) { range.disabled = false; range.max = String(horizon); if (Number(range.value) > horizon) range.value = String(horizon); }
  const output = document.querySelector("[data-ope-horizon-output]"); if (output) output.textContent = `${selected} / ${horizon}`;
  const rows = document.querySelector("[data-ope-stat-rows]"); if (rows) {
    rows.replaceChildren(); let cumulative = 0;
    statistics.numerators.forEach((numerator, index) => {
      const contribution = numerator / statistics.denominators[index]; cumulative += contribution;
      const tr = document.createElement("tr"); [index + 1, numerator, statistics.denominators[index], statistics.counts[index], contribution, cumulative, variant.mode].forEach((value) => { const td = document.createElement("td"); td.textContent = typeof value === "number" ? format(value) : value; tr.append(td); }); rows.append(tr);
    });
  }
  renderOPEPlot(statistics, selected);
  const uncertainty = ope.uncertainty; ["lower", "estimate", "upper"].forEach((name) => setNamed("uncertainty", name, format(uncertainty[name])));
  setNamed("uncertainty", "method", `${uncertainty.mode} / ${uncertainty.method} / ${uncertainty.samples} resamples / seed ${uncertainty.seed}`);
  const span = uncertainty.upper - uncertainty.lower || 1; const center = (uncertainty.estimate - uncertainty.lower) / span * 100;
  const point = document.querySelector(".interval-point"); if (point) point.style.left = `${Math.max(0, Math.min(100, center))}%`;
  const diagnostics = ope.diagnostics; setNamed("diagnostic", "ess", format(Math.min(...diagnostics.per_horizon_ess))); setNamed("diagnostic", "support", diagnostics.support_failures); setNamed("diagnostic", "clipped", `${format(100 * diagnostics.clipped_fraction, 2)}%`);
  setNamed("receipt", "depth", variant.receipt.computation.required_multiplicative_depth); setNamed("receipt", "security", variant.receipt.context.effective_security_level); setNamed("receipt", "agreement", ope.exact_canary.call.output_matches_integer_reference ? "EXACT CANARY PASS" : "FAILED");
}

function initialize(publication) {
  state.publication = publication; document.body.dataset.evidenceState = "loaded";
  document.querySelectorAll("[data-flagship-status]").forEach((node) => { node.textContent = `VERIFIED COPY · ${publication.release.release_id}`; });
  setField("shield-mode", publication.shield.canary.mode); setField("shield-disclosure", publication.shield.run.disclosure.replaceAll("_", " ")); setField("shield-source", publication.shield.canary.source_sha256); setField("shield-receipt", publication.shield.canary.receipt.spec_digest);
  setNamed("accounting", "retained", publication.shield.summary.requested_retained); setNamed("accounting", "override", publication.shield.summary.override_to_certified); setNamed("accounting", "emergency", publication.shield.summary.emergency_brake); setNamed("accounting", "total", publication.shield.summary.total_steps);
  setField("ope-mode", publication.ope.variant.mode); setField("ope-policy", publication.ope.target_policy.policy_sha256); setField("ope-receipt", publication.ope.variant.receipt.source_sha256);
  const smoke = publication.smoke; const smokeSummary = smoke.evidence_summary;
  setField("smoke-run", smoke.run_id); setField("smoke-index", smoke.evidence_index_sha256);
  setNamed("smoke", "planned", smoke.planned_jobs); setNamed("smoke", "terminal", smoke.status_counts.succeeded + smoke.status_counts.rejected); setNamed("smoke", "succeeded", smoke.status_counts.succeeded); setNamed("smoke", "rejected", smoke.status_counts.rejected);
  setNamed("smoke", "status", Object.values(smoke.gate_pass).every(Boolean) ? "CLOSED · ALL GATES PASS" : "CLOSED · GATES FAILED");
  setNamed("smoke", "shield-margins", `${smokeSummary.shield_fhe.accounting.margin_matches} / ${smokeSummary.shield_fhe.accounting.decoded_margins}`);
  setNamed("smoke", "ope-bias", format(gateObserved(smokeSummary.ope, "maximum_normalized_bias"), 3));
  setNamed("smoke", "return-gap", format(smokeSummary.integration.pooled_discrepancy.return, 3));
  setNamed("smoke", "timing", `${smokeSummary.systems.successful_measured_requests} / ${smokeSummary.systems.measured_request_denominator}`);
  const step = document.querySelector("[data-shield-step]"); if (step) step.addEventListener("input", () => { state.shieldStep = Number(step.value); state.candidate = publication.shield.run.decisions[state.shieldStep].selected_action; renderShield(); });
  document.querySelector("[data-shield-prev]")?.addEventListener("click", () => { state.shieldStep = (state.shieldStep - 1 + publication.shield.run.decisions.length) % publication.shield.run.decisions.length; renderShield(); });
  document.querySelector("[data-shield-next]")?.addEventListener("click", () => { state.shieldStep = (state.shieldStep + 1) % publication.shield.run.decisions.length; renderShield(); });
  document.querySelector("[data-shield-play]")?.addEventListener("click", (event) => { if (state.timer) { clearInterval(state.timer); state.timer = null; event.currentTarget.textContent = "Play replay"; } else { state.timer = setInterval(() => { state.shieldStep = (state.shieldStep + 1) % publication.shield.run.decisions.length; renderShield(); }, 1200); event.currentTarget.textContent = "Pause replay"; } });
  const opeRange = document.querySelector("[data-ope-horizon]"); if (opeRange) { opeRange.value = String(publication.ope.variant.statistics.numerators.length); opeRange.addEventListener("input", renderOPE); }
  if (document.body.dataset.page === "shield") { state.candidate = publication.shield.run.decisions[0].selected_action; renderShield(); }
  if (document.body.dataset.page === "ope") renderOPE();
}

function failClosed(error) {
  document.body.dataset.evidenceState = "unavailable";
  document.querySelectorAll("[data-flagship-status]").forEach((node) => { node.textContent = `EVIDENCE UNAVAILABLE · ${error.message}`; });
  console.error(error);
}

async function load() {
  try {
    const expected = document.body.dataset.flagshipSha256;
    digest(expected, "pinned publication digest");
    const response = await fetch(FLAGSHIP_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`publication request returned ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const observed = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))).map((value) => value.toString(16).padStart(2, "0")).join("");
    if (observed !== expected) throw new Error("publication byte digest mismatch");
    initialize(validatePublication(JSON.parse(new TextDecoder().decode(bytes))));
  } catch (error) { failClosed(error instanceof Error ? error : new Error("unknown flagship validation failure")); }
}

load();
