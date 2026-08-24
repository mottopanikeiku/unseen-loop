const publicationUrl = "data/release-analysis.json";
const checksumLedgerPublicationSha256 = "4a38c55363a7c442c9322a7d12b49e8761cb3813746dca66ba9d1fb12ba94aa3";
const publicationUnavailable = "NOT VERIFIED";

const studyFormat = {
  integer: (value) => new Intl.NumberFormat("en-US").format(value),
  decimal: (value, digits = 3) => Number(value).toFixed(digits),
  signed: (value, digits = 3) => {
    const magnitude = Math.abs(Number(value)).toFixed(digits);
    if (value < 0) return `−${magnitude}`;
    if (value > 0) return `+${magnitude}`;
    return Number(value).toFixed(digits);
  },
  percent: (value, digits = 2) => `${(Number(value) * 100).toFixed(digits)}%`,
  points: (value, digits = 2) => `${studyFormat.signed(Number(value) * 100, digits)} pp`,
  milliseconds: (value, digits = 3) => `${(Number(value) / 1e6).toFixed(digits)} ms`,
  bytes: (value) => `${new Intl.NumberFormat("en-US").format(value)} B`,
};

function requireObject(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value;
}

function requireArray(value, name, minimumLength = 1) {
  if (!Array.isArray(value) || value.length < minimumLength) {
    throw new Error(`${name} must contain at least ${minimumLength} item(s)`);
  }
  return value;
}

function requireString(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} must be a non-empty string`);
  }
  return value;
}

function requireNumber(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function requireCount(value, name, allowZero = true) {
  if (!Number.isSafeInteger(value) || value < (allowZero ? 0 : 1)) {
    throw new Error(`${name} must be a ${allowZero ? "nonnegative" : "positive"} safe integer`);
  }
  return value;
}

function requireBoolean(value, name) {
  if (typeof value !== "boolean") throw new Error(`${name} must be boolean`);
  return value;
}

function requireDigest(value, name) {
  const digest = requireString(value, name);
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`${name} must be a lowercase SHA-256 digest`);
  }
  return digest;
}

function assertSchema(value, expected, name) {
  if (value !== expected) throw new Error(`${name} is not ${expected}`);
}

function assertClose(actual, expected, name) {
  const tolerance = Math.max(1, Math.abs(expected)) * 1e-12;
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`${name} is inconsistent with its numerator and denominator`);
  }
}

function assertRatio(record, name, rateField = "rate") {
  const ratio = requireObject(record, name);
  const numerator = requireCount(ratio.numerator, `${name}.numerator`);
  const denominator = requireCount(ratio.denominator, `${name}.denominator`, false);
  if (numerator > denominator) throw new Error(`${name}.numerator exceeds its denominator`);
  if (rateField in ratio) {
    const rate = requireNumber(ratio[rateField], `${name}.${rateField}`);
    if (rate < 0 || rate > 1) throw new Error(`${name}.${rateField} is outside [0, 1]`);
    assertClose(rate, numerator / denominator, `${name}.${rateField}`);
  }
  return ratio;
}

function assertInterval(record, name, requireEpisodeScope = true) {
  const interval = requireObject(record, name);
  const estimate = requireNumber(interval.estimate, `${name}.estimate`);
  const low = requireNumber(interval.ci95_low, `${name}.ci95_low`);
  const high = requireNumber(interval.ci95_high, `${name}.ci95_high`);
  if (low > estimate || estimate > high) throw new Error(`${name} interval does not contain its estimate`);
  requireString(interval.method, `${name}.method`);
  if (requireEpisodeScope) requireString(interval.episode_scope, `${name}.episode_scope`);
  requireCount(interval.repetitions, `${name}.repetitions`, false);
  const seed = requireNumber(interval.seed, `${name}.seed`);
  if (!Number.isInteger(seed) || seed < 0) throw new Error(`${name}.seed must be a nonnegative integer`);
  return interval;
}

function assertSummaryStats(record, name, expectedN) {
  const stats = requireObject(record, name);
  ["mean", "median", "q1", "q3", "iqr", "sd"].forEach((field) => requireNumber(stats[field], `${name}.${field}`));
  if (requireCount(stats.n, `${name}.n`, false) !== expectedN) throw new Error(`${name}.n does not match its evaluation denominator`);
  if (stats.q1 > stats.median || stats.median > stats.q3 || stats.iqr < 0 || stats.sd < 0) {
    throw new Error(`${name} contains malformed distribution statistics`);
  }
  return stats;
}

function assertDigestTree(value, path = "publication") {
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertDigestTree(item, `${path}[${index}]`));
    return;
  }
  if (!value || typeof value !== "object") return;
  Object.entries(value).forEach(([key, child]) => {
    const childPath = `${path}.${key}`;
    const digestKey = key.toLowerCase().includes("sha256") || key.toLowerCase().includes("digest");
    if (digestKey && typeof child === "string") requireDigest(child, childPath);
    else if (
      digestKey
      && child
      && typeof child === "object"
      && !Array.isArray(child)
      && Object.values(child).every((digest) => typeof digest === "string")
    ) {
      Object.entries(child).forEach(([nestedKey, digest]) => requireDigest(digest, `${childPath}.${nestedKey}`));
    } else {
      assertDigestTree(child, childPath);
    }
  });
}

function assertNamedCounts(record, name) {
  const counts = requireObject(record, name);
  Object.entries(counts).forEach(([key, value]) => requireCount(value, `${name}.${key}`));
  return counts;
}

function assertEqualCounts(observed, planned, name) {
  const observedKeys = Object.keys(observed).sort();
  const plannedKeys = Object.keys(planned).sort();
  if (observedKeys.join("|") !== plannedKeys.join("|")) throw new Error(`${name} observed/planned fields differ`);
  observedKeys.forEach((key) => {
    if (observed[key] !== planned[key]) throw new Error(`${name}.${key} is incomplete`);
  });
}

function assertPublication(data) {
  const root = requireObject(data, "publication");
  assertSchema(root.schema_version, "unseen-loop/publication-evidence-v1", "publication.schema_version");

  const analysis = requireObject(root.analysis, "publication.analysis");
  assertSchema(analysis.schema_version, "unseen-loop/release-analysis-v1", "analysis.schema_version");
  const analysisId = requireString(analysis.analysis_id, "analysis.analysis_id");
  const observedRows = assertNamedCounts(analysis.observed_rows, "analysis.observed_rows");
  const claimScope = requireObject(analysis.claim_scope, "analysis.claim_scope");
  ["causal_scope", "clear_privacy_claim", "release_label"].forEach((key) => requireString(claimScope[key], `analysis.claim_scope.${key}`));
  const statistics = requireObject(analysis.statistics, "analysis.statistics");
  Object.entries(statistics).forEach(([key, value]) => {
    if (key === "bootstrap_repetitions") requireCount(value, `analysis.statistics.${key}`, false);
    else requireString(value, `analysis.statistics.${key}`);
  });

  const evidenceIndex = requireObject(root.evidence_index, "publication.evidence_index");
  assertSchema(evidenceIndex.schema_version, "unseen-loop/evidence-index-v1", "evidence_index.schema_version");
  if (evidenceIndex.analysis_id !== analysisId) throw new Error("analysis IDs do not match");
  const sources = requireArray(evidenceIndex.sources, "evidence_index.sources");
  const sourceIds = new Set();
  sources.forEach((source, index) => {
    requireObject(source, `evidence_index.sources[${index}]`);
    const sourceId = requireString(source.study_id, `evidence_index.sources[${index}].study_id`);
    if (sourceIds.has(sourceId)) throw new Error(`duplicate source study ID ${sourceId}`);
    sourceIds.add(sourceId);
    requireString(source.backend, `evidence_index.sources[${index}].backend`);
    requireString(source.trust_label, `evidence_index.sources[${index}].trust_label`);
    requireString(source.path, `evidence_index.sources[${index}].path`);
    const observed = assertNamedCounts(source.observed, `source ${sourceId}.observed`);
    const planned = assertNamedCounts(source.planned, `source ${sourceId}.planned`);
    if (source.backend === "QUANTIZED CLEAR") {
      assertEqualCounts(observed, planned, `source ${sourceId}`);
      const failures = assertNamedCounts(source.failures, `source ${sourceId}.failures`);
      if (failures.checksum !== 0 || failures.incomplete_denominators !== 0) {
        throw new Error(`source ${sourceId} has checksum or denominator failures`);
      }
      requireCount(source.child_ledger_count, `source ${sourceId}.child_ledger_count`, false);
      if (Object.keys(requireObject(source.child_ledger_sha256, `source ${sourceId}.child_ledger_sha256`)).length !== source.child_ledger_count) {
        throw new Error(`source ${sourceId} child-ledger denominator is inconsistent`);
      }
    } else if (source.backend === "REAL FHE") {
      requireCount(source.failures, `source ${sourceId}.failures`);
    } else {
      throw new Error(`source ${sourceId} has an unknown backend`);
    }
  });
  requireArray(evidenceIndex.allowed_claims, "evidence_index.allowed_claims").forEach((claim, index) => requireString(claim, `allowed_claims[${index}]`));
  requireArray(evidenceIndex.forbidden_claims, "evidence_index.forbidden_claims").forEach((claim, index) => requireString(claim, `forbidden_claims[${index}]`));

  const environments = requireArray(root.expanded_environments, "expanded_environments");
  if (environments.length !== observedRows.expanded_environments) throw new Error("expanded environment denominator is inconsistent");
  let expandedCheckpoints = 0;
  let expandedPairs = 0;
  environments.forEach((environment, index) => {
    const name = `expanded_environments[${index}]`;
    requireObject(environment, name);
    requireString(environment.environment, `${name}.environment`);
    const studyId = requireString(environment.study_id, `${name}.study_id`);
    if (!sourceIds.has(studyId)) throw new Error(`${name} has no indexed source`);
    const checkpoints = requireCount(environment.checkpoints, `${name}.checkpoints`, false);
    const pairs = requireCount(environment.evaluation_pairs, `${name}.evaluation_pairs`, false);
    expandedCheckpoints += checkpoints;
    expandedPairs += pairs;
    if (requireArray(environment.champion_digests, `${name}.champion_digests`).length !== checkpoints) {
      throw new Error(`${name} champion denominator is inconsistent`);
    }
    ["paired_return_delta", "student_cost", "student_return", "teacher_cost", "teacher_return"].forEach((field) => {
      const record = requireObject(environment[field], `${name}.${field}`);
      if (field === "paired_return_delta") {
        assertSummaryStats(record, `${name}.${field}`, pairs);
        assertInterval(record.bootstrap, `${name}.${field}.bootstrap`, false);
        assertClose(record.mean, record.bootstrap.estimate, `${name}.${field}.bootstrap.estimate`);
      } else {
        assertSummaryStats(record, `${name}.${field}`, pairs);
      }
    });
    assertRatio(environment.action_certificate, `${name}.action_certificate`);
    assertRatio(environment.teacher_agreement, `${name}.teacher_agreement`);
    requireCount(environment.action_certificate.float_integer_mismatches, `${name}.action_certificate.float_integer_mismatches`);
    requireCount(environment.action_certificate.certified_mismatches, `${name}.action_certificate.certified_mismatches`);
  });
  if (expandedCheckpoints !== observedRows.expanded_checkpoints) throw new Error("expanded checkpoint denominator is inconsistent");
  const expandedStudyId = environments[0].study_id;
  if (environments.some((environment) => environment.study_id !== expandedStudyId)) throw new Error("expanded environments do not share one indexed study");
  const expandedSource = sources.find((source) => source.study_id === expandedStudyId);
  if (!expandedSource || expandedSource.observed.runs !== expandedCheckpoints || expandedSource.observed.paired_evaluation_episodes !== expandedPairs) {
    throw new Error("expanded source accounting does not match the publication table");
  }
  if (expandedSource.observed.long_form_evaluation_rows !== expandedPairs * 2) throw new Error("expanded long-form evaluation denominator is inconsistent");

  const cells = requireArray(root.ablation_cells, "ablation_cells");
  if (cells.length !== observedRows.ablation_cells) throw new Error("ablation cell denominator is inconsistent");
  const checkpointDenominators = new Set();
  cells.forEach((cell, index) => {
    const name = `ablation_cells[${index}]`;
    requireObject(cell, name);
    requireBoolean(cell.certificate_weighting, `${name}.certificate_weighting`);
    requireBoolean(cell.occupancy_refinement_bundle, `${name}.occupancy_refinement_bundle`);
    const studyId = requireString(cell.study_id, `${name}.study_id`);
    const source = sources.find((candidate) => candidate.study_id === studyId);
    if (!source) throw new Error(`${name} has no indexed source`);
    const checkpoints = requireCount(cell.checkpoints, `${name}.checkpoints`, false);
    const pairs = requireCount(cell.evaluation_pairs, `${name}.evaluation_pairs`, false);
    checkpointDenominators.add(checkpoints);
    if (source.observed.runs !== checkpoints || source.observed.paired_evaluation_episodes !== pairs || source.observed.long_form_evaluation_rows !== pairs * 2) {
      throw new Error(`${name} source accounting is inconsistent`);
    }
    assertInterval(cell.paired_return_delta, `${name}.paired_return_delta`);
    const selection = assertRatio(cell.champion_selection_certificate_coverage, `${name}.champion_selection_certificate_coverage`, "missing");
    assertInterval(selection.bootstrap, `${name}.champion_selection_certificate_coverage.bootstrap`);
    assertClose(selection.bootstrap.estimate, selection.numerator / selection.denominator, `${name}.champion_selection_certificate_coverage.bootstrap.estimate`);
    const heldout = assertRatio(cell.postselection_heldout_certificate_coverage, `${name}.postselection_heldout_certificate_coverage`, "missing");
    requireCount(heldout.certified_mismatches, `${name}.postselection_heldout_certificate_coverage.certified_mismatches`);
    requireString(heldout.bootstrap_not_computed, `${name}.postselection_heldout_certificate_coverage.bootstrap_not_computed`);
  });
  if (checkpointDenominators.size !== 1 || !checkpointDenominators.has(observedRows.ablation_checkpoint_contrasts)) {
    throw new Error("ablation checkpoint contrast denominator is inconsistent");
  }

  const effects = requireArray(root.ablation_effects, "ablation_effects");
  if (effects.length !== observedRows.ablation_factorial_effects) throw new Error("ablation effect denominator is inconsistent");
  const requiredEffects = new Set(["weighting_main_effect", "occupancy_refinement_bundle_main_effect", "interaction"]);
  effects.forEach((effect, index) => {
    const name = `ablation_effects[${index}]`;
    requireObject(effect, name);
    const effectName = requireString(effect.effect, `${name}.effect`);
    requiredEffects.delete(effectName);
    requireString(effect.definition, `${name}.definition`);
    requireString(effect.scope, `${name}.scope`);
    assertInterval(effect.paired_return_delta, `${name}.paired_return_delta`);
    assertInterval(effect.champion_selection_certificate_coverage, `${name}.champion_selection_certificate_coverage`);
  });
  if (requiredEffects.size) throw new Error("factorial effects are incomplete");

  const scoped = requireObject(root.scoped_fhe, "scoped_fhe");
  assertSchema(scoped.schema_version, "unseen-loop/scoped-fhe-evidence-v1", "scoped_fhe.schema_version");
  const nonlinear = requireObject(scoped.nonlinear, "scoped_fhe.nonlinear");
  const nonlinearRaw = assertNamedCounts(nonlinear.raw_accounting, "scoped_fhe.nonlinear.raw_accounting");
  if (nonlinearRaw.planned_attempts !== nonlinearRaw.observed_attempts || nonlinearRaw.failures !== 0) throw new Error("nonlinear REAL-FHE call accounting is incomplete");
  const nonlinearSource = requireObject(nonlinear.source_summary_exact, "scoped_fhe.nonlinear.source_summary_exact");
  assertSchema(nonlinearSource.schema_version, "unseen-loop/modal-nonlinear-challenge-study-v1", "nonlinear source schema");
  const challenge = requireObject(nonlinearSource.challenge_summary, "nonlinear challenge_summary");
  assertSchema(challenge.schema_version, "unseen-loop/nonlinear-fhe-challenge-v1", "nonlinear challenge schema");
  ["domain_points", "real_domain_rows", "canary_codes", "canary_repetitions_per_code", "canary_rows", "canary_distinct_request_hashes", "real_fhe_rows", "simulation_rows", "quadratic_feature_products_per_inference"].forEach((field) => requireCount(challenge[field], `challenge_summary.${field}`, false));
  if (
    challenge.domain_points !== challenge.real_domain_rows
    || challenge.canary_codes * challenge.canary_repetitions_per_code !== challenge.canary_rows
    || challenge.canary_distinct_request_hashes !== challenge.canary_rows
    || challenge.real_domain_rows + challenge.canary_rows !== challenge.real_fhe_rows
    || challenge.real_fhe_rows !== nonlinearRaw.observed_attempts
    || nonlinearSource.execution.real_fhe_calls !== challenge.real_fhe_rows
    || challenge.real_fhe_all_match !== true
    || challenge.canary_randomness_passed !== true
    || challenge.backend !== "REAL FHE"
  ) {
    throw new Error("nonlinear REAL-FHE denominators or exactness claims are inconsistent");
  }
  if (nonlinearSource.study_id !== sources.find((source) => source.study_id === nonlinearSource.study_id)?.study_id) throw new Error("nonlinear source is not indexed");

  const timing = requireObject(scoped.timing, "scoped_fhe.timing");
  const timingRaw = assertNamedCounts(timing.raw_accounting, "scoped_fhe.timing.raw_accounting");
  if (timingRaw.planned_attempts !== timingRaw.observed_attempts || timingRaw.failures !== 0 || timingRaw.measured_attempts + timingRaw.warmup_attempts !== timingRaw.observed_attempts) {
    throw new Error("timing attempt accounting is incomplete");
  }
  const timingSource = requireObject(timing.source_summary_exact, "scoped_fhe.timing.source_summary_exact");
  assertSchema(timingSource.schema_version, "unseen-loop/modal-fhe-timing-study-v1", "timing source schema");
  const timingSummary = requireObject(timingSource.timing_summary, "timing_summary");
  assertSchema(timingSummary.schema_version, "unseen-loop/timing-summary-v1", "timing_summary.schema_version");
  const denominators = requireObject(timingSummary.denominators, "timing_summary.denominators");
  ["attempted_measured_requests", "failed_measured_requests", "successful_measured_requests"].forEach((field) => requireCount(denominators[field], `timing_summary.denominators.${field}`));
  if (
    denominators.attempted_measured_requests !== timingRaw.measured_attempts
    || denominators.failed_measured_requests !== timingRaw.failures
    || denominators.successful_measured_requests + denominators.failed_measured_requests !== denominators.attempted_measured_requests
    || denominators.success_fraction !== `${denominators.successful_measured_requests}/${denominators.attempted_measured_requests}`
  ) {
    throw new Error("timing measured-request denominator is inconsistent");
  }
  const grouping = requireObject(timingSummary.grouping, "timing_summary.grouping");
  const containers = requireArray(grouping.containers, "timing_summary.grouping.containers");
  if (grouping.container_count !== containers.length || grouping.container_count !== timingRaw.distinct_containers) throw new Error("timing container denominator is inconsistent");
  const grouped = { measured: 0, successful: 0, failed: 0, warmup_excluded: 0, total: 0 };
  containers.forEach((container, index) => {
    requireObject(container, `timing container ${index}`);
    requireString(container.container_id, `timing container ${index}.container_id`);
    Object.keys(grouped).forEach((field) => {
      grouped[field] += requireCount(container[field], `timing container ${index}.${field}`);
    });
    if (container.successful + container.failed !== container.measured) throw new Error(`timing container ${index} denominator is inconsistent`);
  });
  if (grouped.measured !== timingRaw.measured_attempts || grouped.successful !== denominators.successful_measured_requests || grouped.failed !== timingRaw.failures || grouped.warmup_excluded !== timingRaw.warmup_attempts || grouped.total !== timingRaw.observed_attempts) {
    throw new Error("grouped timing denominators do not match raw accounting");
  }
  const rowCounts = assertNamedCounts(timingSummary.row_counts, "timing_summary.row_counts");
  if (rowCounts.measured !== grouped.measured || rowCounts.successful !== grouped.successful || rowCounts.failed !== grouped.failed || rowCounts.warmup_excluded !== grouped.warmup_excluded || rowCounts.total !== grouped.total) {
    throw new Error("timing row-count denominators do not match grouped accounting");
  }
  [timingSummary.timing_ns, timingSummary.byte_metrics].forEach((collection, collectionIndex) => {
    Object.entries(requireObject(collection, collectionIndex ? "timing_summary.byte_metrics" : "timing_summary.timing_ns")).forEach(([field, metric]) => {
      const name = `timing metric ${field}`;
      requireObject(metric, name);
      const n = requireCount(metric.n, `${name}.n`, false);
      const containerCount = requireCount(metric.container_count, `${name}.container_count`, false);
      const p50 = requireNumber(metric.p50, `${name}.p50`);
      const p95 = requireNumber(metric.p95, `${name}.p95`);
      if (n !== denominators.successful_measured_requests || containerCount !== containers.length || p50 > p95) throw new Error(`${name} denominator or ordering is inconsistent`);
      const confidence = requireObject(metric.confidence_interval, `${name}.confidence_interval`);
      ["p50", "p95"].forEach((quantile) => {
        const bounds = requireArray(confidence[quantile], `${name}.confidence_interval.${quantile}`, 2);
        if (bounds.length !== 2 || !bounds.every(Number.isFinite) || bounds[0] > metric[quantile] || metric[quantile] > bounds[1]) throw new Error(`${name}.${quantile} confidence interval is malformed`);
      });
      requireCount(confidence.replicates, `${name}.confidence_interval.replicates`, false);
      requireNumber(confidence.confidence_level, `${name}.confidence_interval.confidence_level`);
    });
  });
  if (timingSource.study_id !== sources.find((source) => source.study_id === timingSource.study_id)?.study_id) throw new Error("timing source is not indexed");

  assertDigestTree(root);
  return root;
}

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addCell(row, tag, content, className) {
  const cell = createElement(tag, className);
  if (tag === "th") cell.scope = "row";
  if (content instanceof Node) cell.append(content);
  else cell.textContent = content;
  row.append(cell);
  return cell;
}

function addMetric(parent, label, value, note, className = "") {
  const metric = createElement("article", `metric ${className}`.trim());
  metric.append(createElement("span", "label", label), createElement("strong", "metric-value", value), createElement("small", "", note));
  parent.append(metric);
  return metric;
}

function sourceFor(data, studyId) {
  const source = data.evidence_index.sources.find((candidate) => candidate.study_id === studyId);
  if (!source) throw new Error(`source ${studyId} disappeared after validation`);
  return source;
}

function intervalText(interval, formatter = studyFormat.signed) {
  return `[${formatter(interval.ci95_low)}, ${formatter(interval.ci95_high)}]`;
}

function outcomeClass(interval) {
  if (interval.estimate < 0) return "result-negative";
  if (interval.ci95_low <= 0 && interval.ci95_high >= 0) return "result-mixed";
  return "result-positive";
}

function outcomeLabel(interval) {
  const direction = interval.estimate < 0 ? "NEGATIVE ESTIMATE" : interval.estimate > 0 ? "POSITIVE ESTIMATE" : "ZERO ESTIMATE";
  const certainty = interval.ci95_low <= 0 && interval.ci95_high >= 0 ? "INTERVAL SPANS ZERO" : "INTERVAL EXCLUDES ZERO";
  return `${direction} · ${certainty}`;
}

function resultBlock(interval) {
  const block = createElement("div", `result-block ${outcomeClass(interval)}`);
  block.append(createElement("strong", "result-estimate", studyFormat.signed(interval.estimate)), createElement("span", "result-interval", intervalText(interval)), createElement("span", "result-kicker", outcomeLabel(interval)));
  return block;
}

function countList(record) {
  return Object.entries(record).map(([key, value]) => `${key.replaceAll("_", " ")}: ${studyFormat.integer(value)}`).join(" · ");
}

function renderCompleteness(data) {
  const grid = document.querySelector("#completeness-grid");
  const source = sourceFor(data, data.expanded_environments[0].study_id);
  grid.replaceChildren();
  addMetric(grid, "Executed runs", `${studyFormat.integer(source.observed.runs)} / ${studyFormat.integer(source.planned.runs)}`, `observed / planned · ${source.study_id}`);
  addMetric(grid, "Candidate rows", `${studyFormat.integer(source.observed.candidate_rows)} / ${studyFormat.integer(source.planned.candidate_rows)}`, "observed / planned across the expanded matrix");
  addMetric(grid, "Selection episode rows", `${studyFormat.integer(source.observed.selection_episode_rows)} / ${studyFormat.integer(source.planned.selection_episode_rows)}`, "observed / planned checkpoint-selection rows");
  addMetric(grid, "Held-out evaluation rows", studyFormat.integer(source.observed.long_form_evaluation_rows), `${studyFormat.integer(source.observed.paired_evaluation_episodes)} paired episodes · observed equals planned`);
}

function renderExpanded(data) {
  const body = document.querySelector("#expanded-body");
  body.replaceChildren();
  data.expanded_environments.forEach((environment) => {
    const row = document.createElement("tr");
    const identity = createElement("div", "study-identity");
    identity.append(createElement("strong", "", environment.environment), createElement("code", "study-id", environment.study_id));
    addCell(row, "th", identity);
    addCell(row, "td", `${studyFormat.integer(environment.checkpoints)} checkpoints · ${studyFormat.integer(environment.evaluation_pairs)} paired episodes`);
    const returns = createElement("div", "stacked-value");
    returns.append(createElement("span", "", `Student ${studyFormat.decimal(environment.student_return.mean)}`), createElement("span", "", `Teacher ${studyFormat.decimal(environment.teacher_return.mean)}`));
    addCell(row, "td", returns);
    addCell(row, "td", resultBlock(environment.paired_return_delta.bootstrap));
    const certificate = createElement("div", "stacked-value");
    certificate.append(createElement("strong", "", `${studyFormat.integer(environment.action_certificate.numerator)} / ${studyFormat.integer(environment.action_certificate.denominator)}`), createElement("span", "", `${studyFormat.percent(environment.action_certificate.rate)} certified`), createElement("span", environment.action_certificate.float_integer_mismatches > 0 ? "limit" : "property", `${studyFormat.integer(environment.action_certificate.float_integer_mismatches)} float/integer mismatches · ${studyFormat.integer(environment.action_certificate.certified_mismatches)} certified mismatches`));
    addCell(row, "td", certificate);
    const agreement = createElement("div", "stacked-value");
    agreement.append(createElement("strong", "", `${studyFormat.integer(environment.teacher_agreement.numerator)} / ${studyFormat.integer(environment.teacher_agreement.denominator)}`), createElement("span", "", studyFormat.percent(environment.teacher_agreement.rate)));
    addCell(row, "td", agreement);
    body.append(row);
  });
  const source = sourceFor(data, data.expanded_environments[0].study_id);
  const gates = document.querySelector("#expanded-gates");
  const gateCount = source.failures.suite_gate_failures;
  gates.className = `finding-callout ${gateCount > 0 ? "result-negative" : "result-positive"}`;
  gates.textContent = `${studyFormat.integer(gateCount)} suite-gate failures retained · ${studyFormat.integer(source.failures.checksum)} checksum failures · ${studyFormat.integer(source.failures.incomplete_denominators)} incomplete denominators.`;
  const bootstrap = data.expanded_environments[0].paired_return_delta.bootstrap;
  document.querySelector("#expanded-method").textContent = `${bootstrap.method}; ${studyFormat.integer(bootstrap.repetitions)} deterministic repetitions. ${data.analysis.statistics.sd}. ${data.analysis.statistics.quartiles}.`;
}

function effectTitle(effect) {
  return {
    weighting_main_effect: "Certificate weighting / main effect",
    occupancy_refinement_bundle_main_effect: "Occupancy refinement / main effect",
    interaction: "Weighting × refinement / interaction",
  }[effect] || effect.replaceAll("_", " ");
}

function renderFactorial(data) {
  const ledger = document.querySelector("#effect-ledger");
  ledger.replaceChildren();
  data.ablation_effects.forEach((effect) => {
    const article = createElement("article", `effect-entry ${outcomeClass(effect.paired_return_delta)}`);
    article.append(
      createElement("p", "label", effectTitle(effect.effect)),
      createElement("h3", "effect-number", studyFormat.signed(effect.paired_return_delta.estimate)),
      createElement("p", "effect-ci", `Return interval ${intervalText(effect.paired_return_delta)}`),
      createElement("p", "result-kicker", outcomeLabel(effect.paired_return_delta)),
      createElement("p", "effect-definition", effect.definition),
      createElement("p", "effect-certificate", `Selection-certificate effect ${studyFormat.points(effect.champion_selection_certificate_coverage.estimate)} · interval ${intervalText(effect.champion_selection_certificate_coverage, (value) => studyFormat.points(value))}`),
      createElement("p", "method-note", effect.scope),
    );
    ledger.append(article);
  });

  const body = document.querySelector("#ablation-body");
  body.replaceChildren();
  data.ablation_cells.forEach((cell) => {
    const row = document.createElement("tr");
    addCell(row, "th", cell.certificate_weighting ? "Enabled" : "Absent");
    addCell(row, "td", cell.occupancy_refinement_bundle ? "Enabled" : "Absent");
    addCell(row, "td", resultBlock(cell.paired_return_delta));
    const selection = cell.champion_selection_certificate_coverage;
    addCell(row, "td", `${studyFormat.integer(selection.numerator)} / ${studyFormat.integer(selection.denominator)} · ${studyFormat.percent(selection.bootstrap.estimate)}`);
    const heldout = cell.postselection_heldout_certificate_coverage;
    const heldoutBlock = createElement("div", "stacked-value");
    heldoutBlock.append(createElement("strong", "", `${studyFormat.integer(heldout.numerator)} / ${studyFormat.integer(heldout.denominator)}`), createElement("span", "", `${studyFormat.integer(heldout.certified_mismatches)} certified mismatches`), createElement("span", "limit", heldout.bootstrap_not_computed));
    addCell(row, "td", heldoutBlock);
    addCell(row, "td", createElement("code", "study-id", cell.study_id));
    body.append(row);
  });
  document.querySelector("#ablation-method").textContent = `${data.analysis.statistics.ablation_return_scope}. ${data.analysis.statistics.ablation_bootstrap_certificate_scope}. ${data.analysis.statistics.heldout_certificate_limit}.`;
}

function renderNonlinear(data) {
  const scoped = data.scoped_fhe.nonlinear;
  const source = scoped.source_summary_exact;
  const challenge = source.challenge_summary;
  document.querySelector("#nonlinear-scope").textContent = scoped.scope;
  const grid = document.querySelector("#challenge-grid");
  grid.replaceChildren();
  addMetric(grid, "Complete declared domain", studyFormat.integer(challenge.real_domain_rows), `${challenge.calibration_strategy} · qmax ${studyFormat.integer(source.configuration.qmax)}`, "metric-real");
  addMetric(grid, "Fresh-randomness canaries", studyFormat.integer(challenge.canary_rows), `${studyFormat.integer(challenge.canary_distinct_request_hashes)} distinct request digests · ${challenge.canary_randomness_passed ? "passed" : "failed"}`, "metric-real");
  addMetric(grid, "Exact REAL-FHE calls", `${studyFormat.integer(challenge.real_fhe_rows)} / ${studyFormat.integer(scoped.raw_accounting.planned_attempts)}`, `${challenge.real_fhe_all_match ? "all integer-clear matches" : "mismatch present"} · ${studyFormat.integer(scoped.raw_accounting.failures)} failures`, "metric-real");
  addMetric(grid, "Encrypted products / inference", studyFormat.integer(challenge.quadratic_feature_products_per_inference), challenge.circuit_family, "metric-real");
  const simulatedMetric = addMetric(grid, "Simulated semantic cross-check", `${studyFormat.integer(challenge.simulation_rows)} / ${studyFormat.integer(challenge.domain_points)}`, challenge.simulation_all_match ? "all clear compiler-semantics matches" : "mismatch present", "metric-simulated");
  simulatedMetric.querySelector(".label").prepend(createElement("span", "mode-shape sim"));

  const accounting = document.querySelector("#challenge-accounting");
  accounting.replaceChildren();
  const domain = createElement("div", "challenge-segment challenge-domain");
  domain.style.flexGrow = String(challenge.real_domain_rows);
  domain.append(createElement("span", "label", "Complete domain"), createElement("strong", "", studyFormat.integer(challenge.real_domain_rows)));
  const canary = createElement("div", "challenge-segment challenge-canary");
  canary.style.flexGrow = String(challenge.canary_rows);
  canary.append(createElement("span", "label", "Canary calls"), createElement("strong", "", studyFormat.integer(challenge.canary_rows)));
  accounting.append(domain, canary);

  const ribbon = document.querySelector("#nonlinear-source");
  ribbon.replaceChildren(createElement("span", "mode-shape real"), createElement("strong", "", source.study_id), createElement("span", "", `${challenge.backend} · ${source.execution.location} · ${source.artifact_path}`), createElement("span", "limit", source.configuration.trust_scope.local_client_remote_server_secrecy_claim ? "Remote-secrecy claim present" : "No local-client/remote-server secrecy claim"));
}

function renderTiming(data) {
  const scoped = data.scoped_fhe.timing;
  const source = scoped.source_summary_exact;
  const summary = source.timing_summary;
  const raw = scoped.raw_accounting;
  document.querySelector("#timing-scope").textContent = `${scoped.scope}. ${source.trust_scope.aggregation_claim}.`;
  const ledger = document.querySelector("#timing-ledger");
  ledger.replaceChildren();
  addMetric(ledger, "Independent contexts", studyFormat.integer(raw.distinct_containers), source.trust_scope.context_model, "metric-real");
  addMetric(ledger, "Excluded warmups", studyFormat.integer(raw.warmup_attempts), summary.method.warmups_excluded ? "excluded from the measured-request distribution" : "included", "metric-real");
  addMetric(ledger, "Measured successes", summary.denominators.success_fraction, `${studyFormat.integer(summary.denominators.successful_measured_requests)} successful measured requests`, "metric-real");
  addMetric(ledger, "Measured failures", studyFormat.integer(summary.denominators.failed_measured_requests), summary.release_quality.eligible ? "release-quality criteria satisfied" : "release-quality criteria not satisfied", "metric-real");

  const operationLabels = {
    server_evaluate_ns: "Server evaluate",
    end_to_end_ns: "End to end",
    encrypt_ns: "Client encrypt",
    decrypt_ns: "Client decrypt",
  };
  const body = document.querySelector("#timing-body");
  body.replaceChildren();
  Object.entries(operationLabels).forEach(([field, label]) => {
    const metric = summary.timing_ns[field];
    const row = document.createElement("tr");
    addCell(row, "th", label);
    addCell(row, "td", `${studyFormat.integer(metric.n)} requests · ${studyFormat.integer(metric.container_count)} contexts`);
    addCell(row, "td", studyFormat.milliseconds(metric.p50));
    addCell(row, "td", `[${studyFormat.milliseconds(metric.confidence_interval.p50[0])}, ${studyFormat.milliseconds(metric.confidence_interval.p50[1])}]`);
    addCell(row, "td", `${metric.p95_status} · ${studyFormat.milliseconds(metric.p95)}`);
    addCell(row, "td", `[${studyFormat.milliseconds(metric.confidence_interval.p95[0])}, ${studyFormat.milliseconds(metric.confidence_interval.p95[1])}]`);
    body.append(row);
  });

  const containerLedger = document.querySelector("#container-ledger");
  containerLedger.replaceChildren(createElement("h3", "subsection-title", "Independent context ledger"));
  const containerGrid = createElement("div", "container-grid");
  summary.grouping.containers.forEach((container) => {
    const entry = createElement("article", "container-entry");
    entry.append(createElement("code", "hash", container.container_id), createElement("strong", "", `${studyFormat.integer(container.successful)} / ${studyFormat.integer(container.measured)} measured successes`), createElement("span", "", `${studyFormat.integer(container.warmup_excluded)} warmups excluded · ${studyFormat.integer(container.failed)} failed`));
    containerGrid.append(entry);
  });
  containerLedger.append(containerGrid);

  const byteLabels = { request_bytes: "Request", response_bytes: "Response", evaluation_key_bytes: "Evaluation key" };
  const byteBody = document.querySelector("#byte-body");
  byteBody.replaceChildren();
  Object.entries(byteLabels).forEach(([field, label]) => {
    const metric = summary.byte_metrics[field];
    const row = document.createElement("tr");
    addCell(row, "th", label);
    addCell(row, "td", `${studyFormat.integer(metric.n)} requests · ${studyFormat.integer(metric.container_count)} contexts`);
    addCell(row, "td", studyFormat.bytes(metric.p50));
    addCell(row, "td", `${metric.p95_status} · ${studyFormat.bytes(metric.p95)}`);
    byteBody.append(row);
  });

  const ribbon = document.querySelector("#timing-source");
  ribbon.replaceChildren(createElement("span", "mode-shape real"), createElement("strong", "", source.study_id), createElement("span", "", `${source.execution.location} · ${source.artifact_path}`), createElement("span", "limit", source.trust_scope.local_client_remote_server_secrecy_claim ? "Remote-secrecy claim present" : "No local-client/remote-server secrecy claim"));
}

function renderSources(data) {
  const body = document.querySelector("#sources-body");
  body.replaceChildren();
  data.evidence_index.sources.forEach((source) => {
    const row = document.createElement("tr");
    const identity = createElement("div", "source-identity");
    identity.append(createElement("span", `mode-shape ${source.backend === "REAL FHE" ? "real" : "quant"}`), createElement("strong", "", source.backend), createElement("code", "study-id", source.study_id), createElement("span", "source-path", source.source_path));
    addCell(row, "th", identity);
    const accounting = createElement("div", "stacked-value");
    accounting.append(createElement("span", "label", "Observed"), createElement("span", "", countList(source.observed)), createElement("span", "label", "Planned"), createElement("span", "", countList(source.planned)));
    addCell(row, "td", accounting);
    const integrity = createElement("div", "stacked-value");
    integrity.append(createElement("span", "", `${studyFormat.integer(source.ledgered_files)} ledgered files`), createElement("code", "hash", source.ledger_sha256), createElement("span", "", `Source summary ${source.source_summary_sha256}`));
    if (source.backend === "QUANTIZED CLEAR") {
      const failures = Object.entries(source.failures).map(([key, value]) => `${studyFormat.integer(value)} ${key.replaceAll("_", " ")}`).join(" · ");
      integrity.append(createElement("span", Object.values(source.failures).some((value) => value > 0) ? "limit" : "property", failures), createElement("span", "", `${studyFormat.integer(source.child_ledger_count)} child ledgers`));
    } else {
      integrity.append(createElement("span", source.failures > 0 ? "limit" : "property", `${studyFormat.integer(source.failures)} failures`));
    }
    addCell(row, "td", integrity);
    addCell(row, "td", source.trust_label, source.backend === "REAL FHE" ? "property" : "limit");
    body.append(row);
  });
}

function renderClaims(data) {
  const allowed = document.querySelector("#allowed-claims");
  const forbidden = document.querySelector("#forbidden-claims");
  allowed.replaceChildren(...data.evidence_index.allowed_claims.map((claim) => createElement("li", "", claim)));
  forbidden.replaceChildren(...data.evidence_index.forbidden_claims.map((claim) => createElement("li", "", claim)));
  const scope = document.querySelector("#scope-strip");
  scope.replaceChildren(createElement("span", "label", "Causal scope"), createElement("strong", "", data.analysis.claim_scope.causal_scope), createElement("span", "label", "Clear-study privacy claim"), createElement("strong", "limit", data.analysis.claim_scope.clear_privacy_claim));
}

function renderPublication(data, digest) {
  document.querySelectorAll('[data-publication="analysis-id"]').forEach((node) => { node.textContent = data.analysis.analysis_id; });
  document.querySelectorAll('[data-publication="release-label"]').forEach((node) => { node.textContent = data.analysis.claim_scope.release_label; });
  document.querySelectorAll('[data-publication="digest"]').forEach((node) => { node.textContent = digest; });
  document.querySelectorAll('[data-publication="hero-boundary"]').forEach((node) => {
    node.textContent = `Clear-study privacy claim: ${data.analysis.claim_scope.clear_privacy_claim}. Causal scope: ${data.analysis.claim_scope.causal_scope}. Colocated FHE studies retain their source trust limits.`;
  });
  renderCompleteness(data);
  renderExpanded(data);
  renderFactorial(data);
  renderNonlinear(data);
  renderTiming(data);
  renderSources(data);
  renderClaims(data);
}

async function sha256Hex(bytes) {
  if (!window.crypto?.subtle) throw new Error("Web Crypto SHA-256 is unavailable");
  const digest = await window.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function setPublicationState(state, message) {
  document.body.dataset.publicationState = state;
  document.querySelectorAll("[data-publication-status]").forEach((node) => { node.textContent = message; });
}

function failPublication(error) {
  document.querySelectorAll("[data-publication]").forEach((node) => { node.textContent = publicationUnavailable; });
  ["#completeness-grid", "#expanded-body", "#effect-ledger", "#ablation-body", "#challenge-grid", "#challenge-accounting", "#timing-ledger", "#timing-body", "#container-ledger", "#byte-body", "#sources-body", "#allowed-claims", "#forbidden-claims", "#scope-strip"].forEach((selector) => {
    document.querySelector(selector)?.replaceChildren();
  });
  const errorNode = document.querySelector("#publication-error");
  errorNode.hidden = false;
  errorNode.textContent = "Publication evidence failed validation. No measured result is displayed.";
  setPublicationState("unavailable", "Publication unavailable — NOT VERIFIED");
  console.error("Unable to render publication evidence", error);
}

async function loadPublication() {
  setPublicationState("loading", "Verifying publication evidence…");
  try {
    const response = await fetch(publicationUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`publication HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    const digest = await sha256Hex(bytes);
    if (digest !== checksumLedgerPublicationSha256) throw new Error("publication digest does not match the source checksum ledger");
    const raw = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    const data = assertPublication(JSON.parse(raw));
    renderPublication(data, digest);
    setPublicationState("loaded", "Publication evidence verified");
  } catch (error) {
    failPublication(error);
  }
}

loadPublication();
