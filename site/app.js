const evidenceUrl = "data/evidence.json";
const notMeasured = "NOT MEASURED";

const format = {
  integer: (value) => new Intl.NumberFormat("en-US").format(value),
  number: (value, digits = 0) => Number(value).toFixed(digits),
  percent: (value, digits = 1) => `${(value * 100).toFixed(digits)}%`,
  ms: (nanoseconds, digits = 2) => `${(nanoseconds / 1e6).toFixed(digits)} ms`,
  bytes: (value) => `${new Intl.NumberFormat("en-US").format(value)} B`,
  compactHash: (value) => `${value.slice(0, 8)}…${value.slice(-5)}`,
  shape: (value) => `[${value.join(" × ")}]`,
  probability: (value) => value.toExponential(0).replace("e", " × 10^"),
};

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor(ordered.length / 2)];
}

function setField(name, value) {
  document.querySelectorAll(`[data-field="${name}"]`).forEach((node) => {
    node.textContent = value;
  });
}

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

function requireNumber(value, name) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function requireString(value, name) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} must be a non-empty string`);
  }
  return value;
}

function requireDigest(value, name) {
  const digest = requireString(value, name);
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`${name} must be a lowercase SHA-256 digest`);
  }
  return digest;
}

function assertEvidence(data) {
  const root = requireObject(data, "evidence");
  if (root.schema_version !== "unseen-loop/modal-evidence-v2") {
    throw new Error("served evidence is not modal-evidence-v2");
  }
  requireString(root.run_id, "run_id");
  const gpu = requireObject(root.gpu_training, "gpu_training");
  const search = requireObject(root.clear_search, "clear_search");
  const receipt = requireObject(root.circuit_receipt, "circuit_receipt");
  const client = requireObject(root.client, "client");
  const envelope = requireObject(root.authenticated_envelope_protocol, "authenticated_envelope_protocol");
  const audit = requireObject(root.artifact_secret_marker_audit, "artifact_secret_marker_audit");
  const canary = requireObject(root.same_input_canary, "same_input_canary");
  const bundle = requireObject(root.nonsecret_bundle, "nonsecret_bundle");
  const loop = requireObject(root.closed_loop_real_fhe, "closed_loop_real_fhe");
  const trials = requireArray(root.real_fhe_trials, "real_fhe_trials", 2);
  const trajectory = requireArray(loop.trajectory, "closed_loop_real_fhe.trajectory", 1);

  [
    [gpu.population, "gpu_training.population"],
    [gpu.iterations, "gpu_training.iterations"],
    [gpu.episodes_per_candidate, "gpu_training.episodes_per_candidate"],
    [gpu.best_vectorized_return, "gpu_training.best_vectorized_return"],
    [search.champion_return_mean, "clear_search.champion_return_mean"],
    [search.teacher_return_mean, "clear_search.teacher_return_mean"],
    [search.certified_coverage, "clear_search.certified_coverage"],
    [search.box_certificate_coverage, "clear_search.box_certificate_coverage"],
    [search.box_certificate_points, "clear_search.box_certificate_points"],
    [client.keygen_ns, "client.keygen_ns"],
    [receipt.security_level, "circuit_receipt.security_level"],
    [receipt.global_p_error, "circuit_receipt.global_p_error"],
    [receipt.server_artifact_bytes, "circuit_receipt.server_artifact_bytes"],
    [receipt.maximum_integer_bit_width, "circuit_receipt.maximum_integer_bit_width"],
    [receipt.complexity, "circuit_receipt.complexity"],
    [receipt.compile_ns, "circuit_receipt.compile_ns"],
    [canary.repetitions, "same_input_canary.repetitions"],
    [canary.distinct_ciphertexts, "same_input_canary.distinct_ciphertexts"],
    [loop.requested_steps, "closed_loop_real_fhe.requested_steps"],
    [loop.completed_steps, "closed_loop_real_fhe.completed_steps"],
    [loop.exact_matches, "closed_loop_real_fhe.exact_matches"],
    [loop.certified_steps, "closed_loop_real_fhe.certified_steps"],
    [loop.return, "closed_loop_real_fhe.return"],
  ].forEach(([value, name]) => requireNumber(value, name));

  [
    [client.location, "client.location"],
    [gpu.device_name, "gpu_training.device_name"],
    [receipt.backend, "circuit_receipt.backend"],
    [receipt.concrete_python_version, "circuit_receipt.concrete_python_version"],
    [receipt.policy_digest, "circuit_receipt.policy_digest"],
    [receipt.server_artifact_sha256, "circuit_receipt.server_artifact_sha256"],
    [receipt.client_specs_sha256, "circuit_receipt.client_specs_sha256"],
    [receipt.mlir_sha256, "circuit_receipt.mlir_sha256"],
    [envelope.name, "authenticated_envelope_protocol.name"],
    [envelope.authentication_algorithm, "authenticated_envelope_protocol.authentication_algorithm"],
    [envelope.request_schema_version, "authenticated_envelope_protocol.request_schema_version"],
    [envelope.response_schema_version, "authenticated_envelope_protocol.response_schema_version"],
    [envelope.replay_protection, "authenticated_envelope_protocol.replay_protection"],
    [bundle.volume_path, "nonsecret_bundle.volume_path"],
    [bundle.evidence_path, "nonsecret_bundle.evidence_path"],
    [bundle.checksums_path, "nonsecret_bundle.checksums_path"],
  ].forEach(([value, name]) => requireString(value, name));
  requireArray(receipt.input_shape, "circuit_receipt.input_shape");
  requireArray(receipt.integer_output_bound, "circuit_receipt.integer_output_bound");
  requireArray(envelope.bindings, "authenticated_envelope_protocol.bindings");
  requireArray(bundle.files, "nonsecret_bundle.files");
  const requiredBindings = [
    "policy_digest",
    "circuit_digest",
    "client_context_digest",
    "evaluation_key_digest",
    "request_digest",
  ];
  const requiredBundleFiles = [
    "checksums.sha256",
    "client-specs.bin",
    "evidence.json",
    "policy.json",
    "receipt.json",
    "server.zip",
  ];
  if (
    requiredBindings.some((name) => !envelope.bindings.includes(name))
    || requiredBundleFiles.some((name) => !bundle.files.includes(name))
  ) {
    throw new Error("served evidence omits required protocol bindings or bundle files");
  }

  if (
    client.secret_key_sent_to_modal !== false
    || loop.server_received_plaintext_observation !== false
    || audit.server_secret_key_marker_present !== false
    || canary.all_match !== true
    || canary.repetitions !== trials.length
    || canary.distinct_ciphertexts !== new Set(trials.map((row) => row.request_sha256)).size
  ) {
    throw new Error("served evidence does not preserve the declared privacy boundary");
  }
  if (
    loop.requested_steps !== 25
    || loop.completed_steps !== 25
    || loop.exact_matches !== 25
    || trajectory.length !== loop.completed_steps
  ) {
    throw new Error("served evidence does not contain the preregistered 25-step exact loop");
  }
  for (const [index, row] of [...trials, ...trajectory].entries()) {
    requireObject(row, `REAL-FHE row ${index}`);
    const protocol = requireObject(row.protocol, `REAL-FHE row ${index}.protocol`);
    if (
      row.mode !== "REAL FHE"
      || row.matches_integer_clear !== true
      || row.server_secret_key_marker_present !== false
      || protocol.name !== envelope.name
      || protocol.authentication_algorithm !== envelope.authentication_algorithm
      || protocol.request_schema_version !== envelope.request_schema_version
      || protocol.response_schema_version !== envelope.response_schema_version
      || protocol.policy_digest !== receipt.policy_digest
      || protocol.circuit_digest !== receipt.server_artifact_sha256
      || protocol.client_context_digest !== receipt.client_specs_sha256
    ) {
      throw new Error(`REAL-FHE row ${index} violates the authenticated evidence boundary`);
    }
    [
      "server_evaluate_ns",
      "encrypt_ns",
      "decrypt_ns",
      "request_bytes",
      "response_bytes",
      "evaluation_key_bytes",
    ].forEach((field) => requireNumber(row[field], `REAL-FHE row ${index}.${field}`));
    [
      [row.request_sha256, "request_sha256"],
      [row.response_sha256, "response_sha256"],
      [protocol.request_envelope_digest, "protocol.request_envelope_digest"],
      [protocol.response_envelope_digest, "protocol.response_envelope_digest"],
      [protocol.evaluation_key_digest, "protocol.evaluation_key_digest"],
    ].forEach(([value, field]) => requireDigest(value, `REAL-FHE row ${index}.${field}`));
    if (
      !Number.isInteger(row.action)
      || row.action < 0
      || row.action >= receipt.integer_output_bound.length
    ) {
      throw new Error(`REAL-FHE row ${index}.action is outside the action space`);
    }
    if (index < trials.length) {
      const outputShape = requireArray(row.output_shape, `REAL-FHE row ${index}.output_shape`);
      if (
        outputShape.length !== 1
        || outputShape[0] !== receipt.integer_output_bound.length
      ) {
        throw new Error(`REAL-FHE row ${index}.output_shape is invalid`);
      }
    }
    if ("online_end_to_end_ns" in row) {
      requireNumber(row.online_end_to_end_ns, `REAL-FHE row ${index}.online_end_to_end_ns`);
    }
    if ("expected" in row || "output" in row || "observation" in row || "quantized_input" in row) {
      throw new Error("served evidence contains a private plaintext vector");
    }
  }
  if (trajectory.some((row) => !Number.isFinite(row.online_end_to_end_ns))) {
    throw new Error("closed-loop evidence is missing end-to-end timing");
  }
  if (root.privacy_evidence !== true || root.all_real_fhe_match !== true) {
    throw new Error("served evidence does not establish a successful REAL-FHE run");
  }
  return root;
}

function populateEvidence(data) {
  const gpu = data.gpu_training;
  const search = data.clear_search;
  const receipt = data.circuit_receipt;
  const cold = data.real_fhe_trials[0];
  const warm = data.real_fhe_trials[1];
  const loop = data.closed_loop_real_fhe;
  const trajectory = loop.trajectory;
  const canaries = data.real_fhe_trials.slice(0, 2);
  const canary = data.same_input_canary;
  const envelope = data.authenticated_envelope_protocol;
  document.querySelectorAll("[data-measurement-state]").forEach((node) => {
    node.textContent = "MEASURED";
  });
  document.querySelectorAll("[data-privacy-state]").forEach((node) => {
    node.textContent = "PROVIDED";
  });
  const bundle = data.nonsecret_bundle;

  setField("run-id", data.run_id);
  setField("gpu-policies", format.integer(gpu.population));
  setField("gpu-device", gpu.device_name);
  setField("gpu-iterations", format.integer(gpu.iterations));
  setField("gpu-episodes", format.integer(gpu.episodes_per_candidate));
  setField("student-return", format.number(search.champion_return_mean));
  setField("teacher-return", format.number(search.teacher_return_mean));
  setField("certified-coverage", format.percent(search.certified_coverage));
  setField("box-coverage", format.percent(search.box_certificate_coverage, 2));
  setField("box-points", format.integer(search.box_certificate_points));
  setField("keygen", format.ms(data.client.keygen_ns));
  setField("encrypt", format.ms(warm.encrypt_ns));
  setField("decrypt", format.ms(warm.decrypt_ns));
  setField("cold-eval", format.ms(cold.server_evaluate_ns));
  setField("warm-eval", format.ms(warm.server_evaluate_ns));
  setField("closed-loop-steps", `${loop.exact_matches} / ${loop.completed_steps}`);
  setField("closed-loop-completed", format.integer(loop.completed_steps));
  setField("closed-loop-requested", format.integer(loop.requested_steps));
  setField("closed-loop-return", format.number(loop.return));
  setField("closed-loop-certified", `${loop.certified_steps} / ${loop.completed_steps}`);
  setField("client-location", data.client.location);
  setField("episode-server-median", format.ms(median(trajectory.map((row) => row.server_evaluate_ns))));
  setField("episode-e2e-median", format.ms(median(trajectory.map((row) => row.online_end_to_end_ns))));
  setField("request-bytes", format.bytes(warm.request_bytes));
  setField("response-bytes", format.bytes(warm.response_bytes));
  setField("evaluation-key-bytes", format.bytes(warm.evaluation_key_bytes));
  setField("concrete-version", receipt.concrete_python_version);
  setField("backend", receipt.backend);
  setField("security-level", String(receipt.security_level));
  setField("global-p-error", format.probability(receipt.global_p_error));
  setField("server-artifact-size", format.bytes(receipt.server_artifact_bytes));
  setField("policy-digest", format.compactHash(receipt.policy_digest));
  setField("server-digest", format.compactHash(receipt.server_artifact_sha256));
  setField("client-specs-digest", format.compactHash(receipt.client_specs_sha256));
  setField("mlir-digest", format.compactHash(receipt.mlir_sha256));
  setField("max-bit-width", `${receipt.maximum_integer_bit_width} bits`);
  setField("complexity", format.integer(receipt.complexity));
  setField("compile-time", format.ms(receipt.compile_ns));
  setField("input-shape", format.shape(receipt.input_shape));
  setField("output-shape", format.shape([receipt.integer_output_bound.length]));
  setField("protocol", envelope.name);
  setField("authentication-algorithm", envelope.authentication_algorithm);
  setField("request-schema", envelope.request_schema_version);
  setField("response-schema", envelope.response_schema_version);
  setField("replay-protection", envelope.replay_protection);
  setField("canary-count", format.integer(canary.repetitions));
  setField("canary-distinct", `${canary.distinct_ciphertexts} / ${canary.repetitions}`);
  setField("canary-matches", canary.all_match ? `${canary.repetitions} / ${canary.repetitions}` : "MISMATCH");
  setField("bundle-path", bundle.volume_path);
  setField("bundle-evidence-path", bundle.evidence_path);
  setField("bundle-checksums-path", bundle.checksums_path);
  setField("bundle-files", bundle.files.join(" · "));
  setField("secret-marker-status", data.artifact_secret_marker_audit.server_secret_key_marker_present ? "PRESENT" : "NONE FOUND");

  const durations = {
    keygen: data.client.keygen_ns,
    encrypt: warm.encrypt_ns,
    "cold-eval": cold.server_evaluate_ns,
    "warm-eval": warm.server_evaluate_ns,
    "episode-server-median": median(trajectory.map((row) => row.server_evaluate_ns)),
    decrypt: warm.decrypt_ns,
  };
  const maximum = Math.max(...Object.values(durations));
  document.querySelectorAll("[data-duration-field]").forEach((bar) => {
    const duration = durations[bar.dataset.durationField];
    bar.dataset.width = String(Math.max(1, (duration / maximum) * 100));
  });
}

function initTrace(data) {
  const list = document.querySelector("#trace-list");
  const output = document.querySelector("#trace-output");
  const zone = document.querySelector("#inspector-zone");
  const state = document.querySelector("#inspector-state");
  const play = document.querySelector("#play-trace");
  const toggle = document.querySelector("#toggle-trial");
  const canaryLabel = document.querySelector("#canary-label");
  if (!list || !output || !zone || !state || !play || !toggle || !canaryLabel) return;

  const receipt = data.circuit_receipt;
  const canaries = data.real_fhe_trials;
  let canaryIndex = 0;
  let current = -1;
  let timer;

  function traceFor(trial) {
    const protocol = trial.protocol;
    return [
      {
        zone: "CLIENT / PRIVATE",
        state: "INPUT HELD",
        text: `observation shape     ${format.shape(receipt.input_shape)}\nprivate input value   WITHHELD FROM EVIDENCE\ncanary relation       same private input across all canaries\n\nNo plaintext observation is persisted or sent to the evaluator.`,
      },
      {
        zone: "CLIENT / PRIVATE",
        state: "ENCRYPTED",
        text: `fresh randomness      OS CSPRNG\nrequest bytes         ${format.integer(trial.request_bytes)}\nrequest SHA-256        ${trial.request_sha256}\nevaluation key        ${format.bytes(trial.evaluation_key_bytes)}\n\nsecret key sent       ${data.client.secret_key_sent_to_modal}`,
      },
      {
        zone: "MODAL / PUBLIC TRANSCRIPT",
        state: "AUTHENTICATED",
        text: `protocol              ${protocol.name}\nauthentication        ${protocol.authentication_algorithm}\nrequest envelope      ${protocol.request_envelope_digest}\npolicy digest         ${protocol.policy_digest}\ncircuit digest        ${protocol.circuit_digest}\nclient context        ${protocol.client_context_digest}\n\nobservation value     CIPHERTEXT / UNREADABLE`,
      },
      {
        zone: "MODAL / HONEST-BUT-CURIOUS",
        state: "EVALUATED",
        text: `backend               ${receipt.backend}\nsecurity category     ${receipt.security_level}\nglobal p(error)       ${format.probability(receipt.global_p_error)}\nserver evaluate       ${format.ms(trial.server_evaluate_ns)}\nresponse bytes        ${format.integer(trial.response_bytes)}\nresponse SHA-256      ${trial.response_sha256}\nserver secret marker  ${trial.server_secret_key_marker_present}`,
      },
      {
        zone: "CLIENT / PRIVATE",
        state: "RESPONSE VERIFIED",
        text: `protocol              ${protocol.name}\nresponse schema       ${protocol.response_schema_version}\nresponse envelope     ${protocol.response_envelope_digest}\nresponse shape        ${format.shape(trial.output_shape || [receipt.integer_output_bound.length])}\n\nPlaintext score values are deliberately absent from persisted evidence.`,
      },
      {
        zone: "CLIENT / CANARY",
        state: trial.matches_integer_clear ? "EXACT MATCH" : "MISMATCH",
        text: `integer-clear match   ${trial.matches_integer_clear}\nselected action       ${trial.action}\ndecrypt               ${format.ms(trial.decrypt_ns)}\n\nThis is one same-input randomness canary, not a step in the sequential control trajectory.`,
      },
    ];
  }

  function show(index) {
    current = index;
    const item = traceFor(canaries[canaryIndex])[index];
    zone.textContent = item.zone;
    state.textContent = item.state;
    output.textContent = item.text;
    list.querySelectorAll(".trace-step").forEach((node, nodeIndex) => {
      const active = nodeIndex === index;
      node.classList.toggle("active", active);
      node.classList.toggle("done", nodeIndex < index);
      if (active) node.setAttribute("aria-current", "step");
      else node.removeAttribute("aria-current");
    });
  }

  list.querySelectorAll(".trace-step").forEach((node) => {
    node.addEventListener("click", () => {
      window.clearInterval(timer);
      show(Number(node.dataset.step));
      play.textContent = "Replay canary trace";
    });
  });

  play.addEventListener("click", () => {
    window.clearInterval(timer);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      show(5);
      play.textContent = "Replay canary trace";
      return;
    }
    show(0);
    play.textContent = "Playing…";
    timer = window.setInterval(() => {
      if (current >= 5) {
        window.clearInterval(timer);
        play.textContent = "Replay canary trace";
        return;
      }
      show(current + 1);
    }, 850);
  });

  function selectCanary(index) {
    canaryIndex = index;
    const selected = canaries[index];
    canaryLabel.textContent = `Canary ${index + 1} of ${canaries.length}`;
    document.querySelectorAll("[data-canary-request-hash]").forEach((node) => {
      node.textContent = format.compactHash(selected.request_sha256);
    });
    const next = (index + 1) % canaries.length;
    toggle.textContent = `Show canary ${next + 1}`;
    if (current >= 0) show(current);
  }

  toggle.addEventListener("click", () => {
    selectCanary((canaryIndex + 1) % canaries.length);
  });
  selectCanary(0);
}

function initCopyButtons() {
  document.querySelectorAll(".copy-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const original = "Copy";
      try {
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard API unavailable");
        const code = button.parentElement.querySelector("code").textContent;
        await navigator.clipboard.writeText(code);
        button.textContent = "Copied";
      } catch (error) {
        button.textContent = "Copy denied";
        console.warn("Clipboard write was denied or unavailable", error);
      } finally {
        window.setTimeout(() => {
          button.textContent = original;
        }, 1600);
      }
    });
  });
}

async function setEvidenceDigest(bytes) {
  const node = document.querySelector("#evidence-digest");
  if (!node) return;
  if (!window.crypto?.subtle) {
    node.textContent = notMeasured;
    return;
  }
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  node.textContent = Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

function setEvidenceState(state, message) {
  document.body.dataset.evidenceState = state;
  document.querySelectorAll("[data-evidence-status]").forEach((node) => {
    node.textContent = message;
  });
}

function failClosed(error) {
  document.querySelectorAll("[data-field], #evidence-digest, [data-canary-request-hash], [data-measurement-state], [data-privacy-state]").forEach((node) => {
    node.textContent = notMeasured;
  });
  document.querySelectorAll("#play-trace, #toggle-trial, .trace-step").forEach((control) => {
    control.disabled = true;
  });
  document.querySelectorAll(".waterfall-bar").forEach((bar) => {
    bar.style.width = "0";
    delete bar.dataset.width;
  });
  const output = document.querySelector("#trace-output");
  if (output) output.textContent = "Recorded evidence is unavailable. No measured trace is displayed.";
  setEvidenceState("unavailable", "Evidence unavailable — NOT MEASURED");
  console.error("Unable to load recorded evidence", error);
}

async function loadEvidence() {
  setEvidenceState("loading", "Loading served evidence…");
  try {
    const response = await fetch(evidenceUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`evidence HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    const raw = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    const data = assertEvidence(JSON.parse(raw));
    populateEvidence(data);
    initTrace(data);
    animateBars();
    await setEvidenceDigest(bytes);
    setEvidenceState("loaded", "V2 REAL-FHE evidence loaded");
  } catch (error) {
    failClosed(error);
  }
}

function animateBars() {
  document.querySelectorAll(".waterfall-bar[data-width]").forEach((bar) => {
    const width = bar.dataset.width;
    bar.style.width = "0";
    requestAnimationFrame(() => {
      bar.style.transition = "width 700ms cubic-bezier(.25, 1, .5, 1)";
      bar.style.width = `${width}%`;
    });
  });
}

initCopyButtons();
loadEvidence();
