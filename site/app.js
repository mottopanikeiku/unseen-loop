const evidenceUrl = "data/evidence.json";

const format = {
  integer: (value) => new Intl.NumberFormat("en-US").format(value),
  percent: (value, digits = 1) => `${(value * 100).toFixed(digits)}%`,
  ms: (nanoseconds, digits = 2) => `${(nanoseconds / 1e6).toFixed(digits)} ms`,
  bytes: (value) => `${new Intl.NumberFormat("en-US").format(value)} B`,
  compactHash: (value) => `${value.slice(0, 8)}…${value.slice(-5)}`,
};

function setField(name, value) {
  document.querySelectorAll(`[data-field="${name}"]`).forEach((node) => {
    node.textContent = value;
  });
}

function populateEvidence(data) {
  const gpu = data.gpu_training;
  const search = data.clear_search;
  const receipt = data.circuit_receipt;
  const cold = data.real_fhe_trials[0];
  const warm = data.real_fhe_trials[1];

  setField("gpu-policies", format.integer(gpu.population));
  setField("student-return", search.champion_return_mean.toFixed(0));
  setField("teacher-return", search.teacher_return_mean.toFixed(0));
  setField("certified-coverage", format.percent(search.certified_coverage));
  setField("box-coverage", format.percent(search.box_certificate_coverage, 2));
  setField("box-points", format.integer(search.box_certificate_points));
  setField("keygen", format.ms(data.client.keygen_ns));
  setField("encrypt", format.ms(warm.encrypt_ns));
  setField("decrypt", format.ms(warm.decrypt_ns));
  setField("cold-eval", format.ms(cold.server_evaluate_ns));
  setField("warm-eval", format.ms(warm.server_evaluate_ns));
  setField("request-bytes", format.bytes(warm.request_bytes));
  setField("response-bytes", format.bytes(warm.response_bytes));
  setField("evaluation-key-bytes", format.bytes(warm.evaluation_key_bytes));
  setField("concrete-version", receipt.concrete_python_version);
  setField("security-level", String(receipt.security_level));
  setField("server-artifact-size", format.bytes(receipt.server_artifact_bytes));
  setField("policy-digest", format.compactHash(receipt.policy_digest));
  setField("server-digest", format.compactHash(receipt.server_artifact_sha256));
  setField("mlir-digest", format.compactHash(receipt.mlir_sha256));
  setField("max-bit-width", `${receipt.maximum_integer_bit_width} bits`);
  setField("complexity", format.integer(receipt.complexity));
  setField("compile-time", format.ms(receipt.compile_ns));

  const requestHash = document.querySelector("#request-hash");
  if (requestHash) requestHash.textContent = format.compactHash(cold.request_sha256);
}

function initTrace(data) {
  const list = document.querySelector("#trace-list");
  const output = document.querySelector("#trace-output");
  const zone = document.querySelector("#inspector-zone");
  const state = document.querySelector("#inspector-state");
  const play = document.querySelector("#play-trace");
  const toggle = document.querySelector("#toggle-trial");
  if (!list || !output || !zone || !state || !play || !toggle) return;

  const trial = data.real_fhe_trials[0];
  const trace = [
    {
      zone: "CLIENT / PRIVATE",
      state: "QUANTIZED",
      text: `observation shape  [4]\nquantized input    [0, 0, 0, 0]\nvalid code range   [-7, 7]⁴\n\nThe raw environment state and quantizer remain client-side.`,
    },
    {
      zone: "CLIENT / PRIVATE",
      state: "ENCRYPTED",
      text: `fresh randomness    OS CSPRNG\nrequest bytes       ${format.integer(trial.request_bytes)}\nrequest SHA-256      ${trial.request_sha256}\nevaluation key      ${format.bytes(trial.evaluation_key_bytes)}\n\nSecret/decryption key: NOT SERIALIZED`,
    },
    {
      zone: "MODAL / PUBLIC TRANSCRIPT",
      state: "ACCEPTED",
      text: `policy digest       ${data.circuit_receipt.policy_digest}\ncircuit digest      ${data.circuit_receipt.server_artifact_sha256}\ninput shape         [4]\nrequest bytes       ${format.integer(trial.request_bytes)}\nrequest hash        ${trial.request_sha256}\n\n<span class="redacted">observation value   [CIPHERTEXT / UNREADABLE]</span>`,
    },
    {
      zone: "MODAL / HONEST-BUT-CURIOUS",
      state: "EVALUATED",
      text: `backend             ${data.circuit_receipt.backend}\nsecurity category   ${data.circuit_receipt.security_level}\nglobal p(error)     ${data.circuit_receipt.global_p_error}\nserver evaluate     ${format.ms(trial.server_evaluate_ns)}\nresponse bytes      ${format.integer(trial.response_bytes)}\nresponse SHA-256    ${trial.response_sha256}\nserver secret key   false`,
    },
    {
      zone: "CLIENT / PRIVATE",
      state: "DECRYPTED",
      text: `integer scores      [${trial.output.join(", ")}]\nexpected clear      [${trial.expected.join(", ")}]\nexact match         ${trial.matches_integer_clear}\ndecrypt             ${format.ms(trial.decrypt_ns)}\n\nThe evaluator never sees these plaintext scores.`,
    },
    {
      zone: "CLIENT / CONTROL LOOP",
      state: "ACTION 1",
      text: `stable argmax       1\nscore margin        ${trial.output[1] - trial.output[0]} integer units\nnext operation      environment.step(1)\n\nThe next state depends on this action: inference fidelity is a sequential contract.`,
    },
  ];

  let current = -1;
  let timer;
  function show(index) {
    current = index;
    const item = trace[index];
    zone.textContent = item.zone;
    state.textContent = item.state;
    output.innerHTML = item.text;
    list.querySelectorAll("li").forEach((node, nodeIndex) => {
      node.classList.toggle("active", nodeIndex === index);
      node.classList.toggle("done", nodeIndex < index);
    });
  }

  list.querySelectorAll("li").forEach((node) => {
    node.tabIndex = 0;
    const activate = () => {
      window.clearInterval(timer);
      show(Number(node.dataset.step));
      play.textContent = "Replay recorded trace";
    };
    node.addEventListener("click", activate);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") activate();
    });
  });

  play.addEventListener("click", () => {
    window.clearInterval(timer);
    show(0);
    play.textContent = "Playing…";
    timer = window.setInterval(() => {
      if (current >= trace.length - 1) {
        window.clearInterval(timer);
        play.textContent = "Replay recorded trace";
        return;
      }
      show(current + 1);
    }, window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 30 : 850);
  });

  let trialIndex = 0;
  toggle.addEventListener("click", () => {
    trialIndex = trialIndex === 0 ? 1 : 0;
    const selected = data.real_fhe_trials[trialIndex];
    document.querySelector("#request-hash").textContent = format.compactHash(selected.request_sha256);
    toggle.textContent = trialIndex === 0 ? "Compare trial 02" : "Return to trial 01";
  });
}

function initCopyButtons() {
  document.querySelectorAll(".copy-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = button.parentElement.querySelector("code").textContent;
      await navigator.clipboard.writeText(code);
      button.textContent = "Copied";
      window.setTimeout(() => { button.textContent = "Copy"; }, 1200);
    });
  });
}

async function setEvidenceDigest(raw) {
  const node = document.querySelector("#evidence-digest");
  if (!node || !window.crypto?.subtle) return;
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(raw));
  node.textContent = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function loadEvidence() {
  try {
    const response = await fetch(evidenceUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`evidence HTTP ${response.status}`);
    const raw = await response.text();
    const data = JSON.parse(raw);
    populateEvidence(data);
    initTrace(data);
    setEvidenceDigest(raw);
  } catch (error) {
    document.querySelectorAll("[data-field]").forEach((node) => {
      node.textContent = "NOT MEASURED";
    });
    console.error("Unable to load recorded evidence", error);
  }
}

function animateBars() {
  document.querySelectorAll(".waterfall-bar").forEach((bar) => {
    const width = bar.dataset.width || 0;
    bar.style.width = "0";
    requestAnimationFrame(() => {
      bar.style.transition = "width 700ms cubic-bezier(.25, 1, .5, 1)";
      bar.style.width = `${width}%`;
    });
  });
}

initCopyButtons();
animateBars();
loadEvidence();
