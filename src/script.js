// Video Prompt Sandbox: two agents, one video. Vanilla JS, no build step.
// Everything from the server is untrusted: text goes in with textContent only, images only from checked data URLs.
"use strict";

let ws = null;
let retryDelayMs = 1000;
let reconnectTimer = null;
const MAX_RETRY_MS = 10000;
const FETCH_TIMEOUT_MS = 10000;
const HISTORY_LENGTH = 8;
const LOG_LENGTH = 6;
const SLOTS = ["A", "B"];
const SERVER_TYPES = ["state", "round", "result", "round_end", "models", "system", "error"];
const METRICS_KEY = "vps.metrics";

const $ = (id) => document.getElementById(id);
const msgInput = $("msgInput");
const presetBar = $("presetBar");
const saveBtn = $("savePresetBtn");

let state = null;  // last `state` frame
const rounds = new Map();  // round id -> {start_s, end_s, label}
const cards = Object.create(null);  // slot ("A" or "B" only) -> {el, role(name), totals, latest}
let thumbUrls = [];  // object URLs of the strip's thumbnails, revoked when replaced
let lastRequest = {};  // request type -> frame, for a confirmed resend

// ---- helpers ------------------------------------------------------------------------------------------------

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
}

function isJpegDataUrl(value) {
    return typeof value === "string" && value.startsWith("data:image/jpeg;base64,")
        && /^[A-Za-z0-9+/]*={0,2}$/.test(value.slice("data:image/jpeg;base64,".length));
}

// A checked JPEG data URL -> an object URL of its bytes. The page never uses a server string as a URL.
function jpegObjectUrl(dataUrl) {
    const binary = atob(dataUrl.slice("data:image/jpeg;base64,".length));
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
}

function num(value) {
    return typeof value === "number" && Number.isFinite(value);
}

function formatCost(cost, source) {
    if (source === "local") return "no API fee";
    if (!num(cost)) return "n/a";
    return `$${cost < 0.01 ? cost.toFixed(5) : cost.toFixed(4)}`;
}

function formatTokens(value) {
    return num(value) ? String(value) : "n/a";
}

function formatMs(value) {
    return num(value) ? (value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`) : "n/a";
}

function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function moment(roundId) {
    const r = rounds.get(roundId);
    if (!r) return `Round ${roundId}`;
    return `Round ${roundId} · ${r.start_s.toFixed(1)}-${r.end_s.toFixed(1)} s · ${r.label}`;
}

function log(kind, text) {
    const box = $("log");
    const line = el("div", kind === "error" ? "text-danger" : "text-secondary", text);
    box.prepend(line);
    while (box.children.length > LOG_LENGTH) box.removeChild(box.lastChild);
}

// ---- connection -----------------------------------------------------------------------------------------------

function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/ws`);
    ws.onopen = () => {
        retryDelayMs = 1000;
        log("system", "Connected");
    };
    ws.onerror = (err) => {
        // onclose always follows and reports it.
        console.error("WebSocket error:", err);
    };
    ws.onclose = () => {
        log("error", `Disconnected, retrying in ${retryDelayMs / 1000}s (analysis pauses while no page is open)`);
        clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connect, retryDelayMs);
        retryDelayMs = Math.min(retryDelayMs * 2, MAX_RETRY_MS);
    };
    ws.onmessage = (event) => {
        try {
            handleServerMessage(JSON.parse(event.data));
        } catch (err) {
            console.error("Failed to handle message:", err);
        }
    };
}

function send(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log("error", "Not connected, nothing sent");
        return false;
    }
    lastRequest[payload.type] = payload;
    ws.send(JSON.stringify(payload));
    return true;
}

async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

function handleServerMessage(data) {
    if (!data || typeof data.type !== "string" || !SERVER_TYPES.includes(data.type)) {
        console.warn("Unhandled message format");
        return;
    }
    if (data.type === "state") renderState(data);
    else if (data.type === "round") renderRound(data);
    else if (data.type === "result") renderResult(data);
    else if (data.type === "round_end") markComparison(data.round_id);
    else if (data.type === "models") renderModels(data);
    else if (data.type === "error") handleError(data);
    else if (typeof data.text === "string") log("system", data.text);
}

function handleError(data) {
    if (data.confirm === "webcam_remote" && lastRequest[data.request]) {
        if (window.confirm(String(data.text))) send({ ...lastRequest[data.request], confirm: true });
        else renderState(state);  // put the pickers back
        return;
    }
    log("error", typeof data.text === "string" ? data.text : "Error");
}

// ---- state ---------------------------------------------------------------------------------------------------

function renderState(data) {
    if (!data) return;
    state = data;
    const run = data.run || {};
    $("runState").textContent = run.running ? "Running" : (run.pause_text || "Paused");
    $("runState").className = `small fw-semibold ${run.running ? "text-success" : "text-secondary"}`;
    $("runBtn").disabled = run.running === true;
    $("pauseBtn").disabled = run.running !== true && run.step_pending !== true;
    $("stepBtn").disabled = run.running === true;

    const slots = data.slots || [];
    const remote = slots.some((s) => s.enabled && s.remote);
    const budget = $("budget");
    if (remote) {
        const usd = num(run.usd_spent) ? run.usd_spent.toFixed(4) : "n/a";
        budget.textContent = `Budget: ${run.rounds_used}/${run.max_rounds} rounds, $${usd} of $${run.max_usd} `
            + `(estimate, list price as of ${data.price_date}); ${run.unpriced_calls} unpriced calls`;
    } else {
        budget.textContent = "Local agents only: no budget or pacing applies.";
    }
    const ollama = slots.filter((s) => s.enabled && s.provider === "ollama").length === 2;
    if (ollama) budget.textContent += " Two Ollama agents on one machine may queue: their latencies are not comparable.";

    const settings = data.settings || {};
    if (document.activeElement !== $("tempSlider") && num(settings.temperature)) {
        $("tempSlider").value = settings.temperature;
        $("tempVal").textContent = settings.temperature.toFixed(2);
    }
    if (document.activeElement !== $("fpsSlider") && num(settings.resample_fps)) {
        $("fpsSlider").value = settings.resample_fps;
        $("fpsVal").textContent = settings.resample_fps;
    }
    $("fpsEffective").textContent = num(settings.effective_fps) ? `(effective ${settings.effective_fps} fps)` : "";
    if (!msgInput.value && typeof settings.prompt === "string") msgInput.placeholder = `Current: ${settings.prompt}`;

    const select = $("sourceSelect");
    select.replaceChildren(...(data.sources || []).map((s) => {
        const option = el("option", null, s.label);
        option.value = s.key;
        return option;
    }));
    const source = data.source || {};
    select.value = source.key;
    $("sourceError").textContent = source.error || "";

    for (const slot of SLOTS) renderSlot(slot, slots.find((s) => s.slot === slot), data.providers || []);
}

function renderSlot(slot, info, providers) {
    const card = cards[slot];
    const r = card.role;
    const select = r("provider");
    const current = info ? info.provider : "demo";
    select.replaceChildren(...providers.map((p) => {
        const option = el("option", null, p.available ? p.id : `${p.id} (unavailable)`);
        option.value = p.id;
        option.disabled = !p.available;
        option.title = p.reason || "";
        return option;
    }));
    if (document.activeElement !== select) select.value = current;
    if (!info) return;
    if (document.activeElement !== r("model")) r("model").value = info.model || "";
    r("enabled").checked = info.enabled;
    r("badge").textContent = info.provider === "demo" ? "demo: no model" : info.label;
    r("badge").className = `badge ${info.provider === "demo" ? "text-bg-warning" : "text-bg-secondary"}`;
    r("privacy").textContent = info.remote
        ? `Frames sent to ${info.destination}; may cost money`
        : "Frames stay on this machine";
    r("privacy").className = `small ${info.remote ? "text-danger" : "text-secondary"}`;
    card.el.classList.toggle("disabled-agent", !info.enabled);
}

// ---- rounds -----------------------------------------------------------------------------------------------------

function renderRound(data) {
    if (!num(data.round_id) || !num(data.start_s) || !num(data.end_s)) return;
    const label = typeof data.source_label === "string" ? data.source_label : "";
    rounds.set(data.round_id, { start_s: data.start_s, end_s: data.end_s, label });
    if (rounds.size > 50) rounds.delete(rounds.keys().next().value);

    const strip = $("strip");
    const thumbs = Array.isArray(data.thumbnails) ? data.thumbnails.filter(isJpegDataUrl) : [];
    const times = Array.isArray(data.frame_times) ? data.frame_times : [];
    thumbUrls.forEach((url) => URL.revokeObjectURL(url));
    thumbUrls = thumbs.map(jpegObjectUrl);
    strip.replaceChildren(...thumbUrls.map((url, i) => {
        const figure = el("figure", "thumb mb-0");
        const img = el("img");
        img.src = url;  // a blob: URL made from checked JPEG bytes
        img.alt = `Frame ${i + 1}`;
        figure.append(img, el("figcaption", null, num(times[i]) ? `${times[i].toFixed(2)} s` : ""));
        return figure;
    }));
    const loop = num(data.loop) ? `, loop ${data.loop + 1}` : "";
    $("stripInfo").textContent = `${moment(data.round_id)}${loop} · ${thumbs.length} frames`;
    for (const slot of SLOTS) {
        if (Array.isArray(data.slots) && data.slots.includes(slot)) {
            cards[slot].role("status").textContent = `Round ${data.round_id}: asking...`;
        }
    }
}

function renderResult(data) {
    if (!SLOTS.includes(data.slot) || !num(data.round_id)) return;
    const card = cards[data.slot];
    const r = card.role;
    const metrics = data.metrics || {};
    const ok = data.status === "ok";
    const errorText = data.error && typeof data.error.message === "string" ? data.error.message : data.status;
    const text = ok ? (data.text || `(empty answer, finish reason: ${data.finish_reason || "n/a"})`) : errorText;
    addHistory(card, data, text);
    if (data.status === "skipped") return;

    // Totals count every call that was made, stale ones included.
    const t = card.totals;
    t.calls += 1;
    if (num(metrics.latency_ms)) t.latencies.push(metrics.latency_ms);
    if (num(metrics.input_tokens)) t.input += metrics.input_tokens;
    if (num(metrics.output_tokens)) t.output += metrics.output_tokens;
    if (num(metrics.cost_usd)) t.cost += metrics.cost_usd;
    if (!ok) t.errors += 1;
    if (ok && data.remote && !num(metrics.cost_usd)) t.unpriced += 1;
    renderTotals(card);

    if (data.stale) return;  // "previous source": in the history only, the live card keeps its answer
    card.latest = { round_id: data.round_id, ok, metrics };
    r("status").textContent = ok ? `Round ${data.round_id}: answered` : `Round ${data.round_id}: ${data.status}`;
    r("status").className = `small fw-semibold ${ok ? "text-success" : "text-danger"}`;
    r("answer").textContent = text;
    r("answer").classList.toggle("text-danger", !ok);
    r("moment").textContent = moment(data.round_id);
    renderRoundMetrics(card, metrics, []);
}

function renderRoundMetrics(card, metrics, marks) {
    const box = card.role("roundMetrics");
    const params = metrics.params_sent && typeof metrics.params_sent === "object"
        ? Object.entries(metrics.params_sent).map(([k, v]) => `${k}=${typeof v === "object" ? "…" : v}`).join(", ")
        : "";
    const lines = [
        `Latency ${formatMs(metrics.latency_ms)}`,
        `Tokens in ${formatTokens(metrics.input_tokens)} / out ${formatTokens(metrics.output_tokens)}`,
        `Cost ${formatCost(metrics.cost_usd, metrics.cost_source)}`,
        `Model ${metrics.reported_model || metrics.requested_model || "n/a"}`,
        `Sent: ${params || "n/a"}`,
    ];
    box.replaceChildren(...lines.map((line) => el("div", null, line)));
    if (marks.length) box.prepend(el("div", "fw-semibold text-primary", marks.join(", ")));
}

function renderTotals(card) {
    const t = card.totals;
    const med = median(t.latencies);
    card.role("totals").replaceChildren(
        el("div", "fw-semibold", "Session"),
        el("div", null, `${t.calls} calls, ${t.errors} errors, median latency ${formatMs(med)}`),
        el("div", null, `Tokens in ${t.input} / out ${t.output}`),
        el("div", null, `Cost ${formatCost(t.cost, null)}${t.unpriced ? `, ${t.unpriced} unpriced` : ""}`),
    );
}

function addHistory(card, data, text) {
    const list = card.role("history");
    const item = el("li", data.stale ? "text-secondary" : null);
    const prefix = `#${data.round_id}${data.stale ? " (previous source)" : ""} ${data.status}: `;
    item.append(el("span", "fw-semibold", prefix), el("span", null, text));
    list.prepend(item);
    while (list.children.length > HISTORY_LENGTH) list.removeChild(list.lastChild);
}

// "faster" / "cheaper" only when both answered the same round with known values; no quality verdict.
function markComparison(roundId) {
    const a = cards.A.latest;
    const b = cards.B.latest;
    if (!a || !b || a.round_id !== roundId || b.round_id !== roundId || !a.ok || !b.ok) return;
    const marks = { A: [], B: [] };
    const compare = (key, word) => {
        const va = a.metrics[key];
        const vb = b.metrics[key];
        if (!num(va) || !num(vb) || va === vb) return;
        marks[va < vb ? "A" : "B"].push(word);
    };
    compare("latency_ms", "faster");
    if (a.metrics.cost_source !== "local" && b.metrics.cost_source !== "local") compare("cost_usd", "cheaper");
    for (const slot of SLOTS) renderRoundMetrics(cards[slot], cards[slot].latest.metrics, marks[slot]);
}

function renderModels(data) {
    const card = SLOTS.map((s) => cards[s]).find((c) => c.pendingList === data.provider);
    if (!card) return;
    card.pendingList = null;
    const models = Array.isArray(data.models) ? data.models : [];
    card.role("models").replaceChildren(...models.map((m) => {
        const option = el("option");
        option.value = String(m.id);
        return option;
    }));
    card.role("listInfo").textContent = data.error
        ? `${data.error}. You can still type a model id.`
        : `${models.length} models listed (vision models; unknown ones kept).`;
}

// ---- cards -----------------------------------------------------------------------------------------------------

function buildCard(slot) {
    const node = $("cardTemplate").content.firstElementChild.cloneNode(true);
    $(`slot${slot}`).appendChild(node);
    const role = (name) => node.querySelector(`[data-role="${name}"]`);
    const card = { el: node, role, latest: null, pendingList: null,
        totals: { calls: 0, errors: 0, latencies: [], input: 0, output: 0, cost: 0, unpriced: 0 } };
    role("title").textContent = `Agent ${slot}`;
    role("models").id = `models${slot}`;
    role("model").setAttribute("list", `models${slot}`);
    role("enabled").id = `enabled${slot}`;
    role("enabledLabel").setAttribute("for", `enabled${slot}`);
    role("listBtn").onclick = () => {
        card.pendingList = role("provider").value;
        role("listInfo").textContent = "Listing...";
        send({ type: "list_models", provider: role("provider").value, refresh: true });
    };
    const apply = () => {
        const model = role("model").value.trim();
        send({ type: "set_agent", slot, provider: role("provider").value, model: model || null,
            enabled: role("enabled").checked });
    };
    role("applyBtn").onclick = apply;
    role("enabled").onchange = apply;
    role("provider").onchange = () => {
        role("model").value = "";
        role("models").replaceChildren();
        role("listInfo").textContent = "";
    };
    renderTotals(card);
    cards[slot] = card;
}

// ---- prompt, presets, settings -----------------------------------------------------------------------------------

function sendUpdate() {
    const payload = {
        type: "update",
        temperature: parseFloat($("tempSlider").value),
        resample_fps: parseInt($("fpsSlider").value, 10),
    };
    const text = msgInput.value.trim();
    if (text) payload.text = text;
    if (!send(payload)) return;
    if (text) log("system", `Prompt set: ${text}`);
    msgInput.value = "";
    updateSaveButtonState();
}

function getRandomColor() {
    const hexColors = ["#f05d5eff", "#0f7173ff", "#272932ff", "#d8a47fff", "#60495A", "#3F7D20"];
    return hexColors[Math.floor(Math.random() * hexColors.length)];
}

function addPresetButton(label, text) {
    const btn = el("button", "btn btn-sm rounded-pill text-white", label);
    btn.style.backgroundColor = getRandomColor();
    btn.title = text;
    btn.onclick = () => {
        msgInput.value = text;
        msgInput.focus();
        updateSaveButtonState();
    };
    presetBar.appendChild(btn);
}

async function loadPresetButtons() {
    let presets;
    try {
        const resp = await fetchWithTimeout("/preset_buttons");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        presets = await resp.json();
    } catch (err) {
        console.error("Failed to load presets:", err);
        log("error", "Could not load presets");
        return;
    }
    presets.forEach((preset) => addPresetButton(preset.label, preset.text));
}

function updateSaveButtonState() {
    saveBtn.disabled = msgInput.value.trim().length === 0;
}

saveBtn.onclick = async () => {
    const text = msgInput.value.trim();
    if (!text) return;
    const label = prompt("Enter a name for this preset (max 25 characters):");
    if (!label || label.trim().length === 0) return;
    const cleanLabel = label.trim().slice(0, 25);
    try {
        const res = await fetchWithTimeout("/save_preset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: cleanLabel, text }),
        });
        const result = await res.json();
        if (!res.ok || result.status !== "ok") {
            throw new Error(result.reason || `HTTP ${res.status}`);
        }
        // Only clear the textarea once the server confirmed the save.
        msgInput.value = "";
        updateSaveButtonState();
        addPresetButton(cleanLabel, text);
    } catch (err) {
        console.error("Failed to save preset:", err);
        log("error", `Failed to save preset: ${err.message}`);
    }
};

// ---- metrics switch ----------------------------------------------------------------------------------------------

function applyMetricsSwitch(on) {
    document.body.classList.toggle("hide-metrics", !on);
    $("metricsSwitch").checked = on;
}

$("metricsSwitch").onchange = () => {
    const on = $("metricsSwitch").checked;
    try {
        localStorage.setItem(METRICS_KEY, on ? "1" : "0");
    } catch (err) {
        console.warn("Could not remember the metrics switch:", err);
    }
    applyMetricsSwitch(on);
};

// ---- wiring --------------------------------------------------------------------------------------------------

$("runBtn").onclick = () => send({ type: "control", action: "run" });
$("pauseBtn").onclick = () => send({ type: "control", action: "pause" });
$("stepBtn").onclick = () => send({ type: "control", action: "step" });
$("resetBudgetBtn").onclick = () => send({ type: "control", action: "reset_budget" });
$("sendBtn").onclick = sendUpdate;
$("sourceSelect").onchange = () => {
    send({ type: "set_source", source_key: $("sourceSelect").value });
    $("video").src = `/video_feed?ts=${Date.now()}`;
};
msgInput.addEventListener("input", updateSaveButtonState);
msgInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) sendUpdate();
});
$("tempSlider").addEventListener("input", () => {
    $("tempVal").textContent = $("tempSlider").value;
});
$("fpsSlider").addEventListener("input", () => {
    $("fpsVal").textContent = $("fpsSlider").value;
});
$("video").addEventListener("contextmenu", (e) => e.preventDefault());

window.onload = () => {
    SLOTS.forEach(buildCard);
    let on = true;
    try {
        on = localStorage.getItem(METRICS_KEY) !== "0";
    } catch (err) {
        console.warn("Could not read the metrics switch:", err);
    }
    applyMetricsSwitch(on);
    connect();
    loadPresetButtons();
    updateSaveButtonState();
};
