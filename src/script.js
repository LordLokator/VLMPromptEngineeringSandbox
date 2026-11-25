
let ws = null;
let retryDelayMs = 1000;
const MAX_RETRY_MS = 10000;
const chat = document.getElementById("chat");
const msgInput = document.getElementById("msgInput");
const MAX_MESSAGES = 100;
const CHAT_ROLES = ["user", "system", "vlm", "error"];

const presetBar = document.getElementById("presetBar");
const saveBtn = document.getElementById("savePresetBtn");

function handleServerMessage(data) {
    // Accepted shapes: new {type: 'vlm'|'error'|'system', text, delay?},
    // legacy {output, delay, status}, legacy {status: 'ERROR', msg}, legacy {role, text}.
    const withDelay = (text, delay) =>
        typeof delay === "number" ? `${text} (Δt: ${delay.toFixed(2)}s)` : text;

    if (typeof data.type === "string" && "text" in data) {
        addChatMessage(data.type, withDelay(data.text, data.delay));
    } else if (data.status === "ERROR") {
        addChatMessage("error", data.msg || "Unknown server error");
    } else if (typeof data.output === "string") {
        // An empty output is a valid (empty) VLM line.
        addChatMessage("vlm", withDelay(data.output, data.delay));
    } else if (data.role && data.text) {
        addChatMessage(data.role, data.text);
    } else {
        console.warn("Unhandled message format:", data);
    }
}

function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${scheme}://${location.host}/ws`);
    ws.onopen = () => {
        retryDelayMs = 1000;
        addChatMessage("system", "Connected");
    };
    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        addChatMessage("error", "WebSocket error");
    };
    ws.onclose = () => {
        addChatMessage("system", `Disconnected, retrying in ${retryDelayMs / 1000}s`);
        setTimeout(connect, retryDelayMs);
        retryDelayMs = Math.min(retryDelayMs * 2, MAX_RETRY_MS);
    };
    ws.onmessage = (event) => {
        try {
            handleServerMessage(JSON.parse(event.data));
        } catch (err) {
            console.error("Failed to parse message:", err, event.data);
        }
    };
}

function send(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addChatMessage("error", "Not connected, message not sent");
        return false;
    }
    ws.send(JSON.stringify(payload));
    return true;
}


function sendMessage() {
    const text = msgInput.value.trim();
    if (!text) return;

    const temperature = parseFloat(document.getElementById("tempSlider").value);
    const payload = {
        role: "user",
        text: text,
        temperature: temperature,
        resample_fps: parseInt(document.getElementById("fpsSlider").value, 10)
    };

    if (!send(payload)) return;
    addChatMessage("user", text);

    msgInput.value = "";
    saveBtn.disabled = true;
}


function toggleStream() {
    const payload = { role: "user", text: "toggle_stream" };
    if (!send(payload)) return;

    addChatMessage("system", "Stream toggle requested");

    const img = document.getElementById("video");
    img.src = `/video_feed?ts=${Date.now()}`;
}

function addChatMessage(role, msg) {
    const line = document.createElement("div");
    const safeRole = CHAT_ROLES.includes(role) ? role : "system";
    line.className = `chat-message ${safeRole}`;
    // textContent only: VLM output and server echoes are untrusted (XSS).
    const label = document.createElement("strong");
    label.textContent = `${capitalize(safeRole)}:`;
    const body = document.createElement("span");
    body.textContent = ` ${msg}`;
    line.append(label, body);
    chat.appendChild(line);

    chat.scrollTop = chat.scrollHeight;
    if (chat.children.length > MAX_MESSAGES) {
        chat.removeChild(chat.firstChild);
    }
}

function capitalize(word) {
    return word.charAt(0).toUpperCase() + word.slice(1);
}

function getRandomColor() {
    const hexColors = [
        "#f05d5eff",
        "#0f7173ff",
        "#272932ff",
        "#d8a47fff",
        "#60495A",
        "#3F7D20",
    ];

    return hexColors[Math.floor(Math.random() * hexColors.length)];
}

const FETCH_TIMEOUT_MS = 10000;

async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

async function loadPresetButtons() {
    let presets;
    try {
        const resp = await fetchWithTimeout("/preset_buttons");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        presets = await resp.json();
    } catch (err) {
        console.error("Failed to load presets:", err);
        addChatMessage("error", "Could not load presets");
        return;
    }
    const presetBar = document.getElementById("presetBar");

    presets.forEach(preset => {
        const btn = document.createElement("button");
        btn.className = `btn btn-sm rounded-pill text-white`;
        btn.style.backgroundColor = getRandomColor();
        btn.textContent = preset.label;
        btn.title = preset.text;
        btn.onclick = () => {
            msgInput.value = preset.text;
            msgInput.focus();
            updateSaveButtonState();
        };
        presetBar.appendChild(btn);
    });
}

function updateSaveButtonState() {
    saveBtn.disabled = msgInput.value.trim().length === 0;
}

msgInput.addEventListener("input", updateSaveButtonState);

saveBtn.onclick = async () => {
    const text = msgInput.value.trim();
    if (!text) return;

    const label = prompt("Enter a name for this preset (max 25 characters):");
    if (!label || label.trim().length === 0) return;

    const cleanLabel = label.trim().slice(0, 25);
    const payload = { label: cleanLabel, text: text };

    try {
        const res = await fetchWithTimeout("/save_preset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const result = await res.json();
        if (!res.ok || result.status !== "ok") {
            throw new Error(result.msg || `HTTP ${res.status}`);
        }

        // Only clear the textarea once the server confirmed the save.
        msgInput.value = "";
        updateSaveButtonState();

        const btn = document.createElement("button");
        btn.className = `btn btn-sm rounded-pill text-white`;
        btn.style.backgroundColor = getRandomColor();
        btn.textContent = cleanLabel;
        btn.title = text;
        btn.onclick = () => {
            msgInput.value = text;
            msgInput.focus();
            updateSaveButtonState();
        };
        presetBar.appendChild(btn);
    } catch (err) {
        console.error("Failed to save preset:", err);
        addChatMessage("error", `Failed to save preset: ${err.message}`);
    }
};

document.getElementById('video').addEventListener('contextmenu', function (e) {
    e.preventDefault();
});

document.getElementById("tempSlider").addEventListener("input", () => {
    const val = document.getElementById("tempSlider").value;
    document.getElementById("tempVal").textContent = val;
});

document.getElementById("fpsSlider").addEventListener("input", () => {
    document.getElementById("fpsVal").textContent = document.getElementById("fpsSlider").value;
});

document.getElementById("clearChatBtn").onclick = () => {
    chat.replaceChildren();
};

window.onload = () => {
    connect();
    loadPresetButtons();
    updateSaveButtonState();
};