
const ws = new WebSocket(`ws://${location.host}/ws`);
const chat = document.getElementById("chat");
const msgInput = document.getElementById("msgInput");
const MAX_MESSAGES = 100;

const presetBar = document.getElementById("presetBar");
const saveBtn = document.getElementById("savePresetBtn");

const pastelColors = ["#FF6633", "#FFB399", "#FF33FF", "#FFFF99", "#00B3E6", "#E6B333", "#3366E6", "#999966", "#809980", "#E6FF80", "#1AFF33", "#999933", "#FF3380", "#CCCC00", "#66E64D", "#4D80CC", "#FF4D4D", "#99E6E6", "#6666FF"
];

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);

        if (data.output) {
            const text = `${data.output} (Δt: ${data.delay.toFixed(2)}s)`;
            addChatMessage("vlm", text);
        } else if (data.role && data.text) {
            addChatMessage(data.role, data.text);
        } else {
            console.warn("Unhandled message format:", data);
        }

    } catch (err) {
        console.error("Failed to parse message:", err, event.data);
    }
};


function sendMessage() {
    const text = msgInput.value.trim();
    if (!text) return;

    const temperature = parseFloat(document.getElementById("tempSlider").value);
    const payload = {
        role: "user",
        text: text,
        temperature: temperature
    };

    ws.send(JSON.stringify(payload));
    addChatMessage("user", text);

    msgInput.value = "";
    saveBtn.disabled = true;
}


function toggleStream() {
    const payload = { role: "user", text: "toggle_stream" };
    ws.send(JSON.stringify(payload));

    addChatMessage("System", "Stream toggle requested");

    const img = document.getElementById("video");
    img.src = `/video_feed?ts=${Date.now()}`;
}

function addChatMessage(role, msg) {
    const line = document.createElement("div");
    line.className = `chat-message ${role}`;
    line.innerHTML = `<strong>${capitalize(role)}:</strong> ${msg}`;
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

async function loadPresetButtons() {
    const resp = await fetch("/preset_buttons");
    const presets = await resp.json();
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
    msgInput.value = "";
    updateSaveButtonState();

    const payload = { label: cleanLabel, text: text };

    const res = await fetch("/save_preset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const result = await res.json();
    if (result.status === "ok") {
        const btn = document.createElement("button");
        btn.className = `btn btn-sm rounded-pill text-white ${getRandomColorClass()}`;
        btn.textContent = cleanLabel;
        btn.title = text;
        btn.onclick = () => {
            msgInput.value = text;
            msgInput.focus();
            updateSaveButtonState();
        };
        presetBar.appendChild(btn);
    } else {
        alert("Failed to save preset.");
    }
};

document.getElementById('video').addEventListener('contextmenu', function (e) {
    e.preventDefault();
});

document.getElementById("tempSlider").addEventListener("input", () => {
    const val = document.getElementById("tempSlider").value;
    document.getElementById("tempVal").textContent = val;
});

document.getElementById("clearChatBtn").onclick = () => {
    chat.innerHTML = "";
};

window.onload = () => {
    loadPresetButtons();
    updateSaveButtonState();
};