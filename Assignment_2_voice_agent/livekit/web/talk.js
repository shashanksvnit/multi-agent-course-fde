import { Room, RoomEvent } from "/node_modules/livekit-client/dist/livekit-client.esm.mjs";

const participantsEl = document.querySelector("#participants");
const providerEl = document.querySelector("#provider");
const transcriptEl = document.querySelector("#transcript");
const formEl = document.querySelector("#agent-form");
const messageEl = document.querySelector("#caller-message");
const speakToggleEl = document.querySelector("#speak-toggle");
const voiceStatusEl = document.querySelector("#voice-status");
let speakReplies = true;
let recorder = null;
let recordedChunks = [];
let listenStream = null;
let audioContext = null;
let analyser = null;
let vadFrame = null;
let silenceStartedAt = null;
let recordingStartedAt = 0;
let agentBusy = false;
let agentSpeaking = false;
let listenCooldownUntil = 0;

const VAD_THRESHOLD = 0.035;
const ENDPOINT_SILENCE_MS = 850;
const MIN_TURN_MS = 650;

const clients = {
  caller: {
    identity: "caller-demo",
    name: "Caller Demo",
    room: null,
    muted: false,
    root: document.querySelector('[data-client="caller"]'),
  },
  agent: {
    identity: "aurora-agent",
    name: "Aurora Agent",
    room: null,
    muted: false,
    root: document.querySelector('[data-client="agent"]'),
  },
};

function control(client, role) {
  return client.root.querySelector(`[data-role="${role}"]`);
}

function setStatus(client, message) {
  control(client, "status").textContent = message;
}

function setControls(client, connected) {
  control(client, "join").disabled = connected;
  control(client, "mute").disabled = !connected;
  control(client, "leave").disabled = !connected;
  client.root.classList.toggle("connected", connected);
}

function participantName(participant) {
  return participant.name || participant.identity;
}

function renderParticipants() {
  const rows = [];
  for (const client of Object.values(clients)) {
    if (!client.room) continue;
    rows.push({
      name: participantName(client.room.localParticipant),
      side: client.name,
      type: "local",
    });
    for (const participant of client.room.remoteParticipants.values()) {
      rows.push({
        name: participantName(participant),
        side: client.name,
        type: "remote",
      });
    }
  }

  participantsEl.innerHTML = "";
  if (rows.length === 0) {
    participantsEl.innerHTML = '<div class="empty">Join one or both panes to see participants.</div>';
    return;
  }

  for (const row of rows) {
    const element = document.createElement("div");
    element.className = "participant";
    element.innerHTML = `
      <strong>${row.name}</strong>
      <span>${row.type} in ${row.side}</span>
    `;
    participantsEl.appendChild(element);
  }
}

function addTranscript(role, text, meta = "") {
  const empty = transcriptEl.querySelector(".empty");
  if (empty) empty.remove();

  const item = document.createElement("div");
  item.className = `bubble ${role}`;
  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = `${role === "caller" ? "Caller Demo" : "Aurora Agent"}${meta ? ` · ${meta}` : ""}`;
  const body = document.createElement("div");
  body.textContent = text;
  item.append(label, body);
  transcriptEl.appendChild(item);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return item;
}

function speak(text) {
  if (!speakReplies || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.96;
  utterance.pitch = 1.02;
  agentSpeaking = true;
  utterance.onstart = () => {
    agentSpeaking = true;
    voiceStatusEl.textContent = "Agent speaking";
  };
  utterance.onend = () => {
    agentSpeaking = false;
    listenCooldownUntil = Date.now() + 700;
    if (listenStream) {
      voiceStatusEl.textContent = "Listening. Speak naturally.";
    }
  };
  window.speechSynthesis.speak(utterance);
}

async function sendToAgent(text) {
  addTranscript("caller", text);
  const pending = document.createElement("div");
  pending.className = "bubble agent pending";
  pending.textContent = "Aurora Agent is thinking...";
  transcriptEl.appendChild(pending);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;

  try {
    const response = await fetch("/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    pending.remove();
    if (!response.ok) {
      throw new Error(payload.error || `Agent request failed: ${response.status}`);
    }
    providerEl.textContent = `Provider: ${payload.provider} · ${payload.model}`;
    addTranscript("agent", payload.reply, payload.action ? `action: ${payload.action}` : "");
    speak(payload.reply);
  } catch (error) {
    pending.remove();
    addTranscript("agent", error.message);
  }
}

async function sendAudioToAgent(audioBlob) {
  agentBusy = true;
  const voicePlaceholder = addTranscript("caller", "Voice message", "transcribing");
  const pending = document.createElement("div");
  pending.className = "bubble agent pending";
  pending.textContent = "Aurora Agent is listening...";
  transcriptEl.appendChild(pending);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;

  try {
    const response = await fetch("/voice-agent", {
      method: "POST",
      headers: { "Content-Type": audioBlob.type || "audio/webm" },
      body: audioBlob,
    });
    const payload = await response.json();
    pending.remove();
    if (!response.ok) {
      throw new Error(payload.error || `Voice request failed: ${response.status}`);
    }
    voicePlaceholder.remove();
    addTranscript("caller", payload.transcript, payload.sttModel ? `STT: ${payload.sttModel}` : "");
    providerEl.textContent = `Provider: ${payload.provider} · ${payload.model}`;
    addTranscript("agent", payload.reply, payload.action ? `action: ${payload.action}` : "");
    speak(payload.reply);
  } catch (error) {
    pending.remove();
    addTranscript("agent", error.message);
  } finally {
    agentBusy = false;
    if (!agentSpeaking && listenStream) {
      voiceStatusEl.textContent = "Listening. Speak naturally.";
    }
  }
}

function stopTurnRecording() {
  if (recorder && recorder.state !== "inactive") {
    recorder.stop();
  }
}

function audioLevel() {
  if (!analyser) return 0;
  const data = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (const sample of data) {
    const value = (sample - 128) / 128;
    sum += value * value;
  }
  return Math.sqrt(sum / data.length);
}

function startTurnRecording() {
  if (!listenStream || recorder || agentBusy || agentSpeaking) return;

  recordedChunks = [];
  recorder = new MediaRecorder(listenStream);
  recordingStartedAt = Date.now();
  silenceStartedAt = null;
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  recorder.onstop = () => {
    const mimeType = recorder.mimeType || "audio/webm";
    const audioBlob = new Blob(recordedChunks, { type: mimeType });
    recorder = null;
    recordedChunks = [];
    if (audioBlob.size < 800) {
      voiceStatusEl.textContent = "Listening. Speak naturally.";
      return;
    }
    voiceStatusEl.textContent = "Transcribing...";
    sendAudioToAgent(audioBlob);
  };
  recorder.start();
  voiceStatusEl.textContent = "Listening to caller...";
}

function vadLoop() {
  if (!listenStream) return;

  const now = Date.now();
  const canListen = !agentBusy && !agentSpeaking && now > listenCooldownUntil;
  const speaking = canListen && audioLevel() > VAD_THRESHOLD;

  if (speaking) {
    if (!recorder) {
      startTurnRecording();
    }
    silenceStartedAt = null;
  } else if (recorder) {
    silenceStartedAt = silenceStartedAt || now;
    const turnLongEnough = now - recordingStartedAt > MIN_TURN_MS;
    const silenceLongEnough = now - silenceStartedAt > ENDPOINT_SILENCE_MS;
    if (turnLongEnough && silenceLongEnough) {
      stopTurnRecording();
    }
  }

  vadFrame = requestAnimationFrame(vadLoop);
}

async function startPhoneListener() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    voiceStatusEl.textContent = "Audio recording is not available in this browser";
    return;
  }
  if (listenStream) return;

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }

  listenStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(listenStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  voiceStatusEl.textContent = "Listening. Speak naturally.";
  vadFrame = requestAnimationFrame(vadLoop);
  speak("Thanks for calling Aurora Hotel reservations. How can I help?");
}

function stopPhoneListener() {
  if (vadFrame) {
    cancelAnimationFrame(vadFrame);
    vadFrame = null;
  }
  stopTurnRecording();
  if (listenStream) {
    listenStream.getTracks().forEach((track) => track.stop());
    listenStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  analyser = null;
  voiceStatusEl.textContent = "Start call to begin listening";
}

async function loadState() {
  try {
    const response = await fetch("/state");
    const state = await response.json();
    providerEl.textContent = `Provider: ${state.agentProvider}`;
  } catch {
    providerEl.textContent = "Provider: unavailable";
  }
}

function attachRoomEvents(client) {
  client.room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind !== "audio") return;
    const element = track.attach();
    element.autoplay = true;
    control(client, "audio").appendChild(element);
  });
  client.room.on(RoomEvent.TrackUnsubscribed, (track) => track.detach());
  client.room.on(RoomEvent.ParticipantConnected, renderParticipants);
  client.room.on(RoomEvent.ParticipantDisconnected, renderParticipants);
  client.room.on(RoomEvent.Disconnected, () => {
    setControls(client, false);
    setStatus(client, "Disconnected");
    renderParticipants();
  });
}

async function join(client) {
  setStatus(client, "Creating token...");
  const params = new URLSearchParams({ identity: client.identity, name: client.name });
  const response = await fetch(`/token?${params.toString()}`);
  if (!response.ok) throw new Error(`Token request failed: ${response.status}`);
  const session = await response.json();

  client.room = new Room({ adaptiveStream: true, dynacast: true });
  attachRoomEvents(client);

  setStatus(client, "Joining room...");
  await client.room.connect(session.url, session.token);
  const publishMic = client.identity === "caller-demo";
  await client.room.localParticipant.setMicrophoneEnabled(publishMic);

  client.muted = !publishMic;
  control(client, "mute").textContent = publishMic ? "Mute" : "Mic off";
  control(client, "mute").disabled = !publishMic;
  setControls(client, true);
  control(client, "mute").disabled = !publishMic;
  setStatus(client, "Connected");
  renderParticipants();
  if (client.identity === "caller-demo") {
    startPhoneListener().catch((error) => {
      voiceStatusEl.textContent = error.message;
    });
  }
}

async function leave(client) {
  if (client.room) {
    client.room.disconnect();
    client.room = null;
  }
  control(client, "audio").innerHTML = "";
  setControls(client, false);
  setStatus(client, "Disconnected");
  renderParticipants();
  if (client.identity === "caller-demo") {
    stopPhoneListener();
  }
}

async function toggleMute(client) {
  if (!client.room) return;
  client.muted = !client.muted;
  await client.room.localParticipant.setMicrophoneEnabled(!client.muted);
  control(client, "mute").textContent = client.muted ? "Unmute" : "Mute";
  setStatus(client, client.muted ? "Muted" : "Connected");
}

for (const client of Object.values(clients)) {
  control(client, "join").addEventListener("click", () => {
    join(client).catch((error) => setStatus(client, error.message));
  });
  control(client, "leave").addEventListener("click", () => {
    leave(client).catch((error) => setStatus(client, error.message));
  });
  control(client, "mute").addEventListener("click", () => {
    toggleMute(client).catch((error) => setStatus(client, error.message));
  });
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageEl.value.trim();
  if (!text) return;

  messageEl.value = "";
  sendToAgent(text);
});

speakToggleEl.addEventListener("click", () => {
  speakReplies = !speakReplies;
  speakToggleEl.textContent = speakReplies ? "Speak replies on" : "Speak replies off";
  if (!speakReplies && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
});

loadState();
