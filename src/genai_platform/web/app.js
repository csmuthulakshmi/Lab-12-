// GenAI Platform Plus - Frontend Client
(function () {
  "use strict";

  // Application State
  const state = {
    apiKey: localStorage.getItem("genai_api_key") || "test-key-123",
    theme: localStorage.getItem("genai_theme") || "dark",
    sessionId: "sess-" + Math.random().toString(36).substring(2, 9),
    mode: "chat", // "chat" | "agent"
    activeSources: [],
    documents: [],
    isStreaming: false,
  };

  // DOM Elements
  const el = {
    themeToggle: document.getElementById("theme-toggle"),
    apiKeyInput: document.getElementById("api-key-input"),
    healthBadge: document.getElementById("health-badge"),
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("file-input"),
    uploadStatus: document.getElementById("upload-status"),
    docList: document.getElementById("doc-list"),
    newChatBtn: document.getElementById("new-chat-btn"),
    modeChat: document.getElementById("mode-chat"),
    modeAgent: document.getElementById("mode-agent"),
    sourceFilter: document.getElementById("source-filter"),
    hybridToggle: document.getElementById("hybrid-toggle"),
    rerankToggle: document.getElementById("rerank-toggle"),
    kSlider: document.getElementById("k-slider"),
    kValue: document.getElementById("k-value"),
    messagesContainer: document.getElementById("messages-container"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    sendBtn: document.getElementById("send-btn"),
    telemetryFooter: document.getElementById("telemetry-footer"),
    drawerOverlay: document.getElementById("drawer-overlay"),
    drawerClose: document.getElementById("drawer-close"),
    drawerContent: document.getElementById("drawer-content"),
  };

  // Initialization
  function init() {
    applyTheme(state.theme);
    el.apiKeyInput.value = state.apiKey;
    setupEventListeners();
    pollHealth();
    loadDocuments();
    setInterval(pollHealth, 15000);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("genai_theme", theme);
    el.themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
  }

  function setupEventListeners() {
    el.themeToggle.addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      applyTheme(state.theme);
    });

    el.apiKeyInput.addEventListener("change", (e) => {
      state.apiKey = e.target.value.trim();
      localStorage.setItem("genai_api_key", state.apiKey);
      loadDocuments();
    });

    el.newChatBtn.addEventListener("click", () => {
      if (state.isStreaming) return;
      state.sessionId = "sess-" + Math.random().toString(36).substring(2, 9);
      el.messagesContainer.innerHTML = "";
      state.activeSources = [];
      updateTelemetry("New session started.");
    });

    el.modeChat.addEventListener("click", () => setMode("chat"));
    el.modeAgent.addEventListener("click", () => setMode("agent"));

    el.kSlider.addEventListener("input", (e) => {
      el.kValue.textContent = e.target.value;
    });

    // File Upload
    el.dropzone.addEventListener("click", () => el.fileInput.click());
    el.dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.dropzone.classList.add("dragover");
    });
    el.dropzone.addEventListener("dragleave", () => el.dropzone.classList.remove("dragover"));
    el.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        handleUpload(e.dataTransfer.files);
      }
    });
    el.fileInput.addEventListener("change", (e) => {
      if (e.target.files.length) {
        handleUpload(e.target.files);
      }
    });

    // Chat submit
    el.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const message = el.chatInput.value.trim();
      if (!message || state.isStreaming) return;
      el.chatInput.value = "";
      sendMessage(message);
    });

    // Drawer close
    el.drawerClose.addEventListener("click", closeDrawer);
    el.drawerOverlay.addEventListener("click", (e) => {
      if (e.target === el.drawerOverlay) closeDrawer();
    });
  }

  function setMode(mode) {
    state.mode = mode;
    if (mode === "chat") {
      el.modeChat.classList.add("active");
      el.modeAgent.classList.remove("active");
    } else {
      el.modeAgent.classList.add("active");
      el.modeChat.classList.remove("active");
    }
  }

  // Health Polling
  async function pollHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (data.status === "ok") {
        el.healthBadge.className = "badge badge-ok";
        el.healthBadge.textContent = "ONLINE";
      } else {
        el.healthBadge.className = "badge badge-degraded";
        el.healthBadge.textContent = "DEGRADED";
      }
    } catch {
      el.healthBadge.className = "badge badge-error";
      el.healthBadge.textContent = "OFFLINE";
    }
  }

  // Documents Management
  async function loadDocuments() {
    try {
      const res = await fetch("/documents", {
        headers: { "X-API-Key": state.apiKey },
      });
      if (!res.ok) return;
      const docs = await res.json();
      state.documents = docs;
      renderDocumentList(docs);
      populateSourceFilter(docs);
    } catch (e) {
      console.warn("Failed to load documents", e);
    }
  }

  function renderDocumentList(docs) {
    el.docList.innerHTML = "";
    if (!docs.length) {
      el.docList.innerHTML = '<div style="font-size:0.8rem;color:var(--text-muted);text-align:center;padding:12px;">No documents ingested yet.</div>';
      return;
    }

    docs.forEach((doc) => {
      const item = document.createElement("div");
      item.className = "doc-item";

      const name = doc.source.split("/").pop();
      const info = document.createElement("div");
      info.className = "doc-info";
      info.innerHTML = `
        <div class="doc-name" title="${escapeHtml(doc.source)}">${escapeHtml(name)}</div>
        <div class="doc-meta">${doc.chunks} chunks | ${doc.pages || 1} pages</div>
      `;

      const delBtn = document.createElement("button");
      delBtn.className = "btn-icon";
      delBtn.title = "Delete document";
      delBtn.innerHTML = "🗑️";
      delBtn.onclick = () => deleteDocument(doc.source, name);

      item.appendChild(info);
      item.appendChild(delBtn);
      el.docList.appendChild(item);
    });
  }

  function populateSourceFilter(docs) {
    const currentVal = el.sourceFilter.value;
    el.sourceFilter.innerHTML = '<option value="">All Sources</option>';
    docs.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d.source;
      opt.textContent = d.source.split("/").pop();
      el.sourceFilter.appendChild(opt);
    });
    el.sourceFilter.value = currentVal;
  }

  async function handleUpload(files) {
    const formData = new FormData();
    for (let f of files) {
      formData.append("files", f);
    }
    el.uploadStatus.textContent = "Uploading & ingesting...";
    el.uploadStatus.style.color = "var(--accent)";

    try {
      const res = await fetch("/documents/upload", {
        method: "POST",
        headers: { "X-API-Key": state.apiKey },
        body: formData,
      });

      if (res.status === 415) {
        throw new Error("Unsupported file type. Use .pdf, .html, .txt, or .md");
      }
      if (res.status === 413) {
        throw new Error("File exceeds max upload size.");
      }
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      const totalChunks = data.reduce((acc, f) => acc + (f.chunks || 0), 0);
      el.uploadStatus.textContent = `Uploaded ${data.length} file(s) (+${totalChunks} chunks)`;
      el.uploadStatus.style.color = "var(--success)";
      loadDocuments();
    } catch (e) {
      el.uploadStatus.textContent = e.message;
      el.uploadStatus.style.color = "var(--danger)";
    }
  }

  async function deleteDocument(source, name) {
    if (!confirm(`Are you sure you want to delete ${name}?`)) return;
    try {
      const encoded = encodeURIComponent(source);
      const res = await fetch(`/documents/${encoded}`, {
        method: "DELETE",
        headers: { "X-API-Key": state.apiKey },
      });
      if (res.ok) {
        loadDocuments();
      } else {
        alert("Failed to delete document.");
      }
    } catch (e) {
      alert("Error deleting document: " + e.message);
    }
  }

  // SSE Stream Processing
  async function sendMessage(text) {
    state.isStreaming = true;
    el.sendBtn.disabled = true;

    // Append User Message Card
    appendUserMessage(text);

    // Create Assistant Message Card
    const assistantCard = createAssistantCard();
    el.messagesContainer.appendChild(assistantCard);
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;

    const endpoint = state.mode === "chat" ? "/chat/stream" : "/agent/stream";
    const payload =
      state.mode === "chat"
        ? {
            message: text,
            session_id: state.sessionId,
            k: parseInt(el.kSlider.value, 10),
            source: el.sourceFilter.value || null,
            hybrid: el.hybridToggle.checked,
            rerank: el.rerankToggle.checked,
          }
        : {
            message: text,
            session_id: state.sessionId,
          };

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": state.apiKey,
        },
        body: JSON.stringify(payload),
      });

      if (res.status === 401) {
        throw new Error("401 Unauthorized: please check your API Key in the left sidebar.");
      }
      if (!res.ok) {
        const err = await res.text();
        throw new Error(`Server returned ${res.status}: ${err}`);
      }

      await parseSSEStream(res.body, assistantCard);
    } catch (err) {
      renderError(assistantCard, err.message);
    } finally {
      state.isStreaming = false;
      el.sendBtn.disabled = false;
    }
  }

  async function parseSSEStream(stream, card) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const textContentEl = card.querySelector(".message-body");
    const timelineEl = card.querySelector(".timeline");
    const badgeEl = card.querySelector(".grounding-badge");

    let currentEvent = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep last incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          currentEvent = null;
          continue;
        }

        if (trimmed.startsWith("event:")) {
          currentEvent = trimmed.replace("event:", "").trim();
        } else if (trimmed.startsWith("data:")) {
          const rawData = trimmed.replace("data:", "").trim();
          let dataObj = null;
          try {
            dataObj = JSON.parse(rawData);
          } catch {
            dataObj = rawData;
          }

          handleStreamEvent(currentEvent, dataObj, {
            card,
            textContentEl,
            timelineEl,
            badgeEl,
          });
        }
      }
    }
  }

  function handleStreamEvent(event, data, ctx) {
    if (event === "sources") {
      state.activeSources = data;
    } else if (event === "token") {
      const token = typeof data === "object" ? data.token || "" : data;
      ctx.textContentEl.textContent += token;
      // Convert citations [1] into clickable chips
      formatCitations(ctx.textContentEl);
      el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
    } else if (event === "citations") {
      if (ctx.badgeEl) {
        ctx.badgeEl.style.display = "inline-block";
        if (data.refused) {
          ctx.badgeEl.className = "badge badge-degraded";
          ctx.badgeEl.textContent = "REFUSED";
        } else if (data.grounded) {
          ctx.badgeEl.className = "badge badge-ok";
          ctx.badgeEl.textContent = "GROUNDED";
        } else {
          ctx.badgeEl.className = "badge badge-error";
          ctx.badgeEl.textContent = "UNGROUNDED";
        }
      }
    } else if (event === "thought" || event === "action" || event === "observation") {
      if (ctx.timelineEl) {
        appendTimelineStep(ctx.timelineEl, event, data);
      }
    } else if (event === "final") {
      if (ctx.textContentEl && !ctx.textContentEl.textContent.trim()) {
        ctx.textContentEl.textContent = data.answer || "";
      }
    } else if (event === "metrics") {
      updateTelemetry(
        `Latency: ${data.total_latency_s || data.retrieval_latency_s || "-"}s | TTFT: ${data.ttft_s || "-"}s | Tokens: ${data.total_tokens || "-"}`
      );
    } else if (event === "error") {
      renderError(ctx.card, typeof data === "object" ? data.error : data);
    }
  }

  function appendUserMessage(text) {
    const card = document.createElement("div");
    card.className = "message-card message-user";
    card.textContent = text;
    el.messagesContainer.appendChild(card);
  }

  function createAssistantCard() {
    const card = document.createElement("div");
    card.className = "message-card message-assistant";
    card.innerHTML = `
      <div class="message-header">
        <span>${state.mode === "chat" ? "Assistant (RAG)" : "ReAct Agent"}</span>
        <span class="badge grounding-badge" style="display:none;"></span>
      </div>
      <div class="timeline" style="${state.mode === "agent" ? "" : "display:none;"}"></div>
      <div class="message-body"></div>
    `;
    return card;
  }

  function appendTimelineStep(container, type, data) {
    const step = document.createElement("details");
    step.className = "timeline-step";
    step.open = type !== "observation";

    let label = type.toUpperCase();
    if (type === "action") label += `: ${data.tool}`;

    let content = typeof data === "object" ? data.content || data.input || JSON.stringify(data) : data;

    step.innerHTML = `
      <summary>${escapeHtml(label)}</summary>
      <div class="timeline-content">${escapeHtml(content)}</div>
    `;
    container.appendChild(step);
  }

  function formatCitations(elNode) {
    const text = elNode.textContent;
    // Replace [1], [2], etc. with chip button elements
    const parts = text.split(/(\[\d+\])/g);
    if (parts.length <= 1) return;

    elNode.innerHTML = "";
    parts.forEach((p) => {
      const match = p.match(/^\[(\d+)\]$/);
      if (match) {
        const num = parseInt(match[1], 10);
        const chip = document.createElement("span");
        chip.className = "citation-chip";
        chip.textContent = `[${num}]`;
        chip.onclick = () => openDrawer(num);
        elNode.appendChild(chip);
      } else {
        elNode.appendChild(document.createTextNode(p));
      }
    });
  }

  function openDrawer(highlightNum) {
    el.drawerContent.innerHTML = "";
    if (!state.activeSources.length) {
      el.drawerContent.innerHTML = "<p>No context passages retrieved for this response.</p>";
    } else {
      state.activeSources.forEach((src) => {
        const isHighlight = src.n === highlightNum;
        const card = document.createElement("div");
        card.className = "source-card";
        if (isHighlight) card.style.borderColor = "var(--accent)";

        const name = src.source ? src.source.split("/").pop() : "Source";
        card.innerHTML = `
          <div class="source-card-title">[${src.n}] ${escapeHtml(name)} (p.${src.page || 1}) - Score: ${src.score}</div>
          <div class="source-card-snippet">${escapeHtml(src.snippet)}</div>
        `;
        el.drawerContent.appendChild(card);
      });
    }
    el.drawerOverlay.classList.add("active");
  }

  function closeDrawer() {
    el.drawerOverlay.classList.remove("active");
  }

  function renderError(card, message) {
    const errBox = document.createElement("div");
    errBox.style.color = "var(--danger)";
    errBox.style.marginTop = "8px";
    errBox.style.fontWeight = "600";
    errBox.style.fontSize = "0.85rem";
    errBox.textContent = `⚠️ ${message}`;
    card.appendChild(errBox);
  }

  function updateTelemetry(text) {
    el.telemetryFooter.textContent = text;
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Run on page load
  document.addEventListener("DOMContentLoaded", init);
})();
