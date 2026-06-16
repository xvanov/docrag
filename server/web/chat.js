/* docrag chat -- client logic. Plain DOM, no framework.
 *
 * Dark UI adapted from the knowledge-RAG doc_chat client, wired to docrag's
 * endpoints (/api/corpora, /api/chat, /api/rate, /api/upload, /source) with
 * all brand / distill / attach / advanced machinery removed.
 *
 * State:
 *   - corpus: selected corpus slug.
 *   - history: recent {role, content} turns sent to the server.
 *   - threads: chats persisted in localStorage.
 *   - activeThreadId: id of the thread shown, or null for a fresh chat
 *     (lazily minted on first send).
 */

(function () {
  "use strict";

  var HISTORY_CAP = 12;            // turns sent to server (server re-budgets)
  var THREAD_HISTORY_CAP = 24;     // messages stored per thread
  var THREAD_CAP = 50;             // total threads kept in localStorage
  var SIDEBAR_RENDER_CAP = 30;
  var PREVIEW_CHARS = 300;
  var STORE_KEY = "docrag_threads_v1";
  var SIDEBAR_KEY = "docrag_sidebar_collapsed_v1";

  var SUGGESTIONS = [
    "What are the minimum egress door width requirements?",
    "How is fire separation distance measured?",
    "What are the setback requirements for an accessory dwelling unit?",
    "Summarize the occupancy classification groups.",
  ];

  var state = {
    corpus: "",
    history: [],
    inflight: false,
    jobs: {},            // threadId -> in-flight job {abort, startedAt, done, statusText, sources, stageAt}
    tickTimer: null,     // shared 1s status ticker
    sourceMap: {},       // assistantMsgId -> array of source-card elements (1-indexed)
    corpora: [],
    sources: [],         // documents in the active corpus (read-only display)
    threads: [],
    activeThreadId: null,
    pickerOpen: false,
    hoverIdx: -1,
    location: "",        // selected location key (global)
    versions: {},        // per-document edition overrides {doc_key: year}
    versionDocs: [],      // versionable documents from /api/facets
  };

  var LS_LOCATION = "docrag.location";
  var LS_VERSIONS = "docrag.versions";

  function loadLocalJson(key, fallback) {
    try { var v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
    catch (e) { return fallback; }
  }
  function saveLocal(key, val) {
    try { localStorage.setItem(key, JSON.stringify(val)); } catch (e) {}
  }

  var els = {
    thread: document.getElementById("thread"),
    input: document.getElementById("input"),
    send: document.getElementById("send"),
    status: document.getElementById("status"),
    corpus: document.getElementById("corpus-select"),
    sourcesOnly: document.getElementById("sources-only"),
    topk: document.getElementById("topk"),
    topkValue: document.getElementById("topk-value"),
    // Sidebar
    sidebar: document.getElementById("history-sidebar"),
    sidebarToggle: document.getElementById("sidebar-toggle"),
    sidebarToggleFloating: document.getElementById("sidebar-toggle-floating"),
    sidebarBackdrop: document.getElementById("sidebar-backdrop"),
    threadList: document.getElementById("thread-list"),
    newChatBtn: document.getElementById("new-chat"),
    // Corpus picker
    pickerBtn: document.getElementById("corpus-picker-btn"),
    pickerName: document.getElementById("corpus-picker-name"),
    pickerCount: document.getElementById("corpus-picker-count"),
    pickerMenu: document.getElementById("corpus-picker-menu"),
    sourcesList: document.getElementById("sources-list"),
    // Settings popover
    settingsBtn: document.getElementById("settings-btn"),
    settingsMenu: document.getElementById("settings-menu"),
    // Upload modal
    uploadBtn: document.getElementById("upload-btn"),
    uploadModal: document.getElementById("upload-modal"),
    uploadCancel: document.getElementById("upload-cancel"),
    uploadCorpus: document.getElementById("upload-corpus"),
    uploadExistingRow: document.getElementById("upload-existing-corpus-row"),
    uploadNewRow: document.getElementById("upload-new-corpus-row"),
    uploadNewCorpus: document.getElementById("upload-new-corpus"),
    uploadModeExisting: document.getElementById("upload-corpus-mode-existing"),
    uploadModeNew: document.getElementById("upload-corpus-mode-new"),
    uploadFile: document.getElementById("upload-file"),
    uploadSubmit: document.getElementById("upload-submit"),
    uploadStatus: document.getElementById("upload-status"),
    // Help modal
    helpBtn: document.getElementById("help-btn"),
    helpModal: document.getElementById("help-modal"),
    helpCancel: document.getElementById("help-cancel"),
    // Welcome / relocatable composer
    welcome: document.getElementById("welcome"),
    welcomeGreeting: document.getElementById("welcome-greeting"),
    welcomeSub: document.getElementById("welcome-sub"),
    suggestions: document.getElementById("suggestions"),
    composer: document.getElementById("composer"),
    composerDockWelcome: document.getElementById("composer-dock-welcome"),
    composerDockBottom: document.getElementById("composer-dock-bottom"),
    composerGrounded: document.getElementById("composer-grounded"),
    // Location + version selectors
    locationSelect: document.getElementById("location-select"),
    versionsSection: document.getElementById("versions-section"),
    versionsList: document.getElementById("versions-list"),
  };

  // ---------- Utils ----------

  function escapeHtml(s) {
    if (s === null || s === undefined) return "";
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function el(tag, opts, children) {
    var node = document.createElement(tag);
    if (opts) {
      if (opts.className) node.className = opts.className;
      if (opts.text !== undefined) node.textContent = opts.text;
      if (opts.html !== undefined) node.innerHTML = opts.html;
      if (opts.attrs) for (var k in opts.attrs)
        if (Object.prototype.hasOwnProperty.call(opts.attrs, k))
          node.setAttribute(k, opts.attrs[k]);
      if (opts.on) for (var ev in opts.on)
        if (Object.prototype.hasOwnProperty.call(opts.on, ev))
          node.addEventListener(ev, opts.on[ev]);
    }
    if (children) for (var i = 0; i < children.length; i++)
      if (children[i] != null) node.appendChild(children[i]);
    return node;
  }

  function setStatus(text, withSpinner) {
    if (!text) { els.status.textContent = ""; els.status._txt = null; return; }
    // Build the spinner + text nodes ONCE and only update the text afterwards,
    // so the spinner element isn't recreated each tick (which restarted its
    // CSS animation and made it stutter). The spinner now spins continuously.
    if (!els.status._txt) {
      els.status.innerHTML = '<span class="spinner"></span><span class="status-text"></span>';
      els.status._spin = els.status.firstChild;
      els.status._txt = els.status.lastChild;
    }
    els.status._spin.style.display = withSpinner ? "" : "none";
    els.status._txt.textContent = text;
  }

  // Live status: a stage message with an always-ticking elapsed clock so the
  // wait never looks frozen, plus (when a stage names sources) a rotating
  // "Consulting <source>" line that reflects the real work in flight.
  function fmtElapsed(ms) {
    var s = Math.max(0, Math.floor(ms / 1000));
    return Math.floor(s / 60) + ":" + ("0" + (s % 60)).slice(-2);
  }
  // Human-friendly total for the finished answer ("0.8 s", "12.3 s", "1 min 4 s").
  function fmtDuration(ms) {
    if (!ms) return "";
    if (ms < 1000) return ms + " ms";
    var s = ms / 1000;
    if (s < 60) return s.toFixed(1) + " s";
    var m = Math.floor(s / 60), r = Math.round(s % 60);
    return m + " min " + (r ? r + " s" : "");
  }
  // ---- Per-chat jobs: each thread may have at most one in-flight query, but
  // several threads can run at once. The composer status reflects ONLY the
  // active thread's job, so switching chats never shows a stale spinner. ----
  function activeJob() { return state.jobs[state.activeThreadId] || null; }
  function anyRunningJob() {
    for (var k in state.jobs)
      if (state.jobs[k] && !state.jobs[k].done) return true;
    return false;
  }
  function renderActiveStatus() {
    var job = activeJob();
    if (!job || job.done) { setStatus(""); return; }
    var base = job.statusText || "Thinking";
    if (job.sources && job.sources.length) {
      var k = Math.floor((Date.now() - (job.stageAt || job.startedAt)) / 1800) % job.sources.length;
      base = "Consulting " + truncate(job.sources[k], 44);
    }
    setStatus(base + "  ·  " + fmtElapsed(Date.now() - job.startedAt), true);
  }
  // One shared 1s ticker drives the elapsed clock + source rotation; it stops
  // itself once no job anywhere is still running.
  function ensureTicker() {
    if (state.tickTimer) return;
    state.tickTimer = setInterval(function () {
      renderActiveStatus();
      if (!anyRunningJob()) { clearInterval(state.tickTimer); state.tickTimer = null; }
    }, 1000);
  }

  var SEND_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 19V5m0 0l-6 6m6-6l6 6"/></svg>';
  var STOP_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
  function updateSendButton() {
    var running = !!(activeJob() && !activeJob().done);
    els.send.classList.toggle("stop", running);
    els.send.innerHTML = running ? STOP_SVG : SEND_SVG;
    els.send.title = running ? "Stop generating" : "Send (Enter)";
    els.send.setAttribute("aria-label", running ? "Stop generating" : "Send");
  }
  function abortActiveJob() {
    var job = activeJob();
    if (job && !job.done) { try { job.abort.abort(); } catch (e) {} }
  }
  function onSendClick() {
    if (activeJob() && !activeJob().done) abortActiveJob();
    else send();
  }

  // ---- Sidebar completion signal (flash + bell until the chat is opened) ----
  function markAttention(tid) {
    var t = findThread(tid);
    if (t) { t.attention = true; renderSidebar(); }
  }
  function clearAttention(tid) {
    var t = findThread(tid);
    if (t && t.attention) { t.attention = false; }
  }
  function removeTrailingUser(tid) {
    var t = findThread(tid);
    if (!t || !t.history.length) return;
    if (t.history[t.history.length - 1].role === "user") {
      t.history.pop(); saveThreadsToStore();
    }
  }

  // ---- Clipboard ----
  function copyText(txt) {
    if (navigator.clipboard && navigator.clipboard.writeText)
      return navigator.clipboard.writeText(txt);
    return new Promise(function (res, rej) {
      try {
        var ta = document.createElement("textarea");
        ta.value = txt; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.focus(); ta.select();
        document.execCommand("copy"); document.body.removeChild(ta); res();
      } catch (e) { rej(e); }
    });
  }
  var COPY_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 012-2h8"/></svg>';
  var CHECK_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 13l4 4L19 7"/></svg>';
  function makeCopyBtn(getText, label) {
    var b = el("button", { className: "copy-btn",
      attrs: { type: "button", title: label || "Copy" }, html: COPY_SVG });
    b.addEventListener("click", function (ev) {
      ev.stopPropagation();
      copyText(getText() || "").then(function () {
        b.classList.add("copied"); b.innerHTML = CHECK_SVG;
        setTimeout(function () { b.classList.remove("copied"); b.innerHTML = COPY_SVG; }, 1200);
      }).catch(function () {});
    });
    return b;
  }

  // Minimal SSE-frame parser: "event:" + (possibly multi-line) "data:".
  function parseSSE(block) {
    var event = "message", data = "";
    var lines = block.split("\n");
    for (var i = 0; i < lines.length; i++) {
      var ln = lines[i];
      if (ln.indexOf("event:") === 0) event = ln.slice(6).trim();
      else if (ln.indexOf("data:") === 0) data += ln.slice(5).trim();
    }
    if (!data) return null;
    try { return { event: event, data: JSON.parse(data) }; }
    catch (e) { return { event: event, data: null }; }
  }

  function truncate(s, n) {
    s = (s || "").replace(/\s+/g, " ").trim();
    return s.length <= n ? s : s.slice(0, n - 1) + "...";
  }

  function relativeTime(ts) {
    var ms = Date.now() - ts;
    if (ms < 60000) return "just now";
    if (ms < 3600000) return Math.floor(ms / 60000) + "m ago";
    if (ms < 86400000) return Math.floor(ms / 3600000) + "h ago";
    if (ms < 604800000) return Math.floor(ms / 86400000) + "d ago";
    return new Date(ts).toLocaleDateString();
  }

  // ---------- Thread persistence ----------

  function loadThreadsFromStore() {
    try {
      var arr = JSON.parse(window.localStorage.getItem(STORE_KEY) || "[]");
      return Array.isArray(arr) ? arr : [];
    } catch (e) { return []; }
  }

  function saveThreadsToStore() {
    try {
      window.localStorage.setItem(STORE_KEY,
        JSON.stringify(state.threads.slice(0, THREAD_CAP)));
    } catch (e) { /* ignore */ }
  }

  function newThreadId() {
    return "t-" + Date.now() + "-" + Math.floor(Math.random() * 1e6);
  }

  function safeChunk(chunk) {
    if (!chunk || typeof chunk !== "object") return chunk;
    var out = {};
    for (var k in chunk) {
      if (!Object.prototype.hasOwnProperty.call(chunk, k)) continue;
      if (k.charAt(0) === "_") continue;
      var v = chunk[k], t = typeof v;
      if (v === null || t === "string" || t === "number" || t === "boolean") out[k] = v;
      else if (Array.isArray(v)) out[k] = v.slice();
      else if (t === "object") { try { out[k] = JSON.parse(JSON.stringify(v)); } catch (e) {} }
    }
    return out;
  }
  function safeChunks(chunks) {
    return Array.isArray(chunks) ? chunks.map(safeChunk) : [];
  }

  function findThread(id) {
    for (var i = 0; i < state.threads.length; i++)
      if (state.threads[i].id === id) return state.threads[i];
    return null;
  }

  function ensureActiveThread(corpus) {
    if (state.activeThreadId) {
      var t = findThread(state.activeThreadId);
      if (t) return t;
    }
    var now = Date.now();
    var thread = { id: newThreadId(), corpus: corpus || state.corpus || "",
                   ts_created: now, ts_last: now, title: "", history: [] };
    state.threads.unshift(thread);
    state.activeThreadId = thread.id;
    if (state.threads.length > THREAD_CAP) state.threads = state.threads.slice(0, THREAD_CAP);
    return thread;
  }

  function trimThreadHistory(thread) {
    if (thread.history.length > THREAD_HISTORY_CAP)
      thread.history = thread.history.slice(-THREAD_HISTORY_CAP);
  }

  function pushUserToThread(query) {
    var thread = ensureActiveThread(state.corpus);
    thread.history.push({ role: "user", content: query });
    if (!thread.title) thread.title = truncate(query, 60);
    thread.ts_last = Date.now();
    trimThreadHistory(thread);
    saveThreadsToStore();
    renderSidebar();
  }

  // (assistant turns are saved per-thread-id via pushAssistantToThreadById,
  // so a background chat's answer lands in the right thread even when you've
  // switched away.)

  // ---------- Sidebar ----------

  function renderSidebar() {
    els.threadList.innerHTML = "";
    if (!state.threads.length) {
      els.threadList.appendChild(el("li", { className: "empty-threads", text: "(no past chats)" }));
      return;
    }
    var limit = Math.min(state.threads.length, SIDEBAR_RENDER_CAP);
    for (var i = 0; i < limit; i++)
      els.threadList.appendChild(buildThreadRow(state.threads[i]));
  }

  var BELL_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 10-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 01-3.4 0"/></svg>';

  function buildThreadRow(thread) {
    var running = !!(state.jobs[thread.id] && !state.jobs[thread.id].done);
    var headKids = [
      el("span", { className: "row-corpus", text: thread.corpus || "?" }),
      el("span", { html: "&middot;" }),
      el("span", { className: "row-time", text: relativeTime(thread.ts_last) }),
    ];
    if (running)
      headKids.push(el("span", { className: "row-status",
        html: '<span class="mini-spinner"></span>',
        attrs: { title: "Working..." } }));
    else if (thread.attention)
      headKids.push(el("span", { className: "row-status bell", html: BELL_SVG,
        attrs: { title: "Answer ready" } }));
    var head = el("div", { className: "row-head" }, headKids);
    var title = el("div", { className: "row-title", text: thread.title || "(empty)" });
    var body = el("div", { className: "row-body" }, [head, title]);
    var del = el("button", {
      className: "row-delete",
      attrs: { type: "button", title: "Delete this thread" }, html: "&times;",
    });
    del.addEventListener("click", function (ev) {
      ev.stopPropagation();
      if (window.confirm("Delete this chat thread?")) deleteThread(thread.id);
    });
    var cls = "thread-row"
      + (thread.id === state.activeThreadId ? " active" : "")
      + (running ? " running" : "")
      + (thread.attention ? " attention" : "");
    var row = el("li", { className: cls, attrs: { "data-id": thread.id } }, [body, del]);
    row.addEventListener("click", function () { loadThread(thread.id); });
    return row;
  }

  function deleteThread(id) {
    var wasActive = id === state.activeThreadId;
    state.threads = state.threads.filter(function (t) { return t.id !== id; });
    saveThreadsToStore();
    if (wasActive) {
      state.activeThreadId = null;
      state.history = [];
      clearThreadDiv();
      showWelcome();
    }
    renderSidebar();
  }

  function clearThreadDiv() {
    els.thread.innerHTML = "";
    state.sourceMap = {};
  }

  function renderThreadHistory(thread) {
    for (var i = 0; i < thread.history.length; i++) {
      var entry = thread.history[i];
      if (entry.role === "user") {
        renderUserMessage(entry.content || "");
      } else if (entry.role === "assistant") {
        var envelope = {
          answer: entry.content || "", citations: entry.citations || [],
          authorities: entry.authorities || [],
          chunks: entry.chunks || [], refused: !!entry.refused,
          refusal_reason: entry.refusal_reason || "", status: entry.status || "",
          sources_only: !!entry.sources_only, tokens: entry.tokens || {},
          elapsed_ms: entry.elapsed_ms || 0,
        };
        var query = "";
        for (var j = i - 1; j >= 0; j--)
          if (thread.history[j].role === "user") { query = thread.history[j].content || ""; break; }
        renderAssistantMessage(envelope, query);
      }
    }
  }

  // Re-draw the active thread from the store (used after switching, aborting,
  // or a background answer landing in the thread you're viewing).
  function renderActiveThread() {
    var t = findThread(state.activeThreadId);
    clearThreadDiv();
    if (!t || !t.history.length) { showWelcome(); return; }
    showConversation();
    renderThreadHistory(t);
  }

  function loadThread(id) {
    var thread = findThread(id);
    if (!thread) return;
    state.activeThreadId = id;
    clearAttention(id);
    if (thread.corpus && thread.corpus !== state.corpus)
      setActiveCorpus(thread.corpus, false);
    renderActiveThread();
    renderSidebar();
    // Reflect THIS thread's job (if any) in the composer -- fixes the stale
    // spinner that used to show when opening a different chat.
    updateSendButton();
    renderActiveStatus();
    if (activeJob() && !activeJob().done) ensureTicker();
    closeSidebarIfMobile();
  }

  function pushAssistantToThreadById(tid, envelope) {
    var thread = findThread(tid);
    if (!thread) return;
    thread.history.push({
      role: "assistant", content: envelope.answer || "",
      citations: (envelope.citations || []).slice(),
      authorities: (envelope.authorities || []).slice(),
      chunks: safeChunks(envelope.chunks),
      refused: !!envelope.refused, refusal_reason: envelope.refusal_reason || "",
      status: envelope.status || "", sources_only: !!envelope.sources_only,
      tokens: envelope.tokens || null, elapsed_ms: envelope.elapsed_ms || 0,
    });
    thread.ts_last = Date.now();
    trimThreadHistory(thread);
    saveThreadsToStore();
  }

  // ---------- Sources popover (read-only; NOT a corpus switcher) ----------
  //
  // The chat is locked onto one unified corpus. The top-left chip just shows
  // which documents are searched for every answer; it never switches corpora.

  function prettyCorpus(slug) {
    if (!slug) return "Sources";
    return slug.replace(/[-_]/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  // Tiers the user has muted (included by default). Persisted across sessions.
  state.excludedTiers = loadLocalJson("docrag.excludedTiers", {});

  var TIER_ORDER = ["STATE", "LOCAL", "LOCAL-GUIDANCE", "MODEL", "FEDERAL", "OTHER"];

  function excludedSourceGlobs() {
    // Map muted tiers -> the member source paths to omit at retrieval time.
    var globs = [];
    state.sources.forEach(function (s) {
      if (state.excludedTiers[s.tier || "OTHER"] && s.path) globs.push(s.path);
    });
    return globs;
  }

  function renderSourcesList() {
    if (!els.sourcesList) return;
    els.sourcesList.innerHTML = "";
    if (!state.sources.length) {
      els.sourcesList.appendChild(el("li", { className: "sources-menu-empty", text: "(no documents indexed)" }));
      return;
    }
    // Group sources by authority tier; each group gets an include/omit checkbox.
    var groups = {};
    state.sources.forEach(function (s) {
      var t = s.tier || "OTHER";
      (groups[t] = groups[t] || { label: s.tier_label || t, items: [] }).items.push(s);
    });
    TIER_ORDER.concat(Object.keys(groups)).forEach(function (t) {
      if (!groups[t] || groups[t]._done) return;
      groups[t]._done = true;
      var g = groups[t], on = !state.excludedTiers[t];
      var cb = el("input", { attrs: { type: "checkbox" } });
      cb.checked = on;
      cb.addEventListener("change", function () {
        if (cb.checked) delete state.excludedTiers[t]; else state.excludedTiers[t] = true;
        saveLocal("docrag.excludedTiers", state.excludedTiers);
        renderSourcesList();
      });
      var head = el("li", { className: "sources-group-head" + (on ? "" : " muted") }, [
        el("label", {}, [cb, el("span", { text: " " + g.label + " (" + g.items.length + ")" })]),
      ]);
      els.sourcesList.appendChild(head);
      g.items.forEach(function (s) {
        els.sourcesList.appendChild(el("li", { className: "sources-menu-item" + (on ? "" : " muted") }, [
          el("span", { className: "src-file", text: s.file || "unknown" }),
          el("span", { className: "src-count", text: (s.chunks || 0).toLocaleString() }),
        ]));
      });
    });
  }

  function openPicker() {
    if (state.pickerOpen) return;
    els.pickerMenu.hidden = false;
    els.pickerBtn.setAttribute("aria-expanded", "true");
    state.pickerOpen = true;
  }
  function closePicker() {
    if (!state.pickerOpen) return;
    els.pickerMenu.hidden = true;
    els.pickerBtn.setAttribute("aria-expanded", "false");
    state.pickerOpen = false;
  }
  function togglePicker() { state.pickerOpen ? closePicker() : openPicker(); }

  function renderPickerButton() {
    els.pickerName.textContent = prettyCorpus(state.corpus);
    var n = state.sources.length;
    els.pickerCount.textContent = n ? (n + (n === 1 ? " doc" : " docs")) : "";
    updateGroundedText();
  }

  function setActiveCorpus(slug, resetHistory) {
    if (!slug) return;
    state.corpus = slug;
    if (els.corpus) els.corpus.value = slug;
    renderPickerButton();
    if (resetHistory) { state.history = []; state.activeThreadId = null; }
  }

  async function loadSources() {
    state.sources = [];
    try {
      var url = "/api/sources" + (state.corpus ? "?corpus=" + encodeURIComponent(state.corpus) : "");
      var r = await fetch(url);
      if (r.ok) {
        var data = await r.json();
        state.sources = data.sources || [];
      }
    } catch (e) { console.warn("sources load failed", e); }
    renderSourcesList();
    renderPickerButton();
  }

  function renderVersionsList() {
    if (!els.versionsList || !els.versionsSection) return;
    els.versionsList.innerHTML = "";
    var versioned = (state.versionDocs || []).filter(function (d) { return d.versioned; });
    if (!versioned.length) { els.versionsSection.hidden = true; return; }
    els.versionsSection.hidden = false;
    versioned.forEach(function (d) {
      var sel = el("select", { className: "version-select",
        attrs: { "data-doc": d.doc_key, "aria-label": d.doc_label + " edition" } });
      var chosen = state.versions[d.doc_key] || d.default_year;
      d.years.forEach(function (y) {
        var o = el("option", { text: y, attrs: { value: y } });
        if (String(y) === String(chosen)) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener("change", function () {
        state.versions[d.doc_key] = sel.value;
        saveLocal(LS_VERSIONS, state.versions);
      });
      els.versionsList.appendChild(el("li", { className: "version-item" }, [
        el("span", { className: "version-doc", text: d.doc_label }),
        sel,
      ]));
    });
  }

  function renderLocationSelect() {
    if (!els.locationSelect) return;
    var locs = state.locations || [];
    els.locationSelect.innerHTML = "";
    locs.forEach(function (loc) {
      var o = el("option", { text: loc.label, attrs: { value: loc.key } });
      if (loc.key === state.location) o.selected = true;
      els.locationSelect.appendChild(o);
    });
  }

  async function loadFacets() {
    try {
      var url = "/api/facets" + (state.corpus ? "?corpus=" + encodeURIComponent(state.corpus) : "");
      var r = await fetch(url);
      if (!r.ok) return;
      var data = await r.json();
      state.locations = data.locations || [];
      state.versionDocs = data.versions || [];
      // Restore persisted selections; default location from server.
      var savedLoc = loadLocalJson(LS_LOCATION, "");
      var valid = state.locations.some(function (l) { return l.key === savedLoc; });
      state.location = valid ? savedLoc : (data.default_location || (state.locations[0] || {}).key || "");
      state.versions = loadLocalJson(LS_VERSIONS, {}) || {};
      renderLocationSelect();
      renderVersionsList();
    } catch (e) { console.warn("facets load failed", e); }
  }

  async function loadCorpora() {
    try {
      var r = await fetch("/api/corpora");
      if (!r.ok) throw new Error("corpora HTTP " + r.status);
      var data = await r.json();
      var corpora = data.corpora || [];
      state.corpora = corpora.slice();   // full list (used by the upload modal)
      // Lock onto the server's primary corpus -- no switching in the UI.
      var primary = data.primary || (corpora.indexOf("building-codes") >= 0 ? "building-codes" : corpora[0]) || "";
      els.corpus.innerHTML = "";
      if (!primary) {
        els.corpus.appendChild(el("option", { text: "(no corpora)", attrs: { value: "" } }));
        els.pickerName.textContent = "(no corpora)";
        updateGroundedText();
        return;
      }
      var o = el("option", { text: primary, attrs: { value: primary } });
      o.selected = true;
      els.corpus.appendChild(o);
      state.corpus = primary;
      renderPickerButton();
      loadSources();
      loadFacets();
    } catch (e) {
      setStatus("corpus load failed");
      console.error(e);
    }
  }

  // ---------- Welcome state + relocatable composer ----------

  function greetingText() {
    var h = new Date().getHours();
    if (h < 12) return "Good morning";
    if (h < 18) return "Good afternoon";
    return "Good evening";
  }

  function updateGroundedText() {
    var corpus = state.corpus || "";
    if (els.welcomeSub) {
      els.welcomeSub.innerHTML = corpus
        ? "Ask anything grounded in the <b>" + escapeHtml(corpus) + "</b> documentation."
        : "Select a corpus to begin.";
    }
    if (els.composerGrounded) {
      if (corpus) {
        els.composerGrounded.textContent = "Grounded in " + corpus;
        els.composerGrounded.classList.remove("cold");
      } else {
        els.composerGrounded.textContent = "No corpus selected";
        els.composerGrounded.classList.add("cold");
      }
    }
  }

  function renderSuggestions() {
    if (!els.suggestions) return;
    els.suggestions.innerHTML = "";
    SUGGESTIONS.forEach(function (text) {
      var card = el("button", {
        className: "suggestion", attrs: { type: "button" },
      }, [
        el("span", { className: "s-arrow", html: "&#9656;" }),
        el("span", { text: text }),
      ]);
      card.addEventListener("click", function () {
        els.input.value = text;
        autoGrowInput();
        send();
      });
      els.suggestions.appendChild(card);
    });
  }

  function showWelcome() {
    document.body.classList.add("is-welcome");
    if (els.composer && els.composerDockWelcome &&
        els.composer.parentNode !== els.composerDockWelcome) {
      els.composerDockWelcome.appendChild(els.composer);
    }
    if (els.welcomeGreeting) els.welcomeGreeting.textContent = greetingText();
    updateGroundedText();
    renderSuggestions();
    autoGrowInput();
  }

  function showConversation() {
    document.body.classList.remove("is-welcome");
    if (els.composer && els.composerDockBottom &&
        els.composer.parentNode !== els.composerDockBottom) {
      els.composerDockBottom.appendChild(els.composer);
    }
    autoGrowInput();
  }

  function autoGrowInput() {
    var ta = els.input;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }

  // ---------- Render ----------

  function renderUserMessage(text) {
    var bubble = el("div", { className: "bubble" });
    bubble.appendChild(document.createTextNode(text));
    bubble.appendChild(makeCopyBtn(function () { return text; }, "Copy question"));
    els.thread.appendChild(el("div", { className: "msg user" }, [bubble]));
    scrollToBottom();
  }

  function sourceUrl(chunk) {
    var corpus = state.corpus || "", path = chunk.path || "";
    if (!corpus || !path) return null;
    var url = "/source?corpus=" + encodeURIComponent(corpus) +
              "&path=" + encodeURIComponent(path);
    var fname = (chunk.source_file || "").toLowerCase();
    if (fname.endsWith(".pdf") && chunk.page) url += "#page=" + chunk.page;
    return url;
  }

  function renderSourceCard(msgId, idx, chunk, query) {
    var kind = chunk.kind || "document";
    var score = typeof chunk.score === "number" ? chunk.score.toFixed(4) : "-";
    var fname = chunk.source_file || "unknown";
    var preview = truncate(chunk.text || "", PREVIEW_CHARS);
    var full = chunk.text || "";
    var openUrl = sourceUrl(chunk);

    var idxBadge = el("span", { className: "idx", text: String(idx) });

    var fnameSpan;
    if (openUrl) {
      fnameSpan = el("a", {
        className: "filename source-link", text: fname,
        attrs: { href: openUrl, target: "_blank", rel: "noopener",
                 title: "Open source" + (chunk.page ? " at page " + chunk.page : "") },
      });
      fnameSpan.addEventListener("click", function (ev) { ev.stopPropagation(); });
    } else {
      fnameSpan = el("span", { className: "filename", text: fname });
    }

    var page = (chunk.page == null) ? "-" : chunk.page;
    var pageEl;
    if (openUrl && chunk.page && fname.toLowerCase().endsWith(".pdf")) {
      pageEl = el("a", { className: "page-link", text: "p." + page,
        attrs: { href: openUrl, target: "_blank", rel: "noopener",
                 title: "Open at page " + chunk.page } });
      pageEl.addEventListener("click", function (ev) { ev.stopPropagation(); });
    } else {
      pageEl = el("span", { className: "page-static", text: "p." + page });
    }
    var meta = el("span", { className: "source-meta" },
      [pageEl, el("span", { text: "  kind=" + kind + "  score=" + score })]);

    var sectionEl = null;
    if (chunk.section_number) {
      var secTxt = "§ " + chunk.section_number +
        (chunk.section_title ? " " + chunk.section_title : "");
      sectionEl = el("span", { className: "src-section", text: secTxt });
    }
    var refBadge = chunk.referenced
      ? el("span", { className: "src-ref-badge",
          text: "cross-ref" + (chunk.referenced_by ? " ← " + chunk.referenced_by : "") })
      : null;

    var rateGood = el("button", {
      attrs: { type: "button", title: "Mark helpful. Click again to clear.", "data-rating": "good" }, text: "+" });
    var rateBad = el("button", {
      attrs: { type: "button", title: "Mark wrong. Click again to clear.", "data-rating": "bad" }, text: "-" });
    var rate = el("span", { className: "rate" }, [rateGood, rateBad]);

    var head = el("div", { className: "source-head" },
      [idxBadge, fnameSpan, sectionEl, refBadge, meta, rate]);
    var previewEl = el("div", { className: "preview", text: preview });
    var card = el("div", {
      className: "source-card",
      attrs: { "data-msg": msgId, "data-n": String(idx) },
    }, [head, previewEl]);
    card._currentRating = "";

    head.addEventListener("click", function (ev) {
      if (ev.target.closest("button, a")) return;
      if (card.classList.contains("expanded")) {
        card.classList.remove("expanded"); previewEl.textContent = preview;
      } else {
        card.classList.add("expanded"); previewEl.textContent = full;
      }
    });

    function onRate(target) {
      return function (ev) {
        ev.stopPropagation();
        var next = card._currentRating === target ? "clear" : target;
        submitRating(card, chunk, idx, query, next, rateGood, rateBad);
      };
    }
    rateGood.addEventListener("click", onRate("good"));
    rateBad.addEventListener("click", onRate("bad"));
    return card;
  }

  async function submitRating(card, chunk, idx, query, rating, goodBtn, badBtn) {
    var payload = {
      corpus: state.corpus, query: query, chunk_id: chunk.chunk_id,
      file: chunk.source_file, path: chunk.path, page: chunk.page,
      kind: chunk.kind, rank: idx, rating: rating,
    };
    try {
      var r = await fetch("/api/rate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error("rate HTTP " + r.status);
      goodBtn.classList.remove("rated-good");
      badBtn.classList.remove("rated-bad");
      if (rating === "good") { goodBtn.classList.add("rated-good"); card._currentRating = "good"; }
      else if (rating === "bad") { badBtn.classList.add("rated-bad"); card._currentRating = "bad"; }
      else card._currentRating = "";
    } catch (e) {
      console.error("rating failed", e);
      setStatus("rating failed");
    }
  }

  // ---------- Markdown ----------

  function inlineMd(s) {
    var safe = escapeHtml(s);
    var codes = [];
    safe = safe.replace(/`([^`]+)`/g, function (_m, b) { codes.push(b); return "\x00CODE" + (codes.length - 1) + "\x00"; });
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/(^|[\s(])\*([^\s*][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
    safe = safe.replace(/(^|[\s(])_([^\s_][^_]*?)_(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
    safe = safe.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    safe = safe.replace(/\x00CODE(\d+)\x00/g, function (_m, i) { return "<code>" + codes[parseInt(i, 10)] + "</code>"; });
    return safe;
  }

  function renderMarkdown(md) {
    if (!md) return "";
    var src = String(md);
    var fences = [];
    src = src.replace(/```([a-z0-9_-]*)\r?\n([\s\S]*?)```/gi, function (_m, _l, body) {
      fences.push(body); return "\n\x00FENCE" + (fences.length - 1) + "\x00\n";
    });
    var lines = src.split(/\r?\n/), out = [], listType = null, inQuote = false, olNext = 1;
    function closeList() { if (listType) { out.push("</" + listType + ">"); listType = null; } }
    function closeQuote() { if (inQuote) { out.push("</blockquote>"); inQuote = false; } }
    function closeBlocks() { closeList(); closeQuote(); }
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i], trimmed = line.trim();
      var fm = trimmed.match(/^\x00FENCE(\d+)\x00$/);
      if (fm) { closeBlocks(); out.push("<pre><code>" + escapeHtml(fences[parseInt(fm[1], 10)]) + "</code></pre>"); continue; }
      if (!trimmed) { closeBlocks(); continue; }
      var hm = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (hm) { closeBlocks(); olNext = 1; var lvl = Math.min(hm[1].length + 1, 6); out.push("<h" + lvl + ">" + inlineMd(hm[2]) + "</h" + lvl + ">"); continue; }
      if (trimmed.indexOf("> ") === 0 || trimmed === ">") {
        closeList(); if (!inQuote) { out.push("<blockquote>"); inQuote = true; }
        out.push("<p>" + inlineMd(trimmed.replace(/^>\s?/, "")) + "</p>"); continue;
      }
      var ulm = trimmed.match(/^[-*]\s+(.*)$/);
      if (ulm) { closeQuote(); if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; } out.push("<li>" + inlineMd(ulm[1]) + "</li>"); continue; }
      var olm = trimmed.match(/^\d+\.\s+(.*)$/);
      if (olm) { closeQuote(); if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; } out.push('<li value="' + olNext + '">' + inlineMd(olm[1]) + "</li>"); olNext++; continue; }
      if (listType && /^\s+/.test(line) && out.length) {
        var prev = out[out.length - 1];
        if (prev.indexOf("<li>") === 0) { out[out.length - 1] = prev.replace(/<\/li>$/, " " + inlineMd(trimmed) + "</li>"); continue; }
      }
      closeBlocks(); out.push("<p>" + inlineMd(trimmed) + "</p>");
    }
    closeBlocks();
    return out.join("\n");
  }

  function renderAnswerWithMarkdown(answerText, msgId, chunks) {
    if (!answerText) return "";
    var chunkCount = (chunks && chunks.length) || 0;
    var citations = [];
    var stashed = String(answerText).replace(/\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g, function (_m, inner) {
      var spans = [];
      inner.split(",").forEach(function (part) {
        var n = parseInt(part.trim(), 10);
        if (!isNaN(n) && n >= 1 && n <= chunkCount)
          spans.push('<span class="cite" data-msg="' + msgId + '" data-n="' + n + '">[' + n + "]</span>");
      });
      citations.push(spans.join(""));
      return "\x00CITE" + (citations.length - 1) + "\x00";
    });
    var html = renderMarkdown(stashed);
    return html.replace(/\x00CITE(\d+)\x00/g, function (_m, i) { return citations[parseInt(i, 10)]; });
  }

  function renderAssistantMessage(envelope, query) {
    var msgId = "m" + Date.now() + "-" + Math.floor(Math.random() * 1e6);
    var chunks = envelope.chunks || [];
    var msg = el("div", { className: "msg assistant", attrs: { "data-id": msgId } });

    var contentBox = null;
    if (envelope.refused) {
      contentBox = el("div", {
        className: "refusal " + (envelope.status === "no_results" ? "no_results" : ""),
        text: "REFUSED: " + (envelope.refusal_reason || envelope.status || "refused"),
      });
    } else if (envelope.answer) {
      contentBox = el("div", { className: "bubble",
        html: renderAnswerWithMarkdown(envelope.answer, msgId, chunks) });
    } else if (envelope.sources_only) {
      contentBox = el("div", { className: "bubble", text: "Sources only (no LLM call)." });
    }
    if (contentBox) {
      var copyable = envelope.answer || envelope.refusal_reason || "";
      if (copyable)
        contentBox.appendChild(makeCopyBtn(function () { return copyable; }, "Copy answer"));
      msg.appendChild(contentBox);
    }

    var authorities = envelope.authorities || [];
    if (!envelope.refused && authorities.length) {
      var authItems = authorities.map(function (a) {
        return el("li", { className: "authority" }, [
          el("span", { className: "cite", attrs: { "data-msg": msgId, "data-n": String(a.n) },
            text: "[" + a.n + "]" }),
          el("span", { text: " " + (a.designation || "") }),
        ]);
      });
      var authBox = el("div", { className: "authorities" }, [
        el("div", { className: "authorities-title", text: "Authorities cited" }),
        el("ul", { className: "authorities-list" }, authItems),
      ]);
      msg.appendChild(authBox);
    }

    if (chunks.length) {
      var cards = [null];
      var details = document.createElement("details");
      details.className = "sources-block";
      details.open = Boolean(envelope.refused || envelope.sources_only);
      for (var i = 0; i < chunks.length; i++) {
        var card = renderSourceCard(msgId, i + 1, chunks[i], query);
        cards.push(card);
        details.appendChild(card);
      }
      state.sourceMap[msgId] = cards;
      var summary = document.createElement("summary");
      summary.textContent = "Sources (" + chunks.length + ")";
      details.insertBefore(summary, details.firstChild);
      msg.appendChild(details);
    }

    // Friendly total generation time, shown under the finished answer.
    if (envelope.elapsed_ms && !envelope.sources_only) {
      msg.appendChild(el("div", { className: "answer-time", html:
        '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="13" r="8"/>'
        + '<path d="M12 9v4l2.5 2"/><path d="M9 2h6"/></svg>'
        + '<span>Generated in ' + escapeHtml(fmtDuration(envelope.elapsed_ms)) + '</span>' }));
    }

    var t = envelope.tokens || {};
    if (t.prompt || t.completion) {
      var bits = [];
      if (t.prompt) bits.push("prompt=" + t.prompt);
      if (t.completion) bits.push("completion=" + t.completion);
      if (envelope.citations && envelope.citations.length) bits.push("cites=" + envelope.citations.join(","));
      msg.appendChild(el("div", { className: "meta", text: bits.join(" | ") }));
    }

    msg.addEventListener("click", function (ev) {
      var target = ev.target;
      if (target && target.classList && target.classList.contains("cite"))
        scrollToSource(target.getAttribute("data-msg"), parseInt(target.getAttribute("data-n"), 10));
    });

    els.thread.appendChild(msg);
    scrollToBottom();
  }

  function renderErrorMessage(text) {
    els.thread.appendChild(el("div", { className: "msg assistant" },
      [el("div", { className: "refusal error", text: "ERROR: " + text })]));
    scrollToBottom();
  }

  function scrollToSource(msgId, n) {
    var cards = state.sourceMap[msgId];
    if (!cards || !cards[n]) return;
    var card = cards[n], details = card.closest("details");
    if (details && !details.open) details.open = true;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.classList.add("flash");
    setTimeout(function () { card.classList.remove("flash"); }, 1200);
  }

  function scrollToBottom() { els.thread.scrollTop = els.thread.scrollHeight; }

  // ---------- Send ----------

  function send() {
    var query = els.input.value.trim();
    if (!query) return;
    var corpus = els.corpus.value;
    if (!corpus) { showConversation(); renderErrorMessage("no corpus selected"); return; }

    var thread = ensureActiveThread(corpus);
    var tid = thread.id;
    if (state.jobs[tid] && !state.jobs[tid].done) return;  // one query per chat

    var sourcesOnly = els.sourcesOnly.checked;
    var topK = parseInt(els.topk.value, 10) || 12;

    // History for the server = this thread's prior turns (before the new one).
    var hist = [];
    for (var i = 0; i < thread.history.length; i++) {
      var e = thread.history[i];
      if ((e.role === "user" || e.role === "assistant") && e.content)
        hist.push({ role: e.role, content: e.content });
    }
    hist = hist.slice(-HISTORY_CAP);

    pushUserToThread(query);          // append + persist + refresh sidebar
    els.input.value = "";
    autoGrowInput();
    showConversation();
    if (tid === state.activeThreadId) renderUserMessage(query);

    var job = { threadId: tid, abort: new AbortController(),
                startedAt: Date.now(), done: false,
                statusText: "Thinking", sources: null, stageAt: Date.now() };
    state.jobs[tid] = job;
    updateSendButton();
    ensureTicker();
    renderActiveStatus();

    runJob(job, {
      corpus: corpus, query: query, history: hist,
      sources_only: sourcesOnly, top_k: topK, thread_id: tid,
    });
  }

  async function runJob(job, payload) {
    var tid = job.threadId;
    try {
      var r = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        signal: job.abort.signal,
        body: JSON.stringify({
          corpus: payload.corpus, query: payload.query, history: payload.history,
          sources_only: payload.sources_only, top_k: payload.top_k, stream: true,
          thread_id: payload.thread_id,
          location: state.location || undefined, versions: state.versions || {},
          exclude_sources: excludedSourceGlobs() }),
      });
      var ctype = r.headers.get("Content-Type") || "";
      var data;
      if (ctype.indexOf("text/event-stream") >= 0 && r.body && r.body.getReader) {
        var reader = r.body.getReader();
        var dec = new TextDecoder();
        var buf = "", finalEnv = null, errMsg = null;
        while (true) {
          var chunk = await reader.read();
          if (chunk.done) break;
          buf += dec.decode(chunk.value, { stream: true });
          var idx;
          while ((idx = buf.indexOf("\n\n")) >= 0) {
            var frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
            var ev = parseSSE(frame);
            if (!ev) continue;
            if (ev.event === "progress") {
              var p = ev.data || {};
              job.statusText = p.message || "Thinking";
              if (p.sources && p.sources.length) { job.sources = p.sources; job.stageAt = Date.now(); }
              else job.sources = null;
              if (tid === state.activeThreadId) renderActiveStatus();
            } else if (ev.event === "done") finalEnv = ev.data;
            else if (ev.event === "error") errMsg = (ev.data && ev.data.error) || "error";
          }
        }
        if (errMsg) throw new Error(errMsg);
        if (!finalEnv) throw new Error("stream ended without an answer");
        data = finalEnv;
      } else {
        try { data = await r.json(); }
        catch (e) { throw new Error("bad JSON from server (HTTP " + r.status + ")"); }
        if (!r.ok) throw new Error(data.error || "HTTP " + r.status);
      }
      job.done = true;
      pushAssistantToThreadById(tid, data);
      if (tid === state.activeThreadId) renderAssistantMessage(data, payload.query);
      else markAttention(tid);          // flash + bell on the background chat
    } catch (e) {
      job.done = true;
      if (e && e.name === "AbortError") {
        removeTrailingUser(tid);          // drop the cancelled question
        if (tid === state.activeThreadId) renderActiveThread();
      } else {
        console.error(e);
        if (tid === state.activeThreadId) renderErrorMessage(e.message || String(e));
        else markAttention(tid);
      }
    } finally {
      delete state.jobs[tid];
      renderSidebar();
      if (tid === state.activeThreadId) {
        updateSendButton();
        renderActiveStatus();
        els.input.focus();
      }
    }
  }

  // ---------- Sidebar collapse + new chat ----------

  function isMobile() {
    return window.matchMedia("(max-width: 860px)").matches;
  }
  function setSidebarCollapsed(collapsed) {
    els.sidebar.classList.toggle("collapsed", collapsed);
    document.body.classList.toggle("sidebar-collapsed", collapsed);
  }
  function applySidebarState() {
    // On phones the sidebar is an off-canvas overlay -- always start closed,
    // regardless of the desktop preference, so it never covers the chat.
    if (isMobile()) { setSidebarCollapsed(true); return; }
    var collapsed = false;
    try { collapsed = window.localStorage.getItem(SIDEBAR_KEY) === "1"; } catch (e) {}
    setSidebarCollapsed(collapsed);
  }
  function toggleSidebar() {
    var collapsed = els.sidebar.classList.toggle("collapsed");
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    // Only persist the preference on desktop; mobile always reopens closed.
    if (!isMobile()) {
      try { window.localStorage.setItem(SIDEBAR_KEY, collapsed ? "1" : "0"); } catch (e) {}
    }
  }
  // After navigating on mobile, slide the overlay sidebar back out of the way.
  function closeSidebarIfMobile() {
    if (isMobile() && !els.sidebar.classList.contains("collapsed")) setSidebarCollapsed(true);
  }
  function newChat() {
    state.activeThreadId = null;     // no job -> composer is idle
    state.history = [];
    clearThreadDiv();
    showWelcome();
    renderSidebar();
    updateSendButton();
    renderActiveStatus();
    closeSidebarIfMobile();
    els.input.focus();
  }

  // ---------- Wire up ----------

  els.send.addEventListener("click", onSendClick);
  els.input.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      if (activeJob() && !activeJob().done) return;  // a query is already running in this chat
      send();
    }
  });
  els.input.addEventListener("input", autoGrowInput);

  els.topk.addEventListener("input", function () { els.topkValue.textContent = els.topk.value; });

  // Sources popover open/close + outside-click + Escape (read-only).
  els.pickerBtn.addEventListener("click", function (ev) { ev.stopPropagation(); togglePicker(); });
  document.addEventListener("click", function (ev) {
    if (!state.pickerOpen) return;
    if (els.pickerMenu.contains(ev.target) || els.pickerBtn.contains(ev.target)) return;
    closePicker();
  });
  document.addEventListener("keydown", function (ev) {
    if (state.pickerOpen && ev.key === "Escape") { closePicker(); ev.preventDefault(); }
  });

  els.sidebarToggle.addEventListener("click", toggleSidebar);
  if (els.sidebarToggleFloating) els.sidebarToggleFloating.addEventListener("click", toggleSidebar);
  if (els.sidebarBackdrop) els.sidebarBackdrop.addEventListener("click", closeSidebarIfMobile);
  els.newChatBtn.addEventListener("click", newChat);
  // If the viewport crosses the mobile breakpoint, re-apply the correct state
  // (e.g. rotating a tablet, or resizing a desktop window down).
  window.addEventListener("resize", applySidebarState);

  // ---------- Settings popover ----------

  function toggleSettingsMenu(force) {
    var open = force !== undefined ? force : els.settingsMenu.hidden;
    els.settingsMenu.hidden = !open;
    els.settingsBtn.setAttribute("aria-expanded", open ? "true" : "false");
  }
  if (els.settingsBtn) {
    els.settingsBtn.addEventListener("click", function (ev) { ev.stopPropagation(); toggleSettingsMenu(); });
    document.addEventListener("click", function (ev) {
      if (els.settingsMenu.hidden) return;
      if (els.settingsMenu.contains(ev.target) || els.settingsBtn.contains(ev.target)) return;
      toggleSettingsMenu(false);
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && !els.settingsMenu.hidden) toggleSettingsMenu(false);
    });
  }

  // ---------- Upload modal ----------

  function openUploadModal() {
    populateUploadCorpusSelect();
    els.uploadStatus.textContent = "";
    els.uploadStatus.className = "upload-status";
    els.uploadFile.value = "";
    els.uploadNewCorpus.value = "";
    els.uploadModeExisting.checked = true;
    els.uploadModeNew.checked = false;
    applyUploadMode();
    els.uploadModal.hidden = false;
  }
  function closeUploadModal() { els.uploadModal.hidden = true; }

  function applyUploadMode() {
    var mode = els.uploadModeNew.checked ? "new" : "existing";
    els.uploadExistingRow.hidden = mode !== "existing";
    els.uploadNewRow.hidden = mode !== "new";
    if (mode === "new") { try { els.uploadNewCorpus.focus(); } catch (e) {} }
    else els.uploadNewCorpus.value = "";
  }

  function populateUploadCorpusSelect() {
    var sel = els.uploadCorpus;
    sel.innerHTML = "";
    var current = state.corpus || "";
    var corpora = (state.corpora || []).slice();
    if (!corpora.length) {
      var o = document.createElement("option");
      o.value = ""; o.textContent = "(no corpora yet)"; o.disabled = true;
      sel.appendChild(o);
      return;
    }
    if (current && corpora.indexOf(current) === -1) corpora.unshift(current);
    corpora.forEach(function (c) {
      var o = document.createElement("option");
      o.value = c; o.textContent = c;
      if (c === current) o.selected = true;
      sel.appendChild(o);
    });
  }

  function setUploadStatus(msg, level) {
    els.uploadStatus.textContent = msg;
    els.uploadStatus.className = "upload-status " + (level || "");
  }

  async function submitUpload() {
    var corpus;
    if (els.uploadModeNew.checked) {
      corpus = (els.uploadNewCorpus.value || "").trim().toLowerCase();
      if (!/^[a-z0-9][a-z0-9_-]*$/.test(corpus)) {
        setUploadStatus("New corpus name must be lowercase a-z, 0-9, _, -", "error");
        return;
      }
    } else {
      corpus = (els.uploadCorpus.value || "").trim().toLowerCase();
      if (!corpus) { setUploadStatus("Pick an existing corpus.", "error"); return; }
    }
    var file = els.uploadFile.files && els.uploadFile.files[0];
    if (!file) { setUploadStatus("Pick a file first.", "error"); return; }
    if (file.size > 100 * 1024 * 1024) { setUploadStatus("File too large (max 100 MB).", "error"); return; }

    var fd = new FormData();
    fd.append("corpus", corpus);
    fd.append("file", file, file.name);

    els.uploadSubmit.disabled = true;
    setUploadStatus("Uploading and indexing... this can take 1-10 min for PDFs. Don't close this window.", "busy");
    try {
      var r = await fetch("/api/upload", { method: "POST", body: fd });
      var data;
      try { data = await r.json(); } catch (e) { throw new Error("server returned non-JSON"); }
      if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
      var msg = "Done. Corpus `" + data.corpus + "`" + (data.new_corpus ? " (created)" : "") +
                ", file `" + data.filename + "`, " + data.size_bytes + " bytes. " +
                "Indexer exit " + data.indexer_exit + ".";
      setUploadStatus(msg, data.ok ? "ok" : "error");
      if (data.ok) { state.corpus = corpus; loadCorpora(); }
      if (data.indexer_summary) {
        var pre = document.createElement("pre");
        pre.className = "upload-summary";
        pre.textContent = data.indexer_summary;
        els.uploadStatus.appendChild(pre);
      }
    } catch (e) {
      console.error(e);
      setUploadStatus("Upload failed: " + (e.message || e), "error");
    } finally {
      els.uploadSubmit.disabled = false;
    }
  }

  if (els.uploadBtn) {
    els.uploadBtn.addEventListener("click", openUploadModal);
    els.uploadCancel.addEventListener("click", closeUploadModal);
    els.uploadModeExisting.addEventListener("change", applyUploadMode);
    els.uploadModeNew.addEventListener("change", applyUploadMode);
    els.uploadSubmit.addEventListener("click", submitUpload);
    els.uploadModal.addEventListener("click", function (ev) {
      if (ev.target === els.uploadModal) closeUploadModal();
    });
  }

  // ---------- Help modal ----------

  if (els.helpBtn && els.helpModal) {
    els.helpBtn.addEventListener("click", function () { els.helpModal.hidden = false; });
    if (els.helpCancel) els.helpCancel.addEventListener("click", function () { els.helpModal.hidden = true; });
    els.helpModal.addEventListener("click", function (ev) {
      if (ev.target === els.helpModal) els.helpModal.hidden = true;
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && !els.helpModal.hidden) els.helpModal.hidden = true;
    });
  }

  if (els.locationSelect) {
    els.locationSelect.addEventListener("change", function () {
      state.location = els.locationSelect.value;
      saveLocal(LS_LOCATION, state.location);
    });
  }

  // ---------- Init ----------
  state.threads = loadThreadsFromStore();
  applySidebarState();
  renderSidebar();
  showWelcome();
  loadCorpora();
})();
