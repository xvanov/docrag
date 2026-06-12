/* docrag chat -- client logic. Plain DOM, no framework.
 *
 * State:
 *   - corpus: selected corpus slug.
 *   - history: last 6 {role, content} turns sent to the server.
 *   - threads: chats persisted in localStorage.
 */

(function () {
  "use strict";

  var HISTORY_CAP = 6;
  var THREAD_HISTORY_CAP = 12;
  var THREAD_CAP = 50;
  var SIDEBAR_RENDER_CAP = 30;
  var PREVIEW_CHARS = 300;
  var STORE_KEY = "docrag_threads_v1";
  var SIDEBAR_KEY = "docrag_sidebar_collapsed_v1";

  var state = {
    corpus: "",
    history: [],
    inflight: false,
    sourceMap: {},
    corpora: [],
    counts: {},          // corpus -> chunk count (null = unknown)
    threads: [],
    activeThreadId: null,
    pickerOpen: false,
    hoverIdx: -1,
  };

  var els = {
    thread: document.getElementById("thread"),
    input: document.getElementById("input"),
    send: document.getElementById("send"),
    status: document.getElementById("status"),
    corpus: document.getElementById("corpus-select"),
    sourcesOnly: document.getElementById("sources-only"),
    topk: document.getElementById("topk"),
    topkValue: document.getElementById("topk-value"),
    sidebar: document.getElementById("history-sidebar"),
    sidebarToggle: document.getElementById("sidebar-toggle"),
    threadList: document.getElementById("thread-list"),
    newChatBtn: document.getElementById("new-chat"),
    pickerBtn: document.getElementById("corpus-picker-btn"),
    pickerName: document.getElementById("corpus-picker-name"),
    pickerCount: document.getElementById("corpus-picker-count"),
    pickerMenu: document.getElementById("corpus-picker-menu"),
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
    if (withSpinner)
      els.status.innerHTML = '<span class="spinner"></span>' + escapeHtml(text);
    else els.status.textContent = text || "";
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

  function pushAssistantToThread(envelope) {
    if (!state.activeThreadId) return;
    var thread = findThread(state.activeThreadId);
    if (!thread) return;
    thread.history.push({
      role: "assistant", content: envelope.answer || "",
      citations: (envelope.citations || []).slice(),
      chunks: safeChunks(envelope.chunks),
      refused: !!envelope.refused, refusal_reason: envelope.refusal_reason || "",
      status: envelope.status || "", sources_only: !!envelope.sources_only,
      tokens: envelope.tokens || null, elapsed_ms: envelope.elapsed_ms || 0,
    });
    thread.ts_last = Date.now();
    trimThreadHistory(thread);
    saveThreadsToStore();
    renderSidebar();
  }

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

  function buildThreadRow(thread) {
    var head = el("div", { className: "row-head" }, [
      el("span", { className: "row-brand", text: thread.corpus || "?" }),
      el("span", { html: "&middot;" }),
      el("span", { className: "row-time", text: relativeTime(thread.ts_last) }),
    ]);
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
    var row = el("li", {
      className: "thread-row" + (thread.id === state.activeThreadId ? " active" : ""),
      attrs: { "data-id": thread.id },
    }, [body, del]);
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
      showEmptyState();
    }
    renderSidebar();
  }

  function clearThreadDiv() {
    els.thread.innerHTML = "";
    state.sourceMap = {};
  }

  function loadThread(id) {
    var thread = findThread(id);
    if (!thread) return;
    state.activeThreadId = id;
    state.history = [];
    clearThreadDiv();
    if (thread.corpus && thread.corpus !== state.corpus)
      setActiveCorpus(thread.corpus, false);
    for (var i = 0; i < thread.history.length; i++) {
      var entry = thread.history[i];
      if (entry.role === "user") {
        renderUserMessage(entry.content || "");
      } else if (entry.role === "assistant") {
        var envelope = {
          answer: entry.content || "", citations: entry.citations || [],
          chunks: entry.chunks || [], refused: !!entry.refused,
          refusal_reason: entry.refusal_reason || "", status: entry.status || "",
          sources_only: !!entry.sources_only, tokens: entry.tokens || {},
          elapsed_ms: entry.elapsed_ms || 0,
        };
        var query = "";
        for (var j = i - 1; j >= 0; j--)
          if (thread.history[j].role === "user") { query = thread.history[j].content || ""; break; }
        renderAssistantMessage(envelope, query);
        if (entry.content) {
          state.history.push({ role: "user", content: query });
          state.history.push({ role: "assistant", content: entry.content });
        }
      }
    }
    if (state.history.length > HISTORY_CAP) state.history = state.history.slice(-HISTORY_CAP);
    renderSidebar();
  }

  // ---------- Corpus picker ----------

  function formatCount(n) {
    if (n === null || n === undefined) return "";
    return n.toLocaleString() + " chunks";
  }

  function renderPickerMenu() {
    els.pickerMenu.innerHTML = "";
    for (var i = 0; i < state.corpora.length; i++) {
      var slug = state.corpora[i];
      var item = el("li", {
        className: "brand-picker-item" + (slug === state.corpus ? " active" : ""),
        attrs: { "data-slug": slug, role: "option" },
      }, [
        el("span", { className: "picker-brand", text: slug }),
        el("span", { className: "picker-count", text: formatCount(state.counts[slug]) }),
      ]);
      (function (s, node) {
        node.addEventListener("click", function () { selectCorpus(s); closePicker(); });
        node.addEventListener("mouseenter", function () {
          state.hoverIdx = indexOfMenuItem(node); updateHover();
        });
      })(slug, item);
      els.pickerMenu.appendChild(item);
    }
  }

  function indexOfMenuItem(node) {
    var items = els.pickerMenu.querySelectorAll(".brand-picker-item");
    for (var i = 0; i < items.length; i++) if (items[i] === node) return i;
    return -1;
  }

  function updateHover() {
    var items = els.pickerMenu.querySelectorAll(".brand-picker-item");
    for (var i = 0; i < items.length; i++)
      items[i].classList.toggle("hover", i === state.hoverIdx);
  }

  function openPicker() {
    if (state.pickerOpen) return;
    renderPickerMenu();
    els.pickerMenu.hidden = false;
    els.pickerBtn.setAttribute("aria-expanded", "true");
    state.pickerOpen = true;
    state.hoverIdx = Math.max(0, state.corpora.indexOf(state.corpus));
    updateHover();
  }

  function closePicker() {
    if (!state.pickerOpen) return;
    els.pickerMenu.hidden = true;
    els.pickerBtn.setAttribute("aria-expanded", "false");
    state.pickerOpen = false;
    state.hoverIdx = -1;
  }

  function togglePicker() { state.pickerOpen ? closePicker() : openPicker(); }

  function renderPickerButton() {
    els.pickerName.textContent = state.corpus || "(none)";
    els.pickerCount.textContent = formatCount(state.counts[state.corpus]);
  }

  function selectCorpus(slug) { setActiveCorpus(slug, true); }

  function setActiveCorpus(slug, resetHistory) {
    if (!slug) return;
    state.corpus = slug;
    if (els.corpus) els.corpus.value = slug;
    renderPickerButton();
    if (resetHistory) { state.history = []; state.activeThreadId = null; }
  }

  async function loadCorpora() {
    try {
      var r = await fetch("/api/corpora");
      if (!r.ok) throw new Error("corpora HTTP " + r.status);
      var data = await r.json();
      var corpora = data.corpora || [];
      state.corpora = corpora.slice();
      els.corpus.innerHTML = "";
      if (!corpora.length) {
        els.corpus.appendChild(el("option", { text: "(no corpora)", attrs: { value: "" } }));
        els.pickerName.textContent = "(no corpora)";
        showEmptyState();
        return;
      }
      var def = state.corpus && corpora.indexOf(state.corpus) >= 0 ? state.corpus : corpora[0];
      for (var i = 0; i < corpora.length; i++) {
        var o = el("option", { text: corpora[i], attrs: { value: corpora[i] } });
        if (corpora[i] === def) o.selected = true;
        els.corpus.appendChild(o);
      }
      state.corpus = def;
      renderPickerButton();
    } catch (e) {
      setStatus("corpus load failed");
      console.error(e);
    }
  }

  function showEmptyState() {
    els.thread.appendChild(el("div", { className: "empty" }, [
      el("h2", { text: "Ask the documents anything." }),
      el("div", {
        text: "Answers are grounded in the indexed corpus. " +
              "Citations [N] link to source chunks below the answer.",
      }),
    ]));
  }

  // ---------- Render ----------

  function renderUserMessage(text) {
    els.thread.appendChild(el("div", { className: "msg user" },
      [el("div", { className: "bubble", text: text })]));
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

    var rateGood = el("button", {
      attrs: { type: "button", title: "Mark helpful. Click again to clear.", "data-rating": "good" }, text: "+" });
    var rateBad = el("button", {
      attrs: { type: "button", title: "Mark wrong. Click again to clear.", "data-rating": "bad" }, text: "-" });
    var rate = el("span", { className: "rate" }, [rateGood, rateBad]);

    var head = el("div", { className: "source-head" }, [idxBadge, fnameSpan, meta, rate]);
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
    var lines = src.split(/\r?\n/), out = [], listType = null, inQuote = false;
    function closeList() { if (listType) { out.push("</" + listType + ">"); listType = null; } }
    function closeQuote() { if (inQuote) { out.push("</blockquote>"); inQuote = false; } }
    function closeBlocks() { closeList(); closeQuote(); }
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i], trimmed = line.trim();
      var fm = trimmed.match(/^\x00FENCE(\d+)\x00$/);
      if (fm) { closeBlocks(); out.push("<pre><code>" + escapeHtml(fences[parseInt(fm[1], 10)]) + "</code></pre>"); continue; }
      if (!trimmed) { closeBlocks(); continue; }
      var hm = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (hm) { closeBlocks(); var lvl = Math.min(hm[1].length + 1, 6); out.push("<h" + lvl + ">" + inlineMd(hm[2]) + "</h" + lvl + ">"); continue; }
      if (trimmed.indexOf("> ") === 0 || trimmed === ">") {
        closeList(); if (!inQuote) { out.push("<blockquote>"); inQuote = true; }
        out.push("<p>" + inlineMd(trimmed.replace(/^>\s?/, "")) + "</p>"); continue;
      }
      var ulm = trimmed.match(/^[-*]\s+(.*)$/);
      if (ulm) { closeQuote(); if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; } out.push("<li>" + inlineMd(ulm[1]) + "</li>"); continue; }
      var olm = trimmed.match(/^\d+\.\s+(.*)$/);
      if (olm) { closeQuote(); if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; } out.push("<li>" + inlineMd(olm[1]) + "</li>"); continue; }
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

    if (envelope.refused) {
      msg.appendChild(el("div", {
        className: "refusal " + (envelope.status === "no_results" ? "no_results" : ""),
        text: "REFUSED: " + (envelope.refusal_reason || envelope.status || "refused"),
      }));
    } else if (envelope.answer) {
      msg.appendChild(el("div", { className: "bubble",
        html: renderAnswerWithMarkdown(envelope.answer, msgId, chunks) }));
    } else if (envelope.sources_only) {
      msg.appendChild(el("div", { className: "bubble", text: "Sources only (no LLM call)." }));
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

    var t = envelope.tokens || {};
    if (t.prompt || t.completion || envelope.elapsed_ms) {
      var bits = [];
      if (t.prompt) bits.push("prompt=" + t.prompt);
      if (t.completion) bits.push("completion=" + t.completion);
      if (envelope.elapsed_ms) bits.push(envelope.elapsed_ms + "ms");
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

  async function send() {
    if (state.inflight) return;
    var query = els.input.value.trim();
    if (!query) return;
    var corpus = els.corpus.value;
    if (!corpus) { renderErrorMessage("no corpus selected"); return; }
    var sourcesOnly = els.sourcesOnly.checked;
    var topK = parseInt(els.topk.value, 10) || 12;

    if (!state.activeThreadId) clearThreadDiv();
    else { var empty = els.thread.querySelector(".empty"); if (empty) empty.remove(); }

    renderUserMessage(query);
    els.input.value = "";
    pushUserToThread(query);

    var historyForServer = state.history.slice(-HISTORY_CAP);
    state.inflight = true;
    els.send.disabled = true;
    setStatus("thinking", true);

    try {
      var r = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corpus: corpus, query: query,
          history: historyForServer, sources_only: sourcesOnly, top_k: topK }),
      });
      var data;
      try { data = await r.json(); }
      catch (e) { throw new Error("bad JSON from server (HTTP " + r.status + ")"); }
      if (!r.ok) throw new Error(data.error || "HTTP " + r.status);
      renderAssistantMessage(data, query);
      pushAssistantToThread(data);
      if (data.answer) {
        state.history.push({ role: "user", content: query });
        state.history.push({ role: "assistant", content: data.answer });
        if (state.history.length > HISTORY_CAP) state.history = state.history.slice(-HISTORY_CAP);
      }
    } catch (e) {
      console.error(e);
      renderErrorMessage(e.message || String(e));
    } finally {
      state.inflight = false;
      els.send.disabled = false;
      setStatus("");
      els.input.focus();
    }
  }

  // ---------- Sidebar collapse + new chat ----------

  function applySidebarState() {
    var collapsed = false;
    try { collapsed = window.localStorage.getItem(SIDEBAR_KEY) === "1"; } catch (e) {}
    els.sidebar.classList.toggle("collapsed", collapsed);
  }
  function toggleSidebar() {
    var collapsed = els.sidebar.classList.toggle("collapsed");
    try { window.localStorage.setItem(SIDEBAR_KEY, collapsed ? "1" : "0"); } catch (e) {}
  }
  function newChat() {
    state.activeThreadId = null;
    state.history = [];
    clearThreadDiv();
    showEmptyState();
    renderSidebar();
    els.input.focus();
  }

  // ---------- Wire up ----------

  els.send.addEventListener("click", send);
  els.input.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); send(); }
  });
  els.corpus.addEventListener("change", function () {
    if (els.corpus.value && els.corpus.value !== state.corpus) {
      state.corpus = els.corpus.value;
      state.history = [];
      state.activeThreadId = null;
      renderPickerButton();
    }
  });
  els.topk.addEventListener("input", function () { els.topkValue.textContent = els.topk.value; });

  els.pickerBtn.addEventListener("click", function (ev) { ev.stopPropagation(); togglePicker(); });
  document.addEventListener("click", function (ev) {
    if (!state.pickerOpen) return;
    if (els.pickerMenu.contains(ev.target) || els.pickerBtn.contains(ev.target)) return;
    closePicker();
  });
  document.addEventListener("keydown", function (ev) {
    if (!state.pickerOpen) return;
    var items = els.pickerMenu.querySelectorAll(".brand-picker-item");
    if (ev.key === "Escape") { closePicker(); ev.preventDefault(); }
    else if (ev.key === "ArrowDown") { state.hoverIdx = Math.min(items.length - 1, state.hoverIdx + 1); updateHover(); ev.preventDefault(); }
    else if (ev.key === "ArrowUp") { state.hoverIdx = Math.max(0, state.hoverIdx - 1); updateHover(); ev.preventDefault(); }
    else if (ev.key === "Enter") {
      if (state.hoverIdx >= 0 && items[state.hoverIdx]) {
        var slug = items[state.hoverIdx].getAttribute("data-slug");
        if (slug) { selectCorpus(slug); closePicker(); }
      }
      ev.preventDefault();
    }
  });

  els.sidebarToggle.addEventListener("click", toggleSidebar);
  els.newChatBtn.addEventListener("click", newChat);

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

  // ---------- Init ----------
  state.threads = loadThreadsFromStore();
  applySidebarState();
  renderSidebar();
  showEmptyState();
  loadCorpora();
})();
