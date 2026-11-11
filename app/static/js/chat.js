// Chat: thread rendering, streaming answers, fallback card, citations, source drawer.

import { askStream, request } from "./api.js";
import { emit, keyHeaders, readyDocs, state } from "./state.js";
import { escapeHtml, highlightWithSpan, icon, js, pageLabel, renderAnswer, showDrawer, toast } from "./ui.js";

const thread = js("thread");
const scroller = js("scroller");
const question = js("question");

/* ---------- empty state ---------- */

const SAMPLE_QUESTIONS = {
  "halcyon_h2_handbook.pdf": [
    "How long is the battery warranty?",
    "What does a solid red light on the battery mean?",
    "How quickly does the battery take over when the grid fails?",
  ],
};

function suggestionsFor(doc) {
  if (SAMPLE_QUESTIONS[doc.filename]) return SAMPLE_QUESTIONS[doc.filename];
  const name = doc.filename.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ");
  return [`What is "${name}" about?`, `Summarize the key points of "${name}".`, `What numbers or deadlines does "${name}" mention?`];
}

export function renderEmpty() {
  const box = js("empty");
  if (thread.children.length) { box.innerHTML = ""; return; }
  const ready = readyDocs();
  const processing = state.docs.some((d) => d.status === "processing");
  if (!ready.length) {
    box.innerHTML = `<div class="c-empty">
      <span class="c-empty__icon">${icon("upload", "c-icon--lg")}</span>
      <h2 class="c-title">${processing ? "Getting your PDF ready" : "Upload a PDF to start"}</h2>
      <p class="c-text u-muted">${processing ? "This usually takes a few seconds. You can watch the progress in the documents panel." : "Answers come only from your documents, with a link to the exact page."}</p>
      ${processing ? "" : `<button type="button" class="c-btn c-btn--primary" data-js="empty-upload">${icon("upload")}Upload PDF</button>`}
    </div>`;
    js("empty-upload", box)?.addEventListener("click", () => js("file-input").click());
    return;
  }
  const doc = ready[0];
  box.innerHTML = `<div class="c-empty">
    <span class="c-empty__icon">${icon("chat", "c-icon--lg")}</span>
    <h2 class="c-title">Ask about your documents</h2>
    <p class="c-text u-muted">Every answer links to the page it came from. Try one of these:</p>
    <div class="c-empty__samples">${suggestionsFor(doc).map((q) =>
      `<button type="button" class="c-sample" data-sample="${escapeHtml(q)}">${icon("chat")}<span>${escapeHtml(q)}</span></button>`).join("")}</div>
  </div>`;
}

js("empty").addEventListener("click", (e) => {
  const b = e.target.closest("[data-sample]");
  if (b && !state.busy) ask(b.dataset.sample);
});

/* ---------- messages ---------- */

function scrollDown(force = false) {
  if (force || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 160) scroller.scrollTop = scroller.scrollHeight;
}

function addUser(text) {
  const el = document.createElement("div");
  el.className = "c-msg c-msg--user";
  el.innerHTML = `<p class="sr-only">You asked:</p><div class="c-msg__body">${escapeHtml(text)}</div>`;
  thread.appendChild(el);
  renderEmpty();
}

function sourcesRow(citations) {
  if (!citations.length) return "";
  return `<div class="c-msg__sources"><span class="c-meta">Sources</span>${citations.map((c) => `
    <button type="button" class="c-chip c-chip--source" data-cite="${c.n}">
      <span class="c-chip__n">[${c.n}]</span><span class="u-truncate">${escapeHtml(c.filename)}</span><span class="c-chip__page">${pageLabel(c.page_start, c.page_end)}</span>
    </button>`).join("")}</div>`;
}

function debugTable(retrieved, citations) {
  if (!retrieved || !retrieved.length) return "";
  const cited = new Set(citations.map((c) => c.n));
  const f = (v, d) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
  return `<div class="c-table-wrap" data-js="debug"${state.debug ? "" : " hidden"}><table class="c-table">
    <caption class="sr-only">Retrieval scores for the top sources</caption>
    <thead><tr><th>#</th><th>Source</th><th class="num">Vector</th><th class="num">BM25</th><th class="num">RRF</th><th class="num">Rerank</th></tr></thead>
    <tbody>${retrieved.map((r) => `<tr${cited.has(r.n) ? ' aria-current="true"' : ""}>
      <td>${r.n}${cited.has(r.n) ? ' <span class="sr-only">(cited)</span>✓' : ""}</td>
      <td class="u-truncate" style="max-width: var(--size-popover)">${escapeHtml(r.filename)} · ${pageLabel(r.page_start, r.page_end)}</td>
      <td class="num">${f(r.vector_score, 3)}</td><td class="num">${f(r.bm25_score, 2)}</td>
      <td class="num">${f(r.rrf_score, 4)}</td><td class="num">${f(r.rerank_score, 2)}</td></tr>`).join("")}</tbody>
  </table></div>`;
}

function fallbackCard() {
  return `<div class="c-fallback">
    <p class="c-fallback__title">${icon("search")}I couldn't find this in your documents.</p>
    <p class="c-help">The answer doesn't seem to be in the selected PDFs, so I won't guess. You can:</p>
    <div class="c-fallback__actions">
      <button type="button" class="c-btn c-btn--secondary c-btn--sm" data-act="rephrase">Rephrase the question</button>
      <button type="button" class="c-btn c-btn--secondary c-btn--sm" data-act="upload">${icon("upload")}Upload a relevant PDF</button>
    </div>
  </div>`;
}

function actions(answerable) {
  return `<div class="c-msg__actions">
    ${answerable ? `<button type="button" class="c-btn c-btn--ghost c-btn--sm" data-act="copy">${icon("copy")}Copy</button>` : ""}
    <button type="button" class="c-btn c-btn--ghost c-btn--sm" data-act="retry">${icon("retry")}Retry</button>
  </div>`;
}

const HINT_KEY = () => `rag-hint-dismissed:${state.user ? state.user.id : "anon"}`;
const hintDismissed = () => { try { return localStorage.getItem(HINT_KEY()) === "1"; } catch (_) { return false; } };

/** Mode chip + confidence in words + the honest label ("Extractive · quoted from your document"). */
function answerMeta(mode, label, confidenceLabel) {
  if (!mode) return "";
  const kind = mode === "local" ? "Extractive" : "Generated";
  const conf = confidenceLabel ? `<span class="c-badge c-badge--neutral">${escapeHtml(confidenceLabel)}</span>` : "";
  return `<div class="c-answer-meta"><span class="c-badge ${mode === "local" ? "c-badge--ready" : "c-badge--neutral"}">${kind}</span>${conf}<span class="c-meta">${escapeHtml(label || "")}</span></div>`;
}

function hintNotice(hint) {
  if (!hint || hintDismissed()) return "";
  return `<div class="c-notice" role="note"><span>${escapeHtml(hint)}</span>
    <span class="flex gap-2"><button type="button" class="c-btn c-btn--secondary c-btn--sm" data-act="switch-api">Switch mode</button>
    <button type="button" class="c-btn c-btn--ghost c-btn--sm" data-act="dismiss-hint">Dismiss</button></span></div>`;
}

/** Render a finished assistant message into `el` (used live and when loading history). */
function renderAssistant(el, { answer, answerable, citations, retrieved, mode, label, confidenceLabel, hint, rewrite }) {
  el.citations = citations;
  el.answer = answer;
  const known = new Set(citations.map((c) => c.n));
  const body = answerable
    ? `<div class="c-msg__body">${renderAnswer(answer, known)}</div>${sourcesRow(citations)}${answerMeta(mode, label, confidenceLabel)}${hintNotice(hint)}`
    : fallbackCard();
  const note = rewrite ? `<p class="c-meta">Follow-up understood as: <em>${escapeHtml(rewrite)}</em></p>` : "";
  el.innerHTML = `<p class="sr-only">Answer:</p>${note}${body}${debugTable(retrieved, citations)}${actions(answerable)}`;
}

function addAssistant(q) {
  const el = document.createElement("div");
  el.className = "c-msg c-msg--assistant";
  el.dataset.question = q;
  el.setAttribute("aria-busy", "true");
  el.innerHTML = `<div class="c-msg__body" data-js="live"><span class="flex items-center gap-2 u-muted"><span class="c-spinner" aria-hidden="true"></span>Searching your documents…</span></div>`;
  thread.appendChild(el);
  return el;
}

function failAssistant(el, message, partial, code = "") {
  el.removeAttribute("aria-busy");
  const local = `<button type="button" class="c-btn c-btn--secondary c-btn--sm" data-act="switch-local">Switch to Local mode</button>`;
  const fixes = {
    no_api_key: `<button type="button" class="c-btn c-btn--secondary c-btn--sm" data-act="settings">${icon("key")}Add API key</button>${local}`,
    llm_error: `<button type="button" class="c-btn c-btn--secondary c-btn--sm" data-act="settings">${icon("key")}Check API key</button>${local}`,
    ollama_error: local,
  };
  const fix = fixes[code] || "";
  el.innerHTML = `${partial ? `<div class="c-msg__body">${escapeHtml(partial)}</div>` : ""}
    <div class="c-msg__error" role="alert">${icon("alert")}<p>${escapeHtml(message)}</p></div>
    <div class="c-msg__actions">${fix}<button type="button" class="c-btn c-btn--ghost c-btn--sm" data-act="retry">${icon("retry")}Retry</button></div>`;
  el.querySelector(".c-msg__actions").dataset.state = "always";
}

/* ---------- asking ---------- */

function setBusy(busy) {
  state.busy = busy;
  js("send-label").textContent = busy ? "Stop" : "Ask";
  js("send-icon").innerHTML = `<use href="#i-${busy ? "stop" : "send"}"/>`;
  js("send").classList.toggle("c-btn--primary", !busy);
  js("send").classList.toggle("c-btn--secondary", busy);
  js("send").setAttribute("aria-label", busy ? "Stop generating" : "Ask");
  emit("composer");
}

export async function ask(text) {
  const q = text.trim();
  if (!q || state.busy) return;
  addUser(q);
  const el = addAssistant(q);
  scrollDown(true);
  setBusy(true);
  state.controller = new AbortController();
  let streamed = "";
  let finished = false;
  let citations = [];
  let retrieved = [];
  let rewrite = "";
  let label = "";
  const live = () => el.querySelector('[data-js="live"]');
  try {
    await askStream(
      {
        question: q,
        conversation_id: state.conversationId,
        doc_ids: state.selected.size ? [...state.selected] : null,
        mode: state.mode,
      },
      {
        keyHeaders: keyHeaders(),
        signal: state.controller.signal,
        on: {
          token: ({ text: t }) => {
            streamed += t;
            const body = live();
            body.dataset.state = "streaming";
            body.textContent = streamed;
            scrollDown();
          },
          mode: (d) => { label = d.label; },
          rewrite: (d) => { rewrite = d.search_query; },
          citations: (d) => { citations = d.citations; retrieved = d.retrieved; },
          done: (d) => {
            finished = true;
            el.removeAttribute("aria-busy");
            renderAssistant(el, {
              answer: d.answer, answerable: d.answerable, citations, retrieved, rewrite,
              mode: d.mode, label: d.label || label, confidenceLabel: d.confidence_label, hint: d.hint,
            });
            const isNew = state.conversationId !== d.conversation_id;
            state.conversationId = d.conversation_id;
            if (isNew) emit("conversations-changed");
            scrollDown();
          },
          error: (d) => {
            finished = true;
            failAssistant(el, d.message, streamed, d.code);
            emit("modes-changed");
          },
        },
      },
    );
    if (!finished) failAssistant(el, "The connection closed before the answer finished. Please retry.", streamed);
  } catch (e) {
    if (e.name === "AbortError") failAssistant(el, "Stopped. Retry to ask again.", streamed);
    else {
      failAssistant(el, e.message, streamed);
      if (e.status === 401) emit("signed-out", "Your session expired. Please sign in again.");
    }
  } finally {
    setBusy(false);
    state.controller = null;
  }
}

export function stop() { state.controller?.abort(); }

thread.addEventListener("click", async (e) => {
  const msg = e.target.closest(".c-msg--assistant");
  if (!msg) return;
  const chip = e.target.closest("[data-cite]");
  if (chip) {
    const c = (msg.citations || []).find((x) => x.n === Number(chip.dataset.cite));
    if (c) openSource(c, msg.dataset.question || "", chip);
    return;
  }
  const act = e.target.closest("[data-act]")?.dataset.act;
  if (act === "copy") {
    try { await navigator.clipboard.writeText(msg.answer || ""); toast("Answer copied."); }
    catch (_) { toast("Couldn't copy. Select the text and copy it instead.", "error"); }
  } else if (act === "retry") {
    ask(msg.dataset.question || "");
  } else if (act === "rephrase") {
    question.value = msg.dataset.question || "";
    question.focus();
    question.select();
  } else if (act === "upload") {
    js("file-input").click();
  } else if (act === "settings") {
    emit("open-settings");
  } else if (act === "switch-local") {
    emit("set-mode", "local");
    toast("Switched to Local mode. Retry to ask again.");
  } else if (act === "switch-api") {
    emit("set-mode", "api");
  } else if (act === "dismiss-hint") {
    try { localStorage.setItem(HINT_KEY(), "1"); } catch (_) { /* storage blocked */ }
    e.target.closest(".c-notice")?.remove();
  }
});

/* ---------- source drawer ---------- */

function openSource(c, q, opener) {
  js("source-n").textContent = `Source [${c.n}]`;
  js("source-file").textContent = c.filename;
  js("source-file").title = c.filename;
  js("source-pages").textContent = c.page_start === c.page_end ? `Page ${c.page_start}` : `Pages ${c.page_start}–${c.page_end}`;
  js("source-text").innerHTML = highlightWithSpan(c.text || c.snippet, c.highlight, q);
  const scores = js("source-scores");
  scores.hidden = !state.debug;
  const f = (v, d) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
  js("source-scores-list").innerHTML = [["Vector (cosine)", f(c.vector_score, 3)], ["Keyword (BM25)", f(c.bm25_score, 2)], ["Fusion (RRF)", f(c.rrf_score, 4)], ["Rerank", f(c.rerank_score, 2)]]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  showDrawer("source-drawer", { opener });
}

/* ---------- history ---------- */

export function clearThread() {
  thread.innerHTML = "";
  state.conversationId = null;
  js("conv-title").textContent = "New conversation";
  renderEmpty();
}

export async function loadConversation(id) {
  if (state.busy) stop();
  const conv = await request("GET", `/conversations/${id}`);
  thread.innerHTML = "";
  state.conversationId = conv.id;
  js("conv-title").textContent = conv.title;
  let lastQuestion = "";
  for (const m of conv.messages) {
    if (m.role === "user") { addUser(m.content); lastQuestion = m.content; continue; }
    const el = document.createElement("div");
    el.className = "c-msg c-msg--assistant";
    el.dataset.question = lastQuestion;
    renderAssistant(el, {
      answer: m.content, answerable: m.answerable, citations: m.citations, retrieved: [], mode: m.mode, label: m.label,
    });
    thread.appendChild(el);
  }
  renderEmpty();
  scrollDown(true);
  return conv;
}

export function setDebug(on) {
  state.debug = on;
  document.querySelectorAll('[data-js="debug"]').forEach((t) => { t.hidden = !on; });
}
