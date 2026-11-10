"use strict";

/* ===================== state & helpers ===================== */
const state = {
  docs: [],
  selected: new Set(),
  debug: false,
  busy: false,
  controller: null,
  memoryKey: "", // fallback when browser storage is blocked
  llm: null, // last /llm/check result
};

const $ = (id) => document.getElementById(id);
const els = {
  fileInput: $("fileInput"), dropzone: $("dropzone"), uploads: $("uploads"),
  docList: $("docList"), docCount: $("docCount"), scopeHint: $("scopeHint"),
  debugToggle: $("debugToggle"), keyStatus: $("keyStatus"), openKey: $("openKey"),
  keyModal: $("keyModal"), keyForm: $("keyForm"), keyInput: $("keyInput"), keyReveal: $("keyReveal"),
  keyRemember: $("keyRemember"), keyResult: $("keyResult"), keyRemove: $("keyRemove"), keyTest: $("keyTest"),
  keySave: $("keySave"), keyClose: $("keyClose"), keyBanner: $("keyBanner"), keyBannerText: $("keyBannerText"),
  keyBannerBtn: $("keyBannerBtn"),
  messages: $("messages"), thread: $("thread"), emptyState: $("emptyState"), suggestions: $("suggestions"),
  form: $("askForm"), question: $("question"), sendBtn: $("sendBtn"), sendLabel: $("sendLabel"), sendIcon: $("sendIcon"),
  drawer: $("drawer"), drawerBackdrop: $("drawerBackdrop"), drawerClose: $("drawerClose"),
  drawerN: $("drawerN"), drawerTitle: $("drawerTitle"), drawerPages: $("drawerPages"),
  drawerSnippet: $("drawerSnippet"), drawerScores: $("drawerScores"),
  toasts: $("toasts"), toggleSidebar: $("toggleSidebar"), sidebarBody: $("sidebarBody"),
};

const SUGGESTIONS = [
  "Summarize the main points of this document.",
  "What are the key numbers or specifications?",
  "What does it say about warranty or support?",
];

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function pageLabel(a, b) {
  return a === b ? `p. ${a}` : `pp. ${a}–${b}`;
}

function fmtScore(v, digits = 3) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
}

function toast(message, kind = "error") {
  const colors = kind === "error"
    ? "border-rose-200 bg-rose-50 text-rose-800 dark:border-rose-500/30 dark:bg-rose-950 dark:text-rose-200"
    : "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-500/30 dark:bg-emerald-950 dark:text-emerald-200";
  const el = document.createElement("div");
  el.className = `pointer-events-auto rounded-xl border px-4 py-3 text-sm shadow-lg transition ${colors}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.textContent = message;
  els.toasts.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; setTimeout(() => el.remove(), 300); }, 5000);
}

async function apiError(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join("; ");
  } catch (_) { /* not JSON */ }
  return `Request failed (${res.status}).`;
}

/* ===================== API key ===================== */
const KEY_STORE = "rag-anthropic-key";

function storage(kind) {
  try { return kind === "local" ? window.localStorage : window.sessionStorage; } catch (_) { return null; }
}

function getKey() {
  for (const kind of ["session", "local"]) {
    try { const v = storage(kind)?.getItem(KEY_STORE); if (v) return v; } catch (_) { /* blocked */ }
  }
  return "";
}

function setKey(key, remember) {
  for (const kind of ["session", "local"]) { try { storage(kind)?.removeItem(KEY_STORE); } catch (_) { /* ignore */ } }
  if (!key) return;
  try { storage(remember ? "local" : "session")?.setItem(KEY_STORE, key); } catch (_) { /* ignore */ }
  state.memoryKey = key; // still usable this page load even if storage is blocked
}

function currentKey() { return getKey() || state.memoryKey || ""; }

function keyHeaders() {
  const k = currentKey();
  return k ? { "X-Anthropic-Api-Key": k } : {};
}

function maskKey(k) { return k.length > 14 ? `${k.slice(0, 7)}…${k.slice(-4)}` : "••••"; }

const STATUS = {
  checking: ["bg-slate-400 animate-pulse", "Checking Claude…"],
  ok: ["bg-emerald-500", ""],
  bad: ["bg-rose-500", ""],
  none: ["bg-amber-500", "No API key yet"],
  offline: ["bg-rose-500", "Server unreachable"],
};

function renderKeyStatus(kind, text) {
  const [dot, fallback] = STATUS[kind];
  els.keyStatus.innerHTML = `<span class="h-2 w-2 shrink-0 rounded-full ${dot}"></span><span class="truncate">${escapeHtml(text || fallback)}</span>`;
  els.keyStatus.title = text || fallback;
  const showBanner = kind === "none" || kind === "bad";
  els.keyBanner.classList.toggle("hidden", !showBanner);
  els.keyBanner.classList.toggle("flex", showBanner);
  els.keyBannerText.textContent = kind === "bad"
    ? "Your API key isn't working. Open the key screen to fix it."
    : "Add your Anthropic API key to get answers. Search works without it.";
  els.keyBannerBtn.textContent = kind === "bad" ? "Fix API key" : "Add API key";
}

function showKeyResult(kind, html) {
  const styles = {
    pending: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200",
    ok: "bg-emerald-50 text-emerald-800 dark:bg-emerald-500/10 dark:text-emerald-200",
    bad: "bg-rose-50 text-rose-800 dark:bg-rose-500/10 dark:text-rose-200",
  };
  els.keyResult.className = `mt-4 rounded-xl px-3 py-2.5 text-sm ${styles[kind]}`;
  els.keyResult.innerHTML = html;
}

/** Ask the server whether the key works. `key` overrides the stored key (used before saving). */
async function checkKey(key = currentKey()) {
  renderKeyStatus("checking");
  try {
    const res = await fetch("/llm/check", { method: "POST", headers: key ? { "X-Anthropic-Api-Key": key } : {} });
    if (!res.ok) throw new Error(await apiError(res));
    const r = await res.json();
    state.llm = r;
    if (r.ok) renderKeyStatus("ok", r.source === "browser" ? `Claude connected · your key` : `Claude connected · server key`);
    else renderKeyStatus(r.source === "none" ? "none" : "bad", r.source === "none" ? "" : "API key not working");
    return r;
  } catch (e) {
    renderKeyStatus("offline");
    return { ok: false, message: e.message || "Server unreachable." };
  }
}

function openKeyModal() {
  const k = currentKey();
  els.keyInput.value = "";
  els.keyInput.placeholder = k ? `Saved: ${maskKey(k)} (paste a new key to replace it)` : "sk-ant-…";
  els.keyRemember.checked = (() => { try { return !!storage("local")?.getItem(KEY_STORE); } catch (_) { return false; } })();
  els.keyRemove.disabled = !k;
  els.keyResult.classList.add("hidden");
  if (state.llm && state.llm.source !== "none") {
    showKeyResult(state.llm.ok ? "ok" : "bad",
      state.llm.ok ? `✅ ${escapeHtml(state.llm.message)} <span class="opacity-70">(${state.llm.source === "browser" ? "your key" : "server key"})</span>`
                   : `❌ ${escapeHtml(state.llm.message)}`);
  }
  els.keyModal.classList.remove("hidden");
  els.keyModal.classList.add("flex");
  setTimeout(() => els.keyInput.focus(), 30);
}

function closeKeyModal() {
  els.keyModal.classList.add("hidden");
  els.keyModal.classList.remove("flex");
  els.keyInput.value = "";
  els.keyInput.type = "password";
  els.keyReveal.textContent = "Show";
}

async function runKeyTest(save) {
  const typed = els.keyInput.value.trim();
  const key = typed || currentKey();
  if (!key) { showKeyResult("bad", "Paste your API key first."); return; }
  if (typed && !typed.startsWith("sk-")) { showKeyResult("bad", "That doesn't look like an Anthropic API key (they start with <code>sk-ant-</code>)."); return; }
  els.keySave.disabled = true; els.keyTest.disabled = true;
  showKeyResult("pending", `<span class="inline-flex items-center gap-2"><span class="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent"></span>Contacting Anthropic…</span>`);
  const r = await checkKey(key);
  els.keySave.disabled = false; els.keyTest.disabled = false;
  if (r.ok) {
    if (save && typed) { setKey(typed, els.keyRemember.checked); els.keyRemove.disabled = false; }
    else if (save && !typed) { setKey(key, els.keyRemember.checked); }
    showKeyResult("ok", `✅ It works! ${escapeHtml(r.message)}${save ? " Key saved, so you can start asking questions." : ""}`);
    if (save) { toast("Claude is connected.", "ok"); setTimeout(closeKeyModal, 1200); }
  } else {
    showKeyResult("bad", `❌ ${escapeHtml(r.message)}${save ? " The key was not saved." : ""}`);
    if (typed) await checkKey(); // restore the sidebar status for the previously saved key
  }
}

els.openKey.addEventListener("click", openKeyModal);
els.keyBannerBtn.addEventListener("click", openKeyModal);
els.keyClose.addEventListener("click", closeKeyModal);
els.keyModal.addEventListener("click", (e) => { if (e.target === els.keyModal) closeKeyModal(); });
els.keyForm.addEventListener("submit", (e) => { e.preventDefault(); runKeyTest(true); });
els.keyTest.addEventListener("click", () => runKeyTest(false));
els.keyReveal.addEventListener("click", () => {
  const hidden = els.keyInput.type === "password";
  els.keyInput.type = hidden ? "text" : "password";
  els.keyReveal.textContent = hidden ? "Hide" : "Show";
});
els.keyRemove.addEventListener("click", async () => {
  setKey("", false);
  state.memoryKey = "";
  els.keyRemove.disabled = true;
  els.keyInput.placeholder = "sk-ant-…";
  const r = await checkKey();
  showKeyResult(r.ok ? "ok" : "pending", r.ok ? "Your key was removed. The server's own key is still active." : "Your key was removed from this browser.");
});

/* ===================== documents ===================== */
async function loadDocs() {
  try {
    const res = await fetch("/documents");
    if (!res.ok) throw new Error(await apiError(res));
    state.docs = await res.json();
    const ids = new Set(state.docs.map((d) => d.doc_id));
    state.selected.forEach((id) => { if (!ids.has(id)) state.selected.delete(id); });
    renderDocs();
  } catch (e) {
    toast(`Could not load documents: ${e.message}`);
  }
}

function renderDocs() {
  els.docCount.textContent = state.docs.length;
  if (!state.docs.length) {
    els.docList.innerHTML = `<li class="px-2 py-6 text-center text-sm text-slate-500 dark:text-slate-400">No documents yet. Upload a PDF to begin.</li>`;
  } else {
    els.docList.innerHTML = state.docs.map((d) => `
      <li class="group flex items-center gap-3 rounded-xl px-2 py-2 hover:bg-slate-100 dark:hover:bg-slate-800/70">
        <input type="checkbox" data-select="${d.doc_id}" ${state.selected.has(d.doc_id) ? "checked" : ""}
          class="h-4 w-4 shrink-0 cursor-pointer rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 dark:border-slate-600 dark:bg-slate-800"
          aria-label="Limit questions to ${escapeHtml(d.filename)}" />
        <div class="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-rose-100 text-[10px] font-bold text-rose-600 dark:bg-rose-500/15 dark:text-rose-300">PDF</div>
        <div class="min-w-0 flex-1">
          <p class="truncate text-sm font-medium" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</p>
          <p class="text-xs text-slate-500 dark:text-slate-400">${d.pages} pages · ${d.chunks} chunks</p>
        </div>
        <button data-delete="${d.doc_id}" class="rounded-lg p-1.5 text-slate-400 opacity-100 transition hover:bg-rose-50 hover:text-rose-600 focus:opacity-100 dark:hover:bg-rose-500/10 md:opacity-0 md:group-hover:opacity-100"
          aria-label="Delete ${escapeHtml(d.filename)}" title="Delete">
          <svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>
        </button>
      </li>`).join("");
  }
  const n = state.selected.size;
  els.scopeHint.textContent = n
    ? `Questions are limited to ${n} selected document${n > 1 ? "s" : ""}.`
    : "Questions search all documents. Tick to narrow the scope.";
  renderSuggestions();
}

els.docList.addEventListener("change", (e) => {
  const id = e.target.dataset.select;
  if (!id) return;
  e.target.checked ? state.selected.add(id) : state.selected.delete(id);
  renderDocs();
});

els.docList.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-delete]");
  if (!btn) return;
  const doc = state.docs.find((d) => d.doc_id === btn.dataset.delete);
  if (!doc || !confirm(`Delete "${doc.filename}"? It will be removed from the index.`)) return;
  btn.disabled = true;
  try {
    const res = await fetch(`/documents/${doc.doc_id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await apiError(res));
    state.selected.delete(doc.doc_id);
    toast(`Deleted ${doc.filename}`, "ok");
    await loadDocs();
  } catch (err) {
    toast(err.message);
    btn.disabled = false;
  }
});

/* ===================== upload ===================== */
function uploadRow(file) {
  const li = document.createElement("li");
  li.className = "rounded-xl border border-slate-200 px-3 py-2 text-xs dark:border-slate-700";
  li.innerHTML = `
    <div class="flex items-center justify-between gap-2">
      <span class="truncate font-medium">${escapeHtml(file.name)}</span>
      <span data-status class="shrink-0 text-slate-500 dark:text-slate-400">0%</span>
    </div>
    <div class="mt-1.5 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
      <div data-bar class="h-full w-0 rounded-full bg-indigo-500 transition-all"></div>
    </div>`;
  els.uploads.prepend(li);
  return {
    progress(pct) { li.querySelector("[data-bar]").style.width = `${pct}%`; li.querySelector("[data-status]").textContent = `${pct}%`; },
    processing() {
      const bar = li.querySelector("[data-bar]");
      bar.style.width = "100%"; bar.classList.add("animate-pulse");
      li.querySelector("[data-status]").textContent = "Indexing…";
    },
    finish(ok, text) {
      const bar = li.querySelector("[data-bar]");
      bar.classList.remove("animate-pulse", "bg-indigo-500");
      bar.classList.add(ok ? "bg-emerald-500" : "bg-rose-500");
      bar.style.width = "100%";
      const s = li.querySelector("[data-status]");
      s.textContent = text;
      s.className = `shrink-0 ${ok ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"}`;
      setTimeout(() => { li.style.opacity = "0"; setTimeout(() => li.remove(), 300); }, ok ? 2500 : 8000);
      li.classList.add("transition-opacity");
    },
  };
}

function uploadFile(file) {
  return new Promise((resolve) => {
    const row = uploadRow(file);
    if (!/\.pdf$/i.test(file.name)) { row.finish(false, "Not a PDF"); toast(`'${file.name}' is not a PDF file.`); return resolve(); }
    if (file.size > 20 * 1024 * 1024) { row.finish(false, "Too large"); toast(`'${file.name}' is larger than 20 MB.`); return resolve(); }
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", "/upload");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) row.progress(Math.round((e.loaded / e.total) * 100)); };
    xhr.upload.onload = () => row.processing();
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) {
        row.finish(true, body.duplicate ? "Already indexed" : `${body.document.chunks} chunks`);
      } else {
        const msg = typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status}).`;
        row.finish(false, "Failed");
        toast(msg);
      }
      resolve();
    };
    xhr.onerror = () => { row.finish(false, "Network error"); toast("Upload failed: network error."); resolve(); };
    xhr.send(form);
  });
}

async function handleFiles(files) {
  for (const f of files) await uploadFile(f); // sequential keeps indexing load predictable
  await loadDocs();
}

els.fileInput.addEventListener("change", (e) => { handleFiles([...e.target.files]); e.target.value = ""; });
els.dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); } });
["dragenter", "dragover"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
  e.preventDefault(); els.dropzone.classList.add("border-indigo-500", "bg-indigo-50", "dark:bg-indigo-500/10");
}));
["dragleave", "drop"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
  e.preventDefault(); els.dropzone.classList.remove("border-indigo-500", "bg-indigo-50", "dark:bg-indigo-500/10");
}));
els.dropzone.addEventListener("drop", (e) => handleFiles([...e.dataTransfer.files]));
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

/* ===================== rendering answers ===================== */
function renderInline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

/** Tiny, safe Markdown subset: paragraphs, bullet/numbered lists, bold, code. Input is escaped first. */
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "", list = null, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${renderInline(para.join(" "))}</p>`; para = []; } };
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trim();
    const ul = line.match(/^[-*•]\s+(.*)$/), ol = line.match(/^\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; }
      html += `<li>${renderInline((ul || ol)[1])}</li>`;
    } else if (!line) {
      flushPara(); closeList();
    } else {
      closeList(); para.push(line);
    }
  }
  flushPara(); closeList();
  return html;
}

function chipHtml(n, known) {
  const base = "mx-0.5 inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded-md px-1 align-text-top text-[11px] font-semibold transition";
  return known
    ? `<button type="button" data-cite="${n}" class="${base} bg-indigo-100 text-indigo-700 hover:bg-indigo-600 hover:text-white dark:bg-indigo-500/20 dark:text-indigo-200 dark:hover:bg-indigo-500" aria-label="Open source ${n}">${n}</button>`
    : `<span class="${base} bg-slate-200 text-slate-500 dark:bg-slate-700 dark:text-slate-400">${n}</span>`;
}

function withChips(html, citations) {
  const known = new Set(citations.map((c) => c.n));
  return html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, group) =>
    group.split(",").map((n) => chipHtml(Number(n.trim()), known.has(Number(n.trim())))).join(""));
}

function scoreTable(rows, citedSet) {
  return `
    <div class="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
      <table class="w-full text-left text-xs">
        <thead class="bg-slate-50 text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
          <tr><th class="px-3 py-2">#</th><th class="px-3 py-2">Source</th><th class="px-3 py-2 text-right">Vector</th><th class="px-3 py-2 text-right">BM25</th><th class="px-3 py-2 text-right">RRF</th><th class="px-3 py-2 text-right">Rerank</th></tr>
        </thead>
        <tbody class="divide-y divide-slate-100 font-mono dark:divide-slate-800">
          ${rows.map((r) => `
            <tr class="${citedSet.has(r.n) ? "bg-indigo-50/60 dark:bg-indigo-500/10" : ""}">
              <td class="px-3 py-1.5 font-sans font-semibold">${r.n}${citedSet.has(r.n) ? " ✓" : ""}</td>
              <td class="max-w-[12rem] truncate px-3 py-1.5 font-sans" title="${escapeHtml(r.filename)}">${escapeHtml(r.filename)} · ${pageLabel(r.page_start, r.page_end)}</td>
              <td class="px-3 py-1.5 text-right">${fmtScore(r.vector_score)}</td>
              <td class="px-3 py-1.5 text-right">${fmtScore(r.bm25_score, 2)}</td>
              <td class="px-3 py-1.5 text-right">${fmtScore(r.rrf_score, 4)}</td>
              <td class="px-3 py-1.5 text-right">${fmtScore(r.rerank_score, 2)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

function addUserMessage(text) {
  els.emptyState.classList.add("hidden");
  const div = document.createElement("div");
  div.className = "flex justify-end";
  div.innerHTML = `<div class="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2.5 text-sm text-white shadow-sm">${escapeHtml(text)}</div>`;
  els.thread.appendChild(div);
}

function addAssistantMessage(question) {
  const wrap = document.createElement("div");
  wrap.className = "flex gap-3";
  wrap.dataset.msg = "assistant";
  wrap.innerHTML = `
    <div class="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-slate-900 text-xs font-bold text-white dark:bg-white dark:text-slate-900">AI</div>
    <div class="min-w-0 flex-1">
      <div data-card class="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-3 text-sm leading-relaxed shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <div data-body class="prose-answer">
          <span class="inline-flex items-center gap-2 text-slate-500 dark:text-slate-400">
            <span class="h-2 w-2 animate-bounce rounded-full bg-indigo-500"></span>
            <span class="h-2 w-2 animate-bounce rounded-full bg-indigo-500 [animation-delay:120ms]"></span>
            <span class="h-2 w-2 animate-bounce rounded-full bg-indigo-500 [animation-delay:240ms]"></span>
            <span class="ml-1 text-xs">Searching your documents…</span>
          </span>
        </div>
        <div data-sources class="mt-3 hidden flex-wrap gap-2 border-t border-slate-100 pt-3 dark:border-slate-800"></div>
      </div>
      <div data-debug class="mt-2 hidden"></div>
    </div>`;
  els.thread.appendChild(wrap);
  const body = wrap.querySelector("[data-body]");
  let text = "";
  let citations = [];
  return {
    el: wrap,
    question,
    append(delta) {
      text += delta;
      body.classList.add("cursor", "whitespace-pre-wrap");
      body.textContent = text;
    },
    finish(done) {
      citations = done.citations;
      wrap.citations = done.citations;
      wrap.question = question;
      body.classList.remove("cursor", "whitespace-pre-wrap");
      if (!done.answerable) {
        wrap.querySelector("[data-card]").className =
          "rounded-2xl rounded-tl-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-relaxed text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-100";
        body.innerHTML = `<p class="flex gap-2"><span aria-hidden="true">🔎</span><span>${escapeHtml(done.answer)}</span></p>`;
      } else {
        body.innerHTML = withChips(renderMarkdown(done.answer), citations);
      }
      const src = wrap.querySelector("[data-sources]");
      if (citations.length) {
        src.innerHTML = citations.map((c) => `
          <button type="button" data-cite="${c.n}" class="inline-flex max-w-full items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2 py-1 text-xs transition hover:border-indigo-400 hover:bg-indigo-50 dark:border-slate-700 dark:bg-slate-800 dark:hover:border-indigo-500 dark:hover:bg-indigo-500/10">
            <span class="font-semibold text-indigo-600 dark:text-indigo-300">[${c.n}]</span>
            <span class="truncate">${escapeHtml(c.filename)}</span>
            <span class="shrink-0 text-slate-500 dark:text-slate-400">${pageLabel(c.page_start, c.page_end)}</span>
          </button>`).join("");
        src.classList.remove("hidden");
        src.classList.add("flex");
      }
      const dbg = wrap.querySelector("[data-debug]");
      if (done.retrieved && done.retrieved.length) {
        dbg.innerHTML = `<p class="mb-1.5 text-xs font-medium text-slate-500 dark:text-slate-400">Retrieved sources (top ${done.retrieved.length})${done.answerable ? "" : " · best rerank score was below the threshold or the model found no answer"}</p>` +
          scoreTable(done.retrieved, new Set(citations.map((c) => c.n)));
        dbg.dataset.has = "1";
      }
      dbg.classList.toggle("hidden", !(state.debug && dbg.dataset.has));
    },
    fail(message) {
      body.classList.remove("cursor", "whitespace-pre-wrap");
      wrap.querySelector("[data-card]").className =
        "rounded-2xl rounded-tl-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-200";
      body.innerHTML = (text ? `<p class="mb-2 whitespace-pre-wrap text-slate-700 dark:text-slate-200">${escapeHtml(text)}</p>` : "") +
        `<p><strong>Something went wrong.</strong> ${escapeHtml(message)}</p>`;
    },
  };
}

function scrollToBottom(force = false) {
  const m = els.messages;
  if (force || m.scrollHeight - m.scrollTop - m.clientHeight < 160) m.scrollTop = m.scrollHeight;
}

/* ===================== drawer ===================== */
function highlight(snippet, question) {
  const stop = new Set("a an and are as at be by for from has have in is it its of on or that the this to was were what when where which who why how with does do did can i my me you".split(" "));
  const terms = [...new Set((question.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => t.length > 2 && !stop.has(t)))];
  let html = escapeHtml(snippet);
  if (!terms.length) return html;
  const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*`, "gi");
  return html.replace(re, "<mark>$&</mark>");
}

function openDrawer(citation, question) {
  els.drawerN.textContent = `Source [${citation.n}]`;
  els.drawerTitle.textContent = citation.filename;
  els.drawerTitle.title = citation.filename;
  els.drawerPages.textContent = citation.page_start === citation.page_end ? `Page ${citation.page_start}` : `Pages ${citation.page_start}–${citation.page_end}`;
  els.drawerSnippet.innerHTML = highlight(citation.text || citation.snippet, question);
  if (state.debug) {
    els.drawerScores.innerHTML = `<p class="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">Retrieval scores</p>
      <dl class="grid grid-cols-2 gap-2 text-sm">
        ${[["Vector (cosine)", fmtScore(citation.vector_score)], ["BM25", fmtScore(citation.bm25_score, 2)], ["RRF", fmtScore(citation.rrf_score, 4)], ["Rerank", fmtScore(citation.rerank_score, 2)]]
          .map(([k, v]) => `<div class="rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-800/60"><dt class="text-xs text-slate-500 dark:text-slate-400">${k}</dt><dd class="font-mono">${v}</dd></div>`).join("")}
      </dl>`;
    els.drawerScores.classList.remove("hidden");
  } else {
    els.drawerScores.classList.add("hidden");
  }
  els.drawerBackdrop.classList.remove("hidden");
  els.drawer.classList.remove("translate-x-full");
  els.drawerClose.focus();
}

function closeDrawer() {
  els.drawer.classList.add("translate-x-full");
  els.drawerBackdrop.classList.add("hidden");
}

els.thread.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-cite]");
  if (!btn) return;
  const msg = btn.closest("[data-msg]");
  const c = (msg && msg.citations || []).find((x) => x.n === Number(btn.dataset.cite));
  if (c) openDrawer(c, msg.question || "");
});
els.drawerClose.addEventListener("click", closeDrawer);
els.drawerBackdrop.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawer(); closeKeyModal(); } });

/* ===================== asking (SSE over fetch) ===================== */
function setBusy(busy) {
  state.busy = busy;
  els.sendLabel.textContent = busy ? "Stop" : "Ask";
  els.sendIcon.classList.toggle("hidden", busy);
  els.sendBtn.classList.toggle("bg-slate-700", busy);
  els.sendBtn.classList.toggle("hover:bg-slate-600", busy);
}

async function ask(question) {
  addUserMessage(question);
  const msg = addAssistantMessage(question);
  scrollToBottom(true);
  setBusy(true);
  state.controller = new AbortController();
  let finished = false;
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream", ...keyHeaders() },
      body: JSON.stringify({ question, doc_ids: state.selected.size ? [...state.selected] : null }),
      signal: state.controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(await apiError(res));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let event = "message", data = "";
        for (const line of raw.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const payload = JSON.parse(data);
        if (event === "token") { msg.append(payload.text); scrollToBottom(); }
        else if (event === "done") { msg.finish(payload); finished = true; scrollToBottom(); }
        else if (event === "error") { msg.fail(payload.message); finished = true; }
      }
    }
    if (!finished) msg.fail("The connection closed before the answer finished.");
  } catch (e) {
    if (e.name === "AbortError") msg.fail("Stopped.");
    else msg.fail(e.message || "Network error.");
  } finally {
    setBusy(false);
    state.controller = null;
    els.question.focus();
  }
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.busy) { state.controller && state.controller.abort(); return; }
  const q = els.question.value.trim();
  if (!q) return;
  if (!state.docs.length) { toast("Upload a PDF first, then ask a question about it."); return; }
  els.question.value = "";
  autosize();
  ask(q);
});

function autosize() {
  els.question.style.height = "auto";
  els.question.style.height = `${Math.min(els.question.scrollHeight, 160)}px`;
}
els.question.addEventListener("input", autosize);
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); els.form.requestSubmit(); }
});

function renderSuggestions() {
  els.suggestions.innerHTML = state.docs.length
    ? SUGGESTIONS.map((s) => `<button type="button" class="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-600 shadow-sm transition hover:border-indigo-400 hover:text-indigo-600 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-indigo-500">${escapeHtml(s)}</button>`).join("")
    : `<p class="text-xs text-slate-400">Start by uploading a PDF in the panel${window.innerWidth < 768 ? " above" : " on the left"}.</p>`;
}
els.suggestions.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b && !state.busy) { els.question.value = b.textContent; els.form.requestSubmit(); }
});

/* ===================== debug & layout ===================== */
els.debugToggle.addEventListener("change", () => {
  state.debug = els.debugToggle.checked;
  try { localStorage.setItem("rag-debug", state.debug ? "1" : "0"); } catch (_) { /* storage unavailable */ }
  document.querySelectorAll("[data-debug]").forEach((d) => d.classList.toggle("hidden", !(state.debug && d.dataset.has)));
});

els.toggleSidebar.addEventListener("click", () => {
  const open = els.sidebarBody.classList.toggle("hidden") === false;
  els.sidebarBody.classList.toggle("flex", open);
  els.toggleSidebar.setAttribute("aria-expanded", String(open));
});

/* ===================== boot ===================== */
try { state.debug = localStorage.getItem("rag-debug") === "1"; } catch (_) { /* ignore */ }
els.debugToggle.checked = state.debug;
checkKey();
loadDocs().then(() => {
  if (!state.docs.length && window.innerWidth < 768) {
    els.sidebarBody.classList.remove("hidden");
    els.sidebarBody.classList.add("flex");
    els.toggleSidebar.setAttribute("aria-expanded", "true");
  }
});
