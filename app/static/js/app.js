// Entry point: auth screen, documents, conversations, settings, shortcuts, layout.

import { request, uploadFile } from "./api.js";
import { ask, clearThread, loadConversation, renderEmpty, setDebug, stop } from "./chat.js";
import { renderStudyForm } from "./study.js";
import {
  MODE_LABELS, emit, getApiKey, getKeyMeta, isKeyRemembered, keyHeaders, on, readyDocs, setApiKey, state,
} from "./state.js";
import {
  closeDrawer, confirmDanger, escapeHtml, fmtDate, icon, isDrawerOpen, js, setTheme, showDrawer, themeChoice, toast, useThemeFor,
} from "./ui.js";

const STAGES = { queued: "Waiting", reading: "Reading", splitting: "Splitting", understanding: "Understanding", ready: "Ready", failed: "Failed" };
const pref = (k) => `rag-${k}:${state.user ? state.user.id : "anon"}`;
const readPref = (k) => { try { return localStorage.getItem(pref(k)); } catch (_) { return null; } };
const writePref = (k, v) => { try { localStorage.setItem(pref(k), v); } catch (_) { /* blocked */ } };

/* ======================= auth ======================= */

let authMode = "login";

function setAuthMode(mode) {
  authMode = mode;
  document.querySelectorAll('[data-js="auth-tab"]').forEach((t) => t.setAttribute("aria-selected", String(t.dataset.mode === mode)));
  js("auth-form").setAttribute("aria-labelledby", mode === "login" ? "tab-login" : "tab-register");
  js("auth-submit").querySelector(".c-btn__label").textContent = mode === "login" ? "Sign in" : "Create account";
  js("auth-password").autocomplete = mode === "login" ? "current-password" : "new-password";
  js("strength").hidden = mode === "login";
  js("password-help").textContent = mode === "login" ? "At least 8 characters." : "At least 8 characters. Longer is stronger.";
  clearAuthErrors();
}

function clearAuthErrors() {
  for (const f of ["email", "password"]) {
    js("auth-form").querySelector(`#auth-${f}`).removeAttribute("aria-invalid");
    js("auth-form").querySelector(`#auth-${f}-error`).hidden = true;
  }
  js("auth-error").hidden = true;
}

function fieldError(field, message) {
  const input = document.getElementById(`auth-${field}`);
  const err = document.getElementById(`auth-${field}-error`);
  input.setAttribute("aria-invalid", "true");
  err.innerHTML = `${icon("alert")}<span>${escapeHtml(message)}</span>`;
  err.hidden = false;
  input.focus();
}

function strength(pw) {
  if (!pw) return 0;
  let score = pw.length >= 8 ? 1 : 0;
  if (pw.length >= 12) score += 1;
  if (/[^a-zA-Z]/.test(pw) && /[a-zA-Z]/.test(pw)) score += 1;
  return Math.min(score, 3);
}

js("auth-password").addEventListener("input", (e) => {
  if (authMode !== "register") return;
  const lvl = strength(e.target.value);
  js("strength").dataset.level = String(lvl);
  js("password-help").textContent = ["At least 8 characters.", "Okay. A longer password is stronger.", "Good.", "Strong password."][lvl];
});

document.querySelectorAll('[data-js="auth-tab"]').forEach((t) => t.addEventListener("click", () => setAuthMode(t.dataset.mode)));

function busyButton(btn, busy, label) {
  btn.setAttribute("aria-busy", String(busy));
  btn.disabled = busy;
  const lbl = btn.querySelector(".c-btn__label");
  if (label) lbl.textContent = label;
  btn.querySelector(".c-spinner")?.remove();
  if (busy) btn.insertAdjacentHTML("afterbegin", '<span class="c-spinner" aria-hidden="true"></span>');
}

js("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  clearAuthErrors();
  const email = js("auth-email").value.trim();
  const password = js("auth-password").value;
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return fieldError("email", "Enter a valid email address, like name@example.com.");
  if (authMode === "register" && password.length < 8) return fieldError("password", "Use at least 8 characters for your password.");
  if (!password) return fieldError("password", "Enter your password.");
  const btn = js("auth-submit");
  const label = authMode === "login" ? "Sign in" : "Create account";
  busyButton(btn, true, authMode === "login" ? "Signing in…" : "Creating account…");
  try {
    const user = await request("POST", `/auth/${authMode === "login" ? "login" : "register"}`, { json: { email, password } });
    await enterApp(user);
  } catch (err) {
    if (err.code === "invalid_email" || err.code === "email_taken") fieldError("email", err.message);
    else if (err.code === "weak_password") fieldError("password", err.message);
    else { js("auth-error").innerHTML = `${icon("alert")}<span>${escapeHtml(err.message)}</span>`; js("auth-error").hidden = false; }
  } finally {
    busyButton(btn, false, label);
  }
});

js("guest").addEventListener("click", async () => {
  const btn = js("guest");
  busyButton(btn, true, "Preparing the sample PDF…");
  try {
    await enterApp(await request("POST", "/auth/guest"));
  } catch (err) {
    toast(err.message, "error");
  } finally {
    busyButton(btn, false, "Try it with a sample PDF");
  }
});

async function signOut(message) {
  stop();
  try { await request("POST", "/auth/logout"); } catch (_) { /* already signed out */ }
  closeDrawer({ restore: false });
  state.user = null;
  state.docs = [];
  state.selected.clear();
  clearThread();
  clearInterval(pollTimer);
  useThemeFor(null);
  js("app-view").hidden = true;
  js("auth-view").hidden = false;
  if (message) toast(message, "error");
  js("guest").focus();
}

on("signed-out", (msg) => signOut(msg));

/* ======================= app shell ======================= */

async function enterApp(user) {
  state.user = user;
  useThemeFor(user.id);
  state.debug = readPref("debug") === "1";
  setDebug(state.debug);
  js("auth-view").hidden = true;
  js("app-view").hidden = false;
  js("account-line").textContent = user.is_guest
    ? "You're using a guest account. It and its documents are deleted after 24 hours or when you sign out."
    : `Signed in as ${user.email}.`;
  renderDocSkeleton();
  await Promise.all([loadDocs(), loadConversations()]);
  clearThread();
  state.mode = null;
  loadModes();
  setView(readPref("view") === "study" ? "study" : "chat");
}

/* ---------- mobile slide-over ---------- */
function setSidebar(open) {
  const sb = js("sidebar");
  sb.dataset.state = open ? "open" : "";
  js("sidebar-scrim").hidden = !open;
  js("sidebar-scrim").dataset.state = open ? "open" : "";
  js("open-sidebar").setAttribute("aria-expanded", String(open));
  if (open) js("close-sidebar").focus();
}
js("open-sidebar").addEventListener("click", () => setSidebar(true));
js("close-sidebar").addEventListener("click", () => { setSidebar(false); js("open-sidebar").focus(); });
js("sidebar-scrim").addEventListener("click", () => setSidebar(false));
const isMobile = () => window.matchMedia("(max-width: 767px)").matches;

/* ---------- sidebar tabs ---------- */
document.querySelectorAll('[data-js="side-tab"]').forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll('[data-js="side-tab"]').forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
  document.getElementById("panel-docs").hidden = tab.dataset.panel !== "docs";
  document.getElementById("panel-convs").hidden = tab.dataset.panel !== "convs";
}));

/* ======================= documents ======================= */

const uploads = new Map(); // temp id -> {name, pct, stage}
let pollTimer = null;

function renderDocSkeleton() {
  js("doc-list").innerHTML = '<li class="c-skeleton c-skeleton--row" aria-hidden="true"></li>'.repeat(2);
}

async function loadDocs() {
  try {
    state.docs = await request("GET", "/documents");
    const ids = new Set(state.docs.map((d) => d.id));
    [...state.selected].forEach((id) => { if (!ids.has(id)) state.selected.delete(id); });
    renderDocs();
    schedulePoll();
  } catch (err) {
    if (err.status === 401) return emit("signed-out", "Your session expired. Please sign in again.");
    js("doc-list").innerHTML = `<li class="c-help px-2">Couldn't load your documents. <button class="c-badge__action" data-js="reload-docs">Try again</button></li>`;
    js("reload-docs").addEventListener("click", loadDocs);
  }
}

function badge(doc, progress) {
  if (doc.status === "ready") return `<span class="c-badge c-badge--ready">${icon("check")}Ready</span>`;
  if (doc.status === "failed") {
    return `<span class="c-badge c-badge--failed">${icon("alert")}Failed</span>
      <button type="button" class="c-badge__action c-meta" data-retry="${doc.id}" aria-label="Retry processing ${escapeHtml(doc.filename)}">Retry</button>`;
  }
  const stage = progress ? `${STAGES[progress.stage] || "Processing"} · ${progress.progress}%` : "Processing";
  return `<span class="c-badge c-badge--processing"><span class="c-spinner" aria-hidden="true"></span>${stage}</span>`;
}

const progressById = new Map();

function renderDocs() {
  const list = js("doc-list");
  js("doc-count").textContent = String(state.docs.length);
  const pending = [...uploads.entries()].map(([id, u]) => `
    <li class="c-doc-row" aria-busy="true">
      <span class="c-doc-row__icon">${icon("upload")}</span>
      <div class="c-doc-row__body">
        <span class="c-doc-row__name u-truncate">${escapeHtml(u.name)}</span>
        <div class="c-progress" role="progressbar" aria-label="Uploading ${escapeHtml(u.name)}" aria-valuenow="${u.pct}" aria-valuemin="0" aria-valuemax="100"><div class="c-progress__bar" style="width:${u.pct}%"></div></div>
        <span class="c-meta">Uploading · ${u.pct}%</span>
      </div>
    </li>`).join("");
  if (!state.docs.length && !uploads.size) {
    list.innerHTML = '<li class="c-help px-2 py-4">No documents yet. Upload a PDF to begin.</li>';
  } else {
    list.innerHTML = pending + state.docs.map((d) => {
      const prog = progressById.get(d.id);
      const selected = state.selected.has(d.id);
      const meta = d.status === "ready" ? `${d.pages} pages · ${fmtDate(d.created_at)}` : fmtDate(d.created_at);
      const bar = d.status === "processing" && prog
        ? `<div class="c-progress" role="progressbar" aria-label="Processing ${escapeHtml(d.filename)}" aria-valuenow="${prog.progress}" aria-valuemin="0" aria-valuemax="100"><div class="c-progress__bar" style="width:${prog.progress}%"></div></div>` : "";
      const err = d.status === "failed" ? `<span class="c-meta c-meta--danger">${escapeHtml(d.error || "Processing failed.")}</span>` : "";
      return `<li class="c-doc-row" data-state="${selected ? "selected" : ""}" data-doc="${d.id}">
        <input type="checkbox" class="c-doc-row__check" data-js-select="${d.id}" ${selected ? "checked" : ""}
          ${d.status !== "ready" ? "disabled" : ""} aria-label="Include ${escapeHtml(d.filename)} in questions" />
        <span class="c-doc-row__icon">${icon("doc")}</span>
        <div class="c-doc-row__body">
          <span class="c-doc-row__name u-truncate" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</span>
          ${bar}
          <span class="c-doc-row__meta"><span class="c-meta">${meta}</span>${badge(d, prog)}</span>
          ${err}
        </div>
        <div class="c-menu">
          <button class="c-btn c-btn--ghost c-btn--icon c-btn--sm" data-menu="${d.id}" aria-haspopup="menu" aria-expanded="false" aria-label="More actions for ${escapeHtml(d.filename)}">${icon("dots")}</button>
          <ul class="c-menu__list" role="menu" hidden>
            <li role="none"><button class="c-menu__item c-menu__item--danger" role="menuitem" data-delete="${d.id}">Delete</button></li>
          </ul>
        </div>
      </li>`;
    }).join("");
  }
  renderScope();
  renderEmpty();
  emit("docs-changed");
}

js("doc-list").addEventListener("change", (e) => {
  const id = e.target.dataset.jsSelect;
  if (!id) return;
  e.target.checked ? state.selected.add(id) : state.selected.delete(id);
  e.target.closest(".c-doc-row").dataset.state = e.target.checked ? "selected" : "";
  renderScope();
});

/* overflow menus (documents and conversations share the behaviour) */
function closeMenus(except) {
  document.querySelectorAll("[data-menu]").forEach((b) => {
    if (b === except) return;
    b.setAttribute("aria-expanded", "false");
    b.nextElementSibling.hidden = true;
  });
}
document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-menu]");
  if (btn) {
    const open = btn.getAttribute("aria-expanded") !== "true";
    closeMenus(btn);
    btn.setAttribute("aria-expanded", String(open));
    btn.nextElementSibling.hidden = !open;
    if (open) btn.nextElementSibling.querySelector("button").focus();
    return;
  }
  if (!e.target.closest(".c-menu")) closeMenus();
});

js("doc-list").addEventListener("click", async (e) => {
  const retry = e.target.closest("[data-retry]");
  if (retry) {
    try {
      await request("POST", `/documents/${retry.dataset.retry}/retry`);
      progressById.delete(retry.dataset.retry);
      await loadDocs();
    } catch (err) { toast(err.message, "error"); }
    return;
  }
  const del = e.target.closest("[data-delete]");
  if (!del) return;
  closeMenus();
  const doc = state.docs.find((d) => d.id === del.dataset.delete);
  if (!doc) return;
  const ok = await confirmDanger({ title: "Delete this document?", body: `"${doc.filename}" and everything indexed from it will be removed. This can't be undone.` });
  if (!ok) return;
  try {
    await request("DELETE", `/documents/${doc.id}`);
    state.selected.delete(doc.id);
    toast(`Deleted ${doc.filename}.`);
    await loadDocs();
  } catch (err) { toast(err.message, "error"); }
});

async function handleFiles(files) {
  for (const file of files) {
    if (!/\.pdf$/i.test(file.name)) { toast(`"${file.name}" isn't a PDF. Choose a .pdf file.`, "error"); continue; }
    if (file.size > 100 * 1024 * 1024) { toast(`"${file.name}" is larger than 100 MB.`, "error"); continue; }
    const tmp = `up-${Math.random().toString(36).slice(2)}`;
    uploads.set(tmp, { name: file.name, pct: 0 });
    renderDocs();
    try {
      await uploadFile(file, (pct) => { uploads.get(tmp).pct = pct; renderDocs(); });
      uploads.delete(tmp);
      await loadDocs();
    } catch (err) {
      uploads.delete(tmp);
      renderDocs();
      toast(err.message, err.code === "duplicate" ? "success" : "error");
    }
  }
}

function schedulePoll() {
  clearInterval(pollTimer);
  const processing = state.docs.filter((d) => d.status === "processing");
  if (!processing.length) return;
  pollTimer = setInterval(async () => {
    let changed = false;
    for (const d of state.docs.filter((x) => x.status === "processing")) {
      try {
        const s = await request("GET", `/documents/${d.id}/status`);
        progressById.set(d.id, s);
        if (s.status !== "processing") {
          changed = true;
          if (s.status === "ready") toast(`${d.filename} is ready. Ask away.`);
          else toast(`${d.filename}: ${s.error}`, "error");
        }
      } catch (err) { if (err.status === 404) changed = true; }
    }
    if (changed) await loadDocs(); else renderDocs();
  }, 1000);
}

const dz = js("dropzone");
js("file-input").addEventListener("change", (e) => { handleFiles([...e.target.files]); e.target.value = ""; });
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.dataset.state = "dragover"; }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.dataset.state = ""; }));
dz.addEventListener("drop", (e) => handleFiles([...e.dataTransfer.files]));
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

/* ======================= scope & composer ======================= */

function renderScope() {
  const ready = readyDocs();
  const n = state.selected.size;
  const text = js("scope-text");
  if (!ready.length) {
    text.textContent = state.docs.some((d) => d.status === "processing") ? "Your PDF is still being prepared. You can ask once it's ready." : "Upload a PDF to start";
  } else if (n) {
    text.innerHTML = `Searching <strong>${n} of ${ready.length}</strong> document${ready.length > 1 ? "s" : ""}`;
  } else {
    text.innerHTML = `Searching <strong>all ${ready.length}</strong> document${ready.length > 1 ? "s" : ""}`;
  }
  updateComposer();
}

function updateComposer() {
  const noDocs = !readyDocs().length;
  js("question").disabled = noDocs && !state.busy;
  js("send").disabled = noDocs && !state.busy;
}
on("composer", updateComposer);

js("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.busy) return stop();
  const q = js("question").value.trim();
  if (!q) return js("question").focus();
  js("question").value = "";
  autosize();
  ask(q);
});

function autosize() {
  const q = js("question");
  q.style.height = "auto";
  q.style.height = `${Math.min(q.scrollHeight, 160)}px`;
  q.style.overflowY = q.scrollHeight > 160 ? "auto" : "hidden";
}
js("question").addEventListener("input", autosize);
js("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); js("composer").requestSubmit(); }
});

/* ======================= conversations ======================= */

async function loadConversations() {
  try {
    state.conversations = await request("GET", "/conversations");
    renderConversations();
  } catch (err) { if (err.status !== 401) toast(err.message, "error"); }
}
on("conversations-changed", loadConversations);

function renderConversations() {
  const list = js("conv-list");
  if (!state.conversations.length) {
    list.innerHTML = '<li class="c-help px-2 py-4">Your questions and answers will be saved here.</li>';
    return;
  }
  list.innerHTML = state.conversations.map((c) => `
    <li class="c-conv-row" data-conv="${c.id}" aria-current="${c.id === state.conversationId}">
      <button class="c-conv-row__open u-truncate" data-open="${c.id}" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</button>
      <div class="c-menu">
        <button class="c-btn c-btn--ghost c-btn--icon c-btn--sm" data-menu="${c.id}" aria-haspopup="menu" aria-expanded="false" aria-label="More actions for ${escapeHtml(c.title)}">${icon("dots")}</button>
        <ul class="c-menu__list" role="menu" hidden>
          <li role="none"><button class="c-menu__item" role="menuitem" data-rename="${c.id}">Rename</button></li>
          <li role="none"><button class="c-menu__item c-menu__item--danger" role="menuitem" data-delete-conv="${c.id}">Delete</button></li>
        </ul>
      </div>
    </li>`).join("");
}

js("conv-list").addEventListener("click", async (e) => {
  const open = e.target.closest("[data-open]");
  const rename = e.target.closest("[data-rename]");
  const del = e.target.closest("[data-delete-conv]");
  if (open) {
    try {
      if (state.view === "study") setView("chat");
      await loadConversation(open.dataset.open); renderConversations(); if (isMobile()) setSidebar(false);
    }
    catch (err) { toast(err.message, "error"); loadConversations(); }
  } else if (rename) {
    closeMenus();
    startRename(rename.dataset.rename);
  } else if (del) {
    closeMenus();
    const conv = state.conversations.find((c) => c.id === del.dataset.deleteConv);
    if (!conv || !(await confirmDanger({ title: "Delete this conversation?", body: `"${conv.title}" and its answers will be removed. This can't be undone.` }))) return;
    try {
      await request("DELETE", `/conversations/${conv.id}`);
      if (state.conversationId === conv.id) clearThread();
      toast("Conversation deleted.");
      loadConversations();
    } catch (err) { toast(err.message, "error"); }
  }
});

function startRename(id) {
  const row = js("conv-list").querySelector(`[data-conv="${id}"]`);
  const conv = state.conversations.find((c) => c.id === id);
  row.innerHTML = `<form class="flex flex-1 gap-2 py-1" data-rename-form>
    <label class="sr-only" for="rename-${id}">Conversation name</label>
    <input class="c-input" id="rename-${id}" maxlength="120" value="${escapeHtml(conv.title)}" />
    <button class="c-btn c-btn--primary c-btn--sm" type="submit">Save</button>
  </form>`;
  const input = row.querySelector("input");
  input.focus();
  input.select();
  const cancel = () => renderConversations();
  input.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); cancel(); } });
  row.querySelector("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = input.value.trim();
    if (!title) return cancel();
    try {
      await request("PATCH", `/conversations/${id}`, { json: { title } });
      if (state.conversationId === id) js("conv-title").textContent = title;
      await loadConversations();
    } catch (err) { toast(err.message, "error"); cancel(); }
  });
}

for (const name of ["new-chat", "new-chat-top"]) {
  js(name).addEventListener("click", () => {
    stop();
    clearThread();
    renderConversations();
    if (isMobile()) setSidebar(false);
    if (state.view === "study") setView("chat");
    js("question").focus();
  });
}

/* ======================= settings ======================= */

const MODE_HELP = {
  local: "Finds and quotes the answer directly from your PDF. Free, private, works offline.",
  ollama: "Writes an answer using a model running on your computer.",
  api: "Writes a fuller answer using your AI provider (Claude, OpenAI, Gemini and more). Needs an API key.",
};

/** Load which modes can run (the pasted key travels in a header, never stored) and pick one. */
async function loadModes() {
  try {
    const m = await request("GET", "/chat/modes", { headers: keyHeaders() });
    state.modes = m;
    const available = new Set(m.modes.filter((x) => x.available).map((x) => x.mode));
    const preferred = m.default_mode !== "auto" && available.has(m.default_mode) ? m.default_mode : null;
    if (!state.mode || !available.has(state.mode) || preferred) state.mode = preferred || m.recommended;
    renderModes();
  } catch (err) {
    js("llm-status").textContent = "Couldn't reach the server";
    js("llm-dot").className = "c-dot c-dot--bad";
  }
}
on("modes-changed", () => loadModes());

function renderModes() {
  const m = state.modes;
  if (!m) return;
  const byMode = Object.fromEntries(m.modes.map((x) => [x.mode, x]));
  document.querySelectorAll('[data-js="mode-group"] [data-mode]').forEach((b) => {
    const info = byMode[b.dataset.mode];
    b.disabled = !info.available;
    b.title = info.available ? info.description : info.reason;
    b.setAttribute("aria-checked", String(b.dataset.mode === state.mode));
  });
  const unavailable = m.modes.filter((x) => !x.available);
  const setup = unavailable.length
    ? ` <span class="u-muted">${unavailable.map((x) => `${escapeHtml(x.title)}: ${escapeHtml(x.reason)}`).join(" ")}</span> <button type="button" class="c-link" data-js="mode-setup">Set up</button>`
    : "";
  js("mode-help").innerHTML = `${escapeHtml(MODE_HELP[state.mode] || "")}${setup}`;
  js("mode-setup")?.addEventListener("click", openSettings);
  js("key-invite").hidden = !(state.mode === "local" && !byMode.api.available);
  js("llm-dot").className = "c-dot c-dot--ok";
  js("llm-status").textContent = `Answering: ${MODE_LABELS[state.mode]}`;
  const hasKey = Boolean(getApiKey());
  const provider = m.providers.find((p) => p.id === getKeyMeta().provider);
  js("key-badge").className = `c-badge ${hasKey ? "c-badge--ready" : "c-badge--neutral"}`;
  js("key-badge").innerHTML = hasKey ? `${icon("check")}${escapeHtml(provider ? provider.name : "Saved in this browser")}` : "Not set";
  renderProviders(m.providers);
  js("default-mode").value = m.default_mode;
  const select = js("ollama-model");
  select.innerHTML = m.ollama_models.length
    ? m.ollama_models.map((n) => `<option value="${escapeHtml(n)}"${n === m.ollama_model || n === `${m.ollama_model}:latest` ? " selected" : ""}>${escapeHtml(n)}</option>`).join("")
    : `<option value="">${escapeHtml(byMode.ollama.reason)}</option>`;
  select.disabled = !m.ollama_models.length;
  emit("modes-loaded");
}

/* ======================= chat / study view ======================= */

function setView(view) {
  state.view = view;
  const study = view === "study";
  document.querySelectorAll('[data-js="view-switch"] [data-view]').forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  js("scroller").hidden = study;
  js("composer").hidden = study;
  js("study-view").hidden = !study;
  js("new-chat-top").hidden = study;
  js("conv-title").textContent = study ? "Study material" : (state.conversations.find((c) => c.id === state.conversationId)?.title || "New conversation");
  writePref("view", view);
  if (study) { renderStudyForm(); js("study-doc").focus(); } else js("question").focus();
}
js("view-switch").addEventListener("click", (e) => {
  const b = e.target.closest("[data-view]");
  if (b) setView(b.dataset.view);
});

async function savePreferences(body) {
  try {
    await request("PUT", "/me/preferences", { json: body });
  } catch (err) { toast(err.message, "error"); }
}

function setMode(mode) {
  const info = state.modes && state.modes.modes.find((x) => x.mode === mode);
  if (!info || !info.available) {
    toast(info ? `${info.title} isn't available: ${info.reason}` : "That mode isn't available.", "error");
    if (mode === "api") openSettings();
    return;
  }
  state.mode = mode;
  renderModes();
  savePreferences({ default_mode: mode }); // the chosen mode is saved per user
}
on("set-mode", setMode);

js("mode-group").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (b && !b.disabled) setMode(b.dataset.mode);
});
js("key-invite-btn").addEventListener("click", () => openSettings());

js("default-mode").addEventListener("change", async (e) => {
  await savePreferences({ default_mode: e.target.value });
  state.mode = null;
  await loadModes();
});
js("ollama-model").addEventListener("change", (e) => { if (e.target.value) savePreferences({ ollama_model: e.target.value }); });

/** Fill the provider list once (it comes from /chat/modes, so the server decides what's supported). */
function renderProviders(providers) {
  const select = js("key-provider");
  if (select.options.length > 1 || !providers?.length) return;
  select.insertAdjacentHTML("beforeend", providers.map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.name)}</option>`).join(""));
  select.value = getKeyMeta().provider || "auto";
  updateKeyLink();
}

function updateKeyLink() {
  const providers = state.modes?.providers || [];
  const chosen = providers.find((p) => p.id === js("key-provider").value) || providers.find((p) => p.id === "anthropic");
  if (chosen) js("key-link").href = chosen.key_url;
  js("key-link").textContent = chosen && js("key-provider").value !== "auto" ? `Get a ${chosen.name} key` : "Where do I get a key?";
}

function renderModelChoices(models, selected) {
  const field = js("key-model-field");
  field.hidden = !models.length;
  js("key-model").innerHTML = models.map((id) => `<option value="${escapeHtml(id)}"${id === selected ? " selected" : ""}>${escapeHtml(id)}</option>`).join("");
}

js("key-provider").addEventListener("change", () => {
  updateKeyLink();
  renderModelChoices([], ""); // the model list belongs to the previous provider
});

js("key-model").addEventListener("change", (e) => {
  const key = getApiKey();
  if (!key) return;
  setApiKey(key, isKeyRemembered(), { ...getKeyMeta(), model: e.target.value });
  js("key-result").textContent = `Model set to ${e.target.value}.`;
});

function openSettings() {
  const k = getApiKey();
  const meta = getKeyMeta();
  js("key-input").value = "";
  js("key-input").placeholder = k ? `Saved key ending in ${k.slice(-4)}. Paste a new one to replace it.` : "Paste your key";
  if (js("key-provider").options.length > 1) js("key-provider").value = meta.provider || "auto";
  renderModelChoices(meta.model ? [meta.model] : [], meta.model);
  updateKeyLink();
  js("key-remember").checked = isKeyRemembered();
  js("key-remember-warn").hidden = !js("key-remember").checked;
  js("key-remove").disabled = !k;
  js("key-error").hidden = true;
  js("key-input").removeAttribute("aria-invalid");
  js("key-result").textContent = "";
  const choice = themeChoice();
  document.querySelectorAll("[data-theme-choice]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeChoice === choice)));
  js("debug-toggle").checked = state.debug;
  loadModes();
  showDrawer("settings-drawer", { focus: js("key-input") });
}
js("open-settings").addEventListener("click", openSettings);
on("open-settings", openSettings);
js("key-remember").addEventListener("change", (e) => { js("key-remember-warn").hidden = !e.target.checked; });

function keyError(message) {
  const err = js("key-error");
  err.innerHTML = `${icon("alert")}<span>${escapeHtml(message)}</span>`;
  err.hidden = false;
  js("key-input").setAttribute("aria-invalid", "true");
  js("key-input").focus();
}

js("key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  js("key-error").hidden = true;
  js("key-input").removeAttribute("aria-invalid");
  const typed = js("key-input").value.trim();
  const key = typed || getApiKey();
  if (!key) return keyError("Paste your API key first.");
  if (/\s/.test(key)) return keyError("The key contains spaces. Copy it again without extra text.");
  const provider = js("key-provider").value;
  const headers = { "X-LLM-Key": key, "X-LLM-Provider": provider };
  if (!js("key-model-field").hidden && js("key-model").value) headers["X-LLM-Model"] = js("key-model").value;
  const btn = js("key-save");
  busyButton(btn, true, "Testing…");
  js("key-result").textContent = "Checking the key with your provider…";
  let r;
  try {
    r = await request("POST", "/chat/test-key", { headers });
  } catch (err) {
    r = { ok: false, message: err.message };
  }
  busyButton(btn, false, "Test and save key");
  if (r.ok) {
    setApiKey(key, js("key-remember").checked, { provider: r.provider, model: r.model });
    js("key-provider").value = r.provider;
    updateKeyLink();
    renderModelChoices(r.models, r.model);
    js("key-remove").disabled = false;
    js("key-input").value = "";
    js("key-input").placeholder = `Saved key ending in ${key.slice(-4)}.`;
    js("key-result").textContent = r.message;
    toast(`${r.provider_name} is ready. API key mode will use ${r.model}.`);
    await loadModes();
  } else {
    keyError(`${r.message} The key was not saved.`);
    js("key-result").textContent = "";
  }
});

js("key-reveal").addEventListener("click", () => {
  const input = js("key-input");
  const show = input.type === "password";
  input.type = show ? "text" : "password";
  js("key-reveal").textContent = show ? "Hide" : "Show";
  js("key-reveal").setAttribute("aria-pressed", String(show));
});

js("key-remove").addEventListener("click", async () => {
  setApiKey("", false);
  js("key-remove").disabled = true;
  js("key-input").placeholder = "Paste your key";
  js("key-provider").value = "auto";
  renderModelChoices([], "");
  updateKeyLink();
  js("key-result").textContent = "Your key was removed from this browser.";
  state.mode = null;
  await loadModes();
});

js("theme-group").addEventListener("click", (e) => {
  const b = e.target.closest("[data-theme-choice]");
  if (!b) return;
  setTheme(b.dataset.themeChoice);
  document.querySelectorAll("[data-theme-choice]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
});

js("debug-toggle").addEventListener("change", (e) => {
  setDebug(e.target.checked);
  writePref("debug", e.target.checked ? "1" : "0");
});

js("sign-out").addEventListener("click", () => signOut());

document.querySelectorAll('[data-js="close-drawer"]').forEach((b) => b.addEventListener("click", () => closeDrawer()));
js("drawer-scrim").addEventListener("click", () => closeDrawer());

/* ======================= keyboard ======================= */

function toggleShortcuts(force) {
  const pop = document.getElementById("shortcuts");
  const open = force ?? pop.hidden;
  pop.hidden = !open;
  js("shortcuts-btn").setAttribute("aria-expanded", String(open));
}
js("shortcuts-btn").addEventListener("click", () => toggleShortcuts());

document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
  if (e.key === "Escape") {
    if (!document.getElementById("shortcuts").hidden) { toggleShortcuts(false); js("shortcuts-btn").focus(); return; }
    if (isDrawerOpen()) { closeDrawer(); return; }
    if (js("sidebar").dataset.state === "open") { setSidebar(false); js("open-sidebar").focus(); return; }
    closeMenus();
    return;
  }
  if (typing || js("app-view").hidden) return;
  if (e.key === "/" && state.view === "chat") { e.preventDefault(); js("question").focus(); }
  else if (e.key === "?") { e.preventDefault(); toggleShortcuts(true); }
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".c-menu")) toggleShortcuts(false);
});

/* ======================= boot ======================= */

(async function boot() {
  setAuthMode("login");
  try {
    await enterApp(await request("GET", "/auth/me"));
  } catch (_) {
    js("auth-view").hidden = false;
    js("guest").focus();
  }
})();
