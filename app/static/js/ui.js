// Shared UI primitives: DOM helpers, toasts, confirm modal, focus trap, drawers, theme, markdown.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const js = (name, root = document) => root.querySelector(`[data-js="${name}"]`);

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export const icon = (name, extra = "") => `<svg class="c-icon ${extra}" aria-hidden="true"><use href="#i-${name}"/></svg>`;

export function pageLabel(a, b) { return a === b ? `p. ${a}` : `pp. ${a}–${b}`; }

export function fmtDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/* ---------- toasts: success auto-dismisses after 5s, errors persist until dismissed ---------- */
export function toast(message, kind = "success") {
  const box = js("toasts");
  const el = document.createElement("div");
  el.className = `c-toast c-toast--${kind}`;
  el.setAttribute("role", kind === "error" ? "alert" : "status");
  el.innerHTML = `<span class="c-toast__icon">${icon(kind === "error" ? "alert" : "check")}</span>
    <p class="c-toast__msg">${escapeHtml(message)}</p>
    <button class="c-btn c-btn--ghost c-btn--icon c-btn--sm" aria-label="Dismiss">${icon("x")}</button>`;
  const remove = () => { el.dataset.state = "leaving"; setTimeout(() => el.remove(), 200); };
  el.querySelector("button").addEventListener("click", remove);
  box.appendChild(el);
  if (kind !== "error") setTimeout(remove, 5000);
}

/* ---------- focus trap ---------- */
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select, [tabindex]:not([tabindex="-1"])';

export function trapFocus(container) {
  const onKey = (e) => {
    if (e.key !== "Tab") return;
    const items = $$(FOCUSABLE, container).filter((el) => el.offsetParent !== null);
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };
  container.addEventListener("keydown", onKey);
  return () => container.removeEventListener("keydown", onKey);
}

/* ---------- drawers (one open at a time); focus returns to the opener ---------- */
let openDrawer = null;

export function showDrawer(name, { opener, focus } = {}) {
  closeDrawer({ restore: false });
  const el = js(name);
  const scrim = js("drawer-scrim");
  el.hidden = false;
  scrim.hidden = false;
  requestAnimationFrame(() => { el.dataset.state = "open"; scrim.dataset.state = "open"; });
  const release = trapFocus(el);
  openDrawer = { el, scrim, opener: opener || document.activeElement, release };
  (focus || js("close-drawer", el)).focus();
}

export function closeDrawer({ restore = true } = {}) {
  if (!openDrawer) return false;
  const { el, scrim, opener, release } = openDrawer;
  openDrawer = null;
  release();
  el.dataset.state = "";
  scrim.dataset.state = "";
  setTimeout(() => { if (el.dataset.state !== "open") { el.hidden = true; scrim.hidden = true; } }, 200);
  if (restore && opener && document.contains(opener)) opener.focus();
  return true;
}

export const isDrawerOpen = () => openDrawer !== null;

/* ---------- confirm modal (destructive actions only; default focus on Cancel) ---------- */
export function confirmDanger({ title, body, confirmLabel = "Delete" }) {
  return new Promise((resolve) => {
    const wrap = js("modal");
    const opener = document.activeElement;
    js("modal-title").textContent = title;
    js("modal-body").textContent = body;
    js("modal-confirm").textContent = confirmLabel;
    wrap.hidden = false;
    const release = trapFocus(wrap);
    const done = (value) => {
      wrap.hidden = true;
      release();
      document.removeEventListener("keydown", onKey, true);
      cleanup.forEach((fn) => fn());
      if (opener && document.contains(opener)) opener.focus();
      resolve(value);
    };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); done(false); } };
    const handlers = [["modal-cancel", () => done(false)], ["modal-confirm", () => done(true)], ["modal-scrim", () => done(false)]];
    const cleanup = handlers.map(([n, fn]) => { const b = js(n); b.addEventListener("click", fn); return () => b.removeEventListener("click", fn); });
    document.addEventListener("keydown", onKey, true);
    js("modal-cancel").focus();
  });
}

/* ---------- theme: follows the OS unless the user picks one; remembered per user ---------- */
const media = window.matchMedia("(prefers-color-scheme: dark)");
let themeKey = "rag-theme:last";

function store(key, value) { try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch (_) { /* blocked */ } }
function read(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }

export function themeChoice() { return read(themeKey) || "system"; }

export function applyTheme(choice = themeChoice()) {
  const dark = choice === "dark" || (choice === "system" && media.matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

export function setTheme(choice) {
  store(themeKey, choice === "system" ? null : choice);
  store("rag-theme:last", choice === "system" ? null : choice);
  applyTheme(choice);
}

export function useThemeFor(userId) {
  themeKey = userId ? `rag-theme:${userId}` : "rag-theme:last";
  applyTheme();
}

media.addEventListener("change", () => { if (themeChoice() === "system") applyTheme("system"); });

/* ---------- tiny, safe Markdown subset (input escaped first) with [n] → citation chips ---------- */
function inline(s) {
  return s.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

export function renderAnswer(text, knownNumbers) {
  const lines = escapeHtml(text).split("\n");
  let html = "";
  let list = null;
  let para = [];
  const flush = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = raw.trim();
    const ul = line.match(/^[-*•]\s+(.*)$/);
    const ol = line.match(/^\d+[.)]\s+(.*)$/);
    const quote = line.match(/^&gt;\s?(.*)$/); // "> quoted sentence [n]" (escaped by escapeHtml)
    if (quote) {
      flush(); close();
      html += `<blockquote>${inline(quote[1])}</blockquote>`;
    } else if (ul || ol) {
      flush();
      const tag = ul ? "ul" : "ol";
      if (list !== tag) { close(); html += `<${tag}>`; list = tag; }
      html += `<li>${inline((ul || ol)[1])}</li>`;
    } else if (!line) { flush(); close(); } else { close(); para.push(line); }
  }
  flush(); close();
  return html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, group) => group.split(",").map((n) => {
    const num = Number(n.trim());
    return knownNumbers.has(num)
      ? `<button type="button" class="c-chip" data-cite="${num}" aria-label="Open source ${num}">${num}</button>`
      : `<span class="c-chip c-chip--inactive" aria-label="Source ${num}">${num}</span>`;
  }).join(""));
}

/** Escape text, mark the exact extracted answer [start, end), and emphasise question words elsewhere. */
export function highlightWithSpan(text, range, question) {
  if (!range || range.length !== 2 || range[0] >= range[1] || range[1] > text.length) return highlight(text, question);
  const [a, b] = range;
  return `${highlight(text.slice(0, a), question)}<mark class="c-mark--answer">${escapeHtml(text.slice(a, b))}</mark>${highlight(text.slice(b), question)}`;
}

/** Escape text and emphasise words from the question (for the source drawer). */
export function highlight(text, question) {
  const stop = new Set("a an and are as at be by for from has have in is it its of on or that the this to was were what when where which who why how with does do did can i my me you".split(" "));
  const terms = [...new Set((question.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => t.length > 2 && !stop.has(t)))];
  const safe = escapeHtml(text);
  if (!terms.length) return safe;
  const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*`, "gi");
  return safe.replace(re, "<mark>$&</mark>");
}
