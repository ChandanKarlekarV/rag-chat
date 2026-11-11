// Study view: make notes, sticky notes, a mind map, flowcharts, key terms and flashcards from one PDF.

import { request } from "./api.js";
import { MODE_LABELS, keyHeaders, on, readyDocs, state } from "./state.js";
import { escapeHtml, icon, js, toast } from "./ui.js";

const KIND_TITLES = {
  notes: "Short notes", sticky: "Sticky notes", mindmap: "Mind map", flow: "Flowchart", glossary: "Key terms", flashcards: "Flashcards",
};
const MODE_HELP = {
  local: "Quotes the most important sentences straight from your PDF. Free, private, a few seconds.",
  ollama: "Rewrites the material in plain words with the model on your computer. Takes up to a minute.",
  api: "Rewrites the material in plain words with your AI provider. Takes up to a minute.",
};

let current = null; // last StudyPack
let busy = false;

const page = (n) => (n ? `<span class="c-page">p. ${n}</span>` : "");

/* ======================= form ======================= */

export function renderStudyForm() {
  const select = js("study-doc");
  const ready = readyDocs();
  const keep = select.value;
  select.innerHTML = ready.length
    ? ready.map((d) => `<option value="${d.id}">${escapeHtml(d.filename)}</option>`).join("")
    : '<option value="">Upload a PDF first</option>';
  if (ready.some((d) => d.id === keep)) select.value = keep;
  else if (state.selected.size) select.value = [...state.selected].find((id) => ready.some((d) => d.id === id)) || select.value;
  select.disabled = !ready.length;
  renderModeChoices();
  js("study-go").disabled = busy || !ready.length;
}

function renderModeChoices() {
  const modes = state.modes?.modes || [];
  const select = js("study-mode");
  const keep = select.value || state.mode || "local";
  select.innerHTML = modes.map((m) => `<option value="${m.mode}" ${m.available ? "" : "disabled"}>${escapeHtml(MODE_LABELS[m.mode])}${m.available ? "" : " (not set up)"}</option>`).join("");
  const available = modes.filter((m) => m.available).map((m) => m.mode);
  select.value = available.includes(keep) ? keep : available[0] || "local";
  js("study-mode-help").textContent = MODE_HELP[select.value] || "";
}

js("study-mode").addEventListener("change", (e) => { js("study-mode-help").textContent = MODE_HELP[e.target.value] || ""; });
on("docs-changed", renderStudyForm);
on("modes-loaded", renderStudyForm);

js("study-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  const err = js("study-error");
  err.hidden = true;
  const docId = js("study-doc").value;
  const kinds = [...js("study-kinds").querySelectorAll("input:checked")].map((i) => i.value);
  if (!docId) return showError("Upload a PDF first, then come back here.");
  if (!kinds.length) return showError("Choose at least one kind of study material.");
  const mode = js("study-mode").value || null; // null = the saved default / automatic choice
  setBusy(true, mode);
  try {
    current = await request("POST", "/study", { json: { doc_id: docId, kinds, mode }, headers: keyHeaders() });
    renderResult(current);
    js("study-result").querySelector("h2")?.focus();
  } catch (ex) {
    showError(ex.message);
    js("study-result").innerHTML = "";
  } finally {
    setBusy(false);
  }
});

function showError(message) {
  const err = js("study-error");
  err.innerHTML = `${icon("alert")}<span>${escapeHtml(message)}</span>`;
  err.hidden = false;
}

function setBusy(on, mode) {
  busy = on;
  const btn = js("study-go");
  btn.disabled = on;
  btn.setAttribute("aria-busy", String(on));
  btn.querySelector(".c-spinner")?.remove();
  btn.querySelector(".c-btn__label").textContent = on ? "Making your study material…" : "Make study material";
  if (on) btn.insertAdjacentHTML("afterbegin", '<span class="c-spinner" aria-hidden="true"></span>');
  const out = js("study-result");
  out.setAttribute("aria-busy", String(on));
  if (on) {
    const wait = mode === "local" ? "This takes a few seconds." : "The model is writing. This can take up to a minute.";
    out.innerHTML = `<div class="c-card flex flex-col gap-3"><p class="c-text">${escapeHtml(wait)}</p>
      <div class="c-skeleton c-skeleton--row" aria-hidden="true"></div><div class="c-skeleton c-skeleton--row" aria-hidden="true"></div></div>`;
  }
}

/* ======================= result ======================= */

function renderResult(pack) {
  const tabs = pack.kinds.map((k, i) => `<button class="c-tabs__tab" role="tab" id="stab-${k}" aria-selected="${i === 0}" aria-controls="spanel-${k}" tabindex="${i === 0 ? 0 : -1}" data-study-tab="${k}">${KIND_TITLES[k]}</button>`).join("");
  const panels = pack.kinds.map((k, i) => `<div class="c-study__panel" role="tabpanel" id="spanel-${k}" aria-labelledby="stab-${k}" tabindex="0" ${i === 0 ? "" : "hidden"}>${PANELS[k](pack)}</div>`).join("");
  const warnings = pack.warnings.map((w) => `<p class="c-notice">${icon("alert")}<span class="flex-1">${escapeHtml(w)}</span></p>`).join("");
  js("study-result").innerHTML = `
    <div class="flex flex-col gap-4">
      <div class="c-study__head">
        <div class="min-w-0">
          <h2 class="c-heading" tabindex="-1">${escapeHtml(pack.title)}</h2>
          <p class="c-meta">${escapeHtml(pack.label)} · ${pack.pages} page${pack.pages === 1 ? "" : "s"} · ${pack.seconds.toFixed(1)} s</p>
        </div>
        <div class="flex flex-wrap gap-2" data-js="study-actions">
          <button type="button" class="c-btn c-btn--secondary c-btn--sm" data-js="study-download">${icon("doc")}Download as Markdown</button>
          <button type="button" class="c-btn c-btn--ghost c-btn--sm" data-js="study-print">Print</button>
        </div>
      </div>
      ${warnings}
      <div class="c-tabs" role="tablist" aria-label="Study material">${tabs}</div>
      ${panels}
    </div>`;
  js("study-download").addEventListener("click", () => download(pack));
  js("study-print").addEventListener("click", () => window.print());
}

const PANELS = {
  notes: (p) => (p.notes.length ? `<div class="c-notes">${p.notes.map((s) => `
    <article class="c-card"><h3 class="c-label flex items-center gap-2">${escapeHtml(s.title)} ${page(s.page)}</h3>
      <ul>${s.points.map((pt) => `<li>${escapeHtml(pt.text)} ${page(pt.page)}</li>`).join("")}</ul></article>`).join("")}</div>` : empty("notes")),
  sticky: (p) => (p.sticky.length ? `<ul class="c-stickies">${p.sticky.map((n) => `
    <li class="c-sticky" data-tone="${n.tone}"><p class="c-sticky__title">${escapeHtml(n.title)}</p>
      <p class="c-sticky__text">${escapeHtml(n.text)}</p>${page(n.page)}</li>`).join("")}</ul>` : empty("sticky notes")),
  mindmap: (p) => (p.mindmap ? mindmap(p.mindmap) : empty("a mind map")),
  flow: (p) => (p.flows.length ? p.flows.map((f) => `
    <section class="flex flex-col gap-3" style="margin-bottom: var(--space-8)"><h3 class="c-label">${escapeHtml(f.title)}</h3>
      <ol class="c-flow">${f.steps.map((s, i) => `<li class="c-flow__step"><span class="c-flow__num" aria-hidden="true">${i + 1}</span>
        <p class="c-flow__text"><span class="sr-only">Step ${i + 1}: </span>${escapeHtml(s.text)} ${page(s.page)}</p></li>`).join("")}</ol></section>`).join("")
    : empty("a step-by-step process")),
  glossary: (p) => (p.glossary.length ? `<dl class="c-glossary">${p.glossary.map((g) => `
    <div><dt>${escapeHtml(g.term)} ${page(g.page)}</dt><dd>${escapeHtml(g.definition)}</dd></div>`).join("")}</dl>` : empty("key terms")),
  flashcards: (p) => (p.flashcards.length ? `<p class="c-help" style="margin-bottom: var(--space-3)">Click a card to see the answer.</p>
    <ul class="c-flashcards">${p.flashcards.map((c, i) => `<li><button type="button" class="c-flashcard" aria-pressed="false" data-card="${i}">
      <span class="c-flashcard__side">Question ${i + 1}</span><span class="c-flashcard__text">${escapeHtml(c.front)}</span>
      <span class="c-meta">Show answer</span></button></li>`).join("")}</ul>` : empty("flashcards")),
};

function empty(what) {
  return `<p class="c-help">We couldn't find enough in this document to make ${escapeHtml(what)}.</p>`;
}

js("study-result").addEventListener("click", (e) => {
  const tab = e.target.closest("[data-study-tab]");
  if (tab) return selectTab(tab);
  const card = e.target.closest("[data-card]");
  if (card && current) flip(card);
});

js("study-result").addEventListener("keydown", (e) => {
  const tab = e.target.closest("[data-study-tab]");
  if (!tab || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
  const all = [...js("study-result").querySelectorAll("[data-study-tab]")];
  const i = all.indexOf(tab);
  const next = { ArrowLeft: all[(i - 1 + all.length) % all.length], ArrowRight: all[(i + 1) % all.length], Home: all[0], End: all[all.length - 1] }[e.key];
  e.preventDefault();
  selectTab(next);
  next.focus();
});

function selectTab(tab) {
  js("study-result").querySelectorAll("[data-study-tab]").forEach((t) => {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
  });
}

function flip(btn) {
  const card = current.flashcards[Number(btn.dataset.card)];
  const showBack = btn.getAttribute("aria-pressed") !== "true";
  btn.setAttribute("aria-pressed", String(showBack));
  const n = Number(btn.dataset.card) + 1;
  btn.innerHTML = showBack
    ? `<span class="c-flashcard__side">Answer ${n}</span><span class="c-flashcard__text">${escapeHtml(card.back)}</span><span class="c-meta">${card.page ? `Page ${card.page} · ` : ""}Show question</span>`
    : `<span class="c-flashcard__side">Question ${n}</span><span class="c-flashcard__text">${escapeHtml(card.front)}</span><span class="c-meta">Show answer</span>`;
}

/* ======================= mind map (SVG, laid out here) ======================= */

// Layout constants (px, in SVG user units; the SVG scales down to fit and scrolls below MIN_WIDTH).
const LINE_H = 18;
const PAD_X = 12;
const PAD_Y = 7;
const GAP_Y = 8; // between leaves
const GROUP_GAP = 20; // between branches on one side
const COL_GAP = 40;
const MAX_TEXT = { root: 200, branch: 170, leaf: 170 };

function textWidth(text, weight) {
  const ctx = (textWidth.ctx ||= document.createElement("canvas").getContext("2d"));
  const css = getComputedStyle(document.documentElement);
  ctx.font = `${weight} ${css.getPropertyValue("--text-sm").trim() || "14px"} ${css.getPropertyValue("--font-sans")}`;
  return Math.ceil(ctx.measureText(text).width);
}

/** Wrap a label into at most two lines that fit `max` px. */
function wrap(label, weight, max) {
  const words = label.split(/\s+/);
  const lines = [""];
  for (const w of words) {
    const tryLine = lines[lines.length - 1] ? `${lines[lines.length - 1]} ${w}` : w;
    if (textWidth(tryLine, weight) <= max || !lines[lines.length - 1]) lines[lines.length - 1] = tryLine;
    else if (lines.length < 2) lines.push(w);
    else { lines[1] += ` ${w}`; }
  }
  if (lines[1] && textWidth(lines[1], weight) > max) {
    let s = lines[1];
    while (s.length > 1 && textWidth(`${s}…`, weight) > max) s = s.slice(0, -1);
    lines[1] = `${s.trimEnd()}…`;
  }
  return lines;
}

function box(node, kind) {
  const weight = kind === "leaf" ? 400 : 600;
  const lines = wrap(node.label, weight, MAX_TEXT[kind]);
  const w = Math.max(...lines.map((l) => textWidth(l, weight))) + PAD_X * 2;
  return { ...node, kind, lines, w, h: lines.length * LINE_H + PAD_Y * 2 };
}

/** A two-sided mind map: the topic in the middle, main ideas on both sides, details further out. */
function mindmap(root) {
  const branches = (root.children.length ? root.children : [{ label: "No sections found", children: [] }])
    .map((b) => ({ ...box(b, "branch"), leaves: (b.children || []).map((l) => box(l, "leaf")) }));
  for (const b of branches) {
    const leavesH = b.leaves.reduce((s, l) => s + l.h, 0) + GAP_Y * Math.max(0, b.leaves.length - 1);
    b.blockH = Math.max(b.h, leavesH);
  }
  // Balance the two sides by height (greedy, keeps reading order within a side).
  const sides = { right: [], left: [] };
  const sideH = { right: 0, left: 0 };
  branches.forEach((b) => {
    const side = sideH.right <= sideH.left ? "right" : "left";
    sides[side].push(b);
    sideH[side] += b.blockH + GROUP_GAP;
  });
  const center = box(root, "root");
  const height = Math.max(sideH.right, sideH.left, center.h + GROUP_GAP) - GROUP_GAP;
  const branchW = { right: Math.max(0, ...sides.right.map((b) => b.w)), left: Math.max(0, ...sides.left.map((b) => b.w)) };
  const leafW = {
    right: Math.max(0, ...sides.right.flatMap((b) => b.leaves.map((l) => l.w))),
    left: Math.max(0, ...sides.left.flatMap((b) => b.leaves.map((l) => l.w))),
  };
  const leftExtent = sides.left.length ? leafW.left + COL_GAP + branchW.left + COL_GAP : 0;
  center.x = leftExtent;
  center.cy = height / 2;
  const nodes = [center];
  const edges = [];
  const curve = (ax, ay, bx, by) => {
    const mid = (ax + bx) / 2;
    return `M${ax},${ay} C${mid},${ay} ${mid},${by} ${bx},${by}`;
  };
  for (const side of ["right", "left"]) {
    const list = sides[side];
    let y = (height - (sideH[side] - GROUP_GAP)) / 2;
    const dir = side === "right" ? 1 : -1;
    const branchEdgeX = side === "right" ? center.x + center.w + COL_GAP : center.x - COL_GAP; // inner edge of branch column
    for (const b of list) {
      b.cy = y + b.blockH / 2;
      b.x = side === "right" ? branchEdgeX : branchEdgeX - b.w;
      nodes.push(b);
      edges.push(curve(side === "right" ? center.x + center.w : center.x, center.cy, side === "right" ? b.x : b.x + b.w, b.cy));
      const leavesH = b.leaves.reduce((s, l) => s + l.h, 0) + GAP_Y * Math.max(0, b.leaves.length - 1);
      let ly = b.cy - leavesH / 2;
      const leafEdgeX = branchEdgeX + dir * (branchW[side] + COL_GAP);
      for (const l of b.leaves) {
        l.cy = ly + l.h / 2;
        l.x = side === "right" ? leafEdgeX : leafEdgeX - l.w;
        nodes.push(l);
        edges.push(curve(side === "right" ? b.x + b.w : b.x, b.cy, side === "right" ? l.x : l.x + l.w, l.cy));
        ly += l.h + GAP_Y;
      }
      y += b.blockH + GROUP_GAP;
    }
  }
  const totalW = center.x + center.w + (sides.right.length ? COL_GAP + branchW.right + COL_GAP + leafW.right : 0);
  const svgNodes = nodes.map((n) => {
    const top = n.cy - n.h / 2;
    const tspans = n.lines.map((line, i) => `<tspan x="${n.x + PAD_X}" y="${top + PAD_Y + LINE_H * i + LINE_H / 2}">${escapeHtml(line)}</tspan>`).join("");
    return `<g class="c-mindmap__node c-mindmap__node--${n.kind}"><title>${escapeHtml(n.label)}${n.page ? ` (page ${n.page})` : ""}</title>
      <rect x="${n.x}" y="${top}" width="${n.w}" height="${n.h}" rx="${n.kind === "root" ? n.h / 2 : 8}"/><text>${tspans}</text></g>`;
  }).join("");
  const svgEdges = edges.map((d) => `<path class="c-mindmap__edge" d="${d}"/>`).join("");
  const outline = `<ul class="sr-only">${branches.map((b) => `<li>${escapeHtml(b.label)}<ul>${b.leaves.map((l) => `<li>${escapeHtml(l.label)}</li>`).join("")}</ul></li>`).join("")}</ul>`;
  return `<div class="c-mindmap"><svg viewBox="-2 -2 ${totalW + 4} ${height + 4}" style="width:100%;min-width:${Math.min(totalW, 720)}px;aspect-ratio:${(totalW + 4) / (height + 4)}" role="img" aria-label="Mind map of ${escapeHtml(root.label)}">${svgEdges}${svgNodes}</svg>${outline}</div>`;
}

/* ======================= export ======================= */

function toMarkdown(p) {
  const pg = (n) => (n ? ` (p. ${n})` : "");
  const out = [`# ${p.title}`, "", `_${p.label}_`, ""];
  if (p.notes.length) {
    out.push("## Short notes", "");
    p.notes.forEach((s) => { out.push(`### ${s.title}${pg(s.page)}`, ...s.points.map((pt) => `- ${pt.text}${pg(pt.page)}`), ""); });
  }
  if (p.sticky.length) out.push("## Key points", "", ...p.sticky.map((n) => `- **${n.title}:** ${n.text}${pg(n.page)}`), "");
  if (p.mindmap) {
    out.push("## Mind map", "", `- ${p.mindmap.label}`);
    p.mindmap.children.forEach((s) => { out.push(`  - ${s.label}`, ...s.children.map((l) => `    - ${l.label}`)); });
    out.push("");
  }
  p.flows.forEach((f) => { out.push(`## ${f.title}`, "", ...f.steps.map((s, i) => `${i + 1}. ${s.text}${pg(s.page)}`), ""); });
  if (p.glossary.length) out.push("## Key terms", "", ...p.glossary.map((g) => `- **${g.term}:** ${g.definition}${pg(g.page)}`), "");
  if (p.flashcards.length) {
    out.push("## Flashcards", "");
    p.flashcards.forEach((c, i) => out.push(`${i + 1}. **Q:** ${c.front}`, `   **A:** ${c.back}${pg(c.page)}`));
  }
  return out.join("\n");
}

function download(pack) {
  const blob = new Blob([toMarkdown(pack)], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${pack.filename.replace(/\.pdf$/i, "")} - study notes.md`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  toast("Study notes downloaded.");
}
