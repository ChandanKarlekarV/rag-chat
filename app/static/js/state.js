// Single shared state object and tiny pub/sub so modules stay decoupled.

export const state = {
  user: null,
  docs: [],
  selected: new Set(), // doc ids ticked "include in questions"
  conversations: [],
  conversationId: null,
  busy: false,
  controller: null,
  debug: false,
  modes: null, // last GET /chat/modes result
  mode: null, // the mode used for the next question: "local" | "ollama" | "api"
  memoryKey: "",
  memoryMeta: {},
  view: "chat", // "chat" | "study"
};

export const MODE_LABELS = { local: "Local · no key", ollama: "Ollama", api: "API key" };

/** Headers carrying the user's pasted key, its provider and model (never stored server-side). */
export function keyHeaders() {
  const k = getApiKey();
  if (!k) return {};
  const { provider, model } = getKeyMeta();
  const h = { "X-LLM-Key": k };
  if (provider) h["X-LLM-Provider"] = provider;
  if (model) h["X-LLM-Model"] = model;
  return h;
}

const listeners = {};
export const on = (name, fn) => { (listeners[name] ||= []).push(fn); };
export const emit = (name, data) => (listeners[name] || []).forEach((fn) => fn(data));

/* ---------- API key: per browser, session by default, "remember" uses localStorage ---------- */
const KEY = "rag-anthropic-key"; // name kept so keys saved by earlier versions still work
const META = "rag-llm-key-meta";
const store = (kind) => { try { return kind === "local" ? localStorage : sessionStorage; } catch (_) { return null; } };

export function getApiKey() {
  for (const kind of ["session", "local"]) {
    try { const v = store(kind)?.getItem(KEY); if (v) return v; } catch (_) { /* blocked */ }
  }
  return state.memoryKey;
}

export function isKeyRemembered() {
  try { return Boolean(store("local")?.getItem(KEY)); } catch (_) { return false; }
}

/** Provider id ("auto" or e.g. "openai") and model chosen for the saved key. */
export function getKeyMeta() {
  for (const kind of ["session", "local"]) {
    try { const v = store(kind)?.getItem(META); if (v) return JSON.parse(v); } catch (_) { /* blocked or bad JSON */ }
  }
  return state.memoryMeta;
}

export function setApiKey(key, remember, meta = {}) {
  for (const kind of ["session", "local"]) {
    try { store(kind)?.removeItem(KEY); store(kind)?.removeItem(META); } catch (_) { /* ignore */ }
  }
  state.memoryKey = key || "";
  state.memoryMeta = key ? { provider: meta.provider || "", model: meta.model || "" } : {};
  if (key) {
    try {
      const s = store(remember ? "local" : "session");
      s?.setItem(KEY, key);
      s?.setItem(META, JSON.stringify(state.memoryMeta));
    } catch (_) { /* memory only */ }
  }
}

export const readyDocs = () => state.docs.filter((d) => d.status === "ready");
