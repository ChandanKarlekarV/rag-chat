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
};

export const MODE_LABELS = { local: "Local · no key", ollama: "Ollama", api: "Claude API" };

/** Headers carrying the user's pasted key (never stored server-side). */
export function keyHeaders() {
  const k = getApiKey();
  return k ? { "X-LLM-Key": k } : {};
}

const listeners = {};
export const on = (name, fn) => { (listeners[name] ||= []).push(fn); };
export const emit = (name, data) => (listeners[name] || []).forEach((fn) => fn(data));

/* ---------- API key: per browser, session by default, "remember" uses localStorage ---------- */
const KEY = "rag-anthropic-key";
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

export function setApiKey(key, remember) {
  for (const kind of ["session", "local"]) { try { store(kind)?.removeItem(KEY); } catch (_) { /* ignore */ } }
  state.memoryKey = key || "";
  if (key) { try { store(remember ? "local" : "session")?.setItem(KEY, key); } catch (_) { /* memory only */ } }
}

export const readyDocs = () => state.docs.filter((d) => d.status === "ready");
