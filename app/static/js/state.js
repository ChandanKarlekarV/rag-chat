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
  llm: null, // last /llm/check result
  memoryKey: "",
};

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
