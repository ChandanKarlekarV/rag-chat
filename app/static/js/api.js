// Thin fetch wrapper: CSRF header on writes, friendly errors, SSE parsing for /chat/ask.

export class ApiError extends Error {
  constructor(code, message, status) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

const FRIENDLY = {
  0: "You appear to be offline. Check your connection and try again.",
  413: "That file is too large. The limit is 100 MB.",
  500: "Something went wrong on our side. Please try again.",
};

async function toError(res) {
  try {
    const body = await res.json();
    if (body && body.error) return new ApiError(body.error.code, body.error.message, res.status);
  } catch (_) { /* not JSON */ }
  return new ApiError("http_error", FRIENDLY[res.status] || "Something went wrong. Please try again.", res.status);
}

/** JSON request. Throws ApiError with a user-friendly message. */
export async function request(method, path, { json, headers = {}, signal } = {}) {
  const init = { method, headers: { ...headers }, credentials: "same-origin", signal };
  if (method !== "GET") init.headers["X-Requested-With"] = "fetch";
  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("offline", FRIENDLY[0], 0);
  }
  if (!res.ok) throw await toError(res);
  if (res.status === 204) return null;
  return res.json();
}

/** Upload with progress (XHR, because fetch has no upload progress events). */
export function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", "/documents");
    xhr.setRequestHeader("X-Requested-With", "fetch");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100)); };
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (_) { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      const err = body && body.error
        ? new ApiError(body.error.code, body.error.message, xhr.status)
        : new ApiError("http_error", FRIENDLY[xhr.status] || "The upload failed. Please try again.", xhr.status);
      reject(err);
    };
    xhr.onerror = () => reject(new ApiError("offline", FRIENDLY[0], 0));
    xhr.send(form);
  });
}

/** POST /chat/ask and dispatch SSE events to `on[eventName](data)`. */
export async function askStream(body, { keyHeaders = {}, signal, on }) {
  const headers = { "Content-Type": "application/json", Accept: "text/event-stream", "X-Requested-With": "fetch", ...keyHeaders };
  let res;
  try {
    res = await fetch("/chat/ask", { method: "POST", headers, body: JSON.stringify(body), signal, credentials: "same-origin" });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("offline", FRIENDLY[0], 0);
  }
  if (!res.ok || !res.body) throw await toError(res);
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
      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data && on[event]) on[event](JSON.parse(data));
    }
  }
}
