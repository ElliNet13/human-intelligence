"use strict";

const STATUS_LABELS = {
  pending: "Waiting for you",
  forwarding: "AI working…",
  forwarded: "Answered by AI",
  answered: "Answered by you",
  rejected: "Error returned",
  error: "Error",
  cancelled: "Client left",
};

const MESSAGE_COLLAPSE_LIMIT = { system: 0, tool: 1500, user: 6000, assistant: 6000 };

const state = {
  requests: [],
  selectedId: null,
  detail: null,
  controls: null,
  seen: null,
};

// Per-request unsent work, so switching between requests never loses anything.
const drafts = new Map();
// Handles to live parts of the composer, filled in by renderComposer().
const ui = {};
let audioCtx = null;

const $ = (selector) => document.querySelector(selector);

// ---------------------------------------------------------------- helpers

async function errorText(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

const api = {
  async get(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(await errorText(response));
    return response.json();
  },
  async post(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    if (!response.ok) throw new Error(await errorText(response));
    return response.json();
  },
};

// Builds DOM nodes. All dynamic text goes through text nodes, never innerHTML.
function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "list") el.setAttribute("list", value);
    else if (key in el) el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  appendAll(el, children);
  return el;
}

function appendAll(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

function truncate(text, limit) {
  return text.length <= limit ? text : text.slice(0, limit - 1) + "…";
}

function fmtTime(seconds) {
  return new Date(seconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

let toastTimer = null;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4500);
}

function beep() {
  if (!audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.25);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.26);
  } catch {
    // Audio is a nicety; ignore failures.
  }
}

function draftFor(id) {
  let draft = drafts.get(id);
  if (!draft) {
    draft = { text: "", staged: [], values: {}, open: new Set(), filter: "" };
    drafts.set(id, draft);
  }
  return draft;
}

// ----------------------------------------------------------- live updates
// One WebSocket carries everything the server pushes, so the page never polls.

let socket = null;
let retryMs = 500;
let retryTimer = null;

function setConnection(kind) {
  const el = $("#conn");
  el.className = `conn ${kind}`;
  el.textContent = kind === "live" ? "live" : kind === "down" ? "reconnecting…" : "connecting…";
  el.title = kind === "live" ? "Connected: updates arrive instantly" : "No connection to the server; retrying";
}

function connect() {
  clearTimeout(retryTimer);
  setConnection("connecting");
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${scheme}//${location.host}/ws`);
  socket = ws;
  ws.addEventListener("open", () => {
    retryMs = 500;
    setConnection("live");
  });
  ws.addEventListener("message", (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    handleEvent(msg);
  });
  ws.addEventListener("close", () => {
    if (socket !== ws) return;
    setConnection("down");
    retryTimer = setTimeout(connect, retryMs);
    retryMs = Math.min(retryMs * 2, 10000);
  });
  ws.addEventListener("error", () => {
    try { ws.close(); } catch { /* already closing */ }
  });
}

function handleEvent(msg) {
  switch (msg.type) {
    case "hello": return onHello(msg);
    case "request": return onRequest(msg.request);
    case "removed": return onRemoved(msg.ids);
    case "controls": return onControls(msg);
    case "settings": return applySettings(msg.settings);
  }
}

// Background tasks (titles, tags...) are answered by the AI on their own; they are not announced.
function announceNew(summaries) {
  let announce = false;
  for (const r of summaries) {
    if (!state.seen.has(r.id)) {
      state.seen.add(r.id);
      if (r.status === "pending" && !r.auto) announce = true;
    }
  }
  if (announce && $("#sound").checked) beep();
}

function clearSelection() {
  state.selectedId = null;
  state.detail = null;
  state.controls = null;
  renderDetail();
}

function autoSelect() {
  const first = heads()[0];
  if (first) select(first.id);
}

// When the chat you are looking at gets a newer turn, move to it. An unsent draft comes along.
function followChat(newer) {
  if (newer.auto || !state.selectedId || newer.id === state.selectedId) return false;
  const current = state.requests.find((r) => r.id === state.selectedId);
  if (!current || current.auto || current.conversation !== newer.conversation || newer.created <= current.created) return false;
  const draft = drafts.get(current.id);
  if (draft && current.status !== "answered" && current.status !== "forwarded") drafts.set(newer.id, draft);
  drafts.delete(current.id);
  select(newer.id);
  return true;
}

// Sent on every (re)connect: the full queue, so nothing missed while offline stays missed.
function onHello(msg) {
  state.requests = msg.requests;
  applySettings(msg.settings);
  if (state.seen === null) state.seen = new Set(msg.requests.map((r) => r.id));
  else announceNew(msg.requests);
  if (state.selectedId && !state.requests.some((r) => r.id === state.selectedId)) clearSelection();
  renderQueue();
  if (!state.selectedId) {
    autoSelect();
    return;
  }
  const current = state.requests.find((r) => r.id === state.selectedId);
  const head = current.auto ? null : heads().find((r) => r.conversation === current.conversation);
  if (head && followChat(head)) return;
  if (!state.detail || state.detail.status !== current.status) loadDetail(current.id);
  else if (current.status === "pending" && state.detail.tool_names.length) fetchControls(current.id);
}

function onRequest(summary) {
  const index = state.requests.findIndex((r) => r.id === summary.id);
  if (index === -1) state.requests.push(summary);
  else state.requests[index] = summary;
  announceNew([summary]);
  if (followChat(summary)) return;
  renderQueue();
  if (!state.selectedId) {
    autoSelect();
    return;
  }
  if (summary.id === state.selectedId && state.detail && state.detail.status !== summary.status) loadDetail(summary.id);
}

function onRemoved(ids) {
  const gone = new Set(ids);
  state.requests = state.requests.filter((r) => !gone.has(r.id));
  if (gone.has(state.selectedId)) clearSelection();
  renderQueue();
  if (!state.selectedId) autoSelect();
}

function onControls(controls) {
  if (controls.id !== state.selectedId) return;
  if (state.controls && controls.version < state.controls.version) return; // an older copy arrived late
  state.controls = controls;
  renderControls();
}

function applySettings(settings) {
  if (!settings) return;
  $("#auto-forward").checked = settings.auto_forward;
  $("#auto-utility").checked = settings.auto_utility;
}

// ------------------------------------------------------------------ queue

// One entry per chat: its newest request, except that a request still waiting for you is never hidden
// behind a newer finished one. Background tasks are listed apart.
function heads() {
  const newest = new Map();
  for (const r of state.requests) {
    if (r.auto) continue;
    const seen = newest.get(r.conversation);
    const waiting = r.status === "pending";
    const seenWaiting = seen && seen.status === "pending";
    if (!seen || (waiting && !seenWaiting) || (waiting === seenWaiting && r.created > seen.created)) newest.set(r.conversation, r);
  }
  const all = [...newest.values()];
  const pending = all.filter((r) => r.status === "pending").sort((a, b) => a.created - b.created);
  const rest = all.filter((r) => r.status !== "pending").sort((a, b) => b.created - a.created);
  return [...pending, ...rest];
}

function autoHandled() {
  return state.requests.filter((r) => r.auto).sort((a, b) => b.created - a.created);
}

function renderQueue() {
  const selected = state.requests.find((r) => r.id === state.selectedId);
  const list = $("#queue");
  list.replaceChildren();
  const items = heads();
  if (!items.length) {
    list.append(h("li", { class: "queue-empty" }, "No chats yet. Point a client at this server's /v1 endpoint and send a message."));
  }
  for (const r of items) {
    const meta = [
      r.model || "?",
      r.turn > 1 ? `turn ${r.turn}` : null,
      `${r.message_count} msg`,
      r.tool_count ? `${r.tool_count} tools` : null,
      r.stream ? "stream" : null,
    ].filter(Boolean).join(" · ");
    const active = selected && !selected.auto && selected.conversation === r.conversation;
    list.append(
      h("li", { class: "queue-item" + (active ? " active" : ""), onclick: () => select(r.id) },
        h("div", { class: "qi-top" },
          h("span", { class: `badge ${r.status}` }, STATUS_LABELS[r.status] ?? r.status),
          h("span", { class: "qi-time" }, fmtTime(r.created))),
        h("div", { class: "qi-meta" }, meta),
        h("div", { class: "qi-preview" }, r.preview))
    );
  }

  const background = autoHandled().slice(0, 30);
  $("#auto-box").hidden = background.length === 0;
  $("#auto-count").textContent = String(background.length);
  $("#auto-queue").replaceChildren(...background.map((r) =>
    h("li", { class: "auto-item" + (r.id === state.selectedId ? " active" : ""), onclick: () => select(r.id) },
      h("span", { class: `badge ${r.status}` }, STATUS_LABELS[r.status] ?? r.status),
      h("span", { class: "auto-kind" }, r.kind),
      h("span", { class: "qi-time" }, fmtTime(r.created)),
      h("div", { class: "qi-preview" }, r.preview))));

  const waiting = items.filter((r) => r.status === "pending").length;
  document.title = waiting ? `(${waiting}) Human LLM` : "Human LLM";
}

async function select(id) {
  state.selectedId = id;
  state.detail = null;
  state.controls = null;
  renderQueue();
  renderDetail();
  await loadDetail(id);
}

async function loadDetail(id, attempt = 0) {
  let detail;
  try {
    detail = await api.get(`/api/requests/${id}`);
  } catch (err) {
    if (state.selectedId === id) toast(err.message);
    return;
  }
  if (state.selectedId !== id) return;
  state.detail = detail;
  renderQueue();
  renderDetail();
  if (detail.status === "pending" && detail.tool_names.length) fetchControls(id);
  // A push may have announced a newer status while this response was in flight.
  const latest = state.requests.find((r) => r.id === id);
  if (latest && latest.status !== detail.status && attempt < 3) await loadDetail(id, attempt + 1);
}

// ----------------------------------------------------------------- detail

function renderDetail() {
  const root = $("#detail");
  root.replaceChildren();
  ui.controlsBox = null;
  ui.controlsStatus = null;
  ui.controlsBar = null;
  ui.cards = new Map();
  ui.renderStaged = null;
  const d = state.detail;
  if (!d) {
    root.append(h("div", { class: "empty" },
      state.selectedId ? "Loading…" : "Nothing selected. Requests show up on the left as clients send them."));
    return;
  }
  const pending = d.status === "pending";
  const conversation = h("section", { class: "col conv" },
    renderParams(d),
    d.messages.map(messageCard),
    renderOutcome(d));
  root.append(
    renderHeader(d),
    h("div", { class: "detail-body" + (pending ? " split" : "") },
      conversation,
      pending ? renderComposer(d) : null));
  conversation.scrollTop = conversation.scrollHeight;
}

function renderHeader(d) {
  const actions = d.status === "pending"
    ? h("div", { class: "head-actions" },
        h("button", { class: "btn primary", type: "button", onclick: (e) => forward(d, e.currentTarget) }, "Forward to AI"),
        h("button", { class: "btn danger", type: "button", onclick: (e) => reject(d, e.currentTarget) }, "Return error"))
    : null;
  return h("header", { class: "detail-head" },
    h("div", { class: "grow" },
      h("h2", {}, `Request ${d.id}`),
      h("div", { class: "sub" }, [
        d.model || "no model",
        fmtTime(d.created),
        d.auto ? `background task: ${d.kind}` : d.turn > 1 ? `turn ${d.turn} of this chat` : null,
        d.stream ? "streaming" : null,
      ].filter(Boolean).join(" · ")),
      d.note ? h("div", { class: "note" }, d.note) : null,
      renderGrouping(d)),
    h("span", { class: `badge ${d.status}` }, STATUS_LABELS[d.status] ?? d.status),
    actions);
}

// Shown for a request that started a chat of its own: why it did not match another one, and a way to merge by hand.
function renderGrouping(d) {
  if (d.auto) return null;
  const others = heads().filter((r) => r.conversation !== d.conversation);
  const idLine = d.chat_id ? h("div", { class: "muted small" }, `Chat id sent by the client: ${d.chat_id}`) : null;
  if (d.turn > 1 || !others.length) return idLine;

  const match = d.match;
  const select = h("select", {}, others.map((r) =>
    h("option", { value: r.conversation, selected: match ? match.conversation === r.conversation : false },
      `${fmtTime(r.created)} · ${truncate(r.preview, 70)}`)));
  const merge = h("button", {
    class: "btn small", type: "button",
    onclick: (e) => guarded(e.currentTarget, async () => {
      await api.post(`/api/requests/${d.id}/join`, { conversation: select.value });
      toast("Merged into the chat.");
      await loadDetail(d.id);
    }),
  }, "Merge");

  const parts = [h("summary", {}, "Why is this a separate chat?")];
  if (match) {
    parts.push(
      h("div", { class: "muted small" },
        `Closest other chat (${match.old_count} user/tool messages; this one has ${match.new_count}):`),
      h("table", { class: "cmp" },
        h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "Other chat"), h("th", {}, "This request"), h("th", {}, ""))),
        h("tbody", {}, match.rows.map((row) =>
          h("tr", { class: row.state },
            h("td", {}, row.index), h("td", {}, row.old), h("td", {}, row.new), h("td", {}, row.state))))));
  }
  parts.push(h("div", { class: "merge-row" }, h("span", {}, "Merge into:"), select, merge));
  return h("div", {}, idLine, h("details", { class: "grouping" }, parts));
}

function renderParams(d) {
  const entries = Object.entries(d.params).filter(([key]) => key !== "model" && key !== "stream");
  if (!entries.length) return null;
  return h("div", { class: "params" },
    entries.map(([key, value]) => {
      const full = JSON.stringify(value);
      return h("span", { class: "chip", title: full }, `${key}: ${truncate(full, 60)}`);
    }));
}

function callBlock(call) {
  return h("div", { class: "call" },
    h("code", { class: "call-name" }, `${call.name}()`),
    h("pre", { class: "call-args" }, call.arguments));
}

function longText(text, role) {
  const limit = MESSAGE_COLLAPSE_LIMIT[role] ?? 6000;
  if (text.length <= limit) return h("pre", { class: "msg-text" }, text);
  return h("details", { class: "msg-long" },
    h("summary", {}, `${truncate(text.replace(/\s+/g, " "), 140)}  (${text.length} chars)`),
    h("pre", { class: "msg-text" }, text));
}

function messageCard(m) {
  const label = m.role === "tool" ? "tool result" : m.role;
  return h("div", { class: `msg role-${m.role}` },
    h("div", { class: "msg-role" },
      label,
      m.name ? ` · ${m.name}` : null,
      m.tool_call_id ? h("span", { class: "muted" }, ` ${m.tool_call_id}`) : null),
    m.text ? longText(m.text, m.role) : null,
    m.tool_calls.map(callBlock));
}

function renderOutcome(d) {
  if (d.status === "pending") return null;
  const box = h("div", { class: `msg outcome ${d.status}` },
    h("div", { class: "msg-role" }, STATUS_LABELS[d.status] ?? d.status));
  if (d.status === "forwarding") box.append(h("div", { class: "muted" }, "Waiting for the upstream model…"));
  if (d.error) box.append(h("pre", { class: "msg-text err" }, d.error));
  if (d.answer) {
    if (d.answer.content) box.append(h("pre", { class: "msg-text" }, d.answer.content));
    for (const call of d.answer.tool_calls || []) box.append(callBlock(call));
  }
  return box;
}

// --------------------------------------------------------------- composer

function renderComposer(d) {
  const draft = draftFor(d.id);

  const text = h("textarea", {
    class: "reply-text",
    rows: 6,
    placeholder: "Your reply as the assistant…",
    value: draft.text,
    oninput: () => { draft.text = text.value; },
    onkeydown: (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        send(d, sendButton);
      }
    },
  });

  const staged = h("div", { class: "staged" });
  ui.renderStaged = () => {
    staged.replaceChildren();
    draft.staged.forEach((call, index) => {
      staged.append(
        h("div", { class: "staged-item" },
          h("div", { class: "what" },
            h("code", {}, `${call.name}`),
            " ",
            h("code", { class: "args" }, truncate(JSON.stringify(call.arguments), 160))),
          h("button", {
            class: "btn small", type: "button", title: "Remove this call",
            onclick: () => { draft.staged.splice(index, 1); ui.renderStaged(); },
          }, "×")));
    });
  };
  ui.renderStaged();

  const sendButton = h("button", { class: "btn primary", type: "button", onclick: (e) => send(d, e.currentTarget) }, "Send response");

  const parts = [
    h("h3", {}, "Respond yourself"),
    text,
    staged,
  ];

  if (d.tool_names.length) {
    ui.controlsStatus = h("span", { class: "controls-status" }, "Loading controls…");
    ui.controlsBox = h("div", { class: "controls" });
    ui.controlsBar = h("div", { class: "bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", hidden: true },
      h("div", { class: "fill" }));
    const filter = h("input", {
      type: "text",
      placeholder: "Filter tools…",
      value: draft.filter,
      oninput: () => { draft.filter = filter.value; renderControls(); },
    });
    parts.push(
      h("div", { class: "tools-head" },
        h("h3", { class: "grow" }, `Tools (${d.tool_names.length})`),
        ui.controlsStatus,
        h("button", { class: "btn small", type: "button", onclick: () => regenerateControls(d.id) }, "Re-read tools")),
      ui.controlsBar,
      d.tool_names.length > 6 ? filter : null,
      ui.controlsBox,
      customCallCard(d, draft));
  } else {
    parts.push(customCallCard(d, draft));
  }

  return h("section", { class: "col compose" },
    h("div", { class: "compose-scroll" }, parts),
    h("div", { class: "send-bar" },
      h("span", { class: "muted" }, "Ctrl+Enter sends"),
      sendButton));
}

function customCallCard(d, draft) {
  const name = h("input", { type: "text", placeholder: "tool name", list: "tool-names" });
  const datalist = h("datalist", { id: "tool-names" }, d.tool_names.map((n) => h("option", { value: n })));
  const args = h("textarea", { class: "mono", rows: 3, value: "{}" });
  const err = h("div", { class: "tool-err" });
  return h("div", { class: "custom" },
    h("strong", {}, "Custom tool call"),
    h("span", { class: "muted" }, "Pick any tool name and write the arguments as JSON."),
    name, datalist, args, err,
    h("div", {}, h("button", {
      class: "btn small", type: "button",
      onclick: () => {
        err.textContent = "";
        const toolName = name.value.trim();
        if (!toolName) { err.textContent = "Enter a tool name."; return; }
        let parsed;
        try { parsed = JSON.parse(args.value || "{}"); } catch { err.textContent = "Arguments are not valid JSON."; return; }
        if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
          err.textContent = "Arguments must be a JSON object.";
          return;
        }
        draft.staged.push({ name: toolName, arguments: parsed });
        ui.renderStaged();
      },
    }, "Add call")));
}

// ---------------------------------------------------------- tool controls

function controlsStatusText(c) {
  if (c.state === "working") {
    const progress = c.total ? ` ${c.done} of ${c.total}` : "";
    return { text: `Reading tools…${progress} (basic controls shown meanwhile)`, cls: "working" };
  }
  if (c.error) return { text: `AI could not help (${c.error}). Using controls built from the schemas.`, cls: "problem" };
  return { text: "Controls ready", cls: "" };
}

// One fetch when a request is opened; after that the server pushes every change.
async function fetchControls(id) {
  let controls;
  try {
    controls = await api.get(`/api/requests/${id}/controls`);
  } catch {
    return;
  }
  onControls(controls);
}

async function regenerateControls(id) {
  try {
    await api.post(`/api/requests/${id}/controls/regenerate`);
  } catch (err) {
    toast(err.message);
  }
}

function renderControls() {
  const box = ui.controlsBox;
  const c = state.controls;
  if (!box || !c || !state.detail) return;
  const draft = draftFor(state.detail.id);
  const status = controlsStatusText(c);
  ui.controlsStatus.textContent = status.text;
  ui.controlsStatus.className = `controls-status ${status.cls}`;
  const working = c.state === "working";
  if (ui.controlsBar) {
    const percent = c.total ? Math.min(100, Math.round((100 * c.done) / c.total)) : 0;
    ui.controlsBar.hidden = !working;
    ui.controlsBar.firstElementChild.style.width = `${percent}%`;
    ui.controlsBar.setAttribute("aria-valuenow", String(percent));
  }
  const needle = draft.filter.trim().toLowerCase();
  const visible = c.controls.filter((spec) =>
    !needle || `${spec.name} ${spec.title} ${spec.summary}`.toLowerCase().includes(needle));
  // Cards that did not change keep their DOM, so a form being typed into is not rebuilt by each progress push.
  const wanted = visible.map((spec) => {
    const tag = working ? (spec.source === "ai" ? "read" : "waiting") : "";
    const signature = JSON.stringify([spec, tag]);
    const cached = ui.cards.get(spec.name);
    if (cached && cached.signature === signature) return cached.node;
    const node = toolCard(spec, draft, tag);
    ui.cards.set(spec.name, { signature, node });
    return node;
  });
  const keep = new Set(wanted);
  for (const child of [...box.children]) if (!keep.has(child)) child.remove();
  wanted.forEach((node, index) => {
    if (box.children[index] !== node) box.insertBefore(node, box.children[index] || null);
  });
}

function toolCard(spec, draft, tag) {
  const err = h("div", { class: "tool-err" });
  const details = h("details", {
    class: "tool",
    open: draft.open.has(spec.name),
    ontoggle: () => {
      if (details.open) draft.open.add(spec.name);
      else draft.open.delete(spec.name);
    },
  },
    h("summary", {},
      h("span", { class: "tool-title" }, spec.title),
      h("code", { class: "tool-name" }, spec.name),
      tag ? h("span", { class: `tool-tag ${tag}` }, tag === "read" ? "read" : "waiting…") : null,
      spec.summary ? h("span", { class: "tool-summary" }, spec.summary) : null),
    h("div", { class: "fields" },
      spec.fields.length ? spec.fields.map((f) => fieldRow(spec, f, draft)) : h("div", { class: "muted" }, "No parameters.")),
    err,
    h("div", { class: "tool-actions" },
      h("button", { class: "btn small primary", type: "button", onclick: () => addCall(spec, draft, err) }, "Add call")));
  return details;
}

const TEXT_WIDGETS = ["text", "textarea", "password", "combo"];

function defaultText(f) {
  const value = f.default;
  if (value === null || value === undefined) return "";
  switch (f.widget) {
    case "select":
    case "radio":
    case "multiselect":
    case "json":
      return JSON.stringify(value);
    case "list":
      return Array.isArray(value) ? value.join("\n") : "";
    case "checkbox":
      return value ? "true" : "false";
    default:
      return String(value);
  }
}

function isEmptyValue(f, raw) {
  if (raw === "") return true;
  if (f.widget === "multiselect") return raw === "[]";
  if (f.widget === "list") return raw.trim() === "";
  return false;
}

// Each widget keeps its current value as a string in values[key]; collectArgs() turns it into JSON.
function fieldInput(f, initial, setValue, idBase) {
  switch (f.widget) {
    case "checkbox": {
      const box = h("input", { type: "checkbox", checked: initial === "true" });
      box.addEventListener("change", () => setValue(box.checked ? "true" : "false"));
      return h("span", { class: "check-row" }, box, h("span", { class: "muted" }, "enabled"));
    }

    case "radio": {
      const group = h("div", { class: "options" });
      const choices = f.required ? f.options : [null, ...f.options];
      for (const option of choices) {
        const value = option === null ? "" : JSON.stringify(option);
        const radio = h("input", { type: "radio", name: idBase, value, checked: initial === value });
        radio.addEventListener("change", () => setValue(value));
        group.append(h("label", { class: "opt" }, radio, option === null ? "none" : String(option)));
      }
      return group;
    }

    case "multiselect": {
      let selected = [];
      try { selected = JSON.parse(initial || "[]"); } catch { selected = []; }
      const group = h("div", { class: "options" });
      const boxes = f.options.map((option) => {
        const box = h("input", { type: "checkbox", checked: selected.includes(option) });
        box.addEventListener("change", () => {
          setValue(JSON.stringify(f.options.filter((_, i) => boxes[i].checked)));
        });
        group.append(h("label", { class: "opt" }, box, String(option)));
        return box;
      });
      return group;
    }

    case "slider": {
      const range = h("input", { type: "range", min: f.min, max: f.max, step: f.step ?? (f.type === "integer" ? 1 : "any") });
      if (initial !== "") range.value = initial;
      const readout = h("span", { class: "slider-val" }, initial === "" ? "unset" : initial);
      range.addEventListener("input", () => { setValue(range.value); readout.textContent = range.value; });
      const clear = h("button", {
        class: "btn small", type: "button", title: "Leave this unset",
        onclick: () => { setValue(""); readout.textContent = "unset"; },
      }, "×");
      return h("span", { class: "slider-row" }, h("span", { class: "muted" }, f.min), range, h("span", { class: "muted" }, f.max), readout, clear);
    }

    case "select": {
      const select = h("select", {});
      if (!f.required) select.append(h("option", { value: "" }, "—"));
      const options = f.options;
      for (const option of options) select.append(h("option", { value: JSON.stringify(option) }, String(option)));
      select.value = initial;
      select.addEventListener("change", () => setValue(select.value));
      return select;
    }

    case "textarea":
    case "json":
    case "list": {
      const placeholder = f.widget === "list" ? "One item per line" : f.widget === "json" ? "JSON" : "";
      const area = h("textarea", {
        rows: f.widget === "json" ? 4 : 3,
        class: f.widget === "json" ? "mono" : "",
        placeholder,
        value: initial,
      });
      area.addEventListener("input", () => setValue(area.value));
      return area;
    }

    case "number": {
      const input = h("input", { type: "number", step: f.type === "integer" ? "1" : "any", value: initial });
      input.addEventListener("input", () => setValue(input.value));
      return input;
    }

    case "combo": {
      const listId = `dl-${idBase}`;
      const input = h("input", { type: "text", list: listId, value: initial });
      input.addEventListener("input", () => setValue(input.value));
      return h("span", { class: "combo" }, input, h("datalist", { id: listId }, f.options.map((o) => h("option", { value: o }))));
    }

    default: {
      const input = h("input", { type: f.widget === "password" ? "password" : "text", value: initial });
      input.addEventListener("input", () => setValue(input.value));
      return input;
    }
  }
}

function fieldRow(spec, f, draft) {
  const values = (draft.values[spec.name] ??= {});
  const nested = f.widget === "array" || f.widget === "object";
  // A value kept from before the control type changed (say list -> array) no longer fits; start over.
  if (values[f.key] !== undefined && (typeof values[f.key] === "string") === nested) delete values[f.key];
  const idBase = `f-${spec.name}-${f.key}`.replace(/[^A-Za-z0-9_-]/g, "_");
  let input;
  let kind = f.type === "array" ? `${f.item_type || "value"}[]` : f.type;
  if (nested) {
    const schema = f.schema || fallbackSchema(f);
    if (values[f.key] === undefined) {
      values[f.key] = f.default !== null && f.default !== undefined ? rawFrom(f.default, schema) : freshRaw(schema);
    }
    input = buildEditor(schema, values, f.key, f.key, f.required);
    kind = f.widget === "object" ? "object" : `${kindOf(isPlainObject(schema.items) ? schema.items : {})}[]`;
  } else {
    const initial = values[f.key] !== undefined ? values[f.key] : defaultText(f);
    input = fieldInput(f, initial, (value) => { values[f.key] = value; }, idBase);
  }

  // Radio, multiselect and nested editors hold several inputs, so they use a div rather than a label.
  const wrapper = nested || f.widget === "radio" || f.widget === "multiselect" ? "div" : "label";
  return h(wrapper, { class: "field" },
    h("span", { class: "field-label" },
      f.label,
      f.required ? h("em", { class: "req" }, " *") : null,
      h("span", { class: "field-type" }, kind)),
    input,
    f.help ? h("span", { class: "field-help" }, f.help) : null);
}

function toItem(text, itemType) {
  if (itemType === "integer" || itemType === "number") {
    const num = Number(text);
    if (!Number.isFinite(num) || (itemType === "integer" && !Number.isInteger(num))) {
      throw new Error(`"${text}" is not a valid ${itemType}`);
    }
    return num;
  }
  return text;
}

function collectArgs(spec, draft) {
  const values = draft.values[spec.name] || {};
  const args = {};
  const problems = [];
  for (const f of spec.fields) {
    if (f.widget === "array" || f.widget === "object") {
      try {
        const schema = f.schema || fallbackSchema(f);
        const tree = values[f.key] !== undefined && typeof values[f.key] !== "string"
          ? values[f.key]
          : f.default !== null && f.default !== undefined ? rawFrom(f.default, schema) : freshRaw(schema);
        const value = toValue(tree, schema, f.label);
        if (isOmittable(value) && (!f.required || value === undefined)) {
          if (f.required) problems.push(`${f.label} is required`);
          continue;
        }
        args[f.key] = value;
      } catch (err) {
        problems.push(err.message);
      }
      continue;
    }
    const raw = values[f.key] !== undefined ? values[f.key] : defaultText(f);
    if (isEmptyValue(f, raw)) {
      if (!f.required) continue;
      if (f.widget === "checkbox") args[f.key] = false;
      else if (f.widget === "list" || f.widget === "multiselect") args[f.key] = [];
      else problems.push(`${f.label} is required`);
      continue;
    }
    try {
      if (TEXT_WIDGETS.includes(f.widget)) {
        args[f.key] = raw;
      } else if (f.widget === "number" || f.widget === "slider") {
        const num = Number(raw);
        if (!Number.isFinite(num)) throw new Error(`${f.label} must be a number`);
        if (f.type === "integer" && !Number.isInteger(num)) throw new Error(`${f.label} must be a whole number`);
        args[f.key] = num;
      } else if (f.widget === "checkbox") {
        args[f.key] = raw === "true";
      } else if (f.widget === "list") {
        const lines = raw.split("\n").map((line) => line.trim()).filter(Boolean);
        args[f.key] = lines.map((line) => toItem(line, f.item_type));
      } else {
        args[f.key] = JSON.parse(raw);
      }
    } catch (err) {
      problems.push(err instanceof SyntaxError ? `${f.label} is not valid JSON` : `${f.label}: ${err.message}`);
    }
  }
  return { args, problems };
}

function addCall(spec, draft, errBox) {
  const { args, problems } = collectArgs(spec, draft);
  errBox.textContent = problems.join(" · ");
  if (problems.length) return;
  draft.staged.push({ name: spec.name, arguments: args });
  ui.renderStaged();
}

// ---------------------------------------------------------------- actions

async function guarded(button, task) {
  if (button) button.disabled = true;
  try {
    await task();
  } catch (err) {
    toast(err.message);
  } finally {
    if (button) button.disabled = false;
  }
}

async function afterDecision(id) {
  if (state.selectedId === id) await loadDetail(id);
}

function forward(d, button) {
  return guarded(button, async () => {
    await api.post(`/api/requests/${d.id}/forward`);
    await afterDecision(d.id);
  });
}

function reject(d, button) {
  const message = window.prompt("Error message to return to the client:", "The operator declined this request.");
  if (message === null) return Promise.resolve();
  return guarded(button, async () => {
    await api.post(`/api/requests/${d.id}/reject`, { message });
    await afterDecision(d.id);
  });
}

function send(d, button) {
  const draft = draftFor(d.id);
  if (!draft.text.trim() && !draft.staged.length) {
    toast("Write a message or add a tool call first.");
    return Promise.resolve();
  }
  return guarded(button, async () => {
    await api.post(`/api/requests/${d.id}/respond`, { content: draft.text, tool_calls: draft.staged });
    drafts.delete(d.id);
    await afterDecision(d.id);
  });
}

// --------------------------------------------------------------- settings

async function loadSettings() {
  try {
    const settings = await api.get("/api/settings");
    applySettings(settings);
    return settings;
  } catch {
    return null;
  }
}

async function openSettings() {
  const settings = await loadSettings();
  if (!settings) { toast("Could not load settings."); return; }
  $("#set-ollama-url").textContent = settings.ollama_url || "not set - add OLLAMA_URL to .env";
  $("#set-forward-model").value = settings.forward_model;
  $("#set-forward-model").placeholder = settings.default_model;
  $("#set-controls-model").value = settings.controls_model;
  $("#settings-dialog").showModal();
}

async function saveSettings(event) {
  event.preventDefault();
  const body = {
    forward_model: $("#set-forward-model").value,
    controls_model: $("#set-controls-model").value,
  };
  try {
    await api.post("/api/settings", body);
    $("#settings-dialog").close();
    toast("Settings saved.");
  } catch (err) {
    toast(err.message);
  }
}

// ------------------------------------------------------------------- init

function init() {
  $("#settings-btn").addEventListener("click", openSettings);
  $("#settings-cancel").addEventListener("click", () => $("#settings-dialog").close());
  $("#settings-form").addEventListener("submit", saveSettings);

  for (const [selector, key] of [["#auto-forward", "auto_forward"], ["#auto-utility", "auto_utility"]]) {
    $(selector).addEventListener("change", async (e) => {
      try {
        await api.post("/api/settings", { [key]: e.target.checked });
      } catch (err) {
        toast(err.message);
        e.target.checked = !e.target.checked;
      }
    });
  }

  $("#sound").addEventListener("change", (e) => {
    if (e.target.checked && !audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (e.target.checked) beep();
  });

  $("#clear-btn").addEventListener("click", async () => {
    try {
      await api.post("/api/requests/clear");
    } catch (err) {
      toast(err.message);
    }
  });

  document.addEventListener("visibilitychange", () => {
    // A sleeping laptop can leave a dead socket behind; reconnect as soon as the tab is back.
    const alive = socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING);
    if (!document.hidden && !alive) connect();
  });

  renderDetail();
  connect();
}

init();