"use strict";

const POLL_MS = 1000;

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
let controlsEpoch = 0;
let audioCtx = null;

const $ = (selector) => document.querySelector(selector);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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

// ------------------------------------------------------------------ queue

function sortedRequests() {
  const pending = state.requests.filter((r) => r.status === "pending").sort((a, b) => a.created - b.created);
  const rest = state.requests.filter((r) => r.status !== "pending").sort((a, b) => b.created - a.created);
  return [...pending, ...rest];
}

function renderQueue() {
  const list = $("#queue");
  list.replaceChildren();
  const items = sortedRequests();
  if (!items.length) {
    list.append(h("li", { class: "queue-empty" }, "No requests yet. Point a client at this server's /v1 endpoint and send one."));
  }
  for (const r of items) {
    const meta = [
      r.model || "?",
      `${r.message_count} msg`,
      r.tool_count ? `${r.tool_count} tools` : null,
      r.stream ? "stream" : null,
    ].filter(Boolean).join(" · ");
    list.append(
      h("li", { class: "queue-item" + (r.id === state.selectedId ? " active" : ""), onclick: () => select(r.id) },
        h("div", { class: "qi-top" },
          h("span", { class: `badge ${r.status}` }, STATUS_LABELS[r.status] ?? r.status),
          h("span", { class: "qi-time" }, fmtTime(r.created))),
        h("div", { class: "qi-meta" }, meta),
        h("div", { class: "qi-preview" }, r.preview))
    );
  }
  const waiting = state.requests.filter((r) => r.status === "pending").length;
  document.title = waiting ? `(${waiting}) Human LLM` : "Human LLM";
}

async function refreshList() {
  let list;
  try {
    list = await api.get("/api/requests");
  } catch {
    return;
  }
  state.requests = list;

  if (state.seen === null) {
    state.seen = new Set(list.map((r) => r.id));
  } else {
    let announce = false;
    for (const r of list) {
      if (!state.seen.has(r.id)) {
        state.seen.add(r.id);
        if (r.status === "pending") announce = true;
      }
    }
    if (announce && $("#sound").checked) beep();
  }

  if (state.selectedId && !list.some((r) => r.id === state.selectedId)) {
    state.selectedId = null;
    state.detail = null;
    renderDetail();
  }
  if (!state.selectedId && list.length) {
    await select(sortedRequests()[0].id);
    return;
  }
  renderQueue();

  const current = list.find((r) => r.id === state.selectedId);
  if (current && state.detail && state.detail.id === current.id && state.detail.status !== current.status) {
    await loadDetail(current.id);
  }
}

async function select(id) {
  state.selectedId = id;
  state.detail = null;
  state.controls = null;
  renderQueue();
  renderDetail();
  await loadDetail(id);
}

async function loadDetail(id) {
  let detail;
  try {
    detail = await api.get(`/api/requests/${id}`);
  } catch (err) {
    toast(err.message);
    return;
  }
  if (state.selectedId !== id) return;
  state.detail = detail;
  renderQueue();
  renderDetail();
  if (detail.status === "pending" && detail.tool_names.length) loadControls(id);
}

// ----------------------------------------------------------------- detail

function renderDetail() {
  const root = $("#detail");
  root.replaceChildren();
  ui.controlsBox = null;
  ui.controlsStatus = null;
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
      h("div", { class: "sub" }, `${d.model || "no model"} · ${fmtTime(d.created)}${d.stream ? " · streaming" : ""}`)),
    h("span", { class: `badge ${d.status}` }, STATUS_LABELS[d.status] ?? d.status),
    actions);
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
        h("button", { class: "btn small", type: "button", onclick: () => loadControls(d.id, true) }, "Re-read tools")),
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
  if (c.state === "working") return { text: "AI is reading the tools… basic controls shown meanwhile", cls: "working" };
  if (c.error) return { text: `AI could not help (${c.error}). Using controls built from the schemas.`, cls: "problem" };
  return { text: "Controls ready", cls: "" };
}

async function loadControls(id, regenerate = false) {
  const epoch = ++controlsEpoch;
  if (regenerate) {
    try {
      await api.post(`/api/requests/${id}/controls/regenerate`);
    } catch (err) {
      toast(err.message);
      return;
    }
  }
  let failures = 0;
  for (;;) {
    let controls;
    try {
      controls = await api.get(`/api/requests/${id}/controls`);
      failures = 0;
    } catch {
      if (++failures >= 3) return;
      await sleep(1500);
      continue;
    }
    if (epoch !== controlsEpoch || state.selectedId !== id) return;
    state.controls = controls;
    renderControls();
    if (controls.state === "ready") return;
    await sleep(700);
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
  const needle = draft.filter.trim().toLowerCase();
  const visible = c.controls.filter((spec) =>
    !needle || `${spec.name} ${spec.title} ${spec.summary}`.toLowerCase().includes(needle));
  box.replaceChildren(...visible.map((spec) => toolCard(spec, draft)));
}

function toolCard(spec, draft) {
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
  const initial = values[f.key] !== undefined ? values[f.key] : defaultText(f);
  const idBase = `f-${spec.name}-${f.key}`.replace(/[^A-Za-z0-9_-]/g, "_");
  const input = fieldInput(f, initial, (value) => { values[f.key] = value; }, idBase);
  const kind = f.type === "array" ? `${f.item_type || "value"}[]` : f.type;

  // Radio and multiselect groups hold several inputs, so they use a div rather than a label.
  const wrapper = f.widget === "radio" || f.widget === "multiselect" ? "div" : "label";
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
  await refreshList();
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
    $("#auto-forward").checked = settings.auto_forward;
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

  $("#auto-forward").addEventListener("change", async (e) => {
    try {
      await api.post("/api/settings", { auto_forward: e.target.checked });
    } catch (err) {
      toast(err.message);
      e.target.checked = !e.target.checked;
    }
  });

  $("#sound").addEventListener("change", (e) => {
    if (e.target.checked && !audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (e.target.checked) beep();
  });

  $("#clear-btn").addEventListener("click", async () => {
    try {
      await api.post("/api/requests/clear");
      await refreshList();
    } catch (err) {
      toast(err.message);
    }
  });

  renderDetail();
  loadSettings();
  refreshList();
  setInterval(refreshList, POLL_MS);
}

init();