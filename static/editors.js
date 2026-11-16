"use strict";

// Form editors for arrays and objects, built from a JSON Schema.
//
// While a form is being filled in, every value lives in a "raw tree" kept in the draft: scalars are
// strings (booleans are "true"/"false", enum choices are JSON text), arrays are lists of raw values and
// objects are plain objects of raw values. The editors change that tree in place, and toValue() turns
// it into the real JSON value when a call is added. Requires h() from app.js.

const LONG_TEXT_KEY = /content|text|code|body|script|prompt|patch|diff|message|description/i;
const SCALAR_KINDS = ["string", "integer", "number", "boolean"];

function humanizeKey(key) {
  const spaced = String(key).replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/[_-]+/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function scalarEnum(schema) {
  const options = Array.isArray(schema.enum) ? schema.enum : [];
  return options.length && options.every((o) => ["string", "number", "boolean"].includes(typeof o)) ? options : [];
}

// What kind of editor a schema needs: string, integer, number, boolean, enum, array, object or json.
function kindOf(schema) {
  if (scalarEnum(schema).length) return "enum";
  const type = schema.type;
  if (type === "array") return "array";
  if (type === "object") return schema.properties && Object.keys(schema.properties).length ? "object" : "json";
  return SCALAR_KINDS.includes(type) ? type : "json";
}

function propTitle(schema, key) {
  return typeof schema.title === "string" && schema.title.trim() ? schema.title.trim() : humanizeKey(key);
}

// The raw tree for an actual JSON value.
function rawFrom(value, schema) {
  switch (kindOf(schema)) {
    case "string": return typeof value === "string" ? value : value === null || value === undefined ? "" : String(value);
    case "integer":
    case "number": return typeof value === "number" ? String(value) : "";
    case "boolean": return value === true ? "true" : "false";
    case "enum": return scalarEnum(schema).includes(value) ? JSON.stringify(value) : "";
    case "array": {
      const items = isPlainObject(schema.items) ? schema.items : {};
      return (Array.isArray(value) ? value : []).map((item) => rawFrom(item, items));
    }
    case "object": {
      const source = isPlainObject(value) ? value : {};
      const tree = {};
      for (const [key, sub] of Object.entries(schema.properties)) {
        tree[key] = source[key] === undefined ? freshRaw(sub) : rawFrom(source[key], sub);
      }
      return tree;
    }
    default: return value === undefined || value === null ? "" : JSON.stringify(value, null, 2);
  }
}

// A new, untouched value: the schema's default when it has one, otherwise empty.
function freshRaw(schema) {
  if (schema.default !== undefined && schema.default !== null) return rawFrom(schema.default, schema);
  switch (kindOf(schema)) {
    case "boolean": return "false";
    case "array": return [];
    case "object": {
      const tree = {};
      for (const [key, sub] of Object.entries(schema.properties)) tree[key] = freshRaw(sub);
      return tree;
    }
    default: return "";
  }
}

// An array row nobody has touched is left out rather than reported as incomplete. Booleans always count.
function isUntouched(raw, schema) {
  return kindOf(schema) !== "boolean" && JSON.stringify(raw) === JSON.stringify(freshRaw(schema));
}

function isOmittable(value) {
  return value === undefined || (Array.isArray(value) && value.length === 0) || (isPlainObject(value) && Object.keys(value).length === 0);
}

function checkRange(num, schema, label) {
  if (typeof schema.minimum === "number" && num < schema.minimum) throw new Error(`${label} must be at least ${schema.minimum}`);
  if (typeof schema.maximum === "number" && num > schema.maximum) throw new Error(`${label} must be at most ${schema.maximum}`);
}

// The real JSON value for a raw tree; undefined means "nothing entered". Errors name where the problem is.
function toValue(raw, schema, label) {
  switch (kindOf(schema)) {
    case "string": return raw === "" ? undefined : raw;
    case "integer":
    case "number": {
      if (String(raw).trim() === "") return undefined;
      const num = Number(raw);
      if (!Number.isFinite(num)) throw new Error(`${label} must be a number`);
      if (schema.type === "integer" && !Number.isInteger(num)) throw new Error(`${label} must be a whole number`);
      checkRange(num, schema, label);
      return num;
    }
    case "boolean": return raw === "true";
    case "enum": return raw === "" ? undefined : JSON.parse(raw);
    case "array": {
      const items = isPlainObject(schema.items) ? schema.items : {};
      const out = [];
      raw.forEach((item, index) => {
        if (isUntouched(item, items)) return;
        const place = `${label} → item ${index + 1}`;
        const value = toValue(item, items, place);
        if (value === undefined) throw new Error(`${place} is empty`);
        out.push(value);
      });
      return out;
    }
    case "object": {
      const required = new Set(Array.isArray(schema.required) ? schema.required : []);
      const out = {};
      for (const [key, sub] of Object.entries(schema.properties)) {
        const place = `${label} → ${propTitle(sub, key)}`;
        const value = toValue(raw[key], sub, place);
        if (isOmittable(value) && !required.has(key)) continue;
        if (value === undefined) {
          if (kindOf(sub) === "array") out[key] = [];
          else throw new Error(`${place} is required`);
          continue;
        }
        out[key] = value;
      }
      return out;
    }
    default: {
      if (String(raw).trim() === "") return undefined;
      try {
        return JSON.parse(raw);
      } catch {
        throw new Error(`${label} is not valid JSON`);
      }
    }
  }
}

// ------------------------------------------------------------------- DOM

function scalarEditor(schema, kind, hintKey, holder, slot, required) {
  const set = (value) => { holder[slot] = value; };
  const initial = holder[slot];
  switch (kind) {
    case "boolean": {
      const box = h("input", { type: "checkbox", checked: initial === "true" });
      box.addEventListener("change", () => set(box.checked ? "true" : "false"));
      return h("span", { class: "check-row" }, box, h("span", { class: "muted" }, "enabled"));
    }
    case "enum": {
      const select = h("select", {});
      if (!required) select.append(h("option", { value: "" }, "—"));
      for (const option of scalarEnum(schema)) select.append(h("option", { value: JSON.stringify(option) }, String(option)));
      select.value = initial;
      select.addEventListener("change", () => set(select.value));
      return select;
    }
    case "integer":
    case "number": {
      const input = h("input", {
        type: "number", step: kind === "integer" ? "1" : "any", value: initial,
        min: typeof schema.minimum === "number" ? schema.minimum : null,
        max: typeof schema.maximum === "number" ? schema.maximum : null,
      });
      input.addEventListener("input", () => set(input.value));
      return input;
    }
    case "json": {
      const area = h("textarea", { rows: 3, class: "mono", placeholder: "JSON", value: initial });
      area.addEventListener("input", () => set(area.value));
      return area;
    }
    default: {
      if (LONG_TEXT_KEY.test(hintKey) || schema.format === "textarea") {
        const area = h("textarea", { rows: 3, value: initial });
        area.addEventListener("input", () => set(area.value));
        return area;
      }
      const input = h("input", { type: "text", value: initial });
      input.addEventListener("input", () => set(input.value));
      return input;
    }
  }
}

// The editor for holder[slot], whatever its schema. Arrays and objects edit holder[slot] in place.
function buildEditor(schema, holder, slot, hintKey, required) {
  const kind = kindOf(schema);
  if (kind === "array") return arrayEditor(schema, holder[slot]);
  if (kind === "object") return objectEditor(schema, holder[slot]);
  return scalarEditor(schema, kind, hintKey, holder, slot, required);
}

function iconButton(label, title, onclick, disabled) {
  return h("button", { class: "btn small icon", type: "button", title, "aria-label": title, disabled: disabled || null, onclick }, label);
}

function arrayEditor(schema, list) {
  const items = isPlainObject(schema.items) ? schema.items : {};
  const itemKind = kindOf(items);
  const nested = itemKind === "array" || itemKind === "object";
  const rows = h("div", { class: "arr-rows" });
  const count = h("span", { class: "arr-count muted" });

  function move(index, by) {
    const target = index + by;
    if (target < 0 || target >= list.length) return;
    [list[index], list[target]] = [list[target], list[index]];
    render();
  }

  function row(index) {
    return h("div", { class: "arr-row" + (nested ? " nested" : "") },
      h("span", { class: "arr-index" }, String(index + 1)),
      h("div", { class: "arr-item" }, buildEditor(items, list, index, "", true)),
      h("span", { class: "arr-ctl" },
        iconButton("↑", "Move up", () => move(index, -1), index === 0),
        iconButton("↓", "Move down", () => move(index, 1), index === list.length - 1),
        iconButton("×", "Remove this item", () => { list.splice(index, 1); render(); })));
  }

  function render(focusIndex) {
    rows.replaceChildren(...list.map((_, index) => row(index)));
    count.textContent = list.length === 1 ? "1 item" : list.length ? `${list.length} items` : "No items yet";
    if (focusIndex !== undefined) {
      const control = rows.children[focusIndex]?.querySelector("input, textarea, select");
      if (control) control.focus();
    }
  }

  const add = h("button", {
    class: "btn small", type: "button",
    onclick: () => { list.push(freshRaw(items)); render(list.length - 1); },
  }, "+ Add item");
  render();
  return h("div", { class: "arr" }, rows, h("div", { class: "arr-foot" }, add, count));
}

function objectEditor(schema, tree) {
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const box = h("div", { class: "obj" });
  for (const [key, sub] of Object.entries(schema.properties)) {
    if (tree[key] === undefined) tree[key] = freshRaw(sub);
    const kind = kindOf(sub);
    const grouped = kind === "array" || kind === "object";
    const description = typeof sub.description === "string" ? sub.description.trim() : "";
    box.append(h(grouped ? "div" : "label", { class: "obj-field" },
      h("span", { class: "obj-label" },
        propTitle(sub, key),
        required.has(key) ? h("em", { class: "req" }, " *") : null,
        h("span", { class: "field-type" }, kind === "enum" ? "choice" : kind)),
      buildEditor(sub, tree, key, key, required.has(key)),
      description ? h("span", { class: "field-help" }, description.length > 200 ? description.slice(0, 199) + "…" : description) : null));
  }
  return box;
}

// Used when a cached form predates schemas being sent along.
function fallbackSchema(f) {
  if (f.widget === "object") return { type: "object", properties: {} };
  return { type: "array", items: f.item_type ? { type: f.item_type } : {} };
}