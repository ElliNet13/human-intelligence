"""Human-in-the-loop, OpenAI-compatible chat endpoint.

Clients talk to POST /v1/chat/completions like they would to any OpenAI-style
server. Each request lands in a queue shown in the web UI (served at /), where
the operator either forwards it to a real upstream model or answers it by hand.
When answering by hand, an upstream model reads the request's tool definitions
once, caches a form-control description per tool, and the UI renders those
controls so tool calls can be built without reading the raw schemas.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import hashlib
import json
import logging
import os
import re
import secrets
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeGuard, cast
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("human-llm")

BASE_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, '#' comments, real environment variables win."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


load_env_file(BASE_DIR / ".env")

STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
SETTINGS_FILE = DATA_DIR / "settings.json"
CACHE_FILE = DATA_DIR / "controls_cache.json"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8787"))
API_KEY = os.environ.get("API_KEY", "")
# Extra browser origins allowed to open the operator WebSocket (comma separated), e.g. https://llm.example.com
ALLOWED_ORIGINS = {o.strip().lower() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "").strip()
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "").strip() or "mimo_v2.6:latest"

MAX_HISTORY = 100
KEEPALIVE_SECONDS = 15.0
HUB_QUEUE_LIMIT = 500
AUTO_HISTORY = 30  # finished background-task requests kept for inspection; separate from MAX_HISTORY
CHAT_MEMORY_LIMIT = 300
# Finished background tasks remove themselves from the list after this long (errors stay 10x longer); 0 keeps them.
AUTO_CLEAR_SECONDS = float(os.environ.get("AUTO_CLEAR_SECONDS", "60"))
UTILITY_SCAN_CHARS = 8000
UTILITY_PATTERNS_FILE = Path(os.environ.get("UTILITY_PATTERNS_FILE", str(BASE_DIR / "utility_patterns.txt")))
CHAT_ID_HEADERS = ("x-openwebui-chat-id", "x-chat-id", "x-conversation-id", "x-thread-id")
CHAT_ID_FIELDS = ("chat_id", "conversation_id", "thread_id")
CONTROLS_BATCH = 6
MAX_SCHEMA_DEPTH = 6  # how deep arrays and objects nest before a value falls back to a JSON box
SPEC_VERSION = "2"  # bump when the shape of a control spec changes, so cached specs are rebuilt
UPSTREAM_TIMEOUT = httpx.Timeout(600.0, connect=15.0)

CONTROLS_PROMPT = (BASE_DIR / "controls_prompt.txt").read_text(encoding="utf-8")
PROMPT_SALT = hashlib.sha256(CONTROLS_PROMPT.encode("utf-8")).hexdigest()[:8]

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Max-Age": "86400",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return cast(list[Any], value)
    return []


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def error_body(message: str, kind: str = "human_operator_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(error_body(message), status_code=status)


def pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except ValueError:
        return text


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in as_list(content):
        part = as_dict(item)
        kind = part.get("type")
        if kind == "text":
            parts.append(str(part.get("text", "")))
        elif kind == "image_url":
            parts.append("[image]")
        elif part:
            parts.append(f"[{kind or 'part'}]")
    return "\n".join(parts)


def humanize(key: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", key)
    spaced = spaced.replace("_", " ").replace("-", " ").strip()
    return spaced[:1].upper() + spaced[1:] if spaced else key


def short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Settings and the persisted control cache
# --------------------------------------------------------------------------


@dataclass
class Settings:
    forward_model: str = DEFAULT_MODEL  # model used when forwarding; blank keeps the client's model name
    controls_model: str = ""  # model that reads tools; blank means forward_model
    auto_forward: bool = False
    auto_utility: bool = True  # answer the client's titles/tags/follow-ups with the AI, without asking


def load_settings() -> Settings:
    settings = Settings()
    try:
        raw = as_dict(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return settings
    for name in ("forward_model", "controls_model"):
        value = raw.get(name)
        if isinstance(value, str):
            setattr(settings, name, value)
    for flag in ("auto_forward", "auto_utility"):
        value = raw.get(flag)
        if isinstance(value, bool):
            setattr(settings, flag, value)
    return settings


def save_settings() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(asdict(SETTINGS), indent=2), encoding="utf-8")
    try:
        os.chmod(SETTINGS_FILE, 0o600)
    except OSError:
        pass


def load_cache() -> dict[str, dict[str, Any]]:
    try:
        raw = as_dict(json.loads(CACHE_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}
    return {key: as_dict(value) for key, value in raw.items() if isinstance(value, dict)}


def save_cache() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(CONTROLS_CACHE, ensure_ascii=False, indent=1), encoding="utf-8")


SETTINGS = load_settings()
CONTROLS_CACHE: dict[str, dict[str, Any]] = load_cache()
CONTROLS_LOCK = asyncio.Lock()


# --------------------------------------------------------------------------
# Upstream access
# --------------------------------------------------------------------------


def upstream_url(path: str) -> str:
    """Ollama's OpenAI-compatible API lives under /v1 on the server given in .env."""
    base = OLLAMA_URL.rstrip("/")
    if not base:
        raise RuntimeError("OLLAMA_URL is not set. Add it to the .env file next to server.py.")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + path


def upstream_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return headers


def upstream_error_message(status: int, text: str) -> str:
    try:
        message = as_dict(as_dict(json.loads(text)).get("error")).get("message")
        if isinstance(message, str) and message:
            return f"Upstream returned {status}: {message}"
    except ValueError:
        pass
    return f"Upstream returned {status}: {short(text, 300)}"


def describe_exception(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".rstrip(": ")


async def stream_upstream_text(model: str, system: str, user: str) -> AsyncIterator[str]:
    """One streaming chat call that should answer with a JSON object; yields the reply text as it arrives."""
    if not model:
        raise RuntimeError("No model to use for control generation (set one in Settings).")
    url = upstream_url("/chat/completions")
    payload: dict[str, Any] = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        for use_json_mode in (True, False):
            data = dict(payload)
            if use_json_mode:
                data["response_format"] = {"type": "json_object"}
            async with client.stream("POST", url, json=data, headers=upstream_headers()) as resp:
                if resp.status_code == 400 and use_json_mode:
                    await resp.aread()
                    continue  # some servers reject response_format; retry without it
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    raise RuntimeError(upstream_error_message(resp.status_code, text))
                if "json" in resp.headers.get("content-type", "") and "event-stream" not in resp.headers.get("content-type", ""):
                    # A server that ignored stream=true and sent the whole answer at once.
                    whole = as_dict(json.loads((await resp.aread()).decode("utf-8", "replace")))
                    choices = as_list(whole.get("choices"))
                    message = as_dict(as_dict(choices[0]).get("message")) if choices else {}
                    yield content_text(message.get("content"))
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        chunk = as_dict(json.loads(raw))
                    except ValueError:
                        continue
                    for choice in as_list(chunk.get("choices")):
                        piece = as_dict(as_dict(choice).get("delta")).get("content")
                        if isinstance(piece, str) and piece:
                            yield piece
                return
    raise RuntimeError("Upstream rejected the control-generation request.")


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks, including one that is still being written."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:  # a model that starts inside its reasoning
        text = text.split("</think>", 1)[1]
    open_at = text.find("<think>")
    return (text[:open_at] if open_at != -1 else text).strip()


def extract_json(text: str) -> Any:
    text = strip_reasoning(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


# --------------------------------------------------------------------------
# Tool controls: schema -> fields, AI polish, cache
# --------------------------------------------------------------------------


def normalize_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tools as [{name, description, parameters}] from tools[] or legacy functions[]."""
    found: list[dict[str, Any]] = []
    for entry in as_list(body.get("tools")):
        tool = as_dict(entry)
        if tool.get("type", "function") == "function":
            found.append(as_dict(tool.get("function")))
    for entry in as_list(body.get("functions")):
        found.append(as_dict(entry))
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fn in found:
        name = fn.get("name")
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        tools.append(
            {
                "name": name,
                "description": str(fn.get("description") or ""),
                "parameters": as_dict(fn.get("parameters")),
            }
        )
    return tools


def tool_key(tool: dict[str, Any]) -> str:
    raw = json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((PROMPT_SALT + SPEC_VERSION + raw).encode("utf-8")).hexdigest()[:32]


def schema_type(schema: dict[str, Any]) -> str:
    kind = schema.get("type")
    if isinstance(kind, list):
        non_null = [k for k in cast(list[Any], kind) if k != "null"]
        kind = non_null[0] if len(non_null) == 1 else None
    return kind if isinstance(kind, str) else ""


SCALARS = (str, int, float, bool)
LONG_TEXT_KEYS = re.compile(r"content|text|code|body|script|prompt|patch|diff|message|description", re.I)


def scalar_enum(value: Any) -> list[Any]:
    items = as_list(value)
    return items if items and all(isinstance(i, SCALARS) for i in items) else []


def is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


KEPT_SCHEMA_KEYS = ("description", "title", "default", "enum", "format", "minimum", "maximum")


def resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    node: Any = root
    for part in ref[2:].split("/"):
        node = as_dict(node).get(part.replace("~1", "/").replace("~0", "~"))
    return as_dict(node)


def settle_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow $ref and fold "X or null" unions, so the schema says directly what the value is."""
    for _ in range(6):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            schema = {**resolve_ref(ref, root), **{k: v for k, v in schema.items() if k != "$ref"}}
            continue
        folded = False
        for union in ("anyOf", "oneOf", "allOf"):
            options = [as_dict(o) for o in as_list(schema.get(union))]
            real = [o for o in options if schema_type(o) != "null"]
            if options and len(real) == 1:
                rest = {k: v for k, v in schema.items() if k != union}
                schema = {**real[0], **rest}
                folded = True
                break
        if not folded:
            break
    return schema


def simplify_schema(schema: dict[str, Any], root: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    """A property's schema reduced to what the form needs: type, enum, bounds, items, properties and required."""
    schema = settle_schema(schema, root)
    out: dict[str, Any] = {k: schema[k] for k in KEPT_SCHEMA_KEYS if k in schema}
    items = schema.get("items")
    props = as_dict(schema.get("properties"))
    kind = schema_type(schema)
    if not kind:
        kind = "object" if props else "array" if isinstance(items, dict) else ""
    if kind:
        out["type"] = kind
    if depth >= MAX_SCHEMA_DEPTH:
        return out
    if isinstance(items, dict):
        out["items"] = simplify_schema(cast(dict[str, Any], items), root, depth + 1)
    if props:
        out["properties"] = {key: simplify_schema(as_dict(value), root, depth + 1) for key, value in props.items()}
        out["required"] = [r for r in as_list(schema.get("required")) if isinstance(r, str)]
    return out


def describe_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a (simplified) property schema to what decides which widgets make sense."""
    kind = schema_type(schema)
    options = scalar_enum(schema.get("enum"))
    if options and kind not in ("string", "integer", "number", "boolean"):
        first = options[0]
        kind = "boolean" if isinstance(first, bool) else "string" if isinstance(first, str) else "number"
    if kind == "object" and not as_dict(schema.get("properties")):
        kind = ""  # a free-form object has no fields to build a form from
    info: dict[str, Any] = {
        "kind": kind if kind in ("string", "integer", "number", "boolean", "array", "object") else "json",
        "options": options,
        "item_kind": "",
        "item_options": [],
    }
    if info["kind"] == "array":
        items = as_dict(schema.get("items"))
        item_kind = schema_type(items)
        info["item_kind"] = item_kind if item_kind in ("string", "integer", "number") else ""
        info["item_options"] = scalar_enum(items.get("enum"))
    return info


def numeric_range(schema: dict[str, Any], hint: dict[str, Any]) -> tuple[float, float] | None:
    low = schema.get("minimum", hint.get("min"))
    high = schema.get("maximum", hint.get("max"))
    if is_number(low) and is_number(high) and low < high:
        return float(low), float(high)
    return None


def allowed_widgets(info: dict[str, Any], ranged: bool) -> list[str]:
    kind = info["kind"]
    if info["options"]:
        return ["select", "radio"]
    if kind == "string":
        return ["text", "textarea", "password", "combo"]
    if kind in ("integer", "number"):
        return ["number", "slider"] if ranged else ["number"]
    if kind == "boolean":
        return ["checkbox", "select", "radio"]
    if kind == "array":
        widgets = ["multiselect"] if info["item_options"] else []
        widgets.append("array")  # rows that can be added, removed and reordered
        if info["item_kind"]:
            widgets.append("list")
        widgets.append("json")
        return widgets
    if kind == "object":
        return ["object", "json"]
    return ["json"]


def widget_options(widget: str, info: dict[str, Any], hint: dict[str, Any]) -> list[Any]:
    if widget in ("select", "radio"):
        return list(info["options"]) if info["options"] else [True, False]
    if widget == "multiselect":
        return list(info["item_options"])
    if widget == "combo":
        return [str(o) for o in scalar_enum(hint.get("options"))[:20]]
    return []


def schema_for_widget(widget: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    """The nested editors build themselves from the schema, so the form carries it along."""
    return schema if widget in ("array", "object") else None


def fallback_field(key: str, schema: dict[str, Any], required: bool) -> dict[str, Any]:
    info = describe_schema(schema)
    kind = info["kind"]
    if info["options"]:
        widget = "select"
    elif kind == "string":
        widget = "textarea" if LONG_TEXT_KEYS.search(key) else "text"
    elif kind in ("integer", "number"):
        widget = "number"
    elif kind == "boolean":
        widget = "checkbox"
    elif kind == "array":
        widget = "multiselect" if info["item_options"] else "array"
    elif kind == "object":
        widget = "object"
    else:
        widget = "json"
    return {
        "key": key,
        "label": humanize(key),
        "type": kind,
        "item_type": info["item_kind"],
        "widget": widget,
        "required": required,
        "options": widget_options(widget, info, {}),
        "min": None,
        "max": None,
        "step": None,
        "default": schema.get("default"),
        "help": short(str(schema.get("description") or ""), 300),
        "schema": schema_for_widget(widget, schema),
    }


def default_fits(spec: dict[str, Any], value: Any) -> bool:
    widget = spec["widget"]
    if value is None:
        return False
    if widget in ("text", "textarea", "password", "combo"):
        return isinstance(value, str)
    if widget in ("number", "slider"):
        return is_number(value) and (spec["type"] != "integer" or float(value).is_integer())
    if widget == "checkbox":
        return isinstance(value, bool)
    if widget in ("select", "radio"):
        return value in spec["options"]
    if widget == "multiselect":
        return isinstance(value, list) and all(v in spec["options"] for v in cast(list[Any], value))
    if widget == "list":
        return isinstance(value, list) and all(isinstance(v, (str, int, float)) for v in cast(list[Any], value))
    if widget == "array":
        return isinstance(value, list)
    if widget == "object":
        return isinstance(value, dict)
    return True


def build_spec(tool: dict[str, Any], ai: dict[str, Any] | None) -> dict[str, Any]:
    """Merge schema-derived fields (authoritative) with AI-written labels, hints and widget choices."""
    ai = ai or {}
    params = as_dict(tool.get("parameters"))
    props = as_dict(params.get("properties"))
    required = {r for r in as_list(params.get("required")) if isinstance(r, str)}
    ai_fields: dict[str, dict[str, Any]] = {}
    for item in as_list(ai.get("fields")):
        entry = as_dict(item)
        key = entry.get("key")
        if isinstance(key, str):
            ai_fields[key] = entry

    fields: list[dict[str, Any]] = []
    for key, raw_schema in props.items():
        schema = simplify_schema(as_dict(raw_schema), params)
        spec = fallback_field(key, schema, key in required)
        info = describe_schema(schema)
        hint = ai_fields.get(key, {})

        label = hint.get("label")
        if isinstance(label, str) and label.strip():
            spec["label"] = short(label, 60)
        helptext = hint.get("help")
        if isinstance(helptext, str):
            spec["help"] = short(helptext, 300)

        value_range = numeric_range(schema, hint)
        allowed = allowed_widgets(info, value_range is not None)
        widget = hint.get("widget")
        if not isinstance(widget, str) and isinstance(hint.get("multiline"), bool) and info["kind"] == "string":
            widget = "textarea" if hint["multiline"] else "text"  # older-style hint
        if isinstance(widget, str) and widget in allowed:
            spec["widget"] = widget
        spec["options"] = widget_options(spec["widget"], info, hint)
        spec["schema"] = schema_for_widget(spec["widget"], schema)
        if spec["widget"] == "slider" and value_range is not None:
            spec["min"], spec["max"] = value_range
            step = hint.get("step")
            spec["step"] = step if is_number(step) and step > 0 else None

        if spec["default"] is not None and not default_fits(spec, spec["default"]):
            spec["default"] = None
        if spec["default"] is None and default_fits(spec, hint.get("default")):
            spec["default"] = hint["default"]
        fields.append(spec)

    title = ai.get("title")
    summary = ai.get("summary")
    return {
        "name": tool["name"],
        "title": short(title, 60) if isinstance(title, str) and title.strip() else humanize(tool["name"]),
        "summary": short(summary, 240)
        if isinstance(summary, str) and summary.strip()
        else short(str(tool.get("description") or ""), 160),
        "fields": fields,
        "source": "ai" if ai else "schema",
    }


def provisional_controls(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [CONTROLS_CACHE.get(tool_key(t)) or build_spec(t, None) for t in tools]


def finished_controls(text: str) -> list[dict[str, Any]]:
    """The tool objects already complete inside the "controls" array of a reply that may still be arriving."""
    text = strip_reasoning(text)
    opening = re.search(r'"controls"\s*:\s*\[', text)
    if opening is None:
        return []
    found: list[dict[str, Any]] = []
    depth, start = 0, -1
    in_string = escaped = False
    for index in range(opening.end(), len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    found.append(as_dict(json.loads(text[start : index + 1])))
                except ValueError:
                    pass
                start = -1
        elif char == "]" and depth == 0:
            break
    return found


async def read_tools(tools: list[dict[str, Any]], model: str, on_tool: Callable[[dict[str, Any]], None]) -> None:
    """Have the AI read `tools`, calling `on_tool` with each tool's hints the moment that tool is complete."""
    trimmed = [{**t, "description": t["description"][:2000]} for t in tools]
    delivered: set[str] = set()

    def deliver(entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            name = entry.get("name")
            if isinstance(name, str) and name not in delivered:
                delivered.add(name)
                on_tool(entry)

    text = ""
    async for piece in stream_upstream_text(model, CONTROLS_PROMPT, json.dumps({"tools": trimmed}, ensure_ascii=False)):
        text += piece
        if "}" in piece:
            deliver(finished_controls(text))
    deliver(finished_controls(text))
    try:
        parsed = as_dict(extract_json(text))
    except ValueError:
        if not delivered:
            raise RuntimeError("The AI's reply was not valid JSON") from None
        return
    deliver([as_dict(item) for item in as_list(parsed.get("controls"))])


# --------------------------------------------------------------------------
# Recognising a client's own background prompts, and which chat a request belongs to
# --------------------------------------------------------------------------


def load_utility_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    try:
        lines = UTILITY_PATTERNS_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        log.warning("No patterns file at %s: background prompts will not be recognised", UTILITY_PATTERNS_FILE)
        return patterns
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label, separator, expression = line.partition(" :: ")
        if not separator:
            label, expression = "background task", line
        try:
            patterns.append((label.strip(), re.compile(expression.strip(), re.IGNORECASE)))
        except re.error as exc:
            log.warning("%s line %d skipped, bad pattern: %s", UTILITY_PATTERNS_FILE.name, number, exc)
    return patterns


UTILITY_PATTERNS = load_utility_patterns()


def last_user_text(body: dict[str, Any]) -> str:
    for item in reversed(as_list(body.get("messages"))):
        message = as_dict(item)
        if message.get("role") == "user":
            return content_text(message.get("content"))
    return ""


def utility_kind(body: dict[str, Any]) -> str:
    """The label of the matching background-task pattern (e.g. "tags"), or "" for a normal request."""
    text = last_user_text(body)[:UTILITY_SCAN_CHARS]
    for label, pattern in UTILITY_PATTERNS:
        if pattern.search(text):
            return label
    return ""


Key = tuple[str, str]  # (role, text with whitespace collapsed)
WRAP_MARGIN = 40  # how much text a client must have added around a message for us to call it "wrapped"


def conversation_keys(body: dict[str, Any]) -> list[Key]:
    """The conversation as comparable keys: user and tool messages only.

    System prompts are skipped because clients change them between turns, and so are assistant messages
    because clients rewrite them (reasoning blocks, citations, ...). What the user and the tools said is
    what identifies a chat.
    """
    keys: list[Key] = []
    for item in as_list(body.get("messages")):
        message = as_dict(item)
        role = str(message.get("role", ""))
        if role in ("user", "tool"):
            keys.append((role, " ".join(content_text(message.get("content")).split())))
    return keys


FUZZY_RATIO = 0.93  # how alike two messages must be to count as "the same, slightly edited"
FUZZY_MIN_CHARS = 20
FUZZY_SAMPLE = 1500
EXTEND_MIN_CHARS = 12


def texts_close(a: str, b: str) -> bool:
    """Equal, or so alike that a client's small edit (a date, a trailing note) explains the difference."""
    if a == b:
        return True
    longest = max(len(a), len(b))
    if min(len(a), len(b)) < FUZZY_MIN_CHARS or abs(len(a) - len(b)) > longest * 0.1:
        return False
    if longest > FUZZY_SAMPLE * 2:
        a, b = a[:FUZZY_SAMPLE] + a[-FUZZY_SAMPLE:], b[:FUZZY_SAMPLE] + b[-FUZZY_SAMPLE:]
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return matcher.quick_ratio() >= FUZZY_RATIO and matcher.ratio() >= FUZZY_RATIO


def key_close(old: Key, new: Key) -> bool:
    return old[0] == new[0] and texts_close(old[1], new[1])


def wrapped(old: Key, new: Key) -> bool:
    """`old` is `new` with a lot of text around it, e.g. a client's retrieval template around the question."""
    return old[0] == new[0] and len(new[1]) >= 4 and new[1] in old[1] and len(old[1]) - len(new[1]) >= WRAP_MARGIN


def extends(old: Key, new: Key) -> bool:
    """`new` starts with all of `old` and goes on: a client that keeps the whole transcript inside one message."""
    return old[0] == new[0] and len(old[1]) >= EXTEND_MIN_CHARS and len(new[1]) > len(old[1]) and new[1].startswith(old[1])


def prefix_close(old: list[Key], new: list[Key]) -> bool:
    return all(key_close(o, n) for o, n in zip(old, new))


def continues(old: list[Key], new: list[Key]) -> bool:
    """Whether `new` is `old` plus later messages, the way a client resends a chat's history."""
    n = len(old)
    if n == 0 or not new:
        return False
    if len(new) < n:
        return prefix_close(old, new)  # the client rewound the chat and resent the start of it
    for offset in range(n):
        overlap = n - offset
        if offset:
            if overlap < 2:
                break  # too little in common to trust once the client has dropped old messages
            if len(new) <= overlap or not key_close(old[offset], new[0]):
                continue
        if not prefix_close(old[offset : n - 1], new[: overlap - 1]):
            continue
        last_old, last_new = old[n - 1], new[overlap - 1]
        # The newest message of `old` may since have been edited, lost the text the client wrapped around it,
        # or grown into a transcript of everything said since.
        if key_close(last_old, last_new) or extends(last_old, last_new):
            return True
        if len(new) > overlap and wrapped(last_old, last_new):
            return True
    return False


def compare_keys(old: list[Key], new: list[Key], limit: int = 12) -> list[dict[str, Any]]:
    """Side-by-side rows saying where two chats agree, for the "why is this a separate chat?" panel."""
    rows: list[dict[str, Any]] = []
    for index in range(min(max(len(old), len(new)), limit)):
        before = old[index] if index < len(old) else None
        after = new[index] if index < len(new) else None
        if before is None:
            state = "only-new"
        elif after is None:
            state = "only-old"
        elif before == after:
            state = "same"
        elif key_close(before, after):
            state = "close"
        else:
            state = "differs"
        rows.append(
            {
                "index": index + 1,
                "state": state,
                "old": f"{before[0]}: {short(before[1], 90)}" if before else "",
                "new": f"{after[0]}: {short(after[1], 90)}" if after else "",
            }
        )
    return rows


def conversation_hint(request: Request, body: dict[str, Any]) -> str:
    """A chat id the client sent explicitly (header or body field), which beats guessing from the history."""
    for name in CHAT_ID_HEADERS:
        value = request.headers.get(name, "").strip()
        if value:
            return value[:100]
    for name in CHAT_ID_FIELDS:
        value = body.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:100]
    return ""


# --------------------------------------------------------------------------
# Request queue
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    arguments: str  # JSON text
    id: str = field(default_factory=lambda: "call_" + secrets.token_hex(12))


@dataclass
class Decision:
    kind: Literal["forward", "manual", "error"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=lambda: [])
    error: str = ""


@dataclass
class PendingRequest:
    id: str
    created: float
    body: dict[str, Any]
    stream: bool
    future: asyncio.Future[Decision]
    status: str = "pending"  # pending|forwarding|forwarded|answered|rejected|error|cancelled
    error: str = ""
    answer: dict[str, Any] | None = None
    controls_state: str = "ready"  # working|ready
    controls_error: str = ""
    controls_version: int = 0
    controls: list[dict[str, Any]] = field(default_factory=lambda: [])
    controls_task: asyncio.Task[None] | None = None
    conversation: str = ""  # requests that continue one another share this
    turn: int = 1
    auto: bool = False  # a client background task, answered by the AI without asking
    kind: str = ""  # which background task (title, tags, ...)
    keys: list[Key] = field(default_factory=lambda: [])
    note: str = ""  # why this request did not join a chat it nearly matched
    hint: str = ""  # chat id the client sent, if any
    clear_scheduled: bool = False
    controls_done: int = 0  # tools read so far
    controls_total: int = 0


REQUESTS: dict[str, PendingRequest] = {}
ACTIVE_STATUSES = ("pending", "forwarding")


@dataclass
class ChatMemory:
    """What we remember of a chat after its requests were cleared, so it can still be continued."""

    conversation: str
    keys: list[Key]
    turn: int
    stamp: float


CHAT_MEMORY: dict[str, ChatMemory] = {}
HINTS: dict[str, str] = {}  # chat id a client sent -> the conversation it belongs to


def remember(pr: PendingRequest) -> None:
    if pr.auto or not pr.keys:
        return
    CHAT_MEMORY.pop(pr.conversation, None)  # re-insert so the dict stays ordered oldest first
    CHAT_MEMORY[pr.conversation] = ChatMemory(pr.conversation, pr.keys, pr.turn, pr.created)
    while len(CHAT_MEMORY) > CHAT_MEMORY_LIMIT:
        CHAT_MEMORY.pop(next(iter(CHAT_MEMORY)))


def chat_candidates() -> list[ChatMemory]:
    found = list(CHAT_MEMORY.values())
    found.extend(ChatMemory(pr.conversation, pr.keys, pr.turn, pr.created) for pr in REQUESTS.values() if not pr.auto and pr.keys)
    return found


def latest_turn(conversation: str) -> int:
    turns = [pr.turn for pr in REQUESTS.values() if pr.conversation == conversation]
    remembered = CHAT_MEMORY.get(conversation)
    return max(turns + ([remembered.turn] if remembered else []), default=0)


def find_continued(keys: list[Key]) -> ChatMemory | None:
    """The earlier chat state this request continues, preferring the one with the most in common."""
    best: ChatMemory | None = None
    for candidate in chat_candidates():
        if continues(candidate.keys, keys) and (
            best is None or (len(candidate.keys), candidate.stamp) > (len(best.keys), best.stamp)
        ):
            best = candidate
    return best


def explain_no_match(keys: list[Key]) -> str:
    """For a request that joined no chat: the earlier chat it almost matched, and where they differ."""
    best_common, best_old = 0, None
    for candidate in chat_candidates():
        common = 0
        for old_key, new_key in zip(candidate.keys, keys):
            if not key_close(old_key, new_key):
                break
            common += 1
        if common > best_common and common < len(candidate.keys) and common < len(keys):
            best_common, best_old = common, candidate
    if best_old is None:
        return ""
    old_role, old_text = best_old.keys[best_common]
    new_role, new_text = keys[best_common]
    return (
        f"Not joined to an earlier chat that matches up to message {best_common}: "
        f"it then has {old_role} “{short(old_text, 70)}” but this has {new_role} “{short(new_text, 70)}”"
    )


def match_report(pr: PendingRequest) -> dict[str, Any] | None:
    """How this request compares with the closest other chat, so a missed merge can be understood."""
    if pr.auto or not pr.keys:
        return None
    others = [c for c in chat_candidates() if c.conversation != pr.conversation]
    if not others:
        return None
    best = max(others, key=lambda c: (sum(1 for o, n in zip(c.keys, pr.keys) if key_close(o, n)), c.stamp))
    return {
        "conversation": best.conversation,
        "turn": best.turn,
        "old_count": len(best.keys),
        "new_count": len(pr.keys),
        "rows": compare_keys(best.keys, pr.keys),
    }


def assign_conversation(keys: list[Key], hint: str) -> tuple[str, int, str]:
    """(chat id, turn number, note). The note explains a near miss and is empty when there is nothing to say."""
    if hint:
        conversation = HINTS.get(hint)
        if conversation is None:
            # A chat we followed by its text before the client started sending an id; one that already has its own id is not it.
            earlier = find_continued(keys)
            adopt = earlier is not None and earlier.conversation not in HINTS.values()
            conversation = earlier.conversation if earlier is not None and adopt else secrets.token_hex(4)
            HINTS[hint] = conversation
            while len(HINTS) > CHAT_MEMORY_LIMIT * 2:
                HINTS.pop(next(iter(HINTS)))
        return conversation, latest_turn(conversation) + 1, ""
    earlier = find_continued(keys)
    if earlier is not None:
        return earlier.conversation, earlier.turn + 1, ""
    return secrets.token_hex(4), 1, explain_no_match(keys)


class Hub:
    """Fans server events out to every connected operator UI, so the UI never has to poll."""

    def __init__(self) -> None:
        self.clients: set[asyncio.Queue[str | None]] = set()

    def subscribe(self) -> asyncio.Queue[str | None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=HUB_QUEUE_LIMIT)
        self.clients.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str | None]) -> None:
        self.clients.discard(queue)

    def publish(self, event: dict[str, Any]) -> None:
        if not self.clients:
            return
        text = json.dumps(event, ensure_ascii=False)
        for queue in list(self.clients):
            try:
                queue.put_nowait(text)
            except asyncio.QueueFull:
                # A stalled client: drop its backlog and tell it to reconnect, which resyncs it from a fresh snapshot.
                self.clients.discard(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)


HUB = Hub()


def drop_auto(request_id: str) -> None:
    pr = REQUESTS.get(request_id)
    if pr is None or not pr.auto or pr.status in ACTIVE_STATUSES:
        return
    del REQUESTS[request_id]
    HUB.publish({"type": "removed", "ids": [request_id]})


def publish_request(pr: PendingRequest) -> None:
    HUB.publish({"type": "request", "request": summarize(pr)})
    if pr.auto and AUTO_CLEAR_SECONDS > 0 and not pr.clear_scheduled and pr.status not in ACTIVE_STATUSES:
        pr.clear_scheduled = True  # a finished background task tidies itself away
        keep = AUTO_CLEAR_SECONDS * (10 if pr.status == "error" else 1)
        asyncio.get_running_loop().call_later(keep, drop_auto, pr.id)


def controls_payload(pr: PendingRequest) -> dict[str, Any]:
    return {
        "id": pr.id,
        "version": pr.controls_version,
        "state": pr.controls_state,
        "error": pr.controls_error,
        "done": pr.controls_done,
        "total": pr.controls_total,
        "controls": pr.controls,
    }


def bump_controls(pr: PendingRequest) -> None:
    """Call after any change to a request's controls; the version lets the UI ignore out-of-order copies."""
    pr.controls_version += 1
    HUB.publish({"type": "controls", **controls_payload(pr)})


def cancel_request(pr: PendingRequest) -> None:
    if pr.status in ACTIVE_STATUSES:
        pr.status = "cancelled"
        publish_request(pr)


def controls_model_for(pr: PendingRequest) -> str:
    return SETTINGS.controls_model or SETTINGS.forward_model or DEFAULT_MODEL


def accept_control(
    pr: PendingRequest,
    tools: list[dict[str, Any]],
    wanted: set[str],
    accepted: set[str],
    entry: dict[str, Any],
) -> None:
    """Take one tool's hints from the AI the moment they are complete: cache them and show the progress."""
    name = entry.get("name")
    if not isinstance(name, str) or name not in wanted or name in accepted:
        return
    tool = next(t for t in tools if t["name"] == name)
    CONTROLS_CACHE[tool_key(tool)] = build_spec(tool, entry)
    accepted.add(name)
    pr.controls_done += 1
    pr.controls = provisional_controls(tools)
    bump_controls(pr)


async def prepare_controls(pr: PendingRequest, force: bool = False) -> None:
    tools = normalize_tools(pr.body)
    errors: list[str] = []
    try:
        async with CONTROLS_LOCK:
            todo = [t for t in tools if force or tool_key(t) not in CONTROLS_CACHE]
            pr.controls_total = len(tools)
            pr.controls_done = len(tools) - len(todo)
            bump_controls(pr)
            for start in range(0, len(todo), CONTROLS_BATCH):
                batch = todo[start : start + CONTROLS_BATCH]
                wanted = {t["name"] for t in batch}
                accepted: set[str] = set()
                failed = False
                try:
                    await read_tools(
                        batch,
                        controls_model_for(pr),
                        lambda entry: accept_control(pr, tools, wanted, accepted, entry),
                    )
                except Exception as exc:  # noqa: BLE001 - any failure falls back to schema controls
                    failed = True
                    errors.append(describe_exception(exc))
                if accepted:
                    save_cache()
                for tool in batch:
                    if tool["name"] not in accepted:
                        if not failed:
                            errors.append(f"the AI left out {tool['name']}")
                        pr.controls_done += 1  # nothing more will come for it; the schema form stands in
                pr.controls = provisional_controls(tools)
                bump_controls(pr)
    except Exception as exc:  # noqa: BLE001
        errors.append(describe_exception(exc))
    pr.controls = provisional_controls(tools)
    pr.controls_done = pr.controls_total
    pr.controls_error = short("; ".join(dict.fromkeys(errors)), 400)
    pr.controls_state = "ready"
    bump_controls(pr)


def prune() -> None:
    """Keep history bounded. Background tasks have their own budget so they never push real chats out."""
    finished = [rid for rid, pr in REQUESTS.items() if pr.status not in ACTIVE_STATUSES]
    auto_total = sum(1 for pr in REQUESTS.values() if pr.auto)
    regular_total = len(REQUESTS) - auto_total
    auto_ids = [rid for rid in finished if REQUESTS[rid].auto]
    regular_ids = [rid for rid in finished if not REQUESTS[rid].auto]
    removed = auto_ids[: max(0, auto_total - AUTO_HISTORY)] + regular_ids[: max(0, regular_total - MAX_HISTORY)]
    for rid in removed:
        del REQUESTS[rid]
    if removed:
        HUB.publish({"type": "removed", "ids": removed})


def create_request(body: dict[str, Any], generate_controls: bool, hint: str = "", kind: str = "") -> PendingRequest:
    pr = PendingRequest(
        id=secrets.token_hex(4),
        created=time.time(),
        body=body,
        stream=bool(body.get("stream")),
        future=asyncio.get_running_loop().create_future(),
    )
    pr.kind = kind
    pr.auto = bool(kind)
    if pr.auto:
        pr.conversation = secrets.token_hex(4)
    else:
        pr.keys = conversation_keys(body)
        pr.conversation, pr.turn, pr.note = assign_conversation(pr.keys, hint)
        remember(pr)
        if pr.note:
            log.info("request %s: %s", pr.id, pr.note)
    pr.hint = hint
    tools = normalize_tools(body)
    pr.controls = provisional_controls(tools)
    missing = [t for t in tools if tool_key(t) not in CONTROLS_CACHE]
    pr.controls_total = len(tools)
    pr.controls_done = len(tools) - len(missing)
    if missing and generate_controls:
        pr.controls_state = "working"
        pr.controls_task = asyncio.create_task(prepare_controls(pr))
    REQUESTS[pr.id] = pr
    prune()
    publish_request(pr)
    log.info(
        "request %s: model=%s messages=%d tools=%d stream=%s chat=%s turn=%d shape=[%s]%s",
        pr.id,
        body.get("model"),
        len(as_list(body.get("messages"))),
        len(tools),
        pr.stream,
        pr.conversation,
        pr.turn,
        " ".join(f"{role[0]}{len(text)}" for role, text in pr.keys),
        f" background={pr.kind}" if pr.auto else "",
    )
    return pr


def get_request(request_id: str) -> PendingRequest:
    pr = REQUESTS.get(request_id)
    if pr is None:
        raise HTTPException(404, "Unknown request")
    return pr


def resolve(pr: PendingRequest, decision: Decision, answer: dict[str, Any] | None = None) -> None:
    if pr.status != "pending" or pr.future.done():
        raise HTTPException(409, "This request is no longer waiting for an answer")
    pr.status = {"forward": "forwarding", "manual": "answered", "error": "rejected"}[decision.kind]
    if decision.kind == "error":
        pr.error = decision.error
    if answer is not None:
        pr.answer = answer
    pr.future.set_result(decision)
    publish_request(pr)


def fail(pr: PendingRequest, message: str) -> None:
    pr.status = "error"
    pr.error = message
    publish_request(pr)
    log.warning("request %s failed: %s", pr.id, message)


# --------------------------------------------------------------------------
# Building OpenAI-shaped responses
# --------------------------------------------------------------------------


def completion_id() -> str:
    return "chatcmpl-" + secrets.token_hex(12)


def model_name(pr: PendingRequest) -> str:
    return str(pr.body.get("model") or "human")


def build_message(decision: Decision) -> dict[str, Any]:
    has_text = bool(decision.content)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": decision.content if has_text or not decision.tool_calls else None,
    }
    if decision.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in decision.tool_calls
        ]
    return message


def manual_completion(pr: PendingRequest, decision: Decision) -> dict[str, Any]:
    return {
        "id": completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name(pr),
        "choices": [
            {
                "index": 0,
                "message": build_message(decision),
                "finish_reason": "tool_calls" if decision.tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def manual_stream_chunks(pr: PendingRequest, decision: Decision) -> list[str]:
    cid, created, model = completion_id(), int(time.time()), model_name(pr)

    def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
        return sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    out = [chunk({"role": "assistant", "content": ""})]
    if decision.content:
        out.append(chunk({"content": decision.content}))
    for index, call in enumerate(decision.tool_calls):
        out.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                    ]
                }
            )
        )
    out.append(chunk({}, "tool_calls" if decision.tool_calls else "stop"))
    if as_dict(pr.body.get("stream_options")).get("include_usage"):
        out.append(
            sse(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            )
        )
    out.append("data: [DONE]\n\n")
    return out


def answer_from_message(message: dict[str, Any]) -> dict[str, Any]:
    calls: list[dict[str, str]] = []
    for item in as_list(message.get("tool_calls")):
        fn = as_dict(as_dict(item).get("function"))
        calls.append({"name": str(fn.get("name", "")), "arguments": pretty_json(str(fn.get("arguments", "")))})
    return {"content": content_text(message.get("content")), "tool_calls": calls}


# --------------------------------------------------------------------------
# Forwarding to the upstream model
# --------------------------------------------------------------------------


def forward_payload(pr: PendingRequest) -> dict[str, Any]:
    payload = dict(pr.body)
    if SETTINGS.forward_model:
        payload["model"] = SETTINGS.forward_model
    return payload


class StreamAccumulator:
    """Collects the assistant text and tool calls out of an upstream SSE stream."""

    def __init__(self) -> None:
        self.content = ""
        self.calls: dict[int, dict[str, str]] = {}

    def feed(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return
        try:
            obj = as_dict(json.loads(data))
        except ValueError:
            return
        for choice in as_list(obj.get("choices")):
            delta = as_dict(as_dict(choice).get("delta"))
            text = delta.get("content")
            if isinstance(text, str):
                self.content += text
            for item in as_list(delta.get("tool_calls")):
                call = as_dict(item)
                index = call.get("index", 0)
                slot = self.calls.setdefault(index if isinstance(index, int) else 0, {"name": "", "arguments": ""})
                fn = as_dict(call.get("function"))
                if isinstance(fn.get("name"), str):
                    slot["name"] += fn["name"]
                if isinstance(fn.get("arguments"), str):
                    slot["arguments"] += fn["arguments"]

    def answer(self) -> dict[str, Any]:
        calls = [
            {"name": slot["name"], "arguments": pretty_json(slot["arguments"])}
            for _, slot in sorted(self.calls.items())
        ]
        return {"content": self.content, "tool_calls": calls}


async def forward_json(pr: PendingRequest) -> Response:
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(
                upstream_url("/chat/completions"), json=forward_payload(pr), headers=upstream_headers()
            )
    except (httpx.HTTPError, RuntimeError) as exc:
        message = f"Upstream request failed: {describe_exception(exc)}"
        fail(pr, message)
        return error_response(502, message)
    try:
        data: Any = resp.json()
    except ValueError:
        data = None
    if resp.status_code >= 400:
        message = upstream_error_message(resp.status_code, resp.text)
        fail(pr, message)
        content = data if isinstance(data, dict) else error_body(message, "upstream_error")
        return JSONResponse(content, status_code=resp.status_code)
    if not isinstance(data, dict):
        message = "Upstream returned a non-JSON response"
        fail(pr, message)
        return error_response(502, message)
    choices = as_list(cast(dict[str, Any], data).get("choices"))
    first = as_dict(choices[0]) if choices else {}
    pr.answer = answer_from_message(as_dict(first.get("message")))
    pr.status = "forwarded"
    publish_request(pr)
    return JSONResponse(data)


async def forward_stream(pr: PendingRequest) -> AsyncIterator[str]:
    accumulator = StreamAccumulator()
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream(
                "POST", upstream_url("/chat/completions"), json=forward_payload(pr), headers=upstream_headers()
            ) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    message = upstream_error_message(resp.status_code, text)
                    fail(pr, message)
                    yield sse(error_body(message, "upstream_error"))
                    yield "data: [DONE]\n\n"
                    return
                async for line in resp.aiter_lines():
                    accumulator.feed(line)
                    yield line + "\n"
    except (httpx.HTTPError, RuntimeError) as exc:
        message = f"Upstream request failed: {describe_exception(exc)}"
        fail(pr, message)
        yield sse(error_body(message, "upstream_error"))
        yield "data: [DONE]\n\n"
        return
    pr.answer = accumulator.answer()
    pr.status = "forwarded"
    publish_request(pr)


# --------------------------------------------------------------------------
# The OpenAI-compatible endpoints
# --------------------------------------------------------------------------


class V1CorsMiddleware:
    """CORS for /v1 only, so browser-based clients work but the operator UI API stays same-origin."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not str(scope["path"]).startswith("/v1"):
            await self.app(scope, receive, send)
            return
        if scope["method"] == "OPTIONS":
            requested = Headers(scope=scope).get("access-control-request-headers", "authorization, content-type")
            preflight = Response(
                status_code=204, headers={**CORS_HEADERS, "Access-Control-Allow-Headers": requested}
            )
            await preflight(scope, receive, send)
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in CORS_HEADERS.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_with_cors)


app = FastAPI(title="Human LLM", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(V1CorsMiddleware)


def check_auth(request: Request) -> Response | None:
    if not API_KEY:
        return None
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else ""
    if secrets.compare_digest(token.encode(), API_KEY.encode()):
        return None
    return error_response(401, "Invalid or missing API key")


@app.get("/v1/models")
async def list_models(request: Request) -> Response:
    denied = check_auth(request)
    if denied:
        return denied
    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": "human", "object": "model", "created": int(time.time()), "owned_by": "human-llm"}],
        }
    )


async def wait_for_decision(request: Request, pr: PendingRequest) -> Decision | None:
    while not pr.future.done():
        await asyncio.wait({pr.future}, timeout=0.5)
        if not pr.future.done() and await request.is_disconnected():
            return None
    return pr.future.result()


async def json_flow(request: Request, pr: PendingRequest) -> Response:
    try:
        decision = await wait_for_decision(request, pr)
        if decision is None:
            cancel_request(pr)
            return Response(status_code=499)
        if decision.kind == "forward":
            return await forward_json(pr)
        if decision.kind == "error":
            return error_response(500, decision.error or "The operator declined this request")
        return JSONResponse(manual_completion(pr, decision))
    except asyncio.CancelledError:
        cancel_request(pr)
        raise


async def stream_flow(request: Request, pr: PendingRequest) -> AsyncIterator[str]:
    try:
        last_ping = time.monotonic()
        while not pr.future.done():
            await asyncio.wait({pr.future}, timeout=0.5)
            if pr.future.done():
                break
            if await request.is_disconnected():
                cancel_request(pr)
                return
            if time.monotonic() - last_ping >= KEEPALIVE_SECONDS:
                last_ping = time.monotonic()
                yield ": keepalive\n\n"
        decision = pr.future.result()
        if decision.kind == "manual":
            for piece in manual_stream_chunks(pr, decision):
                yield piece
        elif decision.kind == "error":
            yield sse(error_body(decision.error or "The operator declined this request"))
            yield "data: [DONE]\n\n"
        else:
            async for piece in forward_stream(pr):
                yield piece
    except asyncio.CancelledError:
        cancel_request(pr)
        raise


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    denied = check_auth(request)
    if denied:
        return denied
    try:
        body: Any = await request.json()
    except ValueError:
        return error_response(400, "Request body must be valid JSON")
    if not isinstance(body, dict) or not isinstance(cast(dict[str, Any], body).get("messages"), list):
        return error_response(400, "Request body must be an object with a 'messages' array")
    payload = cast(dict[str, Any], body)

    kind = utility_kind(payload) if SETTINGS.auto_utility else ""
    auto = SETTINGS.auto_forward or bool(kind)  # either way the AI answers without asking the operator
    pr = create_request(payload, generate_controls=not auto, hint=conversation_hint(request, payload), kind=kind)
    if auto:
        resolve(pr, Decision("forward"))

    if pr.stream:
        return StreamingResponse(
            stream_flow(request, pr),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await json_flow(request, pr)


# --------------------------------------------------------------------------
# Operator UI API
# --------------------------------------------------------------------------


def messages_view(body: dict[str, Any]) -> list[dict[str, Any]]:
    view: list[dict[str, Any]] = []
    for item in as_list(body.get("messages")):
        message = as_dict(item)
        calls: list[dict[str, str]] = []
        for raw_call in as_list(message.get("tool_calls")):
            call = as_dict(raw_call)
            fn = as_dict(call.get("function"))
            calls.append(
                {
                    "id": str(call.get("id", "")),
                    "name": str(fn.get("name", "")),
                    "arguments": pretty_json(str(fn.get("arguments", ""))),
                }
            )
        legacy = as_dict(message.get("function_call"))
        if legacy:
            calls.append({"id": "", "name": str(legacy.get("name", "")), "arguments": pretty_json(str(legacy.get("arguments", "")))})
        view.append(
            {
                "role": str(message.get("role", "?")),
                "name": str(message.get("name") or ""),
                "tool_call_id": str(message.get("tool_call_id") or ""),
                "text": content_text(message.get("content")),
                "tool_calls": calls,
            }
        )
    return view


def summarize(pr: PendingRequest) -> dict[str, Any]:
    view = messages_view(pr.body)
    preview = ""
    for message in reversed(view):
        if message["role"] != "system" and (message["text"] or message["tool_calls"]):
            text = message["text"] or f"{message['tool_calls'][0]['name']}(…)"
            preview = f"{message['role']}: {short(text, 140)}"
            break
    if pr.auto:
        reply = str((pr.answer or {}).get("content") or "")
        preview = f"{pr.kind}: {short(reply, 120)}" if reply else pr.kind
    return {
        "id": pr.id,
        "conversation": pr.conversation,
        "turn": pr.turn,
        "auto": pr.auto,
        "kind": pr.kind,
        "note": pr.note,
        "chat_id": pr.hint,
        "created": pr.created,
        "status": pr.status,
        "model": str(pr.body.get("model") or ""),
        "stream": pr.stream,
        "message_count": len(view),
        "tool_count": len(normalize_tools(pr.body)),
        "preview": preview,
    }


@app.get("/api/requests")
async def api_list_requests() -> list[dict[str, Any]]:
    return [summarize(pr) for pr in REQUESTS.values()]


@app.post("/api/requests/clear")
async def api_clear_requests() -> dict[str, int]:
    finished = [rid for rid, pr in REQUESTS.items() if pr.status not in ACTIVE_STATUSES]
    for rid in finished:
        del REQUESTS[rid]
    if finished:
        HUB.publish({"type": "removed", "ids": finished})
    return {"removed": len(finished)}


@app.get("/api/requests/{request_id}")
async def api_get_request(request_id: str) -> dict[str, Any]:
    pr = get_request(request_id)
    params = {k: v for k, v in pr.body.items() if k not in ("messages", "tools", "functions")}
    return {
        **summarize(pr),
        "error": pr.error,
        "answer": pr.answer,
        "messages": messages_view(pr.body),
        "tool_names": [t["name"] for t in normalize_tools(pr.body)],
        "params": params,
        "match": match_report(pr) if pr.turn == 1 else None,
    }


@app.get("/api/requests/{request_id}/controls")
async def api_get_controls(request_id: str) -> dict[str, Any]:
    pr = get_request(request_id)
    return controls_payload(pr)


@app.post("/api/requests/{request_id}/controls/regenerate")
async def api_regenerate_controls(request_id: str) -> dict[str, str]:
    pr = get_request(request_id)
    if pr.controls_state == "working":
        raise HTTPException(409, "Controls are already being generated")
    pr.controls_state = "working"
    pr.controls_error = ""
    pr.controls_total = len(normalize_tools(pr.body))
    pr.controls_done = 0
    bump_controls(pr)
    pr.controls_task = asyncio.create_task(prepare_controls(pr, force=True))
    return {"state": pr.controls_state}


class JoinIn(BaseModel):
    conversation: str


@app.post("/api/requests/{request_id}/join")
async def api_join(request_id: str, payload: JoinIn) -> dict[str, Any]:
    """Merge this request's chat into another one by hand, when automatic matching could not tell they belong together."""
    pr = get_request(request_id)
    if pr.auto:
        raise HTTPException(422, "Background tasks are not part of a chat")
    target, source = payload.conversation, pr.conversation
    if target == source:
        return {"conversation": target}
    if target not in CHAT_MEMORY and not any(r.conversation == target and not r.auto for r in REQUESTS.values()):
        raise HTTPException(404, "That chat no longer exists")
    turn = latest_turn(target)
    moved = sorted((r for r in REQUESTS.values() if r.conversation == source and not r.auto), key=lambda r: r.created)
    for item in moved:
        turn += 1
        item.conversation, item.turn, item.note = target, turn, ""
    CHAT_MEMORY.pop(source, None)
    for hint, conversation in list(HINTS.items()):
        if conversation == source:
            HINTS[hint] = target
    members = [r for r in REQUESTS.values() if r.conversation == target and not r.auto]
    if members:
        remember(max(members, key=lambda r: r.created))
    for item in moved:
        publish_request(item)
    log.info("request %s: chat %s merged into %s by hand", pr.id, source, target)
    return {"conversation": target, "turn": pr.turn}


class ToolCallIn(BaseModel):
    name: str
    arguments: Any = Field(default_factory=dict)


class RespondIn(BaseModel):
    content: str = ""
    tool_calls: list[ToolCallIn] = Field(default_factory=lambda: [])


class RejectIn(BaseModel):
    message: str = ""


def arguments_json(call: ToolCallIn) -> str:
    if isinstance(call.arguments, str):
        try:
            json.loads(call.arguments)
        except ValueError:
            raise HTTPException(422, f"Arguments for {call.name} are not valid JSON") from None
        return call.arguments
    return json.dumps(call.arguments, ensure_ascii=False)


@app.post("/api/requests/{request_id}/respond")
async def api_respond(request_id: str, payload: RespondIn) -> dict[str, str]:
    pr = get_request(request_id)
    calls: list[ToolCall] = []
    for item in payload.tool_calls:
        name = item.name.strip()
        if not name:
            raise HTTPException(422, "Every tool call needs a name")
        calls.append(ToolCall(name=name, arguments=arguments_json(item)))
    if not payload.content.strip() and not calls:
        raise HTTPException(422, "Write a message or add at least one tool call")
    decision = Decision("manual", content=payload.content, tool_calls=calls)
    resolve(pr, decision, answer_from_message(build_message(decision)))
    return {"status": pr.status}


@app.post("/api/requests/{request_id}/forward")
async def api_forward(request_id: str) -> dict[str, str]:
    pr = get_request(request_id)
    resolve(pr, Decision("forward"))
    return {"status": pr.status}


@app.post("/api/requests/{request_id}/reject")
async def api_reject(request_id: str, payload: RejectIn) -> dict[str, str]:
    pr = get_request(request_id)
    resolve(pr, Decision("error", error=payload.message.strip() or "The operator declined this request"))
    return {"status": pr.status}


class SettingsIn(BaseModel):
    forward_model: str | None = None
    controls_model: str | None = None
    auto_forward: bool | None = None
    auto_utility: bool | None = None


def settings_view() -> dict[str, Any]:
    return {
        "ollama_url": OLLAMA_URL,
        "default_model": DEFAULT_MODEL,
        "forward_model": SETTINGS.forward_model,
        "controls_model": SETTINGS.controls_model,
        "auto_forward": SETTINGS.auto_forward,
        "auto_utility": SETTINGS.auto_utility,
        "client_api_key_required": bool(API_KEY),
    }


@app.get("/api/settings")
async def api_get_settings() -> dict[str, Any]:
    return settings_view()


@app.post("/api/settings")
async def api_set_settings(payload: SettingsIn) -> dict[str, Any]:
    if payload.forward_model is not None:
        SETTINGS.forward_model = payload.forward_model.strip()
    if payload.controls_model is not None:
        SETTINGS.controls_model = payload.controls_model.strip()
    if payload.auto_forward is not None:
        SETTINGS.auto_forward = payload.auto_forward
    if payload.auto_utility is not None:
        SETTINGS.auto_utility = payload.auto_utility
    save_settings()
    HUB.publish({"type": "settings", "settings": settings_view()})
    return settings_view()


# --------------------------------------------------------------------------
# Live updates for the operator UI
# --------------------------------------------------------------------------


def origin_allowed(ws: WebSocket) -> bool:
    """Browsers don't apply CORS to WebSockets, so refuse pages from other sites reading the queue."""
    origin = ws.headers.get("origin")
    if not origin or origin.lower() in ALLOWED_ORIGINS:
        return True
    return urlsplit(origin).netloc.lower() == ws.headers.get("host", "").lower()


async def ws_pump(ws: WebSocket, queue: asyncio.Queue[str | None]) -> None:
    while True:
        item = await queue.get()
        if item is None:
            await ws.close(code=1013)  # backlog overflowed; the client reconnects and resyncs
            return
        await ws.send_text(item)


async def ws_watch(ws: WebSocket) -> None:
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    if not origin_allowed(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    queue = HUB.subscribe()  # subscribe before the snapshot so nothing that happens in between is lost
    tasks: list[asyncio.Task[None]] = []
    try:
        hello = {
            "type": "hello",
            "requests": [summarize(pr) for pr in REQUESTS.values()],
            "settings": settings_view(),
        }
        await ws.send_text(json.dumps(hello, ensure_ascii=False))
        tasks = [asyncio.create_task(ws_pump(ws, queue)), asyncio.create_task(ws_watch(ws))]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception:  # noqa: BLE001 - a dropped connection is routine
        pass
    finally:
        HUB.unsubscribe(queue)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await ws.close()


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    shown_host = "localhost" if HOST in ("0.0.0.0", "127.0.0.1") else HOST
    log.info("Operator UI:  http://%s:%d/", shown_host, PORT)
    log.info("API base URL: http://%s:%d/v1", shown_host, PORT)
    log.info("AI upstream:  %s (model %s)", OLLAMA_URL or "NOT SET - add OLLAMA_URL to .env", SETTINGS.forward_model or "client's")
    if HOST == "0.0.0.0":
        log.warning("Listening on all interfaces. The operator UI has no login; keep this port off the open internet.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")