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
import hashlib
import json
import logging
import os
import re
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeGuard, cast

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
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

OLLAMA_URL = os.environ.get("OLLAMA_URL", "").strip()
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "").strip() or "mimo_v2.6:latest"

MAX_HISTORY = 100
KEEPALIVE_SECONDS = 15.0
CONTROLS_BATCH = 6
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
    auto = raw.get("auto_forward")
    if isinstance(auto, bool):
        settings.auto_forward = auto
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


async def call_upstream_json(model: str, system: str, user: str) -> str:
    """One non-streaming chat call that should answer with a JSON object."""
    if not model:
        raise RuntimeError("No model to use for control generation (set one in Settings).")
    url = upstream_url("/chat/completions")
    payload: dict[str, Any] = {
        "model": model,
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
            resp = await client.post(url, json=data, headers=upstream_headers())
            if resp.status_code == 400 and use_json_mode:
                continue  # some servers reject response_format; retry without it
            if resp.status_code >= 400:
                raise RuntimeError(upstream_error_message(resp.status_code, resp.text))
            choices = as_list(as_dict(resp.json()).get("choices"))
            message = as_dict(as_dict(choices[0]).get("message")) if choices else {}
            return content_text(message.get("content"))
    raise RuntimeError("Upstream rejected the control-generation request.")


def extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
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
    return hashlib.sha256((PROMPT_SALT + raw).encode("utf-8")).hexdigest()[:32]


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


def describe_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a JSON Schema property to what decides which widgets make sense."""
    kind = schema_type(schema)
    options = scalar_enum(schema.get("enum"))
    if options and kind not in ("string", "integer", "number", "boolean"):
        first = options[0]
        kind = "boolean" if isinstance(first, bool) else "string" if isinstance(first, str) else "number"
    info: dict[str, Any] = {"kind": kind if kind in ("string", "integer", "number", "boolean", "array") else "json"}
    info["options"] = options
    info["item_kind"] = ""
    info["item_options"] = []
    if info["kind"] == "array":
        items = as_dict(schema.get("items"))
        item_kind = schema_type(items)
        info["item_kind"] = item_kind if item_kind in ("string", "integer", "number") else ""
        info["item_options"] = scalar_enum(items.get("enum"))
        if not info["item_kind"] and not info["item_options"]:
            info["kind"] = "json"
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
        return ["multiselect", "list", "json"] if info["item_options"] else ["list", "json"]
    return ["json"]


def widget_options(widget: str, info: dict[str, Any], hint: dict[str, Any]) -> list[Any]:
    if widget in ("select", "radio"):
        return list(info["options"]) if info["options"] else [True, False]
    if widget == "multiselect":
        return list(info["item_options"])
    if widget == "combo":
        return [str(o) for o in scalar_enum(hint.get("options"))[:20]]
    return []


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
        widget = "multiselect" if info["item_options"] else "list"
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
        schema = as_dict(raw_schema)
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


async def ai_controls(tools: list[dict[str, Any]], model: str) -> dict[str, dict[str, Any]]:
    trimmed = [{**t, "description": t["description"][:2000]} for t in tools]
    text = await call_upstream_json(model, CONTROLS_PROMPT, json.dumps({"tools": trimmed}, ensure_ascii=False))
    parsed = as_dict(extract_json(text))
    result: dict[str, dict[str, Any]] = {}
    for item in as_list(parsed.get("controls")):
        entry = as_dict(item)
        name = entry.get("name")
        if isinstance(name, str):
            result[name] = entry
    return result


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
    controls: list[dict[str, Any]] = field(default_factory=lambda: [])
    controls_task: asyncio.Task[None] | None = None


REQUESTS: dict[str, PendingRequest] = {}
ACTIVE_STATUSES = ("pending", "forwarding")


def controls_model_for(pr: PendingRequest) -> str:
    return SETTINGS.controls_model or SETTINGS.forward_model or DEFAULT_MODEL


async def prepare_controls(pr: PendingRequest, force: bool = False) -> None:
    tools = normalize_tools(pr.body)
    errors: list[str] = []
    try:
        async with CONTROLS_LOCK:
            todo = [t for t in tools if force or tool_key(t) not in CONTROLS_CACHE]
            changed = False
            for start in range(0, len(todo), CONTROLS_BATCH):
                batch = todo[start : start + CONTROLS_BATCH]
                try:
                    ai = await ai_controls(batch, controls_model_for(pr))
                except Exception as exc:  # noqa: BLE001 - any failure falls back to schema controls
                    errors.append(describe_exception(exc))
                    continue
                for tool in batch:
                    hint = ai.get(tool["name"])
                    if hint is None:
                        errors.append(f"the AI left out {tool['name']}")
                        continue
                    CONTROLS_CACHE[tool_key(tool)] = build_spec(tool, hint)
                    changed = True
                pr.controls = provisional_controls(tools)
            if changed:
                save_cache()
    except Exception as exc:  # noqa: BLE001
        errors.append(describe_exception(exc))
    pr.controls = provisional_controls(tools)
    pr.controls_error = short("; ".join(dict.fromkeys(errors)), 400)
    pr.controls_state = "ready"


def prune() -> None:
    excess = len(REQUESTS) - MAX_HISTORY
    if excess <= 0:
        return
    finished = [rid for rid, pr in REQUESTS.items() if pr.status not in ACTIVE_STATUSES]
    for rid in finished[:excess]:
        del REQUESTS[rid]


def create_request(body: dict[str, Any], generate_controls: bool) -> PendingRequest:
    pr = PendingRequest(
        id=secrets.token_hex(4),
        created=time.time(),
        body=body,
        stream=bool(body.get("stream")),
        future=asyncio.get_running_loop().create_future(),
    )
    tools = normalize_tools(body)
    pr.controls = provisional_controls(tools)
    missing = any(tool_key(t) not in CONTROLS_CACHE for t in tools)
    if missing and generate_controls:
        pr.controls_state = "working"
        pr.controls_task = asyncio.create_task(prepare_controls(pr))
    REQUESTS[pr.id] = pr
    prune()
    log.info(
        "request %s: model=%s messages=%d tools=%d stream=%s",
        pr.id,
        body.get("model"),
        len(as_list(body.get("messages"))),
        len(tools),
        pr.stream,
    )
    return pr


def get_request(request_id: str) -> PendingRequest:
    pr = REQUESTS.get(request_id)
    if pr is None:
        raise HTTPException(404, "Unknown request")
    return pr


def resolve(pr: PendingRequest, decision: Decision) -> None:
    if pr.status != "pending" or pr.future.done():
        raise HTTPException(409, "This request is no longer waiting for an answer")
    pr.status = {"forward": "forwarding", "manual": "answered", "error": "rejected"}[decision.kind]
    if decision.kind == "error":
        pr.error = decision.error
    pr.future.set_result(decision)


def fail(pr: PendingRequest, message: str) -> None:
    pr.status = "error"
    pr.error = message
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
    names = ["human"]
    if SETTINGS.forward_model:
        names.append(SETTINGS.forward_model)
    created = int(time.time())
    return JSONResponse(
        {
            "object": "list",
            "data": [{"id": n, "object": "model", "created": created, "owned_by": "human-llm"} for n in names],
        }
    )


async def wait_for_decision(request: Request, pr: PendingRequest) -> Decision | None:
    while not pr.future.done():
        await asyncio.wait({pr.future}, timeout=1.0)
        if not pr.future.done() and await request.is_disconnected():
            return None
    return pr.future.result()


async def json_flow(request: Request, pr: PendingRequest) -> Response:
    try:
        decision = await wait_for_decision(request, pr)
        if decision is None:
            pr.status = "cancelled"
            return Response(status_code=499)
        if decision.kind == "forward":
            return await forward_json(pr)
        if decision.kind == "error":
            return error_response(500, decision.error or "The operator declined this request")
        return JSONResponse(manual_completion(pr, decision))
    except asyncio.CancelledError:
        if pr.status in ACTIVE_STATUSES:
            pr.status = "cancelled"
        raise


async def stream_flow(request: Request, pr: PendingRequest) -> AsyncIterator[str]:
    try:
        last_ping = time.monotonic()
        while not pr.future.done():
            await asyncio.wait({pr.future}, timeout=1.0)
            if pr.future.done():
                break
            if await request.is_disconnected():
                pr.status = "cancelled"
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
        if pr.status in ACTIVE_STATUSES:
            pr.status = "cancelled"
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

    auto = SETTINGS.auto_forward
    pr = create_request(payload, generate_controls=not auto)
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
    return {
        "id": pr.id,
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
    }


@app.get("/api/requests/{request_id}/controls")
async def api_get_controls(request_id: str) -> dict[str, Any]:
    pr = get_request(request_id)
    return {"state": pr.controls_state, "error": pr.controls_error, "controls": pr.controls}


@app.post("/api/requests/{request_id}/controls/regenerate")
async def api_regenerate_controls(request_id: str) -> dict[str, str]:
    pr = get_request(request_id)
    if pr.controls_state == "working":
        raise HTTPException(409, "Controls are already being generated")
    pr.controls_state = "working"
    pr.controls_error = ""
    pr.controls_task = asyncio.create_task(prepare_controls(pr, force=True))
    return {"state": pr.controls_state}


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
    resolve(pr, decision)
    pr.answer = answer_from_message(build_message(decision))
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


def settings_view() -> dict[str, Any]:
    return {
        "ollama_url": OLLAMA_URL,
        "default_model": DEFAULT_MODEL,
        "forward_model": SETTINGS.forward_model,
        "controls_model": SETTINGS.controls_model,
        "auto_forward": SETTINGS.auto_forward,
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
    save_settings()
    return settings_view()


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