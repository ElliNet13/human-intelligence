"""
human-llm-server
=================

A drop-in server that speaks the OpenAI Chat Completions API and the
Anthropic Messages API on the wire -- but there is no model behind it.
Every request lands in a queue, a human opens the "console" page, reads
the exact messages + tool definitions a model would have received, and
either writes a reply or issues tool calls. That answer is translated
back into the shape the calling SDK/client expects.

Run:
    pip install -r requirements.txt
    python app.py
    # Console:            http://localhost:8787/
    # OpenAI endpoint:     http://localhost:8787/v1/chat/completions
    # Anthropic endpoint:  http://localhost:8787/v1/messages

Point any OpenAI or Anthropic SDK at this server's base_url and use any
api_key value (unless SERVER_API_KEY is set, see below).
"""

import json
import os
import time
import uuid
import threading
from queue import Queue

from flask import Flask, request, jsonify, Response, render_template, stream_with_context

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# If set, incoming API requests must present this value (Anthropic: x-api-key
# header, OpenAI: Authorization: Bearer <key>). The console UI is never
# gated by this -- it's assumed to be trusted/local.
SERVER_API_KEY = os.environ.get("SERVER_API_KEY", "")

# How long an API caller will block waiting for a human to answer.
RESPONSE_TIMEOUT_SECONDS = int(os.environ.get("RESPONSE_TIMEOUT_SECONDS", "1800"))

# ---------------------------------------------------------------------------
# In-memory job store
#
# A "job" is one inbound completion request, normalized into a common shape,
# waiting for a human answer. Nothing here is persisted -- restart the
# process and in-flight jobs are gone (callers will just time out).
# ---------------------------------------------------------------------------

jobs = {}              # job_id -> job dict
jobs_lock = threading.Lock()
new_job_queue = Queue()  # push job_ids here so the console can long-poll efficiently


def new_job(fmt, model, system_text, thread_messages, tools, extra):
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "format": fmt,                 # "openai" | "anthropic"
        "created": time.time(),
        "status": "pending",           # pending | answered | timeout | error
        "model": model,
        "system": system_text,
        "messages": thread_messages,   # normalized: [{role, blocks:[...]}]
        "tools": tools,                # normalized: [{name, description, input_schema}]
        "extra": extra,                # raw bits worth showing (max_tokens, temperature, stream...)
        "event": threading.Event(),
        "response_blocks": None,       # filled in by the human
        "stop_reason": None,
        "error": None,
    }
    with jobs_lock:
        jobs[job_id] = job
    new_job_queue.put(job_id)
    return job


def wait_for_human(job, timeout=RESPONSE_TIMEOUT_SECONDS):
    answered = job["event"].wait(timeout=timeout)
    if not answered:
        with jobs_lock:
            job["status"] = "timeout"
        return False
    return True


# ---------------------------------------------------------------------------
# Normalization helpers -- turn OpenAI-shaped or Anthropic-shaped request
# bodies into one common internal representation for the console to render.
# ---------------------------------------------------------------------------

def normalize_openai_request(body):
    tools = []
    for t in body.get("tools") or []:
        fn = t.get("function", t)
        tools.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        })

    system_text = None
    thread_messages = []

    for m in body.get("messages", []):
        role = m.get("role")
        if role == "system":
            # OpenAI allows multiple system messages; concatenate them.
            piece = m.get("content") or ""
            system_text = (system_text + "\n\n" + piece) if system_text else piece
            continue

        blocks = []
        content = m.get("content")
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") in ("image_url", "input_image"):
                    blocks.append({"type": "image", "text": "[image omitted]"})

        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = fn.get("arguments")
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })

        if role == "tool":
            blocks.append({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else json.dumps(content),
            })
            role = "tool"

        thread_messages.append({"role": role, "blocks": blocks})

    extra = {
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "temperature": body.get("temperature"),
        "tool_choice": body.get("tool_choice"),
        "stream": bool(body.get("stream")),
    }
    return system_text, thread_messages, tools, extra


def normalize_anthropic_request(body):
    tools = []
    for t in body.get("tools") or []:
        tools.append({
            "name": t.get("name"),
            "description": t.get("description", ""),
            "input_schema": t.get("input_schema", {}),
        })

    system_field = body.get("system")
    if isinstance(system_field, list):
        system_text = "\n\n".join(b.get("text", "") for b in system_field if b.get("type") == "text")
    else:
        system_text = system_field

    thread_messages = []
    for m in body.get("messages", []):
        role = m.get("role")
        content = m.get("content")
        blocks = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "tool_use":
                    blocks.append({
                        "type": "tool_use",
                        "id": part.get("id", ""),
                        "name": part.get("name", ""),
                        "input": part.get("input", {}),
                    })
                elif ptype == "tool_result":
                    c = part.get("content", "")
                    if isinstance(c, list):
                        c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict))
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": part.get("tool_use_id", ""),
                        "content": c,
                        "is_error": part.get("is_error", False),
                    })
                elif ptype == "image":
                    blocks.append({"type": "image", "text": "[image omitted]"})
        thread_messages.append({"role": role, "blocks": blocks})

    extra = {
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "tool_choice": body.get("tool_choice"),
        "stream": bool(body.get("stream")),
    }
    return system_text, thread_messages, tools, extra


# ---------------------------------------------------------------------------
# Response builders -- turn the human's answer (a list of {type: text|tool_use}
# blocks) into an OpenAI or Anthropic completion response body.
# ---------------------------------------------------------------------------

def build_openai_response(job):
    blocks = job["response_blocks"]
    text_parts = [b["text"] for b in blocks if b["type"] == "text" and b.get("text")]
    tool_blocks = [b for b in blocks if b["type"] == "tool_use"]

    message = {"role": "assistant", "content": "\n".join(text_parts) or None}
    finish_reason = "stop"
    if tool_blocks:
        message["tool_calls"] = [
            {
                "id": tb.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tb["name"],
                    "arguments": json.dumps(tb.get("input", {})),
                },
            }
            for tb in tool_blocks
        ]
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{job['id']}",
        "object": "chat.completion",
        "created": int(job["created"]),
        "model": job["model"],
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_anthropic_response(job):
    blocks = job["response_blocks"]
    content = []
    has_tool_use = False
    for b in blocks:
        if b["type"] == "text" and b.get("text"):
            content.append({"type": "text", "text": b["text"]})
        elif b["type"] == "tool_use":
            has_tool_use = True
            content.append({
                "type": "tool_use",
                "id": b.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": b["name"],
                "input": b.get("input", {}),
            })

    return {
        "id": f"msg_{job['id']}",
        "type": "message",
        "role": "assistant",
        "model": job["model"],
        "content": content,
        "stop_reason": "tool_use" if has_tool_use else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Streaming emulation -- we don't get real token-by-token output from a
# human, so we chunk the finished answer into an SSE stream shaped like the
# real thing, which is enough for SDKs/clients that only support streaming.
# ---------------------------------------------------------------------------

def _chunks(s, n=24):
    for i in range(0, len(s), n):
        yield s[i:i + n]


def stream_openai(job):
    resp = build_openai_response(job)
    choice = resp["choices"][0]
    base = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"], "model": resp["model"]}

    def event(delta, finish_reason=None):
        payload = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish_reason}])
        return f"data: {json.dumps(payload)}\n\n"

    yield event({"role": "assistant", "content": ""})
    content = choice["message"].get("content") or ""
    for piece in _chunks(content):
        yield event({"content": piece})

    tool_calls = choice["message"].get("tool_calls")
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield event({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                          "function": {"name": tc["function"]["name"], "arguments": ""}}]})
            for piece in _chunks(tc["function"]["arguments"]):
                yield event({"tool_calls": [{"index": i, "function": {"arguments": piece}}]})

    yield event({}, finish_reason=choice["finish_reason"])
    yield "data: [DONE]\n\n"


def stream_anthropic(job):
    resp = build_anthropic_response(job)

    def sse(event_name, data):
        return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

    yield sse("message_start", {"type": "message_start", "message": {
        "id": resp["id"], "type": "message", "role": "assistant", "model": resp["model"],
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }})

    for idx, block in enumerate(resp["content"]):
        if block["type"] == "text":
            yield sse("content_block_start", {"type": "content_block_start", "index": idx,
                                                "content_block": {"type": "text", "text": ""}})
            for piece in _chunks(block["text"]):
                yield sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                    "delta": {"type": "text_delta", "text": piece}})
            yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        elif block["type"] == "tool_use":
            yield sse("content_block_start", {"type": "content_block_start", "index": idx,
                                                "content_block": {"type": "tool_use", "id": block["id"],
                                                                   "name": block["name"], "input": {}}})
            partial = json.dumps(block["input"])
            for piece in _chunks(partial):
                yield sse("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                    "delta": {"type": "input_json_delta", "partial_json": piece}})
            yield sse("content_block_stop", {"type": "content_block_stop", "index": idx})

    yield sse("message_delta", {"type": "message_delta",
                                  "delta": {"stop_reason": resp["stop_reason"], "stop_sequence": None},
                                  "usage": {"output_tokens": 0}})
    yield sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def check_api_key():
    if not SERVER_API_KEY:
        return True
    provided = request.headers.get("x-api-key") or ""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        provided = provided or auth[len("Bearer "):]
    return provided == SERVER_API_KEY


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions():
    if not check_api_key():
        return jsonify({"error": {"message": "invalid api key", "type": "authentication_error"}}), 401

    body = request.get_json(force=True, silent=True) or {}
    system_text, thread_messages, tools, extra = normalize_openai_request(body)
    job = new_job("openai", body.get("model", "human"), system_text, thread_messages, tools, extra)

    if not wait_for_human(job):
        return jsonify({"error": {"message": "no human answered in time", "type": "timeout"}}), 504

    if extra.get("stream"):
        return Response(stream_with_context(stream_openai(job)), mimetype="text/event-stream")
    return jsonify(build_openai_response(job))


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages():
    if not check_api_key():
        return jsonify({"type": "error", "error": {"type": "authentication_error", "message": "invalid api key"}}), 401

    body = request.get_json(force=True, silent=True) or {}
    system_text, thread_messages, tools, extra = normalize_anthropic_request(body)
    job = new_job("anthropic", body.get("model", "human"), system_text, thread_messages, tools, extra)

    if not wait_for_human(job):
        return jsonify({"type": "error", "error": {"type": "timeout", "message": "no human answered in time"}}), 504

    if extra.get("stream"):
        return Response(stream_with_context(stream_anthropic(job)), mimetype="text/event-stream")
    return jsonify(build_anthropic_response(job))


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({"object": "list", "data": [{"id": "human", "object": "model", "owned_by": "you"}]})


# ---------------------------------------------------------------------------
# Console (human operator UI)
# ---------------------------------------------------------------------------

@app.route("/")
def console():
    return render_template("console.html")


@app.route("/api/jobs/pending")
def api_jobs_pending():
    with jobs_lock:
        pending = [strip_job(j) for j in jobs.values() if j["status"] == "pending"]
    pending.sort(key=lambda j: j["created"])
    return jsonify(pending)


@app.route("/api/jobs/<job_id>")
def api_job_detail(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        return jsonify(strip_job(job, full=True))


@app.route("/api/jobs/<job_id>/respond", methods=["POST"])
def api_job_respond(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "not found"}), 404
        if job["status"] != "pending":
            return jsonify({"error": f"job already {job['status']}"}), 409

        body = request.get_json(force=True, silent=True) or {}
        blocks = body.get("blocks", [])
        for b in blocks:
            if b.get("type") not in ("text", "tool_use"):
                return jsonify({"error": f"unknown block type {b.get('type')!r}"}), 400
            if b.get("type") == "tool_use" and not b.get("name"):
                return jsonify({"error": "tool_use block missing name"}), 400

        job["response_blocks"] = blocks
        job["status"] = "answered"
        job["event"].set()
    return jsonify({"ok": True})


def strip_job(job, full=False):
    out = {
        "id": job["id"],
        "format": job["format"],
        "model": job["model"],
        "created": job["created"],
        "status": job["status"],
        "preview": _preview(job),
    }
    if full:
        out["system"] = job["system"]
        out["messages"] = job["messages"]
        out["tools"] = job["tools"]
        out["extra"] = job["extra"]
    return out


def _preview(job):
    for m in reversed(job["messages"]):
        if m["role"] == "user":
            for b in m["blocks"]:
                if b["type"] == "text" and b.get("text"):
                    return b["text"][:140]
    return "(no user text)"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8787"))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
