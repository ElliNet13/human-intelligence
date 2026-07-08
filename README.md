# human-llm-server

A server that speaks the **OpenAI Chat Completions API** and the **Anthropic
Messages API** on the wire, but the "model" is a human. Point any OpenAI or
Anthropic SDK/client at it, and every request lands in a queue on a console
page where a person sees exactly what a model would see — system prompt,
full message history, tool definitions — and replies or issues tool calls
themselves. The answer gets translated back into a real
`chat.completion` / Anthropic `message` response, streaming included.

## Run it

```bash
pip install -r requirements.txt
python app.py
```

- Console (for the human): http://localhost:8787/
- OpenAI-compatible endpoint: `POST http://localhost:8787/v1/chat/completions`
- Anthropic-compatible endpoint: `POST http://localhost:8787/v1/messages`
- `GET /v1/models` — returns a single dummy `"human"` model.

Point a client at it, e.g. with the OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8787/v1", api_key="anything")
resp = client.chat.completions.create(model="human", messages=[{"role": "user", "content": "hi"}])
```

or the Anthropic SDK:

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8787", api_key="anything")
resp = client.messages.create(model="human", max_tokens=512, messages=[{"role": "user", "content": "hi"}])
```

Both requests will sit and wait until someone opens the console, reads the
request, and sends a reply. Nothing is persisted — restart the server and
any in-flight requests are gone (callers will just time out).

## How answering works

Open the console, click a pending "ticket" in the left column. You'll see:

- the system prompt (if any)
- the full message thread, including prior tool calls and tool results
- the tool definitions (name, description, JSON schema) available for this
  request, via the "Tools" toggle

Write a text reply, and/or click **+ Add tool call** to pick a tool and fill
in its JSON arguments — you can add as many as you like, in any order, just
like a model choosing to call several tools in one turn. Hit **Send reply**
and the waiting client gets back a properly-shaped response:
`finish_reason: "tool_calls"` / `stop_reason: "tool_use"` if you issued any
tool calls, `"stop"` / `"end_turn"` otherwise.

Note that — same as a real model — the human doesn't execute tools. You only
decide *that* a tool should be called and with what arguments; the actual
execution and the tool's result come back from the calling application in
its next request, appearing in the thread as a `tool_result` block.

## Config

- `SERVER_API_KEY` — if set, callers must send this value as `x-api-key`
  (Anthropic-style) or `Authorization: Bearer <key>` (OpenAI-style). Unset
  by default, since this is meant for local/dev use.
- `RESPONSE_TIMEOUT_SECONDS` — how long a caller blocks waiting for a human
  before getting a 504 (default 1800 = 30 min).
- `PORT` — default 8787.

## Limitations

- No persistence — it's a single in-memory queue, run one process.
- "Streaming" is emulated: the human's finished answer is chunked into a
  real SSE stream shape (SSE event types match the real APIs), but it isn't
  token-by-token from a live model.
- No image rendering in the console yet (image blocks show as a placeholder).
- This is a devtime/novelty tool, not hardened for the open internet — put
  it behind your own auth/reverse proxy if it's reachable beyond localhost.
