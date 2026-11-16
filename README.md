# Human LLM

An OpenAI-compatible endpoint where **you** decide, per request, whether a real
AI answers or you do.

- API: `http://localhost:8787/v1` (`POST /v1/chat/completions`, `GET /v1/models`; streaming and non-streaming)
- Operator UI: `http://localhost:8787/`

## Run

```bash
pip install -r requirements.txt
cp .env.example .env      # then set OLLAMA_URL
python server.py
```

It listens on `0.0.0.0:8787`. The operator UI has no login, so keep the port off the open internet or put it behind something like Cloudflare Access.

## How it works

1. A client sends a chat completion. It shows up in the queue in the UI and the client waits.
2. You pick one:
   - **Forward to AI**: the request goes to your Ollama server and the answer, streamed or not, is passed straight back.
   - **Respond yourself**: type a reply and/or build tool calls, then Send.
3. For tool calls, the AI reads the request's tool definitions once and writes a title, summary, labels, hints and a control type for each parameter. The UI turns them into forms. Types, required flags, enums and number ranges always come from the JSON schema; an AI choice that doesn't fit the parameter is ignored. Results are cached per tool in `data/controls_cache.json`, so a tool you have seen before costs nothing. If the AI is unavailable, forms built from the schema alone are used.

### Control types

| Parameter | Controls the AI can pick |
|---|---|
| string | text (one line), textarea (multi-line), password, combo (text plus suggestions) |
| enum | select (dropdown), radio |
| integer / number | number, slider (needs a min and max) |
| boolean | checkbox, select, radio |
| array (strings, numbers, objects, nested arrays) | array: rows you add, remove and reorder with ↑ ↓ ×, each row built from the item schema (an object becomes a small form) |
| array of enum values | multiselect (checkboxes), array |
| array of strings/numbers | array, list (one per line), json |
| object with known properties | object (nested form), json |
| free-form object / other | json |

`$ref`, `$defs` and "X or null" unions are resolved. Empty optional rows and fields are left out of the call; an incomplete row is reported by place, e.g. `Edits → item 2 → New is required`.

While the AI reads the tools, a progress bar and "Reading tools… 3 of 8" show how far it is. The reply is streamed, so each tool's form upgrades the moment the AI finishes it, and tools already cached count as read.

## Chats

Clients resend the whole history with every request, so the queue shows one entry per chat, not one per request. Chats are matched on what the user and the tools said (user and tool messages). System prompts and assistant messages are ignored, because clients change and rewrite them between turns. A request joins an earlier chat when it continues it, and these all count:

- the next turn, a regenerated answer, an edited message, or an identical request sent again while the first is still waiting (the client has moved on)
- a client that dropped its oldest messages to fit its context window (at least two messages must still overlap)
- a client that rewound the chat and resent only the start of it
- a question the client had wrapped in a long template (for example retrieval context) that it later resends plain

A request still waiting for you is never hidden behind a newer finished one in the same chat. Chats are remembered after "Clear finished", so one can be continued afterwards.

If the client sends a chat id, that is used instead of guessing: the headers `X-OpenWebUI-Chat-Id`, `X-Chat-Id`, `X-Conversation-Id` or `X-Thread-Id`, or a `chat_id`, `conversation_id` or `thread_id` field in the request body. In Open WebUI, set `ENABLE_FORWARD_USER_INFO_HEADERS=true` so it sends its chat id. Without an id, two brand-new chats that start with exactly the same message (for example "hi") look like one chat.

When a request almost matched an earlier chat but diverged (the first messages agree, then one differs), the chat's header and the server log say where, for example `matches up to message 3: it then has user "..." but this has user "..."`. That is the quickest way to see what a client changes in its history.

If a chat still does not merge, open its request: a **Why is this a separate chat?** panel compares it message by message with the closest other chat (same / close / differs) and lets you **Merge** it into any chat by hand; later turns then follow the merged chat. Matching also tolerates small edits to a message (a date a client rewrote), a whole transcript stuffed into one message, and a chat id that only starts arriving after the first turns. The server log prints each request's shape, e.g. `shape=[u24 u40]` (role letter and text length per user/tool message).

The UI follows a chat to its newest turn, carrying over a reply you were still typing. When a different chat arrives, your current view is left alone.

## Background tasks

Clients such as Open WebUI send their own prompts in the background (chat titles, tags, follow-up suggestions, search queries, autocomplete). These are recognised, answered by the AI automatically, and listed under **Auto-handled** in the sidebar instead of the queue, so they never wait for you or count as pending. Open an entry to see what was asked and what the AI answered. The checkbox "Auto-handle titles, tags & follow-ups" turns this off.

Finished background tasks remove themselves from the list after 60 seconds (`AUTO_CLEAR_SECONDS`; one that failed stays ten times longer so you can see the error). "Clear finished" removes them straight away.

Recognition uses `utility_patterns.txt`, one `label :: regular expression` per line, matched against the start of the last user message. It ships with the Open WebUI prompts; add lines for other clients (the file is read at startup). Your normal Open WebUI answer prompts are not matched.

## Live updates

The UI opens one WebSocket (`/ws`) and the server pushes every change: new requests, status changes, tool controls as they finish generating, removals and settings. Nothing polls. On a dropped connection the UI shows "reconnecting…", retries with backoff, and resyncs from a fresh snapshot, so requests that arrived while it was offline still appear. Browsers don't apply CORS to WebSockets, so the server only accepts connections from pages served by itself; if you reach the UI through a proxy that changes the Host header, list the page's origin in `ALLOWED_ORIGINS`.

## Configuration

`.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | none | Your Ollama server, e.g. `http://localhost:11434` (`/v1` is added automatically) |
| `OLLAMA_API_KEY` | empty | Only if something in front of Ollama wants a bearer token |
| `OLLAMA_MODEL` | `mimo_v2.6:latest` | Initial AI model |
| `PORT` | `8787` | Listen port |
| `HOST` | `0.0.0.0` | Listen address |
| `API_KEY` | empty | If set, clients must send `Authorization: Bearer <key>` to `/v1/*` |
| `DATA_DIR` | `./data` | Where settings and the control cache are stored |
| `ALLOWED_ORIGINS` | empty | Extra page origins allowed to open the UI WebSocket, comma separated |
| `UTILITY_PATTERNS_FILE` | `./utility_patterns.txt` | Patterns for recognising background tasks |
| `AUTO_CLEAR_SECONDS` | `60` | How long a finished background task stays listed; `0` keeps them until cleared |

The gear icon in the UI changes the AI model, and optionally a separate model for reading tools (blank means the same model). These are saved in `data/settings.json`. When forwarding, the client's model name is replaced by the AI model; clear the field to pass the client's name through.

The prompt that turns tool definitions into controls is `controls_prompt.txt`; editing it invalidates the cache automatically.