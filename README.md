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
| array of enum values | multiselect (checkboxes), list, json |
| array of strings/numbers | list (one per line), json |
| object / other | json |

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

The gear icon in the UI changes the AI model, and optionally a separate model for reading tools (blank means the same model). These are saved in `data/settings.json`. When forwarding, the client's model name is replaced by the AI model; clear the field to pass the client's name through.

The prompt that turns tool definitions into controls is `controls_prompt.txt`; editing it invalidates the cache automatically.