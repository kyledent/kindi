# Using Kindi with agents and chat clients

Kindi is an OpenAI-compatible endpoint. Use these values:

- Base URL: `http://<host>:8093/v1`.
- Model: `bonsai2-2b` (or your `model_name`).
- API key: your `api_key`. If you did not set an `api_key`, any string works.

For tool calls, Kindi uses the standard `tools` / `tool_calls` fields.

## What to set in any client

1. **Context window: 24,576 tokens** (or your `ctx`). Clients that summarise old turns ("compaction") should do it
   before that.
2. **A long request timeout - 30 to 60 minutes.** Kindi reads new prompt text at about 25-33 tokens/s. The first request
   from an agent that writes code often has 5,000+ tokens of system prompt and tool definitions. The first read of
   that text takes about 3 minutes. The cache then saves it (see the README), so later turns and later sessions are
   fast. Some clients stop after 60-120 s and send the request again. This makes the problem worse: the retry waits
   behind the request that Kindi still reads.
3. **One retry at most**, and no parallel requests: Kindi serves one request at a time.
4. **A short system prompt and few tools.** Every tool schema is prompt text. Keep the tool list fixed between calls.
   A different set of tools per call changes the start of the prompt, and then the cache cannot help.
5. **Temperature 0.1-0.7.** Greedy (`0`) is reproducible; the model publisher recommends 0.7 / top_p 0.8 / top_k 20.

Watch a long first read with `curl -s localhost:8093/health` - the `busy` field shows `prompt_read` progress.

## Examples

**Any OpenAI SDK**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8093/v1", api_key="none", timeout=3600, max_retries=1)
```

**Agents for code that have an OpenAI-compatible provider setting** (for example pi, or similar terminal agents):

1. Add a provider with these values:
   - Base URL: `http://localhost:8093/v1`.
   - API style: "openai-completions" / "chat completions".
   - Model: `bonsai2-2b`.
   - Context window: `24576`.
   - Max output: `2048`.
2. Increase the request timeout of the agent, as above.

A shorter custom system prompt helps the first turn.

**Chat UIs** (Open WebUI, LibreChat and similar): add an OpenAI connection with the base URL above. Turn off extra
background requests, such as title, tag and follow-up generation (in Open WebUI: Admin Panel, Settings, Interface).
Each one is a new prompt that waits for the GPU.

## Thinking

Two kinds of request turn thinking on. Clients of the Qwen kind send `chat_template_kwargs.enable_thinking`. Clients of
the OpenAI kind send `reasoning_effort` alone. An explicit `enable_thinking` wins over an effort.

| `reasoning_effort` | Thinking | 27B template effort |
|---|---|---|
| `none` | off | |
| `minimal`, `low` | on | `low` |
| `medium` | on | `medium` |
| `high`, `xhigh`, `max` | on | `xhigh` |

The 2B, 4B and 9B templates have no effort, so there it only selects a thinking budget. Kindi also reads
`reasoning.effort`. The budget comes from the request (`thinking_budget_tokens`), then the server's `thinking_budgets`,
then its `thinking_budget`: see [CONFIGURATION.md](CONFIGURATION.md). Set a budget for the 27B, which can think for
thousands of tokens at each step. All of that thinking stays in the context of an agent.

**pi** has both modes. In `models.json`, this sends the level as `reasoning_effort`, and the server's
`thinking_budgets` sets each level's budget:

```json
{
  "compat": {"supportsDeveloperRole": false, "supportsReasoningEffort": true},
  "models": [{
    "id": "bonsai2-27b", "reasoning": true,
    "thinkingLevelMap": {"minimal": null, "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}
  }]
}
```

pi offers `xhigh` only when the map names it. `high` and `xhigh` then share the 27B's `xhigh` budget.

## When a prompt is too long

A prompt that does not fit the context gets an HTTP 400, also for a stream, because Kindi counts the prompt before it
sends anything:

```json
{"error": {"message": "This model's maximum context length is 24576 tokens. However, your messages resulted in 25000 tokens. Please reduce the length of the messages.",
           "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}
```

This is OpenAI's error, so clients that handle it know what it means. We read how these clients detect it, in their
source code on 2026-10-30:

| Client | What it looks for |
|---|---|
| pi | `maximum context length is N tokens`, `reduce the length of the messages` |
| opencode | `maximum context length is N tokens`, `context_length_exceeded` |
| Cline | code `context_length_exceeded`; status 400 with `context length` or `maximum context` |
| Roo Code | status 400 with `context length` or `maximum context` |
| LiteLLM (and clients built on it) | `this model's maximum context length is` |
| Continue CLI | `maximum context length`, `context_length_exceeded` (it does not retry these) |

Only pi was tested against a running server: on this error it compacts the conversation and retries by itself. One-shot
`pi -p` does not wait for the retry, so use the interactive mode or RPC mode. What the other clients do after they
detect the error depends on the client.

**Compaction takes time on a slow card.** A client's summary request is a new prompt, so Kindi reads all the text to
be summarised, at 30-60 tokens/s on a TITAN X Maxwell, then reads the summary and the kept messages again. Leave a
margin that holds a whole run of tool calls, because pi, for one, checks the context size only when a run ends. With
pi, for a 100K context, in `settings.json`:

```json
{"compaction": {"enabled": true, "reserveTokens": 24576, "keepRecentTokens": 20000}}
```

## What to expect from a 2B model

For its size, the model does short shell and Python tasks, file edits, factual questions, summaries and tool calls
well. It also makes the usual mistakes of a small model:

- Wrong tool arguments. Agents usually recover when they send the error back to the model.
- Repetition in long answers.
- Weak multi-step reasoning.

Give it small, concrete steps.
