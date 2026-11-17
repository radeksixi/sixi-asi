# HTTP API

`sixi-server` (started by `./sixi serve`) speaks the OpenAI chat-completions protocol on `http://127.0.0.1:8089`
by default. With `--api-key KEY` (or `SIXI_API_KEY`) every endpoint except `/health` needs
`Authorization: Bearer KEY`.

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | chat completion, streamed or not |
| `GET /v1/models` | the loaded model |
| `GET /health` | status, model, context size, chat template, active requests |
| `GET /metrics` | request counters, last TTFT / TPOT, mean accepted draft length, prefix-cache hits |
| `POST /apply-template` | the prompt text the chat template renders for a request body, without running it |

## `POST /v1/chat/completions`

| Field | Meaning |
|---|---|
| `messages` | `[{"role": "system" \| "user" \| "assistant" \| "tool", "content": "..."}]`; assistant turns may carry `reasoning_content` and `tool_calls`, tool turns `tool_call_id` and `name` |
| images | `content` may be a list of parts: `{"type": "text", "text": "..."}` and `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`; see below |
| `prompt` | a raw prompt instead of `messages`: the chat template is not applied |
| `max_tokens` (`max_completion_tokens`) | limit on generated tokens, reasoning included; also capped by the context window |
| `temperature`, `top_k`, `top_p`, `seed` | sampling; `temperature` 0 is greedy (the default). Same seed, same answer. At `temperature` 1 without `top_p` the model now and then ends its turn at once; `top_p` 0.9 or a lower temperature avoids that |
| `repetition_penalty`, `presence_penalty` | applied in `--mode adaptive` only; the default strict-replay mode ignores them |
| `stream` | `true`: server-sent events, ending with `data: [DONE]` |
| `stop` | a string or a list of strings; the answer ends before the first match |
| `enable_thinking` | `false` skips the reasoning phase (also accepted inside `chat_template_kwargs`) |
| `reasoning_effort`, `preserve_thinking` | passed to the Qwen3.8 chat template |
| `reasoning_budget_tokens` (`reasoning_budget`) | when the reasoning reaches it, the server closes the reasoning block and the model answers |
| `answer_budget_tokens` | limit on answer tokens alone |
| `response_format` | structured answer, see below |
| `tools` | OpenAI tool definitions; calls come back validated against their schema as `tool_calls` |
| `priority` | `"interactive"` (default) or `"background"`; see below |
| `n` | several samples of one prompt with seeds `seed .. seed + n - 1`, decoded as parallel streams (`n` may not exceed `--max-concurrent`, which is at least `--parallel`; not with `stream`) |
| `vote` | with `n`: `true` or `{"answer_regex": "...", "stop_on_agreement": true}`; the response carries the majority answer |
| `stop_on_loop`, `stop_on_reasoning_loop` | what to do when the output starts repeating itself |
| `include_rendered_prompt` | debugging: echo the rendered prompt |

The response is the OpenAI shape (`choices[0].message.content`, `finish_reason`, `usage`), plus
`choices[0].message.reasoning_content` and a `metrics` object: `ttft_ms`, `tpot_ms`, `prefill_tok_per_s`,
`decode_tok_per_s`, `mean_acceptance_length` (tokens accepted per speculative step), `prefill_ms`, `snapshot_ms`,
`reasoning_forced_closed`, `constrained`, `constraint_rows_changed`, `constraint_drafts_refused`.

`finish_reason`: `stop`, `length`, `tool_calls`, `reasoning_budget_exceeded`, `answer_budget_exceeded`,
`loop_detected`, `cancelled`.

### Structured answers: `response_format`

```json
{"type": "json_object"}
{"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": { ... }}}
```

The answer (`message.content`) is decoded under a grammar: with `json_object` it is exactly one JSON object, with
`json_schema` a document that matches the schema. The model can only choose tokens the grammar allows, so a
finished answer always parses, and generation stops at the end of the document. This holds with speculative drafts
and with parallel streams: a drafted token that breaks the grammar is rejected when the target verifies it, and
each request has its own grammar state. Reasoning is not constrained; the grammar starts after it.

- Describe the format in the prompt as well (paste the schema, or name the keys). The grammar guarantees the
  form; the prompt is what makes the model want those keys and values. A model that has not been told the schema
  is overruled at every key, which costs quality and speed, because fewer drafted tokens survive verification.
- `temperature` 0 picks the most likely allowed token at every position.
- `max_tokens` reached inside the document: `finish_reason` is `length` and the content is incomplete. A document
  that closes on the last allowed token is `stop`.
- Supported schema keywords: `type` (also a list), `properties`, `required`, `additionalProperties`, `items`,
  `prefixItems`, `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `allOf`, `$ref` with `$defs`, `minimum` /
  `maximum` / `exclusiveMinimum` / `exclusiveMaximum` for integers, `minLength`, `maxLength`, `pattern` (anchored
  with `^` and `$`), `format` (`date`, `time`, `date-time`, `uuid`). Annotations (`title`, `description`,
  `default`, `examples`) are accepted and have no effect.
- Anything else (`oneOf`, `not`, `if`, `uniqueItems`, `multipleOf`, bounds on non-integers, other formats, unknown
  keywords) is refused with HTTP 400 and the place in the schema, for example
  `JSON schema: #/properties/a: keyword 'uniqueItems' is not supported`. Nothing is ignored silently.
- `response_format` together with `tools` is refused with HTTP 400 for now.
- `metrics.constraint_rows_changed` is 0 when the model wrote the document unaided and above 0 when the grammar
  overruled its choice at least once. The grammar guarantees the syntax, not the meaning: text that is legal inside
  a string stays there even when the model meant it as a closing (a typographic quote followed by a brace, for
  instance), so an answer with a non-zero count is worth a second look.

### Images

With a projector loaded (`--mmproj`; `./sixi` adds the one of the model's registry entry) a message's `content` can
mix text and images. An image is an `image_url` part with a `data:` URI (base64 of a jpg, png, bmp or gif file);
remote URLs are not fetched. Parts are joined in order without separators, as in llama.cpp's server, and the same
loaded model answers: there is no second model.

- An image costs rows of context and loading time in proportion to its size: about 300 rows and 0.6 s for a
  small picture, over 2000 rows and 4 s for a large one. `--image-max-tokens N` bounds the rows per image (512
  gives about 1.2 s).
- The model sees a picture at one row per 32 x 32 pixels, so a small picture is a coarse one: a 480 x 480 picture
  is 225 rows, and a character 50 pixels wide is covered by two or three of them. To read text or numbers, give
  the picture more rows: `--image-min-tokens N` enlarges every picture to at least N rows. Measured on fifteen
  480 x 480 pictures of a six-digit number (`scripts/image_digits_check.py`): 5 read at the picture's own 225
  rows, 12 with `--image-min-tokens 512`, 14 with `--image-min-tokens 1024 --image-max-tokens 1024` (1.4 s to
  load a picture instead of 0.4 s). llama.cpp's server reads the same pictures the same way at each setting.
- An image is part of the prompt prefix like text: the next turn of the same conversation does not load it again
  (first token after about 0.1 s), and the same picture sent by another request is recognised by its content.
- Requests with images run in parallel streams with requests without; drafts continue after an image.
- `usage.prompt_tokens` counts the image's rows.

### Prompt reuse

The server keeps what it computed for the prompts it has read; a request whose prompt begins like one of them
reads only the rest. `usage.prompt_cached_tokens` says how many tokens of the prompt were not read again.

- The next turn of a conversation continues where the last one ended.
- The end of a long system prompt (512 tokens or more) is kept on its own: every later request with the same
  system prompt starts from there, whatever follows it and whichever conversation it belongs to. Put what stays
  the same (the role, the rules, tool definitions) into the system message and what changes into the user message.
- `--device-slots N` is how many such states are kept: 4, or 8 from `--parallel 4`. All but two of them can hold
  system prompts, so the default serves six different ones; an agent with more roles than that should raise it
  (up to 32). A slot costs the memory of one more stream, about 0.9 GB at `--ctx 8192`, and no speed: a single
  stream decodes as fast with 16 of them as with 4 (`--device-slots 4` is the setting of 1.0.0-rc2).

Measured on a Jetson AGX Thor (`scripts/roles_bench.py`: sessions of three roles, each role with its own
1,500-token system prompt, a growing conversation and a short JSON answer; `--parallel 8`):

| Sessions at once | 1 | 5 | 8 |
|---|---|---|---|
| Seconds a call, 1.0.0-rc2 | 3.4 | 15.7 | 26.8 |
| Seconds a call, from 1.0.0-rc3 | 1.3 | 4.5 | 6.6 |
| Calls a second in all, rc2 -> rc3 | 0.29 -> 0.59 | 0.36 -> 1.17 | 0.34 -> 1.24 |
| Prompt tokens reused, rc2 -> rc3 | 0% -> 86% | 7% -> 86% | 5% -> 86% |

Calls like these are almost all prompt (1,700 tokens in, 20 out), and prompt reading does not get faster with
more requests at once: what decides their cost is how much of the prompt is still there.

At temperature 0 a call gives the same answer whether its prompt was read or reused, as long as calls come one at
a time (18 of 18 in the test above). Requests that run at the same time share passes, and a few percent of their
answers differ in wording from one run to the next with or without reuse (7 to 11 of 90).

### Priority

Work that nobody is waiting for can say so: `"priority": "background"`. Such a request

- loads its prompt in small chunks and hands the device to other requests between them,
- lets interactive requests go first wherever requests queue,
- waits for a free slot (up to two minutes) instead of failing with 429, and never takes the slots set aside
  with `--reserve-interactive N`.

Measured on a Jetson AGX Thor with `--parallel 4 --reserve-interactive 1`: an interactive turn gets its first
token after 0.14 s alone and after a median of 0.26 s (0.7 s at worst) while four background requests load
4k-token prompts; without the priority field the same turn waits about 1.8 s behind one such load.

### Errors

Errors are JSON: `{"error": {"code": "...", "message": "..."}}`.

| Status | When |
|---|---|
| 400 | malformed body, a field of the wrong type, an unsupported schema, an invalid tool call |
| 401 | missing or wrong API key |
| 429 | more requests than `--max-concurrent` (at least `--parallel`) |
| 503 | the model was not admitted at start (memory) |
| 500 | anything else, with the backend's message |

In a streamed response an error that happens after the stream has started arrives as a
`data: {"error": {...}}` event.
