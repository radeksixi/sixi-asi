# Changelog

Versions of the Sixi-ASI Engine, newest first. Numbers are from NVIDIA Jetson AGX Thor; how each was measured is in
[docs/RESULTS.md](docs/RESULTS.md).

## 1.0.0

The three release candidates ran under two agents (one that talks and listens, one that asks for many samples of
an answer) and next to another engine's published recipe. What that brought:

| Area | Change |
|---|---|
| Prompt reuse | requests for the same prompt that arrive together load it once: the first loads it, the others start from its state. Four requests for one 4,100-token prompt are answered after 8.0 s instead of 31.2 s. Until now only `n` shared a prompt load, and `n` cannot stream |
| Prompt reuse | `"cache_control": {"type": "ephemeral"}` on a content part keeps what was computed up to its end, like the end of a long system prompt: a second question about a 3,000-token document inside a user message, asked after ten other prompts, loads 20 tokens instead of all of them (0.5 s instead of 5 s). A client can mark a document, a list of tools, or the message it may come back to |
| API | `stream_options.include_usage`; `usage.prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` next to the engine's own names; `thinking_budget` as a name of the reasoning budget; `POST /tokenize`; a stream that has nothing to send for 15 s (a long prompt is loading) gets a comment line, so that a proxy on the way keeps the connection (`--sse-keepalive`) |
| Checks | `"draft": false` decodes one request without drafts, and `scripts/exact_check.py` asks every prompt both ways on the running server: 9 of 12 answers identical, 3.3x faster with drafts. `scripts/streams_bench.py` measures many requests at once with a prompt of its own for every request and reports decoding and wall-clock figures apart |
| Speed | a request that is alone on a server started with `--parallel 8` writes 55 instead of 48.5 tok/s: the rollback ring stays 16 deep there (0.9 GB) |
| Flash-Next | prompts load 1.8 times faster: 18,500 tokens in 31 s instead of 56 s, 4,400 in 7.6 s instead of 13.9 s (one pass of 2,048 rows: 2.5 s instead of 4.6 s). On the Thor the expert layers of a wide pass had been multiplied one expert at a time |
| Bursts | `--merge-loads` (off by default): the prompts of requests that arrive at the same moment load in shared passes, about a fifth less time for such a burst |
| Memory | with `--draft-model` the model's built-in draft head is not loaded (`--no-mtp` as a second name of `--no-mtp-load`) |

## 1.0.0-rc3

Found in two days of use by an agent that talks and listens (about 4,500 chat requests and 2,600 speech requests,
none failed):

| Area | Change |
|---|---|
| Fix | a request with `temperature` above 0 for a prompt that had been answered at temperature 0 before began its answer with a stray character; with a prompt that demands JSON the answer was that character alone. Sampling now starts from the right numbers in every case (`scripts/sampling_check.py`) |
| Agents with roles | the end of a long system prompt is kept and reused by every later call with it: three roles with 1,500-token system prompts take 1.3 s a call instead of 3.4 s alone, 4.5 s instead of 15.7 s with five sessions at once, 6.6 s instead of 26.8 s with eight (`scripts/roles_bench.py`). The server keeps 8 prompt states instead of 4 from `--parallel 4` (`--device-slots`) |
| Logs | the engine's log no longer fills with the library's debug lines (2.1 million lines in two days of use), and a log over 64 MB is set aside at the next start; the speech service's log has timings and character counts only, no recognised or spoken text (`--log-text` turns it on for debugging) |
| Speech | short answers are heard in their language: with `language=pl`, `de`, `it` or `fr` a clip of up to 2 s is recognised after a phrase in that language (a Polish "nie" used to come back as "Yeah."): 201 instead of 80 of 300 one-word clips of Polish speakers right, 21 instead of 153 wrong; German 126 instead of 41 of 280, French 146 instead of 76 of 317, Italian 76 instead of 38 of 161 (`scripts/speech_short_check.py`). `certain=1` in the request returns a short clip's text only when it was heard the same way twice. Silent audio returns an empty text. `"deterministic": true` in a synthesis request: the same audio for the same text |
| Images | documented: a small picture needs `--image-min-tokens` for its text to be read (14 of 15 six-digit numbers with 1024 rows, 5 of 15 at the default; `scripts/image_digits_check.py`) |
| Checks | `scripts/speed_set.py` (single-stream speed over 60 prompts) and the four scripts above are in the archive |

## 1.0.0-rc2

| Area | Change |
|---|---|
| Fix | a streamed answer with a character split across two tokens (an emoji, for instance) could stop the server; fixed |
| Images | a message can carry pictures (`image_url` parts); the same loaded Qwen3.8-27B answers, in parallel with text requests, and a picture stays in the conversation's memory |
| Priority | `"priority": "background"` and `--reserve-interactive N`: batch work no longer delays the turn a person is waiting for (first token 0.26 s instead of 1.8 s next to a long prompt being loaded) |
| Speed | 2 and 4 requests at once 64 -> 70 and 98 -> 108 tok/s in total; long runs 3-7% faster; quality gates unchanged |
| Speech | `speech/sixi-speech`: speech recognition and synthesis on the CPU with the OpenAI audio API, for a talking agent ([docs/VOICE-AGENT.md](docs/VOICE-AGENT.md)) |

## 1.0.0-rc1

| Area | Change |
|---|---|
| Download | The engine comes as one archive for the Jetson AGX Thor: nothing to build |
| One command | `./sixi` checks the machine, downloads a model by name (resumable, checksummed), chats in the terminal, runs one prompt, serves the API |
| Structured answers | `response_format` `json_object` and `json_schema`: a finished answer always parses and nothing follows it, also with drafts and parallel streams. Unsupported schema keywords are refused with HTTP 400 and the place in the schema |
| Background service | `./sixi service install`: the server as a systemd user service, like a local model service ([docs/SERVICE.md](docs/SERVICE.md)) |
| Backend | llama.cpp v1.0.0 |
| Qwen3.8-Flash-Next | drafts 11% faster on code and 8% in conversation (52.8 and 44.6 tok/s); the first request after a start no longer waits for the model's tables |
| Many streams | 4 streams 90 -> 98 tok/s in total on the default weights; 8-32 streams as in 0.11 |
| Versions | one version for all programs (`--version`), reported in the server banner |

## 0.11

| Area | Effect |
|---|---|
| Many streams | 32 streams 129 -> 169 tok/s in total, 16 streams 129 -> 154 |
| Memory | context memory for 32 streams 70.2 -> 27.1 GB |
| Long prompts under load | 16 agents: a follow-up waits under 3 s instead of up to 442 s |
| Memory admission | the server refuses a configuration that does not fit and says what would |
| Default weights | Unsloth Dynamic v3.1 UD-Q4_K_XL: KL divergence to BF16 0.008 instead of 0.025 for 2-7% of the speed |
| Voting | `"vote": true` with `n` samples: the majority answer, and the remaining samples stop once it is decided |

## 0.10

| Area | Effect |
|---|---|
| Parallel streams | `--parallel N`: 4 streams 94.2 tok/s in total, 24-26 tok/s each |
| Prompt loading | 571-711 prompt tokens per second instead of 299-337 |
| Prompt loading under load | a document being loaded no longer stalls other chats |
| First token | short prompts: 327 -> about 210 ms |
