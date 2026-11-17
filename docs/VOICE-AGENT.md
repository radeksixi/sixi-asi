# A talking agent on one Jetson AGX Thor

How to run a conversational agent (several language-model threads, vision, speech in and out) on one Thor with
the Sixi-ASI Engine. Two local services, both with the OpenAI API:

| Service | What | Where it runs | Address |
|---|---|---|---|
| `sixi-asi` | Qwen3.8-27B: text, images, JSON answers, tool calls | GPU | `http://127.0.0.1:8089/v1` |
| `sixi-speech` | speech recognition (Parakeet TDT v3; Whisper turbo for other languages) and speech synthesis (Piper voices) | CPU | `http://127.0.0.1:8088/v1` |

The language model is bound by the GPU's memory bandwidth, so everything else is kept off the GPU: speech runs
on CPU cores, and the vision model is the language model itself (its projector), not a second model.

## Start

```bash
sixi-asi service install --parallel 8 --ctx 8192 -- --reserve-interactive 1 --image-max-tokens 512
sixi-asi service start
sixi-speech setup                 # once: Python environment, runtime, models (about 1.8 GB)
sixi-speech service install
sixi-speech service start
sixi-asi service status; sixi-speech service status
```

`--parallel 8`: up to eight requests share the model. `--ctx 8192`: tokens of context per request (raise it for
long documents; memory grows with it). `--reserve-interactive 1`: one slot that background work never takes.
`--image-max-tokens 512`: an image loads in about 1.2 s.

## How many threads

Total speed grows with the number of simultaneous requests, speed per request falls:

| Requests at once | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Tokens per second, each | 44 | 33 | 26 | 15 | 9.5 |
| Tokens per second, in total | 44 | 66 | 102 | 122 | 148 |

Speech needs 3 to 5 tokens per second, but a turn feels fast when its first sentence is ready early. Keep 4 to 6
requests active at a time; 8 is the limit worth configuring for a conversation.

## Agents with several roles

A turn of an agent is often several calls, each with its own long system prompt (a reader, a writer, a coder)
and a short answer. Such calls are almost all prompt, and their cost is how much of the prompt the engine still
has. It keeps the end of every long system prompt (512 tokens or more) and starts later calls from there:

| Sessions at once, three roles with 1,500-token system prompts | 1 | 5 | 8 |
|---|---|---|---|
| Seconds a call | 1.3 | 4.5 | 6.6 |
| Calls a second in all | 0.59 | 1.17 | 1.24 |

- Keep the role's text in the system message and identical from call to call; put the conversation and the
  instruction of the moment into the user message. `usage.prompt_cached_tokens` shows what was reused.
- Up to six different system prompts are kept with `--parallel 4` or more; for more roles add
  `--device-slots N` (two more than the number of system prompts) after the `--` of `service install`.
- More than five to eight calls in flight add little: prompt reading does not get faster with parallel requests.
- A call that arrives while `--parallel` calls are running is answered 429: keep the calls in flight at or below
  that number, or send the call again a moment later (ten callers on eight slots: 7.4 s a call, the same
  throughput).
- `"priority": "background"` on every call changes nothing between them; it only orders them behind the calls
  that do not carry it.

## The turn that a person waits for

Send it streamed, without the reasoning phase, and start speaking at the first sentence:

```python
from openai import OpenAI
llm = OpenAI(base_url="http://127.0.0.1:8089/v1", api_key="none")
stream = llm.chat.completions.create(
    model="qwen3.8-27b", messages=history, stream=True, max_tokens=200,
    extra_body={"enable_thinking": False})
for chunk in stream:
    piece = chunk.choices[0].delta.content or ""
    ...   # collect until a sentence ends, hand the sentence to the speech service
```

- **Barge-in**: close the stream (drop the connection). The engine stops generating at once
  (`finish_reason: cancelled`) and the slot is free for the next turn.
- **Conversation memory**: send the whole history every turn. What was already sent is not computed again; a
  turn that continues a conversation starts after 0.1 to 0.3 s however long the history is.
- **Structured decisions** (intent, slots, tool arguments): `response_format` with a JSON schema; put the schema
  in the prompt too.

## Everything else is background

Threads that plan, summarise, watch or load documents should mark their requests:

```python
extra_body={"priority": "background", "enable_thinking": False}
```

A background request loads its prompt in small pieces, lets the interactive turn go first, and waits for a slot
instead of failing. Measured with `--parallel 4 --reserve-interactive 1` (first token of the interactive turn):

| Meanwhile | Median | Worst |
|---|---|---|
| nothing | 0.14 s | 0.16 s |
| 4 background requests generating | 0.26 s | 1.2 s |
| 4 background requests loading 4k-token prompts | 0.26 s | 0.7 s |
| 2 generating and 2 loading | 0.26 s | 0.8 s |
| one request loading a 4k-token prompt **without** the priority field | 1.8 s | 2.2 s |

## Vision

An image goes into a message as an `image_url` part with a `data:` URI:

```python
import base64
frame = base64.b64encode(open("frame.jpg", "rb").read()).decode()
messages = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
    {"type": "text", "text": "What is in front of you?"}]}]
```

- Image requests are ordinary requests: they run next to the conversations in the same model, at the same
  speed per request. Two agents that look at pictures and four that talk are six requests.
- A new image takes the GPU alone while it loads (0.6 s for about 300 rows, 1.2 s for 512), and the other
  requests pause for that time. Send a frame when it is needed, not every second.
- The same image in the following turns costs nothing: it stays in the conversation's memory.
- Measured: a new camera frame every turn with three background threads generating: first token after 1.5 s.
- **Reading text or numbers** needs more rows than the default gives a small picture (one row per 32 x 32
  pixels). Of fifteen 480 x 480 pictures of a six-digit number, 5 were read at the default, 12 with
  `--image-min-tokens 512` and 14 with `--image-min-tokens 1024 --image-max-tokens 1024`; a picture then takes
  1.4 s to load instead of 0.4 s. Check your own pictures with `scripts/image_digits_check.py`.

## Speech

`sixi-speech` listens on `http://127.0.0.1:8088` with the OpenAI audio API and runs on CPU cores.

```python
speech = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="none")
text = speech.audio.transcriptions.create(model="parakeet", file=open("utterance.wav", "rb")).text
audio = speech.audio.speech.create(model="piper", voice="de", input="Guten Morgen!", response_format="wav")
audio.write_to_file("reply.wav")
```

| | Model | Languages | Measured on the Thor's CPU |
|---|---|---|---|
| Recognition | NVIDIA Parakeet TDT 0.6b v3 | 25 European languages, among them English, German, Italian, French, Polish; the language is detected | a 4-6 s utterance in 0.2-0.35 s; four at once in 0.9 s |
| Recognition, other languages | Whisper turbo (`language=zh`, or `model=whisper`) | about a hundred, Chinese among them | a 4-6 s utterance in 2-3 s |
| Synthesis | Piper voices `en`, `de`, `it`, `fr`, `pl`, `pl-female` | one voice per language | 2-3 s of speech in 0.3-0.4 s |

- **Recognition** takes a finished utterance (any WAV; other formats through ffmpeg) and returns its text. Cut
  the microphone stream at pauses on the agent's side and send each utterance when it ends.
- **Synthesis** returns one audio file per request: send the answer sentence by sentence as the language model
  streams it, and play the files in order. `response_format`: `wav` (default), `pcm` (16-bit mono, the rate is in
  the `X-Sample-Rate` header, 22050 Hz), `opus`, `flac`; `mp3` only if the machine's ffmpeg can write it (this
  one cannot), so name a format explicitly with OpenAI clients, whose default is mp3.
- **The same audio twice**: a voice draws random noise, so the same sentence is different audio every time.
  `"deterministic": true` in the request (or `"noise_scale": 0, "noise_scale_w": 0`) gives the same audio for
  the same text, a little flatter; use it for measurements.
- **Give `language` in a recognition request whenever you know it**, above all for short answers. Parakeet has
  no input for the language and decides from the audio; a word of one syllable is too little, and it writes an
  English word (a Polish "nie" came back as "Yeah.", "tak" in Cyrillic). With `language=pl`, `de`, `it` or `fr`
  the service hears a clip of up to 2 s together with a phrase in that language, spoken by its own voice, and
  takes the phrase off the text again (0.2 to 0.35 s a clip). Measured on one-word clips spoken by many people
  (Multilingual Spoken Words Corpus: words cut from sentences, so harder than an answer said alone); right / a
  wrong word / nothing:

  | | Polish, 300 clips | German, 280 | French, 317 | Italian, 161 |
  |---|---|---|---|---|
  | without `language` (and up to 1.0.0-rc2 with it) | 80 / 153 / 67 | 41 / 171 / 68 | 76 / 172 / 69 | 38 / 98 / 25 |
  | with `language` | 201 / 21 / 78 | 126 / 18 / 136 | 146 / 35 / 136 | 76 / 33 / 52 |
  | Whisper turbo, `model=whisper`, with `language`, 2 s a clip | 132 / 168 / 0 | | | |

  "nie": 0 of 40 without, 24 of 40 with; "nein" 26 -> 38, "non" 8 -> 29, "oui" 27 -> 38. An empty text means
  nothing was heard for certain: let the agent ask again. English needs no phrase. `lead_in=0` in the form turns
  the phrases off for one request; `scripts/speech_short_check.py` runs the comparison with the service's voices
  or with your own recordings.
- **Where a wrong word is worse than no answer** (a "yes" that was a "no"), add `certain=1` to the form: a short
  clip's text comes back only when it was heard the same way twice (with the phrase in front and with a phrase
  behind), otherwise the text is empty. On the 300 Polish clips: 180 right, 12 a wrong word, 108 nothing,
  against 201 / 21 / 78 without it.
- `language` also chooses the recogniser: Parakeet for its 25 languages, Whisper for the others. Audio with
  nothing above -60 dBFS returns an empty text without asking a recogniser (Whisper writes a sentence for
  silence).
- A voice speaks its own language: pick the voice by the language of the answer. OpenAI voice names (`alloy`,
  ...) map to the service's default voice (`--default-voice`, `en`).
- Checked by a round trip (synthesise, then recognise) in the five languages: all sentences came back; numbers
  come back as digits, and one French time was misheard ("10 heures" as "1h").
- **What the log holds.** `~/.sixi/speech/service.log` (the service's standard output) has one line per request:
  audio length, time taken, model or voice, number of characters. It holds no recognised or spoken text. For
  debugging, `sixi-speech serve --log-text` (or `sixi-speech service install --log-text`, or
  `SIXI_SPEECH_LOG_TEXT=1`) adds the first 60-80 characters of each text; the first line of the log says which
  mode is on. Audio is decoded in memory and never written to disk. Up to 1.0.0-rc2 the text was always logged:
  replace `speech/sixi-speech` with the newer file, or start the service with its output discarded. The engine's
  own log (`~/.sixi/service.log`) holds no request text.
- **Silence.** Whisper can return invented text for audio without speech (reported from real recordings: 41
  words for 9 s of silence); Parakeet returned nothing for the same file. Keep Parakeet for the languages it
  covers and send Whisper only audio that a voice-activity check has passed.
- Each Piper voice has its own license (file `MODEL_CARD` in its folder under `~/.sixi/speech/models`); the
  Parakeet model is CC-BY-4.0. The `pl-female` voice has no such file: it comes from the Hugging Face repository
  `WitoldG/polish_piper_models`, whose page says "license: mit" and names no licence for its recordings.
- Names: the engine accepts any `model` value (one model is loaded); a recognition `model` beginning with
  `whisper` selects Whisper, anything else Parakeet; the `model` of a synthesis request is ignored.

Larger speech models that sound more natural or stream partial results exist (they need the GPU, and some are
licensed for non-commercial use only); they would compete with the language model for the GPU, so they are not
the default here.

## A whole spoken turn

`scripts/voice_turn_demo.py` runs one turn through both services and prints the time of every stage; it is a
reference for the agent's main loop (recognise, stream the answer, speak each sentence as it is finished).

```bash
python3 scripts/voice_turn_demo.py --say "What is the tallest mountain in Italy, and how high is it?" --voice en
python3 scripts/voice_turn_demo.py --question question.wav --voice de --image frame.jpg
```

Measured with both services running (`--parallel 8`), the question spoken by the speech service itself:

| Question | Understood after | First token of the answer | First audio of the answer, after the question ended | Whole turn |
|---|---|---|---|---|
| English | 0.16 s | 0.27 s | 1.4 s | 1.8 s for 7.0 s of speech |
| Italian | 0.15 s | 0.21 s | 1.1 s | 1.6 s for 5.4 s of speech |
| German (the answer was one long sentence) | 0.14 s | 0.21 s | 2.5 s | 2.5 s for 6.9 s of speech |
| Polish (one long sentence) | 0.15 s | 0.20 s | 3.8 s | 3.8 s for 7.8 s of speech |
| English, about an image | 0.11 s | 1.2 s | 2.2 s | 2.2 s for 2.8 s of speech |

The first audio waits for the first complete sentence. When answers come as one long sentence, cut at commas
too, or ask for short sentences in the system prompt (the demo does).

Speech on the CPU does not slow the language model: the first token of a turn comes after 0.14 s with the speech
service idle and after 0.14 s while it recognises and speaks without a pause in four loops; with three
background model threads on top, after 0.18 s (0.28 s at worst).

## Memory

| | |
|---|---|
| Qwen3.8-27B with drafter and projector, `--parallel 8 --ctx 8192` | about 34 GB |
| the same with 32768 tokens of context | 48 GB (44 GiB) |
| speech service: Parakeet and the six voices | about 1.5 GB of ordinary memory; 3 GB once Whisper has been used |

## Check it on your machine

```bash
python3 scripts/voice_latency.py --url http://127.0.0.1:8089 --turns 20                 # alone
python3 scripts/voice_latency.py --url http://127.0.0.1:8089 --turns 20 --background 3   # with other threads
python3 scripts/voice_latency.py --url http://127.0.0.1:8089 --turns 10 --image frame.jpg
```

It prints the time to the first token, the time to the end of the first sentence and the stream rate of an
interactive turn. Run it once alone and once while the speech service is working: the difference is what the
neighbours cost.
