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
- A voice speaks its own language: pick the voice by the language of the answer. OpenAI voice names (`alloy`,
  ...) map to the service's default voice (`--default-voice`, `en`).
- Checked by a round trip (synthesise, then recognise) in the five languages: all sentences came back; numbers
  come back as digits, and one French time was misheard ("10 heures" as "1h").
- Each Piper voice has its own license (file `MODEL_CARD` in its folder under `~/.sixi/speech/models`); the
  Parakeet model is CC-BY-4.0.

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
