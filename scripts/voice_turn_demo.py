#!/usr/bin/env python3
"""One spoken turn through the two local services, with the time of every stage.

  recorded question (WAV) -> sixi-speech (text) -> sixi-asi (streamed answer) -> sixi-speech (audio, sentence by sentence)

It is a reference for an agent's main loop: the answer is spoken from its first sentence on, while the language
model is still writing the rest. Standard library only.

  python3 scripts/voice_turn_demo.py --question question.wav --voice en
  python3 scripts/voice_turn_demo.py --say "What is the tallest mountain in Italy?" --voice en     # question made by the speech service
  python3 scripts/voice_turn_demo.py --say "Was siehst du?" --voice de --image frame.jpg
"""
import argparse
import base64
import json
import os
import time
import urllib.request

SENTENCE_END = (".", "!", "?", ":", ";")


def post_json(url, body, timeout=120):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def transcribe(speech, wav_bytes):
    boundary = "turn%d" % int(time.time() * 1000)
    body = (b"--" + boundary.encode() + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"q.wav\"\r\n\r\n" + wav_bytes +
            b"\r\n--" + boundary.encode() + b"--\r\n")
    req = urllib.request.Request(speech + "/v1/audio/transcriptions", body, {"Content-Type": "multipart/form-data; boundary=" + boundary})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["text"]


def synthesize(speech, text, voice):
    return post_json(speech + "/v1/audio/speech", {"input": text, "voice": voice, "response_format": "wav"}).read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="http://127.0.0.1:8089")
    ap.add_argument("--speech", default="http://127.0.0.1:8088")
    ap.add_argument("--question", default="", help="WAV file with the spoken question")
    ap.add_argument("--say", default="", help="text to turn into the spoken question first (instead of --question)")
    ap.add_argument("--voice", default="en", help="voice of the answer (and of --say)")
    ap.add_argument("--image", default="", help="image the question is about")
    ap.add_argument("--system", default="You are a voice assistant. Answer in the language of the question, in one to three short sentences, without lists or markup.")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--out", default="", help="folder for the answer's audio files (one per sentence)")
    a = ap.parse_args()

    if a.say:
        wav = synthesize(a.speech, a.say, a.voice)
    elif a.question:
        with open(a.question, "rb") as f:
            wav = f.read()
    else:
        ap.error("give --question FILE or --say TEXT")

    t0 = time.time()
    question = transcribe(a.speech, wav)
    t_stt = time.time() - t0
    print("heard      (%.2f s): %s" % (t_stt, question))

    content = question
    if a.image:
        with open(a.image, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        mime = "png" if a.image.lower().endswith(".png") else "jpeg"
        content = [{"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % (mime, data)}}, {"type": "text", "text": question}]
    body = {"messages": [{"role": "system", "content": a.system}, {"role": "user", "content": content}],
            "max_tokens": a.max_tokens, "temperature": 0, "stream": True, "enable_thinking": False}

    t1 = time.time()
    first_token = first_audio = None
    pending = ""
    spoken = []
    audio_s = 0.0
    with post_json(a.llm + "/v1/chat/completions", body, timeout=600) as resp:
        def speak(sentence):
            nonlocal first_audio, audio_s
            sentence = sentence.strip()
            if not sentence:
                return
            t = time.time()
            audio = synthesize(a.speech, sentence, a.voice)
            if first_audio is None:
                first_audio = time.time() - t0
            seconds = max(0, len(audio) - 44) / 2.0 / 22050.0
            audio_s += seconds
            spoken.append(sentence)
            print("  spoke    (+%.2f s, %.1f s of audio): %s" % (time.time() - t, seconds, sentence))
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, "answer_%02d.wav" % len(spoken)), "wb") as f:
                    f.write(audio)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if "error" in ev:
                raise SystemExit("language model: %s" % ev["error"])
            delta = (ev.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if not delta:
                continue
            if first_token is None:
                first_token = time.time() - t1
            pending += delta
            # hand over every finished sentence as soon as it is there
            while True:
                cut = next((i for i, ch in enumerate(pending) if ch in SENTENCE_END and i + 1 < len(pending) and pending[i + 1] in " \n"), -1)
                if cut < 0:
                    break
                speak(pending[: cut + 1])
                pending = pending[cut + 1:]
        speak(pending)
    total = time.time() - t0
    print("first token of the answer   %.2f s after the question was understood" % (first_token or 0.0))
    print("first audio of the answer   %.2f s after the question ended" % (first_audio or 0.0))
    print("whole turn                  %.2f s for %.1f s of spoken answer" % (total, audio_s))


if __name__ == "__main__":
    main()
