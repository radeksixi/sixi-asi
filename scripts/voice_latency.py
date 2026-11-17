#!/usr/bin/env python3
"""Latency of an interactive turn on a Sixi-ASI server, alone and next to other work.

A talking agent cares about three numbers per turn: time to the first token, time to the end of the first
sentence (what a TTS can start speaking), and tokens per second after that. This script sends short streamed
turns one after another and reports the three, optionally while

  --background N     N other threads keep generating long answers on the same server (other agent threads)
  --loaders N        N other threads keep loading long new prompts (documents), the worst neighbour for a turn
  --image FILE       every interactive turn carries this image (camera frame); needs a server with --mmproj

Run it once on an idle machine and once while the speech stack (STT, TTS, VLM) is working on the GPU: the
difference is what sharing the GPU costs.

  python3 scripts/voice_latency.py --url http://127.0.0.1:8089 --turns 20
  python3 scripts/voice_latency.py --url http://127.0.0.1:8089 --turns 20 --background 3 --image frame.jpg

Standard library only.
"""
import argparse
import base64
import json
import statistics
import threading
import time
import urllib.request

QUESTIONS = [
    "What should I check before leaving the house?",
    "Give me one tip for a better night's sleep.",
    "How long does it take to boil an egg?",
    "What is a good name for a small robot?",
    "Why is the sky blue? Keep it short.",
    "Suggest something quick to cook tonight.",
    "What is seven times eight, and how do you know?",
    "Tell me a fact about the Moon.",
]
IMAGE_QUESTIONS = [
    "What do you see in front of you?",
    "Is there any text in view? Read it out.",
    "Describe the scene in one sentence.",
    "What is the most noticeable object here?",
]
LONG_TASK = "Write a detailed, step by step guide to planning a week-long hiking trip, covering gear, food, safety and navigation."


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def stream_turn(url, messages, max_tokens, extra):
    """One streamed request. Returns (first token s, first sentence s, total s, tokens, text, error)."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True, "enable_thinking": False}
    body.update(extra)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = sentence = None
    text = ""
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if "error" in ev:
                    return None, None, time.time() - t0, 0, "", str(ev["error"])[:200]
                delta = (ev.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if not delta:
                    continue
                now = time.time() - t0
                if first is None:
                    first = now
                text += delta
                tokens += 1
                if sentence is None and any(text.rstrip().endswith(p) for p in (".", "!", "?", ":")) and len(text.strip()) >= 12:
                    sentence = now
    except Exception as e:  # noqa: BLE001 - report and carry on
        return first, sentence, time.time() - t0, tokens, text, str(e)[:200]
    return first, sentence if sentence is not None else time.time() - t0, time.time() - t0, tokens, text, None


def background(url, stop, counters, kind, index):
    n = 0
    while not stop.is_set():
        n += 1
        if kind == "gen":
            messages = [{"role": "user", "content": LONG_TASK + f" (variant {index}.{n})"}]
            max_tokens = 400
        else:   # a long new prompt every time: nothing of it is in the prefix cache
            filler = " ".join(f"Record {index}.{n}.{i}: sensor {i % 17} read {i * 7 % 101} at step {i}." for i in range(counters["loader_records"]))
            messages = [{"role": "user", "content": filler + "\nHow many records are there? Answer with a number."}]
            max_tokens = 16
        _, _, _, tokens, _, err = stream_turn(url, messages, max_tokens, counters["extra"])
        with counters["lock"]:
            counters[kind] += tokens
            if err:
                counters["errors"] += 1
                counters["last_error"] = err
        if err:
            time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--background", type=int, default=0, help="threads generating long answers meanwhile")
    ap.add_argument("--loaders", type=int, default=0, help="threads loading long new prompts meanwhile")
    ap.add_argument("--loader-records", type=int, default=200, help="size of a loader's prompt (one record is about 22 tokens)")
    ap.add_argument("--neighbour-priority", default="background", choices=["background", "interactive"],
                    help="priority the neighbours send (default: background, as other agent threads should)")
    ap.add_argument("--image", default="", help="image file sent with every turn")
    ap.add_argument("--same-image", action="store_true", help="keep one conversation, so the image is loaded once")
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between turns")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    image_part = None
    if a.image:
        with open(a.image, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        mime = "png" if a.image.lower().endswith(".png") else "jpeg"
        image_part = {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}}

    stop = threading.Event()
    counters = {"gen": 0, "load": 0, "errors": 0, "last_error": "", "loader_records": a.loader_records,
                "extra": ({"priority": "background"} if a.neighbour_priority == "background" else {}), "lock": threading.Lock()}
    threads = [threading.Thread(target=background, args=(a.url, stop, counters, "gen", i), daemon=True) for i in range(a.background)]
    threads += [threading.Thread(target=background, args=(a.url, stop, counters, "load", i), daemon=True) for i in range(a.loaders)]
    for t in threads:
        t.start()
    if threads:
        time.sleep(4.0)   # let the neighbours get going

    rows = []
    history = []
    t_start = time.time()
    for i in range(a.turns):
        if image_part is not None:
            q = IMAGE_QUESTIONS[i % len(IMAGE_QUESTIONS)]
            content = [image_part, {"type": "text", "text": q}] if (not a.same_image or not history) else q
        else:
            q = QUESTIONS[i % len(QUESTIONS)]
            content = q + f" (turn {i})"   # a new prompt each time: no free ride from the cache
        messages = (history if a.same_image else []) + [{"role": "user", "content": content}]
        first, sentence, total, tokens, text, err = stream_turn(a.url, messages, a.max_tokens, {})
        if err or first is None:
            print(f"turn {i}: failed: {err}")
            continue
        rate = (tokens - 1) / (total - first) if tokens > 1 and total > first else 0.0
        rows.append({"first_token_s": first, "first_sentence_s": sentence, "total_s": total, "tokens": tokens, "tok_s": rate})
        if a.same_image:
            history = messages + [{"role": "assistant", "content": text}]
        time.sleep(a.pause)
    elapsed = time.time() - t_start
    stop.set()

    if not rows:
        print("no turn succeeded")
        return 1
    def line(key, unit, scale=1.0):
        v = [r[key] * scale for r in rows]
        return f"median {statistics.median(v):7.1f} {unit}   p90 {pct(v, 0.9):7.1f} {unit}   worst {max(v):7.1f} {unit}"
    print(f"{len(rows)} turns, {a.background} generating and {a.loaders} loading neighbours" + (", with an image" if a.image else ""))
    print("  first token     " + line("first_token_s", "ms", 1000.0))
    print("  first sentence  " + line("first_sentence_s", "ms", 1000.0))
    print("  stream events   " + line("tok_s", "per s") + "   (an event can carry several tokens)")
    if threads:
        with counters["lock"]:
            print(f"  neighbours: {counters['gen'] / elapsed:.1f} stream events/s generated in total, {counters['errors']} errors"
                  + (f" (last: {counters['last_error']})" if counters["errors"] else ""))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "turns": rows, "neighbour_tok_s": counters["gen"] / elapsed}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
