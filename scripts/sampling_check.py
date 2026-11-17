#!/usr/bin/env python3
"""Sampled answers after a greedy answer to the same prompt (and the other way round).

The server keeps what it computed for a prompt, so the same prompt costs nothing the second time. Up to
1.0.0-rc2 a request that sampled (temperature above 0) after the same prompt had been answered at temperature 0
began its answer with a stray character ('*', '+', '%', a digit) nine times out of ten. Fixed in 1.0.0-rc3; this
is the check that would have found it.

Each case asks a question whose answer is one certain word, so that sampling at temperature 1 must give it too:

  greedy, then sampled with several seeds     every sampled answer starts with the word
  sampled first on a new prompt, then greedy  both answers start with the word

  python3 scripts/sampling_check.py --url http://127.0.0.1:8089
Exit status 0 when every answer is right. Standard library only.
"""
import argparse
import json
import random
import sys
import urllib.request

CASES = [
    ("Reply with exactly one word and nothing else: Yes", "yes"),
    ("What is 2 + 2? Answer with the number only.", "4"),
    ("Repeat this word and nothing else: banana", "banana"),
    ("Complete with one word and nothing else: The capital of France is", "paris"),
    ("Answer with this JSON object and nothing else: {\"ok\": true}", "{\"ok\": true}"),
]


def ask(url, text, temperature, seed=None, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    # top_p keeps the rare tail out (at temperature 1 with top_k alone the model ends its turn at once in 2 of 40
    # tries after a long system prompt); the defect this checks for gave a wrong first character in 9 of 10
    body = {"messages": msgs, "max_tokens": 24, "temperature": temperature, "enable_thinking": False, "top_k": 40, "top_p": 0.9}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return (r["choices"][0]["message"].get("content") or "").strip()


def right(answer, word):
    return answer.lower().strip(" .!\n`\"'").startswith(word.strip("\"'")) or answer.lower().startswith(word)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--system-tokens", type=int, default=0, help="put a system prompt of about this many tokens in front (a long shared prefix)")
    a = ap.parse_args()
    filler = ("You are a careful assistant. Follow the user's instruction to the letter. " * (a.system_tokens // 14)) or None
    bad = total = 0
    for text, word in CASES:
        # a number in front makes the prompt new to the server without changing the answer
        tag = "(request %d) " % random.randrange(10 ** 6)
        g = ask(a.url, tag + text, 0, system=filler)
        sampled = [ask(a.url, tag + text, 1.0, seed=s, system=filler) for s in range(1, a.seeds + 1)]
        tag2 = "(request %d) " % random.randrange(10 ** 6)
        s_first = ask(a.url, tag2 + text, 1.0, seed=1, system=filler)
        g_after = ask(a.url, tag2 + text, 0, system=filler)
        rows = [("greedy", g), ("greedy after sampled", g_after), ("sampled first", s_first)] + [("sampled seed %d" % (i + 1), s) for i, s in enumerate(sampled)]
        wrong = [(k, v) for k, v in rows if not right(v, word)]
        total += len(rows)
        bad += len(wrong)
        print("%-62s %d of %d right%s" % (text[:60], len(rows) - len(wrong), len(rows), "" if not wrong else "   WRONG: " + "; ".join("%s -> %r" % (k, v[:16]) for k, v in wrong[:4])))
    print("sampling check: %d of %d answers right" % (total - bad, total))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
