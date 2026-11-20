#!/usr/bin/env python3
"""Does drafting change an answer? Asks the running server every prompt twice at temperature 0, once as usual and
once with `"draft": false` (no drafts for that request), and compares the two answers.

Drafts are only proposals: the model checks every drafted token, so the answer should be the one it would have
written alone. The two can still part where two candidate tokens are nearly tied, because a pass that checks
several rows runs other kernels than a pass with one row and rounds differently. This script counts how often
that happens on the machine it runs on, and what drafting gives in speed. It also asks one prompt twice with
sampling and a seed (those two must agree) and once more without drafts (reported, not required: the check that
accepts a draft uses the random numbers in another order).

Standard library only.

  python3 scripts/exact_check.py --url http://127.0.0.1:8089
  python3 scripts/exact_check.py --url http://127.0.0.1:8089 --min-same 0.9     exit code 1 below 90% identical
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring.",
    "A train leaves at 9:40 and arrives at 13:05. It stops for 12 minutes on the way. How long is it moving? Show the steps.",
    "Explain in four sentences why the seasons change during the year.",
    "Write a SQL query that returns the three customers with the highest total order value from tables customers(id, name) and orders(id, customer_id, value).",
    "List five things to check before a long car trip, one sentence each.",
    "A shop sells pens at 3 for 2.40. How much do 14 pens cost? Give the calculation and the result.",
    "Write a C function that reverses a string in place.",
    "Summarise the story of the tortoise and the hare in about 80 words.",
    "What is the difference between a list and a tuple in Python? Answer briefly with one example.",
    "Write a polite two-paragraph email asking to move a meeting from Tuesday to Thursday.",
    "Convert this to JSON with the keys name, age and city: Anna is 31 and lives in Lyon.",
    "Explain what a hash table is to someone who knows arrays but not hashing.",
]


def ask(url, text, max_tokens, **more):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False}
    body.update(more)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    m = r.get("metrics") or {}
    return {"text": r["choices"][0]["message"].get("content") or "", "finish": r["choices"][0].get("finish_reason"),
            "tok_s": m.get("decode_tok_per_s") or 0.0, "accepted": m.get("mean_acceptance_length") or 0.0,
            "tokens": (r.get("usage") or {}).get("completion_tokens") or 0}


def common_prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS), help="how many of the %d prompts" % len(PROMPTS))
    ap.add_argument("--min-same", type=float, default=0.0, help="exit code 1 when a smaller share of the answers is identical")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = []
    for i, text in enumerate(PROMPTS[:a.prompts]):
        with_drafts = ask(a.url, text, a.max_tokens)
        alone = ask(a.url, text, a.max_tokens, draft=False)
        same = with_drafts["text"] == alone["text"]
        rows.append({"prompt": text, "same": same, "with_drafts": with_drafts, "alone": alone,
                     "same_until": common_prefix(with_drafts["text"], alone["text"])})
        print("%2d  %s  with drafts %5.1f tok/s (%.2f tokens a step), without %5.1f tok/s%s"
              % (i + 1, "same     " if same else "DIFFERENT", with_drafts["tok_s"], with_drafts["accepted"], alone["tok_s"],
                 "" if same else "  | same for the first %d of %d characters" % (rows[-1]["same_until"], len(alone["text"]))), flush=True)
    if not rows:
        sys.exit("no prompts")
    if all(r["alone"]["accepted"] > 1.05 for r in rows):
        print("\nThis server drafted for the requests that said \"draft\": false: it does not know the field (it came after 1.0.0-rc3).")
        sys.exit(2)
    n_same = sum(r["same"] for r in rows)
    fast = sum(r["with_drafts"]["tokens"] for r in rows) / max(1e-9, sum(r["with_drafts"]["tokens"] / r["with_drafts"]["tok_s"] for r in rows if r["with_drafts"]["tok_s"] > 0))
    slow = sum(r["alone"]["tokens"] for r in rows) / max(1e-9, sum(r["alone"]["tokens"] / r["alone"]["tok_s"] for r in rows if r["alone"]["tok_s"] > 0))
    print("\nidentical answers: %d of %d | with drafts %.1f tok/s, without %.1f tok/s (%.2fx)" % (n_same, len(rows), fast, slow, fast / slow if slow else 0.0))

    text = PROMPTS[2]
    s1 = ask(a.url, text, a.max_tokens, temperature=0.7, top_p=0.8, seed=11)
    s2 = ask(a.url, text, a.max_tokens, temperature=0.7, top_p=0.8, seed=11)
    s3 = ask(a.url, text, a.max_tokens, temperature=0.7, top_p=0.8, seed=11, draft=False)
    repeat = s1["text"] == s2["text"]
    print("sampled with a seed, asked twice: %s | the same without drafts: %s (not required)"
          % ("same" if repeat else "DIFFERENT", "same" if s1["text"] == s3["text"] else "different"))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "identical": n_same, "sampled_repeats": repeat, "sampled_same_without_drafts": s1["text"] == s3["text"]}, f, indent=1)
    if n_same < a.min_same * len(rows) or not repeat:
        sys.exit(1)


if __name__ == "__main__":
    main()
