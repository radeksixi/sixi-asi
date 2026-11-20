#!/usr/bin/env python3
"""Single-stream speed over a set of prompts large enough to compare two builds.

The five-prompt `ab_bench.py` is decided by which text the model happens to write: two builds whose answers differ
in a few tokens can differ by ten percent on it, because the drafter agrees with the model for a different share of
that text. This script uses 60 prompts (30 GSM8K problems, 20 HumanEval tasks, 10 everyday questions; the same
sample every time; `--temperature`, `--top-p`, `--top-k` and `--request-seed` measure sampled answers) and reports, next to tokens per second, the two things tok/s is made of:

  ms per speculative step   what the engine and the kernels cost
  tokens accepted per step  what the drafter and the text give

A faster build has a lower step time; a luckier text has a higher acceptance. Greedy, 256 tokens, no reasoning
phase by default (`--thinking` keeps it). Standard library only; needs the problem sets `intel_eval.py` downloads.

  python3 scripts/speed_set.py --url http://127.0.0.1:8089 --runs 3 --out speed.json
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intel_eval  # noqa: E402  (the problem sets and their cache)

EVERYDAY = [
    "Explain to a new colleague how to prepare for a code review, in a few short paragraphs.",
    "Write a polite email that asks a supplier for a delivery date and a corrected invoice.",
    "Summarise the plot of Romeo and Juliet in about 120 words.",
    "List the steps to change a flat bicycle tyre, with one sentence each.",
    "Describe how a refrigerator works to a ten-year-old.",
    "Draft a packing list for a three-day hiking trip in autumn and explain the three most important items.",
    "Compare renting and buying a home in five balanced points.",
    "Write a short product description for a stainless steel water bottle.",
    "Explain what an API rate limit is and how a client should behave when it is reached.",
    "Give a weekly study plan for learning basic Italian in a month.",
]


def prompts(n_gsm, n_he, seed):
    out = []
    for q in intel_eval.load_gsm8k(n_gsm, seed):
        out.append(("gsm8k", q["question"] + "\n\nSolve step by step, then give the final answer as a single number on the last line."))
    for q in intel_eval.load_humaneval(n_he, seed):
        out.append(("humaneval", "Complete the following Python function. Return the full function in one code block.\n\n" + q["prompt"]))
    for q in EVERYDAY:
        out.append(("everyday", q))
    return out


def ask(url, text, max_tokens, thinking, sampling=None):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0, "enable_thinking": thinking}
    body.update(sampling or {})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    m = r.get("metrics") or {}
    tokens = (r.get("usage") or {}).get("completion_tokens") or 0
    tps = m.get("decode_tok_per_s") or 0.0
    acc = m.get("mean_acceptance_length") or 1.0
    return {"tokens": tokens, "tok_s": tps, "accepted": acc, "ttft_ms": m.get("ttft_ms") or 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-gsm", type=int, default=30)
    ap.add_argument("--n-he", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--thinking", action="store_true", help="keep the reasoning phase")
    ap.add_argument("--temperature", type=float, default=0.0, help="above 0: sampled answers (send --request-seed to repeat them)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--request-seed", type=int, default=None, help="the seed field of every request")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sampling = {"temperature": a.temperature}
    for field, value in (("top_p", a.top_p), ("top_k", a.top_k), ("seed", a.request_seed)):
        if value is not None:
            sampling[field] = value

    ps = prompts(a.n_gsm, a.n_he, a.seed)
    runs = []
    for run in range(a.runs):
        rows = []
        t0 = time.time()
        for kind, text in ps:
            r = ask(a.url, text, a.max_tokens, a.thinking, sampling)
            r["kind"] = kind
            rows.append(r)
        use = [r for r in rows if r["tokens"] >= 8 and r["tok_s"] > 0]
        seconds = sum(r["tokens"] / r["tok_s"] for r in use)
        tokens = sum(r["tokens"] for r in use)
        steps = sum(r["tokens"] / r["accepted"] for r in use)
        summary = {"tok_s": tokens / seconds, "step_ms": 1000.0 * seconds / steps, "accepted": tokens / steps,
                   "median_tok_s": statistics.median(r["tok_s"] for r in use), "tokens": tokens, "prompts": len(use),
                   "first_token_ms": statistics.median(r["ttft_ms"] for r in use), "wall_s": time.time() - t0}
        by_kind = {}
        for kind in ("gsm8k", "humaneval", "everyday"):
            k = [r for r in use if r["kind"] == kind]
            if k:
                ks = sum(r["tokens"] / r["tok_s"] for r in k)
                by_kind[kind] = sum(r["tokens"] for r in k) / ks
        summary["by_kind"] = by_kind
        runs.append({"summary": summary, "rows": rows})
        print("run %d: %.1f tok/s over %d prompts and %d tokens | %.1f ms per step | %.2f accepted per step | median prompt %.1f tok/s | by kind %s"
              % (run + 1, summary["tok_s"], summary["prompts"], tokens, summary["step_ms"], summary["accepted"], summary["median_tok_s"],
                 " ".join("%s %.1f" % (k, v) for k, v in by_kind.items())), flush=True)
    if a.runs > 1:
        v = [r["summary"]["tok_s"] for r in runs]
        s = [r["summary"]["step_ms"] for r in runs]
        print("all runs: %.1f tok/s (%.1f to %.1f), %.1f ms per step (%.1f to %.1f)" % (statistics.mean(v), min(v), max(v), statistics.mean(s), min(s), max(s)))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "runs": runs}, f, indent=1)


if __name__ == "__main__":
    main()
