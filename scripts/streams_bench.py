#!/usr/bin/env python3
"""Throughput with N requests at once, every request a different prompt (standard library only).

`concurrency_bench.py` sends the five `ab_bench` prompts from every stream, so with 8 or 32 streams the same
prompts come again and again: some prompt loads are answered from kept states, and the streams write the same
texts. This script gives every request its own prompt from a pool of 240 (120 GSM8K problems, 80 HumanEval
tasks, 40 everyday questions, in the proportions of `speed_set.py`, whose 60 prompts are the start of the pool)
and reports two figures per stream count:

  decode   N x (tokens after the first / time from the first to the last token), over all requests: what the
           streams write while they write, prompt loading and queueing left out
  wall     all completion tokens / wall time of the run: prompt loading, queueing and the tail in which some
           streams have finished are in it

Greedy, 256 tokens, no reasoning phase by default. Needs the problem sets `intel_eval.py` downloads.

  streams_bench.py --url http://127.0.0.1:8091 --streams 8,16,32 --out streams.json
  streams_bench.py --url http://127.0.0.1:8091 --streams 8 --temperature 0.7 --top-p 0.8 --seed 7
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_bench  # noqa: E402  (the streaming client)
import intel_eval  # noqa: E402  (the problem sets and their cache)
import speed_set  # noqa: E402  (its ten everyday questions and the prompt wording)

EVERYDAY_MORE = [
    "Explain the difference between a debit card and a credit card to a teenager.",
    "Write a short note to neighbours announcing a street party next month.",
    "Describe how to brew a good cup of filter coffee, step by step.",
    "Give five practical tips for a first job interview and a sentence on each.",
    "Explain why the sky is blue in plain language.",
    "Write a friendly reminder to a team about submitting timesheets by Friday.",
    "Summarise the story of Cinderella in about 100 words.",
    "Describe the water cycle to a class of nine-year-olds.",
    "Compare travelling by train and by car for a 400 km trip in five points.",
    "Write a short welcome message for a new member of a chess club.",
    "Explain what compound interest is, with one small worked example in words.",
    "List what to check before buying a used bicycle, one sentence each.",
    "Describe a typical day of a lighthouse keeper a hundred years ago.",
    "Write a polite complaint to a hotel about a noisy room and ask for a partial refund.",
    "Explain how vaccines train the immune system, in a few short paragraphs.",
    "Give a simple plan for tidying a cluttered garage over one weekend.",
    "Describe the rules of football to someone who has never seen a match.",
    "Write a short speech thanking volunteers after a charity run.",
    "Explain what a database index is and when it does not help.",
    "Suggest a three-course dinner for six guests, one of them vegetarian, and say why the dishes fit together.",
    "Describe how a sailing boat can move against the wind.",
    "Write a paragraph that introduces a small town museum on its website.",
    "Explain the difference between weather and climate.",
    "Give advice to a student who keeps putting off writing an essay.",
    "Describe how bread rises, from mixing the dough to the oven.",
    "Write a short message declining a meeting and proposing two other times.",
    "Explain what version control is to a writer who has never programmed.",
    "List the main things to do in the first week after moving to a new city.",
    "Describe how noise-cancelling headphones work.",
    "Write a short review of a quiet seaside cafe.",
]


def pool():
    """240 prompts in a repeating pattern of three GSM8K problems, two HumanEval tasks and one everyday question."""
    base = speed_set.prompts(120, 80, 1234)
    gsm = [p for k, p in base if k == "gsm8k"]
    he = [p for k, p in base if k == "humaneval"]
    ev = [p for k, p in base if k == "everyday"] + EVERYDAY_MORE
    out = []
    for i in range(40):
        out += [("gsm8k", gsm[3 * i]), ("gsm8k", gsm[3 * i + 1]), ("gsm8k", gsm[3 * i + 2]),
                ("humaneval", he[2 * i]), ("humaneval", he[2 * i + 1]), ("everyday", ev[i])]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--streams", default="8,16,32")
    ap.add_argument("--per-stream", type=int, default=0, help="prompts per stream, a multiple of 6 (default: 6 to 24, by the stream count)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thinking", action="store_true", help="keep the reasoning phase")
    ap.add_argument("--extra-json", default=None, help="JSON merged into every request body")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    prompts = pool()
    extra = {"enable_thinking": bool(a.thinking)}
    if a.top_p is not None:
        extra["top_p"] = a.top_p
    if a.top_k is not None:
        extra["top_k"] = a.top_k
    if a.extra_json:
        extra.update(json.loads(a.extra_json))
    rows = []
    for n in [int(x) for x in a.streams.split(",")]:
        per = a.per_stream or 6 * max(1, min(4, 40 // n))
        if per % 6 or n * per > len(prompts):
            sys.exit("streams x per-stream must be a multiple of 6 per stream and at most %d prompts" % len(prompts))
        res = [[] for _ in range(n)]
        err = [None] * n

        def run(i):
            try:
                for kind, text in prompts[i * per:(i + 1) * per]:
                    r = ab_bench.stream_chat(a.url, [{"role": "user", "content": text}], a.max_tokens, a.temperature, a.seed + i, extra=extra)
                    r["kind"] = kind
                    r.pop("reasoning", None)
                    r.pop("content", None)
                    res[i].append(r)
            except Exception as e:  # noqa: BLE001  (reported with the row)
                err[i] = repr(e)

        t0 = time.perf_counter()
        th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.perf_counter() - t0
        done = [r for rs in res for r in rs]
        toks = sum(r["completion_tokens"] for r in done)
        dec = [r for r in done if r["decode_tok_s"] > 0 and r["completion_tokens"] > 1]
        dec_s = sum((r["completion_tokens"] - 1) / r["decode_tok_s"] for r in dec)
        per_stream = (sum(r["completion_tokens"] - 1 for r in dec) / dec_s) if dec_s > 0 else 0.0
        ptok = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in done)
        cached = sum((r.get("usage") or {}).get("prompt_cached_tokens") or 0 for r in done)
        acc = [(r.get("metrics") or {}).get("mean_acceptance_length") for r in done]
        acc = [x for x in acc if x]
        ttft = sorted(r["ttft_s"] for r in done)
        row = {"streams": n, "per_stream_prompts": per, "requests": len(done), "errors": [e for e in err if e],
               "wall_s": wall, "tokens": toks, "wall_tok_s": toks / wall if wall > 0 else 0.0,
               "decode_tok_s_per_stream": per_stream, "decode_tok_s": n * per_stream,
               "ttft_median_s": statistics.median(ttft) if ttft else None, "ttft_worst_s": ttft[-1] if ttft else None,
               "prompt_tokens": ptok, "prompt_cached_tokens": cached,
               "accepted_per_step": (statistics.mean(acc) if acc else None), "detail": res}
        rows.append(row)
        print("streams=%d: decode %.1f tok/s (%.1f per stream) | wall %.1f tok/s | first token %.2f s median, %.2f worst | "
              "%d requests, %d tokens in %.1f s | prompt tokens %d, from kept states %d%s%s"
              % (n, row["decode_tok_s"], per_stream, row["wall_tok_s"], row["ttft_median_s"] or 0, row["ttft_worst_s"] or 0,
                 len(done), toks, wall, ptok, cached,
                 (" | accepted per step %.2f" % row["accepted_per_step"]) if acc else "",
                 (" | ERRORS %d" % len(row["errors"])) if row["errors"] else ""), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
