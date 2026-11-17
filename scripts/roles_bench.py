#!/usr/bin/env python3
"""Several agent sessions, each calling a few roles with long system prompts: what does a call cost?

The workload a user of the service described (2026-11-17): every turn of a session makes three calls one after
another (a reader, a writer, a coder), each with its own system prompt of 1,000 to 2,500 tokens and the
conversation so far, and each answers with a short JSON object. Such calls are mostly prompt: about 2,000
tokens in, 50 out. Their cost is decided by how much of the prompt the server still has from an earlier call
(`usage.prompt_cached_tokens`), and that by the number of prompt states it keeps (`--device-slots`).

  python3 scripts/roles_bench.py --url http://127.0.0.1:8091 --sessions 1
  python3 scripts/roles_bench.py --url http://127.0.0.1:8091 --sessions 10 --out roles10.json
Standard library only.
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request

ROLES = ["reader", "writer", "coder", "actor"]
WORDS = ("session caller line marker quote remark reason question signal turn answer object field value rule "
         "example careful exact short return never always when then first second third note check list").split()


def system_prompt(role, tokens, rnd):
    # distinct text per role, the same in every session (as a real role prompt is)
    body = " ".join(rnd.choice(WORDS) for _ in range(int(tokens * 0.95)))
    return ("You are the %s of a conversation system. Rules follow.\n%s\n"
            "Answer with exactly one JSON object {\"role\": \"%s\", \"note\": <one short sentence>} and nothing else." % (role, body, role))


def call(url, system, user, max_tokens, background):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0, "enable_thinking": False}
    if background:
        body["priority"] = "background"
    t = time.time()
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    busy = 0
    while True:
        # more callers than the server has slots (--parallel): it answers 429 and the caller tries again
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or time.time() - t > 600:
                raise
            busy += 1
            time.sleep(0.2)
    u = r.get("usage") or {}
    return {"s": time.time() - t, "busy": busy, "in": u.get("prompt_tokens", 0), "cached": u.get("prompt_cached_tokens", 0), "out": u.get("completion_tokens", 0),
            "prefill_ms": (r.get("metrics") or {}).get("prefill_ms", 0.0), "text": r["choices"][0]["message"].get("content") or ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--sessions", type=int, default=1, help="sessions running at the same time")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--roles", type=int, default=3, help="calls per turn, each with its own system prompt")
    ap.add_argument("--system-tokens", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--background", action="store_true", help="send every call with priority background")
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", default="", help="an --out file of an earlier run with the same arguments: count the answers that are the same text")
    a = ap.parse_args()

    systems = [system_prompt(ROLES[i % len(ROLES)], a.system_tokens, random.Random(100 + i)) for i in range(a.roles)]
    rows = []
    lock = threading.Lock()

    def session(sid):
        rnd = random.Random(1000 + sid)
        history = []
        for turn in range(a.turns):
            history.append("Caller: " + " ".join(rnd.choice(WORDS) for _ in range(25)) + ".")
            for ri, system in enumerate(systems):
                user = "Conversation so far:\n" + "\n".join(history) + "\n\nDo your part for the last line."
                r = call(a.url, system, user, a.max_tokens, a.background)
                r.update({"session": sid, "turn": turn, "role": ri})
                with lock:
                    rows.append(r)
            history.append("Agent: " + " ".join(rnd.choice(WORDS) for _ in range(30)) + ".")

    t0 = time.time()
    threads = [threading.Thread(target=session, args=(i,)) for i in range(a.sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    later = [r for r in rows if r["turn"] > 0]      # the first turn loads every system prompt once
    use = later or rows
    tin, tcached, tout = (sum(r[k] for r in use) for k in ("in", "cached", "out"))
    summary = {"sessions": a.sessions, "calls": len(rows), "wall_s": wall, "calls_per_s": len(rows) / wall,
               "call_s_median": statistics.median(r["s"] for r in use), "call_s_first_turn": statistics.median(r["s"] for r in rows if r["turn"] == 0),
               "turn_s_median": statistics.median(sum(r["s"] for r in use if r["session"] == s and r["turn"] == t) for s in range(a.sessions) for t in range(1 if later else 0, a.turns)),
               "prompt_tokens_per_call": tin / len(use), "cached_share": tcached / max(1, tin), "out_tokens_per_call": tout / len(use),
               "calls_answered_429_first": sum(1 for r in rows if r["busy"])}
    print("%2d sessions: call %.2f s median after the first turn (first turn %.2f s), turn of %d calls %.1f s, %.2f calls/s; "
          "%d prompt tokens a call, %.0f%% of them cached; %d tokens out%s" % (a.sessions, summary["call_s_median"], summary["call_s_first_turn"], a.roles,
                                                                           summary["turn_s_median"], summary["calls_per_s"], summary["prompt_tokens_per_call"], 100 * summary["cached_share"], summary["out_tokens_per_call"],
                                                                           "" if not summary["calls_answered_429_first"] else "; %d of %d calls had to wait for a slot (429, tried again)" % (summary["calls_answered_429_first"], len(rows))), flush=True)
    if a.compare:
        with open(a.compare) as f:
            before = {(r["session"], r["turn"], r["role"]): r.get("text") for r in json.load(f)["rows"]}
        both = [r for r in rows if before.get((r["session"], r["turn"], r["role"])) is not None]
        same = sum(1 for r in both if before[(r["session"], r["turn"], r["role"])] == r["text"])
        summary["same_text_as_compared"] = [same, len(both)]
        print("answers identical to %s: %d of %d" % (a.compare, same, len(both)))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "summary": summary, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
