#!/usr/bin/env python3
"""Gate for schema-constrained answers (response_format) on an OpenAI-compatible chat endpoint.

Two sets of 3 prompts per schema: "general" (30 schemas over the features the converter supports) and "agent"
(10 schemas in the shapes of the agent that asked for the feature: flat objects with required strings, arrays of
objects, a 60-value string enum, arrays of integers; free text in Polish, English, German, French and Italian
with typographic quotes, escaped quotes and line breaks). For every case the answer must be valid JSON and
validate against its schema. With --compare-free the same prompt is also sent without response_format (the
schema is pasted into the prompt instead), which shows what the constraint removes (text around the document,
broken endings) and what it costs in speed on identical prompts. --mode object sends
{"type": "json_object"} (any object; the schema goes into the prompt and is still validated afterwards).

Besides the verdict, a row is marked "overruled" when the grammar refused the model's own choice at least once
(metrics.constraint_rows_changed > 0) and "suspect" when a string value ends like a broken closing (a typographic
quote followed by a brace, for instance): valid JSON, but the model meant to close the document there.

  schema_eval.py --url http://127.0.0.1:8091 [--set general|agent|all] [--mode schema|object] [--streams 8]
                 [--temperature 0.7] [--thinking] [--compare-free] [--max-tokens 512] [--limit N] --out results.json

Exit code 0 only when every constrained answer parses and validates (cases the server refuses with 400 are
listed separately: they are schema features the converter cannot express, not wrong answers).
Needs the `jsonschema` package for validation; without it only parsing is checked and the run is marked so.
"""
import argparse, concurrent.futures, json, re, statistics, sys, time, urllib.error, urllib.request

try:
    import jsonschema
except ImportError:  # parsing only
    jsonschema = None


FORMATS = None
if jsonschema is not None:
    # "time" is RFC 3339 full-time (with an offset); the library's built-in check accepts HH:MM:SS only
    FORMATS = jsonschema.FormatChecker()
    FORMATS.checks("time")(lambda v: not isinstance(v, str) or
                           re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)", v) is not None)


def obj(props, required=None, extra=False):
    return {"type": "object", "properties": props, "required": list(props) if required is None else required,
            "additionalProperties": extra}


S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}

# (name, schema, three prompts)
CASES = [
    ("person", obj({"name": S, "age": I, "email": S}),
     ["Invent a person and describe them.", "Describe Ada Lovelace as a record.", "Give me a fictional sailor."]),
    ("tags", obj({"title": S, "tags": {"type": "array", "items": S}}),
     ["Title and tags for an article about memory bandwidth.", "Title and tags for a recipe for pierogi.",
      "Title and tags for a talk about tide pools."]),
    ("enum_sentiment", obj({"sentiment": {"enum": ["positive", "neutral", "negative"]}, "confidence": N}),
     ["Classify: 'The battery died after two days.'", "Classify: 'Works as described.'",
      "Classify: 'Best purchase this year, by far.'"]),
    ("nested_address", obj({"name": S, "address": obj({"street": S, "city": S, "zip": S})}),
     ["A bakery in Gdansk.", "A bookshop in Lisbon.", "A bike repair shop in Utrecht."]),
    ("list_of_objects", obj({"items": {"type": "array", "items": obj({"name": S, "qty": I, "unit": S})}}),
     ["Shopping list for pancakes for four.", "Parts list for a small bookshelf.", "Packing list for a day hike."]),
    ("min_max_items", obj({"steps": {"type": "array", "items": S, "minItems": 3, "maxItems": 5}}),
     ["How do I brew filter coffee?", "How do I change a bicycle tube?", "How do I repot a plant?"]),
    ("integer_range", obj({"rating": {"type": "integer", "minimum": 1, "maximum": 5}, "reason": S}),
     ["Rate the idea of a four-day work week.", "Rate plain oatmeal as a breakfast.", "Rate night trains."]),
    ("booleans", obj({"is_prime": B, "is_even": B, "n": I}),
     ["Tell me about the number 97.", "Tell me about the number 1024.", "Tell me about the number 51."]),
    ("nullable", obj({"name": S, "nickname": {"type": ["string", "null"]}}),
     ["A scientist with no nickname.", "A boxer with a nickname.", "A river (rivers have no nicknames)."]),
    ("any_of", obj({"value": {"anyOf": [I, S, {"type": "array", "items": I}]}}),
     ["Give the value forty-two as a number.", "Give the value 'forty-two' as text.", "Give the first five primes."]),
    ("const_kind", obj({"kind": {"const": "event"}, "name": S, "year": I}),
     ["The first Moon landing.", "The fall of the Berlin Wall.", "The first transatlantic telegraph cable."]),
    ("optional_fields", obj({"city": S, "country": S, "population": I, "note": S}, required=["city", "country"]),
     ["Krakow.", "A small invented town.", "Reykjavik, with a note about its weather."]),
    ("matrix", obj({"rows": {"type": "array", "items": {"type": "array", "items": I, "minItems": 3, "maxItems": 3},
                             "minItems": 3, "maxItems": 3}}),
     ["A 3x3 identity matrix.", "A 3x3 magic square.", "A 3x3 matrix counting from 1 to 9."]),
    ("tuple", {"type": "array", "prefixItems": [S, I, B], "minItems": 3, "maxItems": 3},
     ["Name, age and whether retired, for Marie Curie in 1930.", "A city, its founding year, whether a capital.",
      "A planet, its number of moons, whether it has rings."]),
    ("top_level_array", {"type": "array", "items": obj({"word": S, "length": I}), "minItems": 2, "maxItems": 6},
     ["Words in 'the quick brown fox' with their lengths.", "Four fruits with the lengths of their names.",
      "The words of 'to be or not' with lengths."]),
    ("string_lengths", obj({"code": {"type": "string", "minLength": 3, "maxLength": 3}, "name": S}),
     ["Currency code and name for Poland.", "IATA code and name for Tokyo Haneda.", "Currency code and name for Japan."]),
    ("date_format", obj({"event": S, "date": {"type": "string", "format": "date"}}),
     ["When did Apollo 11 land on the Moon?", "When did the Titanic sink?", "When was the Eiffel Tower opened?"]),
    ("time_format", obj({"task": S, "at": {"type": "string", "format": "time"}}),
     ["Schedule a stand-up at nine in the morning.", "Schedule lunch at half past twelve.", "Schedule a call at 17:45."]),
    ("uuid_format", obj({"id": {"type": "string", "format": "uuid"}, "label": S}),
     ["Make up an id for a test order.", "Make up an id for a sensor.", "Make up an id for a user session."]),
    ("pattern", obj({"plate": {"type": "string", "pattern": "^[A-Z]{2,3} [0-9]{4,5}$"}, "country": S}),
     ["A Polish-style licence plate.", "A made-up licence plate from Krakow.", "A made-up licence plate from Gdansk."]),
    ("defs_ref", {"$defs": {"point": obj({"x": N, "y": N})},
                  **obj({"from": {"$ref": "#/$defs/point"}, "to": {"$ref": "#/$defs/point"}})},
     ["A segment from the origin to (3, 4).", "A horizontal segment of length two.", "A segment along the diagonal."]),
    ("recursive_tree", {"$defs": {"node": {"type": "object", "properties": {
                           "name": S, "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "maxItems": 3}},
                           "required": ["name", "children"], "additionalProperties": False}},
                        "$ref": "#/$defs/node"},
     ["A small family tree, two generations.", "A folder with two subfolders, one of them with a file.",
      "A menu with two sections of two dishes."]),
    ("deep_nesting", obj({"a": obj({"b": obj({"c": obj({"d": S, "n": I})})})}),
     ["Put the word 'deep' and the number 4 at the bottom.", "Put 'core' and 7 at the bottom.",
      "Put 'leaf' and 0 at the bottom."]),
    ("escapes", obj({"quote": S, "path": S}),
     ["A sentence with \"double quotes\" inside, and a Windows path.", "A two-line quote (with a newline) and a Unix path.",
      "A quote with a backslash and a tab, and any path."]),
    ("unicode", obj({"word": S, "language": S, "meaning": S}),
     ["A Polish word with diacritics.", "A Japanese word.", "A Greek word."]),
    ("numbers", obj({"pi": N, "avogadro": N, "negative": N, "zero": I}),
     ["Fill in the constants; negative is minus one half.", "Fill in the constants; negative is -273.15.",
      "Fill in the constants; negative is any negative number."]),
    ("tool_like", obj({"name": {"enum": ["get_weather", "search", "calculator"]},
                       "arguments": obj({"query": S, "units": {"enum": ["metric", "imperial"]}}, required=["query"])}),
     ["Which tool call answers: weather in Oslo, in Celsius?", "Which tool call answers: what is 17 times 23?",
      "Which tool call answers: who wrote Solaris?"]),
    ("long_array", obj({"squares": {"type": "array", "items": I, "minItems": 20, "maxItems": 20}}),
     ["The squares of 1 to 20.", "The squares of 11 to 30.", "The squares of the first twenty even numbers."]),
    ("free_object", {"type": "object"},
     ["Any small JSON object about a cat.", "Any small JSON object about a bridge.", "Any small JSON object about tea."]),
    ("enum_array", obj({"days": {"type": "array", "items": {"enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]},
                                 "minItems": 1, "maxItems": 7}}),
     ["Which days is a weekend?", "Which days are working days?", "Which days start with the letter t?"]),
]


VERBS = ["book", "cancel", "change", "check", "find", "track"]
THINGS = ["flight", "hotel", "train", "order", "ticket", "table", "taxi", "payment", "account", "delivery"]
INTENTS = [f"{v}_{t}" for v in VERBS for t in THINGS]   # 60 values
LANGS = {"enum": ["pl", "en", "de", "fr", "it"]}

AGENT_CASES = [
    ("utterance_quote", obj({"utterance": S}),
     ["Napisz jedno zdanie po polsku, które kończy się cytatem w polskim cudzysłowie („…”).",
      "Schreibe einen deutschen Satz, der mit einem Zitat in Anführungszeichen („…“) endet.",
      "Écris une phrase en français qui se termine par une citation entre guillemets (« … »)."]),
    ("utterance_escapes", obj({"utterance": S}),
     ["Write two short lines separated by a line break; the second line contains a word in \"double quotes\".",
      "Scrivi due righe brevi separate da un a capo; la seconda contiene una parola tra \"virgolette\".",
      "Napisz dwie krótkie linie oddzielone znakiem nowej linii; druga zawiera słowo w \"cudzysłowie\"."]),
    ("translations", obj({"pl": S, "en": S, "de": S, "fr": S, "it": S}),
     ["Translate into all five languages: 'The train leaves in ten minutes.'",
      "Translate into all five languages: 'She said \"no\" and smiled.'",
      "Translate into all five languages: 'Where is the nearest pharmacy?'"]),
    ("dialogue", obj({"turns": {"type": "array", "items": obj({"speaker": {"enum": ["user", "assistant"]}, "text": S})}}),
     ["A four-turn dialogue in Polish about booking a table.", "Ein Dialog mit vier Wendungen über ein verspätetes Paket.",
      "Un dialogo di quattro battute su un treno in ritardo."]),
    ("intent_60", obj({"intent": {"type": "string", "enum": INTENTS}, "reply": S}),
     ["Classify and reply in Polish: 'Chciałbym odwołać rezerwację hotelu.'",
      "Classify and reply in French: 'Où est mon colis ?'",
      "Classify and reply in German: 'Ich möchte einen Tisch für vier reservieren.'"]),
    ("integers", obj({"numbers": {"type": "array", "items": I}}),
     ["The first twelve prime numbers.", "The numbers from -5 to 5.", "The years of the last six summer Olympic Games before 2024."]),
    ("summary", obj({"title": S, "summary": S, "language": LANGS}),
     ["Podsumuj po polsku w dwóch zdaniach, czym jest pamięć podręczna.",
      "Résume en deux phrases ce qu'est une marée.", "Riassumi in due frasi che cos'è un ghiacciaio."]),
    ("segments", obj({"segments": {"type": "array", "items": obj({"start": I, "end": I, "text": S})}}),
     ["Split 'Dzień dobry. Jak się masz? Do zobaczenia!' into sentences with character offsets.",
      "Split 'Guten Morgen. Wie geht es dir?' into sentences with character offsets.",
      "Split 'Bonjour ! Ça va ? À demain.' into sentences with character offsets."]),
    ("quotes", obj({"quotes": {"type": "array", "items": S}}),
     ["Trois citations célèbres en français, chacune entre guillemets « ».",
      "Trzy znane polskie przysłowia, każde w cudzysłowie „”.", "Three short English sayings, each in \"double quotes\"."]),
    ("quiz", obj({"question": S, "answers": {"type": "array", "items": S}, "correct": {"type": "array", "items": I}}),
     ["Pytanie quizowe po polsku z czterema odpowiedziami; podaj indeksy poprawnych.",
      "Una domanda a quiz in italiano con quattro risposte; indica gli indici corretti.",
      "Eine Quizfrage auf Deutsch mit vier Antworten; gib die Indizes der richtigen an."]),
]

BROKEN_TAILS = ("}", ")", ">", "]", '"', "”}", "“}", "»}")


def suspect_values(doc):
    """String values that end like a closing the model meant for the document, not for the text."""
    out = []
    def walk(v):
        if isinstance(v, str):
            t = v.rstrip()
            if t.endswith(BROKEN_TAILS) and not (t.endswith(")") and "(" in t) and not (t.endswith('"') and t.count('"') % 2 == 0) \
                    and not (t.endswith("]") and "[" in t) and not (t.endswith(">") and "<" in t) and not (t.endswith("}") and "{" in t):
                out.append(t[-12:])
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
    walk(doc)
    return out


def post(url, body, timeout):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read()), time.time() - t0
        except Exception:
            return e.code, {}, time.time() - t0


def run_case(a, name, schema, prompt, constrained):
    in_prompt = "\n\nAnswer with one JSON document that matches this JSON schema, and nothing else:\n" + json.dumps(schema, ensure_ascii=False)
    as_schema = constrained and a.mode == "schema"
    # The schema is in the prompt in every case, so the constrained and the free run answer the same prompt.
    # --blind leaves it out of the constrained prompt: the grammar alone then has to steer the model to the keys.
    body = {"messages": [{"role": "user", "content": prompt + ("\n\nAnswer as JSON." if (as_schema and a.blind) else in_prompt)}],
            "max_tokens": a.max_tokens, "temperature": a.temperature, "enable_thinking": a.thinking, "seed": a.seed}
    if as_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
    elif constrained:
        body["response_format"] = {"type": "json_object"}
    status, resp, wall = post(a.url, body, a.timeout)
    out = {"schema": name, "prompt": prompt, "constrained": constrained, "status": status, "wall_s": round(wall, 3)}
    if status != 200:
        out["error"] = (resp.get("error") or {}).get("message") if isinstance(resp.get("error"), dict) else str(resp)[:300]
        out["verdict"] = "refused" if status == 400 else "http_error"
        return out
    ch = resp["choices"][0]
    text = (ch.get("message") or {}).get("content") or ""
    m = resp.get("metrics") or {}
    out.update(finish=ch.get("finish_reason"), completion_tokens=(resp.get("usage") or {}).get("completion_tokens"),
               tok_s=m.get("decode_tok_per_s"), rows_changed=m.get("constraint_rows_changed"),
               drafts_refused=m.get("constraint_drafts_refused"), accept_len=m.get("mean_acceptance_length"),
               text=text if len(text) <= 2000 else text[:2000] + "...", overruled=bool(m.get("constraint_rows_changed")))
    try:
        doc = json.loads(text)
    except ValueError as e:
        out["verdict"] = "truncated" if ch.get("finish_reason") == "length" else "not_json"
        out["error"] = str(e)
        return out
    if jsonschema is not None:
        try:
            jsonschema.validate(doc, schema, format_checker=FORMATS)
        except jsonschema.ValidationError as e:
            # json_object promises one object, not the schema: a mismatch there is the model's, not the constraint's
            object_only = constrained and a.mode == "object" and isinstance(doc, dict)
            out["verdict"] = "ok_object" if object_only else "invalid"
            out["error"] = e.message[:300]
            return out
    out["verdict"] = "ok"
    sus = suspect_values(doc)
    if sus:
        out["suspect"] = sus
    return out


def summary(rows):
    n = len(rows)
    by = {}
    for r in rows:
        by[r["verdict"]] = by.get(r["verdict"], 0) + 1
    speeds = [r["tok_s"] for r in rows if r.get("tok_s")]
    toks = sum(r.get("completion_tokens") or 0 for r in rows)
    return {"cases": n, "verdicts": by, "median_tok_s": round(statistics.median(speeds), 2) if speeds else None,
            "completion_tokens": toks, "overruled_answers": sum(1 for r in rows if r.get("overruled")),
            "suspect_answers": sum(1 for r in rows if r.get("suspect")),
            "rows_changed": sum(r.get("rows_changed") or 0 for r in rows),
            "drafts_refused": sum(r.get("drafts_refused") or 0 for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--compare-free", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0, help="first N schemas only")
    ap.add_argument("--blind", action="store_true", help="constrained run without the schema in the prompt")
    ap.add_argument("--set", default="all", choices=["general", "agent", "all"])
    ap.add_argument("--mode", default="schema", choices=["schema", "object"],
                    help="object: response_format json_object, the schema only in the prompt")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cases = {"general": CASES, "agent": AGENT_CASES, "all": AGENT_CASES + CASES}[a.set]
    cases = cases[:a.limit] if a.limit else cases
    jobs = [(name, schema, p) for name, schema, prompts in cases for p in prompts]
    result = {"url": a.url, "set": a.set, "mode": a.mode, "blind": a.blind, "streams": a.streams, "temperature": a.temperature, "thinking": a.thinking,
              "max_tokens": a.max_tokens, "validated_with_jsonschema": jsonschema is not None}
    for label, constrained in [("constrained", True)] + ([("free", False)] if a.compare_free else []):
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.streams) as ex:
            rows = list(ex.map(lambda j: run_case(a, j[0], j[1], j[2], constrained), jobs))
        wall = time.time() - t0
        s = summary(rows)
        s["wall_s"] = round(wall, 1)
        s["aggregate_tok_s"] = round(s["completion_tokens"] / wall, 2) if wall > 0 else None
        result[label] = {"summary": s, "rows": rows}
        print(f"{label}: {json.dumps(s)}")
        for r in rows:
            if r["verdict"] not in ("ok",):
                print(f"  {r['verdict']:10s} {r['schema']}: {(r.get('error') or '')[:160]}")
            elif r.get("suspect"):
                print(f"  suspect    {r['schema']}: ...{r['suspect'][0]}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
    bad = [r for r in result["constrained"]["rows"] if r["verdict"] not in ("ok", "ok_object", "refused")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
