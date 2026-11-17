#!/usr/bin/env python3
"""Short spoken answers through sixi-speech: is a one-word answer heard in its language?

The recogniser has no input for the language, and a word of one syllable is too little for it to tell: a Polish
"nie" used to come back as "Yeah.". With `language` in the request the service now puts a phrase in that language
next to a clip of up to 2 s (Polish, German, Italian, French). This check says a few short answers per language
with the service's own voices and recognises them four ways: without `language`, with it, with it but with the
phrase turned off (`lead_in=0`), and with `certain=1` (a text only when it was heard the same way twice).

  python3 scripts/speech_short_check.py --url http://127.0.0.1:8088
  python3 scripts/speech_short_check.py --clips DIR --language pl     your own recordings: DIR/<what is said>.wav
Synthetic voices are easier to hear than people; the second form is the test that counts. Standard library only.
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request

ANSWERS = {
    "pl": ["Nie.", "Tak.", "Już.", "Dobrze.", "Nie wiem.", "Pomocy.", "Tak, proszę.", "Stop.", "Dość.", "Koniec."],
    "de": ["Nein.", "Ja.", "Stopp.", "Hilfe.", "Ich weiß nicht."],
    "it": ["No.", "Sì.", "Basta.", "Non lo so."],
    "fr": ["Non.", "Oui.", "Stop.", "Je ne sais pas."],
    "en": ["No.", "Yes.", "Stop.", "Help.", "I don't know."],
}
VOICES = {"pl": ["pl", "pl-female"], "de": ["de"], "it": ["it"], "fr": ["fr"], "en": ["en"]}


def plain(text):
    t = " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split()).replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def say(url, text, voice, speed):
    body = json.dumps({"input": text, "voice": voice, "response_format": "wav", "speed": speed}).encode()
    return urllib.request.urlopen(urllib.request.Request(url + "/v1/audio/speech", body, {"Content-Type": "application/json"}), timeout=120).read()


def hear(url, audio, name="clip.wav", **fields):
    b = "----sixi" + os.urandom(8).hex()
    parts = b""
    for k, v in fields.items():
        if v is not None:
            parts += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode()
    parts += ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\nContent-Type: application/octet-stream\r\n\r\n" % (b, name)).encode()
    parts += audio + ("\r\n--%s--\r\n" % b).encode()
    req = urllib.request.Request(url + "/v1/audio/transcriptions", parts, {"Content-Type": "multipart/form-data; boundary=" + b})
    t = time.time()
    text = json.loads(urllib.request.urlopen(req, timeout=300).read())["text"]
    return text, time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8088")
    ap.add_argument("--languages", default="pl,de,it,fr,en")
    ap.add_argument("--clips", default="", help="a folder of recordings named after what is said in them (nie.wav, tak_2.wav, nie wiem.wav)")
    ap.add_argument("--language", default="pl", help="with --clips: the language of the recordings")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ways = ["no language", "with language", "with language, phrase off", "with language, certain"]
    rows = []
    if a.clips:
        clips = []
        for f in sorted(os.listdir(a.clips)):
            said = re.sub(r"[_ ]*\d*$", "", os.path.splitext(f)[0]).replace("_", " ")
            with open(os.path.join(a.clips, f), "rb") as fh:
                clips.append((a.language, said, f, fh.read()))
    else:
        clips = []
        for lang in a.languages.split(","):
            for text in ANSWERS.get(lang, []):
                for voice in VOICES[lang]:
                    for speed in (0.9, 1.0, 1.15):
                        clips.append((lang, text, "%s %.2f" % (voice, speed), say(a.url, text, voice, speed)))
    for lang, text, what, audio in clips:
        row = {"language": lang, "said": text, "clip": what}
        for name, fields in zip(ways, ({}, {"language": lang}, {"language": lang, "lead_in": "0"}, {"language": lang, "certain": "1"})):
            row[name], row[name + " s"] = hear(a.url, audio, **fields)
        rows.append(row)
    print("right / a wrong word / nothing, by how the request was made")
    print("%-9s %-6s %-14s %-15s %-26s %-23s %s" % ("language", "clips", *ways, "seconds a clip (with language, median)"))
    for lang in sorted({r["language"] for r in rows}):
        rr = [r for r in rows if r["language"] == lang]
        cells = []
        for name in ways:
            right = sum(plain(r[name]) == plain(r["said"]) for r in rr)
            nothing = sum(not plain(r[name]) for r in rr)
            cells.append("%d/%d/%d" % (right, len(rr) - right - nothing, nothing))
        times = sorted(r[ways[1] + " s"] for r in rr)
        print("%-9s %-6d %-14s %-15s %-26s %-23s %.2f" % (lang, len(rr), *cells, times[len(times) // 2]))
    wrong = [r for r in rows if plain(r[ways[1]]) != plain(r["said"])]
    for r in wrong[:20]:
        print("  not heard with language: %-3s %-18s (%s) -> %r" % (r["language"], r["said"], r["clip"], r[ways[1]]))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "rows": rows}, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
