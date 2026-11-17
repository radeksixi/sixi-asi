#!/usr/bin/env python3
"""Can the model read a number from a picture? A check of image input with a known answer.

Each picture is white with one number in black, drawn in a 5x7 block font (the test a user of the service
proposed on 2026-11-17: 480x480, six digits, each 50x70 pixels). The model is asked for the number; the answer
must be exactly it. Run it against this engine and against llama-server with the same weights and projector:
a difference between the two is the engine's, a failure of both is the model's or the projector's.

  python3 scripts/image_digits_check.py --url http://127.0.0.1:8091 --count 15
  python3 scripts/image_digits_check.py --url http://127.0.0.1:8099 --count 15 --cell 20 --size 960
Standard library only (it writes the PNG itself). Exit status 0 when every number is read.
"""
import argparse
import base64
import json
import random
import struct
import sys
import urllib.request
import zlib

FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def png(number, size, cell):
    """A size x size 8-bit grayscale PNG with `number` centred; one font cell is `cell` pixels."""
    gap = cell
    w = len(number) * 5 * cell + (len(number) - 1) * gap
    x0, y0 = (size - w) // 2, (size - 7 * cell) // 2
    rows = [bytearray(b"\xff" * size) for _ in range(size)]
    for i, ch in enumerate(number):
        for r, line in enumerate(FONT[ch]):
            for c, bit in enumerate(line):
                if bit == "1":
                    x = x0 + i * (5 * cell + gap) + c * cell
                    for y in range(y0 + r * cell, y0 + (r + 1) * cell):
                        rows[y][x:x + cell] = b"\x00" * cell
    raw = b"".join(b"\x00" + bytes(row) for row in rows)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def ask(url, image, model):
    body = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
        {"type": "text", "text": "Which number does the picture show? Answer with the number only."}]}],
        "max_tokens": 24, "temperature": 0, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
    if model:
        body["model"] = model
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return (r["choices"][0]["message"].get("content") or "").strip(), (r.get("usage") or {}).get("prompt_tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8089")
    ap.add_argument("--count", type=int, default=15)
    ap.add_argument("--digits", type=int, default=6)
    ap.add_argument("--size", type=int, default=480, help="picture width and height in pixels")
    ap.add_argument("--cell", type=int, default=10, help="pixels per font cell (a digit is 5 x 7 cells)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="")
    ap.add_argument("--save", default="", help="folder to keep the pictures in")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    good = 0
    tokens = None
    for i in range(a.count):
        number = "".join(rnd.choice("0123456789") for _ in range(a.digits))
        image = png(number, a.size, a.cell)
        if a.save:
            import os
            os.makedirs(a.save, exist_ok=True)
            with open(os.path.join(a.save, "%s.png" % number), "wb") as f:
                f.write(image)
        answer, tokens = ask(a.url, image, a.model)
        ok = answer.strip(" .\n") == number
        good += ok
        print("%s -> %-24r %s" % (number, answer[:22], "ok" if ok else "WRONG"), flush=True)
    print("image digits: %d of %d read (%dx%d, digit %dx%d px, prompt tokens %s)" % (good, a.count, a.size, a.size, 5 * a.cell, 7 * a.cell, tokens))
    sys.exit(0 if good == a.count else 1)


if __name__ == "__main__":
    main()
