#!/usr/bin/env python3
"""Decode-path test: greedy-generate GEN tokens after fixed prompt windows of the corpus; save the token ids."""
import argparse, json, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gen", type=int, default=200)
ap.add_argument("--url", default="http://localhost:8001"); a = ap.parse_args()
def post(p, b):
    r = urllib.request.Request(a.url + p, json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))
ids = post("/tokenize", {"model": "kvq", "prompt": open("corpus.txt", encoding="utf-8", errors="replace").read(), "add_special_tokens": False})["tokens"]
# (start, length): windows of different depth; starts differ so the texts differ
W = [(s, n) for n in (4096, 32768, 90000) for s in (0, 100000, 200000, 300000, 400000, 500000)]
res = []
for s, n in W:
    r = post("/v1/completions", {"model": "kvq", "prompt": ids[s:s+n], "max_tokens": a.gen, "temperature": 0, "logprobs": 1})
    c = r["choices"][0]
    res.append({"start": s, "n": n, "tokens": c["logprobs"]["tokens"], "lp": c["logprobs"]["token_logprobs"]})
    print(s, n, "".join(c["logprobs"]["tokens"][:12]).replace("\n", " "), flush=True)
json.dump(res, open(a.out, "w"))
