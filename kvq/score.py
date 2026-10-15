#!/usr/bin/env python3
"""Teacher-forced scoring: send a fixed token sequence, read back prompt_logprobs (top-K + actual token)."""
import argparse, json, sys, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True); ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=20)
ap.add_argument("--url", default="http://localhost:8001")
a = ap.parse_args()
def post(path, body):
    r = urllib.request.Request(a.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))
text = open(a.corpus, encoding="utf-8", errors="replace").read()
ids = post("/tokenize", {"model": "kvq", "prompt": text, "add_special_tokens": False})["tokens"][:a.tokens]
print(f"scoring {len(ids):,} tokens", flush=True)
r = post("/v1/completions", {"model": "kvq", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": a.k})
pl = r["choices"][0]["prompt_logprobs"]
out = {"n": len(ids), "ids": ids, "lp": [], "top": []}
for t, d in zip(ids, pl):
    if d is None: out["lp"].append(None); out["top"].append(None); continue
    out["lp"].append(d[str(t)]["logprob"])
    out["top"].append({k: v["logprob"] for k, v in d.items()})
json.dump(out, open(a.out, "w"))
print("saved", a.out)
