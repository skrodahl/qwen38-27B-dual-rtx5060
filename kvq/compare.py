#!/usr/bin/env python3
"""compare.py REF.json TEST.json : per-position-bucket stats of TEST vs REF (teacher-forced, same tokens)."""
import json, math, sys
ref = json.load(open(sys.argv[1])); tst = json.load(open(sys.argv[2]))
assert ref["ids"][:min(ref["n"], tst["n"])] == tst["ids"][:min(ref["n"], tst["n"])], "token mismatch"
n = min(ref["n"], tst["n"])
edges = [1, 4096, 16384, 32768, 65536, 98304, 131072, 200001]
def kl(rt, tt):
    # KL(ref||test) over ref's top-K; test's missing ids floored at its own smallest top-K logprob
    floor = min(tt.values()); s = 0.0; mr = 0.0
    for k, lr in rt.items():
        p = math.exp(lr); mr += p
        s += p * (lr - tt.get(k, floor))
    return s / mr  # renormalised to the covered mass
print(f"ref={sys.argv[1]} test={sys.argv[2]}")
print(f"{'positions':>16} {'n':>7} {'ref NLL':>8} {'test NLL':>8} {'dNLL':>8} {'mean|dlp|':>9} {'top1 agree':>10} {'KL(topK)':>9}")
for lo, hi in zip(edges, edges[1:]):
    hi = min(hi, n)
    if lo >= hi: break
    a = b = ad = kk = ag = c = 0
    for i in range(lo, hi):
        if ref["lp"][i] is None: continue
        c += 1; a += -ref["lp"][i]; b += -tst["lp"][i]; ad += abs(ref["lp"][i] - tst["lp"][i])
        rt, tt = ref["top"][i], tst["top"][i]
        ag += max(rt, key=rt.get) == max(tt, key=tt.get); kk += kl(rt, tt)
    print(f"{lo:>7}-{hi:<8} {c:>7} {a/c:8.4f} {b/c:8.4f} {(b-a)/c:+8.4f} {ad/c:9.4f} {ag/c:10.4f} {kk/c:9.5f}")
