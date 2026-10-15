#!/usr/bin/env python3
"""gencmp.py REF TEST : per-window first-divergence index of greedy generations (of GEN tokens), by prompt depth."""
import json, sys, statistics as st
ref = json.load(open(sys.argv[1])); tst = json.load(open(sys.argv[2]))
print(f"ref={sys.argv[1]} test={sys.argv[2]}")
by = {}
for r, t in zip(ref, tst):
    assert (r["start"], r["n"]) == (t["start"], t["n"])
    k = next((i for i, (x, y) in enumerate(zip(r["tokens"], t["tokens"])) if x != y), len(r["tokens"]))
    by.setdefault(r["n"], []).append(k)
allk = []
for n, ks in sorted(by.items()):
    allk += ks
    print(f"  prompt {n:>6}: first divergence at token {ks}  median {st.median(ks)}  identical {sum(k==len(ref[0]['tokens']) for k in ks)}/{len(ks)}")
print(f"  ALL: median {st.median(allk)}, mean {st.mean(allk):.1f}, identical {sum(k==len(ref[0]['tokens']) for k in allk)}/{len(allk)}")
