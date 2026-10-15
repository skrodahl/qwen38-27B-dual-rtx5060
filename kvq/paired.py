#!/usr/bin/env python3
"""paired.py REF TEST [lo hi] : mean dNLL (test-ref) with 95% block-bootstrap CI (blocks of 512 tokens)."""
import json, random, sys, statistics as st
ref = json.load(open(sys.argv[1])); tst = json.load(open(sys.argv[2]))
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 1
hi = min(int(sys.argv[4]) if len(sys.argv) > 4 else 10**9, ref["n"], tst["n"])
d = [tst["lp"][i] and -(tst["lp"][i]) - -(ref["lp"][i]) for i in range(lo, hi)]
B = 512; blocks = [d[i:i+B] for i in range(0, len(d), B)]
mean = sum(d) / len(d)
random.seed(1); bs = []
for _ in range(2000):
    s = [random.choice(blocks) for _ in blocks]
    bs.append(sum(sum(b) for b in s) / sum(len(b) for b in s))
bs.sort()
print(f"{sys.argv[2]:>26} vs {sys.argv[1]:<18} [{lo}-{hi}] n={len(d):>6}  dNLL {mean:+.5f}  95% CI [{bs[50]:+.5f}, {bs[1949]:+.5f}]  ppl x{2.718281828**mean:.4f}")
