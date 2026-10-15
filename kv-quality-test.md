# How much does a compressed KV cache cost in quality? bf16 vs FP8 vs NVFP4 on Qwen 3.8 27B

**Short answer:** on this model and this hardware, FP8 KV is indistinguishable from bf16 KV
in everything I measured. NVFP4 KV costs roughly **0.2-0.5 % of perplexity**
(0.002-0.005 nats per token), a shift that is just large enough to detect with 90,000-110,000
scored tokens and that does **not** grow with context depth in my data. Greedy generations
diverge between *any* two configurations, including two bf16 runs that differ only in prefill chunk size,
so "the output is not bit-identical" is not evidence of anything. Everything below says how
that was measured, what the controls were, and what the test cannot show.

This is a measurement of *this* setup (one model, one checkpoint, one GPU pair, one vLLM build, two text
corpora). It is not a general statement about KV quantization.

## 1. Setup

| | |
|---|---|
| Model | Qwen 3.8 27B, NVFP4 weights, with the MTP head (`gittensor-NVFP4-MTP`). A hybrid model: Gated DeltaNet (linear-attention) layers plus full-attention layers. Only the full-attention layers carry a KV cache that grows with context. |
| Hardware | 2 x RTX 5060 Ti 16 GB (SM120), tensor-parallel 2, GPU clock locked to 300-2500 MHz (not relevant to numerics) |
| vLLM | `vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989` (v0.32.0rc1.dev100). Needed because NVFP4 KV on SM12x is in this nightly, not in 0.30.0. |
| Common flags | `--tensor-parallel-size 2 --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching --enable-chunked-prefill --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 --compilation-config {"cudagraph_mode":"PIECEWISE"} --max-logprobs 20` |
| Switched between runs | `--kv-cache-dtype auto` (bf16) / `fp8` / `nvfp4`; `--kv-cache-memory` 3.5e9 (bf16), 4.9e9 (fp8, nvfp4). Pools: bf16 208,310 tok, fp8 295,406 tok, nvfp4 520,104 tok. |
| Deliberately off | prefix caching (it would skip the prefill being measured), speculative decoding, KV offload. |
| Determinism | Two runs of the *same* config gave **bit-identical** logprobs at every position (bf16, fp8, nvfp4). So there is no run-to-run jitter: any difference between two configs is a real function of the config. |

The server is started fresh for every configuration by `kvq/run-server.sh <dtype> <kv-memory> <max-len> <chunk>`.

### Corpora

Two fixed token sequences, tokenized by the model's own tokenizer through the server, no special tokens, no chat template:

- **Corpus 1** (`corpus.txt`): this project's `STATUS.md` (0-89,851 tok) + `vllm-setup.md` (to 105,037) + Vim's `todo.txt` (to 194,013) + `options.txt` + `version9.txt`. First 200,000 tokens scored.
- **Corpus 2** (`corpus2.txt`): the same material reordered: Vim `todo.txt` (0-88,976), then `STATUS.md`, `vllm-setup.md`, then `options.txt`. First 200,000 tokens scored.

Corpus 2 exists because corpus 1 confounds depth with text type (project notes at shallow depth, Vim help files deep). Mean loss differs a lot by text type, so I reran it with the order swapped. The project notes are text the model cannot have seen; Vim's help files are public and may be in its training data.

## 2. Methods

### 2.1 Teacher-forced scoring (prefill path)

`score.py` sends the token sequence as one `/v1/completions` request with `max_tokens=1`
and `prompt_logprobs=20`. vLLM returns, for every position, the log-probability of the actual next token
and the top-20 alternatives. All positions are processed through chunked prefill (2,048 tokens per chunk),
so each chunk attends to the previous chunks **through the quantized KV cache**.

For two configurations A (reference) and B, on the same tokens:

- **dNLL**: mean over positions of `(-logp_B) - (-logp_A)`, in nats per token. `exp(dNLL)` is the relative perplexity change. This is exact (it uses the real token's log-probability).
- **95 % confidence interval** by block bootstrap: 512-token blocks, 2,000 resamples, because neighbouring tokens are not independent. (`paired.py`)
- **mean |dlogp|**, **top-1 agreement**, **KL(top-20)**: per-position disagreement measures (`compare.py`).
  The KL is **an approximation**: vLLM only returns the top 20, so it is KL(A||B) over A's top-20, renormalised to the mass they cover, with any token missing from B's top-20 assigned B's smallest top-20 logprob. It is biased and only indicative. A full-vocabulary KL would need the raw logits from inside vLLM.

### 2.2 The noise-floor control

The question "is the difference caused by the KV dtype?" needs a baseline for "how much does *any*
harmless change move the numbers?". The control is the **same dtype with a different prefill chunk size**
(`--max-num-batched-tokens` 1024 and 4096 instead of 2048). That changes the order of floating-point operations and
nothing else. Every KV-dtype effect is read against this.

### 2.3 Decode-path test

Teacher-forced scoring never runs the decode kernel, and decode is where NVFP4 uses its own attention kernel.
`gen.py` therefore generates **200 tokens greedily** (temperature 0) after 18 fixed prompts
(6 corpus offsets x prompt lengths 4,096 / 32,768 / 90,000 tokens), in four configurations
(bf16, bf16 with chunk 1024, fp8, nvfp4). `gencmp.py` records the token index at which two generations first differ.

### 2.4 Depth coverage

- bf16 KV only fits **up to ~90,000 tokens** on 16 GB cards at this pool size, so bf16 is compared against fp8/nvfp4 on positions 1 to 90,000 of the same 200K-token prompts.
- Beyond 90,000 there is no bf16 reference; the comparison there is **NVFP4 vs FP8** (FP8 is itself within noise of bf16, see below).

## 3. Results

### 3.1 Noise floor and determinism

| Control | dNLL (nats/tok) | 95 % CI | perplexity |
|---|---:|---|---:|
| bf16 chunk 1024 vs bf16 chunk 2048, positions 1-90K | -0.0012 | [-0.0037, +0.0014] | x0.9988 |
| bf16 chunk 4096 vs bf16 chunk 2048, positions 1-90K | -0.0012 | [-0.0036, +0.0013] | x0.9988 |
| fp8 chunk 1024 vs fp8 chunk 2048, 1-90K (corpus 1) | -0.0020 | [-0.0047, +0.0008] | x0.9980 |
| fp8 chunk 1024 vs fp8 chunk 2048, 90K-200K (corpus 1) | +0.0005 | [-0.0019, +0.0028] | x1.0005 |
| fp8 chunk 1024 vs fp8 chunk 2048, 1-89K (corpus 2) | +0.0003 | [-0.0024, +0.0028] | x1.0003 |
| fp8 chunk 1024 vs fp8 chunk 2048, 105K-200K (corpus 2) | +0.0005 | [-0.0020, +0.0030] | x1.0005 |

Changing only the chunk size moves the mean by up to ~0.002 nats and the CI half-width is ~0.0025.
**A shift smaller than about 0.003 nats cannot be resolved with this method.**

### 3.2 bf16 vs FP8 vs NVFP4, positions 1-90,000 (corpus 1, project notes)

Reference: bf16 KV. (Absolute: perplexity 4.058 on these 90K tokens under bf16.)

| Positions | FP8: dNLL [95 % CI] | NVFP4: dNLL [95 % CI] |
|---|---|---|
| 1 - 32,768 | -0.0039 [-0.0081, +0.0004] | +0.0022 [-0.0022, +0.0063] |
| 32,768 - 90,000 | +0.0031 [+0.0004, +0.0060] | +0.0036 [+0.0009, +0.0063] |
| **1 - 90,000 (all)** | **+0.0006 [-0.0019, +0.0030]** (x1.0006) | **+0.0031 [+0.0007, +0.0055]** (x1.0031) |

FP8 over the whole range is inside the noise floor. NVFP4 over the whole range is just outside it (+0.31 % perplexity).
The 32K-90K slice shows FP8 marginally positive too, with its CI barely excluding zero; I do not read that as a robust FP8 effect.
The 1-32K slice for FP8 is negative, which is noise.

### 3.3 NVFP4 vs FP8 on the deeper positions (corpus 1, mostly Vim help text beyond 105K)

Reference: FP8 KV. Control: FP8 at chunk 1024.

| Positions | NVFP4 vs FP8: dNLL [95 % CI] | Control (fp8 chunk 1024): dNLL [95 % CI] |
|---|---|---|
| 1 - 90,000 | +0.0026 [+0.0001, +0.0049] | -0.0020 [-0.0047, +0.0008] |
| 90,000 - 131,072 | +0.0055 [+0.0024, +0.0091] | +0.0004 [-0.0032, +0.0042] |
| 131,072 - 200,000 | +0.0051 [+0.0018, +0.0084] | +0.0006 [-0.0024, +0.0035] |
| **90,000 - 200,000** | **+0.0053 [+0.0028, +0.0077]** (x1.0053) | +0.0005 [-0.0019, +0.0028] |

### 3.4 Is it depth, or is it the text? (corpus 2: order swapped)

Reference: FP8 KV. Corpus 2 puts the Vim help text at shallow depth and the project notes deep.

| Positions | Text | NVFP4 vs FP8 [95 % CI] | Control [95 % CI] |
|---|---|---|---|
| 1 - 89,000 | Vim help (shallow) | **+0.0035 [+0.0011, +0.0060]** (x1.0035) | +0.0003 [-0.0024, +0.0028] |
| 89,000 - 105,000 | project notes | +0.0003 [-0.0063, +0.0070] | -0.0005 [-0.0063, +0.0058] |
| 105,000 - 200,000 | project notes (deep) | +0.0020 [-0.0004, +0.0044] (x1.0020) | +0.0005 [-0.0020, +0.0030] |

Put side by side, NVFP4's extra loss is about **+0.0035-0.0055 on Vim help text wherever it sits**
(shallow in corpus 2, deep in corpus 1) and about **+0.002-0.003 on the project notes wherever they sit**
(shallow in corpus 1: +0.0026; deep in corpus 2: +0.0020, CI includes zero). The shift tracks the
**text**, not the depth. There is no sign in these data that it grows with context length, but with CIs this wide
I cannot exclude a small depth effect either.

### 3.5 Per-position disagreement measures do **not** separate the dtypes

Positions 1-90,000, reference bf16 (`results-raw.txt`):

| Comparison | mean abs dlogp | top-1 agreement | KL(top-20), approx. |
|---|---:|---:|---:|
| bf16 chunk 1024 vs bf16 | 0.18-0.22 | 87-90 % | 0.049-0.067 |
| bf16 chunk 4096 vs bf16 | 0.18-0.23 | 88-91 % | 0.050-0.069 |
| fp8 vs bf16 | 0.18-0.22 | 88-90 % | 0.050-0.067 |
| nvfp4 vs bf16 | 0.19-0.23 | 87-90 % | 0.054-0.073 |

All four rows are nearly the same, **including the control that changes no precision at all**. On this hybrid
model, ordinary floating-point reordering (chunk size) already moves individual token probabilities by about as
much as the KV dtype does, and top-1 flips on roughly 1 token in 10 for either reason. The text is high-entropy
(mean loss 1.3-2.3 nats/token), so many positions are near-ties. NVFP4 is ~5-10 % higher on every one of these
columns, which is consistent with the dNLL result but is well inside what the control spans.
Only the **paired mean shift** (3.2-3.4), which averages over ~100,000 positions, resolves a difference.

A leftover oddity: in the first 64 tokens, FP8 vs bf16 is much closer (mean abs dlogp 0.064) than NVFP4 vs bf16 (0.25). That
window is tiny and I did not pursue it.

### 3.6 Decode-path test: greedy generations diverge for every pair

First index (0-199) at which two greedy 200-token generations differ, 18 prompts per row:

| Pair | median | mean | identical to the end |
|---|---:|---:|---:|
| bf16 vs bf16 chunk 1024 (**control**) | 9.0 | 14.9 | 0 / 18 |
| bf16 vs fp8 | 15.0 | 17.4 | 0 / 18 |
| bf16 vs nvfp4 | 8.0 | 12.1 | 0 / 18 |
| fp8 vs nvfp4 | 10.0 | 13.7 | 0 / 18 |

Every pair diverges within a few tens of tokens, including the control. The NVFP4 numbers (median 8-10) are close to
the control (9) and FP8 is, if anything, later (15). The test **cannot** tell the dtypes apart, and "NVFP4 outputs differ from bf16 outputs" is
equally true of bf16 against bf16. What it does show is that decode ran without gross failure on the NVFP4 path: all 18 generations ran the full 200 tokens, none was empty or garbage, and none looped worse than the other dtypes (largest repeated-8-gram count in one generation: bf16 6, nvfp4 9, fp8 14; the outputs mostly continue the prompt's own formatting, which is repetitive in Vim help text). Checked by that crude metric plus reading samples, at prompt depths up to 90,000 tokens.

## 4. What this does and does not show

**Supported by the data**

1. FP8 KV vs bf16 KV: no measurable quality difference at the resolution of this method (about 0.003 nats).
2. NVFP4 KV vs FP8 KV: a small, real loss. Between +0.002 and +0.0055 nats/token depending on the text, i.e. **about +0.2 % to +0.55 % perplexity**; it exceeds the chunk-size noise on the Vim text and on the combined 1-90K range, and is not resolved on the project notes alone at depth.
3. No evidence in these runs that the NVFP4 loss grows with context depth up to 200K tokens.
4. Run-to-run determinism is exact, so there is no hidden random component.
5. NVFP4 decode at depth showed no gross failure (18 of 18 generations completed, none empty, none degenerate by a crude repetition check).

**Not shown, and why**

- **Task quality.** Perplexity of text is a proxy. I did not measure coding, tool-calling, or agent-loop success rates under NVFP4 vs FP8; a 0.3 % perplexity shift is small, but whether it matters for a given task is a separate experiment.
- **Rare-token and long-range failures.** An average hides a tail: a mean shift of 0.003 nats could be a uniform nudge or a few rare, large errors (the "missing parenthesis" worry). Average loss cannot tell those apart. Needle-in-a-haystack at 250K passed on both dtypes earlier in this project (and is a weak test by itself).
- **Decode-kernel accuracy in numbers.** The decode path was only checked by generation divergence and fluency, which has no resolving power here. A teacher-forced score *through the decode kernel* would need changes inside vLLM.
- **Full-vocabulary KL.** Only a top-20 approximation was available.
- **Generality.** One model, one checkpoint, one vLLM nightly, two text types, one 200K prompt per corpus (so effectively n=1 per text region; the bootstrap CI reflects token-level, not document-level, sampling). Vim's help text may be in the training data.
- **bf16 beyond 90K.** It does not fit, so the deep comparison is against FP8 only.
- **Resolution.** Anything below about 0.003 nats is indistinguishable from chunk-size noise, in either direction.

## 5. Reproducing

Scripts and text results are in [`kvq/`](kvq/): `run-server.sh` (private vLLM instance on port 8001), `score.py`,
`paired.py`, `compare.py`, `gen.py`, `gencmp.py`, and the raw outputs `results-raw.txt` (per-position measures),
`paired-raw.txt` (paired dNLL, corpus 1), `reordered-paired-raw.txt` (corpus 2) and `gen-results-raw.txt` (decode test).

**Not included:** the per-position logprob dumps (~120 MB each) and the two corpus files. The corpora contain the
author's internal project notes (`STATUS.md`), which are not published here. To reproduce, build your own
`corpus.txt` from any long text (the scripts read `corpus.txt` in the working directory) and expect different absolute
numbers; the method, not the exact text, is what transfers. The depth/text-type confound in section 3.4 is the reason to
test with the text order swapped as well.

Procedure, per configuration: start the server with `./run-server.sh <auto|fp8|nvfp4> <kv-cache-memory> <max-model-len> <chunk>`;
run `./score.py --corpus corpus.txt --tokens 200000 --out <file>`; compare two outputs with
`./paired.py REF.json TEST.json [lo hi]`. The model path and image tag in `run-server.sh` are this machine's.
