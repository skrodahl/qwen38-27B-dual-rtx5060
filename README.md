# Qwen 3.8 27B NVFP4 with NVFP4 KV cache on vLLM 0.32.0

**A 27B model with its full 262K context on two midrange 16 GB cards: stock
vLLM, no patches, 78 tok/s on prose and 125 tok/s on structured output
(measured with fp8 KV; NVFP4 KV runs within about 5 % of that).**

## Highlights

- **Hardware:** two RTX 5060 Ti 16 GB, tensor parallel. Midrange consumer cards, no 5090.
- **Model:** Qwen 3.8 27B, NVFP4 weights, NVFP4 KV cache, MTP speculative decoding (k=4).
- **Speed:** decode up to **125 tok/s** on structured output (JSON, code, tool calls),
  **78 tok/s** on prose, against 42 tok/s with MTP off. Measured with fp8 KV;
  NVFP4 KV runs within about 5 %. About 4 % lower with the 2500 MHz clock cap
  the rig now uses (see [Power](#power)).
- **Context:** the full **262K** window, with a KV cache of about **475K tokens**
  (1.81× the window), so nearly two full-context sessions fit at once.
  (About 451K with KV offload switched on.)
- **Checked, not assumed:** a 250K needle test, a bf16 / FP8 / NVFP4
  [KV quality comparison](kv-quality-test.md) (NVFP4 costs roughly 0.2-0.5 %
  perplexity), and a full MTP sweep from k=0 to 7.
- **Optional:** KV offload to host RAM, so a session pushed out of the GPU
  comes back in seconds instead of being prefilled again. A commented-out block in the recipe.
- **Stock and reproducible:** the official vLLM 0.32.0 Docker image and one
  Compose file. No patches.

**Who it's for:** anyone who wants long-context agentic or coding work on
mid-range consumer GPUs and is comfortable with Docker and vLLM. It is tuned for
**one user, or a few streams** (`max-num-seqs 2`); more concurrent streams make
the per-stream rate collapse on these cards, so it is not a many-user server.

- **What:** Qwen 3.8 27B (NVFP4 weights, NVFP4 KV cache) on 2 × RTX 5060 Ti,
  tensor parallel, with MTP speculative decoding, on a headless Ubuntu 26.04
  box.
- **How:** the official vLLM 0.32.0 Docker image and one Compose file. The full
  recipe is at the bottom. The tuning that matters is `max-num-batched-tokens
  4096`, FlashInfer autotune off, and MTP k=4.
- **Why:** to show that full-context agentic work doesn't need a 5090. It's
  verified with a 250K needle test, measured against a no-MTP baseline, and
  comes with the gotchas that cost context or crash the server.
- **NVFP4 vs FP8 KV, in short:** NVFP4 KV (in vLLM 0.32.0, 2026-10-18) gives a
  **+70 % KV pool** (474,032 vs 278,927 tokens) for about **−5 % steps/s**, and
  costs roughly **0.2-0.5 % perplexity** against bf16, with FP8 indistinguishable
  from bf16: see the [KV-cache quality test](kv-quality-test.md). It also makes
  **KV offload to host RAM** worth having; the recipe has a commented-out block.
- **Which numbers are which:** the decode, depth and k-sweep tables below were
  measured with **FP8 KV on vLLM 0.30.0**, the baseline. The NVFP4 and offload
  results, and the quality test, were measured on the **pre-release nightly**
  (v0.32.0rc1.dev100), same code as the tagged 0.32.0 for this feature. See
  [What would make this faster](#what-would-make-this-faster).

## Contents

- [Model](#model) · [Hardware](#hardware) · [Software](#software)
- [Performance](#performance)
  - [Decode speed](#decode-speed) — MTP k=4 against MTP off
  - [Decode at depth](#decode-at-depth) — why tok/s falls as context fills
  - [Choosing k: the full MTP sweep](#choosing-k-the-full-mtp-sweep) — k=0..7, and where each workload peaks
  - [Power](#power) · [Long context: 250K needle test](#long-context-250k-needle-test) · [Time to first token](#time-to-first-token)
- [What didn't work, and other gotchas](#what-didnt-work-and-other-gotchas) — 8192 batched tokens, TRITON_ATTN, full CUDA graphs, async scheduling
- [Tips and tricks](#tips-and-tricks) — what each knob does, and how to test a change without fooling yourself
- [What would make this faster](#what-would-make-this-faster) — fused draft decode, NVFP4 KV (in 0.32.0, measured on nightly), KV offload to RAM, DSpark
- [vLLM recipe (Docker Compose)](#vllm-recipe-docker-compose) — quick start and the whole config, with commented-out NVFP4 KV + offload

## Model

[gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/commit/0cc27958cefbbe231782ec8511de8c4eb5233348)
(pinned to commit `0cc2795`)

Pinned because this is the latest release of the checkpoint that still
includes the **MTP head**. Later commits don't include it, and without it
`--speculative-config` MTP won't work: decode drops to the ~42 tok/s no-MTP
baseline.

**Downloading the pinned commit** (~18.8 GB; public, so no token needed). Use
the full commit hash. Downloading `main` gets you the build without MTP.

With the Hugging Face CLI (`pip install -U huggingface_hub`, which provides `hf`):

```bash
hf download gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --revision 0cc27958cefbbe231782ec8511de8c4eb5233348 \
  --local-dir ~/ai/models/gittensor-NVFP4
```

With plain `wget`, fetching the file list at that commit:

```bash
REPO=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
REV=0cc27958cefbbe231782ec8511de8c4eb5233348
mkdir -p ~/ai/models/gittensor-NVFP4 && cd ~/ai/models/gittensor-NVFP4
for f in config.json generation_config.json hf_quant_config.json \
         model.safetensors.index.json model-00001-of-00003.safetensors \
         model-00002-of-00003.safetensors model-00003-of-00003.safetensors \
         tokenizer.json tokenizer_config.json vocab.json merges.txt \
         chat_template.jinja preprocessor_config.json processor_config.json \
         video_preprocessor_config.json; do
  wget -c "https://huggingface.co/$REPO/resolve/$REV/$f"
done
```

`-c` resumes a partial download, so you can simply rerun the loop if it gets
interrupted. Or skip the download altogether and let vLLM fetch the pinned
commit itself (see [the recipe](#vllm-recipe-docker-compose)).

## Hardware

| Component   | Details                     |
|-------------|-----------------------------|
| CPU         | AMD Ryzen 7 9800X3D         |
| GPUs        | 2 × RTX 5060 Ti 16 GB       |
| Motherboard | MSI MPG X670E CARBON WIFI   |
| RAM         | 64 GB DDR5                  |
| Storage     | NVMe Gen4                   |
| PCIe        | Gen5 x8 / x8 (cards are x8 electrically; GPU0 sits in an x16 slot) |
| GPU power   | Capped at 150 W per card (stock max 198 W) |
| GPU clock   | Uncapped for the tables below; since 2026-10-14 the rig runs with a 2500 MHz cap (see [Power](#power)) |

## Software

| Component     | Version              |
|---------------|----------------------|
| OS            | Ubuntu 26.04 LTS     |
| Kernel        | 7.0.0-31-generic     |
| NVIDIA driver | 595.91.07            |
| CUDA          | 13.2                 |
| vLLM          | 0.32.0 recipe (official `vllm/vllm-openai:v0.32.0` image, stock, no patches); fp8 baseline tables measured on 0.30.0, NVFP4 results on the 0.32.0 nightly |

## Performance

All runs in this section (the fp8 baseline): vLLM 0.30.0, fp8 KV cache, `max-num-seqs 2`, `max-model-len 262,144`.

### Decode speed

The operating config, MTP k=4, against the same workload with MTP off. Both
runs 2026-09-23, decode tok/s excludes prefill.

| Output type | No MTP tok/s | MTP k=4 tok/s | Speed-up | Steps/s (no MTP / MTP) | Acceptance (MTP) | Reasoning tokens (no MTP / MTP) |
|-------------|-------------:|--------------:|---------:|-----------------------:|-----------------:|--------------------------------:|
| Prose       | 41.99        | 78.31         | 1.9×     | 41.92 / 28.57          | 2.730            | 342 / 347                       |
| Structured  | 41.99        | 124.89        | 3.0×     | 41.93 / 28.55          | 4.348            | 165 / 166                       |
| Mixed       | 41.92        | 81.61         | 1.9×     | 41.86 / 28.56          | 2.847            | 519 / 525                       |

| Config  | KV cache pool | Headroom over max-model-len |
|---------|--------------:|----------------------------:|
| No MTP  | 351,618 tok   | +89,474                     |
| MTP k=4 | 278,927 tok   | +16,783                     |

**MTP k=4** (4 speculative tokens) is the setting this config runs. It is the
peak for prose and for mixed prose+JSON output; a higher k trades prose speed
for structured speed. See "Choosing k" below for the full k=0..7 sweep. Decode
speed is the only metric that differs between the MTP and non-MTP runs.

Without MTP, every output type decodes at the same speed, since nothing is
drafted. That is about 88 % of the theoretical ceiling set by memory bandwidth
(~47 tok/s).

The arithmetic ceiling at k=4 is 5 tokens per step, or ~143 tok/s; structured
output measures 125, so it fills 4.35 of the 5 slots. (At k=6 the ceiling is
~176 tok/s and highly repetitive output has been seen to reach ~160.)

### Decode at depth

Decode speed is not a constant: it falls as the context fills. Measured at
three depths, decode only (prefill excluded from the tok/s). **These tables are
the k=6 run**, kept because the per-position detail below was captured there;
the same depths for every k, including the operating k=4, are in "Choosing k".

| Depth (context tokens) | Prose tok/s | Structured tok/s | Steps/s | Acceptance, prose / structured |
|-----------------------:|------------:|-----------------:|--------:|-------------------------------:|
| 0                      | 74.23       | 130.48           | 25.21   | 2.936 / 5.207                  |
| 32,768                 | 67.73       | 127.04           | 23.79   | 2.862 / 5.348                  |
| 131,072                | 55.99       | 103.23           | 20.55   | 2.735 / 5.085                  |

At 128K of context, prose decodes at 75 % of its empty-context speed and
structured at 79 %. The depth-0 numbers differ by ~2 % from the Decode speed
table above because they are a different run; that is ordinary run-to-run
variation, not a config difference.

**Why it falls.** Decode speed is the product of two terms:

    tok/s = steps/s x acceptance

Both decay with depth, but the step rate does almost all of it.

- **Steps/s: −18 % at 128K.** Every decode step reads the model weights once
  *plus the entire KV cache*. The weights are fixed: 18.8 GB NVFP4, ~8.75 GiB
  per GPU at TP=2. The KV cache is not. This pool is 4.78 GiB per worker for
  272,533 tokens, so ~18.4 KiB per token per GPU, and 131,072 tokens adds
  ~2.30 GiB to the bytes every single step has to read. That is 26 % more
  traffic for the same weights, and decode is memory-bandwidth-bound, so the
  step rate should drop by about that much. It does:

  | Depth   | KV bytes/GPU | Predicted steps/s | Measured |
  |--------:|-------------:|------------------:|---------:|
  | 0       | 0            | 25.21 (reference) | 25.21    |
  | 32,768  | 0.575 GiB    | 23.65             | 23.79    |
  | 131,072 | 2.30 GiB     | 19.96             | 20.55    |

  This is the same mechanism that makes prefill slow down with depth (every
  new token attends to all the earlier ones), showing up on the decode side.

- **Acceptance: −7 % prose, −2 % structured.** The draft head also gets
  slightly worse at depth, but it is a minor term. Per-position acceptance
  (the probability that draft token 1, 2, ... 6 is accepted; acceptance length
  is 1 + their sum):

  | Depth   | Prose                            | Structured                       |
  |--------:|----------------------------------|----------------------------------|
  | 0       | 0.75 0.51 0.31 0.18 0.11 0.08    | 0.91 0.83 0.72 0.67 0.59 0.48    |
  | 32,768  | 0.73 0.51 0.29 0.16 0.11 0.06    | 0.96 0.83 0.74 0.70 0.62 0.50    |
  | 131,072 | 0.69 0.42 0.26 0.16 0.12 0.07    | 0.90 0.79 0.74 0.62 0.58 0.47    |

  The prose loss is concentrated in draft positions 2 and 3 (0.51 → 0.42,
  0.31 → 0.26); the tail was already near zero and has nothing left to lose.
  Structured output is within noise of flat — at 32K it is actually the best
  of the three. Predictable output stays predictable at depth.

**Steps/s is identical for prose and structured at every depth** (25.21/25.21,
23.79/23.79, 20.55/20.54). The cost of a step is set by the model and the
depth, not by what is being generated. Content only moves acceptance. That is
what makes the two-term model above usable: the terms are independent.

**Without MTP, measured at the same depths** (pool 351,618, 5.47 GiB/worker):

| Depth (context tokens) | Prose tok/s | Structured tok/s | Steps/s |
|-----------------------:|------------:|-----------------:|--------:|
| 0                      | 41.99       | 42.00            | 41.90   |
| 32,768                 | 39.98       | 39.98            | 39.90   |
| 131,072                | 34.66       | 34.66            | 34.56   |

Prose and structured are identical at every depth, to 0.01 tok/s, because
nothing is drafted: one step, one token, whatever the content. The same
weights-plus-cache model predicts the decay here too — this pool is 16.3 KiB
per token per GPU, so 131,072 tokens add ~2.04 GiB to the ~8.75 GiB of
weights, predicting 33.98 steps/s against 34.56 measured.

**The MTP speed-up survives depth**, with a small erosion from acceptance:

| Depth   | Prose speed-up | Structured speed-up |
|--------:|---------------:|--------------------:|
| 0       | 1.77×          | 3.11×               |
| 32,768  | 1.69×          | 3.18×               |
| 131,072 | 1.62×          | 2.98×               |

The step rate falls by nearly the same fraction either way (−17.5 % without
MTP, −18.5 % with it at 128K), which is the bandwidth term and it is common to
both. What erodes the ratio is the acceptance term, and only for prose.

Practically: a 130K-deep agentic session still decodes prose at ~56 tok/s and
JSON at ~103 tok/s, against ~35 tok/s without MTP at the same depth.

### Choosing k: the full MTP sweep

Every k from 0 to 7, same config, same workload, one vLLM restart each.
`decode` and `decode_depth` stages, fp8 KV, `max-num-seqs 2`, len 262,144.

| k | Steps/s | Prose tok/s | Structured tok/s | Mixed tok/s | Acceptance, prose / structured | KV pool |
|--:|--------:|------------:|-----------------:|------------:|-------------------------------:|--------:|
| 0 | 41.92   | 41.99       | 41.99            | 41.92       | — (no drafting)                | 351,618 |
| 1 | 35.94   | 63.91       | 70.27            | 63.41       | 1.78 / 1.95                    | 290,655 |
| 2 | 33.26   | 75.21       | 95.40            | 77.54       | 2.25 / 2.86                    | 285,321 |
| 3 | 30.74   | 78.10       | 108.73           | **83.01**   | 2.53 / 3.52                    | 284,319 |
| 4 | 28.57   | **78.31**   | 124.89           | 81.61       | 2.73 / 4.35                    | 278,927 |
| 5 | 26.77   | 72.66       | 135.58           | 79.03       | 2.70 / 5.03                    | 274,815 |
| 6 | 25.19   | 73.09       | 135.36           | 74.67       | 2.89 / 5.33                    | 272,533 |
| 7 | 23.80   | 70.84       | **138.08**       | 76.50       | 2.96 / 5.76                    | 268,596 |

**Mixed** is one prompt that answers in prose and then emits a JSON summary —
the closest stand-in here for a real agent turn. It peaks at **k=3-4**, with
prose, not at k=5-7 with structured. Treat the column as indicative rather than
exact: sampling is greedy, so each k produced one fixed output, and the
reasoning-token counts across these runs ranged from 275 to 616. The shape
(rise to k=3-4, slow decline after) is consistent across all three workloads,
but any single cell is one sample.

**The step rate is a straight line in k.** Fitting the measured step times
gives:

    step time = 25.4 ms + 2.37 ms x k        (steps/s = 1000 / step time)

Every k from 1 to 7 lands within 0.2 steps/s of that line. The 25.4 ms base is
a plain forward pass (23.9 ms measured at k=0) plus ~1.5 ms of fixed drafting
overhead, and each draft position costs a flat 2.37 ms after that. So the
question for any k is only: does draft position k get accepted often enough to
earn 2.37 ms?

**Prose peaks at k=4, structured keeps climbing.** Prose gains +22.0, +11.3,
+2.9, +0.2 tok/s for k=1..4, then loses ground: acceptance stops rising (2.73
at k=4, 2.70 at k=5) while every step keeps getting longer. Structured keeps
filling the extra slots — per-position acceptance is still 0.46 at position 7 —
but its gains flatten too: 135.58, 135.36, 138.08 for k=5,6,7, against +16.2
from k=3 to k=4.

**Memory is not a reason to pick a low k.** Turning MTP on at all costs 61K
tokens of KV pool (351,618 -> 290,655): that is the draft head itself. Each
position after that costs only 3-5K. k=7 still leaves 6,452 tokens of headroom
over the full 262,144 context.

**At depth the choice sharpens.** Same runs, decode tok/s at three depths:

| k | Prose @0 | @32K | @131K | Structured @0 | @32K | @131K |
|--:|---------:|-----:|------:|--------------:|-----:|------:|
| 0 | 41.99    | 39.98 | 34.66 | 42.00        | 39.98 | 34.66 |
| 1 | 63.58    | 59.44 | 53.30 | 70.41        | 67.40 | 58.10 |
| 2 | 76.15    | 72.63 | 61.99 | 95.43        | 87.24 | 79.62 |
| 3 | 78.12    | 72.08 | 66.97 | 109.70       | 105.13 | 92.36 |
| 4 | 80.10    | 69.60 | **68.83** | 123.33   | 115.61 | 92.41 |
| 5 | 73.39    | 65.79 | 59.02 | 135.05       | 123.14 | 101.67 |
| 6 | 74.23    | 67.73 | 55.99 | 130.48       | 127.04 | 103.23 |
| 7 | 70.39    | 84.49* | 55.48 | 136.12      | 125.38 | 108.31 |

Prose at 131K peaks at k=4 (68.83) and falls 14 % by k=5 — a sharper penalty
than the 7 % it costs at depth 0. Structured at 131K keeps climbing all the way
to k=7 (108.31). So the prose/structured trade-off *widens* with depth. Here is
why, in three steps.

**Step 1 — the step-time law is depth-dependent.** Fit `base + cost x k` to the
measured step times at each depth separately (k=1..7, prose):

| Depth   | Base step | Cost per draft position | Fit |
|--------:|----------:|------------------------:|-----|
| 0       | 25.35 ms  | 2.37 ms                 | all k within 0.2 steps/s |
| 32,768  | 26.24 ms  | 2.61 ms                 | +4 % base, +10 % per position |
| 131,072 | 29.36 ms  | 3.19 ms                 | +16 % base, **+35 % per position** |

The base grows because every decode step re-reads the KV cache along with the
weights, and the cache grows with depth. But the *per-position* cost grows
faster — and that is the part that matters here. A draft pass is itself a
forward pass through the draft head, so it also reads the growing cache. **Deep
context makes each draft token more expensive, not just each step.**

**Step 2 — acceptance does not improve to compensate.** Prose acceptance is
essentially depth-independent (2.80 at depth 0 and 2.88 at 131K for k=4; 2.95
and 2.89 for k=7). The draft head is no better at guessing deep in a context
than shallow in one. So at depth you pay 35 % more per position and get the
same number of tokens back.

**Step 3 — do the arithmetic for one case.** k=4 against k=7 at 131K, prose:

    k=4:  step 40.83 ms -> 24.5 steps/s x 2.876 accepted = 68.8 tok/s
    k=7:  step 51.71 ms -> 19.3 steps/s x 2.885 accepted = 55.5 tok/s

Three extra draft positions add 10.9 ms to every step and return 0.009 more
accepted tokens. That is the whole story of the prose penalty at depth.

Structured output escapes it because its acceptance *does* keep rising with k:
3.890 at k=4 against 5.679 at k=7, same depth. The extra positions are filled
often enough to outrun the rising per-position cost:

    k=4:  24.5 steps/s x 3.890 = 92.4 tok/s
    k=7:  19.3 steps/s x 5.679 = 108.3 tok/s

(The k=4 figure is the soft one: a rerun of the same stage measured 4.269
acceptance and 103.5 tok/s at that depth, so structured acceptance at k=4 is
sample-dependent and does not clearly fall at depth. The direction — more
positions still filled at k=7 — holds either way, but the gap is smaller than
this one pair suggests.)

**The rule that falls out:** the deeper your typical context, the more each
draft position has to earn. Predictable output (JSON, code, tool calls) keeps
earning it; prose stops at k=4 and the penalty for overshooting roughly doubles
by 131K. If your workload is mixed and long, k=4 is the safer end of the range.

\* The 84.49 at k=7 / 32K is a content artefact, not a depth effect. Sampling is
greedy (`temperature 0`), so each config produces one fixed output; that run's
output happened to be unusually predictable (acceptance 3.77 against 2.95 at
depth 0). A k=6 run shows the same artefact at the same depth (140.13 tok/s).
Treat single tok/s cells as one sample; steps/s is the stable column.

**This config runs k=4** — the peak for prose and for mixed output, and the
setting whose depth penalty is smallest. Agent turns here are prose-heavy in
practice: the reasoning block is prose, and it is often several hundred tokens
before any JSON appears. k=5-7 is the choice only for output that is
overwhelmingly structured.

### Power

**Clock cap (added 2026-10-14).** To keep the cards cooler and the fans quieter
under long prefills, the rig now locks the SM clock to 2500 MHz
(`nvidia-smi -lgc 300,2500`; 150 W is already the power floor, so the clock is
the only lever left). It costs speed: about **4 % on decode** (38.7 vs 37.2 ms
per step) and **5.6 % on prefill** (2,880 vs 3,051 tok/s at a cold 32K), from
one back-to-back flip of the cap, n=1 each. **The decode, depth, k-sweep, NVFP4
and 250K needle tables were measured before the cap, uncapped.** In section 2b,
the restore timings were also taken before the cap, while the 250K needle at
4.6e9 and 4.9e9 and the concurrent 2 × 200K test were run with it on. Expect roughly 4 % lower decode than the tables on
the capped rig.

The 150 W cap does not limit decode: the cards draw ~123 W while decoding.
Decode is limited by memory bandwidth, not compute, so a higher power limit
buys nothing here.

### Long context: 250K needle test

A ~250K-token prompt with a passphrase buried at 50 % depth, then asked to
recall it. This tests that the KV cache is correct at depth, not just that the
server survives the prefill.

| Prompt tokens | Prefill time | Prefill tok/s | Result |
|--------------:|-------------:|--------------:|--------|
| 249,450       | 246.7 s      | 1,011         | Passphrase recovered |

Measured at MTP k=6, `gpu-memory-utilization 0.95`; prefill is unaffected by
k, since speculation applies to decode only.
Prefill rate falls with depth because every new token attends to all the
earlier ones: ~4,200 tok/s at 4K, ~3,100 at 32K, ~1,000 at 250K.

### Time to first token

Prompts of different lengths, showing how the prefill rate holds up with depth.

| Prompt tokens | TTFT     | Prefill tok/s |
|--------------:|---------:|--------------:|
| 523           | 0.141 s  | 3,712         |
| 4,123         | 0.987 s  | 4,176         |
| 32,812        | 10.521 s | 3,119         |

## What didn't work, and other gotchas

- **`--max-num-batched-tokens 8192`.** It makes the vLLM warning about
  speculative decoding and `max_num_scheduled_tokens` go away, but it gives no
  measurable speed gain: TTFT and decode are the same as at 4096. It needs more
  activation memory, which costs context: at 8192 anything above
  `gpu-memory-utilization 0.91` OOMs, and 0.91 fits 200K (KV pool 201,562),
  against 262K at 0.95 with 4096. The
  warning says "*may* lead to suboptimal performance", and at `max-num-seqs 2`
  it doesn't: 2 sequences × 7 draft slots is 14 of the 4096 per step.
- **FlashInfer autotune** OOMs on vLLM versions after 0.28.0. Use
  `--no-enable-flashinfer-autotune`. Turning it off also frees the headroom to
  run `gpu-memory-utilization` at 0.95.
- **MTP k, the full sweep.** See "Choosing k" above. Prose peaks at k=4 and
  declines gently after; structured is flat from k=5 on. An earlier note here
  claimed prose "collapses to ~20 tok/s above k=6" — **that does not reproduce
  on this config**: k=7 measures 70.4 tok/s on prose. The claim came from a
  different build and has been withdrawn.
- **`NCCL_P2P_DISABLE=1` and `--disable_custom_all_reduce`.** GeForce cards
  don't support direct GPU-to-GPU (P2P) transfers, so tensor-parallel traffic
  goes through system RAM. vLLM's custom all-reduce kernel works by having each
  GPU write straight into the other's memory, so it needs P2P. Disabling it
  leaves the all-reduce to NCCL, which handles the no-P2P path. vLLM usually
  detects missing P2P and falls back on its own; the flag makes that explicit.
- **`cudagraph_mode`: PIECEWISE is what this config runs; FULL_AND_PIECEWISE
  was measured and is a valid alternative that buys almost nothing here.**
  vLLM 0.30.0 offers NONE, PIECEWISE, FULL, FULL_DECODE_ONLY and
  FULL_AND_PIECEWISE. With chunked prefill, FULL_AND_PIECEWISE is the one to
  try: full graphs for decode-only batches, piecewise for the mixed prefill
  batches. Note that under PIECEWISE the *draft* steps get no graphs at all
  (`speculator.py`: "PIECEWISE cudagraphs are not supported for draft
  decodes"), so this is where a gain would come from. Measured at k=4, same
  workload:

  | | PIECEWISE | FULL_AND_PIECEWISE |
  |---|---:|---:|
  | Steps/s @0 / 32K / 131K | 28.55 / 27.36 / 24.17 | 29.29 / 28.02 / 24.68 |
  | Step time saved          | —                     | ~0.87 ms at every depth |
  | Prose / structured tok/s | 78.31 / 124.89        | 77.47 / 124.44 |
  | TTFT 32K                 | 10.521 s              | 10.624 s |
  | KV pool                  | 278,927               | 275,730 (−3,197) |
  | 250K needle              | recovered             | recovered, clean |

  So: **no throughput gain.** The step rate rises 2.1-2.6 % at every depth, but
  measured tok/s did not follow — prose and structured both came out slightly
  *lower*, because acceptance on these prompts was 1-3 % lower under full
  graphs. Whether that acceptance difference is specific to these prompts or
  systematic is not established; either way, the faster steps did not turn into
  tokens. Prefill is unchanged, and 3,197 tokens of pool are spent. Graph
  capture took 19 s and 0.35 GiB per worker. An older note here warned that FULL graphs plus MTP silently
  corrupt decoding into loops or empty answers — **that does not reproduce on
  stock 0.30.0**; it came from a different build. PIECEWISE stays because full
  graphs bought no throughput here, not because FULL is unsafe.
- **`--attention-backend TRITON_ATTN` — do not.** FlashInfer logs "Fused
  multi-step draft decode is not supported by attention backend(s)
  FLASHINFER; falling back to rebuilding attention metadata between draft
  steps", which looks like free performance: TRITON_ATTN supports the fused
  path, so the message disappears, and it also frees FlashInfer's 64 MB
  workspace (pool 283,722 against 278,927). Measured, it is a disaster at
  depth:

  | Depth | Steps/s FlashInfer | TRITON_ATTN |
  |------:|-------------------:|------------:|
  | 0     | 28.55              | 28.76 (+0.7 %) |
  | 32,768 | 27.36             | 18.25 (**−33 %**) |
  | 131,072 | 24.17            | 8.71 (**−64 %**) |

  At 131K that is 25.8 tok/s prose — slower than no speculative decoding at
  all. Prefill suffers too (2,670 tok/s at 32K against 3,119), and the cards
  run 6-9 °C hotter for the extra work. **The fused draft path is worth ~0.7 %;
  FlashInfer's attention kernel is worth 2.8x at 131K.** The flat depth curve in
  this README is FlashInfer's doing. Vision still worked under TRITON_ATTN,
  despite the startup line about `mm_prefix` being disabled — that message
  means a backend constraint was relaxed (this config disables video), not that
  multimodal support was dropped.
- **`--async-scheduling` changes nothing here.** It overlaps the CPU's
  scheduling of the next step with the current step's GPU work, which only pays
  when the CPU is the bottleneck. At `max-num-seqs 2` with ~35 ms steps it is
  not: steps/s, acceptance and reasoning length came back byte-identical to the
  baseline. Worth revisiting for high-concurrency serving, not for a
  single-user rig.
- **x8 PCIe is not the decode bottleneck.** Decode reaches ~88 % of the
  memory-bandwidth ceiling even with tensor-parallel traffic going through host
  RAM.
- **`--limit-mm-per-prompt` caps images per *conversation*, not per message.**
  Chat clients resend the whole conversation every turn, so `{"image": 2}`
  means two images per chat, ever. The third returns HTTP 400 and breaks the
  conversation. Leaving the `image` key out removes the cap, which is why the
  recipe only disables video.
- **`max-num-seqs` and `max-num-batched-tokens` change the KV pool size.**
  Re-read `GPU KV cache size` in the startup log after changing either.

## Tips and tricks

**What each knob does, if you change it from this recipe:**

| Knob | Here | Change it and… |
|------|------|----------------|
| `max-num-batched-tokens` | 4096 | 8192 buys no speed and caps context at 200K |
| MTP `num_speculative_tokens` | 4 | prose peaks at 4; structured keeps rising to 7; see "Choosing k" |
| `cudagraph_mode` | PIECEWISE | FULL_AND_PIECEWISE: +2.4 % steps/s, no tok/s gain, −3,197 pool |
| attention backend | FlashInfer (default) | TRITON_ATTN: −33 % at 32K, −64 % at 131K |
| `async-scheduling` | off | no effect at `max-num-seqs 2` |
| `max-num-seqs` | 2 | higher collapses per-stream rate on these cards (measured outside this repo) |
| `flashinfer-autotune` | off | on: OOMs on vLLM after 0.28.0 |
| `gpu-memory-utilization` | 0.94 (nvfp4; fp8 ran 0.95) | above 0.91 OOMs *if* batched tokens is 8192 |
| `kv-cache-dtype` | nvfp4 | fp8 (the baseline in the tables): 278,927-token pool instead of 474,032, ~5 % more steps/s; bf16 roughly doubles KV bytes/token vs fp8 and 262K would not fit |
| KV offload to RAM | off | commented-out block in the recipe; measured with nvfp4 KV, needs 24 GiB RAM |

**Method — how to test a change without fooling yourself:**

- **Re-read `GPU KV cache size` after every change.** Most knobs move the pool:
  MTP k, batched tokens, graph mode, attention backend. The startup log is the
  only truth.
- **Scope the log to the container's `StartedAt`.** `compose up -d` keeps the
  old log, so an unscoped grep can read the *previous* boot and show a false
  READY:
  `docker logs --since "$(docker inspect -f '{{.State.StartedAt}}' <container>)" <container>`
- **Compare steps/s, not tok/s.** Sampling here is greedy (`temperature 0`), so
  each config produces one fixed output; tok/s then depends on how predictable
  that particular text is. Steps/s is set by the model, the depth and k, and it
  reproduces to ±0.02.
- **Check the reasoning-token count before trusting a tok/s comparison.** A
  reply that thinks for 800 tokens and one that thinks for 150 are not the same
  workload, and the difference is worth 40 tok/s on the structured prompt.
- **Test at depth, not just at depth 0.** TRITON_ATTN looked 0.7 % *faster*
  empty and was 64 % slower at 131K. Depth-0 numbers hide the thing you
  actually care about.
- **A speed test cannot catch a broken config.** Silent KV/graph corruption
  shows up as looping or empty output while tok/s stays healthy. End every
  config change with the 250K needle and *read the reply text*, not just the
  pass flag.
- **Warm up.** The first run after a restart is slower — ~3 % on prose here.
  Discard it.
- **One variable at a time, and write down the pool.** Two of the "findings"
  this README started with turned out to be artefacts of an older build; both
  died the moment they were re-measured in isolation.

## What would make this faster

Three things could move these numbers, in rough order of how much they would
change the tables above.

**1. Fused multi-step draft decode on FlashInfer.** Every MTP step here rebuilds
the attention metadata between draft positions, because FlashInfer does not
declare support for updating it in place:

    Fused multi-step draft decode is not supported by attention backend(s)
    FLASHINFER; falling back to rebuilding attention metadata between draft steps.

That fallback is part of the flat **2.37 ms per draft position** measured in
"Choosing k". Upstream is moving: vLLM
[PR #55292](https://github.com/vllm-project/vllm/pull/55292) lets each metadata
builder declare multi-step support instead of relying on a central list, and
[PR #57443](https://github.com/vllm-project/vllm/pull/57443) enabled the fused
path for another model with a reported **13.3 % end-to-end gain at concurrency
1** — single-stream, which is this rig's case.
[Issue #54369](https://github.com/vllm-project/vllm/issues/54369) goes further
and argues the rebuild fallback is what "caps useful MTP depth at k=4" — still
active as of September 2026.

**If that is right, the k curve in this README is partly an artefact of the
fallback, not of the draft head's quality.** A cheaper per-position cost moves
the break-even point, and k=5-7 could start paying for prose too. The sweep
would be worth re-running the day FlashInfer declares support. Switching to a
backend that already has it is not the answer — see TRITON_ATTN under "What
didn't work".

**2. NVFP4 KV cache for GDN models. NOW IN vLLM 0.32.0 (released 2026-10-18).**
Everything above is fp8 KV on vLLM 0.30.0, ~16.3 KiB per token per GPU. As of
[PR #60452](https://github.com/vllm-project/vllm/pull/60452) (merged 2026-10-10,
"Decode an NVFP4 KV cache with XQA on SM12x") NVFP4 KV decodes through FlashInfer
XQA on this card family. It is **not in the 0.30.0 image used for the tables
above**. It is in the tagged
[v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0), whose
notes list it under "Low-precision KV cache everywhere" (#46963 for SM8x/SM12x
support, #60452 for XQA decode on SM12x).

**Everything measured below was done on the nightly image**
(`vllm/vllm-openai:nightly`, v0.32.0rc1.dev100, before the release), **not on the
tagged 0.32.0 image.** The feature is the same code, but I have not re-run these
tables on the release.

Measured on that nightly with `--kv-cache-dtype nvfp4`,
`gpu-memory-utilization 0.94`, MTP k=4, `max-num-seqs 2`, `max-model-len 262,144`.
The fp8 run differs in more than the KV dtype: vLLM 0.30.0 against the nightly,
NVIDIA driver 595.91.07 against 595.99.02, kernel 7.0.0-31 against -38, and
utilization 0.95 against 0.94. This is a build comparison, not a controlled
KV-dtype experiment. The fp8 comparison is the profiler run
of 2026-09-30 on 0.30.0 (same workload version, k=4); it is **not** the table
under "Decode speed", which came from an earlier workload.

| | fp8, 0.30.0 | NVFP4, nightly | |
|---|---:|---:|---|
| KV cache pool | 278,927 tok | **474,032 tok** | +70 % |
| Concurrency at 262,144 | 1.06x | **1.81x** | |
| Decode step time (p50) | 35.0 ms | 36.9 ms | +5 % |
| Steps/s | 28.5 | 27.1 | −5 % |
| Structured tok/s | 89.7 | 90.6 | +1 % |
| 250K needle | passed (246.7 s, 1,011 tok/s)¹ | **passed** (274.8 s, 912.7 tok/s) | prefill −10 % |

¹ From "Long context" below; 249,450 tokens at k=6, so prefill-only and
comparable. A second NVFP4 run at k=5 recovered the same passphrase
(250,414 tokens, 272.2 s).

What this shows:

- **The pool is the win.** 474K tokens holds nearly two full 262K contexts.
  Steps/s is a few per cent lower at depth 0, which is the price.
- **The prefill gap is a depth effect.** At ~32.7K tokens NVFP4 prefilled at
  3,051 tok/s (k=4) and 3,056 tok/s (k=5), against 3,025-3,105 tok/s across the fp8
  `ttft_sweep` runs of 2026-09-22 to 2026-10-02 (0.30.0 and 0.30.1rc1, k=4/5/6), which is
  inside their run-to-run spread. The same holds at ~4K (4,093-4,178 against
  4,172-4,333). The -10 % in the table only shows up at 250K, so whatever costs the time
  grows with context. Attention taking a larger share of the prefill at depth would fit,
  but I have not profiled it. Each figure is one prompt per depth, on different builds.
- **Quality cost is measured.** The [KV-cache quality test](kv-quality-test.md)
  compares bf16, FP8 and NVFP4 KV on this model: FP8 is indistinguishable from
  bf16, NVFP4 costs roughly 0.2-0.5 % perplexity (0.002-0.005 nats/token), and the
  shift does not grow with context depth in that data. It covers one model, one
  GPU pair and one build, so it is not a general statement about KV quantization.
- **Correctness at depth held.** The 250K needle passed twice (k=4 and k=5), with
  no Xid and no container restart. Upstream's own SM120 accuracy evidence is one
  third-party GSM8K run at short context (0.962 NVFP4, 0.962 FP8).
- **Step-rate decay at depth looks smaller, not proven.** At k=4, steps/s goes
  27.1 → 26.2 → 23.6 at 0 / 32K / 131K, which is −13 % at 131K, against −18 % for
  fp8 in "Decode at depth". Different builds and dates, so treat it as a lead.
- **Prose tok/s is not comparable between runs.** In the NVFP4 run the prose
  prompt spent all 800 tokens reasoning; in the fp8 run it spent 202 and then
  answered. Acceptance moves with content, so only step time and the structured
  rows are quoted above. Mean acceptance read 2.72 against 2.94 (−7.5 %); that
  may be content, or NVFP4 rounding in the KV, and this run cannot separate them.
- **Heat under the 250K prefill:** peak 82 / 81 °C at 43–47 % fan, 128–131 W per
  card, 93–96 % of samples unthrottled. No fp8 run of the same stage exists to
  compare against.

**2b. KV offload to host RAM works with NVFP4 KV (2026-10-14/15).**
vLLM's native CPU offload keeps evicted KV blocks in system RAM, so a session that
was pushed out of the GPU pool restores in seconds instead of being prefilled again.
It works on this hybrid (Gated DeltaNet) NVFP4 model, with the NVFP4 KV cache and MTP
k=4 on. Measured on the same nightly as above.

**What it is for.** A bigger pool (section 2) lets more sessions stay resident on
the GPU. Offload covers the case where that still isn't enough: when a session
is evicted from the pool by newer work, its KV blocks go to host RAM instead of
being thrown away, and coming back to it costs a copy (~2.6 s for 93K tokens)
instead of a fresh prefill (~50 s). It pays off if you juggle several long
sessions (agents, subagents, a few chats left open) on one server, and does
nothing if everything you do fits in the pool anyway. It is not persistence:
RAM only, so a container restart empties it. It needs spare host RAM (24 GiB
below) and a little GPU memory for the connector, which is why the pool shrinks
slightly (below). The recipe at the bottom has it as a commented-out block.

The flags, added to the `command:` list of the recipe:

```yaml
      - --kv-cache-memory
      - "4900000000"            # bytes of KV per GPU; see "Sizing" below
      - --enable-cumem-allocator
      - --kv-offloading-backend
      - native
      - --kv-offloading-size
      - "24"                    # GiB of host RAM, total
```

and, at service level, `ulimits: { memlock: -1 }` (see "Pinning").

What was measured (restore tests, `n=1` per row unless noted):

| Prompt | Cold prefill | Restored from offload | Output |
|---|---:|---:|---|
| ~46K tokens (pool 72K) | 17.5 s | 1.67 s | byte-identical, 2 of 2 |
| ~93K tokens (pool 109K) | 49.9 s | 2.63 s | byte-identical |
| ~93K tokens (pool 423,776) | 49.6 s | 2.6 s | byte-identical |

- **Cost:** the store side was 7-30 ms per 10 s window in normal use, so effectively
  free. KV is ~20.5 KB per token across both GPUs, so 24 GiB holds roughly 1.26M tokens.
- **RAM only:** a container restart empties it. Nothing survives a restart.
- **At 250K:** the needle test passed with offload on (249,823 tokens, correct
  passphrase), and so did two concurrent ~200K requests (both needles correct,
  1,064 tok/s aggregate).

**Sizing.** `--kv-cache-memory` skips memory profiling, and the offload connector
takes GPU memory of its own, so a value that works without it can fail with it.
`5120000000` passed the 250K needle without offload. With offload it **boots cleanly
and then OOMs on the first deep prefill** (93K in my test, GPU 1 down to 92 MiB free),
so a clean start proves nothing: always run a deep prefill before trusting a value. The
startup log's own suggestion ("--kv-cache-memory=5484572672 to fully utilize gpu memory")
did the same thing even without offload: started in ~40 s, then CUDA-OOMed on a 32K
prefill. Do not use it. Gate any value on a 32K prefill first, then the 250K needle.

For reference, `5120000000` is what normal profiling picked at
`--gpu-memory-utilization 0.94` (pool 471,315 vs 474,032, -0.6 %), and setting it
directly cut startup from ~2 min to ~40 s.

| `--kv-cache-memory` | KV pool | Concurrency at 262,144 | Free at 250K prefill peak |
|---:|---:|---:|---|
| 4,600,000,000 | 423,776 | 1.62x | not recorded |
| 4,900,000,000 | 450,942 | 1.72x | ~580 MiB per card |

Both passed the 250K needle. The pool is smaller than the 474,032 tokens in the table
above, which is the no-offload run.

**Gotchas.**

- Any KV connector refuses `expandable_segments` unless `--enable-cumem-allocator` is
  on. The recipe sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, so you need both.
- **Pinning.** Docker's default locked-memory limit is 8 MiB, and with it vLLM logs
  `cudaHostRegister failed (code=1)`: transfers still work, but the offload region
  stays pageable (unpinned DMA). `ulimits: memlock: -1` fixes that. The restore times
  above were measured **before** that fix, and the 4.9e9 runs after it. I have not
  re-measured restore speed pinned.
- `VLLM_SERVER_DEV_MODE=1` exposes endpoints such as `/reset_prefix_cache` **without
  the API key** on the listening port. I used it to force evictions in testing; do not
  leave it on.
- Needs host RAM: 24 GiB for the offload pool. On a 60 GB machine, about 21 GiB stayed
  available with the model loaded and offload pinned.

**LMCache did not work.** LMCache 0.5.5 (the version in the nightly image), run as a
sidecar, booted fine on this model but returned looping garbage on every restore, from
both its RAM tier and its disk tier. A control run without the connector was correct.
I suspect it does not handle NVFP4 KV on a hybrid model, but I did not isolate the cause
and did not report it upstream. Native offload does not have this problem.

**3. The DSpark draft head — TRIED 2026-09-23, does not load on vLLM 0.30.0.**
The checkpoint here is pinned to the last release that includes the MTP head;
upstream moved to DSpark, published as a separate 1.4 GB drafter repo
(`Qwen3.8-27B-DSpark-NVFP4`) alongside a checkpoint whose `lm_head` is NVFP4 and
whose MTP head is gone.

Everything looked promising from the outside: vLLM 0.30.0 ships
`qwen3_dspark.py`, resolves the architecture (`Resolved architecture:
Qwen3DSparkModel`), accepts `{"method": "dspark", "model": "/drafter",
"num_speculative_tokens": 7}` — 7 because the drafter's `block_size` is 7 and k
must divide by it — and DSpark drafts a block **in parallel** (vLLM sets
`parallel_drafting = True`), so it would not pay MTP's flat ~2.37 ms per draft
position. The drafter ships no `lm_head` or `embed_tokens`, sharing the
target's.

It dies in weight loading, on both workers:

    File "vllm/model_executor/models/qwen3_dflash.py", line 693, in load_weights
    File "vllm/model_executor/layers/vocab_parallel_embedding.py", line 507
    RuntimeError: The size of tensor a (128) must match the size of tensor b (256)
                  at non-singleton dimension 1

That is an NVFP4 packing mismatch — two values per byte, so a 256-wide row
arrives as 128 bytes — on a vocabulary-sized tensor, i.e. the shared NVFP4
`lm_head`. Not fixable with flags. **The drafter's model card says it requires
the Qwen3.8 SGLang build, and that is correct**; read the card before planning
around it (this was learned the expensive way). Note also that
`restart: unless-stopped` will cycle the container on this failure.

Expected trade if it ever does load on vLLM: the weights are 17.91 GB + 1.4 GB
= 19.31 GB against 18.8 GB here, ~0.5 GB more, which works out at a pool near
263K — 262,144 would still fit, with the headroom cut from 16,783 to a few
thousand.

## vLLM recipe (Docker Compose)

Stock vLLM 0.32.0 with NVFP4 KV, the operating config. (The fp8 baseline on
0.30.0, adopted 2026-09-22, is the fallback described below the quick start.)

**Quick start.** You need the NVIDIA driver, Docker with the Compose plugin,
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
# 1. Pull the image. The recipe sets pull_policy: never, so Compose won't pull it for you
docker pull vllm/vllm-openai:v0.32.0

# 2. Get the checkpoint (see "Model" above), or use the --revision variant below

# 3. Save the YAML below as compose.yaml, change /home/user to your own paths
mkdir -p ~/ai/cache/vllm-stock
docker compose up -d

# 4. Follow the startup log. The first start compiles CUDA graphs, so it takes a while
docker compose logs -f
#    Look for "GPU KV cache size: ... tokens", which should be above 262,144

# 5. Wait for the server to be ready, then send a test request
until curl -sf localhost:8000/health; do sleep 5; done
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer api-key" -H "Content-Type: application/json" \
  -d '{"model": "vllm-qwen38-27b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Change `--api-key` before you expose port 8000 to anything beyond localhost.

**Switching on KV offload.** The recipe below runs NVFP4 KV without offload,
the measured 474,032-token pool. Offload (see "What would make this faster",
section 2b) is a commented-out block. In the YAML:

1. Comment out `--gpu_memory_utilization` and `"0.94"`, and uncomment the
   `--kv-cache-memory` line: it sets KV bytes per GPU directly and skips
   profiling. Don't use both.
2. Uncomment the `ulimits:` block (pinned transfers) and the
   `--enable-cumem-allocator` and `--kv-offloading-*` lines.
3. `docker compose up -d`, read `GPU KV cache size` in the log (see
   "Method"), then run a deep prefill (32K, then ~250K) before trusting it.
   A clean boot proves nothing; a too-high `--kv-cache-memory` OOMs on the
   first deep request.

To size it for your machine: `--kv-offloading-size` is GiB of host RAM, total,
so leave room for the OS and the model server. `--kv-cache-memory` is bytes
per GPU; `4900000000` suits 16 GB cards at 262K, and lower it if a deep
prefill OOMs. To go back, re-comment the block.

**Falling back to fp8.** Set `--kv-cache-dtype` to `fp8` and
`--gpu_memory_utilization` to `0.95`. That is the baseline in the tables above
(278,927-token pool, 1.06x at 262K). It also works on older vLLM, where NVFP4 KV
does not exist: the fp8 tables were measured on 0.30.0.

**Letting vLLM download the pinned commit instead.** vLLM takes `--revision`,
so you can pass the Hugging Face repo name and the commit hash and skip the
manual download. The tokenizer uses the same revision by default. Make these
changes to the recipe:

```yaml
    volumes:
      # replaces the /model mount: the Hugging Face cache, so the download is kept
      - /home/user/.cache/huggingface:/root/.cache/huggingface
      - /home/user/ai/cache/vllm-stock:/root/.cache/vllm
    command:
      - --model
      - gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
      - --revision
      - 0cc27958cefbbe231782ec8511de8c4eb5233348
      # ...everything else unchanged
```

The first start downloads ~18.8 GB before it loads. `--served-model-name`
keeps the API model name at `vllm-qwen38-27b` either way. All the measurements
here were made with the local-folder mount, not this variant.

```yaml
services:
  vllm-qwen38-27b:
    # Image pinned, for reference and to avoid version drift
    image: vllm/vllm-openai:v0.32.0
    # Comment this line out if pulling a new version is required.
    pull_policy: never
    container_name: vllm-qwen38-27b
    restart: unless-stopped
    ipc: host
    # KV offload (optional, see "Switching on NVFP4 KV and KV offload"):
    # lifts Docker's 8 MiB locked-memory limit so the RAM region can be pinned.
    #ulimits:
    #  memlock: -1
    ports:
      - 8000:8000
    volumes:
      # Qwen 3.8 27B NVFP4 checkpoint, manually downloaded to this folder
      - /home/user/ai/models/gittensor-NVFP4:/model
      # vLLM cache folder, for faster startup
      - /home/user/ai/cache/vllm-stock:/root/.cache/vllm
    environment:
      - NCCL_P2P_DISABLE=1
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      - VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities:
                - gpu
    command:
      - --model
      - /model
      - --served-model-name
      - vllm-qwen38-27b
      - --tensor-parallel-size
      - "2"
      # --no-enable-flashinfer-autotune is required for vLLM > v0.28.0, as
      # autotune OOMs. Autotune also requires unassigned VRAM; disabling it
      # provides the headroom to raise gpu-memory-utilization to 0.95-0.96.
      - --no-enable-flashinfer-autotune
      # No P2P on GeForce: let NCCL do the all-reduce (see gotchas)
      - --disable_custom_all_reduce
      - --api-key
      - api-key
      - --load-format
      - ipc_cache
      - --trust-remote-code
      # nvfp4 needs vLLM >= 0.32.0; fp8 is the fallback (see "Falling back to fp8")
      - --kv-cache-dtype
      - nvfp4
      # 262144 is the project's target. With nvfp4 at gpu-memory-utilization
      # 0.94 and MTP k=4: GPU KV cache size 474,032 tokens, 1.81x at 262,144.
      # (fp8 at 0.95: 278,927 tokens, 1.06x.)
      - --max-model-len
      - "262144"
      # --max-num-seqs 2 since 2026-09-01. Was 1; the comment below is kept
      # because the reasoning still holds, only the conclusion changed.
      # OpenCode's main agent IDLES while its subagent runs, so OpenCode is
      # effectively ONE concurrent request -- but a 2nd request (Open WebUI's
      # background title generation, or a subagent) used to QUEUE behind a
      # deep prefill.
      - --max-num-seqs
      - "2"
      # PIECEWISE, not FULL_AND_PIECEWISE: the latter was measured and gave
      # +2.4% steps/s but no tok/s gain, for 3,197 tokens of pool. See gotchas.
      - --compilation-config
      - '{"cudagraph_mode": "PIECEWISE"}'
      - --mamba-cache-dtype
      - bfloat16
      - --mamba-ssm-cache-dtype
      - bfloat16
      - --gpu_memory_utilization
      - "0.94"
      # --- KV offload to host RAM: OPTIONAL, off by default ---
      # Replaces gpu_memory_utilization: comment that
      # one out and uncomment this block. Also uncomment ulimits (top).
      # Measured pool: 450,942 tokens. Validate with a 32K prefill, then the 250K needle.
      #- --kv-cache-memory
      #- "4900000000"       # bytes of KV per GPU; lower it if a deep prefill OOMs
      #- --enable-cumem-allocator
      #- --kv-offloading-backend
      #- native
      #- --kv-offloading-size
      #- "24"               # GiB of host RAM, total
      - --max-num-batched-tokens
      - "4096"
      - --enable-prefix-caching
      - --enable-chunked-prefill
      # Vision: disable video input, limit image size
      - --limit-mm-per-prompt
      - '{"video": 0}'
      - --mm-processor-kwargs
      - '{"max_pixels": 1000000}'
      # MTP k: measured sweep k=0..7 (see README "Choosing k"). 4 is the peak
      # for prose (~78 t/s) and for mixed prose+JSON (~82 t/s); structured is
      # ~125 t/s here and would reach ~138 at k=7, at the cost of prose.
      # Step time is 25.4 ms + 2.37 ms per draft position, so each position
      # must earn its 2.37 ms in accepted tokens.
      - --speculative-config
      - '{"method": "mtp", "num_speculative_tokens": 4}'
      - --reasoning-parser
      - qwen3
      - --tool-call-parser
      - qwen3_xml
      - --enable-auto-tool-choice
      - --default-chat-template-kwargs
      - '{"preserve_thinking": true}'
networks: {}
```
