# Qwen 3.8 27B NVFP4 with FP8 KV cache on vLLM 0.30.0

**A 27B model with its full 262K context on two midrange 16 GB cards: stock
vLLM, no patches, 72 tok/s on prose and 133 tok/s on structured output.**

- **What:** Qwen 3.8 27B (NVFP4 weights, FP8 KV cache) on 2 × RTX 5060 Ti,
  tensor parallel, with MTP speculative decoding, on a headless Ubuntu 26.04
  box.
- **How:** the official vLLM 0.30.0 Docker image and one Compose file. The full
  recipe is at the bottom. The tuning that matters is `max-num-batched-tokens
  4096`, FlashInfer autotune off, and MTP k=6.
- **Why:** to show that full-context agentic work doesn't need a 5090. It's
  verified with a 250K needle test, measured against a no-MTP baseline, and
  comes with the gotchas that cost context or crash the server.

## Model

[gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090/commit/0cc27958cefbbe231782ec8511de8c4eb5233348)
(pinned to commit `0cc2795`)

Pinned because this is the latest release of the checkpoint that still
includes the **MTP head**. Later commits don't include it, and without it
`--speculative-config` MTP won't work: decode drops to the ~42 tok/s no-MTP
baseline.

## Hardware

| Component   | Details                     |
|-------------|-----------------------------|
| CPU         | AMD Ryzen 7 9800X3D         |
| GPUs        | 2 × RTX 5060 Ti 16 GB       |
| Motherboard | MSI MPG X670E CARBON WIFI   |
| RAM         | 32 GB DDR5-6000             |
| Storage     | NVMe Gen4                   |
| PCIe        | Gen5 x8 / x8 (cards are x8 electrically; GPU0 sits in an x16 slot) |
| GPU power   | Capped at 150 W per card (stock max 198 W) |

## Software

| Component     | Version              |
|---------------|----------------------|
| OS            | Ubuntu 26.04 LTS     |
| Kernel        | 7.0.0-31-generic     |
| NVIDIA driver | 595.91.07            |
| CUDA          | 13.2                 |
| vLLM          | 0.30.0 (official `vllm/vllm-openai:v0.30.0` image, stock, no patches) |

## Performance

All runs: vLLM 0.30.0, fp8 KV cache, `max-num-seqs 2`, `max-model-len 262,144`.

### Decode speed

| Output type | No MTP tok/s | MTP k=6 tok/s | Speed-up | Steps/s (no MTP / MTP) | Acceptance (MTP) | Reasoning tokens (no MTP / MTP) |
|-------------|-------------:|--------------:|---------:|-----------------------:|-----------------:|--------------------------------:|
| Prose       | 41.72        | 72.45         | 1.7×     | 41.84 / 25.21          | 2.888            | 411 / 511                       |
| Structured  | 41.72        | 132.75        | 3.2×     | 41.81 / 25.20          | 5.333            | 150 / 158                       |
| Mixed       | 41.70        | 73.84         | 1.8×     | 41.79 / 25.19          | 2.952            | 522 / 275                       |

| Config  | KV cache pool | Headroom over max-model-len |
|---------|--------------:|----------------------------:|
| No MTP  | 351,618 tok   | +89,474                     |
| MTP k=6 | 272,533 tok   | +10,389                     |

**MTP k=6** (6 speculative tokens) gives the fastest structured output while
keeping prose close to its best. Decode speed is the only metric that differs
between the MTP and non-MTP runs.

Without MTP, every output type decodes at the same speed, since nothing is
drafted. That is about 88 % of the theoretical ceiling set by memory bandwidth
(~47 tok/s).

On highly repetitive output, MTP k=6 reaches ~160 tok/s. The ceiling at k=6 is
7 tokens per step, or ~176 tok/s.

### Decode at depth

Decode speed is not a constant: it falls as the context fills. Measured at
three depths, MTP k=6, decode only (prefill excluded from the tok/s).

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
| 0 | 41.90   | 41.99       | 42.00            | 41.85       | — (no drafting)                | 351,618 |
| 1 | 35.94   | 63.55       | 69.83            | 63.41       | 1.78 / 1.95                    | 290,655 |
| 2 | 33.26   | 74.71       | 94.62            | 77.54       | 2.25 / 2.86                    | 285,321 |
| 3 | 30.74   | 77.53       | 107.20           | **83.01**   | 2.53 / 3.52                    | 284,319 |
| 4 | 28.57   | **77.73**   | 122.81           | 81.61       | 2.73 / 4.35                    | 278,927 |
| 5 | 26.77   | 72.14       | 132.93           | 79.03       | 2.70 / 5.03                    | 274,815 |
| 6 | 25.21   | 72.45       | 132.75           | 74.67       | 2.89 / 5.33                    | 272,533 |
| 7 | 23.80   | 70.36       | **135.21**       | 76.50       | 2.96 / 5.76                    | 268,596 |

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

**Prose peaks at k=4, structured at k=5.** Prose gains +21.6, +11.2, +2.8 tok/s
for k=1..4, then loses ground: acceptance stops rising (2.73 at k=4, 2.70 at
k=5) while every step keeps getting longer. Structured keeps filling the extra
slots — per-position acceptance is still 0.46 at position 7 — but the gains
flatten from k=5: 132.93, 132.75, 135.21.

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

**The rule that falls out:** the deeper your typical context, the more each
draft position has to earn. Predictable output (JSON, code, tool calls) keeps
earning it; prose stops at k=4 and the penalty for overshooting roughly doubles
by 131K. If your workload is mixed and long, k=4 is the safer end of the range.

\* The 84.49 at k=7 / 32K is a content artefact, not a depth effect. Sampling is
greedy (`temperature 0`), so each config produces one fixed output; that run's
output happened to be unusually predictable (acceptance 3.77 against 2.95 at
depth 0). A k=6 run shows the same artefact at the same depth (140.13 tok/s).
Treat single tok/s cells as one sample; steps/s is the stable column.

**This config runs k=6** — structured-heavy agentic work, where the ~7 % prose
loss against k=4 buys ~8 % on structured output. For prose-heavy use, k=4.

### Power

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

Same config as the recipe below (MTP k=6, `gpu-memory-utilization 0.95`).
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
- **`cudagraph_mode: PIECEWISE`, not FULL.** In earlier testing on this rig,
  FULL CUDA graphs combined with MTP silently corrupted decoding: the output
  looped or came back empty, with no crash and no error. PIECEWISE avoids that.
  FULL hasn't been re-tested on this exact config (fp8 KV, vLLM 0.30.0). It
  would save some kernel-launch overhead, part of the ~12 % gap between no-MTP
  decode and the memory-bandwidth ceiling. If you try it, check the output, not
  just the speed.
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

## vLLM recipe (Docker Compose)

Stock vLLM 0.30.0, the operating config, adopted 2026-09-22.

```yaml
services:
  vllm-qwen38-27b:
    # Image pinned, for reference and to avoid version drift
    image: vllm/vllm-openai:v0.30.0
    # Comment this line out if pulling a new version is required.
    pull_policy: never
    container_name: vllm-qwen38-27b
    restart: unless-stopped
    ipc: host
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
      - --kv-cache-dtype
      - fp8
      # 262144 is the project's target, and it FITS on stock fp8:
      # at gpu-memory-utilization=0.95, GPU KV cache size: 272,533 tokens,
      # maximum concurrency for 262,144 tokens per request: 1.04x
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
      # FULL graphs + MTP silently corrupted output in earlier testing
      # (loops / empty answers, no crash). See gotchas.
      - --compilation-config
      - '{"cudagraph_mode": "PIECEWISE"}'
      - --mamba-cache-dtype
      - bfloat16
      - --mamba-ssm-cache-dtype
      - bfloat16
      - --gpu_memory_utilization
      - "0.95"
      - --max-num-batched-tokens
      - "4096"
      - --enable-prefix-caching
      - --enable-chunked-prefill
      # Vision: disable video input, limit image size
      - --limit-mm-per-prompt
      - '{"video": 0}'
      - --mm-processor-kwargs
      - '{"max_pixels": 1000000}'
      # MTP: 6 gives maximum structured output (up to ~160 t/s on highly
      # repetitive output, ~133 t/s on typical JSON), while affecting prose
      # minimally (~72-75 t/s)
      - --speculative-config
      - '{"method": "mtp", "num_speculative_tokens": 6}'
      - --reasoning-parser
      - qwen3
      - --tool-call-parser
      - qwen3_xml
      - --enable-auto-tool-choice
      - --default-chat-template-kwargs
      - '{"preserve_thinking": true}'
networks: {}
```
