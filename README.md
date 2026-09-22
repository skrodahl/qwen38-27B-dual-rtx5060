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
- **MTP k > 6.** Prose collapses to ~20 tok/s, half the speed of no speculation
  at all. k=4 gives slightly better prose, k=6 gives much faster structured
  output.
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
