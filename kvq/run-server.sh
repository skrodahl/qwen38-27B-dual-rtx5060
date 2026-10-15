#!/bin/bash
# usage: run-server.sh <kv-dtype: auto|fp8|nvfp4> <kv-cache-memory bytes> [max-model-len]
# Private vLLM instance for the KV-quality test: port 8001, no prefix cache, no spec-decode, no offload.
set -e
DT=$1; MEM=$2; LEN=${3:-262144}; CHUNK=${4:-2048}
docker rm -f vllm-kvq >/dev/null 2>&1 || true
docker run -d --name vllm-kvq --gpus all --ipc host -p 8001:8000 \
  -v /home/skrodahl/ai/models/gittensor-NVFP4-MTP:/model \
  -v /home/skrodahl/ai/cache/vllm-kvq:/root/.cache/vllm \
  -e NCCL_P2P_DISABLE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864 \
  vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989 \
  --model /model --served-model-name kvq --tensor-parallel-size 2 \
  --no-enable-flashinfer-autotune --disable_custom_all_reduce --trust-remote-code \
  --kv-cache-dtype $DT --max-model-len $LEN --max-num-seqs 1 \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
  --kv-cache-memory $MEM --enable-cumem-allocator \
  --max-num-batched-tokens $CHUNK --no-enable-prefix-caching --enable-chunked-prefill \
  --limit-mm-per-prompt '{"video": 0, "image": 0}' --max-logprobs 20
