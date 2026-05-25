#!/usr/bin/env bash
# Quick decode-rate probe (plain decode, no spec) for A/B testing OS levers. Prints decode tok/s.
set -uo pipefail
BIN=/home/rpi/dllama-spec/dllama
MODEL=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_model_deepseek-r1-distill-llama-8b_q40.m
TOK=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_tokenizer_deepseek-r1-distill-llama-8b.t
WORKERS="192.168.1.77:9998 192.168.1.75:9998 192.168.1.76:9998"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF="${MALLOC_CONF:-narenas:4,tcache:true,dirty_decay_ms:30000}"
STEPS="${STEPS:-160}"; N="${N:-2}"
for r in $(seq 1 "$N"); do
  out=$("$BIN" inference --model "$MODEL" --tokenizer "$TOK" --buffer-float-type q80 --max-seq-len 4096 \
    --temperature 0 --seed 42 --prompt "The Eiffel Tower is one of the most famous landmarks in the world" \
    --steps "$STEPS" --nthreads 3 --workers $WORKERS 2>/dev/null)
  pr=$(echo "$out" | grep "tokens/s" | sed -n 2p | awk '{print $2}')
  echo "  decode=${pr:-?} tok/s"
done
