#!/usr/bin/env bash
# Record-comparable benchmark: plain decode (NO speculation), community prompts, dllama CLI
# Prediction tok/s metric, same as the b4rtaz discussion records (#162 DeepSeek 6.43).
set -uo pipefail
BIN=/home/rpi/dllama-spec/dllama    # optimized build; no --spec-ngram => identical to plain `inference`
MODEL=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_model_deepseek-r1-distill-llama-8b_q40.m
TOK=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_tokenizer_deepseek-r1-distill-llama-8b.t
WORKERS="192.168.1.77:9998 192.168.1.75:9998 192.168.1.76:9998"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
STEPS="${STEPS:-256}"; NT="${NT:-3}"

for PROMPT in "Hello World" "The Eiffel Tower is" "Write a short story about the sea"; do
  echo "=== prompt: \"$PROMPT\"  (steps=$STEPS, nthreads=$NT, plain decode) ==="
  for r in 1 2 3; do
    out=$("$BIN" inference --model "$MODEL" --tokenizer "$TOK" --buffer-float-type q80 --max-seq-len 4096 \
      --temperature 0 --seed 42 --prompt "$PROMPT" --steps "$STEPS" --nthreads "$NT" --workers $WORKERS 2>/dev/null)
    ev=$(echo "$out" | grep "tokens/s" | sed -n 1p | awk '{print $2}')
    pr=$(echo "$out" | grep "tokens/s" | sed -n 2p | awk '{print $2}')
    printf "  run %d: Evaluation(prefill)=%s tok/s | Prediction(decode)=%s tok/s\n" "$r" "${ev:-?}" "${pr:-?}"
  done
done
