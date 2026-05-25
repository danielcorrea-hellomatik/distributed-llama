#!/usr/bin/env bash
# Authoritative n=20 record-comparable benchmark for the DeepSeek-8B paper.
# Plain decode (no spec), optimized stack, CLI Prediction tok/s metric (same as record #162).
set -uo pipefail
BIN=/home/rpi/dllama-spec/dllama
MODEL=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_model_deepseek-r1-distill-llama-8b_q40.m
TOK=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_tokenizer_deepseek-r1-distill-llama-8b.t
WORKERS="192.168.1.77:9998 192.168.1.75:9998 192.168.1.76:9998"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
PROMPT="The Eiffel Tower is one of the most famous landmarks in the world"
STEPS=256
echo "# DeepSeek-R1-Distill-8B Q40, 4xPi5 16GB, optimized stack (nthreads3+jemalloc), plain decode"
echo "# warmup x2"
for i in 1 2; do "$BIN" inference --model "$MODEL" --tokenizer "$TOK" --buffer-float-type q80 --max-seq-len 4096 --temperature 0 --seed 42 --prompt "$PROMPT" --steps 48 --nthreads 3 --workers $WORKERS >/dev/null 2>&1; done
echo "run,eval_prefill_toks,pred_decode_toks"
for i in $(seq 1 20); do
  out=$("$BIN" inference --model "$MODEL" --tokenizer "$TOK" --buffer-float-type q80 --max-seq-len 4096 --temperature 0 --seed 42 --prompt "$PROMPT" --steps "$STEPS" --nthreads 3 --workers $WORKERS 2>/dev/null)
  ev=$(echo "$out" | grep "tokens/s" | sed -n 1p | awk '{print $2}')
  pr=$(echo "$out" | grep "tokens/s" | sed -n 2p | awk '{print $2}')
  echo "$i,$ev,$pr"
done | tee /tmp/paperbench_raw.csv
python3 - <<'PY'
import statistics, math
rows=[l.strip().split(',') for l in open('/tmp/paperbench_raw.csv') if l[0].isdigit()]
dec=sorted(float(r[2]) for r in rows if len(r)>=3 and r[2])
pre=[float(r[1]) for r in rows if len(r)>=2 and r[1]]
def pct(v,p):
    k=(len(v)-1)*p; f=math.floor(k); c=math.ceil(k)
    return v[f] if f==c else v[f]+(v[c]-v[f])*(k-f)
print(f"DECODE n={len(dec)} mean={statistics.mean(dec):.3f} stdev={statistics.stdev(dec):.3f} "
      f"95CI=+/-{1.96*statistics.stdev(dec)/math.sqrt(len(dec)):.3f} min={min(dec):.3f} max={max(dec):.3f} "
      f"p50={pct(dec,.5):.3f} p90={pct(dec,.9):.3f} p99={pct(dec,.99):.3f}")
print(f"PREFILL n={len(pre)} mean={statistics.mean(pre):.3f}")
PY
