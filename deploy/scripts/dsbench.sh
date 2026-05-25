#!/usr/bin/env bash
# dsbench.sh — single-arm decode benchmark for distributed-llama CLI `inference`.
# Runs on the ROOT node. Workers must already be listening on :9998 with the matching binary.
# Env in: BIN NT JEM RUNS STEPS MODEL TOK WORKERS PROMPT
# Out: each run's Prediction (decode) + Evaluation (prefill) tok/s, then mean (all) and mean (warmup-discarded).
set -uo pipefail

BIN="${BIN:?}"; NT="${NT:?}"; JEM="${JEM:-0}"; RUNS="${RUNS:-5}"; STEPS="${STEPS:-256}"
MODEL="${MODEL:?}"; TOK="${TOK:?}"; WORKERS="${WORKERS:?}"
PROMPT="${PROMPT:-Write a detailed technical explanation of distributed inference across multiple machines.}"

PRE=""
if [ "$JEM" = "1" ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
  export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
fi

echo "BIN=$BIN NT=$NT JEM=$JEM RUNS=$RUNS STEPS=$STEPS"
echo "binary bytes: $(stat -c%s "$BIN")"

decodes=(); prefills=()
for i in $(seq 1 "$RUNS"); do
  out=$("$BIN" inference \
    --model "$MODEL" --tokenizer "$TOK" \
    --buffer-float-type q80 --max-seq-len 4096 \
    --prompt "$PROMPT" --steps "$STEPS" --nthreads "$NT" \
    --workers $WORKERS 2>/dev/null)
  # two "tokens/s:" lines: Evaluation (prefill) then Prediction (decode)
  pf=$(echo "$out" | grep "tokens/s:" | head -1 | awk '{print $2}')
  dc=$(echo "$out" | grep "tokens/s:" | tail -1 | awk '{print $2}')
  if [ -z "$dc" ]; then
    echo "Run $i: ERROR (no output) -- last lines:"; echo "$out" | tail -4
    continue
  fi
  decodes+=("$dc"); prefills+=("$pf")
  printf "Run %d: prefill=%s tok/s | decode=%s tok/s\n" "$i" "$pf" "$dc"
done

# stats with python (mean all + mean discarding first warmup)
python3 - "${decodes[@]}" <<'PY'
import sys, statistics
vals=[float(x) for x in sys.argv[1:]]
if not vals:
    print("no successful runs"); sys.exit(0)
def stat(v,label):
    m=statistics.mean(v)
    sd=statistics.stdev(v) if len(v)>1 else 0.0
    print(f"{label}: n={len(v)} mean={m:.3f} stdev={sd:.3f} min={min(v):.3f} max={max(v):.3f}")
stat(vals,"DECODE all")
if len(vals)>1: stat(vals[1:],"DECODE warmup-discarded")
PY
