#!/usr/bin/env bash
# dstest.sh — one deterministic decode run: tok/s + Pred/Sync split + bit-exact output signature.
# Env in: BIN NT JEM TILE STEPS MODEL TOK WORKERS PROMPT WARMUP
set -uo pipefail
BIN="${BIN:?}"; NT="${NT:?}"; JEM="${JEM:-0}"; TILE="${TILE:-0}"; STEPS="${STEPS:-96}"; WARMUP="${WARMUP:-8}"
MODEL="${MODEL:?}"; TOK="${TOK:?}"; WORKERS="${WORKERS:?}"
PROMPT="${PROMPT:-Write a detailed technical explanation of distributed inference across multiple machines.}"
if [ "$JEM" = "1" ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
  export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
fi
TILEARG=""; [ "$TILE" -gt 0 ] && TILEARG="--tile-sync $TILE"
out=$("$BIN" inference --model "$MODEL" --tokenizer "$TOK" \
  --buffer-float-type q80 --max-seq-len 4096 --temperature 0 --seed 42 \
  --prompt "$PROMPT" --steps "$STEPS" --nthreads "$NT" $TILEARG --workers $WORKERS 2>&1)
dc=$(echo "$out" | grep "tokens/s:" | tail -1 | awk '{print $2}')
sig=$(echo "$out" | grep "Pred" | grep "Sync" | sed 's/.*| //' | tr -d '\n' | sha256sum | cut -c1-16)
echo "$out" | awk -v warm="$WARMUP" '
/Pred/ && /Sync/ { p=s=0; for(i=1;i<=NF;i++){if($i=="Pred")p=$(i+1); if($i=="Sync")s=$(i+1)} n++; if(n<=warm)next; c++; sp+=p; ss+=s }
END{ if(c){ap=sp/c;as=ss/c;t=ap+as; printf "  Pred=%.1fms(%.0f%%) Sync=%.1fms(%.0f%%) total=%.1fms\n",ap,100*ap/t,as,100*as/t,t} }'
echo "  TILE=$TILE NT=$NT -> decode=$dc tok/s  OUTPUT_SHA=$sig"
