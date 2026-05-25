#!/usr/bin/env bash
# dsdiag.sh — decode bottleneck decomposition for distributed-llama CLI.
# Parses the per-token "Pred X ms Sync Y ms | Sent Z kB Recv W kB" lines.
# Env in: BIN NT JEM STEPS MODEL TOK WORKERS PROMPT WARMUP
set -uo pipefail
BIN="${BIN:?}"; NT="${NT:?}"; JEM="${JEM:-0}"; STEPS="${STEPS:-60}"; WARMUP="${WARMUP:-8}"
MODEL="${MODEL:?}"; TOK="${TOK:?}"; WORKERS="${WORKERS:?}"
PROMPT="${PROMPT:-Write a detailed technical explanation of distributed inference across multiple machines.}"
if [ "$JEM" = "1" ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
  export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
fi
out=$("$BIN" inference --model "$MODEL" --tokenizer "$TOK" \
  --buffer-float-type q80 --max-seq-len 4096 \
  --prompt "$PROMPT" --steps "$STEPS" --nthreads "$NT" --workers $WORKERS 2>/dev/null)
echo "$out" | grep "tokens/s:" | sed 's/^/SUMMARY /'
# decompose decode tokens (lines with Pred & Sync), skip WARMUP
echo "$out" | awk -v warm="$WARMUP" '
/Pred/ && /Sync/ {
  pred=sync=sent=recv=0
  for(i=1;i<=NF;i++){ if($i=="Pred")pred=$(i+1); if($i=="Sync")sync=$(i+1); if($i=="Sent")sent=$(i+1); if($i=="Recv")recv=$(i+1) }
  n++; if(n<=warm) next
  c++; sp+=pred; ss+=sync; se+=sent; sr+=recv
}
END{
  if(c==0){print "no decode tokens parsed"; exit}
  ap=sp/c; as=ss/c; tot=ap+as
  printf "DECODE tokens analyzed: %d (warmup %d skipped)\n", c, warm
  printf "  avg Pred (compute+mem): %.1f ms/tok  (%.1f%%)\n", ap, 100*ap/tot
  printf "  avg Sync (network)    : %.1f ms/tok  (%.1f%%)\n", as, 100*as/tot
  printf "  total                 : %.1f ms/tok  -> %.2f tok/s\n", tot, 1000/tot
  printf "  net per tok           : Sent %.0f kB  Recv %.0f kB\n", se/c, sr/c
}'
