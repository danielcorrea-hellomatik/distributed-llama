#!/usr/bin/env bash
# Bit-exact + speedup test for prompt-lookup speculative decoding.
# Runs greedy (no drafts) and speculative via the SAME code path, compares generated-text SHA.
set -uo pipefail
BIN=/home/rpi/dllama-spec/dllama
MODEL=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_model_deepseek-r1-distill-llama-8b_q40.m
TOK=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_tokenizer_deepseek-r1-distill-llama-8b.t
WORKERS="192.168.1.77:9998 192.168.1.75:9998 192.168.1.76:9998"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
PROMPT="${PROMPT:-Write a detailed technical explanation of distributed inference across multiple machines.}"
STEPS="${STEPS:-128}"
NGRAM="${NGRAM:-6}"
MIN="${MIN:-2}"

run() { # $1=ngram $2=min $3=outfile
  "$BIN" inference --model "$MODEL" --tokenizer "$TOK" --buffer-float-type q80 --max-seq-len 4096 \
    --temperature 0 --seed 42 --prompt "$PROMPT" --steps "$STEPS" --nthreads 3 \
    --spec-ngram "$1" --spec-min "$2" --workers $WORKERS > "$3" 2>/dev/null
}

run 1 250    /tmp/greedy.out   # spec-min huge -> never matches -> pure greedy through spec path
run "$NGRAM" "$MIN" /tmp/spec.out

PROMPT="$PROMPT" python3 - <<'PY'
import hashlib, os
PROMPT=os.environ['PROMPT']
def gen(path):
    t=open(path,encoding='utf-8',errors='replace').read()
    k=t.rfind(PROMPT+'\n')                       # generated text starts after the prompt echo
    body=t[k+len(PROMPT)+1:] if k>=0 else t
    j=body.find('\nPrediction (speculative')     # ends at the stats block
    return body[:j] if j>=0 else body
g=gen('/tmp/greedy.out'); s=gen('/tmp/spec.out')
hg=hashlib.sha256(g.encode()).hexdigest()[:16]; hs=hashlib.sha256(s.encode()).hexdigest()[:16]
print(f"greedy  sha={hg} len={len(g)}")
print(f"spec    sha={hs} len={len(s)}")
print("BIT-EXACT: " + ("PASS (identical output)" if hg==hs else "FAIL (output differs!)"))
PY
echo "--- greedy ---"; grep -E "tokens/s|accept|nTokens|nSteps" /tmp/greedy.out | sed 's/^/  /'
echo "--- spec ---";   grep -E "tokens/s|accept|nTokens|nSteps" /tmp/spec.out | sed 's/^/  /'
