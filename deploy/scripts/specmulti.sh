#!/usr/bin/env bash
# Multi-prompt characterization of prompt-lookup speculative decoding (bit-exact gate per prompt).
set -uo pipefail
BIN=/home/rpi/dllama-spec/dllama
MODEL=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_model_deepseek-r1-distill-llama-8b_q40.m
TOK=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_tokenizer_deepseek-r1-distill-llama-8b.t
WORKERS="192.168.1.77:9998 192.168.1.75:9998 192.168.1.76:9998"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
STEPS="${STEPS:-140}"; NGRAM="${NGRAM:-10}"; MIN="${MIN:-2}"

run(){ # ngram min prompt outfile
  "$BIN" inference --model "$MODEL" --tokenizer "$TOK" --buffer-float-type q80 --max-seq-len 4096 \
    --temperature 0 --seed 42 --prompt "$3" --steps "$STEPS" --nthreads 3 \
    --spec-ngram "$1" --spec-min "$2" --workers $WORKERS > "$4" 2>/dev/null
}

test_prompt(){ # label prompt
  run 1 250 "$2" /tmp/g.out               # greedy (no drafts)
  run "$NGRAM" "$MIN" "$2" /tmp/s.out      # speculative
  LBL="$1" PR="$2" python3 - <<'PY'
import hashlib,os
P=os.environ['PR']; L=os.environ['LBL']
def gen(p):
    t=open(p,encoding='utf-8',errors='replace').read()
    k=t.rfind(P+'\n'); b=t[k+len(P)+1:] if k>=0 else t
    j=b.find('\nPrediction (speculative'); return b[:j] if j>=0 else b
def field(p,key,split):
    for ln in open(p):
        if key in ln: return ln.split(split)[1].strip()
    return '?'
g=gen('/tmp/g.out'); s=gen('/tmp/s.out')
be='PASS' if g==s else 'FAIL'
gt=field('/tmp/g.out','tokens/s','tokens/s:').split()[0]
st=field('/tmp/s.out','tokens/s','tokens/s:').split()[0]
acc=field('/tmp/s.out','accept','accept:')
try: spd=float(st)/float(gt)
except: spd=0
print(f"{L:13s} bitexact={be} | greedy {gt:>5} -> spec {st:>6} tok/s  ({spd:.2f}x) | accept {acc}")
PY
}

echo "=== STEPS=$STEPS NGRAM=$NGRAM MIN=$MIN ==="
test_prompt "repetitive" "Repeat exactly: All work and no play makes Jack a dull boy. All work and no play makes Jack a dull boy. All work and no play makes Jack a dull boy."
test_prompt "list-50"     "List all 50 US states in alphabetical order, one per line, nothing else."
test_prompt "json"        "Output only a JSON array of 15 objects, each with fields id, name and value."
test_prompt "code-tests"  "Write a Python function is_prime(n), then 15 assert-based unit tests, one per line."
test_prompt "rag-quote"   "Text: The mitochondria is the powerhouse of the cell and produces ATP. Task: copy that exact sentence four times."
test_prompt "reasoning"   "A train leaves city A at 60 km per hour and another leaves city B at 80 km per hour. Explain step by step how to find when they meet."
