#!/usr/bin/env bash
# Realistic-workload characterization of adaptive PLD speculative decoding (bit-exact gate per prompt).
set -uo pipefail
BIN=/home/rpi/dllama-spec/dllama
MODEL=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_model_deepseek-r1-distill-llama-8b_q40.m
TOK=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40/dllama_tokenizer_deepseek-r1-distill-llama-8b.t
WORKERS="192.168.1.77:9998 192.168.1.75:9998 192.168.1.76:9998"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000
STEPS="${STEPS:-160}"; NGRAM="${NGRAM:-10}"; MIN="${MIN:-2}"

DOC="Document: HelloMatik is an AI customer support platform. It provides retrieval augmented generation over your documents, so answers are grounded in your own content. It supports chat across web, WhatsApp, Telegram and Discord. It also includes voice agents that answer phone calls automatically. The starter plan costs forty nine euros per month. The growth plan costs ninety nine euros per month. The enterprise plan has custom pricing. All plans include analytics and a shared team inbox."

run(){ # ngram min prompt outfile
  "$BIN" inference --model "$MODEL" --tokenizer "$TOK" --buffer-float-type q80 --max-seq-len 4096 \
    --temperature 0 --seed 42 --prompt "$3" --steps "$STEPS" --nthreads 3 \
    --spec-ngram "$1" --spec-min "$2" --workers $WORKERS > "$4" 2>/dev/null
}
test_prompt(){ # label prompt
  run 1 250 "$2" /tmp/g.out
  run "$NGRAM" "$MIN" "$2" /tmp/s.out
  LBL="$1" PR="$2" python3 - <<'PY'
import os
P=os.environ['PR']; L=os.environ['LBL']
def gen(p):
    t=open(p,encoding='utf-8',errors='replace').read()
    k=t.rfind(P+'\n'); b=t[k+len(P)+1:] if k>=0 else t
    j=b.find('\nPrediction (speculative'); return b[:j] if j>=0 else b
def f(p,key,sp):
    for ln in open(p):
        if key in ln: return ln.split(sp)[1].strip()
    return '?'
g=gen('/tmp/g.out'); s=gen('/tmp/s.out')
be='PASS' if g==s else 'FAIL'
gt=f('/tmp/g.out','tokens/s','tokens/s:').split()[0]
st=f('/tmp/s.out','tokens/s','tokens/s:').split()[0]
acc=f('/tmp/s.out','accept','accept:')
try: spd=float(st)/float(gt)
except: spd=0
print(f"{L:14s} bitexact={be} | greedy {gt:>5} -> spec {st:>6} ({spd:.2f}x) | accept {acc}")
PY
}
echo "=== ADAPTIVE realistic bench  STEPS=$STEPS NGRAM(max)=$NGRAM MIN=$MIN ==="
test_prompt "rag-extract"  "$DOC Task: list each plan and its exact price, copying the wording from the document above."
test_prompt "format-quote" "$DOC Task: rewrite the document above as a bulleted list, keeping every sentence word for word."
test_prompt "summarize"    "$DOC Task: summarize the document above in two sentences in your own words."
test_prompt "qa-grounded"  "$DOC Question: which channels are supported and how much is the growth plan? Answer using the document."
test_prompt "long-repeat"  "Repeat this line exactly forty times: HelloMatik makes customer support effortless and fast."
