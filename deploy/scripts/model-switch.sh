#!/usr/bin/env bash
# model-switch.sh — sirve UN modelo a la vez en el cluster 4xPi5, en :9999.
#   El cluster no puede correr dos modelos a la vez (los workers hacen busy-spin
#   y saturan los 4 cores). Este script alterna Qwen <-> DeepSeek limpiamente.
#
# Uso:  ./model-switch.sh status | qwen | deepseek
#   qwen      -> Qwen3-30B-A3B (produccion, via systemd dllama-api/dllama-worker)
#   deepseek  -> DeepSeek-R1-8B (transitorio via systemd-run, mismo :9999)
#
# Ejecutar desde el Mac (usa IPs LAN). Requiere sudo sin password en los nodos.
set -euo pipefail

ROOT=192.168.1.74
WORKERS=(192.168.1.77 192.168.1.75 192.168.1.76)   # mismo orden que usa Qwen
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"

JEM="--setenv=LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 --setenv=MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000"
DS_DIR=/home/rpi/distributed-llama/models/deepseek_r1_distill_llama_8b_q40
DS_MODEL=$DS_DIR/dllama_model_deepseek-r1-distill-llama-8b_q40.m
DS_TOK=$DS_DIR/dllama_tokenizer_deepseek-r1-distill-llama-8b.t
DLLAMA=/home/rpi/distributed-llama/dllama
DLLAMA_API=/home/rpi/distributed-llama/dllama-api

stop_deepseek() {
  $SSH rpi@$ROOT 'sudo systemctl stop dllama-ds-api 2>/dev/null; sudo systemctl reset-failed dllama-ds-api 2>/dev/null' || true
  for w in "${WORKERS[@]}"; do
    $SSH rpi@$w 'sudo systemctl stop dllama-ds-worker 2>/dev/null; sudo systemctl reset-failed dllama-ds-worker 2>/dev/null' || true
  done
}
stop_qwen() {
  $SSH rpi@$ROOT 'sudo systemctl stop dllama-api' || true
  for w in "${WORKERS[@]}"; do $SSH rpi@$w 'sudo systemctl stop dllama-worker' || true; done
}

wait_ready() {  # $1=port
  local port=$1 i
  for i in $(seq 1 50); do
    if $SSH rpi@$ROOT "ss -tln 2>/dev/null | grep -q ':$port'"; then echo "  :$port escuchando (~$((i*3))s)"; return 0; fi
    sleep 3
  done
  echo "  ERROR: :$port no escucha. Log:"; $SSH rpi@$ROOT 'journalctl -u dllama-ds-api --no-pager -n 12' 2>/dev/null || true
  return 1
}

case "${1:-status}" in
  deepseek)
    echo "==> Cambiando a DeepSeek-R1-8B (Qwen se detiene)"
    stop_deepseek; stop_qwen; sleep 2
    echo "  arrancando workers DeepSeek (:9998, nthreads 4)"
    for w in "${WORKERS[@]}"; do
      $SSH rpi@$w "sudo systemctl reset-failed dllama-ds-worker 2>/dev/null; sudo systemd-run --unit=dllama-ds-worker --collect -p LimitMEMLOCK=infinity -p Nice=-10 -p Restart=on-failure -p RestartSec=5 $JEM $DLLAMA worker --port 9998 --nthreads 4" >/dev/null
    done
    sleep 3
    echo "  arrancando api DeepSeek (:9997, nthreads 4)"
    $SSH rpi@$ROOT "sudo systemctl reset-failed dllama-ds-api 2>/dev/null; sudo systemd-run --unit=dllama-ds-api --collect -p LimitMEMLOCK=infinity -p Nice=-10 -p Restart=on-failure -p RestartSec=8 $JEM $DLLAMA_API --host 0.0.0.0 --port 9997 --model $DS_MODEL --tokenizer $DS_TOK --buffer-float-type q80 --workers ${WORKERS[0]}:9998 ${WORKERS[1]}:9998 ${WORKERS[2]}:9998 --nthreads 4 --max-seq-len 4096" >/dev/null
    wait_ready 9997 && echo "==> DeepSeek listo en :9997"
    ;;
  qwen)
    echo "==> Cambiando a Qwen3-30B-A3B (produccion)"
    stop_deepseek; sleep 2
    echo "  arrancando workers Qwen (:9998)"
    for w in "${WORKERS[@]}"; do $SSH rpi@$w 'sudo systemctl start dllama-worker'; done
    sleep 3
    echo "  arrancando api Qwen (:9999)"
    $SSH rpi@$ROOT 'sudo systemctl start dllama-api'
    wait_ready 9999 && echo "==> Qwen listo en :9999"
    ;;
  status)
    echo "=== root :9999 ==="
    $SSH rpi@$ROOT 'echo -n "  qwen-api(systemd): "; systemctl is-active dllama-api; echo -n "  ds-api(transient): "; systemctl is-active dllama-ds-api 2>/dev/null || echo inactive; echo -n "  load: "; awk "{print \$1,\$2,\$3}" /proc/loadavg'
    ;;
  *) echo "Uso: $0 [status|qwen|deepseek]"; exit 1 ;;
esac
