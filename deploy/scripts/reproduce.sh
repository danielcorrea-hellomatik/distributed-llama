#!/usr/bin/env bash
#
# One-command reproduction of the headline result.
# Run from an operator machine with SSH access to all 4 nodes.
#
#   ROOT=rpi-1005 WORKERS="rpi-1006 rpi-1007 rpi-1008" ./deploy/scripts/reproduce.sh
#
# Steps: cold restart (workers -> root) -> wait ready -> bit-exact check -> n=20 benchmark.
# A correct cluster prints "PASS (bit-exact)" and a mean of ~14.3 tok/s (95% CI +/- ~0.05).
set -euo pipefail

ROOT="${ROOT:-rpi-1005}"
WORKERS="${WORKERS:-rpi-1006 rpi-1007 rpi-1008}"
USER_AT="${USER_AT:-rpi}"
REPO="${REPO:-\$HOME/distributed-llama}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"

echo "==> [1/4] cold restart (workers first, then root)"
$SSH "$USER_AT@$ROOT" 'sudo systemctl stop dllama-api' || true
for w in $WORKERS; do $SSH "$USER_AT@$w" 'sudo systemctl stop dllama-worker' & done; wait
for w in $WORKERS; do $SSH "$USER_AT@$w" 'sudo systemctl start dllama-worker' & done; wait
$SSH "$USER_AT@$ROOT" 'sudo systemctl reset-failed dllama-api 2>/dev/null || true; sudo systemctl start dllama-api'

echo "==> [2/4] waiting for the model to load (up to ~200 s)"
$SSH "$USER_AT@$ROOT" '
  for i in $(seq 1 40); do
    r=$(curl -s -m 25 http://127.0.0.1:9999/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":3,\"temperature\":0}" 2>/dev/null)
    echo "$r" | grep -q completion_tokens && { echo "ready after ~$((i*5))s"; exit 0; }
    sleep 5
  done
  echo "ERROR: cluster did not become ready"; exit 1
'

echo "==> [3/4] bit-exact validation (SHA-256 of deterministic output)"
$SSH "$USER_AT@$ROOT" "python3 $REPO/deploy/scripts/verify_bitexact.py"

echo "==> [4/4] n=20 throughput benchmark (cold cluster)"
$SSH "$USER_AT@$ROOT" "python3 $REPO/deploy/scripts/academic_bench.py"
