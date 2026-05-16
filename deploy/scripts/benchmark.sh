#!/usr/bin/env bash
# benchmark.sh -- wrapper for benchmark.py
# Runs from the root node and reports throughput statistics.

set -euo pipefail

if [ -z "${SSH_CONNECTION:-}" ] && [ "$(hostname)" != *rpi* ]; then
    # Running from operator machine
    REMOTE="${1:-rpi-1005}"
    echo ">>> Running benchmark on $REMOTE (20 runs)"
    ssh "rpi@$REMOTE" 'python3 ~/distributed-llama/scripts/benchmark.py' 2>/dev/null \
        || ssh "rpi@$REMOTE" 'python3 /tmp/academic_bench.py'
else
    # Running on the node itself
    python3 ~/distributed-llama/scripts/benchmark.py 2>/dev/null \
        || python3 /tmp/academic_bench.py
fi
