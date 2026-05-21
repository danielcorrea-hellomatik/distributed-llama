#!/usr/bin/env bash
# Run the 3 pre-deploy audits in sequence and produce one report.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/dllama-audit-$(date +%Y%m%d-%H%M%S).log"

{
  echo "distributed-llama pre-deploy audit  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  echo "================================================================"
  echo "AUDIT 1 — cpufreq governor"
  echo "================================================================"
  bash "$HERE/audit-1-governor.sh"; r1=$?
  echo
  echo "================================================================"
  echo "AUDIT 2 — thermal throttle"
  echo "================================================================"
  bash "$HERE/audit-2-thermal.sh";  r2=$?
  echo
  echo "================================================================"
  echo "AUDIT 3 — SDOT in binary"
  echo "================================================================"
  bash "$HERE/audit-3-sdot.sh";     r3=$?
  echo
  echo "================================================================"
  echo "SUMMARY"
  echo "================================================================"
  printf "  Audit 1 governor : exit=%d\n" "$r1"
  printf "  Audit 2 thermal  : exit=%d\n" "$r2"
  printf "  Audit 3 sdot     : exit=%d\n" "$r3"
  if [ $r1 -eq 0 ] && [ $r2 -eq 0 ] && [ $r3 -eq 0 ]; then
    echo "OVERALL: PASS — safe to proceed with code changes"; exit 0
  else
    echo "OVERALL: NOT-PASS — investigate before deploy"; exit 1
  fi
} | tee "$LOG"
echo "Full log: $LOG"
