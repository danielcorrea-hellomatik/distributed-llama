#!/usr/bin/env bash
# Audit 3: confirm deployed dllama binary uses ARMv8.2 sdot + CPU exposes dotprod.
# Read-only.
set -u

PIS=(rpi-1005 rpi-1006 rpi-1007 rpi-1008)
SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

printf "%-12s | %-7s | %-9s | %-9s | %s\n" "HOST" "CORES" "DOTPROD" "SDOT_INS" "STATUS"
printf -- "-------------+---------+-----------+-----------+--------\n"

fail=0; warn=0
for h in "${PIS[@]}"; do
  out=$(ssh $SSH_OPTS "rpi@$h" '
    bin=$HOME/distributed-llama/dllama
    [ -x "$bin" ] || bin=$(command -v dllama 2>/dev/null)
    [ -x "$bin" ] || bin=$(find $HOME /opt /usr/local -maxdepth 4 -name dllama -type f -executable 2>/dev/null | head -1)
    [ -x "$bin" ] || { echo "NO_BIN"; exit 0; }
    cores=$(grep -c "^processor" /proc/cpuinfo)
    # Linux ARM names dotprod feature "asimddp" in /proc/cpuinfo Features line
    dot=$(grep -c asimddp /proc/cpuinfo)
    command -v objdump >/dev/null || { echo "$cores|$dot|NO_OBJDUMP"; exit 0; }
    sdot=$(objdump -d "$bin" 2>/dev/null | grep -cE "\bsdot[[:space:]]+v")
    echo "$cores|$dot|$sdot"
  ' 2>/dev/null) || { printf "%-12s | %s\n" "$h" "UNREACHABLE"; fail=$((fail+1)); continue; }

  if [ "$out" = "NO_BIN" ]; then
    printf "%-12s | %s\n" "$h" "NO_BIN"; fail=$((fail+1)); continue
  fi
  IFS='|' read -r cores dot sdot <<<"$out"
  if [ "$sdot" = "NO_OBJDUMP" ]; then st="WARN"; warn=$((warn+1))
  elif [ "$dot" -ge "$cores" ] && [ "$sdot" -ge 100 ]; then st="PASS"
  elif [ "$sdot" -lt 100 ]; then st="FAIL"; fail=$((fail+1))
  else st="WARN"; warn=$((warn+1)); fi
  printf "%-12s | %-7s | %-9s | %-9s | %s\n" "$h" "$cores" "$dot" "$sdot" "$st"
done

echo
if   [ $fail -gt 0 ]; then echo "VERDICT: FAIL"; exit 1
elif [ $warn -gt 0 ]; then echo "VERDICT: WARN"; exit 2
else                       echo "VERDICT: PASS"; fi
