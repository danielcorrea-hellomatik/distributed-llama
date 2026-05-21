#!/usr/bin/env bash
# Audit 1: cpufreq governor on each Pi. Read-only.
# Expected verdict: PASS if all 4 == "performance".
set -u

PIS=(rpi-1005 rpi-1006 rpi-1007 rpi-1008)
SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

printf "%-12s | %-12s | %-10s | %s\n" "HOST" "GOVERNOR" "STATUS" "CPU0_FREQ_KHZ"
printf -- "-------------+--------------+------------+--------------\n"

fail=0; warn=0
for h in "${PIS[@]}"; do
  out=$(ssh $SSH_OPTS "rpi@$h" '
    g=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "MISSING");
    f=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "?");
    echo "$g|$f"
  ' 2>/dev/null) || { printf "%-12s | %-12s | %-10s | %s\n" "$h" "-" "UNREACHABLE" "-"; fail=$((fail+1)); continue; }

  gov="${out%%|*}"; freq="${out##*|}"
  case "$gov" in
    performance) st="PASS" ;;
    ondemand|schedutil|powersave|conservative) st="FAIL"; fail=$((fail+1)) ;;
    *) st="WARN"; warn=$((warn+1)) ;;
  esac
  printf "%-12s | %-12s | %-10s | %s\n" "$h" "$gov" "$st" "$freq"
done

echo
if   [ $fail -gt 0 ]; then echo "VERDICT: FAIL ($fail host(s) not on performance governor)"; exit 1
elif [ $warn -gt 0 ]; then echo "VERDICT: WARN ($warn host(s) unclear)"; exit 2
else                       echo "VERDICT: PASS (all 4 Pis on performance governor)"; fi
