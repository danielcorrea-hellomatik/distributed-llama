#!/usr/bin/env bash
# Audit 2: thermal/clock throttling under 60s of arbitrary CPU load.
# Spawns 4 `yes > /dev/null` then kills them. No config changes.
# Expected verdict: PASS if clock holds ~2.4 GHz and get_throttled == 0x0.
set -u

PIS=(rpi-1005 rpi-1006 rpi-1007 rpi-1008)
SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
DURATION=60

printf "%-12s | %-9s | %-9s | %-6s | %-9s | %s\n" \
  "HOST" "CLK_PRE" "CLK_LOAD" "TEMP" "THROTTLE" "STATUS"
printf -- "-------------+-----------+-----------+--------+-----------+--------\n"

fail=0; warn=0
for h in "${PIS[@]}"; do
  out=$(ssh $SSH_OPTS "rpi@$h" "
    set -u
    command -v vcgencmd >/dev/null || { echo 'NO_VCGENCMD'; exit 0; }
    clk_pre=\$(vcgencmd measure_clock arm | awk -F'=' '{print \$2}')
    for i in 1 2 3 4; do yes > /dev/null & done
    LOADPIDS=\$(jobs -p)
    sleep $((DURATION/2))
    clk_load=\$(vcgencmd measure_clock arm | awk -F'=' '{print \$2}')
    temp=\$(awk '{printf \"%.1f\", \$1/1000}' /sys/class/thermal/thermal_zone0/temp)
    thr_end=\$(vcgencmd get_throttled | awk -F'=' '{print \$2}')
    sleep $((DURATION/2))
    kill \$LOADPIDS 2>/dev/null; wait 2>/dev/null
    echo \"\$clk_pre|\$clk_load|\$temp|\$thr_end\"
  " 2>/dev/null) || { printf "%-12s | %s\n" "$h" "UNREACHABLE"; fail=$((fail+1)); continue; }

  if [ "$out" = "NO_VCGENCMD" ]; then
    printf "%-12s | %s\n" "$h" "NO_VCGENCMD"; warn=$((warn+1)); continue
  fi
  IFS='|' read -r clk_pre clk_load temp thr_end <<<"$out"
  clk_load_mhz=$(( clk_load / 1000000 ))
  if [ "$thr_end" != "0x0" ]; then st="FAIL"; fail=$((fail+1))
  elif [ "$clk_load_mhz" -lt 2300 ]; then st="WARN"; warn=$((warn+1))
  else st="PASS"; fi
  printf "%-12s | %-9s | %-9s | %-6s | %-9s | %s\n" \
    "$h" "$((clk_pre/1000000))M" "${clk_load_mhz}M" "${temp}C" "$thr_end" "$st"
done

echo
if   [ $fail -gt 0 ]; then echo "VERDICT: FAIL ($fail host(s) throttled)"; exit 1
elif [ $warn -gt 0 ]; then echo "VERDICT: WARN ($warn host(s) below 2.3 GHz under load)"; exit 2
else                       echo "VERDICT: PASS (no throttle on 4 Pis at ${DURATION}s)"; fi
