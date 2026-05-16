#!/usr/bin/env bash
# Cluster Control — operativa rápida del cluster dllama/Hermes en 4× RPi 5
# Uso: ./cluster-control.sh [status|start|stop|restart|temp|test|logs|memory]

set -e

ROOT_PI="rpi-1005"
WORKER_PIS=("rpi-1006" "rpi-1007" "rpi-1008")
ALL_PIS=("$ROOT_PI" "${WORKER_PIS[@]}")

case "${1:-status}" in
  status)
    echo "=== Servicios ==="
    ssh -o BatchMode=yes rpi@$ROOT_PI "echo '$ROOT_PI dllama-api: '\$(sudo systemctl is-active dllama-api)"
    for h in "${WORKER_PIS[@]}"; do
      ssh -o BatchMode=yes rpi@$h "echo '$h dllama-worker: '\$(sudo systemctl is-active dllama-worker)"
    done
    ;;

  start)
    for h in "${WORKER_PIS[@]}"; do
      ssh -o BatchMode=yes rpi@$h "sudo systemctl start dllama-worker"
      echo "$h worker started"
    done
    sleep 3
    ssh -o BatchMode=yes rpi@$ROOT_PI "sudo systemctl start dllama-api"
    echo "$ROOT_PI api started"
    ;;

  stop)
    ssh -o BatchMode=yes rpi@$ROOT_PI "sudo systemctl stop dllama-api"
    echo "$ROOT_PI api stopped"
    for h in "${WORKER_PIS[@]}"; do
      ssh -o BatchMode=yes rpi@$h "sudo systemctl stop dllama-worker"
      echo "$h worker stopped"
    done
    ;;

  restart)
    for h in "${WORKER_PIS[@]}"; do
      ssh -o BatchMode=yes rpi@$h "sudo systemctl restart dllama-worker" &
    done
    wait
    sleep 3
    ssh -o BatchMode=yes rpi@$ROOT_PI "sudo systemctl restart dllama-api"
    echo "Cluster restarted (workers first, then api)"
    ;;

  temp)
    echo "=== Temperaturas (throttling a 85°C) ==="
    for h in "${ALL_PIS[@]}"; do
      ssh -o BatchMode=yes rpi@$h "echo '$h: '\$(vcgencmd measure_temp) | throttled=\$(vcgencmd get_throttled | cut -d= -f2)"
    done
    ;;

  memory)
    echo "=== Memoria ==="
    for h in "${ALL_PIS[@]}"; do
      ssh -o BatchMode=yes rpi@$h "free -h | head -2 | tail -1 | awk '{print \"$h: used=\"\$3\" available=\"\$7}'"
    done
    ;;

  test)
    echo "=== Smoke test inferencia distribuida ==="
    ssh -o BatchMode=yes rpi@$ROOT_PI 'time curl -s -m 60 http://localhost:9999/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"llama\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words\"}],\"max_tokens\":15,\"temperature\":0.5}"'
    ;;

  logs)
    target="${2:-api}"
    if [ "$target" = "api" ]; then
      ssh -t rpi@$ROOT_PI "sudo journalctl -u dllama-api -f"
    else
      ssh -t rpi@$target "sudo journalctl -u dllama-worker -f"
    fi
    ;;

  *)
    echo "Uso: $0 [status|start|stop|restart|temp|memory|test|logs <host>]"
    exit 1
    ;;
esac
