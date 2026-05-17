#!/bin/bash
# dllama-runtime-tweaks.sh — apply Pi5 LAN tunings for distributed-llama
#
# Applies the following bit-exact runtime tweaks (all NIC/scheduling, NO model math):
#   - RX ring buffer 512 -> 4096    (absorbs bursts, prevents drops)
#   - RFS enabled with CPU 1-3 mask (spreads softirq RX off CPU0)
#   - NAPI defer hard IRQs = 2      (delays hard IRQ in favour of NAPI poll)
#   - GRO flush timeout = 20 us     (low latency batching)
#
# Install to /usr/local/sbin/ and enable the matching systemd unit:
#   sudo cp dllama-runtime-tweaks.sh /usr/local/sbin/
#   sudo chmod +x /usr/local/sbin/dllama-runtime-tweaks.sh
#   sudo cp dllama-runtime-tweaks.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now dllama-runtime-tweaks.service

set -e

# RX/TX ring buffer (max for macb on Pi5 is 8192 / 4096)
ethtool -G eth0 rx 4096 tx 2048 2>/dev/null || true

# Receive Flow Steering: spread softirq RX to CPU 1-3 (CPU0 stays for IRQ)
sysctl -w net.core.rps_sock_flow_entries=32768 >/dev/null
echo 4096 > /sys/class/net/eth0/queues/rx-0/rps_flow_cnt
echo e > /sys/class/net/eth0/queues/rx-0/rps_cpus

# NAPI tuning: defer hard IRQs and bound GRO flush
echo 2 > /sys/class/net/eth0/napi_defer_hard_irqs
echo 20000 > /sys/class/net/eth0/gro_flush_timeout

exit 0
