# Per-node deployment guide

This document describes how to bootstrap a fresh Raspberry Pi 5 (16 GB) node and join it to the cluster. The first node becomes the root (rank 0, serves HTTP API). Subsequent nodes become workers.

## Prerequisites

- Raspberry Pi 5 16 GB with active cooling (passive will throttle under sustained load)
- NVMe HAT and NVMe SSD (>= 100 GB; the model uses ~18 GB)
- Gigabit Ethernet, all nodes on the same LAN segment
- Raspberry Pi OS Lite 64-bit (Debian 13 trixie, kernel 6.12+)
- SSH access with key authentication; user is `rpi`
- (Optional) Tailscale enrolment for remote operation

## Bootstrap a node from scratch

On your operator machine:

```bash
ssh-copy-id rpi@<node-hostname>
```

On the new node:

```bash
sudo apt-get update
sudo apt-get install -y -qq \
    build-essential git python3-pip python3-venv \
    curl wget rsync ethtool libjemalloc2 bc
```

Configure sudo without password (required for systemd management):

```bash
echo 'rpi ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/010_rpi-nopasswd
sudo chmod 440 /etc/sudoers.d/010_rpi-nopasswd
```

Create the swap file on NVMe (16 GB on root, 4 GB on workers):

```bash
# Root only
sudo fallocate -l 16G /swapfile_16g
sudo chmod 600 /swapfile_16g
sudo mkswap /swapfile_16g
sudo swapon /swapfile_16g
echo '/swapfile_16g none swap sw,pri=10 0 0' | sudo tee -a /etc/fstab
```

```bash
# Workers
sudo fallocate -l 4G /swapfile_4g
sudo chmod 600 /swapfile_4g
sudo mkswap /swapfile_4g
sudo swapon /swapfile_4g
echo '/swapfile_4g none swap sw,pri=10 0 0' | sudo tee -a /etc/fstab
```

## Clone distributed-llama (Pi 5 cluster fork)

```bash
cd ~
git clone -b pi5-cluster https://github.com/danielcorrea-hellomatik/distributed-llama.git
cd distributed-llama
```

The `pi5-cluster` branch already includes the ARM Cortex-A76 optimisations, the socket tuning patches and all other fixes integrated into the source. No separate patch step is required.

Optional: run the unattended install script for the rest of the system setup (sudo NOPASSWD, swap, systemd units, sysctl):

```bash
bash deploy/scripts/install-node.sh root      # on the root node
bash deploy/scripts/install-node.sh worker    # on each worker
```

## Compile with the optimised Makefile

The patches include the Makefile change so the build picks up the ARM-specific flags automatically:

```bash
cd ~/distributed-llama
make clean
make dllama dllama-api -j4
```

Verify dotprod kernels are present:

```bash
objdump -d ~/distributed-llama/dllama | grep -cE 'udot|sdot'
# Expected: > 200 (we see 322)
```

## Download the model

```bash
cd ~/distributed-llama
python3 launch.py qwen3_30b_a3b_q40 -skip-run
```

This downloads ~18 GB to `~/distributed-llama/models/qwen3_30b_a3b_q40/`.

Distribute the same files to all worker nodes:

```bash
# From root node:
for h in worker-1 worker-2 worker-3; do
    rsync -az --info=progress2 \
        ~/distributed-llama/models/qwen3_30b_a3b_q40/ \
        rpi@$h:~/distributed-llama/models/qwen3_30b_a3b_q40/ &
done
wait
```

## Install systemd units

Copy from this repository and enable:

```bash
# All nodes
sudo cp systemd/cpu-performance.service /etc/systemd/system/
sudo cp systemd/eth0-tuning.service /etc/systemd/system/

# Root only
sudo cp systemd/dllama-api.service /etc/systemd/system/
sudo cp systemd/dllama-proxy.service /etc/systemd/system/

# Workers only
sudo cp systemd/dllama-worker.service /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable cpu-performance eth0-tuning
# Root
sudo systemctl enable dllama-api dllama-proxy
# Workers
sudo systemctl enable dllama-worker
```

Adjust the IPs in `dllama-api.service` to match your actual workers, e.g.:

```
--workers 192.168.1.77:9998 192.168.1.75:9998 192.168.1.76:9998
```

## Apply sysctl tuning

```bash
sudo cp sysctl/99-dllama-extra.conf /etc/sysctl.d/
sudo cp sysctl/99-dllama-sched.conf /etc/sysctl.d/
sudo sysctl --system
```

## Install the Python proxy (root only)

```bash
sudo apt-get install -y python3-venv
python3 -m venv ~/litellm-env
~/litellm-env/bin/pip install --quiet flask waitress requests
cp scripts/dllama_proxy.py ~/dllama_proxy.py
sudo systemctl start dllama-proxy
```

## Disable parasitic timers

```bash
for t in apt-daily.timer apt-daily-upgrade.timer man-db.timer e2scrub_all.timer rpi-zram-writeback.timer; do
    sudo systemctl mask $t
done
```

## Start the cluster

Order matters: workers must be up before the root contacts them.

```bash
# Workers (parallel from operator)
for h in worker-1 worker-2 worker-3; do
    ssh rpi@$h 'sudo systemctl start dllama-worker' &
done
wait

# Root
ssh rpi@root-node 'sudo systemctl start dllama-api dllama-proxy'
```

## Verify

```bash
# Endpoint is up:
curl -s http://root-node:9999/v1/models | head

# Quick inference:
curl -s -m 30 http://root-node:9999/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3","messages":[{"role":"user","content":"Reply OK"}],"max_tokens":10}'

# Full benchmark (20 runs with statistics):
ssh rpi@root-node 'python3 ~/distributed-llama/scripts/benchmark.py'
```

Expected result on a correctly-set 4-Pi cluster: ~13.8 tok/s mean, 95% CI +/- 0.05 tok/s.

## Operations

The `deploy/scripts/cluster-control.sh` helper handles day-to-day operations:

```bash
./scripts/cluster-control.sh status     # what is running
./scripts/cluster-control.sh test       # quick smoke test
./scripts/cluster-control.sh temp       # temperatures
./scripts/cluster-control.sh memory     # RAM usage per node
./scripts/cluster-control.sh restart    # ordered restart
./scripts/cluster-control.sh stop       # graceful stop
./scripts/cluster-control.sh logs api   # tail logs of dllama-api
```

## Tuning notes

- Active cooling is mandatory. Without it, sustained inference will throttle below 75% of peak after ~10 minutes.
- Use Gigabit Ethernet, never Wi-Fi. Network throughput collapses ~30% on Wi-Fi.
- The node count must be a power of 2 (1, 2, 4, 8). Three or five nodes will not boot.
- `--nthreads` must equal the number of physical cores. For Pi 5 use `4`. Higher values are rejected by dllama.
- `--max-seq-len` must fit in RAM. We use 32768; going to 65536 forced swap and reduced throughput.
- All workers must run the SAME `dllama` binary version, including patches. Mismatch breaks the all-reduce protocol silently.

## Troubleshooting

```bash
# Cluster will not start
sudo journalctl -u dllama-api -u dllama-worker -n 40 --no-pager

# Workers run but root crashes on first request
# -> Usually means worker has different model file. Re-rsync.

# Heavy swap pressure
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo swapoff -a && sudo swapon -a

# Test individual worker connectivity
nc -zv <worker-ip> 9998

# Confirm dotprod kernels were compiled
objdump -d /home/rpi/distributed-llama/dllama | grep -cE 'udot|sdot'
# Must be > 200; if zero, recompile with the Makefile patch applied.
```
