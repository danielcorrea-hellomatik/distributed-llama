#!/usr/bin/env bash
# install-node.sh -- bootstrap a fresh Raspberry Pi 5 16GB node
#                    for the dllama-hermes-cluster.
#
# Run as user 'rpi' with sudo NOPASSWD already configured.
# This script is idempotent.

set -euo pipefail

ROLE="${1:-worker}"   # 'root' or 'worker'
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> Bootstrap node as: $ROLE"
echo ">>> Repository: $REPO_DIR"

# --- System packages ---
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential git python3-pip python3-venv \
    curl wget rsync ethtool libjemalloc2 bc

# --- Sudo NOPASSWD ---
if [ ! -f /etc/sudoers.d/010_rpi-nopasswd ]; then
    echo 'rpi ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/010_rpi-nopasswd
    sudo chmod 440 /etc/sudoers.d/010_rpi-nopasswd
fi

# --- Swap on NVMe ---
SWAPFILE="/swapfile_4g"
SWAPSIZE="4G"
if [ "$ROLE" = "root" ]; then
    SWAPFILE="/swapfile_16g"
    SWAPSIZE="16G"
fi
if [ ! -f "$SWAPFILE" ]; then
    sudo fallocate -l $SWAPSIZE $SWAPFILE
    sudo chmod 600 $SWAPFILE
    sudo mkswap $SWAPFILE
    sudo swapon $SWAPFILE
    echo "$SWAPFILE none swap sw,pri=10 0 0" | sudo tee -a /etc/fstab
fi

# --- Clone or update distributed-llama ---
cd "$HOME"
if [ ! -d distributed-llama ]; then
    git clone https://github.com/b4rtaz/distributed-llama.git
fi
cd distributed-llama
git fetch --tags --depth 100 2>/dev/null || true
git checkout v0.16.5

# --- Apply patches ---
echo ">>> Applying patches"
for p in "$REPO_DIR"/patches/*.patch; do
    if [ -f "$p" ]; then
        # Skip if already applied
        if patch --dry-run -p0 -R < "$p" > /dev/null 2>&1; then
            echo "    $(basename $p) already applied"
        else
            patch -p0 < "$p" || echo "    (patch may already be applied)"
        fi
    fi
done

# --- Compile ---
echo ">>> Compiling dllama with ARM tuning + LTO"
make clean > /dev/null
make dllama dllama-api -j4 2>&1 | tail -5

# Verify NEON dotprod
DOTPROD_COUNT=$(objdump -d ~/distributed-llama/dllama 2>/dev/null | grep -cE 'udot|sdot' || echo 0)
echo "    NEON dotprod instructions in dllama: $DOTPROD_COUNT (expect > 200)"
if [ "$DOTPROD_COUNT" -lt 200 ]; then
    echo "    WARNING: dotprod kernels not detected. Recompile with the Makefile patch."
fi

# --- systemd units ---
echo ">>> Installing systemd units"
sudo cp "$REPO_DIR/systemd/cpu-performance.service" /etc/systemd/system/
sudo cp "$REPO_DIR/systemd/eth0-tuning.service" /etc/systemd/system/
if [ "$ROLE" = "root" ]; then
    sudo cp "$REPO_DIR/systemd/dllama-api.service" /etc/systemd/system/
    sudo cp "$REPO_DIR/systemd/dllama-proxy.service" /etc/systemd/system/
else
    sudo cp "$REPO_DIR/systemd/dllama-worker.service" /etc/systemd/system/
fi

# --- sysctl ---
echo ">>> Installing sysctl tuning"
sudo cp "$REPO_DIR/sysctl/"*.conf /etc/sysctl.d/
sudo sysctl --system > /dev/null

# --- Mask parasitic timers ---
echo ">>> Masking parasitic timers"
for t in apt-daily.timer apt-daily-upgrade.timer man-db.timer e2scrub_all.timer rpi-zram-writeback.timer; do
    sudo systemctl mask $t 2>/dev/null | grep -v already || true
done

# --- Enable services ---
sudo systemctl daemon-reload
sudo systemctl enable cpu-performance eth0-tuning
if [ "$ROLE" = "root" ]; then
    sudo systemctl enable dllama-api dllama-proxy
else
    sudo systemctl enable dllama-worker
fi

# --- Python proxy (root only) ---
if [ "$ROLE" = "root" ]; then
    if [ ! -d "$HOME/litellm-env" ]; then
        python3 -m venv "$HOME/litellm-env"
        "$HOME/litellm-env/bin/pip" install --quiet flask waitress requests
    fi
    cp "$REPO_DIR/scripts/dllama_proxy.py" "$HOME/dllama_proxy.py"
fi

echo ""
echo ">>> Bootstrap complete for $(hostname) as $ROLE"
echo ">>> Next steps:"
echo "    1. Download model:  cd ~/distributed-llama && python3 launch.py qwen3_30b_a3b_q40 -skip-run"
echo "    2. (Workers): receive model via rsync from root"
echo "    3. Edit /etc/systemd/system/dllama-api.service with the actual worker IPs"
echo "    4. Start the cluster (workers first):"
echo "       sudo systemctl start dllama-worker     # on each worker"
echo "       sudo systemctl start dllama-api dllama-proxy   # on root"
echo "    5. Verify:  curl http://localhost:9999/v1/models"
