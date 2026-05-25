# DeepSeek-R1-Distill-8B — raw measurements log

> **SUPERSEDED (nthreads sweep, §4):** the old nthreads sweep below (nt3 optimal, nt4 regressing)
> is superseded by the clean-room n=8 sweep in `cleanroom_sweep_2026-05-25.md`, which finds
> **nt4 optimal (8.36 tok/s, +16% over nt3)** under the current kernel. The apparent reversal is
> attributed to the kernel upgrade (6.18.29, 2026-05-12); RPS/IRQ steering was tested directly and
> found to be a no-op (NIC has one rx queue; its hardware IRQ and the RPS target both land on CPU3).
> The old data below is kept for the record but is no longer load-bearing for the thread-count optimum.

Cluster: 4×Raspberry Pi 5 16GB (rpi-1005 root + 1006/1007/1008 workers), Gigabit LAN.
Model: `deepseek_r1_distill_llama_8b_q40` (Q40, dense 8B, Llama-3 arch, 128k tok), 6.32 GB.
Constraints: 2.4 GHz (no overclock), bit-exact (temp 0, seed 42), no model change.
Metric: CLI `dllama inference` Prediction (decode) tok/s — same as record #162.
Scripts: `deploy/scripts/{recordbench,qdecode,qd2,paperbench,dsbench,dsdiag,spectest,specmulti,specnight}.sh`.

## 1. Record-comparable decode (plain, no spec, optimized stack, steps 256)
Prompt-independent (steady-state):
- "Hello World":                      prefill 8.00/8.08/8.16,  decode 7.67/7.68/7.65
- "The Eiffel Tower is...":           prefill 12.46/13.62/13.69, decode 7.69/7.67/7.66
- "Write a short story about the sea": prefill 15.98/15.96/15.89, decode 7.66/7.66/7.67
=> decode ≈ 7.66 tok/s, identical across prompts. n=20 authoritative run: see paperbench_n20.log.

## 2. Attribution vs clean upstream (git clone b4rtaz/distributed-llama, default make), same boards
- A) ours (optimized binary, nthreads 3, jemalloc):     7.63/7.57/7.68  (mean ~7.63)
- B) upstream out-of-box (clean binary, nthreads 4, no jemalloc): 7.81/7.59/7.65 (mean ~7.68)
- C) upstream binary + our runtime (nthreads 3, jemalloc): 7.11/7.13/7.12 (mean ~7.12)
=> At each one's optimum, ours ≈ upstream (7.63 ≈ 7.68). Our source/runtime contribution to DENSE decode ≈ 0%.
   Our binary IS faster at equal nthreads (7.63 vs 7.12 at nt3) but upstream recovers with nt4.
Upstream Makefile flags: `-march=native -mtune=native -O3` (no -flto/-mcpu=cortex-a76/+dotprod/kernel-fusion).

## 3. Vanilla vs optimized (earliest A/B, n=5 × 256 steps)
- VANILLA (stock binary, nthreads 4, no jemalloc): decode mean 7.480 (σ0.052), prefill ~8.52
- OPTIMIZED (custom binary, nthreads 3, jemalloc):  decode mean 7.492 (σ0.049), prefill ~15.81
=> decode +0.2% (null), prefill +86%. (All bit-exact, SHA c4802bf0825d0760.)

## 4. nthreads sweep (all nodes) — SUPERSEDED by `cleanroom_sweep_2026-05-25.md` (nt4 optimal, 8.36)
- nt2 = 6.63 / 6.815
- nt3 = 7.49 / 7.69  (OPTIMAL — stale; nt4 now optimal under kernel 6.18.29)
- nt4 = 7.21 / 7.40  (4th thread contends with pinned NIC IRQ; prefill higher ~18)
> Superseded by the clean-room n=8 sweep: nt2=5.36, nt3=7.20, **nt4=8.36 (optimal, +16% over nt3)**.
> The reversal is attributed to the kernel upgrade (not isolated); RPS/IRQ steering tested = no-op.

## 5. Bottleneck decomposition (script dsdiag.sh, steps 60, warmup 8)
- Pred (compute+memory): 111.6 ms/tok (87.2%)
- Sync (network all-reduce): 16.4 ms/tok (12.8%)
- net per token: Sent 864 kB, Recv 1191 kB
- roofline: ~1.1 GB/node / 0.1116 s ≈ 10.1 GB/s per node (vendor ceiling ~17 GB/s)
- perf during steady decode: IPC 0.81, cores not saturated (memory-stalled)

## 6. Tile-sync (Phase B / FlashOverlap, --tile-sync K), bit-exact (SHA c4802bf0825d0760 all)
- K0 = 7.31, K2 = 7.29, K4 = 7.40, K8 = 7.12  => no overlap benefit in batch-1 decode (Sync rises with K)

## 7. Speculative decoding — prompt-lookup (PLD), bit-exact (sha identical greedy vs spec)
FIXED draft ngram=10 min=2 (steps 140): repetitive 1.76×; rag 1.09×; code 0.79×; reasoning 0.71×; json 0.56×; list 0.36×
  => large fixed draft LOSES on non-repetitive (batch>1 compute-bound, ~3× cost; break-even ≈3 tok/step)
CONSERVATIVE ngram=4 min=3: repetitive 1.74×; rag 1.16×; reasoning 1.02×; list 0.98×; json 0.97×; code 0.91×
ADAPTIVE (kCur grow+2/shrink/probe), ngram max 10 min 2:
  repetitive 1.96× (93/93, 8.75 tok/step); rag-quote 1.18× (47/69); reasoning 1.01×; code 0.99×; list 0.99×; json 0.98×
  realistic RAG (STEPS 160/400, R1 thinks first → novel): rag-extract 0.95-0.99×, summarize 0.92-1.02×, qa-grounded 1.02-1.06×, long-repeat 1.73-1.92×
  => adaptive never penalizes (≥~0.95×); real gains only on repetitive / quoted-from-context output.
Implementation: src/dllama.cpp inferenceSpec() (root-only, reuses batched forward; workers unchanged). Flags --spec-ngram/--spec-min.

## 8. OS levers A/B (qdecode.sh, vs baseline ~7.66), all bit-exact by construction
- governor performance: already set; 16KB pages: already active; THP: NOT available in kernel
- NIC ring/coalescing (ethtool -G/-C): FAIL (Pi5 driver unsupported); decode unchanged 7.67
- jemalloc dirty/muzzy_decay_ms:-1: 7.72/7.73/7.75 then 7.64/7.67/7.63 combined => within noise
- swapoff -a + swappiness=1: 7.64/7.67/7.63 => no gain (swap unused, weights mlock'd)
=> No OS lever moves decode. Memory wall confirmed.

## 9. Thermals (during/after sustained spec runs, 2400 MHz)
- rpi-1005..1008: 53-55°C, throttled=0x0 (no throttle). Sustained 400-step run bit-exact.

## 10. Records landscape (web research)
- 4×Pi5: 6.43 (#162, v0.12.2), 6.06 (Seeed 8+4GB). 2×Pi5: 3.54.
- 1×Pi5 (ollama/llama.cpp) DeepSeek-8B: ~1.2-1.4 tok/s. 1.5B distill: ~6.
- 8×Intel AVX2 + 2.5GbE: 16.63 (NOT Pi). "200 tok/s" = Hailo NPU accelerator (excluded).
=> world record for the model on 4×Pi5 (any software) = 6.43. This work 7.66 = +19%.

## 11. OC incident (negative result)
arm_freq=3000 + reboot → rpi-1005 & rpi-1006 failed to boot (silicon lottery; 1007/1008 OK at 3.0GHz).
Recovery: rescue microSD (remove NVMe → boot SD → dtparam=pciex1 + BOOT_ORDER=0xf461 via rpi-eeprom-config --apply,
VERIFY:SUCCESS → reboot → reconnect NVMe → mount /dev/nvme0n1p1, delete OC lines → remove SD). Back to 2400, no FS damage.
2.8GHz (before 3.0 failure) showed bit-exact +14% decode — but remote OC that can strand a node is not a defensible lever.
