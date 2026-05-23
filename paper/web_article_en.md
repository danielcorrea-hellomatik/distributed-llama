<!-- paper-url: https://github.com/hellomatik-org/distributed-llama/blob/pi5-cluster/paper/main_en.pdf -->

A 30-billion-parameter Mixture-of-Experts model (Qwen3-30B-A3B) runs on **four Raspberry Pi 5 boards** — CPU-only hardware, no GPU, no NPU — at **15.143 tok/s decode**: bit-exact and **+16.1% above the highest publicly documented result** for this model and hardware class (13.04 tok/s, b4rtaz #255). On identical 16 GB silicon the same result is **+15.2%** over the vanilla `distributed-llama` baseline (13.15 → 15.143 tok/s) — a confound-free figure attributable to our software and configuration alone; the cross-SKU comparison carries no measurable hardware confound either, since vanilla on our 16 GB cluster measures 13.15 tok/s, within run-to-run noise of the 8 GB record. This is a condensed technical write-up of our report; the full paper — every table, figure and raw log — is one click away.

## Results at a glance

| Metric | Value |
|---|---|
| Decode throughput (vs the public ceiling) | **15.143 tok/s** |
| Improvement over b4rtaz #255 (13.04 tok/s, decode) | **+16.1%** |
| Improvement over vanilla on identical 16 GB silicon (confound-free) | **+15.2%** |
| Sustained serving throughput (prefill included) | 14.449 tok/s |
| Time-to-first-token (TTFT) | 557 ms |
| Per-node DRAM bandwidth (sustained / vendor ceiling) | 11.4 / 17 GB/s |
| Hardware | 4× Pi 5 16GB |
| Constraints honoured | bit-exact, no overclock, no model change |

```chart
{"type":"bar","xKey":"system","yDomain":[0,16.6],"highlightLast":"#1f9d57","unit":"tok/s",
"series":[{"key":"decode","label":"Decode tok/s","color":"#9aa3b2"}],
"data":[{"system":"b4rtaz #255 (public record)","decode":13.04},{"system":"This work (clean-room)","decode":15.143}],
"caption":"Same model and hardware class (Qwen3-30B-A3B Q40, 4× Raspberry Pi 5), decode metric — a bit-exact +16.1% improvement (n=20)."}
```

## Why this is hard

Edge inference of LLMs on consumer single-board computers is increasingly studied as a privacy-preserving, low-cost alternative to the cloud. The Raspberry Pi 5 is the most widely deployed ARM SBC with enough RAM (16 GB) to host quantised models, and its Gigabit Ethernet lets small clusters be formed. But the Pi 5 has **no usable accelerator** — the VideoCore VII GPU lacks general compute shaders — so inference runs on the CPU, where decode is **memory-bandwidth-bound**.

We adopt **distributed-llama** v0.16.5 with **twelve source-level changes** developed in this work (eight framework patches plus four bit-exact kernel and op-fusion optimisations), plus persistent runtime kernel tuning.

### Five hard constraints

Every optimisation obeys five rules, adopted as design rules. They rule out most of the lossy speedups in the literature — which is exactly what makes what survives production-deployable, with **zero risk of quality regression**:

- **Bit-exact output** — SHA-256 of the first 100 generated token-ids matches a fixed reference (`seed=42`, `temperature=0`).
- **No kernel rebuild** — stock Pi OS kernel 6.12.75.
- **No model change** — Qwen3-30B-A3B Q40 is fixed.
- **No CPU overclock** — silicon stays at the nominal 2.4 GHz.
- **No quality reduction** — top-k unchanged at 8, no down-quantisation, no expert pruning.

## System design

**Per node:** Broadcom BCM2712 (4× Cortex-A76 @ 2.4 GHz, ARMv8.2-A); 16 GB LPDDR4X @ 4267 MT/s (≈17 GB/s theoretical); NVMe via PCIe Gen 2; integrated Gigabit Ethernet (measured intra-cluster latency 0.226 ms); Debian 13, kernel 6.12.75 aarch64; no usable accelerator. The cluster is one **root** coordinator (also serving the HTTP API) plus three **workers**, full-duplex Gigabit Ethernet, with tensor-parallel synchronisation per transformer layer.

**Software:** distributed-llama v0.16.5 + our patches; model Qwen3-30B-A3B Q40 (30B total, 128 experts, top-k = 8, ~3B active/token); jemalloc 2 via `LD_PRELOAD`; a small Python proxy that sanitises OpenAI-strict client requests.

## Methodology

We follow MLPerf Inference v5.1 conventions: **two warm-up runs (discarded)**, **n = 20 measurement runs**, fixed prompt, `temperature = 0`, single client. Statistics as mean, median, stdev, 95% CI and p50/p90/p99. **Every improvement claim is validated bit-exact** by SHA-256-hashing the generated token-id sequence against a fixed reference, unless stated otherwise.

## The optimisation trajectory

```chart
{"type":"bar","xKey":"stage","yDomain":[0,16.4],"highlightLast":"#1f9d57","unit":"tok/s","height":440,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":13.04,"label":"ceiling 13.04","color":"#d98a00","position":"insideTopLeft"}],
"data":[{"stage":"Llama 3.1 8B (dense)","toks":5.70},{"stage":"Qwen3-30B MoE switch","toks":11.40},{"stage":"max-seq + swap clean","toks":12.71},{"stage":"Stage 8 patches+flags","toks":13.72},{"stage":"Stage 9 TIER-0 sysctls","toks":14.011},{"stage":"Stage 10 SILU·MUL fuse","toks":14.081},{"stage":"Stage 13 true 1-pass","toks":14.27},{"stage":"Stage 14–15 chunk+NIC","toks":14.449},{"stage":"Stage 17 clean-room decode","toks":15.143}],
"caption":"Optimisation trajectory and final clean-room result. Bars 1–8 report sustained throughput; the final green bar is the Stage-17 clean-room #255 replication on the decode metric. Dashed line: the public ceiling."}
```

### The change with the largest effect: dense → MoE

The change with the largest single effect was migrating from **Llama 3.1 8B** (dense, ~5 GB Q40 weights, all parameters active per token) to **Qwen3-30B-A3B** (MoE, 128 experts, top-k = 8, ~3 GB active per token). Bandwidth-bound throughput improved **59%** (7.18 → 11.4 tok/s) with no other change. MoE *is* sparse activation: for top-8-of-128 the activation ratio is 8/128 = 6.25% of expert weights plus shared parameters — and the effective bandwidth multiplier matches the observed speedup almost exactly.

### Eight framework patches

The patches fix critical bugs and unlock optimisation flags. Highlights:

| # | Patch | Why it mattered |
|---|---|---|
| 1 | `NnByte` → `NnUint` for `nBatches` | `uint8_t` overflow: `nbatches=256` silently became 0 (256 mod 256), tripping an embedding-layer assertion. |
| 2 | Force `finish_reason` = `stop`/`length` | Empty `finish_reason` sent strict OpenAI clients into infinite retry loops. |
| 3 | `try/catch` around `json::parse` | Malformed bodies threw uncaught exceptions, abort-trapping the daemon (SIGABRT). |
| 6 | `posix_memalign(64, n)` for pipes | ARM64 default `new[]` is 16 B aligned; cache-line (64 B) alignment is required for vectorised NEON. |
| 7 | TCP `SO_RCVBUF`/`SO_SNDBUF` = 8 MB | Default 208 KiB buffers caused write-blocking under burst sync at end-of-layer. |

### Two counter-intuitive bit-exact changes

Both run against received wisdom on the same code base:

1. **Removing the software prefetch (Stage 12, +1.05%).** The NEON+dotprod matmul inner loop carried two `__builtin_prefetch` calls. Across five swept variants, *removing* both was the only one that helped: the Cortex-A76's hardware prefetcher (stride detection on sequential weight access) outperforms the manual hints, which compete for issue slots in the 4-wide decoder.

2. **A true single-pass SILU·MUL fusion (Stage 13, +1.5%).** The earlier fusion was a two-call wrapper that kept the intermediate resident in memory (3 loads + 2 stores per element). We rewrote it as one NEON loop holding values in registers — 2 loads + 1 store per element — bit-exactness preserved (identical arithmetic; only the intermediate writeback is removed).

![Before (two-pass) vs after (single-pass): the SILU·MUL fusion drops the intermediate store/load between the two loops — 2 loads + 1 store per iteration instead of 3 + 2, bit-exact.](/api/contents/research/0001_19d46624-4ce1-4a02-b01d-db23007f1cc5/images/silu-mul-fusion-en.svg)

### Network chunk-size and runtime NIC tuning (Stages 14–15)

The `writeMany`/`readMany` all-reduce loop capped each `send()`/`recv()` syscall at 4 KB. With the larger TCP buffers from Stage 9 that is 4× more syscalls than needed; widening to 16 KB cut the count fourfold (+1.25%). A runtime NIC bundle (enlarged RX ring, Receive Flow Steering off CPU 0, deferred NAPI), persisted as a `systemd` unit, added +1.99% combined. Both are bit-exact — only NIC scheduling changes, no model arithmetic.

## Tuning the host Linux mattered as much as the code

It is tempting to credit a result like this to the inference engine. On identical 16 GB silicon, **most of the deployable speedup came from re-tuning the host operating system as aggressively as the application** — and every lever below holds the bit-exact constraint, so none costs output quality:

- **`SDRAM_BANKLOW=1`** in the bootloader EEPROM — a firmware memory-interleave change that lifted effective read bandwidth from 8.3 to 12.5 GB/s per node. **+5.9%**, bit-exact, no overclock.
- **`--nthreads 3` instead of 4**, freeing one core, with the Ethernet IRQ and Receive Packet Steering pinned to that freed core (CPU 3), plus **jemalloc** (tuned arena/tcache). Together **+10.3%**.
- **`performance` CPU governor** at 2.4 GHz, and **`mlock`-ed weights** (~4.45 GB locked/worker) so the model never pages out.
- **Network / VM stack (Stage 9, TIER 0)**: TCP BBR, enlarged `SO_RCVBUF`/`SO_SNDBUF`, **GRO disabled** (it adds 50–200 µs of pure overhead to the 510 kB all-reduce bursts on 1 GbE), `busy_poll`, `vm.swappiness=1`. **+2.12%**.

The whole state is frozen into `systemd` units, so a cold cluster boots straight into the optimised configuration — no manual steps, fully reproducible. The point is general: on a memory-bandwidth-bound workload **the operating system is not a neutral substrate.** Firmware memory interleaving, core/IRQ placement, the allocator, page residency and the network stack each move the needle by single-digit percentages that compound — and on the same hardware they add up to *more than the source-level code changes do*.

## Results: throughput and stability

Run-to-run variance is low — a coefficient of variation of **0.52%**. The distribution below is the n = 20 sample at an intermediate Stage-6 snapshot (mean **12.708 tok/s**); the final configuration reaches **14.449 tok/s** (best run 14.557, 95% CI ±0.038), with the later stages raising throughput without changing this latency behaviour.

```chart
{"type":"line","xKey":"run","yDomain":[12.4,13.0],"unit":"tok/s","height":340,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":12.708,"label":"mean 12.708","color":"#1f9d57"}],
"data":[{"run":1,"toks":12.85},{"run":2,"toks":12.81},{"run":3,"toks":12.74},{"run":4,"toks":12.74},{"run":5,"toks":12.69},{"run":6,"toks":12.67},{"run":7,"toks":12.72},{"run":8,"toks":12.78},{"run":9,"toks":12.62},{"run":10,"toks":12.68},{"run":11,"toks":12.64},{"run":12,"toks":12.77},{"run":13,"toks":12.58},{"run":14,"toks":12.69},{"run":15,"toks":12.64},{"run":16,"toks":12.71},{"run":17,"toks":12.76},{"run":18,"toks":12.65},{"run":19,"toks":12.69},{"run":20,"toks":12.71}],
"caption":"Per-run throughput across 20 measurement runs (Stage 6 snapshot, warm-ups excluded). Coefficient of variation 0.52%."}
```

Time-to-first-token is **557 ms mean** (p50 545 ms, max 638 ms). Throughput scales with response length up to an asymptote — shorter responses are TTFT-dominated:

```chart
{"type":"line","xKey":"max_tokens","logX":true,"yDomain":[11.5,13],"unit":"tok/s","height":320,
"series":[{"key":"toks","label":"sustained tok/s","color":"#F76B1C"}],
"data":[{"max_tokens":50,"toks":11.83},{"max_tokens":100,"toks":12.47},{"max_tokens":200,"toks":12.67},{"max_tokens":400,"toks":12.86},{"max_tokens":800,"toks":12.62}],
"caption":"Sustained throughput vs response length (log x). Converges near 12.86 tok/s for responses ≥ 400 tokens."}
```

**Prefill is the practical limit.** Prefill rate stays above 15 tok/s up to 2K-token prompts, but for 20K-token prompts (large agent system prompts) the projected prefill time exceeds 20 minutes — the upper bound of usability for full agent workloads here.

Under sustained load all nodes sit at 54–56 °C with **zero throttling** (threshold 85 °C). The root carries extra orchestration/serving buffers; workers keep >9 GB of headroom for KV-cache growth.

```chart
{"type":"bar","xKey":"node","stacked":true,"unit":"GB","height":320,"yDomain":[0,16],
"series":[{"key":"used","label":"Used (model + buffers)","color":"#2f6fed"},{"key":"available","label":"Available","color":"#A8C68A"}],
"data":[{"node":"rpi-1005 (root)","used":12,"available":3.0},{"node":"rpi-1006","used":6.5,"available":9.4},{"node":"rpi-1007","used":6.3,"available":9.5},{"node":"rpi-1008","used":6.3,"available":9.5}],
"caption":"Memory utilisation per node during sustained inference."}
```

## Where the time goes: the memory wall

ARM PMU profiling (`perf stat`, 60 s of sustained inference) pins the bottleneck: **49% backend-stalled cycles**, **11.4 of ~17 GB/s** DRAM per node, IPC 1.74. The sub-0.1% TLB and branch-mispredict rates confirm the 16 KB-page setup is already optimal; `objdump` shows 322 `udot`/`sdot` NEON dot-products in the inner loop — already at the ARMv8.2-A limit. The cores sit idle waiting on DRAM, which is why every compute-side experiment returns zero gain.

```chart
{"type":"bar","xKey":"phase","layout":"horizontal","stacked":true,"unit":"%","height":210,"xDomain":[0,100],
"series":[{"key":"matmul","label":"Matmul Q40 (MoE)","color":"#2f6fed"},{"key":"sync","label":"Sync barrier","color":"#d98a00"},{"key":"syscalls","label":"Syscalls (send/recv)","color":"#9aa3b2"},{"key":"other","label":"Orchestration & other","color":"#c7cfdb"}],
"data":[{"phase":"per-token","matmul":56,"sync":17,"syscalls":4.5,"other":22.5}],
"caption":"Per-token wall-clock breakdown. The Q40 matmul dominates; the synchronisation barrier is the only software-addressable slack."}
```

The ceiling is physical. On a log scale, the Pi 5's LPDDR4X is ~16× below an Apple M4 Pro and ~200× below an H100:

```chart
{"type":"bar","xKey":"platform","layout":"horizontal","logX":true,"unit":"GB/s","height":300,
"series":[{"key":"bw","label":"Memory bandwidth (GB/s)","color":"#1b2a4a"}],
"data":[{"platform":"Pi 5 LPDDR4X","bw":17},{"platform":"Mac M4 Pro","bw":273},{"platform":"RTX 3060","bw":360},{"platform":"Mac M3 Ultra","bw":800},{"platform":"H100 SXM5","bw":3350}],
"caption":"Cross-platform memory bandwidth (log scale). This is the physical ceiling the cluster reaches."}
```

## The telemetry finding

During the final clean-room sweep we found two background monitoring agents running on all four nodes (the "idle background workload" of our earlier methodology). On a memory-bandwidth-bound workload these are not free: they periodically walk system memory statistics, consuming DRAM bandwidth the decode phase needs.

| Background | n | Decode (tok/s) | Prefill (tok/s) |
|---|---:|---:|---:|
| Monitoring ON | 10 | 14.397 ± 0.153 | 18.74 |
| **Monitoring OFF (clean-room)** | 20 | **15.143 ± 0.097** | 18.81 |

Stopping the agents raised **decode by +5.18%** (CIs do not overlap) while leaving **prefill unchanged** — a double dissociation: decode is bandwidth-bound, the compute-bound prefill is the negative control. It is also a practical lesson: co-located observability silently taxes memory-bound inference, and a monitored production node under-performs a clean-room benchmark by several percent.

## Stage 16: a WFE/SEV barrier

The inter-step barrier busy-spun on an atomic with an ARM `yield` hint, so three waiting threads continuously re-read a cache line the advancing thread writes — coherency traffic that contends with the thread driving network I/O. We replaced the spin with the ARMv8 event mechanism: waiters issue `wfe` (low-power wait-for-event) and the advancing thread broadcasts `sev`. Signalling-only, so bit-exact by construction; it also lowers power and heat on the waiting cores.

```chart
{"type":"bar","xKey":"build","unit":"tok/s","yDomain":[14.0,14.6],"highlightLast":"#1f9d57","errorKey":"ci","height":320,
"series":[{"key":"mean","label":"mean tok/s","color":"#9aa3b2"}],
"data":[{"build":"yield spin (Stage 15)","mean":14.261,"ci":0.034},{"build":"WFE/SEV (Stage 16)","mean":14.329,"ci":0.042}],
"caption":"Stage 16 WFE/SEV barrier vs yield-spin, same-session cold paired A/B (n=40/arm). +0.48%, bit-exact, Welch t=2.45, p=0.014. Error bars: 95% CI."}
```

## What did not work

In keeping with reproducibility norms, we document every intent-to-treat attempt. A selection of the 26 catalogued dead-ends:

| Configuration | Root cause of failure |
|---|---|
| Llama 3.3 70B Q40 on 4× Pi 5 | 38 GB weights force aggressive swap; 0.15 tok/s under thrashing. |
| EXO framework | Depends on Apple MLX (Metal + Neural Engine + UMA); does not build on ARM Linux. |
| prima.cpp | ZMQ topology discovery hangs >10 min on Pi 5; never bootstraps. |
| llama.cpp + RPC | 25× regression vs single-node (network-overhead-dominated pipeline parallelism). |
| ARM I8MM / SMMLA repack | The Cortex-A76 does **not** implement I8MM (`grep -c i8mm /proc/cpuinfo` = 0); it is an A78+ feature. |
| Transparent hugepages, jumbo frames, NUMA interleave, PGO | Tested and reverted — neutral or harmful on this DRAM-bound workload. |
| Software prefetch (tiered PLDL2KEEP+L1) | −0.47%; the HW prefetcher already wins (see Stage 12). |

The recurring reason for inapplicability is structural: the highest-leverage modern techniques need a newer kernel, a hardware feature the A76 lacks, or a reboot/rebuild we ruled out — which is precisely what makes the surviving bit-exact wins deployable on stock hardware.

## Caveats and limitations

- **The +16.1% headline mixes effects** — our optimised 16 GB cluster vs b4rtaz's vanilla 8 GB cluster (and `--nthreads 3` vs 4). On **identical 16 GB silicon**, vanilla → our build → fully tuned is 13.15 → 13.88 → **15.143** (+15.2%), of which only **+5.6% is attributable to source code**; the rest is the model choice, the firmware flag, runtime configuration and a clean environment. The paper's Table 22 breaks this down lever by lever.
- **"Bit-exact" means SHA-256 vs our own canonical build** (flags include `-ffast-math`), not strict IEEE-754. Energy (J/token) is the one standard edge metric not yet measured. All numbers are for one model, one site, one batch of silicon.

## Conclusion and future work

The headline number matters less than the method behind it. **Most of the speedup came not from the model or from novel kernels, but from treating the operating system as a tunable part of the inference stack** — firmware memory interleaving, core and IRQ placement, the allocator, page residency, and a network stack tuned for the low-latency, bursty all-reduce traffic between nodes. None of it touched the hardware, the clock, or the model's output quality. That is the transferable lesson: **this class of OS- and network-level optimisation applies to any distributed CPU inference system, not only distributed-llama.**

We have reached the memory wall the Pi 5 presents — the bytes moved per token are fixed by the model and its quantisation, and no further bit-exact reduction of that traffic is available. The remaining software lever, Async Tensor Parallelism, overlaps communication with computation rather than moving fewer bytes.

**What's next.** We plan to release a **Linux image tuned for distributed inference** — an OS pre-configured for low inter-node latency on dllama-style clusters — so these gains are available out of the box, **without changing hardware, overclocking, or lowering model quality.** We also plan to run the same OS- and network-level pass on **other open-source models**, to test how far the approach generalises beyond Qwen3-MoE.

## Resources

- **Paper, code & raw data:** [github.com/hellomatik-org/distributed-llama](https://github.com/hellomatik-org/distributed-llama/tree/kernel-opt-t02) (branch `kernel-opt-t02`, `paper/`).
- **The public reference we compare against:** [b4rtaz/distributed-llama discussion #255](https://github.com/b4rtaz/distributed-llama/discussions/255).
