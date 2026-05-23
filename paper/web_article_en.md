A 30-billion-parameter Mixture-of-Experts language model (Qwen3-30B-A3B) runs on **four Raspberry Pi 5 boards** — roughly **€500 of CPU-only hardware**, no GPU, no NPU — at **15.143 tok/s decode**: bit-exact and **+16.1% above the highest publicly documented result** for this model and hardware class (13.04 tok/s, b4rtaz #255). This is the full, web-native version of our technical report; the [original PDF and all raw data](https://github.com/hellomatik-org/distributed-llama/tree/kernel-opt-t02/paper) are public.

## Results at a glance

| Metric | Value |
|---|---|
| **Decode throughput** (apples-to-apples vs the public ceiling) | **15.143 tok/s** |
| Improvement over b4rtaz #255 (13.04 tok/s, decode) | **+16.1%** |
| Sustained serving throughput (prefill included) | 14.449 tok/s |
| Time-to-first-token (TTFT) | 557 ms |
| Per-node DRAM bandwidth (sustained / vendor ceiling) | 11.4 / 17 GB/s |
| Hardware / cost | 4× Pi 5 16GB / ~€500 |
| Telemetry-off gain on decode (paired A/B) | +5.18% |
| Constraints honoured | bit-exact, no overclock, no model change |

```chart
{"type":"bar","xKey":"system","yDomain":[0,16.6],"highlightLast":"#1f9d57","unit":"tok/s",
"series":[{"key":"decode","label":"Decode tok/s","color":"#9aa3b2"}],
"data":[{"system":"b4rtaz #255 (public record)","decode":13.04},{"system":"This work (clean-room)","decode":15.143}],
"caption":"Same model and hardware class (Qwen3-30B-A3B Q40, 4× Raspberry Pi 5), decode metric — a bit-exact +16.1% improvement (n=20, telemetry off)."}
```

## Why this is hard

Edge inference of large language models on consumer single-board computers is increasingly studied as a privacy-preserving, low-cost alternative to the cloud. The Raspberry Pi 5 is the most widely deployed ARM SBC with enough RAM (16 GB) to host quantised models, and its Gigabit Ethernet lets small clusters be formed. But the Pi 5 has **no usable accelerator** — the VideoCore VII GPU lacks general compute shaders — so inference runs on the CPU, where execution is **memory-bandwidth-bound**.

We adopt **distributed-llama** v0.16.5 with **twelve source-level changes** developed in this work (eight framework patches plus four bit-exact kernel and op-fusion optimisations), plus persistent runtime kernel tuning. The trajectory runs from a 5.70 tok/s Llama-3.1-8B dense baseline, through an 11.40 tok/s Qwen3-MoE baseline, to the 15.143 tok/s decode result.

### Five hard constraints

Every optimisation in this work obeys five rules, discovered progressively and adopted as design rules. They rule out most speedups in the recent literature — and that is exactly what makes the surviving optimisations production-deployable with **zero risk of quality regression**:

- **Bit-exact output** — SHA-256 of the first 100 generated token-ids matches a fixed reference (`seed=42`, `temperature=0`).
- **No kernel rebuild** — stock Pi OS kernel 6.12.75.
- **No model change** — Qwen3-30B-A3B Q40 is fixed.
- **No CPU overclock** — silicon stays at the nominal 2.4 GHz.
- **No quality reduction** — top-k unchanged at 8, no down-quantisation, no expert pruning.

## System design

### Hardware (per node)

- **SoC:** Broadcom BCM2712 (4× Arm Cortex-A76 @ 2.4 GHz, ARMv8.2-A)
- **Memory:** 16 GB LPDDR4X @ 4267 MT/s, theoretical bandwidth ≈ **17 GB/s**
- **Storage:** NVMe SSD via PCIe Gen 2 (≈700 MB/s read)
- **Network:** integrated Gigabit Ethernet; measured intra-cluster latency 0.226 ms
- **OS:** Debian 13 *trixie*, kernel 6.12.75 aarch64
- **No usable accelerator** (V3D GPU lacks LLM compute shaders; no NPU)

The cluster is one **root** coordinator (`rpi-1005`, also serving the HTTP API) plus three **workers**, connected over full-duplex Gigabit Ethernet, with tensor-parallel synchronisation per transformer layer.

### Software stack

- **Inference engine:** distributed-llama v0.16.5 + 8 framework patches + 4 bit-exact kernel/op-fusion optimisations.
- **Model:** Qwen3-30B-A3B Q40 (30B total parameters, 128 experts, top-k = 8, ~3B active per token).
- **Allocator:** jemalloc 2 preloaded via `LD_PRELOAD`.
- **HTTP front-end:** a custom Python proxy sanitising OpenAI-strict client requests.

## Methodology

We follow MLPerf Inference v5.1 conventions. Protocol: **two warm-up runs (discarded)**, **n = 20 measurement runs** for the main configuration, fixed prompt, `temperature = 0` for determinism, single client, idle background workload. Statistics reported as mean, median, standard deviation, 95% confidence interval, and p50/p90/p99 percentiles. **Every improvement claim is validated bit-exact** via SHA-256 hashing of the generated token-id sequence against a fixed reference, unless explicitly marked otherwise.

## The optimisation trajectory

```chart
{"type":"bar","xKey":"stage","yDomain":[0,16.4],"highlightLast":"#1f9d57","unit":"tok/s","height":460,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":13.04,"label":"ceiling 13.04","color":"#d98a00","position":"insideTopLeft"}],
"data":[{"stage":"Llama 3.1 8B (dense)","toks":5.70},{"stage":"Qwen3-30B MoE switch","toks":11.40},{"stage":"max-seq + swap clean","toks":12.71},{"stage":"Stage 8 patches+flags","toks":13.72},{"stage":"Stage 9 TIER-0 sysctls","toks":14.011},{"stage":"Stage 10 SILU·MUL fuse","toks":14.081},{"stage":"Stage 13 true 1-pass","toks":14.27},{"stage":"Stage 14–15 chunk+NIC","toks":14.449},{"stage":"Stage 17 clean-room decode","toks":15.143}],
"caption":"Optimisation trajectory and final clean-room result. Bars 1–8 report sustained throughput; the final green bar is the Stage-17 clean-room #255 replication on the decode metric. Dashed line: the public ceiling."}
```

### The single biggest win: dense → MoE

The most impactful change was migrating from **Llama 3.1 8B** (dense, ~5 GB Q40 weights, all parameters active per token) to **Qwen3-30B-A3B** (MoE, 128 experts, top-k = 8, ~3 GB active weights per token). Bandwidth-bound throughput improved by **59%** (7.18 → 11.4 tok/s) with no other change. MoE *is* a form of sparse activation: for top-8-of-128 the activation ratio is 8/128 = 6.25% of expert weights plus shared parameters — and the effective bandwidth multiplier matches the observed speedup almost exactly.

### Eight framework patches

The patches fix critical bugs and unlock optimisation flags. Highlights:

| # | Patch | Why it mattered |
|---|---|---|
| 1 | `NnByte` → `NnUint` for `nBatches` | `uint8_t` overflow: `nbatches=256` silently became 0 (256 mod 256), tripping an embedding-layer assertion. |
| 2 | Force `finish_reason` = `stop`/`length` | Empty `finish_reason` sent strict OpenAI clients into infinite retry loops. |
| 3 | `try/catch` around `json::parse` | Malformed bodies threw uncaught exceptions, abort-trapping the daemon (SIGABRT). |
| 6 | `posix_memalign(64, n)` for pipes | ARM64 default `new[]` is 16 B aligned; cache-line (64 B) alignment is required for vectorised NEON. |
| 7 | TCP `SO_RCVBUF`/`SO_SNDBUF` = 8 MB | Default 208 KiB buffers caused write-blocking under burst sync at end-of-layer. |

### Kernel and OS tuning (Stage 9, TIER 0)

A structured research round found OS-level network and memory tunings worth a measurable, source-free improvement. GRO (Generic Receive Offload) coalesces packets, adding 50–200 µs of latency; for the 510 kB sync bursts of the all-reduce step on a 1 GbE LAN that is pure overhead, so we turn it off. We raise `rmem_max`/`wmem_max` from the 256 kB default so the TCP windows can grow past the burst size, and drop `vm.swappiness` to 1 to keep `mlock`-ed weights resident. Net effect: **13.720 → 14.011 tok/s** (+2.12%).

This Stage alone accounts for **1.5% of the final 16.1% headline improvement**, and it is a critical insight: **stock Debian 13 is not configured for memory-bandwidth-bound inference**. The default kernel parameters assume general-purpose workloads (web servers, databases, interactive shells) where packet coalescing and swap-readiness are sensible defaults. For edge inference on commodity hardware, these defaults actively harm performance. Customizing the Linux kernel without recompilation — via `sysctl` configuration and runtime NIC tuning persisted as a `systemd` unit — is thus *not* a marginal optimisation: it is **essential infrastructure** for achieving competitive throughput on this class of hardware. This finding applies broadly to any memory-bandwidth-bound distributed workload on ARM SBCs, and it underscores why pre-tuned containerised deployment (a single curated kernel configuration distributed to all nodes) is now table stakes for production edge inference.

### Two counterintuitive bit-exact wins

The two most consequential source-level wins run **counter to received wisdom** on the same code base:

1. **Removing the software prefetch (Stage 12, +1.05%).** The NEON+dotprod matmul inner loop carried two `__builtin_prefetch` calls. We swept five variants and found that *removing* both was the only winner: the Cortex-A76's hardware prefetcher (stride detection on sequential weight access) is more efficient than the manual hints, which compete for issue slots in the 4-wide decoder.

2. **A true single-pass SILU·MUL fusion (Stage 13, +1.5%).** The earlier fusion (Stage 10) was a thin two-call wrapper that kept the intermediate result resident in memory: 3 loads + 2 stores per element. We rewrote it as a single NEON loop that holds the values in registers throughout — 2 loads + 1 store per element, a ~33% reduction in memory ops for this kernel — bit-exactness preserved (the arithmetic sequence is identical; only the intermediate writeback is removed).

```text
Algorithm — Stage 13 single-pass silu_mul_F32 (bit-exact NEON fusion)
Input:  output buffer y[0..n-1] (in-place), multiplier m[0..n-1], thread t of T
Output: y[i] <- silu(y[i]) * m[i]
Invariant: arithmetic (vrecpeq_f32 + one Newton-Raphson + multiply) is
           byte-identical to the prior two-pass code; only the intermediate
           writeback between the two passes is removed.

 1  (start, end) <- split_threads(n, T, t)
 2  for i = start to end-4 step 4 do          // NEON inner loop, 4 lanes
 3      x  <- vld1q_f32(y + i)                 // 1 vector load
 4      mv <- vld1q_f32(m + i)                 // 1 vector load
 5      e  <- expf_neon(-x)
 6      d  <- 1 + e
 7      r  <- vrecpeq_f32(d)                   // reciprocal estimate
 8      r  <- r * (2 - d * r)                  // 1 Newton-Raphson iteration
 9      silu <- x * r
10      out  <- silu * mv
11      vst1q_f32(y + i, out)                  // 1 store; no intermediate write
12  end for
13  for remaining i < end do                   // scalar tail
14      y[i] <- (y[i] / (1 + e^-y[i])) * m[i]
15  end for
```

### Network chunk-size and runtime NIC tuning (Stages 14–15)

The `writeMany`/`readMany` all-reduce loop capped each `send()`/`recv()` syscall at 4 KB. With the 8–32 MB TCP buffers tuned in Stage 9, that generates 4× more syscalls than necessary; widening to 16 KB cut the syscall count fourfold (+1.25%). A runtime NIC bundle (enlarged RX ring, Receive Flow Steering off CPU 0, deferred NAPI), persisted as a `systemd` unit, added a further +1.99% combined. Both are bit-exact by construction — only NIC scheduling changes, no model arithmetic.

## Results: throughput and stability

The cluster is exceptionally stable run-to-run — a coefficient of variation of just **0.52%**. The distribution below is the n = 20 sample at an intermediate Stage-6 snapshot (mean **12.708 tok/s**); the final configuration reaches **14.449 tok/s** (best run 14.557, 95% CI ±0.038), with the later stages raising throughput without changing this latency behaviour.

```chart
{"type":"line","xKey":"run","yDomain":[12.4,13.0],"unit":"tok/s","height":340,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":12.708,"label":"mean 12.708","color":"#1f9d57"}],
"data":[{"run":1,"toks":12.85},{"run":2,"toks":12.81},{"run":3,"toks":12.74},{"run":4,"toks":12.74},{"run":5,"toks":12.69},{"run":6,"toks":12.67},{"run":7,"toks":12.72},{"run":8,"toks":12.78},{"run":9,"toks":12.62},{"run":10,"toks":12.68},{"run":11,"toks":12.64},{"run":12,"toks":12.77},{"run":13,"toks":12.58},{"run":14,"toks":12.69},{"run":15,"toks":12.64},{"run":16,"toks":12.71},{"run":17,"toks":12.76},{"run":18,"toks":12.65},{"run":19,"toks":12.69},{"run":20,"toks":12.71}],
"caption":"Per-run throughput across 20 measurement runs (Stage 6 snapshot, warm-ups excluded). Coefficient of variation 0.52% — a very stable cluster."}
```

Time-to-first-token is **557 ms mean** (p50 545 ms, max 638 ms). Throughput scales with response length up to an asymptote — shorter responses are TTFT-dominated:

```chart
{"type":"line","xKey":"max_tokens","logX":true,"yDomain":[11.5,13],"unit":"tok/s","height":320,
"series":[{"key":"toks","label":"sustained tok/s","color":"#F76B1C"}],
"data":[{"max_tokens":50,"toks":11.83},{"max_tokens":100,"toks":12.47},{"max_tokens":200,"toks":12.67},{"max_tokens":400,"toks":12.86},{"max_tokens":800,"toks":12.62}],
"caption":"Sustained throughput vs response length (log x). Converges near 12.86 tok/s for responses ≥ 400 tokens."}
```

**Prefill is the practical limit.** Prefill rate stays above 15 tok/s up to 2K-token prompts, but for 20K-token prompts (e.g. large agent system prompts) the projected prefill time exceeds 20 minutes — the upper bound of usability for full agent workloads on this configuration.

### Memory and thermals

The root carries extra buffers for orchestration and HTTP serving; workers retain >9 GB of headroom for further KV-cache growth. Under sustained load all nodes sit at 54–56 °C with **zero throttling** (throttle threshold 85 °C).

```chart
{"type":"bar","xKey":"node","stacked":true,"unit":"GB","height":340,"yDomain":[0,16],
"series":[{"key":"used","label":"Used (model + buffers)","color":"#2f6fed"},{"key":"available","label":"Available","color":"#A8C68A"}],
"data":[{"node":"rpi-1005 (root)","used":12,"available":3.0},{"node":"rpi-1006","used":6.5,"available":9.4},{"node":"rpi-1007","used":6.3,"available":9.5},{"node":"rpi-1008","used":6.3,"available":9.5}],
"caption":"Memory utilisation per node during sustained inference."}
```

## Where the time goes: the memory wall

ARM PMU profiling (`perf stat` over 60 s of sustained inference) locates the bottleneck unambiguously: **49% backend-stalled cycles** and **11.4 GB/s sustained DRAM traffic per node**, ~67% of the ~17 GB/s vendor ceiling. The CPUs wait on memory roughly half the time.

```chart
{"type":"bar","xKey":"phase","layout":"horizontal","stacked":true,"unit":"%","height":210,"xDomain":[0,100],
"series":[{"key":"matmul","label":"Matmul Q40 (MoE)","color":"#2f6fed"},{"key":"sync","label":"Sync barrier","color":"#d98a00"},{"key":"syscalls","label":"Syscalls (send/recv)","color":"#9aa3b2"},{"key":"other","label":"Orchestration & other","color":"#c7cfdb"}],
"data":[{"phase":"per-token","matmul":56,"sync":17,"syscalls":4.5,"other":22.5}],
"caption":"Per-token wall-clock breakdown. The Q40 matmul dominates; the synchronisation barrier is the only software-addressable slack."}
```

| PMU metric | Root (rpi-1005) | Worker (rpi-1006) |
|---|---:|---:|
| stalled-cycles-backend (% cycles) | **49.25%** | 47.69% |
| DRAM read bandwidth | 2.74 GB/s | 2.61 GB/s |
| DRAM write bandwidth | 8.66 GB/s | 8.47 GB/s |
| Per-node DRAM total | 11.40 GB/s | 11.08 GB/s |
| IPC | 1.74 | — |
| dTLB / iTLB miss rate | 0.08% / 0.01% | — |

The sub-0.1% TLB and branch-mispredict rates confirm the 16 KB-page setup is already optimal (transparent hugepages would not help). `objdump` shows 322 `udot`/`sdot` NEON dot-product instructions in the inner loop at IPC 1.74 — the kernel is already vectorised to the ARMv8.2-A dot-product limit. **This is why every compute-side experiment returns zero gain: the cores sit idle waiting on DRAM, not starved of issue slots.**

The ceiling is physical. On a log scale, the Pi 5's LPDDR4X is ~16× below an Apple M4 Pro and ~200× below an H100:

```chart
{"type":"bar","xKey":"platform","layout":"horizontal","logX":true,"unit":"GB/s","height":300,
"series":[{"key":"bw","label":"Memory bandwidth (GB/s)","color":"#1b2a4a"}],
"data":[{"platform":"Pi 5 LPDDR4X","bw":17},{"platform":"Mac M4 Pro","bw":273},{"platform":"RTX 3060","bw":360},{"platform":"Mac M3 Ultra","bw":800},{"platform":"H100 SXM5","bw":3350}],
"caption":"Cross-platform memory bandwidth (log scale). This is the physical ceiling the cluster reaches."}
```

## The telemetry finding (a clean confirmation)

During the final clean-room sweep we found two background monitoring agents running on all four nodes (the "idle background workload" of our earlier methodology). On a memory-bandwidth-bound workload these are not free: they periodically walk system memory statistics, consuming DRAM bandwidth the decode phase needs. A paired A/B:

| Background | n | Decode (tok/s) | Prefill (tok/s) | vs 13.04 |
|---|---:|---:|---:|---:|
| Telemetry ON (Alloy + cAdvisor) | 10 | 14.397 ± 0.153 | 18.74 | +10.4% |
| **Telemetry OFF (clean-room)** | 20 | **15.143 ± 0.097** | 18.81 | **+16.1%** |

Stopping the two agents raised decode by **+5.18%** (CIs do not overlap), while **prefill was unchanged** (compute-bound, high arithmetic intensity). It is a practical lesson: co-located observability agents silently tax memory-bound inference, and a monitored production node under-performs a clean-room benchmark by several percent.

## Stage 16: a WFE/SEV barrier

The inter-step barrier busy-spun on an atomic with an ARM `yield` hint, so three waiting threads continuously re-read a cache line the advancing thread writes — coherency traffic that contends with the thread driving network I/O. We replaced the spin with the ARMv8 event mechanism: waiters issue `wfe` (low-power wait-for-event) and the advancing thread broadcasts `sev`. Correctness rests on the sticky ARM event register, with the architected-timer event stream as a periodic-wake safety net. Signalling-only, so bit-exact by construction.

```chart
{"type":"bar","xKey":"build","unit":"tok/s","yDomain":[14.0,14.6],"highlightLast":"#1f9d57","errorKey":"ci","height":320,
"series":[{"key":"mean","label":"mean tok/s","color":"#9aa3b2"}],
"data":[{"build":"yield spin (Stage 15)","mean":14.261,"ci":0.034},{"build":"WFE/SEV (Stage 16)","mean":14.329,"ci":0.042}],
"caption":"Stage 16 WFE/SEV barrier vs yield-spin, same-session cold paired A/B (n=40/arm). +0.48%, bit-exact, Welch t=2.45, p=0.014. Error bars: 95% CI."}
```

The change is retained: bit-exact, free at runtime, never slower, and it reduces the power and heat of the waiting cores.

## What did not work

In keeping with reproducibility norms, we document every intent-to-treat attempt. A selection of the 26 catalogued dead-ends:

| Configuration | Root cause of failure |
|---|---|
| Llama 3.3 70B Q40 on 4× Pi 5 | 38 GB weights force aggressive swap; 0.15 tok/s under thrashing. |
| EXO framework | Depends on Apple MLX (Metal + Neural Engine + UMA); does not build on ARM Linux. |
| prima.cpp | ZMQ topology discovery hangs >10 min on Pi 5; never bootstraps. |
| llama.cpp + RPC | 25× regression vs single-node (network-overhead-dominated pipeline parallelism). |
| ARM I8MM / SMMLA repack | The Cortex-A76 does **not** implement I8MM (`grep -c i8mm /proc/cpuinfo` = 0). I8MM is an A78+ feature. |
| Transparent hugepages, jumbo frames, NUMA interleave, PGO | Tested and reverted — neutral or harmful on this DRAM-bound workload. |
| Software prefetch (tiered PLDL2KEEP+L1) | −0.47%; the HW prefetcher already wins (see Stage 12). |

The recurring reason for inapplicability is structural: the highest-leverage modern techniques require a newer kernel, a hardware feature the A76 lacks, or a reboot/rebuild we ruled out — which is precisely what makes the surviving bit-exact wins deployable on stock hardware.

> **"ARM" is not a hardware category.** Apple Silicon, Cortex-A76 SBCs, and server-class Neoverse cores are three different platforms, with a ~30× bandwidth spread and entirely different feature sets (no I8MM, no FP16 GEMM, no SVE on the A76). Frameworks ostensibly written for "ARM Linux" may implicitly require GPU compute, hardware-specific synchronisation, or ISA extensions — and these distinctions surface only at runtime.

## Cost / performance in context

| Setup | tok/s (8B class) | Cost (approx.) |
|---|---:|---:|
| **4× Pi 5 16GB + dllama MoE (this work, decode)** | **15.143** | ~€500 |
| 1× Mac Mini M4 8GB + MLX | 25 | 599 USD |
| 1× NVIDIA Jetson Orin Nano Super | 21.75 | 249 USD |
| 1× desktop + RTX 3060 12GB | ~40 | ~€700 |

The Pi cluster is not the cheapest tok/s, but it delivers fully on-premise inference with no per-token cost and substantial idle headroom for co-located workloads.

## Honest caveats (threats to validity)

- **The headline mixes effects.** The +16.1% compares our optimised 16 GB cluster against b4rtaz's vanilla 8 GB cluster (and `--nthreads 3` vs 4). It is an *end-to-end system comparison*, not an isolated attribution of the speedup to our code. The cleanly isolated quantities we can defend without confound are the same-hardware A/Bs: the telemetry delta (+5.18%) and the OS/`nthreads` deltas. The vanilla-vs-ours benchmark on the *same* 16 GB silicon is the next measurement.
- **Bit-exact ≠ strict IEEE-754.** "Bit-exact" means SHA-256 equality of generated token-ids under the project's fixed build flags (which include `-ffast-math`); it is equality against our own canonical baseline, not an arbitrary third-party build.
- **Energy not measured.** Joules-per-token is the one standard edge metric we do not report (no wall-plug wattmeter was available); protocol drafted.
- **Single model, single site.** All numbers are for Qwen3-30B-A3B Q40 behind one Gigabit switch with nodes from one batch.

## Conclusion: the memory wall

On the decode metric used by the public ceiling, the cluster reaches **15.143 tok/s** (±0.097, n = 20, telemetry off), **+16.1% above the 13.04 tok/s** publicly documented for Qwen3-30B-A3B Q40 on 4× Pi 5; end-to-end sustained serving (prefill included) is 14.449 tok/s. To our knowledge this is the highest rate at which a 30-billion-parameter Mixture-of-Experts model has been driven on this hardware class — bit-exact, with no overclock and no loss of output quality.

At 49% backend-stalled cycles and ~11.4 of ~17 GB/s realised per node, decode is memory-bandwidth-bound: the bytes moved per token are fixed by the model and its quantisation, and **no further bit-exact reduction of that traffic is available**. The one software lever left — **Async Tensor Parallelism** (~250 LOC on the foundation already shipped in the repository) — overlaps communication with computation rather than moving fewer bytes. The telemetry experiment sharpens the point: even the few percent of DRAM bandwidth a background monitor consumes is directly visible in the decode rate, while compute-bound prefill is untouched.

We have, in short, reached the **memory wall** the Raspberry Pi 5 presents. Pushing past it is no longer a software problem but a silicon one: it requires memory with more bandwidth.

## Resources

- **Code, paper PDF and raw data:** [github.com/hellomatik-org/distributed-llama](https://github.com/hellomatik-org/distributed-llama/tree/kernel-opt-t02) (branch `kernel-opt-t02`, `paper/`).
- **The public record we compare against:** [b4rtaz/distributed-llama discussion #255](https://github.com/b4rtaz/distributed-llama/discussions/255).
- **Upstream engine:** [b4rtaz/distributed-llama](https://github.com/b4rtaz/distributed-llama).
