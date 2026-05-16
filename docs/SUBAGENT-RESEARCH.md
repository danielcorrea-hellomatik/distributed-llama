# Consolidated research findings

This document summarises the findings of the 10 research subagents we ran during the project. Their full reports are too verbose for the README; here we keep the highest-signal conclusions.

## 1. Distributed-llama flag exploration

- `--net-turbo` already defaults to `true` in v0.16.5 and only sets `O_NONBLOCK`; it does not enable async pipelining.
- Mesh topology is automatic in v0.11+ (no flag).
- `--nbatches` was hard-coded to 32; our patch exposes it but values > 32 do not improve performance in our regime.
- Buffer-float-type `q80` is the best trade-off; `f16` and `f32` regress.

## 2. Pi 5 OS-level tuning

- CPU governor `performance` gives ~10% over `ondemand`. Applied.
- TCP BBR has a positive but small effect (~2-3%). Applied.
- THP (transparent huge pages) does not exist in the Raspberry Pi OS kernel. Not applicable.
- Disabling `apt-daily`, `man-db.timer` and `e2scrub_all.timer` removes background I/O spikes during inference.
- Active cooling is mandatory: throttling at 80 deg C cuts throughput by 30-50%.

## 3. Memory leak audit

- No real leaks found in dllama core code. Smart pointers are used correctly.
- `nlohmann::json::parse` throws uncaught exceptions; our patch wraps in try/catch.
- Streaming response buffer grows via repeated `resize`; our patch reserves 64 KB upfront.
- Pipe buffers are allocated once at construction and reused; not a leak.
- mmap is used for model weights; the kernel page cache handles eviction.

## 4. Pi 5 hardware accelerator survey

- V3D GPU: render only, no compute shaders for LLM. Skip.
- Hailo-8 M.2: vision-class, useless for transformers. Skip.
- ARM Compute Library / BLIS / OpenBLAS: could theoretically replace ggml matmul, but dllama does not have an integration point. Effort vs reward unclear.

## 5. Forks and community optimisations

- The community ceiling for Qwen3-30B-A3B on 4 x Pi 5 8 GB is 13.04 tok/s, documented in the upstream discussion #255.
- Our 13.82 tok/s sits 6% above the community ceiling, on slightly better hardware (16 GB vs 8 GB).
- No public fork has merged meaningful optimisations beyond the upstream.
- Speculative decoding with a 1 B draft model would theoretically give 2-3x in decode, but the prefill bottleneck dominates and the implementation effort is 2-3 weeks.

## 6. Streaming weights, sparse activation

- MoE (Mixture of Experts) is sparse activation at the architectural level: only k of N experts run per token.
- For Qwen3-30B-A3B with top-8 of 128 experts, ~3 B of 30 B parameters are read per token. This translates to a 60% throughput improvement vs dense 8 B on the same hardware (measured: 7 -> 11.4 tok/s).
- Software-level streaming from NVMe would replace 17 GB/s RAM access with 700 MB/s NVMe access -- 24x slower, useless.
- PowerInfer-style sparse activation requires offline neuron-importance training; not applicable in the available time.

## 7. HALO paper deep dive (arXiv:2601.11676)

- Three techniques: semantic-aware predictor (SAP), overlap scheme, PLR-aware scheduler.
- In clean LAN (0% packet loss, our regime), HALO is only 12% faster than TCP-based dllama.
- Speedup increases with packet loss; in 5% PLR lossy networks, HALO achieves 3.41x.
- Source code is not public.
- Estimated re-implementation: 3-6 months including SAP training.

## 8. Similar papers (TokenWeave, prima.cpp, Lagom, EAGLE-3, RServe)

- TokenWeave: 1.29x speedup, but GPU-only.
- prima.cpp: theoretically applicable, but hangs on Pi 5 in our tests.
- Lagom: 1.07-1.33x, code not public.
- EAGLE-3: speculative decoding, 2-6x for batch <= 4, but llama.cpp + RPC regresses on Pi cluster.
- RServe: multimodal-specific, not applicable.

## 9. dllama internals analysis

- Synchronisation barrier per layer at `nn-network.cpp:609` (`NnNetworkNodeSynchronizer::sync()`).
- `setTurbo(true)` only sets `O_NONBLOCK` on sockets; `readMany` then busy-spins on EAGAIN.
- An async overlap scheme would require ~300 LOC for minimal version, ~1500 LOC for full pipeline.
- Estimated speedup with full async: 1.3-1.5x. Effort: 1-3 weeks engineering.

## 10. Chinese / Asian frameworks

- Alibaba MNN-LLM: claims 8.6x prefill speedup over llama.cpp on ARM CPU, including Qwen3-specific kernels. Migration is 1-2 days but architecturally different.
- Tencent ncnn: ARM-optimised but no LLM cluster support.
- Baidu PaddleLite, Huawei MindSpore Lite: edge-focused but single-device.

## 11. Compiler / binary-level optimisation (the one that worked)

- Default `-march=native` compiles to baseline ARMv8-A; the NEON dotprod ISA is not used.
- Explicit `-mcpu=cortex-a76 -march=armv8.2-a+fp16+dotprod+rcpc` enables 322 `udot/sdot` instructions in the binary.
- Measured impact: +3.8% throughput (13.34 -> 13.82 tok/s).
- LLVM BOLT: ARM64 support is recent, requires perf data, not pursued.
- AutoFDO: requires the kernel PMU; the default Pi OS kernel does not expose it.
- PGO: instrumentation breaks dllama's all-reduce timing. Failed.
- IPA flags (`-fipa-pta -fipa-icf`): applied, negligible measurable effect on its own.

## 12. Kernel-level (io_uring, eBPF, AF_XDP, QUIC)

- io_uring: applicable to socket I/O but no existing LLM inference engine uses it; 400+ LOC port; speedup unclear.
- eBPF sockmap: 10-30% speedup theoretical for TCP bypass. 200-400 LOC port. Not pursued.
- AF_XDP: zero-copy. Pi 5 RP1 driver does not expose XDP. Skip.
- QUIC/UDP custom protocol: 15-30 us per sync if implemented; effort 300-600 LOC.

## What we actually applied (and the cumulative impact)

| Optimisation | Layer       | Cumulative tok/s | Delta  |
| ------------ | ----------- | ---------------- | ------ |
| Baseline (Llama 3.1 8B dense vanilla)            | --        | 5.70  | --     |
| CPU governor + swap NVMe + TCP BBR + jemalloc    | OS        | 6.85  | +20%   |
| 8 source patches in dllama                       | Framework | 7.18  | +5%    |
| Cleanup parasitic processes, mlock model         | OS        | 7.01  | -      |
| Switch to Qwen3-30B-A3B MoE                      | Model     | 11.40 | +63%   |
| max-seq-len 32 K + swap-clean cycle              | OS+Model  | 12.71 | +12%   |
| SO_BUSY_POLL + SO_PRIORITY + SO_INCOMING_CPU     | Network   | 13.34 | +5%    |
| ARM cortex-a76 + dotprod + IPA flags             | Compiler  | 13.82 | +3.8%  |
| **Final**                                        | --        | 13.82 | **+143% vs baseline** |
