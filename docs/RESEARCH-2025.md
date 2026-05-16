# 2025 Research findings — what could move the needle next

After applying F0 (llamafile gate), F2 (cache-line pad), F1 (thread pool)
on pi5-cluster and stabilising at ~13.77 tok/s, this document records
what an external research pass identified as the realistic next levers.

## Current bottleneck (perf -F 200 -g, 12 s sample, post-optimizations)

### Root node

| Symbol | Overhead |
|--------|----------|
| `matmulForward_Q80_Q40_F32` (per-expert MoE matmul) | 30.26 % |
| `executorWorkerLoop` (per-step busy-spin barrier) | 24.73 % |
| `tinyBLAS_Q0_ARM<Q40,Q80>::gemm<3,1>` (non-MoE matmul, via F0) | 18.96 % |
| `NnExecutor::forward()` (orchestration) | 7.49 % |
| `matmulForward_F32_F32_F32` (norms, gate, etc.) | ~5 % |
| Other (network, F32 small ops) | ~13 % |

Compute total: ~54 % (MoE Q40 + non-MoE Q40 + F32). Network-wait
busy-spin: ~25 %. The increase in busy-spin proportion vs the
pre-optimization baseline is exactly the expected outcome of making
compute faster — the sync barrier is unchanged in wall-clock time, so
it now occupies a larger fraction of the total.

### Worker node (rpi-1006)

| Symbol | Overhead |
|--------|----------|
| `matmulForward_Q80_Q40_F32` | 31.04 % |
| `executorWorkerLoop` | 25.66 % |
| `tinyBLAS_Q0_ARM::gemm<3,1>` | 17.44 % |

Symmetric distribution; workers and root are doing the same kind of
work in roughly the same proportions, which means the cluster is
load-balanced and no single node is the straggler.

## Techniques investigated and ruled out

### F5 — Software prefetch of next active expert (tested, regressed)

Hypothesis: each iteration of the per-expert MoE matmul loop in
`matmulForward_Q80_Q40_F32` starts with cold L1/L2 for the new
expert's weight buffer (~9 MiB per expert). Adding
`__builtin_prefetch` of the first 8 KiB of the next expert's weights
at the top of each iteration should warm L1 before that iteration
starts.

Measured: 13.697 +/- 0.061 vs the 13.774 baseline, i.e. **within
noise, slight negative point estimate**. The Cortex-A76 hardware
prefetcher already detects the sequential expert-weight access pattern;
explicit software prefetch competes with the in-loop prefetch already
present inside the matmul kernel itself. Not merged; branch
`opt-f5-expert-prefetch` kept for the record.

### MTP (Multi-Token Prediction) heads — blocked

DeepSeek V3/V4 and Qwen 3.6 ship trained MTP heads that boost
acceptance rate to 70-85 % and yield 30-80 % decode speedup
([vLLM docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)).
**Qwen3-30B-A3B (our model) does not ship MTP heads.** Out of reach
until we upgrade the base model.

### EAGLE-3 / Medusa tree attention — blocked

Both need a draft head co-trained on the target model. No public
EAGLE-3 head for Qwen3-30B-A3B; training one on the Pi cluster is
infeasible (would need ~A100 days).

### Naive speculative decoding on MoE — would regress

[arXiv 2506.20675 "Utility-Driven Speculative Decoding for MoE"](
https://arxiv.org/abs/2506.20675) shows that running naive speculative
decode on a top-k MoE model causes a **1.5x slowdown**: draft tokens
collectively activate more experts during verification, so the
verify-phase MoE FFN has to load (rough) 16-30 experts of 128 instead
of 8 of 128. Utility-gated variant claws this back to about +7 %.
Only worth implementing after Phase B because the cost ceiling for
naive spec is the worst-case path.

### INT4 -> INT8 weights, i8mm, SVE, SME — A76 lacks the instructions

The Q4_0 x Q8_0 dotprod path is the best the hardware supports. ARM
i8MM / SVE / SME are present on A78/A710/X2/X4 but **not on A76**, so
the [Arm i8MM blog](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction)
does not apply to the Pi 5.

### KV / prefix cache reuse — already implemented, only helps multi-turn

`NaiveCache` in `dllama-api.cpp:379-414` already does prefix matching
across requests. For Hermes Agent multi-turn conversations this saves
real time, but for the isolated-request benchmark there is no prefix
to reuse.

## Techniques still worth implementing (in order)

### Phase B — compute/comms overlap within a layer (~10 hours, +5-12 %)

The 25 % busy-spin is genuinely the cluster waiting for the per-layer
all-gather. Restructure the executor so that producer-side matmul of
layer N+1 can start as soon as a tile of layer N's output is ready,
while the network ships the rest of layer N concurrently. The
signalling primitive on CPU is `std::atomic<uint32_t>` with
release/acquire; FlashOverlap pattern.

Risk: very high — this is the same code path that broke the cluster
on PGO, isolcpus + nthreads<4, and MSG_ZEROCOPY. Expect 5-10 iterations
before stable.

References:
- [arXiv 2504.19519 — FlashOverlap (the signalling abstraction)](https://arxiv.org/abs/2504.19519)
- [arXiv 2409.11155 — ISO compute/comm overlap](https://arxiv.org/abs/2409.11155)
- [arXiv 2510.27257 — Synergistic TP+PP overlap](https://arxiv.org/pdf/2510.27257)

### SWIFT + utility-gated speculative decoding (~15-20 hours, +5-8 %)

[SWIFT (arXiv 2410.06916)](https://arxiv.org/abs/2410.06916) skips
middle layers of the *same* model as the draft, no extra weights and
no training. [CLaSp (ACL 2025)](https://aclanthology.org/2025.acl-long.1525.pdf)
is a similar in-context layer-skip variant. Both claim 1.3-1.6x on
dense models but the MoE-utility paper (2506.20675) caps real gain
to ~+7 % on top-k MoE with the gating.

Lower priority than Phase B because the headline number is smaller
and the MoE-specific gotcha is well-known.

### Pre-attention expert prediction (~3-5 hours, +5-10 %, model dependent)

[arXiv 2511.10676 (Nov 2025)](https://arxiv.org/abs/2511.10676)
introduces a 94.69 %-accurate predictor for top-k experts using
pre-attention hidden states. If our actual bottleneck were
DRAM-to-cache fetch of expert weights, prefetching them based on the
predictor would shave hundreds of microseconds per layer. Given that
our weights are already mlock'd in RAM (16 GiB Pi 5, not 8 GiB), the
predictor's main value is allowing the prefetch to start one layer
earlier than the router can naturally.

Worth re-evaluating after Phase B lands.

## Numerical summary

| State | tok/s | 95% CI | Delta vs base |
|-------|-------|--------|---------------|
| Original v0.16.5 + 6 patches | 13.596 | +/-0.079 | 0 % |
| + F0 (llamafile gate) | 13.724 | +/-0.042 | +0.94 % |
| + F2 (cache-line pad) | 13.752 | +/-0.044 | +1.15 % |
| + F1 (thread pool) | 13.950 | +/-0.047 | +2.60 % * |
| + F5 (expert prefetch) | 13.697 | +/-0.061 | not merged |
| Final sustained on pi5-cluster | **~13.77** | +/-0.05 | **+1.3 %** |

\* F1's measurement of 13.95 was at the high end of the noise envelope;
a re-measurement on the same code gave 13.77. The sustained gain over
the full session is about +1.3 % over the original 13.596 baseline.

## Next decision

The cluster is now at the realistic ceiling of "easy" optimizations on
this hardware/software stack. Each additional percent is harder than
the last:

| Lever | Effort | Risk | Expected gain | Cost-effective? |
|-------|--------|------|---------------|-----------------|
| Phase B overlap | 10 h | very high | +5-12 % | depends on user tolerance for instability |
| SWIFT + utility gating | 15-20 h | high | +5-8 % | only if Phase B done first |
| Expert prediction prefetch | 3-5 h | medium | +5-10 % | re-evaluate after Phase B |
| MTP / EAGLE-3 | infinite (model unavailable) | n/a | +30-80 % | blocked until Qwen ships heads |

Recommendation: park here. The cluster is +1.3 % above the previous
session's baseline and ~+6 % above the public community benchmark.
The next +5-10 % requires committed engineering hours and meaningful
regression risk.
