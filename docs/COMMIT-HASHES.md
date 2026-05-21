# Stage → Commit Mapping (pi5-cluster branch)

Reproducibility reference for the technical paper `paper/main_en.tex`.
All commits live on the `pi5-cluster` branch of `danielcorrea-hellomatik/distributed-llama` (this fork).

## Model artefacts (verified SHA-256)

```
8a116523251c62b9ee2f230f9ab1dc68849095ba7d6d70af71943df937c3cc5f
  /home/rpi/distributed-llama/models/qwen3_30b_a3b_q40/dllama_model_qwen3_30b_a3b_q40.m

6076bc3f06adb200bfadad6bbafafdbb78dbd87a2f608578144fd6674b3237d7
  /home/rpi/distributed-llama/models/qwen3_30b_a3b_q40/dllama_tokenizer_qwen3_30b_a3b_q40.t
```

Computed on rpi-1005, 2026-05-18, after final Stage 15 deploy.
Identical hash expected on all 4 nodes (model is mmap'd from local NVMe per node).

## Throughput evolution → commit hash → date → mean tok/s (n=20, paired)

| Stage | Description                                              | Commit    | tok/s mean | 95% CI ± |
|-------|----------------------------------------------------------|-----------|------------|----------|
| 1–6   | Base patches + jemalloc + MoE switch + max-seq-len 32K   | (b598a92) | 12.708     | 0.029    |
| 7     | F0 — `llamafile_sgemm` gate removed for single-token     | 92a20e2   | 13.724     | 0.042    |
| 7b    | F2 — Cache-line padded atomics + relaxed mo              | 79b44ad   | 13.752     | 0.044    |
| 8     | F1 — Persistent worker pool                              | 95045b1   | 13.720     | 0.050    |
| 9     | TIER 0 — GRO off, rmem/wmem 8MB, vm.swappiness=1         | d50d30c   | 14.011     | 0.116    |
| 10    | OP_SILU_MUL bit-exact fusion (2-pass wrapper)            | 106eab3   | 14.081     | 0.055    |
| 11    | Phase B foundation + Round 3/4 sysctls + EEVDF 4 ms      | 850d1e1   | 14.046     | 0.049    |
| 12    | Remove SW prefetch in matmul_Q80_Q40_F32                 | 64ef787   | 13.997     | 0.040    |
| 13    | TRUE single-pass silu_mul_F32 fusion                     | 1af175c   | 14.270     | (R1+R2)  |
| 14    | MAX_CHUNK_SIZE 4 KB → 16 KB                              | f8ed9d2   | **14.449** | 0.038    |
| 15    | Runtime tweaks persisted (RX ring + RFS + NAPI defer)    | edbc4ce   | 14.449     | 0.038    |

**Best individual run measured**: 14.557 tok/s (Stage 15, n=20 paired bench).

## Reproduction protocol

1. Clone fork at the commit of interest:
   ```bash
   git clone https://github.com/danielcorrea-hellomatik/distributed-llama
   cd distributed-llama
   git checkout <commit-hash>
   ```
2. Build on each Pi 5 (4 nodes, ARM Cortex-A76, 16 GB LPDDR4X):
   ```bash
   make clean && make dllama dllama-api -j4
   ```
3. Deploy systemd units from `deploy/systemd/` and start in order:
   `dllama-worker` on workers, then `dllama-api` + `dllama-proxy` on root.
4. Run n=20 paired benchmark:
   ```bash
   python3 deploy/scripts/benchmark.py
   ```
5. Compare against the row in the table above.

For Stages 14–15, also apply runtime kernel tweaks via the new systemd unit
in `deploy/systemd/dllama-runtime-tweaks.service` (see file for install
instructions). These persist across reboots and are required to reproduce
the 14.449 tok/s headline.

## Bit-exact reference (golden output)

Captured on the running cluster with a fixed deterministic request and
confirmed **identical across the unoptimised (yield-spin barrier) and
optimised (WFE/SEV barrier) builds** — i.e. the optimisations do not alter
model output. Verified reproducibly on 2026-05-21 (≥4 independent runs).

```
Prompt      : "List the first 12 prime numbers and then explain what a prime number is in one sentence."
Params      : max_tokens=160, temperature=0, seed=42
SHA-256 of generated text:
  bdbcaec68c56dd4f5cf07f0dc8e60c8d17209fd14e998d2a3c437144f2328645
```

Reproduce / verify in one command (run on the root node, or set ENDPOINT):

```bash
python3 deploy/scripts/verify_bitexact.py
# -> "result : PASS (bit-exact)" and exit code 0
```

Bit-exact validation procedure documented in `paper/main_en.tex` §4.3
(subsection "Bit-exact output validation").

## Session 2026-05-21 — crash fix + WFE/SEV barrier (additive, does not alter the table above)

Two changes landed on top of Stage 15, both **bit-exact** (golden SHA above
unchanged) and measured with a clean paired A/B (cold cluster between arms):

| Change | What | tok/s (n) | vs yield | Significance | Errors |
|--------|------|-----------|----------|--------------|--------|
| `dllama-api` accept-loop `catch(std::exception&)` | stops a client disconnect from aborting the whole API (was SIGABRT) | n/a (stability) | — | 36 bad conns caught, 0 restarts | — |
| WFE/SEV inter-step barrier (replaces `yield` spin) | cores wait on event instead of busy-spinning | 14.329 (n=40) | yield 14.261 (n=40); **+0.48%** | Welch t=2.45, p≈0.014 | 0 in 80×250-tok runs |

Raw per-run CSV data for both arms is in `data/bench/`. Reproduce with
`deploy/scripts/reproduce.sh`.

## Benchmark script versions

- `deploy/scripts/benchmark.py` — n=20 runs after 2 warmup, fixed prompt,
  reports mean/median/stdev/p50/p90/p99/CI95. Source of all tok/s
  measurements in the paper.
- `deploy/scripts/academic_bench.py` — used for Stage 9 onward (post-publication).
- `deploy/scripts/bench_n50.py` — extended n=50 variant for higher
  statistical power, used during Stage 11 Round 3/4 investigation.

Hardware state at measurement:
- All 4 nodes thermally stable 64–74 °C (active cooler ON, throttled=0x0).
- TCP buffers per `/etc/sysctl.d/99-dllama*.conf`.
- Runtime tweaks per `dllama-runtime-tweaks.service` (Stage 15+).
- Cluster cold-started ≥ 60 s before bench, 2 warmup runs discarded.
