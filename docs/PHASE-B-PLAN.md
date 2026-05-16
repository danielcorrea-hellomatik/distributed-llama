# Phase B — Tile-Level Compute/Communication Overlap

> Status: in progress (started 2026-05-16)
> Branch: `phase-b-overlap-v2`
> Risk level: HIGH — touches the same code that broke the cluster 3 times before.

## 1. Why we are doing this

The cluster sits at 13.77 tok/s after F0+F2+F1. Profiling shows the
remaining time budget is:

```
Compute (matmul Q80 + tinyBLAS + F32):  ~49 %
Network barrier busy-spin:              ~25 %     <-- THIS IS THE TARGET
Syscalls / orchestration / misc:        ~26 %
```

Compute is at the hardware ceiling for Cortex-A76 + dotprod. The
busy-spin time is the network round-trip wait for the per-layer
all-gather. Phase B aims to hide most of that 25 % behind compute
that is happening anyway.

## 2. What "overlap" means here

### 2a. The current pattern — serialised per layer

```
Layer N timeline (one node, ~50 ms total)

T=0          15ms         25ms         40ms        50ms
│            │            │            │           │
├──ATT──────►│            │            │           │
             ├──SYNC1────►│            │           │
             │            ├──MoE──────►│           │
             │            │            ├──SYNC2───►│
             │            │            │           │
             ↑                         ↑
     attention compute            ffn compute
     all 4 cores busy             all 4 cores busy

While SYNC1 is in flight, all 4 cores busy-spin doing nothing useful.
While SYNC2 is in flight, same — all 4 cores idle on the barrier.
That is the 25 % wasted CPU time we see in perf.
```

### 2b. The naive "overlap layers" idea — does NOT work

You might think: while SYNC2 of layer N is flying, start ATT of layer
N+1. Problem: ATT(N+1) needs the FULL output of layer N (which is what
SYNC2 is gathering). It cannot start until SYNC2 has completed.

### 2c. The trick that works — tile-level overlap inside one layer

Don't think in layers. Think in TILES of the matmul output. Each layer's
matmul produces, say, K=4 chunks of the output row dimension. Once
tile T is computed, send it on the wire IMMEDIATELY, while the matmul
continues on tile T+1. The peers receive tile T before our matmul
finishes tile T+4.

```
Tile-overlap timeline (one matmul + sync, ~13 ms total instead of ~22 ms)

t=0    2.5    5.0    7.5    10.0   12.5
│      │      │      │      │      │
├──t1──►                                  matmul lane (cores 0-3)
│      ├──t2──►
│      │      ├──t3──►
│      │      │      ├──t4──►
│      │      │      │
│      ┌─send──►                          network lane (drain thread)
│      │      ┌─send──►
│      │      │      ┌─send──►
│      │      │      │      ┌─send──►
                                   ↑
                          only this final tile
                          is on the critical path
```

Effective overlap:
- ~10 ms of matmul becomes mostly free of "sync wait"
- ~3 ms remain for the tail tile to finish flying
- Per-layer saving: 22 ms → ~13 ms = ~40 % on the matmul+sync pair

End-to-end estimate: the matmul+sync pair is ~30 % of token time.
Saving 40 % of that = +12 % token throughput. Realistic gain target:
**+5-12 % end-to-end**, sustained, on top of the current 13.77 tok/s.

## 3. The signalling primitive (FlashOverlap on CPU)

FlashOverlap (arXiv 2504.19519) introduces a kernel-level
"tile ready" signal for GPU. On CPU the equivalent is just
`std::atomic<uint32_t>` with release/acquire semantics, plus a
dedicated network thread (or opportunistic drain by any thread that
hits a wait).

Per-tile state machine:

```
State 0: not yet computed
State 1: matmul finished writing this tile (RELEASE)
State 2: network thread has handed it to send() (ACQUIRE on send side)
State 3: peer has fully received and ACKed (ACQUIRE on recv side)

Atomic per tile:
  alignas(64) std::atomic<uint32_t> tile_ready[K_TILES_PER_LAYER];
```

Producer (matmul):
```cpp
// Compute output[tile_start..tile_end]
// then:
tile_ready[t].store(1, std::memory_order_release);
```

Consumer (network drain thread):
```cpp
while (true) {
    uint32_t s = tile_ready[t].load(std::memory_order_acquire);
    if (s >= 1) {
        send_tile(t);
        tile_ready[t].store(2, std::memory_order_release);
        break;
    }
    cpu_relax();
}
```

The acquire on the send side guarantees the network thread sees
all the byte writes that the matmul did BEFORE the release. This is
the formal guarantee that prevents tearing.

## 4. The 7 sub-phases (plan)

Each sub-phase is independently testable and revertible.

| # | Step | LOC | Risk | Verification |
|---|------|-----|------|--------------|
| **B1** | Add `clockUs()` instrumentation around syncNodeSlices, syncWithRoot, matmul step. Print per-token wall-clock breakdown to a log file. | ~50 | nil | run vs current baseline 13.77, verify zero perf regression and that the printed breakdown matches our earlier perf estimates (49 % matmul, 25 % sync) |
| **B2** | Double-buffer the per-pipe sync buffer in NnNetExecution. Allocate `pipes[layer & 1]` ping-pong so layer N can read while layer N+1 writes. | ~60 | low (more RAM only) | same tok/s, RSS goes up by ~10 MB |
| **B3** | Define `TileSignal` struct + minimal API (`signal_ready`, `wait_ready`, `reset`). Cache-line padded. Test in unit-test mode locally. | ~80 | nil | compile + 1-thread localhost test of the primitive itself |
| **B4** | Modify `matmul_Q80_Q40_F32` to emit tile-level `signal_ready` for K=4 row blocks. SEMANTICS UNCHANGED — the signal is informational only, the function still produces the full output in the same memory layout. | ~80 | low | bit-exact greedy generation vs baseline at `--seed 42`, n=100 tokens |
| **B5** | Add network drain thread (or opportunistic drain hook). On the SEND side, this thread polls tile_ready and issues async sends. On the RECV side, drains incoming tiles into the correct pipe buffer using a per-peer tile counter. | ~150 | **HIGH** | 2-node localhost test (root + 1 worker) with a small model; numerical comparison + 5-minute stability test |
| **B6** | Wire B5 into the actual `syncNodeSlices`: when overlap is enabled, the sync function returns immediately after posting the sends; the next layer's first op waits on `tile_ready` of incoming tiles before consuming them. | ~120 | **HIGHEST** | full 4-node 2-minute soak, then full 60-minute soak at 14 tok/s |
| **B7** | Tuning: tile count K (try 2, 4, 8), send buffer sizing, timeout watchdog values. Final benchmark n=20 vs the 13.77 baseline. | ~30 | low | n=20 paired t-test, accept only if mean gain > 2x stdev |

Total ~570 LOC across 4 files. Cluster downtime per sub-phase is
~3 minutes (rebuild + restart). Each sub-phase is reverted by
`git reset --hard pi5-cluster` and rebuild on all 4 Pis.

## 5. Risks (from external research, 12 categories ranked)

Ranked by likelihood of biting us in the first week of integration.
For each: trigger, manifestation, detection, mitigation.

### 5.1. HIGHEST — Out-of-order tile arrival (wire format has no header)

```
Sender:                     Receiver:
  send(tile_0, sk_A)          recv from sk_A  → ???  (no idea which tile)
  send(tile_1, sk_A)          recv from sk_A  → ???
  send(tile_2, sk_A)
  send(tile_3, sk_A)
```

The current dllama protocol writes raw payload bytes with **no header,
no length prefix, no sequence number**. Receiver knows the byte count
only because both sides agree on the static `NnSyncConfig`. With
tile-overlap, that contract breaks the moment any tile is reordered.

- Trigger: `writeMany` across 3 sockets has no defined cross-socket
  ordering; even within one socket, kernel coalescing under congestion
  can reorder calls. TCP guarantees byte order *within a socket*, not
  *across multiple writes from different threads*.
- Manifests as: silent garbage tokens. No crash, just nonsense logits.
- Mitigation (NON-OPTIONAL): per-tile header
  `{sync_id:u16, tile_idx:u8, n_tiles:u8, payload_len:u32}` = 8 bytes.
  Cost: 8 B × 4 tiles × 98 syncs = 3 KB/token, negligible vs the
  ~6 MB/token payload.

### 5.2. HIGH — Producer-consumer race on the same buffer

ARMv8 / Cortex-A76 is **weakly ordered**: without explicit barriers,
the network thread can observe `tile_ready[T] = 1` BEFORE the matmul's
last `vstX` of that tile range has actually retired from the store
buffer to L1.

- Manifests as: intermittent garbage tokens, worse under load,
  irreproducible. Classic weak-memory race.
- Mitigation: use `__atomic_store_n(&tile_ready[T], 1, __ATOMIC_RELEASE)`
  on the producer and `__ATOMIC_ACQUIRE` on the consumer. On A76 these
  lower to `STLR`/`LDAR` instructions — essentially free.
- **Do NOT use `volatile`.** It does not provide the cross-thread
  ordering we need.

### 5.3. HIGH — Q80 block boundary alignment

Q80 layout is `struct { float16 scale; int8 qs[32]; }` = 34 bytes per
block. If a tile boundary cuts through a Q80 block, the receiver
dequantises that block with the **wrong scale** (from the previous tile)
and the right quants (from the new tile), or vice versa.

- Manifests as: numerical garbage localised to specific tokens at
  particular output positions. Hard to spot in aggregate stats.
- Mitigation: enforce `tile_size_elements % 32 == 0` and
  `tile_size_bytes = (tile_size_elements / 32) * 34` for Q80 tiles.
  Add a compile-time `static_assert` keyed off `Q80_BLOCK_SIZE`.
  For output dims not divisible by 128 (4 tiles × 32 blocks), pad to
  the next multiple OR fall back to fewer tiles for that op.

### 5.4. HIGH — Inter-layer dependency violation

If we mark layer N "done" as soon as the last tile's `send()` returns
(rather than when all peers have received and acked), layer N+1's
first op consumes stale or zero data.

```
node A sends tile 3 last  ─►  recv buffer not yet full at node B
node B starts layer N+1                         ✗ reads zeros
```

- Manifests as: first token fine, then exponential drift / garbage.
  Or deadlock at the next barrier.
- Mitigation: per-layer barrier on `tile_arrived[peer][tile]` bitmap.
  4 peers × 4 tiles = 16 bits; check `popcount == 16` before exiting.
  **Overlap is intra-layer only in v1**; cross-layer overlap is a
  separate, much harder project.

### 5.5. MEDIUM-HIGH — Drain inside compute busy-spin

If the compute thread opportunistically drains the network during its
spin, `recv()` (even non-blocking) can take a kernel mutex while
another thread also wants to memcpy from the recv buffer.

- Manifests as: throughput collapse, no correctness bug. tok/s drops
  below baseline.
- Mitigation: dedicated network thread pinned to a separate core.
  Pi 5 has 4 cores; **budget = 3 compute + 1 network**. Compute parallelism
  drops 25 %, but bandwidth-bound ops more than recover via overlap.
  Validate empirically — if not, kill the project at this point.

### 5.6. MEDIUM — TCP SO_SNDBUF backpressure

Per-link load: ~52 MB/s × 3 peers = 156 MB/s aggregate. Gigabit
Ethernet ceiling is ~118 MB/s. We will fill the 8 MiB `SO_SNDBUF`
within ~70 ms of sustained sending. `send()` returns EAGAIN; current
`writeMany` loops, blocking compute.

- Manifests as: periodic 50–100 ms stalls visible as token-time
  variance, not as failure.
- Mitigation: tile-emit must use non-blocking `send()` and queue
  locally on EAGAIN; the network thread retries on `EPOLLOUT`.
  Do not raise `SO_SNDBUF` blindly — you only shift the stall.

### 5.7. MEDIUM — Signaling-array cache thrash

1 568 flag operations per token (4 peers × 4 tiles × 98 syncs).
Naively cache-line padded: 1 568 × 64 B = 100 KB, fits in L2 (512 KB)
but evicts useful data.

- Manifests as: 5–10 % regression with no obvious culprit.
- Mitigation: pack flags into a `std::atomic<uint64_t>` bitmap per
  sync op (16 bits used, 48 free). 1 cache line per op, 98 lines per
  token = 6 KB working set.

### 5.8. MEDIUM — Thread-pool sizing vs F1

Current F1 has 4 persistent compute workers. Adding a 5th oversubscribes.
Demoting one to network reduces compute by 25 %.

- Mitigation: empirically validate 3 compute + 1 network configuration
  at the B5 milestone. If compute regression exceeds the overlap gain,
  abort Phase B.

### 5.9. LOW for inference — Bit-exactness drift from reordering

F32 accumulation order changes between runs. Different runs → different
last-bit logits. For greedy decode, ~1 token in 10k may differ.

- Mitigation: do not care for inference. Add a `--deterministic` flag
  (fixed tile-index reduction order on receive) for regression tests
  only.

### 5.10. LOW frequency / HIGH severity — Peer crash mid-tile

If peer P dies after tile 2/4, the `tile_arrived` bitmap never reaches
the expected popcount.

- Mitigation: per-sync-op watchdog. 14 tok/s × 98 syncs = ~140 syncs/s,
  ~7 ms budget per sync; set watchdog at 50 ms. On timeout: abort
  token, mark peer dead, fail the inference cleanly.

### 5.11. LOW — Truncation read of zero-initialised tail

Matmul kernel reads tile T memory before full arrival because of bug
2 or bug 1 variant.

- Mitigation: covered by 5.1 (header `payload_len`) + 5.2
  (release/acquire). Sentinel-fill recv buffer with `0xDEADBEEF`
  between tokens; detect sentinel reads in dequant.

### 5.12. LOW — Q80 reduction order changes existing tested tokens

Today's all-gather likely does fixed-order Q80 dequant→sum→requant.
Tile-overlap changes the order. Block-level requant introduces rounding.

- Mitigation: gate tile-overlap behind a feature flag. Require
  sender-tile-index-ordered reduction for "compat mode"; accept drift
  in "fast mode" and re-baseline goldens.

## 5b. Day-1 mandatory mitigations (from the risk analysis)

Before B5 lands, the following MUST be designed in:

1. **8-byte tile header** with `{sync_id, tile_idx, n_tiles, payload_len}`.
2. **Release/acquire** on every `tile_ready` flag, no `volatile`.
3. **`tile_size % 32 == 0`** assert + Q80 stride alignment.
4. **Per-layer barrier** on `popcount(tile_arrived_bitmap) == expected`.
5. **50 ms per-sync watchdog** with clean abort path.

Anything else is optimisation; these 5 are correctness.

## 6. Rollback strategy

Every sub-phase is a separate commit on `phase-b-overlap-v2`. To
revert sub-phase N: `git revert <commit>` and rebuild. To revert
ALL of Phase B at any time: switch the 4 Pis back to `pi5-cluster`
and rebuild.

We will not merge `phase-b-overlap-v2` into `pi5-cluster` until ALL
of B1-B7 are stable for at least 60 minutes of continuous serving
without a regression in either:
- tok/s mean (must be >= baseline 13.77)
- numerical drift (logit diff against `--seed 42` baseline within
  ULP tolerance on F32, within Q80 block-quant noise on Q80 paths)

## 6b. Test plan (from risk analysis)

Paired benchmarks (n=20, baseline vs overlap, same seed):
- **B1 tok/s mean** — catches drain-collapse, SNDBUF stalls, cache thrash, pool sizing.
- **B2 inter-token latency p50 and p99** — catches SNDBUF stalls (show up in p99).
- **B3 per-thread CPU time breakdown** — catches drain-collapse and pool sizing.

Numerical / correctness tests:
- **N1 bit-exact logits, deterministic mode, 1k tokens** — catches reorder, race, Q80 alignment, inter-layer dependency, sentinel-read.
- **N2 last-bit drift histogram, non-deterministic** — quantifies non-associative drift.
- **N3 Q80 boundary fuzz** — random output dims in `{32, 33, 96, 97, 128, 129}` — catches Q80 alignment.
- **N4 tile-header CRC mismatch counter** (dev builds only) — catches reordering.
- **N5 sentinel-fill 0xDEADBEEF detector** — catches truncation read.

Chaos / failure:
- **C1 `kill -9` a peer mid-decode** — catches peer-crash watchdog.
- **C2 `tc qdisc add` 100 ms latency + 1 % loss on one link** — catches stall handling and watchdog.

## 7. Decision points

- **After B1**: if the per-sync wall-clock numbers do NOT show ~12 ms
  per token in syncs, our 25 % busy-spin estimate is wrong and the
  whole Phase B target is wrong. Re-plan before continuing.
- **After B4**: if signal/release adds measurable overhead (>1 %),
  reconsider the design (maybe coarser tiles).
- **After B6**: if cluster doesn't pass the 60-minute soak, do not
  merge regardless of tok/s gain.

## 7b. Implementation blueprints to lift (from external survey)

No one has published an exact CPU + `std::atomic` port of FlashOverlap.
This project is therefore novel applied work. However, the design is a
combination of four open-source codebases, each contributing one piece:

| Source | Contributes | License | URL |
|--------|-------------|---------|-----|
| FlashOverlap | atomic counter signalling pattern (`fetch_add(release)` / `load(acquire)`) | open | https://github.com/infinigence/FlashOverlap (`wait.cuh`, `overlap_impl.cu`) |
| CoCoNet | three-level tiling scheduler (buffer-tile → rank-chunk → channel-chunk); first-tile latency, later tiles bandwidth | MIT | https://github.com/parasailteam/coconet |
| vLLM DBO (PR #23693) | CPU thread-barrier pattern between compute and drain (`UBatchContext` + `dbo_yield`) | Apache-2 | https://github.com/vllm-project/vllm/pull/23693/files |
| Horovod | tensor-fusion buffer coalescing for small tiles over Ethernet/MPI | open | https://github.com/horovod/horovod |

Concrete patterns to copy:

**a) Tile-ready signal** — port of FlashOverlap `wait.cuh`:

```cpp
// Producer (matmul epilogue, after writing tile T's output rows):
tile_counter.fetch_add(1, std::memory_order_release);

// Consumer (network drain thread):
while (tile_counter.load(std::memory_order_acquire) < N) {
#if defined(__aarch64__)
    asm volatile ("yield" ::: "memory");
#endif
}
```

On ARMv8/A76 these compile to `STLR` (release) and `LDAR` (acquire),
which are essentially free.

**b) Tile sizing** — "wave-equal split" from CoCoNet / TokenWeave:

```
tile_rows = ceil(out_rows / (compute_time_per_row / send_time_per_row))
```

So that compute-of-tile-K ≈ send-of-tile-(K-1). For our cluster on
GbE at ~118 MB/s and Pi 5 matmul throughput ~3 GB/s effective on
Q40, the ratio is roughly 25:1 — meaning we want the COMPUTE tile to
be ~25× the byte-size of the SEND tile. With 4 tiles per layer, this
balances naturally for the dominant matmuls.

**c) Coalesce small tiles** — Horovod tensor-fusion buffer threshold
~64 KB on GbE (one TCP window). For our layers with small activations
(< 64 KB) we should NOT tile; only the large MoE-FFN matmuls benefit.

**d) Drain interleaving** — vLLM DBO's cooperative-yield pattern.
On a 4-core Pi 5: dedicate 1 core to socket draining, 3 to matmul.
Validate empirically — if compute regression exceeds overlap gain,
abort Phase B at the B5 milestone.

### Known failure patterns (from the survey)

- **`llama.cpp -sm row`** (tensor-parallel via row split) has been
  documented "slow as molasses" for ~2.5 years (`ikawrakow` issue #254,
  ggml-org issue #13083). Root cause: tile send without scheduling.
  Naïve chunked send REGRESSES. This is why we need the FlashOverlap
  signalling, not just sequential per-tile writes.
- **`llama.cpp` RPC backend** does request/response per op — no
  pipelining. Discussion #9136 documents the Ethernet slowness;
  Jeff Geerling's 25× regression on the 3-Pi cluster came from this.
- **Tile too small** → TCP per-packet overhead (Nagle, segment
  headers) > matmul tile time → regression. Floor tile size at
  ~MTU × N = ~6 KB for GbE.
- **Tile too large** → first-byte latency dominates → effective
  overlap window is small.
- **Same thread doing compute and `recv()`** → matmul stalls on
  network. Needs the DBO yield model.

### Why CoCoNet's three-level scheduling matters for us

```
Level 1: buffer-tile  (the K=4 we discussed)
Level 2: rank-chunk   (per-peer slice of each tile, since N=4 nodes)
Level 3: channel-chunk (TCP-level chunking, MTU-sized fragments)
```

The first buffer-tile pays setup latency (RTT, syscall overhead) —
fine-grained overlap doesn't help here. Subsequent tiles use coarse
pipelining and saturate the GbE link. This split is exactly what
distinguishes CoCoNet's wins from llama.cpp `-sm row`'s losses.

### Bottom line from the survey

The project is novel. The closest open-source ingredients combined:
*FlashOverlap atomic counter on AVX/NEON microkernel epilogue → fuse
into Horovod-style buffer → ship over TCP → DBO-style yielding compute
thread on the receive side*.

The negative results (llama.cpp -sm row) tell us where the cliff is:
naïve "send each chunk as it's ready" without proper scheduling
regresses. So the scheduler is the load-bearing piece.

## 8. References

- [FlashOverlap arXiv 2504.19519](https://arxiv.org/abs/2504.19519) — the signalling abstraction we are porting.
- [ISO compute/comm overlap arXiv 2409.11155](https://arxiv.org/abs/2409.11155) — sequence-level version of the same idea.
- [Synergistic TP+PP arXiv 2510.27257](https://arxiv.org/pdf/2510.27257) — overlap patterns and pitfalls.
- [b4rtaz/distributed-llama issue #58](https://github.com/b4rtaz/distributed-llama/issues/58) — upstream confirms the per-layer barrier wait is real, no roadmap.
- Earlier session profiling: `docs/PROFILING-FINDINGS.md`, `docs/CODE-AUDIT.md`, `docs/RESEARCH-2025.md`.

## 9. Living log

The implementation diary, results of each sub-phase benchmark, and
any failures will be appended to this document as we work through
B1 → B7. Failures count: starting at 0; if we hit 5, we pause and
re-evaluate the design.

## 9. Living log

### 2026-05-16 — B1 instrumentation deployed

Implementation: `nn-network.cpp` wraps `syncNodeSlices`,
`syncWithRoot` and `NnNetworkNodeSynchronizer::sync` with timing on
thread index 0. Atomic counters; periodic stderr print every 50
forwards.

Measured on the live 4-Pi cluster, 5000 forwards:

```
syncNodeSlices:  125.5 us / call  × 98  = 12.3 ms / forward
syncWithRoot:     15.6 us / call  × 1   =  0.0 ms / forward
TOTAL              13.0 ms / forward  (~17.7 % of 73 ms token)
```

n=20 benchmark with instrumentation enabled: **13.720 +/- 0.048**
vs prior baseline 13.77. No regression from instrumentation.

Decision-point passed: sync time per token (~13 ms) matches our
pre-implementation estimate; the 25 % busy-spin observed in `perf`
is roughly 17.7 % sync wall-clock + ~7 % inter-step barrier
spinning. Tile-overlap can therefore hide AT MOST 17.7 % of token
time = upper bound on Phase B gain. Realistic target with wave-
equal split: +5-12 %.

Proceeding to B2 (double-buffer pipes).
