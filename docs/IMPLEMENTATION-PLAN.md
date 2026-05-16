# Async All-Reduce Refactor — Implementation Plan

This document defines the implementation plan for the next-generation refactor of
`distributed-llama` on our 4 x Raspberry Pi 5 16 GB cluster, targeting Qwen3-30B-A3B
MoE inference.

It is informed by:
- A read of the upstream source at tag v0.16.5 (~2500 LOC across the 7 core files).
- A research pass of 7 related areas (compute/comms overlap papers 2023-2026,
  dllama PRs and forks, cross-pollination from vLLM/DeepSpeed/TRT-LLM/MPI/Gloo,
  community blogs).
- Three failure rounds in this same cluster: PGO, MSG_ZEROCOPY/TCP_QUICKACK,
  isolcpus + taskset + nthreads. All documented in `FAILED-ATTEMPTS.md`.

Current state: 13.8 tok/s sustained, 95 % CI +/- 0.05; +6 % above the public
community ceiling for this hardware (b4rtaz issue #255).

## 1. Headline conclusion

Two of the three originally-considered optimisations are out:

| Phase | Original idea | Verdict | Reason |
|-------|---------------|---------|--------|
| Phase 1 | DeepSpeed-style fused attn+MLP AllReduce | **Dropped** | Qwen3-A3B's MoE gate needs the fully-normed vector; routing on a per-node partial sends each node to different experts. Also incompatible with our `--buffer-float-type q80`. |
| Phase 2 | Rabenseifner recursive halving-doubling | **Replaced** | RHRD is latency-optimal; our regime is bandwidth-bound (1.7 MB per sync at 30 syncs/token). RHRD would be neutral or worse. |

What remains is a single coherent two-phase refactor:

- **Phase A**: chunked pipelined ring all-reduce in the network layer.
- **Phase B**: tile-signalling FlashOverlap-style async, so compute on layer N+1
  starts as soon as layer N's partial sums leave the matmul instead of waiting
  for the full sync to complete.

These are independent. We can ship A first, measure, then build B on top.

## 2. What we learned about dllama's current sync path

(Citations are file:line in upstream v0.16.5.)

Single choke point: `NnNetworkNodeSynchronizer::sync()` at
`src/nn/nn-network.cpp:609-632`. Dispatches per sync type:

| Sync type | What it does | Where it fires (per token) |
|-----------|--------------|-----------------------------|
| `SYNC_WITH_ROOT` | Root broadcasts a whole pipe to all workers | Once, after embedding (`llm.cpp:256`) |
| `SYNC_NODE_SLICES` | Every node sends its slice to every other (full all-gather) | Twice per block: end of att (`llm.cpp:403`) and end of ff/MoE (`llm.cpp:554`) |
| `SYNC_NODE_SLICES_EXCEPT_ROOT` | Workers ship slice to root only | Once, on final logits (`llm.cpp:599`) |

For Qwen3 with `N` layers: `1 + 2N + 1` sync ops per token. With N=48,
that's **98 sync calls per token**. At 13.8 tok/s, the cluster is firing
~1350 syncs/sec.

The wire payload has no header, no length prefix and no opcode tag --
both sides agree on byte counts purely from the static `NnSyncConfig`.
This is simple but means any algorithmic change must keep byte counts
identical between sender and receiver, or both ends must change in lockstep.

Threading model is the other big finding:
- Compute threads (`--nthreads N`) are *recreated every token* in
  `NnExecutor::forward` (`executor.cpp:177-202`).
- Barrier between steps is a busy-spin on `currentStepIndex`
  (`executor.cpp:168-171`). No condvar, no `pthread_barrier_t`, no
  `sched_yield`. 100 % CPU during waits.
- All compute threads enter `sync()` *simultaneously*. There is no
  dedicated I/O thread; compute and comms share the same pool.
- `setTurbo(true)` flips sockets to `O_NONBLOCK` but the user-space
  loop (`writeMany`/`readMany`) immediately retries on `EAGAIN`. It is
  "non-blocking" only from the kernel's view; from the executor's view
  it is fully synchronous.

Effective compute/communications overlap today: **zero**.

## 3. The dependency graph that prevents naive fusion

In Qwen3-MoE the att-sync and ff-sync are not back to back. Between them
sits the entire ff segment (16+ ops including the dominant MoE matmuls).
The ff segment's first op (`block_merge_add2`) consumes `zqPipe` written
by the att-sync (`llm.cpp:407`). And the next block's att opens with a
`block_merge_add` that consumes `zqPipe` from the previous ff-sync.

Net: every sync's output has a direct, immediate consumer. There is no
slack we can exploit by reordering steps. The only way to overlap
compute with comms is to **stream tile-by-tile**: each output tile of
the matmul is signalled ready as soon as it is computed, the network
layer begins shipping that tile while the matmul is still working on
the rest. The consumer op of the next layer waits on per-tile signals
rather than a single sync-done signal.

This is precisely the FlashOverlap pattern (arXiv 2504.19519); on CPU
the signalling primitive is `std::atomic<uint32_t>` with
release/acquire semantics. No new runtime, no kernel changes, no
hardware features needed.

## 4. Phase A -- Chunked pipelined ring all-reduce

### A.1 Why chunked ring

The current `syncNodeSlices` (`network.cpp:568-600`) is two phases per
call: first each node writes its own slice to every peer, then reads
every other peer's slice. With `setTurbo` this overlaps across peers
(`writeMany` and `readMany` do round-robin polling, `network.cpp:450-522`)
-- but each *direction* of each *peer link* is utilised only half the
time. Measured GbE throughput during sync is ~55 MB/s (half line rate),
consistent with half-duplex utilisation per link.

A ring reduce-scatter + ring allgather with K=4 chunks utilises both
directions simultaneously and keeps every link full-duplex busy. The
bandwidth term in the cost model is identical to the current code on
paper; the wall-clock gain comes entirely from full-duplex utilisation,
worth ~1.5x to 2x on link-bound steps.

For small messages (< 64 KiB) ring is suboptimal -- the per-hop
latency dominates. Use a binomial tree reduce + broadcast instead
(`log2(N)` hops = 2 for N=4, vs N-1 = 3 for ring).

### A.2 Algorithm specification (N=4)

For tensor of size S, target chunk size 256 KiB, K = max(N, ceil(S / 256 KiB)):

**Reduce-scatter (chunked ring)**:
```
for step = 0 .. (N-1) + (K-1):
    sChunk = chunk_for_step(step, rank, N, +1)
    rChunk = chunk_for_step(step, rank, N, -1)
    async_write(right_neighbour, buf + sChunk*chunkBytes, chunkBytes)
    async_read (left_neighbour,  scratch,                 chunkBytes)
    wait_all()
    accumulate_in_place(buf + rChunk*chunkBytes, scratch, chunkBytes, dtype)
```

**Allgather (chunked ring)**:
```
for step = 0 .. (N-1) + (K-1):
    same dance, no accumulate; received chunk overwrites buf
```

Total wire bytes per node: 2 * (N-1)/N * S = 1.5 S (identical to current).
Steps: 2 * ((N-1) + (K-1)) = ~14 for S=1.7 MB on N=4 K=7. Each step <
chunkBytes / line_rate ~ 2.3 ms.

### A.3 Code changes

| File | LOC delta | Change |
|------|-----------|--------|
| `src/nn/nn-network.hpp` | +30 | Add `chunkedRingAllReduce()`, `treeReduce()`, `treeBroadcast()` declarations; `HM_TREE_THRESHOLD` constant |
| `src/nn/nn-network.cpp` | +220 -10 | Implement both algorithms; modify `NnNetworkNodeSynchronizer::sync()` to dispatch by size |
| -- | -- | Add `asyncWrite()`/`asyncRead()`/`waitAll()` thin wrappers over existing `setNonBlocking` sockets with `MSG_DONTWAIT` + poll |
| Tests | +150 | Unit test in localhost (2-node and 4-node) verifying bit-equality with current path for representative tensor sizes and dtypes |

Total: ~400 LOC.

### A.4 Bit-exactness considerations

Floating-point add is non-associative; chunked-ring reorders the
accumulation. For Q80 (our default `--buffer-float-type`) the
dequant-add-requant chain is already non-associative; dllama tolerates
the drift. For F32 paths (small syncs like the position pipe) we will
pin the accumulation order within each ring step to keep golden tests
stable.

### A.5 Risk and rollback

- Risk: deadlock if any peer stalls (any all-reduce ring has this
  property). Mitigation: per-step watchdog (500 ms) using the existing
  `clockUs()` infrastructure; on timeout abort the token and fall
  through to the legacy `syncNodeSlices` path.
- Risk: SO_SNDBUF/SO_RCVBUF too small for K chunks in flight. Mitigation:
  bump to `4 * chunkBytes` at socket setup.
- Rollback: feature-flag `--allreduce-algo=ring|legacy`, default `legacy`
  until measured stable.

### A.6 Expected gain

For SYNC_NODE_SLICES at 1.7 MB: current ~21 ms wall-clock -> chunked
ring ~11-13 ms. We have ~95 of these per token. Saving 8 ms x ~30 % of
syncs at non-zero load = ~200 ms/token saved, **~1.3 x end-to-end**.

## 5. Phase B -- Tile-signalling async (FlashOverlap on CPU)

### B.1 Why this and not OpenMPI

Both options were investigated. OpenMPI's `MPI_Iallreduce` requires a
dedicated progress thread on at least one core. Pi 5 has 4 cores, no
SMT; dedicating one to MPI progress drops `--nthreads` from 4 to 3, a
~15 % measured compute regression that erodes most of the gain. It
also introduces a 4-6 MB runtime dependency, a new launcher (`mpirun`)
replacing `dllama worker`, and adds failure modes (`MPI_ERRORS_ARE_FATAL`,
no peer-recovery without ULFM).

Hand-rolled FlashOverlap on the existing TCP layer keeps all 4 cores
on compute, adds zero dependencies, and has a smaller blast radius
(~400 LOC in two files already on the hot path).

### B.2 Data plane

Tile-granular signalling:
- A matmul output is split into K tiles.
- Each tile, once written by the producer thread, atomically increments
  `tile_ready[k]` with `memory_order_release`.
- The network drain function (run opportunistically by any thread that
  is about to wait) reads `tile_ready[k]` with `memory_order_acquire`;
  if 1, it starts the non-blocking send for that tile.
- The consumer matmul on the next layer waits with `acquire` on
  `tile_recv_ready[k] == 2` (received + verified) before reading.

Double-buffered: `sendBuf[layer & 1]`, `recvBuf[layer & 1]`. Two layers
in flight simultaneously; never more. Memory cost: 2 x 1.7 MB x 3 peer
sockets = ~10 MB per node. Trivial.

### B.3 Threading

Default: no dedicated network thread; any compute thread on a `wait`
call drains the sockets opportunistically.
Optional escalation: pin one thread to `epoll_wait` if measured
contention exceeds 5 % CPU.

### B.4 Code changes

| File | LOC delta | Change |
|------|-----------|--------|
| `src/nn/nn-network.hpp` | +60 | `postSend()`/`postRecv()`/`wait()` API; per-socket pending-write state |
| `src/nn/nn-network.cpp` | +200 -30 | Convert `writeAll`/`readAll` to stateful enqueue + drain; opportunistic drain |
| `src/nn/nn-executor.cpp` | +120 -10 | Insert `signal_tile_ready` after each producer matmul tile; `wait_tile` before each consumer op |
| Tests | +200 | Inter-thread atomics, ordering, tile-boundary fuzzing |

Total: ~600 LOC.

### B.5 Risk and rollback

- Risk: subtle race condition causing data corruption that only shows
  up at scale. Mitigation: extensive 2-node localhost tests with
  ThreadSanitizer; A/B compare logits for 100 tokens before deploying
  to 4-node.
- Risk: ordering bug deadlocks the cluster (we have seen this 3 times).
  Mitigation: 30-second wall-clock timeout on every wait; on timeout
  fall through to a synchronous resync; log and continue.
- Rollback: feature-flag `--async-sync=on|off`, default `off`.

### B.6 Expected gain

Sync time after Phase A is ~half of current. Tile-signalling overlap
hides ~60 % of remaining sync behind producer compute. Net additional
gain on top of Phase A: ~10-12 % end-to-end.

Stacked (A + B): from 13.8 -> ~17-19 tok/s realistic, possibly 20 if
both phases hit their upper estimates.

## 6. Testing protocol

Every change ships behind a feature flag, default off. For each phase:

1. Build on rpi-1005 first. Run unit tests (localhost, 1-process, 2-process).
2. Deploy to one worker only. Run paired benchmark (current path vs flag-on),
   n = 20, paired t-test. Require p < 0.05 for a positive claim.
3. Deploy to all 4. Run n = 20 again. Compare to documented baseline.
4. Numerical: generate 100 tokens greedy with `--seed 42`, compare to a
   reference trace captured at v0.16.5. Allow drift on Q80 (block-quant
   non-associative), require exact match on F32 paths.
5. If pass: enable flag in `systemd/dllama-api.service`. Keep `.bak`.
   If fail: leave flag off, log to FAILED-ATTEMPTS.md, move on.

## 7. Discarded ideas (with reasons, so future-us doesn't re-try)

- Phase 1 fused attn+MLP AllReduce: incompatible with Qwen3-MoE gate and Q80.
- Rabenseifner: wrong regime (bandwidth-bound, not latency-bound).
- TokenWeave token-split overlap: needs batch >= 1024; we are batch 1.
- Ladder Residual: needs fine-tuning Qwen3, out of scope.
- ScMoE shortcuts: needs MoE retraining, out of scope.
- HALO: no public code; 3-6 month reimplementation.
- OpenMPI transport: -15 % compute due to lost core dominates +24 % overlap.
- TensorRT-LLM oneshot: CUDA NVSwitch features, not portable.
- llama.cpp + RPC: 25x regression documented (Geerling), wrong architecture.
- Vulkan compute on Pi 5 V3D: missing FP16 subgroup, slower than CPU.
- Speculative decoding on MoE A3B: documented to regress (unsloth, hackmd).

## 8. Upstream contribution path

b4rtaz has stated explicitly (Discussion #261) that there is no roadmap.
Issues #58, #256, #262 flag this exact bottleneck and have no upstream
response.

Once Phase A is stable we will open a PR to b4rtaz with measured benchmarks
on 4 x Pi 5 + 2 x x86. Likely outcome: hold our own fork; possibly merge
on a feature flag. Either way, the work is the work.

## 9. Sequencing

| Order | Item | Estimated effort | Risk |
|-------|------|------------------|------|
| 0 | Set up the test harness, capture v0.16.5 golden traces, baseline n=20 benchmark | 2 h | Low |
| 1 | Phase A in branch, unit tests passing on localhost | 4-6 h | Low |
| 2 | Phase A on rpi-1005 + rpi-1006 only (2-node), benchmark | 1 h | Low |
| 3 | Phase A on all 4 nodes, benchmark, decide | 1 h | Medium |
| 4 | Phase B prototype in branch on top of A, localhost tests | 6-8 h | Medium |
| 5 | Phase B on 2-node, careful A/B with logit verification | 2 h | High |
| 6 | Phase B on all 4, benchmark | 1 h | High |
| 7 | Documentation, upstream PR, paper update | 2 h | Low |

Total: ~20-24 hours of focused work spread over several sessions.
First milestone (Phase A working on 4 nodes): ~8 hours.

## 10. Definition of done

- Phase A: cluster runs 60 minutes uninterrupted at >= 17 tok/s (95 % CI +/- 0.1)
  with `--allreduce-algo=ring` on all 4 nodes.
- Phase B: cluster runs 60 minutes uninterrupted at >= 18 tok/s (95 % CI +/- 0.1)
  with `--async-sync=on` and `--allreduce-algo=ring`.
- No measured numerical regression on Qwen3-30B-A3B beyond Q80 block-quant
  noise (verified by 100-token greedy comparison).
- Documentation: this file updated with measured results, `FAILED-ATTEMPTS.md`
  updated if anything regressed.
- Upstream PR: open and benchmarked, even if not merged.
