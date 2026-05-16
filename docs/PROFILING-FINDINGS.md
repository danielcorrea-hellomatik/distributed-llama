# Profiling findings on 4 x Raspberry Pi 5 cluster

This document records the actual measurements taken on the running
production cluster (Qwen3-30B-A3B Q40 at 13.6 tok/s), the bottleneck
they identify, and the design implications for the next round of
optimisations.

It supersedes parts of `IMPLEMENTATION-PLAN.md` that were written
before profiling. Specifically, the "Phase A: chunked pipelined
ring all-reduce" path turns out to be misdirected; the section on
Phase B (tile-level signalling overlap) is reinforced.

## 1. What we measured

Tooling:
- `strace -c -p <pid>` for a single 22-second inference request
- `sudo perf record -F 200 -g -p <pid> sleep 15` on both root and worker
- Custom Python benchmark with n = 20 samples post-warmup, paired vs
  the documented 13.82 tok/s baseline.

Cluster: 4 x Pi 5 16 GB on Gigabit Ethernet through a managed PoE
switch. v0.16.5 + the patches landed in `pi5-cluster` branch.

## 2. Where the CPU actually goes (perf, root and worker)

| Symbol | Root | Worker | Category |
|--------|------|--------|----------|
| `matmulForward_Q80_Q40_F32` | 49 % | 50 % | Q40 MoE matmul (real compute) |
| `executorThreadHandler` | 17 % | 16 % | Busy-spin in the inter-step barrier |
| `NnExecutor::forward()` | 8 %  | 7 %  | Orchestration |
| `matmulForward_F32_F32_F32` | 7 % | 4 % | F32 matmuls (gate, norms) |
| Everything else | ~19 % | ~23 % | Sync I/O, page faults, etc. |

Compute (rows 1 and 4) is 56 % of wall time. The barrier busy-spin
in `nn-executor.cpp:168-171` is **17 %**. That spin is what the
executor does while waiting for a SYNC step to finish on whichever
thread is shouldering the network I/O.

## 3. Syscall counts (strace, one 22 s request)

Total of 49 932 syscalls captured. The interesting ones:

| Syscall | Calls | Errors (EAGAIN) | Wall time (s) | Mean (us) |
|---------|------:|----------------:|--------------:|----------:|
| `recvfrom` | 41 223 | 30 098 | 0.474 | 11 |
| `sendto`   | 7 940  | 0      | 0.465 | 58 |
| `futex`    | 80     | 0      | 0.040 | 501 |
| Total      | 49 932 | 30 104 | 0.980 | 19 |

73 % of recvs return `EAGAIN`. The do-while loops in `writeMany` /
`readMany` (`nn-network.cpp:477-549`) round-robin across peer sockets
and spin on `EAGAIN` until each socket drains. The number of "failed"
recvs is high but each one is cheap (11 us mean).

**Total kernel time is 0.98 s out of 22 s of wall time -- 4.5 %.**
Syscalls are not the bottleneck.

## 4. Network utilisation

Per token: ~98 sync ops, ~1.7 MB sent + received aggregate -> ~24 MB/s
of TX/RX during sustained inference. With four nodes that totals about
50 MB/s on the wire -- **40 % of Gigabit Ethernet line rate**.

The link is not saturated.

## 5. Sync wall-clock budget

17 % busy-spin on each of 4 cores over one second of wall = 0.68 CPU-s
of spin. At 14 tok/s that is ~48 ms of spin per token, distributed
across 4 cores -> **~12 ms of wall-clock per token spent waiting on
sync**. Divided by 98 syncs gives roughly **120 us per sync barrier**.

This is within an order of magnitude of a LAN ping (~200 us). The
current implementation already extracts most of what the hardware
provides per round trip; the bottleneck is round-trip latency, not
bandwidth or syscall overhead.

## 6. Why the MAX_CHUNK_SIZE bump did not move the needle

The original hypothesis (in the first version of the implementation
plan) was that `MAX_CHUNK_SIZE = 4096` in `nn-network.cpp` was
fragmenting per-layer slices into ~100 syscalls per peer per direction
and that the syscall overhead dominated. Strace shows this is wrong:

- send/recv combined cost 0.94 s of kernel time on 22 s of wall time.
- send buffers are already 8 MiB (set in `setSocketBuffers`); a single
  `send()` call could (and largely does) push the whole slice into
  the kernel before yielding.
- The kernel quietly pipelines: while node A is in `writeMany`, the
  peer's data is accumulating in A's recv buffer. By the time
  `readMany` starts, much of the work is done -- "writeMany then
  readMany" is half-duplex in code but full-duplex through the
  kernel.

Measured impact of raising `MAX_CHUNK_SIZE` from 4096 to 1 MiB:
13.559 vs 13.632 tok/s, within the 95 % CI of +/- 0.06 -- statistically
no difference.

Verdict: the change is reverted on `pi5-cluster`. It is preserved
in branch `phase-a-chunked-ring` as a documented dead end.

## 7. The real bottleneck: round-trip latency, expressed as busy-spin

The 17 % busy-spin time is genuinely the cores waiting for the network
to deliver the next sync result. The wait is real, not artefactual:

- Network: ~40 % of GbE used. Not saturated. So adding bandwidth (10 G,
  jumbo frames, etc.) only buys marginal speedup, since each sync is
  latency-limited.
- Syscalls: ~4.5 % of wall. Not significant.
- Sync barrier wall-clock: 12 ms per token. Distributed across cores
  as 4 x 12 = 48 ms of CPU-time spinning.
- Per sync: ~120 us. Roughly the speed of light + Ethernet store-and-
  forward on this hardware.

Halving sync latency through a smarter algorithm is impossible without
RDMA-class hardware. **The only direction left is to hide the latency
behind compute -- the FlashOverlap pattern.**

## 8. Implication for the implementation plan

Phase A as originally proposed (chunked ring all-reduce, Rabenseifner
hybrid, larger chunks) is **dead**:

- chunked ring vs current star-with-parallel-sockets: the latter
  already runs full-duplex through the kernel and ties up the network
  efficiently.
- Rabenseifner: as already noted, optimised for latency at the cost
  of bandwidth; the wrong trade for our payload sizes.
- Larger chunks: measured no improvement.

Phase B (tile-level signalling overlap) remains the real lever:

- Per token, ~12 ms of wall is spent in sync barriers.
- If even 60 % of those 12 ms can be overlapped with the producer-
  side matmul of the next layer, we save ~7 ms/token -> +10-12 %
  throughput at 14 tok/s.
- The technique is FlashOverlap (arXiv 2504.19519) ported from CUDA
  signal/wait to CPU std::atomic. The dllama executor already has
  a per-step atomic barrier (`currentStepIndex`); the change is to
  add per-tile atomics that allow the network thread to start
  shipping bytes as soon as a matmul output tile is ready, instead
  of waiting for the whole output.

## 9. What this changes for the next session

1. The branch `phase-a-chunked-ring` is abandoned (kept on GitHub
   for the audit trail). The cluster is back on `pi5-cluster`.
2. The IMPLEMENTATION-PLAN.md will be updated to:
   - Strike Phase A.
   - Promote Phase B to the next active task.
3. The next session starts a `phase-b-overlap` branch from
   `pi5-cluster` and adds:
   - per-tile atomic ready flags in `NnExecutor`
   - a network drain function (`drive_network`) callable from any
     thread waiting on a tile
   - matmul output split into K tiles (start with K = 4, equal to
     the number of compute threads)
   - benchmarking harness with bit-exact comparison of logits at
     `--seed 42` against the current baseline

Risk acknowledgement: the cluster has shown repeatedly that the
all-reduce timing is fragile (PGO, MSG_ZEROCOPY, isolcpus + nthreads
all broke it). Phase B touches the same code paths; expect 5-10
iterations before a stable version. Mitigation: feature-flagged behind
`--async-sync=on`, default off.

## 10. Numeric reference table

For future regressions, the steady-state baseline measured on the
operational cluster running v0.16.5 + patches on `pi5-cluster`:

| Metric | Value |
|--------|-------|
| Throughput (mean of n = 20) | 13.6 tok/s |
| 95 % CI | +/- 0.06 |
| Per-token wall | ~72 ms |
| Sync barrier per token | ~12 ms (17 % of wall) |
| Syscall total per token | ~3.3 ms (4.5 % of wall) |
| Matmul (Q80 + F32) | ~40 ms (56 % of wall) |
| Network throughput sustained | ~50 MB/s (40 % of GbE) |
| Sync ops per token | ~98 |
| Wall-clock per sync | ~120 us |
| recvfrom calls per request | 41 k (73 % EAGAIN) |
| sendto calls per request | 7.9 k |
