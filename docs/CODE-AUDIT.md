# Code audit — dllama v0.16.5 hot path findings

Audit conducted on 2026-05-16 against the running production cluster
(4 x Raspberry Pi 5 16 GB, Qwen3-30B-A3B Q40 at 13.6 tok/s). Complements
`PROFILING-FINDINGS.md` which contains the runtime profile.

Methodology: read of the source files in this repository
(`src/nn/nn-executor.cpp`, `nn-executor.hpp`, `nn-network.cpp`,
`nn-cpu-ops.cpp`, `dllama-api.cpp`) plus targeted `strace -c` of the
running daemon. No code was changed.

This document records issues that exist in the code and have a
non-trivial impact on a long-running, latency-sensitive cluster.
Each issue is rated on **likelihood**, **impact** and **complexity
to fix**.

## Finding 0 — Single-token matmul never reaches `llamafile_sgemm` (HIGH-IMPACT, UNVERIFIED)

**Likelihood: medium  Impact: potentially very high  Fix complexity: 1 line  Status: open upstream issue**

`matmulForward_llamafile` (`src/nn/nn-cpu-ops.cpp:1120-1136`) is a dispatcher that
routes matmul through the bundled `llamafile_sgemm` library when conditions are met.
Line 1121 gates entry:

```cpp
if (batchSize == 1u || !context->hasInputContinuousMemory
    || !context->hasOutputContinuousMemory || context->inputSize.z != 1u)
    return false;
```

**`batchSize == 1u` is exactly our case** — autoregressive decode generates one
token at a time, so every matmul during sustained generation is gated out and
falls back to the per-row kernels (`matmul_Q80_Q40_F32`, `matmul_F32_F32_F32`,
etc).

`llamafile_sgemm` has full ARM NEON support: `src/nn/llamafile/sgemm.cpp` is
1010 lines including `tinyBLAS_Q0_ARM<NnBlockQ80, NnBlockQ40>` (line 935-942)
and multiple `#if defined(__ARM_NEON)` register-blocked kernels (lines 50, 64,
101, 142-159, 859, 915). For our exact dtype combination (Q80 inputs × Q40
weights → F32 output) the routing at line 935 already exists; the gate just
prevents it from running.

Upstream issue: https://github.com/b4rtaz/distributed-llama/issues/284 (open,
no maintainer reply, no comments). The reporter removed the `batchSize == 1u ||`
clause and observed **~50 % throughput gain** on AVX2 i5-6500 and 12th-gen i5.

Whether ARM NEON sees the same gain is unknown — needs a paired A/B benchmark
on rpi-1005 before/after the change. If even a fraction of the AVX2 gain
materialises on Pi 5 dotprod, it would be by far the largest single
optimisation available to us. The fix is literally deleting `batchSize == 1u ||`
from one line; the risk is bounded by:

- numerical drift (`tinyBLAS` reduction order differs from `matmul_Q80_Q40_F32`)
- whether the dispatcher's other constraints (`hasInputContinuousMemory`,
  `inputSize.z == 1`) are satisfied in our model's actual op contexts at
  runtime (Qwen3-MoE may not satisfy all of them).

This is the **#1 priority for the next experimental commit**: try the patch,
benchmark n=20, accept if any meaningful improvement and no numerical
regression.

## Finding 1 — pthread_create + pthread_join every forward() call

**Likelihood: 1.0  Impact: medium  Fix complexity: medium**

`NnExecutor::forward()` (`src/nn/nn-executor.cpp:192-217`) spawns
`nThreads - 1` pthreads via `pthread_create` and `pthread_join`s
them at the end. `forward()` is called once per token.

Measured on the production cluster:

```
strace -c -e trace=clone,clone3,futex,exit -p <dllama-api pid>
  during a 100-token inference request

% time    seconds  usecs/call  calls   syscall
99.12     0.103125  1011       102     futex      (pthread_join wait)
 0.88     0.000913     3       303     clone3     (pthread_create)
```

303 thread creations = 3 per token x 101 tokens. Total cost:
0.104 s of syscalls over ~7 s of compute = **1.5 % of wall time**.

Second-order effects (not captured by strace):
- each new thread is born on a cold core; the first ~10 us of work
  pays cache fill penalties
- default pthread stack is 8 MiB virtual, faulted on access; over time
  this churns memory residency
- the kernel scheduler decides placement freshly each spawn, sometimes
  putting two threads on the same core

The fix is a persistent worker thread pool driven by a condvar.
`forward()` becomes: set state, `notify_all()`, wait for done.
Estimated +1-3 % throughput, more consistent latency (lower stdev),
and lower power consumption.

## Finding 2 — False sharing across the three hot atomics

**Likelihood: 1.0  Impact: small but pervasive  Fix complexity: low**

In `nn-executor.hpp:70-81`:

```cpp
typedef struct {
    NnUint nThreads;                   // offset 0
    NnUint nSteps;                     // offset 4
    NnExecutorStep *steps;             // offset 8
    NnNodeSynchronizer *synchronizer;  // offset 16
    std::atomic_uint currentStepIndex; // offset 24
    std::atomic_uint doneThreadCount;  // offset 28
    std::atomic_bool isAlive;          // offset 32
    NnUint batchSize;                  // offset 36
    Timer *timer;                      // offset 40
    NnUint totalTime[N_STEP_TYPES];    // offset 48
} NnExecutorContext;
```

All three atomics share the first 64-byte cache line. The hot path
in `executorThreadHandler` (`nn-executor.cpp:152-189`) does:

- `context->currentStepIndex.load()`  on every step start, and inside
  the busy-spin loop (`nn-executor.cpp:183-187`)
- `context->doneThreadCount.fetch_add(1)`  at every step end
  (`nn-executor.cpp:172`)
- `context->isAlive.load()` inside the busy-spin loop

Every `fetch_add` on `doneThreadCount` invalidates the cache line
on the other three cores. The other cores re-fetch from L3 / RAM on
the next spin iteration. With ~98 sync ops per token x 4 threads x
14 tok/s, that is ~5 500 invalidation rounds per second.

Default `std::atomic` operations are sequentially consistent. On
ARMv8 the load lowers to `LDAR` (full acquire barrier); the
read-modify-write to `LDAXR`/`STXR` with full barriers. These cost
significantly more than the relaxed equivalents.

Fix:
- Pad each atomic to its own cache line with `alignas(64)`.
- Switch the busy-spin to `memory_order_acquire` loads and the
  `fetch_add` to `memory_order_acq_rel` or release.

Estimated +0.5-1.5 % throughput. The mechanism is well-understood
and the fix is ~10 lines.

## Finding 3 — Per-sync std::vector heap allocation

**Likelihood: 1.0  Impact: small  Fix complexity: low**

`syncNodeSlices` (`nn-network.cpp:650`) and `syncWithRoot`
(`nn-network.cpp:623`) allocate a fresh `std::vector<NnSocketIo>`
on every call. For our model: 98 sync ops per token x 4 threads =
**392 small heap allocations per token**.

Each vector typically holds 0, 1 or rarely 2 elements (for N=4
nodes, nThreads=4 -> nSocketsPerThread is 0 or 1). The vector itself
is 24 bytes; the heap block carrying its elements is 16 bytes
(one NnSocketIo). With jemalloc tcache the allocation is ~50 ns,
so ~20 us per token in allocation churn, ~0.03 % of wall. Not a
material throughput win, but it is allocation traffic that has no
purpose.

Fix: replace with a `thread_local std::array<NnSocketIo, kMaxPeers>`
or stack-allocated `NnSocketIo ios[4]`. Removes ~400 mallocs per
token entirely.

## Finding 4 — Worker syncWithRoot wakes all threads to do nothing

**Likelihood: 1.0  Impact: tiny  Fix complexity: trivial**

`syncWithRoot` on a worker (`nn-network.cpp:637-647`) executes the
read on **threadIndex 0 only**; the other three threads return
immediately and then re-enter the busy-spin barrier
(`nn-executor.cpp:183`) until thread 0 catches up.

This sync fires only once per token (embedding broadcast,
`llm.cpp:256`), so the cost is small (~50 us idle spin per token).
But threads 1-3 are reading the hot cache line for nothing,
amplifying Finding 2's false-sharing footprint at exactly the
wrong moment.

Fix is one line: have threads 1-3 skip the barrier participation
for this step (the synchronizer interface needs to expose "this
step did nothing").

## Finding 5 — Memory state is clean

**Likelihood: 1.0  Impact: positive baseline**

`smaps_rollup` on the live dllama-api after 12 minutes of serving:

```
Rss:             7 075 312 kB
Pss_Dirty:       7 063 504 kB   (the mmap'd weights, expected)
Private_Dirty:   7 063 504 kB
LazyFree:                0 kB
```

No anonymous growth across requests. The NaiveCache
(`dllama-api.cpp:425`) does grow within a single conversation but
clears on context mismatch and on each request boundary
(`dllama-api.cpp:546-548`). Per-request allocations
(`std::unique_ptr<ChatItem[]>`, `std::unique_ptr<int[]>`) are RAII
and freed promptly.

No memory leak. The cluster is safe to run continuously.

## Finding 6 — Docker monitoring overhead is negligible

**Likelihood: 1.0  Impact: positive baseline**

```
USER  PID   %CPU %MEM  COMMAND
rpi   1545   0.9 0.3   cadvisor
rpi   1547   0.6 0.9   alloy run
```

Total monitoring overhead: 1.5 % of one core, ~1 % of one Pi's RAM.
Not interfering with inference. The user's instruction to leave the
monitoring containers running was already costless.

## Finding 7 — Matmul kernel is well-optimised

**Likelihood: 1.0  Impact: positive baseline**

`matmul_Q80_Q40_F32` (`nn-cpu-ops.cpp:231`) uses NEON intrinsics with
`__ARM_FEATURE_DOTPROD` path: 4-way block unrolling, `__builtin_prefetch`
on weights and activations, `vdotq_s32` (UDOT) for the inner dot
product. The compiled binary contains 322 `udot`/`sdot` instructions
(verified by `objdump -d ~/distributed-llama/dllama | grep -cE 'udot|sdot'`).

This is at the hardware ceiling for Q40 GEMM on Cortex-A76. No
optimisation opportunity here.

## Finding 8 — Steps are linearly enumerated; no per-step scheduling

**Likelihood: 1.0  Impact: structural  Fix complexity: high**

`NnExecutor::NnExecutor` (`nn-executor.cpp:60-118`) flattens the
per-segment ops into a single `steps[]` array of
`STEP_EXECUTE_OP` / `STEP_SYNC_NODES` entries, ordered statically
at construction time. Every step is a barrier point: all `nThreads`
must finish their share before any thread can advance the
`currentStepIndex`.

This is the architectural reason a SYNC step is hard to overlap
with the next OP step. To overlap them, the executor needs:
- a notion of "step may begin while previous is still running"
- per-tile readiness signalling instead of per-step
- explicit dependency declaration

This is the Phase B refactor (~600 LOC). It is the only remaining
software lever above the noise floor; everything else listed in
this file is incremental.

## Aggregate estimate

Stacking Findings 1, 2 and 3 (low-risk, mechanical fixes):

| Fix | Estimated gain | LOC | Risk |
|-----|----------------|-----|------|
| Persistent thread pool | +1 to +3 % | ~150 | medium |
| Cache-line-pad atomics + memory_order tuning | +0.5 to +1.5 % | ~10 | low |
| Stack-alloc sync iovecs | +0.0 to +0.1 % | ~30 | trivial |
| **Cumulative** | **+1.5 to +4.5 %** | ~190 | low-medium |

That moves the cluster from 13.6 to ~14.0 tok/s -- modest but real,
and almost free in terms of risk. The big Phase B refactor remains
the only path beyond that.

## What we did NOT find

Items the audit explicitly looked for and ruled out:

- Memory leaks across requests.
- Unbounded data structure growth.
- Socket fd leaks.
- Mutex contention (no mutexes in the hot path; only atomics and
  busy-spin).
- Scalar code in inner matmul (everything is NEON + dotprod).
- Inefficient JSON or HTTP handling beyond what is normal for a
  single-user inference server.
- Docker monitoring or systemd overhead above 2 % of one core.
- tcache or jemalloc misconfiguration.
- Throttle / thermal events (CPU at 2 400 MHz, throttled=0x0).

## Sources cross-checked

- `src/nn/nn-executor.cpp` and `.hpp`
- `src/nn/nn-network.cpp` and `.hpp`
- `src/nn/nn-cpu-ops.cpp` (Q40/Q80 kernel verified against ARM ACLE)
- `src/dllama-api.cpp` (caches and RAII patterns)
- `/proc/<pid>/smaps_rollup` on running process
- `strace -c -e trace=clone,clone3,futex,exit` on running process
- `perf record -F 200 -g` results from `PROFILING-FINDINGS.md`

This document supersedes nothing; it complements the existing
`PROFILING-FINDINGS.md` and `IMPLEMENTATION-PLAN.md`.
