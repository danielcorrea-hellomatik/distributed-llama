# Phase B — Wiring guide for next session

Status: all building-block code is committed on `opt-b5b6-tiled-sync`.
This file documents what remains to integrate them.

## What's already in code (no deploy needed)

```
src/nn/nn-tile-signal.hpp       — TileSignal primitive (release/acquire)
src/nn/nn-async-sync.hpp        — NnAsyncSyncContext + API
src/nn/nn-async-sync.cpp        — drain thread (header wire, abort path)
src/nn/nn-cpu-ops.cpp           — matmul_Q80_Q40_F32_tiled (producer)
src/app.cpp / .hpp              — --tile-sync K CLI flag
src/nn/nn-network.cpp           — syncNodeSlicesTiled (chunked-IO fallback)
Makefile                        — links nn-async-sync.o into both binaries
```

## What remains (~60 LOC + testing)

### Step 1 — A new dispatch in `NnNetworkNodeSynchronizer::sync`

When `tileSync > 0` AND the sync type is `SYNC_NODE_SLICES`, the
synchronizer should:

1. Allocate an `NnAsyncSyncContext` (pre-allocate `signals[K]`,
   `peerArrived[(N-1)*K]`, `tileDone[K]` once at construction).
2. Call `nnAsyncSyncReset(&ctx)`.
3. Call `nnAsyncSyncStart(&ctx)` to spawn the drain thread.
4. Run the consumer matmul (the one whose output IS the slice) via
   `matmul_Q80_Q40_F32_tiled(..., &ctx, tileDone)`. The matmul will
   emit signals as it produces tiles.
5. Call `nnAsyncSyncJoin(&ctx)` to wait for all peer tiles to arrive.

But wait — in the current executor, the matmul is a previous step
and the sync is the NEXT step. The synchronizer is invoked AFTER
the matmul has fully completed. There's nothing to overlap with at
that point.

To get real overlap, the executor needs to FUSE the
last-matmul-before-sync with the SYNC step. Two approaches:

**Option A — Fuse at executor level (preferred)**

In `NnExecutor::NnExecutor`, when building the linear `steps[]`
vector, detect the pattern `OP(matmul) → CAST → SYNC` and replace
those three steps with a single `STEP_TILED_MATMUL_SYNC` that
calls the tiled matmul with an active orchestrator.

Estimated LOC:
- New step type in `nn-executor.hpp` (~5 LOC)
- Detect pattern in NnExecutor ctor (~20 LOC)
- Run the fused step in `executeStep` (~30 LOC)
- Total: ~55 LOC

**Option B — Use a peer thread to do matmul while sync drains**

Instead of the calling thread doing both matmul and join, the
calling thread spawns the matmul on a worker (the same persistent
pool used by F1) and spawns the drain. Both run concurrently.
Issue: the F1 pool's threads enter `executorStepLoop` which is the
barrier itself — they're the same threads. Repurposing them mid-step
is intrusive.

**Option C — Restructure the matmul step itself**

Make `matmulForward_Q80_Q40_F32` aware of an attached
`NnAsyncSyncContext` (passed via the op context). When present, it
calls `matmul_Q80_Q40_F32_tiled` instead of the plain kernel, and the
existing SYNC step that follows it consumes the already-running
drain thread's output via `nnAsyncSyncJoin` only.

Cleanest because it touches one op and one synchronizer dispatch.

Recommendation: **Option C**. Outline:

```cpp
// In NnCpuOpContext, add a field:
NnAsyncSyncContext *asyncSync;   // nullptr when overlap disabled

// In matmulForward_Q80_Q40_F32:
if (context->asyncSync != nullptr && !context->asyncSync->started) {
    nnAsyncSyncReset(context->asyncSync);
    nnAsyncSyncStart(context->asyncSync);
    matmul_Q80_Q40_F32_tiled(output, x, w, n, d, nThreads, threadIndex,
                              context->asyncSync, context->asyncSync_tileDone);
    return;
}
// ... existing path ...

// In NnNetworkNodeSynchronizer::sync, when this segment had an async
// matmul, just nnAsyncSyncJoin() — drain has been running concurrently.
```

The plumbing of `asyncSync` into the op context is the only architectural
change. The op builder (`llm.cpp`) needs to attach the context to specific
ops (matmul_wo, the MoE w2 final matmul, etc.) at construction time.

### Step 2 — Build + deploy

When the wiring above is done, build on each Pi, restart services with
`--tile-sync 4` (or 2), and run a 60-minute soak test before merging.

### Step 3 — Test plan from the risk analysis

- Bit-exact greedy at `--seed 42`, 100 tokens: must match baseline.
- Paired n=20 benchmark vs baseline 13.7 tok/s.
- Chaos test: kill -9 one worker during decode; observe clean abort.
- Numerical drift histogram over 1k random prompts.

### Step 4 — Watchdog (mandatory before merge)

Add a 50 ms timeout per tile in the drain thread (currently no
timeout). On timeout: set `ctx->aborted`, set errMsg, return.
`nnAsyncSyncJoin` already surfaces this as an exception.

### Step 5 — Q80 alignment guard

In `matmul_Q80_Q40_F32_tiled`: add `static_assert` and runtime
`assert(d % K == 0 && tileBytes % 34 == 0)` to catch ill-shaped
matmuls before they corrupt data.

## Estimate

- Step 1 (Option C wiring): ~40 LOC across nn-cpu-ops.cpp +
  nn-network.cpp + llm.cpp.
- Step 2-4 (tests + watchdog): ~30 LOC + cluster soak.

Realistic next-session time: 4-6 focused hours plus a 60-minute
soak. The pieces are all in place; this is integration work, not
research.
