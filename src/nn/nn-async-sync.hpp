// Phase B / B5+B6 — async sync orchestrator.
//
// Pairs a producer (matmul that emits per-tile signals) with a network
// drain thread that ships each tile as soon as it is signalled, in parallel
// with the matmul that is still producing.
//
// This file is the public API. Implementation in nn-async-sync.cpp.
//
// Usage pattern at a sync site (pseudocode):
//
//   NnAsyncSyncContext ctx;
//   ctx.tiles    = NN_K_TILES;
//   ctx.network  = network;
//   ctx.nNodes   = nNodes;
//   ctx.nodeIndex = nodeIndex;
//   ctx.buffer   = pipeBatch;        // gets writes from matmul AND from peers
//   ctx.nBytes   = batchBytes;
//   ctx.signals  = signalArray;       // NN_K_TILES entries
//   ctx.peerArrived = peerArrivedArr; // [nNodes-1][NN_K_TILES] of NnTileSignal
//
//   // Reset signals before the matmul writes to the buffer
//   nnAsyncSyncReset(&ctx);
//
//   // Spawn the drain thread (it spins waiting for ctx.signals[t] >= READY)
//   nnAsyncSyncStart(&ctx);
//
//   // Run the matmul. Pass &ctx so the matmul can call nnAsyncSyncEmit(t)
//   // after writing tile t of its output to ctx.buffer.
//   matmul_Q80_Q40_F32_withEmitter(..., &ctx);
//
//   // Block until the drain thread has sent everything AND collected all
//   // peer tiles. Returns when ctx.buffer holds the full gathered result.
//   nnAsyncSyncJoin(&ctx);
//
// Compile-time toggles:
//   NN_ASYNC_SYNC_BUSY_POLL — if defined, drain thread busy-spins. Otherwise
//                            uses condvar (lower CPU but higher latency).

#ifndef NN_ASYNC_SYNC_H
#define NN_ASYNC_SYNC_H

#include "nn-core.hpp"
#include "nn-tile-signal.hpp"
#include "nn-network.hpp"
#include <pthread.h>

struct NnAsyncSyncContext {
    // Topology
    NnNetwork *network;
    NnUint     nNodes;
    NnUint     nodeIndex;       // 0 = root
    NnUint     nThreads;
    NnUint     threadIndex;

    // Data plane
    NnByte    *buffer;          // pipe slice buffer; my slice at offset nodeIndex*sliceBytes
    NnSize     nBytes;          // total pipe size (across all nodes)
    NnUint     tiles;           // K (typically NN_K_TILES)

    // Signals from matmul (one per tile, indicates that my buffer's tile is filled)
    NnTileSignal *signals;      // size = tiles

    // Per-peer per-tile arrival flags (indicates this peer's tile has been received)
    NnTileSignal *peerArrived;  // size = (nNodes-1) * tiles, row-major

    // Drain thread handle (managed internally)
    pthread_t handle;
    volatile bool started;
    volatile bool aborted;       // set on error
    char     errMsg[128];
};

// Reset all signals to state 0. Call BEFORE matmul writes anything to buffer.
void nnAsyncSyncReset(NnAsyncSyncContext *ctx);

// Spawn the drain thread. Returns immediately. The drain thread spins on
// ctx->signals[t] for t = 0..tiles-1, ships the corresponding bytes to all
// peers, and reads the corresponding bytes from all peers into
// ctx->buffer[peer_slice_offset + t*tileBytes].
void nnAsyncSyncStart(NnAsyncSyncContext *ctx);

// Called by the matmul producer after tile t has been written to
// ctx->buffer[ctx->nodeIndex*sliceBytes + t*tileBytes].
// This is just a thin wrapper around ctx->signals[t].signalReady() for
// clarity and to keep producer-side code readable.
static inline void nnAsyncSyncEmit(NnAsyncSyncContext *ctx, NnUint t) {
    ctx->signals[t].signalReady();
}

// Block the calling thread until the drain thread has completed all peer
// receives. After return, ctx->buffer holds the fully-gathered result.
// Joins the drain pthread.
//
// Throws NnTransferSocketException if any peer connection failed (set in
// ctx->aborted + ctx->errMsg by the drain thread).
void nnAsyncSyncJoin(NnAsyncSyncContext *ctx);

// Convenience: drop-in replacement for the existing syncNodeSlices that
// drives the legacy path when the producer did not signal tiles. Reads all
// peer slices in one pass after the matmul has already completed. Useful as
// the unit-test path: with K=1 and a "signal all done" stub it must produce
// bit-exact output vs the current code.
void nnAsyncSyncLegacyFallback(NnAsyncSyncContext *ctx);

#endif // NN_ASYNC_SYNC_H
