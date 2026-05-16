// Phase B / B5+B6 — async sync orchestrator implementation.
//
// Drain thread: spins waiting for signals[t] to become READY, then ships
// the corresponding tile to every peer socket assigned to this thread.
// Once all my tiles are shipped, the same thread does the reads of peer
// tiles into the buffer.

#include "nn-async-sync.hpp"
#include <cstring>
#include <cstdio>
#include <atomic>
#include <stdexcept>

void nnAsyncSyncReset(NnAsyncSyncContext *ctx) {
    for (NnUint t = 0; t < ctx->tiles; t++)
        ctx->signals[t].reset();
    for (NnUint p = 0; p < (ctx->nNodes - 1) * ctx->tiles; p++)
        ctx->peerArrived[p].reset();
    ctx->started = false;
    ctx->aborted = false;
    ctx->errMsg[0] = '\0';
}

// ---------------------------------------------------------------------------
// Drain thread
// ---------------------------------------------------------------------------
//
// Wire layout per tile, per peer:
//   [NnTileHeader (8B)][payload (tileBytes)]
//
// Receive logic uses the header's tileIndex to place the payload into the
// correct slot of ctx->buffer:
//
//   buffer + (peer_slice_index * sliceBytes) + (tileIndex * tileBytes)
//
// where peer_slice_index is the peer's nodeIndex (NOT the socket index --
// they differ by one when the peer is at or beyond this node's nodeIndex).
// ---------------------------------------------------------------------------

static void *drainThreadMain(void *arg) {
    NnAsyncSyncContext *ctx = (NnAsyncSyncContext *)arg;
    NnNetwork *network = ctx->network;
    const NnSize sliceBytes = ctx->nBytes / ctx->nNodes;
    const NnSize tileBytes = sliceBytes / ctx->tiles;
    const NnUint nPeers = ctx->nNodes - 1;
    NnByte *mySliceBase = ctx->buffer + (NnSize)ctx->nodeIndex * sliceBytes;

    // ---- SEND PHASE ----------------------------------------------------
    // For each tile, in order, wait for matmul to emit it, then send to all
    // peers attached to this drain thread.
    try {
        for (NnUint t = 0; t < ctx->tiles; t++) {
            ctx->signals[t].waitAtLeast(1);

            NnByte *tileSrc = mySliceBase + (NnSize)t * tileBytes;

            NnTileHeader hdr;
            hdr.syncId    = 0;  // single-sync use; bumped per call by caller in v2
            hdr.tileIndex = (uint8_t)t;
            hdr.nTiles    = (uint8_t)ctx->tiles;
            hdr.payloadLen = (uint32_t)tileBytes;

            // Issue send to every peer socket; writeMany interleaves under
            // the hood (non-blocking; loops on EAGAIN).
            // We send the header in a separate writeMany batch to keep
            // hdr lifetime simple — total overhead is 8B per peer per tile.
            std::vector<NnSocketIo> hdrIos(network->nSockets);
            std::vector<NnSocketIo> payIos(network->nSockets);
            for (NnUint s = 0; s < network->nSockets; s++) {
                hdrIos[s].socketIndex = s;
                hdrIos[s].data = &hdr;
                hdrIos[s].size = sizeof(NnTileHeader);
                payIos[s].socketIndex = s;
                payIos[s].data = tileSrc;
                payIos[s].size = tileBytes;
            }
            network->writeMany(network->nSockets, hdrIos.data());
            network->writeMany(network->nSockets, payIos.data());

            ctx->signals[t].signalSent();
        }

        // ---- RECEIVE PHASE ---------------------------------------------
        // We expect K tiles from each of (nNodes-1) peers. Receive headers
        // one at a time on each socket and use the tileIndex to dispatch to
        // the correct buffer offset.
        const NnUint expectedPerPeer = ctx->tiles;
        for (NnUint received = 0; received < expectedPerPeer * nPeers; received++) {
            // For simplicity v1: receive in socket-strict order. A future
            // version can use poll/epoll to drain whichever socket is ready.
            NnUint s = received % network->nSockets;
            // The peer's slice index in buffer:
            // socketIndex s addresses peer p; peer p's nodeIndex is
            //   p_node = s >= ctx->nodeIndex ? s + 1 : s
            NnUint peerNodeIndex = (s >= ctx->nodeIndex) ? s + 1 : s;
            NnByte *peerSliceBase = ctx->buffer + (NnSize)peerNodeIndex * sliceBytes;

            NnTileHeader hdr;
            NnSocketIo hdrIo = { s, &hdr, sizeof(NnTileHeader) };
            network->readMany(1, &hdrIo);
            if (hdr.payloadLen != tileBytes || hdr.tileIndex >= ctx->tiles || hdr.nTiles != ctx->tiles) {
                snprintf(ctx->errMsg, sizeof(ctx->errMsg),
                    "drain: bad header from peer node %u: tile=%u/%u payload=%u (expected %u)",
                    peerNodeIndex, (unsigned)hdr.tileIndex, (unsigned)hdr.nTiles,
                    (unsigned)hdr.payloadLen, (unsigned)tileBytes);
                ctx->aborted = true;
                return nullptr;
            }

            NnByte *dst = peerSliceBase + (NnSize)hdr.tileIndex * tileBytes;
            NnSocketIo payIo = { s, dst, tileBytes };
            network->readMany(1, &payIo);

            // Mark this peer's tile as arrived.
            NnUint slot = (peerNodeIndex > ctx->nodeIndex ? peerNodeIndex - 1 : peerNodeIndex) * ctx->tiles + hdr.tileIndex;
            ctx->peerArrived[slot].signalReceived();
        }
    } catch (const std::exception &e) {
        snprintf(ctx->errMsg, sizeof(ctx->errMsg), "drain: %s", e.what());
        ctx->aborted = true;
    }
    return nullptr;
}

void nnAsyncSyncStart(NnAsyncSyncContext *ctx) {
    int rc = pthread_create(&ctx->handle, NULL, drainThreadMain, (void *)ctx);
    if (rc != 0)
        throw std::runtime_error("nnAsyncSyncStart: pthread_create failed");
    ctx->started = true;
}

void nnAsyncSyncJoin(NnAsyncSyncContext *ctx) {
    if (!ctx->started) return;
    pthread_join(ctx->handle, NULL);
    ctx->started = false;
    if (ctx->aborted)
        throw std::runtime_error(std::string("async sync aborted: ") + ctx->errMsg);
}

void nnAsyncSyncLegacyFallback(NnAsyncSyncContext *ctx) {
    // Stub: if the producer did not signal tiles, mark all as ready so the
    // drain thread can ship them in one batch.
    for (NnUint t = 0; t < ctx->tiles; t++)
        ctx->signals[t].signalReady();
    nnAsyncSyncStart(ctx);
    nnAsyncSyncJoin(ctx);
}
