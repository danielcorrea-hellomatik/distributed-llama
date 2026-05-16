// Phase B Tile-level Signalling primitive (CPU port of FlashOverlap).
//
// Header-only. Producer (matmul epilogue) calls signalReady(t) after writing
// tile t. Consumer (network drain thread) calls waitReady(t) before reading.
// On ARMv8 the underlying ops compile to STLR (release) / LDAR (acquire).
//
// State 0: tile not yet computed
// State 1: producer has written all bytes for this tile (RELEASE)
// State 2: drain thread has handed bytes to send()
// State 3: peer has fully received this tile (ack via header CRC)
//
// In Phase B sub-phases B2-B4 the signalling is wired in but NOT consumed.
// Sub-phase B5 adds the drain thread that reads these signals.

#ifndef NN_TILE_SIGNAL_H
#define NN_TILE_SIGNAL_H

#include <atomic>
#include <cstdint>

// Tiles per layer. Picked per the wave-equal-split heuristic (CoCoNet /
// TokenWeave): tile_rows = ceil(out_rows / (compute_per_row / send_per_row)).
// For Pi 5 + GbE the compute/send ratio is ~25:1; K=4 is the rounded-up
// power-of-2 that keeps tile size > MTU * N (avoiding TCP small-write penalty).
#define NN_K_TILES 4

// Header that prefixes every tile payload on the wire. Wire-portable; 8 bytes.
// Without this the receiver cannot distinguish tile 0 from tile 1 -- that is
// the highest-severity risk in the Phase B plan (risk 5.1).
#pragma pack(push, 1)
struct NnTileHeader {
    uint16_t syncId;       // monotonic per-sync identifier
    uint8_t  tileIndex;    // 0 .. NN_K_TILES-1
    uint8_t  nTiles;       // = NN_K_TILES (carried explicit for forward-compat)
    uint32_t payloadLen;   // bytes following this header
};
#pragma pack(pop)
static_assert(sizeof(NnTileHeader) == 8, "tile header must be 8 bytes");

// One cache-line per signal to avoid false sharing between adjacent tiles in
// the array.
struct alignas(64) NnTileSignal {
    std::atomic<uint32_t> state;

    NnTileSignal() : state(0) {}

    inline void reset()                { state.store(0, std::memory_order_relaxed); }
    inline void signalReady()          { state.store(1, std::memory_order_release); }
    inline void signalSent()           { state.store(2, std::memory_order_release); }
    inline void signalReceived()       { state.store(3, std::memory_order_release); }

    inline uint32_t observe() const    { return state.load(std::memory_order_acquire); }

    // Spin until at least the requested state is reached. ARMv8 yield hint
    // keeps the spin from saturating the cluster wakeup latency.
    inline void waitAtLeast(uint32_t want) const {
        while (state.load(std::memory_order_acquire) < want) {
#if defined(__aarch64__) || defined(__arm__)
            asm volatile ("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
            asm volatile ("pause" ::: "memory");
#endif
        }
    }
};

#endif // NN_TILE_SIGNAL_H
