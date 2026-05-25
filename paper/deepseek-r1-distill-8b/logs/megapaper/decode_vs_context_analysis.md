# Decode throughput vs context length — KV-cache scaling (committed raw analysis)

> Source: per-token timing (`🔶 Pred <ms> Sync <ms>`) emitted by `dllama inference`, parsed from
> three long generations at the 5000-token cap, stock upstream binary (sha256 `36e4d0d8…`),
> nthreads=4, temperature 0, seed 42, telemetry OFF. Raw `.out` files in this directory:
> `cx_math-proof.out` (4958 tok), `cx_code-btree.out` (4962 tok), `cx_reasoning-puzzle.out` (4955 tok).

## Headline finding
Dense decode throughput is **not constant**: it falls roughly linearly with context position as the
KV-cache grows. The 8.32 tok/s record is the **short-context** rate; sustained long-form generation
is slower. From the first 200 to the last 200 tokens of a 5000-token generation, decode drops **~40%**.

## 1. Averaged decode-vs-context curve (n=3 prompts, 250-token bins)

| context (mid) | decode tok/s | ±sd | pred ms | sync ms | network % |
|---:|---:|---:|---:|---:|---:|
| 125  | 8.36 | 0.07 | 99.6  | 20.0 | 16.7 |
| 375  | 8.10 | 0.01 | 103.2 | 20.3 | 16.4 |
| 625  | 7.87 | 0.05 | 106.5 | 20.6 | 16.2 |
| 875  | 7.60 | 0.11 | 109.4 | 22.3 | 16.9 |
| 1125 | 7.36 | 0.13 | 112.6 | 23.3 | 17.2 |
| 1375 | 7.09 | 0.14 | 116.5 | 24.5 | 17.4 |
| 1625 | 6.89 | 0.15 | 120.0 | 25.2 | 17.4 |
| 1875 | 6.68 | 0.15 | 122.9 | 26.9 | 18.0 |
| 2125 | 6.49 | 0.17 | 127.0 | 27.1 | 17.6 |
| 2375 | 6.31 | 0.14 | 130.6 | 27.9 | 17.6 |
| 2625 | 6.18 | 0.09 | 133.9 | 27.9 | 17.3 |
| 2875 | 6.02 | 0.10 | 137.8 | 28.4 | 17.1 |
| 3125 | 5.89 | 0.09 | 141.9 | 28.1 | 16.5 |
| 3375 | 5.73 | 0.10 | 145.8 | 28.7 | 16.4 |
| 3625 | 5.59 | 0.07 | 149.3 | 29.5 | 16.5 |
| 3875 | 5.42 | 0.05 | 153.3 | 31.2 | 16.9 |
| 4125 | 5.27 | 0.08 | 156.5 | 33.2 | 17.5 |
| 4375 | 5.16 | 0.07 | 161.5 | 32.5 | 16.7 |
| 4625 | 5.08 | 0.07 | 164.9 | 32.2 | 16.3 |
| 4852 | 4.97 | 0.05 | 168.2 | 33.0 | 16.4 |

Cross-prompt sd ≤ 0.17 tok/s → the curve is **prompt-independent**: decode rate is a function of
context position (KV size), not content. (math vs code vs logic-puzzle overlay within ~3%.)

## 2. Linear KV model

Per-token wall time fits a straight line in context position with near-perfect agreement:

```
t(L) = t0 + k·L   ms/token
t0 = 116.73 ms      (L=0  -> 8.57 tok/s)
k  = 0.01740 ms per context-token
R² = 0.9993
```

Predictions: L=256 → 8.25 tok/s (matches the 256-step headline), L=1024 → 7.43, L=2048 → 6.56,
L=4096 → 5.32, L=8192 → 3.86 tok/s.

## 3. Interpretation
- The slope `k` is the cost of reading one more token's KV (all 32 layers) each step.
  With F32 KV, kvdim=1024: 2·1024·4·32 = 256 KB/context-token total, /4 nodes = **64 KB/node**.
  k = 0.0174 ms ⇒ implied per-node KV bandwidth **≈ 3.8 GB/s** (order-of-magnitude consistent with
  the F32 KV read sharing the same LPDDR4X bus measured at ~10 GB/s for weights).
- **Network stays ~16–18% of per-token time across the whole curve** → the slowdown is entirely
  on-node compute+memory (KV), not synchronisation. The bandwidth wall deepens with context.
- This reframes the record honestly: 8.32 tok/s is the short-context peak; the effective rate for a
  long reasoning trace (R1 routinely emits thousands of tokens) is the average along this curve
  (~6.2 tok/s over 5000 tokens).
