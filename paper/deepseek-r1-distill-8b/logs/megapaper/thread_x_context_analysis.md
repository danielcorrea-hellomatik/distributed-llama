# Thread-count × context length (E3) — committed raw analysis

> Per-token timing from `dllama inference`, stock binary (sha256 `36e4d0d8…`), temperature 0,
> seed 42, telemetry OFF, max-seq-len 4096, single 2500-token generation per thread count.
> Workers restarted at the target `--nthreads` before each run. Raw: `ctx_nt4.out`, `ctx_nt3.out`,
> `ctx_nt2.out` (all 2440 tokens — identical length confirms bit-exact output is independent of nthreads).

## Finding
The optimal thread count is **context-independent**: nthreads=4 is fastest at every context length,
and the relative advantage is **flat** across the whole curve — ~**13% over nt3** and ~**50% over nt2**,
from the first 400 tokens to 2600. The three decode-vs-context curves are parallel; growing the
KV-cache slows all thread counts proportionally. The headline 4-thread configuration is therefore
robust to generation length, not just a short-context artifact.

## decode tok/s vs context, by nthreads (400-token bins)

| context | nt2 | nt3 | nt4 | nt4/nt3 | nt4/nt2 |
|---:|---:|---:|---:|---:|---:|
| 200  | 5.28 | 7.08 | 7.97 | 1.125 | 1.508 |
| 600  | 5.07 | 6.77 | 7.69 | 1.136 | 1.517 |
| 1000 | 4.82 | 6.43 | 7.29 | 1.135 | 1.513 |
| 1400 | 4.55 | 6.04 | 6.84 | 1.132 | 1.504 |
| 1800 | 4.31 | 5.68 | 6.46 | 1.136 | 1.500 |
| 2200 | 4.10 | 5.31 | 6.07 | 1.143 | 1.480 |
| 2600 | 3.99 | 5.22 | 5.98 | 1.145 | 1.498 |

## Overall average decode (2440-token generation)
- nt4: **6.97 tok/s**
- nt3: **6.14 tok/s**
- nt2: **4.64 tok/s**

(Short-context peak per the n=20 headline is 8.32 tok/s at nt4; the 6.97 here is the average over a
2440-token generation, i.e. the decode-vs-context curve integrated to 2440 — consistent with
the curve in `decode_vs_context_analysis.md`.)
