# DeepSeek-R1-Distill-8B — clean-room A/B attribution: optimized FORK vs stock upstream (dense decode)

> **Purpose.** Produce a same-session, clean-room (telemetry OFF) A/B so the paper's attribution
> table can use *consistent* clean-room numbers instead of stale telemetry-on ones. The claim under
> test: **the fork's source/kernel changes contribute ≈ 0 to DENSE decode** (fork-best ≈ stock-best).
>
> **Bottom line.** They do. With a **balanced interleaved design** (A,B in pass 1; B,A in pass 2)
> to cancel thermal drift, pooled **n=16** each:
>
> | Config | binary | nthreads | n | mean tok/s | 95% CI |
> |---|---|---|---|---|---|
> | **A (FORK, optimized)** | `dllama-custom/dllama` | 4 | 16 | **8.081** | [8.044, 8.118] |
> | **B (STOCK, upstream)** | `distributed-llama/dllama` | 4 | 16 | **8.093** | [8.065, 8.121] |
>
> **A − B = −0.012 tok/s (−0.15%), Welch t = −0.54, df ≈ 28, p ≈ 0.59, CIs heavily overlap.**
> → The fork's contribution to dense decode is **statistically indistinguishable from zero**.
> The fork is *neither faster nor slower* than stock upstream for dense 8B decode at the optimum.
>
> **Optimal nthreads.** Both binaries peak at **nt4** (not nt3 as hypothesized). For the fork,
> nt4 (8.081) > nt3 (7.826) by **+3.26%**; for stock, nt4 (8.093) > nt3 (7.201, prior log) by +12.4%.
> Secondary note: the fork *does* help at nt3 (7.826 vs stock 7.201, +8.7%), but that advantage
> **vanishes at the nt4 optimum** — consistent with dense decode being bandwidth/saturation-bound
> at full thread occupancy, where extra compute throughput buys nothing.

---

## Provenance

| Field | Value |
|---|---|
| Date (UTC) | 2026-05-25, runs 10:19Z–11:11Z (single thermal session, A/B interleaved). |
| Cluster | 4×Raspberry Pi 5 16GB. Root `rpi@192.168.1.74` (RPI-0005). Workers `.77` (RPI-0006), `.75` (RPI-0007), `.76` (RPI-0008). Gigabit LAN. |
| Worker order | `192.168.1.77 192.168.1.75 192.168.1.76` (production order). |
| CPU | Cortex-A76 ×4, **2400 MHz, no overclock** (`frequency(0)=2400030464`, verified all 4 nodes). |
| Governors | `performance` ×4 on root (verified). |
| Telemetry | **OFF** — `systemctl is-active docker` = `inactive` on root; **0** `hm_alloy`/`hm_cadvisor`/`hm_*` containers on any of the 4 nodes. Left off. Not reactivated at any point. |
| Throttle | `get_throttled` = `0x0` on `.74/.75/.77` throughout. `.76` shows `0xe0000` (historical bits 17/18/19 only — *past* under-voltage/throttle since boot; **no active low bits**, i.e. not throttling during the runs). Pre-existing flag, also present in the prior sweep log; not induced by this session. Live temps stayed 51–61 °C (well under the 80–85 °C throttle thresholds). |
| FORK binary (A) | `/home/rpi/dllama-custom/dllama` — 364,464 B. Optimized: A76 flags, `-flto`, SILU·MUL fusion, prefetch removal. sha256 `05e0ad218b0e1d32e425e6eb1f5c314d3e5992a170d102c231ba7891f84d314c` — **byte-identical on root + all 3 workers** (verified). Already present additively on all nodes; **no deploy needed**. |
| STOCK binary (B) | `/home/rpi/distributed-llama/dllama` — 302,704 B. Pristine upstream `b4rtaz/distributed-llama` @ `e0c59737e89184cc786598246901c9abd45c8972`. sha256 `36e4d0d84a8414014e98d42e0059046bf299f18537786e2b28c51126457db1fa` — **byte-identical on root + all 3 workers** (verified). |
| Model | `deepseek_r1_distill_llama_8b_q40` (Q40 dense 8B, Llama-3 arch), `dllama_model_deepseek-r1-distill-llama-8b_q40.m` = 6,323,781,792 B. Tokenizer `dllama_tokenizer_deepseek-r1-distill-llama-8b.t`. |
| Determinism | temperature 0, seed 42, `--buffer-float-type q80`, `--max-seq-len 4096`, `--steps 256`. |
| Prompt | `"Write a long continuous paragraph about distributed computing."` |
| Metric | CLI `dllama inference` **Prediction (decode) tok/s**, parsed `awk '/Prediction/{f=1} f&&/tokens/{print $2; exit}'`. |
| Worker runtime | jemalloc preload (`libjemalloc.so.2`, `MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000`), `Nice=-10`, `LimitMEMLOCK=infinity`, transient systemd unit `dllama-ds-worker`, port 9998. **Identical runtime env for both A and B** — only the binary path differs. |
| Method | For each config: stop production api (release workers) → (re)start all 3 workers with the chosen binary at the chosen nthreads → wait ≥6 s + verify all 3 :9998 listening → **2 warmups (48 steps, discarded) + 8 measured runs (256 steps)** on root with the same binary/nthreads. Each CLI run reloads+redistributes the model (~30–40 s) → every measured run is an independent cold-load steady-state decode. |
| Drift control | Both first passes showed mild monotone thermal decay within a block (~0.2 tok/s top→bottom). To prevent run-order from confounding the A/B, the two configs were **interleaved**: Pass 1 = A then B; Pass 2 = B then A. Pooling n=16 each cancels the order effect (see §4). |
| Harness | `/tmp/dsbench_ab_mac.sh <BIN> <NT> <NRUNS> <STEPS> <OUTFILE>` (run from the Mac; drives workers via `ssh -o StrictHostKeyChecking=accept-new` and runs the CLI on root). |

---

## 1. Config A — FORK (`/home/rpi/dllama-custom/dllama`), nthreads = 4

Raw per-run Prediction (decode) tok/s:
```
Pass 1 (A first):  8.21  8.17  8.16  8.14  8.10  8.11  8.09  8.03
Pass 2 (A second): 8.13  8.02  7.98  7.97  8.04  8.03  8.04  8.08
```
| stat | pass 1 (n=8) | pass 2 (n=8) | pooled (n=16) |
|---|---|---|---|
| mean | 8.126 | 8.036 | **8.081** |
| sample σ | 0.056 | 0.052 | 0.070 |
| 95% CI | [8.080, 8.173] | [7.993, 8.079] | **[8.044, 8.118]** |
| min / max | 8.03 / 8.21 | 7.97 / 8.13 | 7.97 / 8.21 |

## 2. Config B — STOCK (`/home/rpi/distributed-llama/dllama`), nthreads = 4

Raw per-run Prediction (decode) tok/s:
```
Pass 1 (B second): 8.17  8.16  8.10  8.08  8.08  8.06  7.96  8.07
Pass 2 (B first):  8.14  8.02  8.09  8.14  8.07  8.10  8.13  8.12
```
| stat | pass 1 (n=8) | pass 2 (n=8) | pooled (n=16) |
|---|---|---|---|
| mean | 8.085 | 8.101 | **8.093** |
| sample σ | 0.065 | 0.041 | 0.053 |
| 95% CI | [8.031, 8.139] | [8.067, 8.136] | **[8.065, 8.121]** |
| min / max | 7.96 / 8.17 | 8.02 / 8.14 | 7.96 / 8.17 |

(Re-confirms the prior log's stock nt4 = 8.360 *directionally* as the optimum; the absolute mean here
is ~3% lower because this whole session ran warmer — but A and B share that thermal window, which is
the point. Same-session A/B is internally valid regardless of the absolute offset.)

## 3. Fork nthreads sweep (to locate fork's optimum)

| binary | nthreads | n | mean | 95% CI | note |
|---|---|---|---|---|---|
| FORK | 3 | 8 | 7.826 | [7.805, 7.847] | `7.86 7.87 7.82 7.82 7.82 7.81 7.81 7.80` |
| FORK | 4 | 16 | **8.081** | [8.044, 8.118] | optimum (above) |
| STOCK | 3 | 8 | 7.201 | [7.185, 7.218] | from prior `cleanroom_sweep_2026-05-25.md` |
| STOCK | 4 | 16 | **8.093** | [8.065, 8.121] | optimum (above) |

- **Fork best = nt4** (8.081 > nt3 7.826, +3.26%). Hypothesis "fork→3" **not confirmed**; fork peaks at nt4 like stock.
- The fork's kernel/flag changes *help at nt3* (7.826 vs stock 7.201, +8.7%) but the gain **disappears at the nt4 optimum** → dense decode at full occupancy is bandwidth/saturation-bound, not compute-bound, so faster kernels yield ≈ 0.

## 4. A vs B verdict (pooled n=16 each, balanced interleaved)

```
A FORK  nt4: mean 8.081  95% CI [8.044, 8.118]
B STOCK nt4: mean 8.093  95% CI [8.065, 8.121]
Delta (A − B) = -0.012 tok/s = -0.15%
Welch t = -0.542, df ≈ 28.1, two-sided p ≈ 0.59
CIs overlap: YES (substantially)
```

Per-pass deltas: **Pass 1 (A first) = +0.51% for fork; Pass 2 (B first) = +0.81% for stock.**
The sign flips with run-order → the apparent difference is **thermal drift / order effect, not the
binary**. Pooled across the balanced design it collapses to **−0.15%, not significant**.

> **VERDICT: The fork's source/kernel changes contribute ≈ 0 to DENSE decode.**
> Fork-best (8.081, nt4) ≈ stock-best (8.093, nt4), within heavily overlapping 95% CIs and with a
> non-significant Welch test (p ≈ 0.59). For the paper's attribution table, the fork's dense-decode
> contribution should be reported as **0 (no measurable effect)**, with these clean-room numbers.

## 5. Caveats / honesty notes

- **Absolute offset vs prior log.** This session's stock nt4 mean (8.093) is ~3% below the earlier
  clean-room sweep (8.360) because the cluster ran warmer this session (steady 58–61 °C vs the
  earlier window). This does **not** affect the A/B conclusion — A and B were measured in the *same*
  thermal window, which is exactly why a same-session A/B was requested. If the paper wants the
  *absolute* dense-decode headline, keep using the cooler-window 8.360 from `cleanroom_sweep`; use
  *this* file only for the **relative fork-vs-stock attribution**.
- **Within-block drift.** ~0.2 tok/s monotone decay within each 8-run block; controlled by
  interleaving + pooling. Per-pass CIs are tighter but order-biased; trust the pooled n=16.
- **Node .76 `0xe0000`.** Historical throttle bits only (no active low bits), pre-existing, also in
  the prior log; live temps never approached the throttle threshold. Not believed to affect results.
- **No deploy was performed.** The fork was already byte-identical on all 4 nodes; verified by
  sha256, so the "deploy fork to workers" step was a no-op (binaries confirmed, not copied).
- **Stock nt3 = 7.201** is taken from the prior same-kernel clean-room log rather than re-run this
  session (it was not needed to establish the optimum, which is nt4 for both binaries).
