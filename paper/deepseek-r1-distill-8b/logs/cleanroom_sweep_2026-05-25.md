# DeepSeek-R1-Distill-8B — clean-room thread-count sweep (committed raw data)

> Produced to settle a reviewer dispute: paper claims decode **8.33 tok/s, nthreads=4 optimal**;
> the earlier `measurements.md` (§4) claimed **nthreads=3 optimal, nt4 regresses** due to NIC IRQ
> contention. This dataset re-measures under the **current** stack and reports raw per-run values.
>
> **Bottom line (n=8 each):** under the current kernel, **nthreads=4 is optimal (8.360 tok/s),
> +16.1% over nt3 (7.201)**. The 8.33 headline is supported. The earlier nt3-optimal log is
> stale (older kernel). The nt4 win is **NOT** attributable to RPS/IRQ steering — an A/B with RPS
> on vs off shows zero difference (+0.01%, Welch t=0.04). The win is simply the 4th compute thread.

---

## Provenance

| Field | Value |
|---|---|
| Date (UTC) | 2026-05-25T10:03:04Z |
| Cluster | 4×Raspberry Pi 5 16GB. Root `rpi@192.168.1.74` (hostname RPI-0005). Workers `.75`, `.76`, `.77`. Gigabit LAN. |
| CPU | Cortex-A76 ×4, 2400 MHz (no overclock; `arm_freq=2400`, `over_voltage=0`). |
| Kernel | `6.18.29+rpt-rpi-2712` (Debian 1:6.18.29-1+rpt1, built 2026-05-12) — **newer than when the old log was written**. |
| Governors | `performance` on all 4 cores. |
| Binary | `/home/rpi/distributed-llama/dllama` — **stock = pristine upstream** `b4rtaz/distributed-llama` @ `e0c59737e89184cc786598246901c9abd45c8972`, clean checkout (no local diffs). |
| Binary sha256 | `36e4d0d84a8414014e98d42e0059046bf299f18537786e2b28c51126457db1fa` (identical on all 4 nodes). |
| Build flags | Default Makefile: `-std=c++11 -march=native -mtune=native -O3`. **No** `-flto` / `-mcpu=cortex-a76` / `+dotprod` / kernel-fusion. |
| Worker runtime | jemalloc preload (`libjemalloc.so.2`, `MALLOC_CONF=narenas:4,tcache:true,dirty_decay_ms:30000`), `Nice=-10`, `LimitMEMLOCK=infinity`, transient systemd unit `dllama-ds-worker`, port 9998. |
| Worker order | `192.168.1.77 192.168.1.75 192.168.1.76` (same as production). |
| Model | `deepseek_r1_distill_llama_8b_q40` (Q40 dense 8B, Llama-3 arch). size 6,323,781,792 B (6.32 GB). model sha256 `9158b4ae9bbb5caca4e9…`. |
| Determinism | temperature 0, seed 42, `--buffer-float-type q80`, `--max-seq-len 4096`, `--steps 256`. |
| Prompt | `"Write a long continuous paragraph about distributed computing."` |
| Metric | CLI `dllama inference` **Prediction (decode) tok/s**, parsed `awk '/Prediction/{f=1} f&&/tokens/{print $2; exit}'`. |
| Telemetry | **OFF** — Docker `inactive` and 0 containers on all 4 nodes (`hm_alloy`/`hm_cadvisor` not running). Left off. |
| Throttle | `get_throttled` = `0x0` on `.74/.75/.77`; `0xe0000` on `.76` (historical bits 17/18/19 only — no active low bits; live temp 60°C). |
| Method | Each thread count: stop+restart all 3 workers at that nthreads, wait ≥5 s, then **2 warmups (48 steps) + 8 measured runs (256 steps)** at the SAME nthreads on root. Each CLI run reloads+redistributes the model (~30-40 s) before generating, so every measured run is an independent cold-load steady-state decode. |
| n | 8 measured runs per point. |
| Helper | `/tmp/dsbench_nt.sh <NT> 8 256` on root (writes `/tmp/nt_<NT>.txt`). |

---

## 1. Thread-count sweep — raw per-run Prediction (decode) tok/s

### nthreads = 2
```
5.36  5.37  5.36  5.37  5.36  5.36  5.36  5.35
```

### nthreads = 3
```
7.24  7.18  7.21  7.19  7.18  7.20  7.21  7.20
```

### nthreads = 4   (current/default RPS config: rps_cpus = 8 = CPU3)
```
8.47  8.38  8.42  8.37  8.38  8.25  8.30  8.31
```

### Statistics (sample stdev; 95% CI via t, df=7, t=2.365)

| nthreads | n | mean | sample σ | SEM | 95% CI | min | max |
|---|---|---|---|---|---|---|---|
| 2 | 8 | **5.361** | 0.006 | 0.002 | [5.356, 5.367] | 5.35 | 5.37 |
| 3 | 8 | **7.201** | 0.020 | 0.007 | [7.185, 7.218] | 7.18 | 7.24 |
| 4 | 8 | **8.360** | 0.071 | 0.025 | [8.301, 8.419] | 8.25 | 8.47 |

Deltas: nt3 vs nt2 **+34.3%** · **nt4 vs nt3 +16.1%** · nt4 vs nt2 +55.9%.

**nthreads = 4 is the optimum.** This contradicts the earlier `measurements.md` §4 (nt3 optimal,
nt4 regressing). See §3 for the reconciliation.

**Headline:** decode = **8.360 tok/s [8.301, 8.419]** (nt4, n=8). The paper's 8.33 tok/s falls
inside the CI → supported. Vs the 4×Pi5 record of 6.43 tok/s (#162) this is **+30.0%**.

---

## 2. RPS / NIC-IRQ A/B at nthreads = 4

The paper attributes the nt4 win to RPS steering the NIC softirq off the compute cores; the old
log attributed an nt4 *regression* to NIC IRQ contention. We tested it directly.

**Current NIC/IRQ config (root, eth0):**
- eth0 has a **single** rx queue (`rx-0`). `rps_cpus = 8` = binary `1000` = **CPU 3**.
- eth0 hardware IRQ is **112** (`rp1_irq_chip`), `smp_affinity = 8`, `smp_affinity_list = 3` → **CPU 3**.
- `/proc/interrupts`: virtually all eth0 interrupts land on **CPU 3** (137M on col-4 vs 56 on col-0).
- `dllama-runtime-tweaks` systemd unit: **does not exist / inactive** — the values above are the
  running default, not from a managed tweak service.

So RPS redistributes packet processing to **CPU 3**, which is exactly the CPU the hardware IRQ
already lands on. RPS is therefore effectively a **no-op** here (it "moves" softirq to the same
core). It does **not** steer the NIC *off* the compute cores; with nt4 the 4 compute threads span
CPU 0-3, so net-rx work shares CPU 3 with a compute thread in **both** RPS states.

**A/B (RPS cleared then restored):** `echo 0 | sudo tee .../rx-0/rps_cpus`, re-measure n=8, then
restore `rps_cpus = 8`. IRQ 112 affinity left untouched throughout.

### nthreads = 4, RPS OFF (rps_cpus = 0) — raw
```
8.44  8.37  8.33  8.41  8.28  8.36  8.35  8.33
```

| config | n | mean | sample σ | 95% CI |
|---|---|---|---|---|
| nt4, RPS ON (rps_cpus=8) | 8 | 8.360 | 0.071 | [8.301, 8.419] |
| nt4, RPS OFF (rps_cpus=0) | 8 | 8.359 | 0.050 | [8.317, 8.400] |

**Delta = +0.001 tok/s (+0.01%). Welch t = 0.04 → no significant difference.**

`rps_cpus` was **restored to 8** after the A/B (verified); IRQ 112 affinity unchanged (8).

**Verdict on nt3 → nt4 (honest):**
- The nt4 win is **real and large (+16.1%)** under the current kernel, and is **NOT** caused by
  RPS/IRQ steering — toggling RPS changes nothing (the RPS target == the IRQ core == CPU 3).
- The paper's *mechanism* claim ("RPS steers the IRQ off the compute cores, enabling nt4") is
  **not supported by the live config**: the NIC sits **on** a compute core (CPU 3) in both states,
  yet nt4 still wins. The 4th thread's compute gain simply dominates whatever net-rx contention
  exists on CPU 3.
- The most plausible explanation for the **reversal vs the old log** (which had nt4 regressing) is
  the **kernel upgrade to 6.18.29 (2026-05-12)** — scheduler / softirq handling changed such that a
  4th compute thread no longer loses to net-rx on the shared core. This is inferred, not isolated:
  we did not (and cannot safely) downgrade the kernel to A/B it. State this caveat in the paper.
- **Recommendation:** the paper should keep "nthreads=4 optimal, 8.33 tok/s" but **drop/soften the
  RPS-steering attribution** — replace it with "the 4th thread's compute gain outweighs net-rx
  contention under kernel ≥6.18; RPS is a measured no-op here (NIC IRQ and RPS both on CPU 3)."

---

## 3. Vanilla-current control (defuses the "+30% vs 2024 8GB/old-software" confound)

**Result: the control IS the headline.** The production "stock" binary at
`/home/rpi/distributed-llama/dllama` is **byte-identical** (sha256 `36e4d0d84a8414…`) to a clean
upstream clone at `/home/rpi/dllama-upstream/dllama`, both git `e0c5973` of
`b4rtaz/distributed-llama`, both clean checkouts with identical `src/` and the default Makefile.

The optimized/custom binaries (`dllama-custom` sha `05e0ad21…`, `dllama-kernel` sha `8d611ce2…`,
both ~364 KB) are **not** what served these measurements. So the entire §1 sweep — including the
8.360 tok/s headline — was measured on the **vanilla current upstream binary on 16GB boards.**

Therefore the vanilla-current control = **8.360 tok/s (nt4, n=8)**, identical to the headline.
No separate build was needed (and none was made). The +30% vs the 6.43 record is **not** a
software-optimization artifact on our side for dense decode — it is the **upstream binary on newer
16GB boards + newer kernel + nthreads=4**. This is consistent with the old log §2/§3 finding that
our custom source/runtime contributes ≈0% to *dense* decode (the gains there were prefill-only).

Caveat: the 6.43 record was on 8GB boards with 2024 software; this control isolates the *binary*
(upstream vs upstream → no difference) but cannot retroactively isolate board RAM vs kernel vs
upstream-version drift. What it **does** prove: the headline number is reproducible with a pristine
upstream binary, so the +30% is not attributable to bespoke kernel-fusion in our fork.

---

## 4. Health / restore (post-run)

- Temps after all runs: `.74` 59.3°C · `.75` 58.7°C · `.76` 60.4°C · `.77` 58.2°C. ARM clock 2400 MHz all. No active throttle.
- Docker inactive + 0 containers on all 4 nodes (telemetry stayed OFF).
- `rps_cpus` restored to 8; IRQ 112 affinity unchanged.
- DeepSeek restored to serving on :9997 at nthreads=4 via `model-switch.sh deepseek` (see report). Qwen left stopped.

## Caveats / anomalies

1. Absolute numbers differ from old `measurements.md` §4 (old nt3=7.49 vs here 7.20; old nt2=6.63
   vs here 5.36). Likely kernel/measurement-context drift; the **ordering** (nt4 > nt3 > nt2) is the
   load-bearing finding and is unambiguous (non-overlapping CIs).
2. nt4 noise is higher (σ 0.05-0.07) than nt2/nt3 (σ 0.006-0.02) — consistent with net-rx jitter on
   the shared CPU 3 — but the nt4 mean is still well above nt3 with non-overlapping CIs.
3. The RPS-vs-kernel attribution is inferred (kernel not downgraded). The RPS no-op finding is
   measured; the kernel-as-cause is the best-supported hypothesis, not a proven isolation.
