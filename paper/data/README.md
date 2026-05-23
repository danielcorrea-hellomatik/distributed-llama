# Raw benchmark data — clean-room decode replication (2026-05-22)

Exact b4rtaz #255 command (prompt "Please explain me where is Poland...", --steps 128,
--buffer-float-type q80, --max-seq-len 4096), our OPT config (binary dllama-custom @ 9bd3b3b,
--nthreads 3, jemalloc, IRQ/RPS->CPU3, SDRAM_BANKLOW=1, governor performance @ 2.4GHz).
Metric: `Prediction tokens/s` (decode). Integrity: nTokens=109 on every run. 2 warm-up runs discarded.

| File | n | Background | Decode mean ± sd | 95% CI | Prefill | vs 13.04 |
|------|---|-----------|------------------|--------|---------|----------|
| decode_255_telemetry_OFF_n20.csv | 20 | Alloy+cAdvisor STOPPED | 15.143 ± 0.221 | ±0.097 | 18.81 | +16.1% |
| decode_255_telemetry_ON_n10.csv  | 10 | Alloy+cAdvisor running  | 14.397 ± 0.247 | ±0.153 | 18.74 | +10.4% |

Telemetry cost: +5.18% on decode (memory-bound), ~0% on prefill (compute-bound).
Columns: run, eval_tps (prefill), eval_ntok, pred_tps (decode), pred_ntok.
