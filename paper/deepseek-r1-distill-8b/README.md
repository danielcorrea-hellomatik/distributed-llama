# DeepSeek-R1-Distill-8B on 4× Raspberry Pi 5 — paper

**Headline:** 8.32 tok/s **short-context** decode (clean-room: 4 threads/node, observability stack off, n=20, 95% CI [8.30, 8.34]),
4×Pi5 16GB, bit-exact, no overclock. New best published result for this model on Raspberry Pi 5:
**+29%** vs the prior 4×Pi5 record (6.43 tok/s, b4rtaz #162), **+37%** vs Seeed (6.06).
(Thread sweep n=8: nt2=5.36, nt3=7.20, nt4=8.36 — the nt4 point sits inside the n=20 CI. Mid-investigation 3-thread config: 7.591, n=20.)

**Decode falls with context (KV-cache):** from per-token timing, decode drops ~linearly with context position — `t(L)=117+0.0174·L` ms/token, **R²=0.999**, prompt-independent — from 8.4 tok/s (short) to ~5.0 at 5,000 tokens (−40%). The 8.32 headline is the short-context peak; a long reasoning trace averages ~6.2 tok/s. The 4-thread optimum holds at every context length (~13% over nt3 throughout). Data: `logs/megapaper/`.

**The honest twist (the actual contribution):** the production binary is byte-identical to clean
upstream `distributed-llama` (git e0c5973, SHA-256 match) — the 8.36 headline is reached by the
stock upstream binary itself, so our fork's source/runtime work adds ≈0 to *dense* decode (it adds
+86% to prefill). Dense decode is memory-bandwidth-bound (~10.1 of 17 GB/s/node,
87% of the per-token budget on-node), so software can't move it — the exact opposite of the
compute-bound MoE model in the companion report (Qwen3-30B-A3B, +15.2% from the same software).
That dense-vs-MoE contrast on identical silicon is the result.

## Thesis (layered optimisation)
Inference performance is a three-layer stack — **hardware/firmware · operating system · application software**.
Profile the binding constraint, then tune the layer it lives in. Here the constraint is memory bandwidth, so the
**+30%** comes from the **OS + hardware** layers (thread count, BANKLOW firmware, newer kernel, 16GB), not the application
kernels. We argue the *method* (not the prescription) should generalise to other inference software — GPU serving, detection models — with cross-stack validation left to future work; the
*paying layer* is workload-dependent (compute-bound → software; bandwidth/system-bound → OS + hardware).

## Files
- `main_en.tex` / `main_en.pdf` — paper (English, 12 pp).
- `main_es.tex` / `main_es.pdf` — Spanish translation (12 pp).
- `logs/measurements.md` — every raw measurement with provenance.
- `logs/paperbench_n20.log` — the authoritative n=20 decode benchmark (raw csv + stats).

## Reproduce
```bash
# record-comparable decode (plain, no spec), n=20:
bash deploy/scripts/paperbench.sh        # on the cluster root
# attribution vs clean upstream (A/B/C):  deploy/scripts/qd2.sh
# speculative decoding (bit-exact gate):  deploy/scripts/spectest.sh / specmulti.sh
# OS-lever A/B sweep:                      deploy/scripts/qdecode.sh
pdflatex main_en.tex && pdflatex main_en.tex
```

## Sections
Intro + layered thesis · record landscape · system design · optimisation stack & where it pays off
· methodology (bit-exact) · clean-room thread-count sweep · attribution A/B/C · prefill · bottleneck/roofline
· null OS levers · adaptive speculative decoding · dense-vs-MoE contrast · layered optimisation & generalisation · conclusion.
