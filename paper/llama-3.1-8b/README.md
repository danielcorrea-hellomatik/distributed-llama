# Llama-3.1-8B-Instruct on 4× Raspberry Pi 5 — data point (not a standalone paper yet)

Llama-3.1-8B-Instruct (dense 8B, Q40) is the *baseline* in the companion MoE paper's optimisation
trajectory: it established the 5.70 tok/s dense starting point before the switch to Qwen3-30B-A3B
(MoE) drove the +59% jump. As a dense 8B model it falls in the **same memory-bandwidth-bound regime
as DeepSeek-R1-Distill-8B** (see `../deepseek-r1-distill-8b/`): software optimisation helps prefill,
not decode.

Status: present on the cluster (`~/distributed-llama/models/llama3_1_8b_instruct_q40`, 5.9 GB).
A full per-model report would mostly replicate the DeepSeek-8B findings (same dense regime); it is
folded into that paper's dense-vs-MoE contrast rather than written separately. Kept as a folder for
organisation and for a future head-to-head dense-8B comparison (DeepSeek vs Llama-3.1) if desired.
