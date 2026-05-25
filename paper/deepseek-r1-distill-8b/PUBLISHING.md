# Publishing checklist — DeepSeek-R1-8B paper

Status: paper revised (peer-reviewed twice, 8/10 → tech-report ready), EN 12 pp / ES 13 pp,
bit-exact, raw data committed in `logs/`. This is the step-by-step to put it online.

## 0. Before upload (one-time)
- [ ] Add your **ORCID** to `CITATION.cff` (line is there, commented `# orcid:`). If you don't have one: https://orcid.org/register (2 min, free).
- [ ] Pick the headline framing for sharing: **lead with the finding** (memory wall / dense-vs-MoE / layered optimisation), NOT "+30% record". The honesty is the selling point; a "+30% record" headline invites the stale-baseline objection.

## 1. Zenodo (gets you a DOI — do this first)
1. https://zenodo.org → New upload.
2. Drag in: `main_en.pdf`, `main_es.pdf`, `main_en.tex`, `main_es.tex`, the whole `logs/` folder, `CITATION.cff`, `LICENSE`.
3. Zenodo reads `.zenodo.json` automatically (title, authors, description, keywords, CC-BY-4.0, the companion DOI as *isPartOf*). Check the fields, set license = **Creative Commons Attribution 4.0**.
4. Under *Related/alternate identifiers*, confirm the companion DOI `10.5281/zenodo.20357376` is linked as "is part of".
5. **Publish** → Zenodo mints a DOI (e.g. `10.5281/zenodo.NNNNNNN`).
6. **Back-fill the DOI**: replace the placeholder `10.5281/zenodo.0000000` in `CITATION.cff`, and (optional) add it to the paper title block. Re-upload a new version if you edit the PDF.

## 2. arXiv (cs.DC) — optional, needs endorsement
- Blocked on the cs.DC endorsement (campaign already in flight — see `~/Desktop/arXiv-endorsement/`).
- Once endorsed: submit `main_en.tex` (+ no external figures, all inline pgfplots), category **cs.DC** (cross-list cs.LG), abstract from the paper, and add the Zenodo DOI in the comments.

## 3. Canonical hub (SEO rule: one canonical page)
- [ ] Publish a `hellomatik.com/research/...` page as the **canonical** home (per the SEO/backlink rule). Link out from there; don't scatter links.
- [ ] Point the Zenodo record and any posts back to the hub.

## 4. Share (with the honest framing)
- distributed-llama discussion (the DeepSeek-8B thread #162 / a new one) — most relevant audience.
- r/LocalLLaMA, Hacker News (edge-AI crowd).
- Frame: *"Honest, bit-exact characterisation of dense-8B decode on 4×Pi5 — it's bandwidth-bound, so OS/config tuning (not kernels) gets you +30%; here's the dense-vs-MoE contrast that proves it."*

## Files in this artifact
- `main_en.{tex,pdf}` / `main_es.{tex,pdf}` — the report (EN/ES).
- `logs/cleanroom_sweep_2026-05-25.md` — committed n=8 thread sweep (the headline's raw data).
- `logs/cleanroom_attribution_2026-05-25.md` — clean-room fork-vs-stock attribution.
- `logs/measurements.md` — full measurement log (old nthreads sweep marked superseded).
- `logs/paperbench_n20.log` — the n=20 mid-investigation benchmark.
- `CITATION.cff`, `LICENSE` (CC-BY-4.0), `.zenodo.json`, `README.md`.
