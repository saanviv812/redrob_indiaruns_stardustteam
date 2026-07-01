# Redrob Ranker — Intelligent Candidate Discovery & Ranking Challenge (Track 1)

Ranks the 100,000-candidate pool (`candidates.jsonl`) against the *Senior AI Engineer — Founding
Team* job description and emits the top-100 as a spec-compliant CSV.

**Design philosophy:** a deterministic, EDA-calibrated, certainty-gated scoring system where every
weight traces to a sentence in the JD (defensible live at Stage 5), all heavy work is precomputed
offline, and the timed ranking step is pure cache-load + vectorized arithmetic. Full rationale is in
[`../UPDATED_ARCHITECTURE.md`](../UPDATED_ARCHITECTURE.md); calibration evidence is in `notebooks/`.

## Reproduce

```bash
pip install -r requirements.txt

# 1. Precompute (offline, ~10 min, unbudgeted) — builds cache/ from the candidate file.
python src/precompute.py --candidates /path/to/candidates.jsonl

# 2. Rank (the BUDGETED step: <5 min, <16 GB, CPU-only, no network) — produces the CSV.
python src/rank.py --candidates /path/to/candidates.jsonl --out submission.csv
```

The single Stage-3 reproduction command is step 2 (`rank.py`); step 1 is the documented
pre-computation that may exceed the 5-minute window (spec §10.3). Both run CPU-only, offline.

## Measured against the real 100K pool

| What | Result |
|---|---|
| `precompute.py` wall-clock | ~577 s (offline, unbudgeted) |
| `rank.py` wall-clock | **~19 s** (scoring 0.5 s; rest is one file re-stream for reasoning) |
| Peak memory (`rank.py`) | well under 16 GB (cache is 14 MB; arrays are float32) |
| Cache disk footprint | **0.014 GB** (budget 5 GB) |
| `validate_submission.py` | **PASS** |
| Honeypots in top-100 | **0** (auto-DQ if >10) |
| Trap-profession titles in top-100 | **0** |
| Compliance (`src/compliance_check.py`) | **COMPLIANT** (no network, no GPU, disk OK) |

## How it works (one screen)

1. **`jd_criteria.py`** — 17-criterion checklist (4 must-have, 5 nice-to-have, 4 text-derived
   soft-dealbreakers), each carrying the verbatim JD sentence that justifies its weight.
2. **`text_match.py`** — TF-IDF + Truncated SVD (LSA), matched at the *career-entry* level with
   max-evidence per criterion (structurally suppresses skills-list keyword-stuffing). No transformer
   weights → deterministic and reproduction-safe.
3. **`hybrid_retrieval.py`** — BM25 lexical lane fused with the semantic lane via Reciprocal Rank
   Fusion, normalized to [0,1] by rank-percentile.
4. **`features.py`** — structural facts from dates/numbers/enums (experience fit, applied-ML years,
   availability, GitHub/assessment positives) + the structural dealbreakers. Recency is anchored to
   the corpus (`max(last_active_date)`), never the wall clock — a confirmed reproducibility bug we
   avoid.
5. **`honeypots.py`** — two EDA-verified hard rules (clean statistical separation) → near-zero
   (×0.01) multiplier, plus a soft skill-assessment overclaim coherence signal.
6. **`rank.py`** — loads cache, applies the §7 blend (honeypot gate → dealbreaker gates →
   weighted qualification → 0.85/0.15 retrieval blend → availability → tiny additive prefs), sorts
   (tie-break `candidate_id` ascending), takes top 100, attaches reasoning, writes the CSV.
7. **`reasoning.py`** — fragment-pool generation: extracts atomic facts from real fields and varies
   wording/order per candidate (deterministic), so reasonings are specific, honest, and non-templated.

## Evaluation (`eval/`)

We have **no ground truth** (revealed only after submissions close), so every label is our own
inference. The trustworthy signal is the *negative* class — honeypots/keyword-stuffers are defined
by the spec, not by us — and our ranker excludes them cleanly. We deliberately **distrust** the
near-perfect NDCG over our own anchors (it's partially circular). See `notebooks/eval_strategy.md`.

```bash
python eval/evaluate.py --candidates /path/to/candidates.jsonl
python -m unittest discover -s tests -v        # 23 tests
python src/compliance_check.py                 # no-network / no-GPU / disk proof
```

## Sandbox demo (`app.py`) — required by spec §10.5

A one-file Streamlit app: upload a ≤100-candidate JSONL sample → the **full pipeline** (offline
precompute + CPU-only, no-network ranking) runs → shows the ranked candidates + reasoning and a
downloadable CSV. Runs in ~1–2s on a small sample (well within the ≤5-min CPU budget).

Run locally: `streamlit run app.py`

**Deploy the required hosted link (free):**
- **Streamlit Community Cloud** (simplest — deploys straight from this GitHub repo): share.streamlit.io
  → "New app" → pick this repo → main file `app.py` → Deploy. It installs `requirements.txt` and serves.
- **HuggingFace Spaces**: create a Streamlit Space, push this repo's contents, `app_file: app.py`.

(`streamlit` is listed separately in `requirements.txt` and is **not** imported by the ranker — the
Stage-3 reproduction of `rank.py` needs only the ranking-pipeline block.)

## Repo layout

```
src/         jd_criteria, aliases, jd_text, text_match, hybrid_retrieval, features,
             honeypots, reasoning, precompute, rank, compliance_check, config, io_utils
eval/        metrics.py, evaluate.py
tests/       test_ranker.py (stdlib unittest)
notebooks/   honeypot_calibration.md, dealbreaker_feature_calibration.md, eval_strategy.md
cache/       generated by precompute.py (gitignored)
requirements.txt, submission_metadata.yaml, README.md
```

## AI tool usage

This system was built with Claude's assistance (architecture discussion, code, EDA). Declared
honestly in `submission_metadata.yaml`. No candidate data was sent to any hosted LLM; the ranking
step makes zero network calls.
