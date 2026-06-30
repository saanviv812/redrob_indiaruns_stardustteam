# Evaluation strategy & results — honest, no ground truth (UPDATED_ARCHITECTURE.md §8)

**Foundational honesty: we have NO ground truth.** Redrob reveals scores only after submissions
close. Every label below is *our own inference*, differing only in confidence. Nothing here is
quoted as ground truth — this is a development compass, not a guarantee.

Run: `python eval/evaluate.py --candidates ./candidates.jsonl` (after `precompute.py`).

## The three layers

- **Layer 1 — our own anchors (automated, in `eval/evaluate.py`).** Honeypots (our detector, §6) →
  tier 0; keyword-stuffers (a trap-profession title + ≥3 AI skill tags, no narrative) → tier 0;
  off-target → 1; background → 2; AI/ML-title candidates → tier 3, strong ones (4–9y + GitHub>60 or
  assessment≥66) → tier 4. **Inference, not ground truth.**
- **Layer 2 — LLM-as-judge (offline, manual).** A strong LLM reads full profiles + the exact JD and
  assigns graded tiers to a stratified ~200–400 sample, blind to our model's outputs. This is the
  genuinely *independent* signal (holistic reading vs. our mechanical scorer; least likely to be
  fooled by keyword-stuffing). **Not run in this repo** — the dev environment is offline by design
  and the ranking step must be no-network; this is the documented next step before final submission.
- **Layer 3 — human spot-check (~30–45 min).** A teammate reads 15–25 of the LLM-judge's labels
  (high/low/borderline) to catch systematic LLM error. Manual; not code.

## Results — read the two kinds of number completely differently

### 1. Gross-error checks — TRUSTWORTHY (least circular)

Honeypots and keyword-stuffers are defined by Redrob's spec/JD, **not by our ranker**, so "our
ranker avoids them" is a real adversarial test, not self-grading.

| Check | Result | Target |
|---|---|---|
| Honeypots in top-100 | **0** | ≤10 (auto-DQ if >10) |
| Honeypots in top-10 | **0** | 0 |
| Honeypot score-percentile (all 68) | median **0.025**, max **0.034** | very low |
| Best (highest) honeypot rank | **96,572 / 100,000** | far from top-100 |
| Negative anchors (honeypot+stuffer, n=468) in top-100 | **0** | low |
| Strong-positive anchors score-percentile | median **0.994** | high |

The 0.01 honeypot multiplier buries all 68 honeypots in the bottom ~2.5% (final scores ≈ 0). This
is the result we actually stand behind.

### 2. NDCG/MAP over the anchored set — DO NOT TRUST THE ABSOLUTE NUMBER

`NDCG@10 = 1.0000, NDCG@50 = 0.978, MAP = 0.934, composite = 0.984` on the anchored set.

**We explicitly distrust this**, for the exact reason the architecture flags a perfect NDCG as a
red flag (one competitor self-reported 1.0000 — a leakage signal, not quality):

- It is **partially circular**: our tier-4 "strong positive" definition (AI/ML title + experience +
  GitHub/assessment) overlaps with what the ranker rewards, so a near-perfect ordering is almost
  tautological for the positive class.
- It is **stratified** (oversamples honeypots/stuffers/boundary), so it is *not* an unbiased
  estimate of true NDCG@10 on a random top-100.

We keep it for one purpose only: **relative change-tracking** (did a code change move this up or
down). The genuinely independent number we would trust is the Layer-2 LLM-judge, not run here.

## Why we do NOT train a learned ranker on these labels (§8)

No historical interaction data exists; every label is a proxy. Pseudo-labels from our own rules are
circular; the LLM-judge set is a few hundred rows against dozens of features (overfitting setup);
the anchors are sparse and mostly define the negative class. So the eval set *validates and lightly
calibrates a few coefficients* — it never trains a flexible model. The EDA-calibrated, certainty-
gated scorer remains the primary ranker (correct for a true cold-start, no-feedback problem).

## `sample_submission.csv` is a FORMAT REFERENCE ONLY

Per the spec: "It is not a high-quality ranking — it's only a format reference." Its trap-profession
composition is explained by base rate (those are 68.8% of the pool), not intent. We infer nothing
about per-candidate relevance from it; we use it solely to confirm our CSV format / validator pass.
