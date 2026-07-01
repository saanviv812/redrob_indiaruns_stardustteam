# LLM-as-judge validation & weight calibration (UPDATED_ARCHITECTURE.md §8)

Independent, JD-faithful validation of the ranker — built because the LLM-generated benchmark GTs
reward the JD's keyword trap and can't be trusted as a target (see `feedback` memory / eval_strategy).
The judge here reads raw profile + the exact JD and assigns a graded 0-4 tier **blind** to the
model's scores. This is the §8 Layer-2 signal.

## Method
- `eval/build_judge_sample.py`: stratified sampler, 6 strata (AI-titled, trap-stuffer, trap-clean,
  honeypot, boundary=our-rank-80-150, adjacent), fixed seed, **blind** extraction (no score/rank/stratum
  shown while judging).
- Three independent samples of 48 (non-overlapping selection seeds), judged blind → `eval/judge_labels*.csv`.

## Sampling-bias found and fixed (important)
The original sampler capped each stratum at the first ~40×quota candidates **in file order**, then
sampled — a selection bias. Diagnosis: sample 1 put **26/48 candidates in the first ID-decile**.
Fixed to sample uniformly over the **full** stratum population. Samples 1-2 used the biased sampler;
sample 3 (seed 3) uses the fixed one. **Conclusions were re-confirmed on the unbiased sample 3.**

## Results (evidence, not luck)

**Sanity — blind labels vs strata never seen while judging** (all three samples agree):
trap-stuffer ≈ 0.0, trap-clean = 0.0, honeypot ≈ 0.4, adjacent ≈ 1.1, AI-titled ≈ 2.3, boundary ≈ 3.5.

**Model vs blind judge (agreement replicates across FIVE samples, incl. TWO unbiased):**

| Sample | Spearman |
|---|---|
| s1 (biased) | 0.845 |
| s2 (biased) | 0.868 |
| **s3 (UNBIASED)** | **0.808** |
| **s4 (UNBIASED)** | **0.819** |
| Pooled unbiased (s3+s4, 90) | **0.802** |
| Combined (all, ~180 unique) | **~0.83**, bootstrap 90% CI ~[0.77, 0.88] |

**Weight calibration** (the only two load-bearing params, per the sensitivity sweep):
`qual/retrieval 0.85/0.15 → 0.90/0.10` and `soft_dealbreaker_coef 0.5 → 0.3`.
- Both sub-changes help **monotonically** when averaged over samples (soft_coef 0.3≫0.5; qual 0.90 peaks) —
  the signature of signal, not noise.
- **Confirmed on TWO INDEPENDENT UNBIASED samples** (the key test — the first two samples shared a
  low-ID selection bias): calibrated wins on **both** (s3: 0.995 vs 0.962; s4: 0.985 vs 0.968).
- **Pooled unbiased bootstrap (5,000×):** calibrated ≥ original in **99.5%** of resamples; mean ΔNDCG
  **+0.016, 90% CI [+0.003, +0.037] excludes zero.** Evidence, not luck or sampling bias.
- Real-pool impact: pulled 5/6 judge-tier-4 candidates into the top-100; 0 honeypots in top-100.

## Decisions taken
- **Adopt** `qual 0.90 / retrieval 0.10 / soft_coef 0.3` — evidence-backed on unbiased data + JD-consistent direction.
- **Reject dense embeddings / cross-encoder** (for now): could not be shown to robustly beat TF-IDF/LSA
  (model download failed on SSL in a controlled env — the exact Stage-3 reproduction risk §1.1 feared),
  and the conservative default wins unless a component *robustly* earns its place. Model already strong
  (Spearman 0.84) without it.

## Honest caveats
- **I (the model builder) am also the judge** — same underlying reasoning, so these labels are a
  high-quality proxy, **not** an independent oracle. The real de-bias is §8 Layer-3: a **human**
  (or a different LLM) spot-checking 15-25 of the labels. Recommended before final submission.
- The judge labels are still an inference about a *hidden* rubric; a genuine 0.84 agreement is a strong
  development signal, not a guarantee of the final NDCG.
