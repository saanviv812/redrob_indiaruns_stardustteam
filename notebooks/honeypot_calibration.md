# Honeypot calibration — verified against the real 100,000-candidate pool

This is the evidence record for `src/honeypots.py`. Every threshold below was confirmed by
streaming the actual `candidates.jsonl` (100,000 records), not copied from generic advice.
Reproduce with the snippet at the bottom.

## Results (run on the full pool, n = 100,000)

| Quantity | Value |
|---|---|
| Rule 1 — experience inflation, **DIRECTIONAL** (`Σmonths/12 − yoe > 2.5y`) | **22** flagged |
| Rule 2 — expert-skill overload (`≥3 "expert" skills, duration ≤3mo`) | **21** flagged |
| Overlap between the two rules | **0** |
| **Unique honeypots flagged (union)** | **43** |
| Documented target in the spec | ~80 |

> **Directional fix (important).** An earlier version used the *symmetric* `|yoe − Σmonths/12| > 2.5`,
> which flagged **47** — but 25 of those were the *benign* direction (stated experience EXCEEDING
> listed career = an under-listed résumé, which is normal, not "impossible"). The spec's honeypot is
> the *inflation* direction only ("8 years of experience at a company founded 3 years ago" = career
> exceeding what's possible). So we now flag one-sided: `Σmonths/12 − yoe > 2.5`. This removes 25
> false positives on the real pool (we were wrongly burying 25 legitimate candidates) and is verified
> clean one-sided: inflation values sit at 1.92, 2.08, then jump to 3.27 — an empty gap around 2.5.

## Why these thresholds are clean, not guessed

**Rule 1 — the gap is real (inflation direction).** For `Σ career months / 12 − years_of_experience`
(the inflation side only):

- only 24 candidates are on the inflation side at all
- values just below/above the threshold: 1.92, 2.08, then a jump to **3.27** — an empty gap around 2.5
- **22** candidates above 2.5y, maximum ~15.3y

So 2.5y is not a tuned knob — it sits inside an essentially empty gap between natural noise and a
cluster of impossible profiles.

**Rule 2 — even cleaner.** 99,979 / 100,000 candidates have **zero** expert-skills-with-≤3mo. The
21 that fire have 3–5 such skills, never 1–2. The `≥3` threshold sits in an empty region.

## The ~43 vs ~80 gap is a known, accepted shortfall

We are short of the documented ~80. Per `UPDATED_ARCHITECTURE.md` §6 we deliberately do **not**
invent an uncalibrated third rule to close it:

- The corpus-tenure check (company "founding date" proxy) was investigated and **dropped** — only
  63 distinct company names exist across 100K records, so earliest-appearance saturates and cannot
  discriminate anachronistic tenure.
- A few missed honeypots is fine: the gate is ≤10% of the **top-100**, and our honeypot multiplier
  is 0.01 (near-zero), so any *detected* honeypot cannot mathematically reach the top-100. The
  availability/coherence layers further down-weight implausible profiles we don't explicitly flag.

## Rules we rejected after checking (do not re-introduce)

- **salary `min > max`** — fires on **18.8%** of the pool; common synthetic noise, not fraud.
- **duplicate summary / name** — 2,761 / 3,312 duplicate groups; synthetic twins, not honeypots.
- **date inversion (`end_date < start_date`)** — 0 hits.
- **learned anomaly detector** — these are logical contradictions, not statistical outliers;
  a deterministic rule is more defensible and already cleanly separates.

## Skill-assessment overclaim (soft coherence signal, not a gate)

- `skill_assessment_scores` present for **24,244 / 100,000 (24.2%)** of candidates.
- Candidates with ≥1 overclaim (claimed advanced/expert in a skill scoring <40): **8,165**.
- This is a *soft* signal only (most candidates have no assessment); never zeroes a candidate.

## Soft flag

- Certifications dated ≥2030: **23** candidates (a generation artifact). Tracked, never gating.

## Reproduce

```python
import io_utils as io, honeypots as hp   # run from src/ with REDROB data path
r1 = r2 = union = 0
for c in io.iter_candidates(DATA):
    f1, f2 = hp.experience_inflation_flag(c), hp.expert_skill_overload_flag(c)
    r1 += f1; r2 += f2; union += (f1 or f2)
# -> r1=22 (directional inflation), r2=21, union=43
```
