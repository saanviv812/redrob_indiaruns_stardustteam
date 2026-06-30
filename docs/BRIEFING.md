# BRIEFING — problem context for the Redrob ranker

A one-page orientation for anyone (teammate or fresh session) picking this up. The full design
rationale lives in `../../UPDATED_ARCHITECTURE.md`; calibration evidence in `../notebooks/`.

## The task

Rank 100,000 synthetic candidate profiles (`candidates.jsonl`) against one fixed job description
(*Senior AI Engineer — Founding Team*, Redrob AI). Output the top-100 as a CSV
(`candidate_id,rank,score,reasoning`). Scored on `0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`
against a hidden ground truth revealed only after submissions close.

## Hard constraints (auto-DQ if violated)

- The ranking step (`rank.py`) must run in **≤5 min, ≤16 GB RAM, CPU-only, no network, ≤5 GB disk**.
- Offline precompute is unlimited and explicitly allowed — only `rank.py` is timed.
- **Honeypot rate in the top-100 must stay ≤10%** (pool has ~80 honeypots) or auto-DQ at Stage 3.

## What makes this hard (the traps, from the JD/spec)

- **Keyword-stuffers.** 68.8% of the pool hold one of 12 non-AI titles (Marketing Manager, etc.).
  Some carry AI skill tags with no supporting career narrative — "skills list full of AI keywords,
  title says Marketing Manager." Ranking on skill-tag count is the explicitly-built trap.
- **Honeypots (~80).** Subtly impossible profiles (e.g. "expert" in 10 skills used 0 months;
  career months wildly exceeding stated years). Forced to relevance tier 0 in the ground truth.
- **Behavioral availability.** A perfect-on-paper candidate who hasn't logged in for months with a
  low recruiter-response rate is "not actually available" — down-weight, don't reward.
- **Read between the lines.** A strong candidate may never write "RAG" or "Pinecone" but shows they
  built a recommendation system at a product company. Title + narrative > keyword presence.

## Key dataset facts (verified against the real data, in `notebooks/`)

- True AI/ML-titled candidates are rare (~0.5%); the JD says so explicitly ("10 great matches > 1000 maybes").
- Honeypots: 2 EDA-verified rules flag 68 with a clean statistical gap (the documented target is ~80).
- `skill_assessment_scores` (objective, 24.2% coverage) catches overclaimers — 33.7% of assessed
  candidates claim advanced/expert in a skill they score <40 on.
- Recency must be anchored to the corpus (`last_active_date` ranges 2025-09-29 … 2026-05-27), never
  `datetime.now()` — otherwise 30.6% falsely look "180+ days inactive" and results drift by run date.
- Several rules that *sound* right are wrong on this data: salary `min>max` (18.8% — noise, not fraud),
  duplicate summary/name (synthetic twins), corpus company-tenure (only 63 distinct company names).

## Evaluation honesty

We have **no ground truth**. Every relevance label we use is our own inference. The trustworthy
signal is that our ranker keeps honeypots and keyword-stuffers out of the top-100 (those classes are
spec-defined, not ours). We explicitly distrust any near-perfect NDCG on our own labels as circular.

## The five evaluation stages

1. Format validation (`validate_submission.py`) → 2. Scoring (hidden) → 3. Code reproduction +
honeypot check (sandboxed, our compute constraints) → 4. Manual review (reasoning quality, git
history authenticity, code quality) → 5. Defend-your-work interview. Build to survive all five.
