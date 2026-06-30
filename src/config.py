"""Central configuration: paths, determinism, and every tunable coefficient.

Why this file exists
--------------------
`UPDATED_ARCHITECTURE.md` §7 ends with a "parameter provenance & calibration status"
table that classifies every magic number as VERIFIED / JD-TRACED / CONVENTION /
GUESS->CALIBRATE. We mirror that table here so that:

  1. All calibratable knobs live in ONE place (the §8 eval set tunes this file, nothing else).
  2. Every constant carries its provenance class inline, so at the Stage-5 interview we can
     answer "did the spec tell you to do that, or did you decide it?" honestly per-number.

Nothing in this module imports heavy dependencies — it is safe to import from anywhere,
including the time-budgeted rank.py.

Provenance classes (see UPDATED_ARCHITECTURE.md §7):
  VERIFIED        — measured against the real 100K data.
  JD-TRACED       — follows from a specific job_description.docx sentence.
  CONVENTION      — a documented standard default (e.g. RRF k=60).
  GUESS->CALIBRATE — a placeholder chosen by judgment; calibrate against the §8 eval set
                     or justify before final submission. Defended at Stage 5 as a choice.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Determinism — reproducibility is a hard requirement (spec §3, README).
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 20260702  # arbitrary fixed seed (module-close date); used everywhere RNG appears.

# ---------------------------------------------------------------------------
# Paths. Resolved relative to the repo root (parent of src/), overridable by env
# so the Stage-3 sandbox / a different checkout can relocate data without code edits.
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = Path(os.environ.get("REDROB_CACHE_DIR", REPO_ROOT / "cache"))

# Cache artifact filenames (written by precompute.py, read by rank.py). Kept as a dict so
# both sides reference the same names and a typo can't desync producer/consumer.
CACHE_FILES = {
    "candidate_ids": "candidate_ids.csv",          # ordering index for every .npy below
    "criteria_scores": "criteria_scores.npy",      # (N, n_criteria) float32
    "structural_features": "structural_features.npz",
    "dealbreakers": "dealbreakers.npz",
    "honeypots": "honeypots.npz",
    "fused_retrieval": "fused_retrieval_score.npy", # (N,) float32, rank-percentile normalized
    "reference_date": "reference_date.txt",         # corpus-anchored "now" (see §7 critical bug)
    "manifest": "manifest.json",                    # provenance / row counts / version stamp
}

# ---------------------------------------------------------------------------
# Honeypot gate (UPDATED_ARCHITECTURE.md §6, §16 item 2).
# ---------------------------------------------------------------------------
# VERIFIED thresholds (clean statistical separation in the real data):
HONEYPOT_EXP_INFLATION_YEARS: float = 2.5   # |yoe - sum(career months)/12| > 2.5y -> flag
HONEYPOT_EXPERT_SKILL_MIN_COUNT: int = 3    # >=3 "expert" skills with duration<=3mo -> flag
HONEYPOT_EXPERT_SKILL_MAX_DURATION_MONTHS: int = 3
# JD-TRACED: near-zero so a detected honeypot cannot mathematically reach the top-100 of 100K
# (>10% honeypots in top-100 is an auto-DQ; 0.2 was rejected in §16 as DQ-risky).
HONEYPOT_MULTIPLIER: float = 0.01

# Skill-assessment overclaim (VERIFIED percentiles of the real score distribution; §6).
ASSESSMENT_OVERCLAIM_SCORE_CUTOFF: float = 40.0  # claimed advanced/expert but score < 40 (~p25)
ASSESSMENT_HIGH_SCORE_CUTOFF: float = 66.0       # >= ~p75 -> positive corroboration

# ---------------------------------------------------------------------------
# Dealbreaker multipliers (UPDATED_ARCHITECTURE.md §1.3, §7 step 3).
# ---------------------------------------------------------------------------
# Structural/certain dealbreakers -> hard multiplicative gate.
HARD_DEALBREAKER_MULTIPLIER: float = 0.1   # GUESS->CALIBRATE (direction JD-traced, value a guess)
# Text-derived/less-certain dealbreakers -> soft penalty scaled by criterion similarity:
#   multiplier = 1 - SOFT_DEALBREAKER_COEF * criterion_score
# Chosen (not left at the rejected 0.7 placeholder): 0.5 keeps a max ~50% haircut even when the
# trap signal is at full strength, so a genuinely strong candidate with one weak soft-flag is not
# annihilated, while a clear trap is meaningfully suppressed. GUESS->CALIBRATE against §8.
SOFT_DEALBREAKER_COEF: float = 0.5

# JD's literal consulting-firm list (job_description.docx). HCL deliberately NOT included so the
# shipped feature matches the JD verbatim (§7 notes 7.0% with the literal 6, 7.6% incl. HCL).
CONSULTING_FIRMS = frozenset({
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
})

# ---------------------------------------------------------------------------
# Experience / availability / soft-preference layers (§7).
# ---------------------------------------------------------------------------
EXPERIENCE_BAND_YEARS = (5.0, 9.0)        # JD-TRACED ("a range not a requirement")
APPLIED_ML_YEARS_TARGET = (4.0, 5.0)      # JD-TRACED ("6-8 total, of which 4-5 in applied ML")
NOTICE_PERIOD_PREFERRED_MAX_DAYS: int = 30  # JD-TRACED ("We'd love sub-30-day notice")

# Availability multipliers, recency buckets in days (CONVENTION edges, GUESS->CALIBRATE values; §7).
# Mirrors the JD's own worked example (inactive + low response -> strong down-weight).
RECENCY_BUCKET_DAYS = (30, 90, 180)
AVAILABILITY_INACTIVE_RESPONSE_CUTOFF: float = 0.05  # near-dead branch (12/100k) — recheck post-anchor
AVAILABILITY_MULTIPLIERS = {
    "inactive_unresponsive": 0.3,  # >180d AND response_rate < cutoff
    "very_stale": 0.6,             # >90d
    "stale": 0.85,                 # >30d
    "active": 1.0,                 # <=30d
}

# Soft additive preferences — deliberately tiny so they cannot override a multiplicative gate
# (§16 item 4 rejected large additive bonuses that resurrect gated candidates). GUESS->CALIBRATE.
SOFT_PREF_LOCATION_BONUS: float = 0.02
SOFT_PREF_NOTICE_BONUS: float = 0.02
SOFT_PREF_EXPERIENCE_COEF: float = 0.03   # (experience_band_fit - 1.0) * coef

# ---------------------------------------------------------------------------
# Retrieval / text-match (§5). CONVENTION + GUESS->CALIBRATE per §7 table.
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES: int = 8000      # GUESS->CALIBRATE
TFIDF_NGRAM_RANGE = (1, 2)          # unigrams + bigrams
SVD_COMPONENTS: int = 128           # GUESS->CALIBRATE
RRF_K: int = 60                     # CONVENTION (original RRF paper default)
# Evidence blend within the semantic lane: entry >> skills (fights keyword-stuffing). §5.
EVIDENCE_BLEND = {"entry": 0.75, "skills": 0.15, "summary": 0.10}  # direction JD-traced, split a guess

# Final blend (§7 step 5). GUESS->CALIBRATE + depends on the resolved RRF normalization (§5 open Q).
QUALIFICATION_WEIGHT: float = 0.85
RETRIEVAL_WEIGHT: float = 0.15

# ---------------------------------------------------------------------------
# Submission constraints (spec §2-3) — referenced by rank.py and validation.
# ---------------------------------------------------------------------------
TOP_N: int = 100
REASONING_MAX_CHARS: int = 250  # submission_spec: 1-2 sentences; we cap hard at 250 chars.
