"""Deterministic, EDA-calibrated honeypot detection (UPDATED_ARCHITECTURE.md §6).

Why this file exists
--------------------
The spec embeds ~80 honeypots (subtly impossible profiles) and disqualifies any submission with
>10% honeypots in its top-100 (Stage 3). We detect them with two hard logical-consistency rules
whose thresholds were calibrated against the real 100K data (clean statistical gaps, not guesses),
plus one objective overclaim coherence signal and one tracked-but-non-gating soft flag.

We deliberately do NOT use a learned anomaly detector (§6 "Explicitly Rejected"): these are clean
logical contradictions, trivially defensible live, and a deterministic rule already produced a
demonstrated clean separation. We also never use salary min>max (fires on 18.8% of the pool —
common noise, not fraud; §6) nor text/identity dedup (thousands of synthetic twins; §16 item 1).

Rules (both calibrated in notebooks/honeypot_calibration.md):
  Rule 1 — Experience inflation:  |yoe - sum(career_history.duration_months)/12| > 2.5y
  Rule 2 — Expert-skill overload: >=3 skills with proficiency=="expert" AND duration_months<=3
A candidate is a honeypot if EITHER rule fires.

Overclaim coherence signal (soft, NOT a gate — only ~24% of candidates have assessments):
  claimed advanced/expert in a skill but its Redrob assessment score < 40  -> coherence penalty.
"""

from __future__ import annotations

from typing import Any

from config import (
    ASSESSMENT_OVERCLAIM_SCORE_CUTOFF,
    HONEYPOT_EXP_INFLATION_YEARS,
    HONEYPOT_EXPERT_SKILL_MAX_DURATION_MONTHS,
    HONEYPOT_EXPERT_SKILL_MIN_COUNT,
)
from io_utils import get_profile, get_signals, get_skills, safe_float

_CLAIMED_HIGH = {"advanced", "expert"}


def experience_inflation_flag(cand: dict) -> bool:
    """Rule 1: stated years_of_experience vs summed career-history months differ by > 2.5y.

    DIRECTIONAL (fixed): only career INFLATION — summed career history EXCEEDING stated experience
    by >2.5y — is the "subtly impossible" pattern the spec names ("8 years of experience at a
    company founded 3 years ago"). The opposite direction (stated experience exceeding listed
    career, i.e. an under-listed resume) is perfectly normal and NOT a honeypot; flagging it is a
    false positive. Verified against the real pool: the inflation side has a clean empty gap
    (values at 1.92, 2.08, then a jump to 3.27 — nothing between), and it is the ONLY side that
    corresponds to impossible profiles. An earlier symmetric `abs(...)` version false-flagged 25
    legitimate under-listed candidates on the real pool (and 18k+ on LLM-generated stress sets).
    Missing years_of_experience -> no flag (missing data never gates — §7).
    """
    prof = get_profile(cand)
    yoe = safe_float(prof.get("years_of_experience"), default=None)
    if yoe is None:
        return False
    total_months = 0
    for entry in cand.get("career_history") or []:
        dm = entry.get("duration_months")
        if isinstance(dm, (int, float)):
            total_months += dm
    career_years = total_months / 12.0
    return (career_years - yoe) > HONEYPOT_EXP_INFLATION_YEARS


def expert_skill_overload_flag(cand: dict) -> bool:
    """Rule 2: >=3 skills claimed "expert" with <=3 months of stated duration.

    Real experts accrue duration; claiming expert mastery in several skills used for <=3 months is
    an objective inconsistency. 99,979/100,000 candidates have zero such skills; the flagged set
    has 3-5, never 1-2 — another clean separation.
    """
    count = 0
    for skill in get_skills(cand):
        prof = skill.get("proficiency")
        dm = skill.get("duration_months")
        if prof == "expert" and isinstance(dm, (int, float)) and dm <= HONEYPOT_EXPERT_SKILL_MAX_DURATION_MONTHS:
            count += 1
            if count >= HONEYPOT_EXPERT_SKILL_MIN_COUNT:
                return True
    return False


def is_honeypot(cand: dict) -> bool:
    """A candidate is a honeypot if EITHER hard rule fires."""
    return experience_inflation_flag(cand) or expert_skill_overload_flag(cand)


def _normalize_skill_name(name: Any) -> str:
    return name.strip().lower() if isinstance(name, str) else ""


def assessment_overclaim_strength(cand: dict) -> float:
    """Soft coherence signal in [0,1]: how strongly the candidate overclaims vs objective tests.

    For each skill the candidate claims at advanced/expert proficiency, if that skill has a Redrob
    assessment score below 40 (~p25 of the real distribution), it is an objective overclaim. We
    return min(1.0, n_overclaimed / 3) — 3+ overclaimed skills saturates the signal. Candidates
    with no assessments (the ~76%) return 0.0 (strictly neutral; §6 mandatory rule).

    This is NOT a honeypot gate; rank.py folds it into the soft-penalty stack.
    """
    signals = get_signals(cand)
    scores = signals.get("skill_assessment_scores")
    if not isinstance(scores, dict) or not scores:
        return 0.0
    # Normalize assessment keys once for case-insensitive matching against skill names.
    norm_scores = {_normalize_skill_name(k): v for k, v in scores.items()}
    n_overclaimed = 0
    for skill in get_skills(cand):
        if skill.get("proficiency") not in _CLAIMED_HIGH:
            continue
        key = _normalize_skill_name(skill.get("name"))
        if not key or key not in norm_scores:
            continue
        score = safe_float(norm_scores[key], default=None)
        if score is not None and score < ASSESSMENT_OVERCLAIM_SCORE_CUTOFF:
            n_overclaimed += 1
    return min(1.0, n_overclaimed / 3.0)


def cert_2030_soft_flag(cand: dict) -> bool:
    """Soft flag (tracked, never zeroing): certification dated 2030 — a known data artifact (§6)."""
    for cert in cand.get("certifications") or []:
        year = cert.get("year")
        if isinstance(year, int) and year >= 2030:
            return True
    return False
