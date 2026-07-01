"""Structural feature extraction + dealbreaker detection (UPDATED_ARCHITECTURE.md §7).

Why this file exists
--------------------
This is the most defensible layer in the system: no text matching for the numeric features —
every value comes directly from dates, numbers, and enums. The dealbreaker detectors here are the
structural/certain ones (hard gates) plus the structural-soft one (closed-source), as opposed to
the text-derived soft dealbreakers (those are scored by the retrieval lanes against the queries in
jd_criteria.SOFT_DEALBREAKERS).

Outputs, per candidate, are plain scalars combined by precompute.py into cache arrays and consumed
by rank.py (§7 blend) and reasoning.py. Every output below is used downstream — no dead features.

Critical correctness rule (§7 "CRITICAL BUG TO AVOID"): recency is computed against a
corpus-anchored `reference_date` passed in by the caller, NEVER `datetime.now()`. A frozen dataset
+ a live clock would make 30.6% of the pool look "180+ days inactive" and break reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

from config import (
    APPLIED_ML_YEARS_TARGET,
    ASSESSMENT_HIGH_SCORE_CUTOFF,
    CONSULTING_FIRMS,
    EXPERIENCE_BAND_YEARS,
    NOTICE_PERIOD_PREFERRED_MAX_DAYS,
)
from io_utils import (
    full_text,
    get_career,
    get_profile,
    get_signals,
    get_skills,
    parse_date,
    safe_float,
)

# Pool-median recruiter_response_rate used when the signal is missing — "unknown" != "unresponsive"
# (§7 missing-data handling). 0.4 ≈ the documented median of the smooth 0.3–0.7 distribution.
DEFAULT_RESPONSE_RATE: float = 0.4

# AI/ML-relevant title keywords (lowercased substring match) — for applied-ML-years estimation.
# Deliberately excludes bare "engineer"/"developer"; "research scientist" is excluded because the
# JD treats pure-research as a dealbreaker, not applied-ML credit.
AIML_TITLE_KEYWORDS = (
    "machine learning", "ml engineer", "ai engineer", "data scientist", "applied scientist",
    "nlp", "deep learning", "research engineer", "ml scientist", "ai/ml", " ml ", " ai ",
)

# Seniority ladder for title_chaser detection. Higher = more senior.
SENIORITY_LADDER = (
    ("intern", 0), ("junior", 0), ("associate", 1),
    ("senior", 2), ("staff", 3), ("lead", 3), ("principal", 4),
    ("architect", 4), ("director", 5), ("vp", 5), ("head of", 5),
)

# Titles indicating a move away from hands-on coding (architect_drift). Near-inert on this dataset
# (no Architect/Director titles in the 47-title vocabulary) — kept as a documented rarely-firing gate.
ARCHITECT_DRIFT_TITLE_KEYWORDS = (
    "architect", "tech lead", "technical lead", "engineering manager",
    "head of", "director", "vp ", "vice president",
)

# Tokens that count as external validation (open-source / papers / talks) — used to NOT fire the
# closed-source dealbreaker. Absence of all of these + 5y+ exp + no GitHub -> closed-source signal.
VALIDATION_TOKENS = (
    "open source", "open-source", "opensource", "github", "oss ", "maintainer",
    "published", "paper", "publication", "talk", "conference", "patent",
    "speaker", "keynote", "blog", "kaggle",
)


@dataclass
class StructuralFeatures:
    """All structural outputs for one candidate. Field names == cache array names."""

    years_of_experience: float          # raw, for reasoning + experience fit
    experience_band_fit: float          # 1.0 inside [5,9]y, linear decay outside
    applied_ml_years_fit: float         # fit of est. applied-ML years against [4,5]y target
    notice_period_ok: float             # 1.0 if notice_period_days <= 30 else 0.0
    location_acceptable: float          # 1.0 if India or willing_to_relocate; else 0.0
    days_since_last_active: float       # vs corpus reference_date; -1.0 if unknown (neutral)
    recruiter_response_rate: float      # pool-median default if missing
    open_to_work: float                 # 1.0/0.0
    github_signal: float                # graded [0,1] = score/100 (0 if absent/-1); JD nice-to-have
    assessment_corroboration: float     # [0,1] high assessment on a JD-relevant-ish skill
    # Dealbreakers:
    consulting_only_career: float       # hard gate (1.0/0.0)
    title_chaser: float                 # hard gate (1.0/0.0)
    architect_drift: float              # hard gate (1.0/0.0)
    closed_source_strength: float       # structural soft dealbreaker strength [0,1]


def _seniority_rank(title: str) -> int:
    t = title.lower()
    rank = 1  # default mid-level if no keyword matches
    for kw, r in SENIORITY_LADDER:
        if kw in t:
            rank = max(rank, r)
    return rank


def experience_band_fit(yoe: float | None) -> float:
    """1.0 inside [5,9]y; linear decay to 0 over a 4-year window on each side (soft, never a gate)."""
    if yoe is None:
        return 1.0  # unknown experience is neutral, not penalized (§7 missing-data)
    lo, hi = EXPERIENCE_BAND_YEARS
    if lo <= yoe <= hi:
        return 1.0
    if yoe < lo:
        return max(0.0, 1.0 - (lo - yoe) / 4.0)
    return max(0.0, 1.0 - (yoe - hi) / 4.0)


def _applied_ml_years(cand: dict) -> float:
    """Estimate years in AI/ML-titled roles at non-consulting (product-proxy) companies."""
    months = 0
    for entry in get_career(cand):
        title = (entry.get("title") or "").lower()
        company = (entry.get("company") or "").lower()
        if company in CONSULTING_FIRMS:
            continue  # services firm, not a product company
        if any(kw in title for kw in AIML_TITLE_KEYWORDS):
            dm = entry.get("duration_months")
            if isinstance(dm, (int, float)):
                months += dm
    return months / 12.0


def applied_ml_years_fit(cand: dict) -> float:
    """Fit estimated applied-ML years against the JD's [4,5]y ideal; 1.0 in-band, linear decay.

    Below the band decays over 4y (a 1-year applied-ML candidate is still partially credited);
    above the band decays gently over 6y (more applied-ML experience is barely a negative).
    """
    yrs = _applied_ml_years(cand)
    lo, hi = APPLIED_ML_YEARS_TARGET
    if lo <= yrs <= hi:
        return 1.0
    if yrs < lo:
        return max(0.0, 1.0 - (lo - yrs) / 4.0)
    return max(0.0, 1.0 - (yrs - hi) / 6.0)


def _days_since_last_active(cand: dict, reference_date: date) -> float:
    """Days between last_active_date and the corpus-anchored reference date; -1.0 if unknown."""
    last = parse_date(get_signals(cand).get("last_active_date"))
    if last is None:
        return -1.0
    return float((reference_date - last).days)


def consulting_only_career(cand: dict) -> bool:
    """Hard gate: every career-history company is in the JD's literal consulting-firm list.

    The JD exempts candidates with prior product-company experience, which this captures exactly:
    a single non-consulting role makes this False. Requires >=1 known company to fire.
    """
    companies = [(e.get("company") or "").strip().lower() for e in get_career(cand)]
    companies = [c for c in companies if c]
    if not companies:
        return False
    return all(c in CONSULTING_FIRMS for c in companies)


def title_chaser(cand: dict) -> bool:
    """Hard gate: escalating seniority across >=3 jobs with short (<18mo) tenures (JD: ~1.5y hops).

    Conservative by design (gates are certain): requires all of —
      * >=3 career entries,
      * a net seniority climb of >=2 ladder levels from first to last role,
      * the majority of *completed* (non-current) roles held <18 months.
    """
    career = get_career(cand)
    if len(career) < 3:
        return False
    # career_history order: assume reverse-chronological is common, so sort by start_date ascending.
    entries = sorted(career, key=lambda e: (e.get("start_date") or ""))
    ranks = [_seniority_rank(e.get("title") or "") for e in entries]
    if ranks[-1] - ranks[0] < 2:
        return False
    completed = [e for e in entries if not e.get("is_current")]
    if not completed:
        return False
    short = sum(1 for e in completed
                if isinstance(e.get("duration_months"), (int, float)) and e["duration_months"] < 18)
    return short >= (len(completed) + 1) // 2  # majority (rounded up)


def architect_drift(cand: dict) -> bool:
    """Hard gate: current role is an architecture/management title with no recent coding evidence.

    Near-inert on this dataset (no such titles in the 47-title vocab) — documented in §15. Fires
    only when the *current* title matches a drift keyword AND its description lacks coding verbs.
    """
    current = None
    for e in get_career(cand):
        if e.get("is_current"):
            current = e
            break
    if current is None:
        return False
    title = (current.get("title") or "").lower()
    if not any(kw in title for kw in ARCHITECT_DRIFT_TITLE_KEYWORDS):
        return False
    desc = (current.get("description") or "").lower()
    coding_evidence = any(v in desc for v in ("implemented", "built", "coded", "developed", "wrote", "engineered"))
    return not coding_evidence


def closed_source_strength(cand: dict) -> float:
    """Structural soft dealbreaker strength (§7): 5y+ experience, no external validation, no GitHub.

    Returns 0.5 (a deliberately moderate strength, NOT 1.0) when it fires, because this is the
    least-certain and broadest-firing of the dealbreakers (64.6% of the pool has github=-1). At
    0.5 strength with the soft coefficient it is a gentle nudge, not a sledgehammer — appropriate
    for a signal that's really "absence of evidence." Returns 0.0 if any validation token appears,
    if GitHub is linked with a real score, or if experience < 5y.
    """
    yoe = safe_float(get_profile(cand).get("years_of_experience"), default=0.0)
    if yoe is None or yoe < 5.0:
        return 0.0
    gh = safe_float(get_signals(cand).get("github_activity_score"), default=-1.0)
    if gh is not None and gh >= 0.0:
        return 0.0  # has a linked GitHub with activity -> external validation exists
    text = full_text(cand).lower()
    if any(tok in text for tok in VALIDATION_TOKENS):
        return 0.0
    return 0.5


def assessment_corroboration(cand: dict) -> float:
    """[0,1] positive signal: high Redrob assessment scores corroborate claimed competence (§7).

    Returns the fraction (capped) of the candidate's *high-scoring* (>=66) assessed skills, scaled
    so that 2+ high assessments saturate. Absent assessments (the ~76%) return 0.0 (neutral). This
    is the objective, hardest-to-game positive counterpart to the overclaim penalty in honeypots.py.
    """
    scores = get_signals(cand).get("skill_assessment_scores")
    if not isinstance(scores, dict) or not scores:
        return 0.0
    high = sum(1 for v in scores.values()
               if isinstance(v, (int, float)) and v >= ASSESSMENT_HIGH_SCORE_CUTOFF)
    return min(1.0, high / 2.0)


def compute_features(cand: dict, reference_date: date) -> StructuralFeatures:
    """Compute all structural features + structural dealbreakers for one candidate."""
    prof = get_profile(cand)
    sig = get_signals(cand)
    yoe = safe_float(prof.get("years_of_experience"), default=None)

    notice = sig.get("notice_period_days")
    notice_ok = 1.0 if isinstance(notice, (int, float)) and notice <= NOTICE_PERIOD_PREFERRED_MAX_DAYS else 0.0

    country = (prof.get("country") or "").strip().lower()
    relocate = bool(sig.get("willing_to_relocate"))
    location_ok = 1.0 if (country == "india" or relocate) else 0.0

    rr = safe_float(sig.get("recruiter_response_rate"), default=None)
    response_rate = rr if rr is not None else DEFAULT_RESPONSE_RATE

    # Graded, not binary: proportional to the organizer-provided 0-100 score; absent/-1 -> 0 (neutral).
    gh = safe_float(sig.get("github_activity_score"), default=-1.0)
    github_signal = (max(0.0, min(100.0, gh)) / 100.0) if (gh is not None and gh >= 0.0) else 0.0

    return StructuralFeatures(
        years_of_experience=yoe if yoe is not None else 0.0,
        experience_band_fit=experience_band_fit(yoe),
        applied_ml_years_fit=applied_ml_years_fit(cand),
        notice_period_ok=notice_ok,
        location_acceptable=location_ok,
        days_since_last_active=_days_since_last_active(cand, reference_date),
        recruiter_response_rate=response_rate,
        open_to_work=1.0 if sig.get("open_to_work_flag") else 0.0,
        github_signal=github_signal,
        assessment_corroboration=assessment_corroboration(cand),
        consulting_only_career=1.0 if consulting_only_career(cand) else 0.0,
        title_chaser=1.0 if title_chaser(cand) else 0.0,
        architect_drift=1.0 if architect_drift(cand) else 0.0,
        closed_source_strength=closed_source_strength(cand),
    )


def feature_names() -> list[str]:
    """Ordered field names (== cache array keys)."""
    return list(asdict(StructuralFeatures(*([0.0] * 14))).keys())
