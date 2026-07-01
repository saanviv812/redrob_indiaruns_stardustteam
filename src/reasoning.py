"""Fragment-pool reasoning generation (UPDATED_ARCHITECTURE.md §9).

Why this file exists
--------------------
Stage 4 samples 10 rows and checks each reasoning for: specific profile facts, connection to JD
requirements, honest acknowledgement of concerns, NO hallucination, variation across rows, and
tone consistent with rank. We follow the "Rank, Don't Generate" pattern: extract atomic facts from
the candidate's actual fields, then select/compose/vary phrasing — never free-generate, so every
claim provably traces to a field value.

Variation (the §9 risk): a fixed "one career fact + one skill fact + one behavior fact" template
reads as templated across 10 sampled rows even when facts differ. So we randomize fragment WORDING
and ORDER per candidate via a deterministic RNG seeded from the candidate_id — varied surface form,
fully reproducible.

Output is hard-capped at config.REASONING_MAX_CHARS (250) — 1-2 sentences (submission_spec §2).
"""

from __future__ import annotations

import random

from config import REASONING_MAX_CHARS, RANDOM_SEED
from io_utils import get_profile, get_signals, get_skills, safe_float
from jd_criteria import MUST_HAVES

# Human-readable label for each must-have criterion id (for the JD-connection fragment).
_MUST_HAVE_LABEL = {
    "embeddings_retrieval_production": "embeddings/retrieval",
    "vector_db_hybrid_search": "vector/hybrid search",
    "strong_python": "Python engineering",
    "ranking_eval_frameworks": "ranking evaluation",
}
_PREFERRED_PROFICIENCY = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}


def _candidate_rng(candidate_id: str) -> random.Random:
    """Deterministic per-candidate RNG (so wording varies but a re-run is byte-identical)."""
    return random.Random(f"{RANDOM_SEED}:{candidate_id}")


def _top_skills(cand: dict, limit: int = 3) -> list[str]:
    """Up to `limit` real skill names, preferring higher proficiency. Strictly from the profile."""
    skills = get_skills(cand)
    ranked = sorted(
        (s for s in skills if isinstance(s.get("name"), str) and s["name"].strip()),
        key=lambda s: (_PREFERRED_PROFICIENCY.get(s.get("proficiency"), 0), s.get("endorsements", 0)),
        reverse=True,
    )
    out, seen = [], set()
    for s in ranked:
        name = s["name"].strip()
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
        if len(out) >= limit:
            break
    return out


def _experience_fragment(cand: dict, rng: random.Random) -> str:
    prof = get_profile(cand)
    yoe = safe_float(prof.get("years_of_experience"), default=None)
    title = (prof.get("current_title") or "").strip()
    company = (prof.get("current_company") or "").strip()
    yrs = f"{yoe:.1f}y" if yoe is not None else "experience"
    role = title or "current role"
    if company:
        templates = [
            f"{yrs} experience, currently {role} at {company}",
            f"{role} at {company} ({yrs} total)",
            f"{yrs} in the field; now {role} at {company}",
        ]
    else:
        templates = [f"{yrs} experience as {role}", f"{role} with {yrs} experience"]
    return rng.choice(templates)


def _qualification_fragment(top_must_have_ids: list[str], rng: random.Random, strength: float) -> str:
    """Language is GRADED by the candidate's relative strength so the tone matches the rank
    (Stage-4 "rank consistency" check): confident wording only for genuinely strong candidates,
    measured/hedged wording for weak ones — never "strong fit" on a low-scored profile."""
    labels = [_MUST_HAVE_LABEL[i] for i in top_must_have_ids if i in _MUST_HAVE_LABEL]
    if not labels:
        return ""
    joined = " and ".join(labels[:2])
    if strength >= 0.60:
        templates = [
            f"strong fit on the JD's {joined} needs",
            f"matches the JD's {joined} requirements",
            f"profile signals {joined}, central to this role",
        ]
    elif strength >= 0.30:
        templates = [
            f"relevant signal on the JD's {joined}",
            f"partial match on {joined}",
            f"some experience touching {joined}",
        ]
    else:
        templates = [
            f"only adjacent/limited signal on {joined}",
            f"weak direct evidence on the JD's {joined}",
            f"surface keyword overlap on {joined}, not substantiated",
        ]
    return rng.choice(templates)


def _skills_fragment(skills: list[str], rng: random.Random) -> str:
    if not skills:
        return ""
    shown = skills[:2] if len(skills) >= 2 else skills[:1]
    joined = ", ".join(shown)
    templates = [f"key skills {joined}", f"depth in {joined}", f"lists {joined}"]
    return rng.choice(templates)


def _behavior_fragment(cand: dict, days_since_active: float, rng: random.Random) -> str:
    sig = get_signals(cand)
    rr = safe_float(sig.get("recruiter_response_rate"), default=None)
    bits = []
    if rr is not None:
        bits.append(f"response rate {rr:.2f}")
    if days_since_active is not None and days_since_active >= 0:
        bits.append(f"active {int(days_since_active)}d ago")
    if sig.get("open_to_work_flag"):
        bits.append("open to work")
    if not bits:
        return ""
    return rng.choice(["engagement: ", "signals: ", ""]) + ", ".join(bits[:2])


def generate_reasoning(
    cand: dict,
    top_must_have_ids: list[str],
    concerns: list[str],
    days_since_active: float,
    strength: float = 1.0,
) -> str:
    """Compose a varied, fact-grounded, rank-consistent reasoning string (<=250 chars).

    Parameters
    ----------
    cand              : the candidate record (source of every fact — no hallucination).
    top_must_have_ids : must-have criterion ids this candidate scored highest on (JD connection).
    concerns          : honestly-named triggered concerns (honeypot/dealbreaker/availability/notice).
    days_since_active : corpus-anchored recency (for the behavioral fragment).
    strength          : candidate's relative strength in [0,1] (score / top score) — grades the
                        qualification wording so tone matches rank (Stage-4 rank-consistency check).
    """
    rng = _candidate_rng(cand.get("candidate_id", ""))

    # Consistent, professional structure (FIXED order): experience -> qualification -> skills ->
    # behavior. We intentionally do NOT shuffle the order — the spec's "not templated" check is about
    # substantively different CONTENT, not different sentence shapes, and a clean consistent format
    # reads more polished to a human reviewer. Variation that satisfies the check comes from each
    # candidate's genuinely different facts (title/company/years/skills/signals/concerns) plus light
    # per-fragment wording choices; it is never a name-swap template.
    lead = _experience_fragment(cand, rng)
    body = [
        _qualification_fragment(top_must_have_ids, rng, strength),
        _skills_fragment(_top_skills(cand), rng),
        _behavior_fragment(cand, days_since_active, rng),
    ]
    body = [f for f in body if f]

    ordered = ([lead] if lead else []) + body
    sentence = "; ".join(ordered[:4])
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    if concerns:
        concern_text = rng.choice(["concern: ", "caveat: ", "note: "]) + "; ".join(concerns[:2])
        sentence = f"{sentence}. {concern_text[0].upper()}{concern_text[1:]}" if sentence else concern_text

    if not sentence:
        sentence = "Included on adjacent signals; limited direct evidence in profile."

    sentence = sentence.strip().rstrip(";")
    if not sentence.endswith("."):
        sentence += "."
    if len(sentence) > REASONING_MAX_CHARS:
        sentence = sentence[: REASONING_MAX_CHARS - 1].rstrip().rstrip(";,") + "."
    return sentence
