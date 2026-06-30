"""rank.py — the ONLY budgeted script (<=5min, <=16GB, CPU, no network). UPDATED_ARCHITECTURE.md §7.

Why this file exists
--------------------
This is the deliverable the grader reproduces. It does NO heavy computation: it loads the cache
produced by precompute.py and applies fast vectorized arithmetic (the §7 blend), sorts, takes the
top 100, attaches fact-grounded reasoning, and writes the submission CSV. No network, no GPU, no
model loading — by construction it cannot violate the compute constraints.

Pipeline (§7 step list):
  honeypot gate -> dealbreaker gates (hard + soft) -> qualification score -> blend retrieval
  -> availability multiplier -> soft additive preferences -> sort (tie-break candidate_id asc)
  -> top 100 -> reasoning -> CSV.

Run:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
(precompute.py must have been run on the same candidates file first; cache/ is consumed here.)
"""

from __future__ import annotations

import argparse
import csv
import logging
import time

import numpy as np

import io_utils as io
from config import (
    AVAILABILITY_INACTIVE_RESPONSE_CUTOFF,
    AVAILABILITY_MULTIPLIERS,
    HARD_DEALBREAKER_MULTIPLIER,
    HONEYPOT_MULTIPLIER,
    NOTICE_PERIOD_PREFERRED_MAX_DAYS,
    QUALIFICATION_WEIGHT,
    RECENCY_BUCKET_DAYS,
    RETRIEVAL_WEIGHT,
    SOFT_DEALBREAKER_COEF,
    SOFT_PREF_EXPERIENCE_COEF,
    SOFT_PREF_LOCATION_BONUS,
    SOFT_PREF_NOTICE_BONUS,
    TOP_N,
)
from jd_criteria import MUST_HAVES, NICE_TO_HAVES, SCORED_CRITERIA, criterion_index
from reasoning import generate_reasoning

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rank")

_IDX = criterion_index()
_SOFT_DB_PLAIN = ("pure_research_only", "langchain_tutorial_only", "framework_enthusiast")
_NLP_IR_COLS = [_IDX["embeddings_retrieval_production"], _IDX["vector_db_hybrid_search"],
                _IDX["ranking_eval_frameworks"]]


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------
def load_cache() -> dict:
    cache = {
        "ids": io.load_candidate_ids(),
        "criteria": io.load_array("criteria_scores"),
        "fused": io.load_array("fused_retrieval"),
        "structural": io.load_npz("structural_features"),
        "dealbreakers": io.load_npz("dealbreakers"),
        "honeypots": io.load_npz("honeypots"),
        "reference_date": io.load_reference_date(),
    }
    n = len(cache["ids"])
    assert cache["criteria"].shape[0] == n, "criteria_scores row count != candidate_ids"
    assert cache["fused"].shape[0] == n, "fused_retrieval row count != candidate_ids"
    return cache


# ---------------------------------------------------------------------------
# Scoring (§7 blend), fully vectorized
# ---------------------------------------------------------------------------
def compute_scores(cache: dict) -> dict:
    crit = cache["criteria"].astype(np.float64)
    struct = cache["structural"]
    db = cache["dealbreakers"]
    hp = cache["honeypots"]
    n = crit.shape[0]

    # 1. honeypot gate.
    honeypot_mult = np.where(hp["is_honeypot"] > 0.5, HONEYPOT_MULTIPLIER, 1.0)

    # 2. hard dealbreaker gate: 0.1 per fired structural gate (multiplicative).
    n_hard = db["consulting_only_career"] + db["title_chaser"] + db["architect_drift"]
    hard_mult = np.power(HARD_DEALBREAKER_MULTIPLIER, n_hard)

    # 3. soft dealbreaker penalties (each: *= 1 - COEF*strength).
    soft_mult = np.ones(n, dtype=np.float64)
    for db_id in _SOFT_DB_PLAIN:
        strength = np.clip(crit[:, _IDX[db_id]], 0.0, 1.0)
        soft_mult *= 1.0 - SOFT_DEALBREAKER_COEF * strength
    # cv/speech/robotics only penalized WITHOUT significant NLP/IR exposure (JD wording):
    nlp_ir = crit[:, _NLP_IR_COLS].max(axis=1)
    cv_strength = np.clip(crit[:, _IDX["cv_speech_robotics_only"]] - nlp_ir, 0.0, 1.0)
    soft_mult *= 1.0 - SOFT_DEALBREAKER_COEF * cv_strength
    # structural-soft closed-source + skill-assessment overclaim coherence penalty:
    soft_mult *= 1.0 - SOFT_DEALBREAKER_COEF * np.clip(db["closed_source_strength"], 0.0, 1.0)
    soft_mult *= 1.0 - SOFT_DEALBREAKER_COEF * np.clip(hp["overclaim_strength"], 0.0, 1.0)

    dealbreaker_mult = hard_mult * soft_mult

    # 4. qualification score = weighted must+nice / total weight.
    total_w = sum(c.weight for c in SCORED_CRITERIA)
    qual = np.zeros(n, dtype=np.float64)
    for c in SCORED_CRITERIA:
        qual += c.weight * crit[:, _IDX[c.id]]
    qual /= total_w

    # 5. blend in retrieval (already rank-percentile normalized to [0,1] in precompute).
    base = QUALIFICATION_WEIGHT * qual + RETRIEVAL_WEIGHT * cache["fused"].astype(np.float64)

    # 6. availability multiplier (corpus-anchored recency; -1 days == unknown -> neutral 1.0).
    days = struct["days_since_last_active"]
    resp = struct["recruiter_response_rate"]
    d30, d90, d180 = RECENCY_BUCKET_DAYS
    availability_mult = np.select(
        [
            (days > d180) & (resp < AVAILABILITY_INACTIVE_RESPONSE_CUTOFF),
            days > d90,
            days > d30,
        ],
        [
            AVAILABILITY_MULTIPLIERS["inactive_unresponsive"],
            AVAILABILITY_MULTIPLIERS["very_stale"],
            AVAILABILITY_MULTIPLIERS["stale"],
        ],
        default=AVAILABILITY_MULTIPLIERS["active"],
    )

    # 7. small additive soft preferences + positive corroboration (kept tiny; §16 item 4).
    soft_pref = (
        SOFT_PREF_LOCATION_BONUS * struct["location_acceptable"]
        + SOFT_PREF_NOTICE_BONUS * struct["notice_period_ok"]
        + SOFT_PREF_EXPERIENCE_COEF * (struct["experience_band_fit"] - 1.0)
        + 0.03 * struct["assessment_corroboration"]   # objective positive (§7)
        + 0.02 * struct["github_positive"]            # real OSS signal (§7)
    ).astype(np.float64)

    # 8. final score. Soft pref multiplied by honeypot_mult so a flagged honeypot can NEVER be
    #    lifted back toward the top-100 by additive bonuses (preserves the §16 0.01-gate guarantee).
    final = base * dealbreaker_mult * honeypot_mult * availability_mult + soft_pref * honeypot_mult

    return {
        "final": final, "qual": qual, "base": base,
        "dealbreaker_mult": dealbreaker_mult, "honeypot_mult": honeypot_mult,
        "availability_mult": availability_mult, "cv_strength": cv_strength,
    }


def select_top(cache: dict, scores: dict, top_n: int = TOP_N) -> list[int]:
    """Indices of the top_n candidates, sorted by score desc then candidate_id asc (validator rule)."""
    final = scores["final"]
    ids = cache["ids"]
    # Sort key: (-score, candidate_id). lexsort is stable and sorts by the LAST key primarily.
    order = sorted(range(len(final)), key=lambda i: (-final[i], ids[i]))
    return order[:top_n]


# ---------------------------------------------------------------------------
# Reasoning support — concerns + top must-haves for the selected candidates
# ---------------------------------------------------------------------------
def _top_must_have_ids(crit_row: np.ndarray, k: int = 2) -> list[str]:
    cols = [(_IDX[c.id], c.id) for c in MUST_HAVES]
    ranked = sorted(cols, key=lambda ci: crit_row[ci[0]], reverse=True)
    return [cid for _, cid in ranked[:k] if crit_row[_IDX[cid]] > 0.05]


def _concerns(cand: dict, cache: dict, scores: dict, i: int) -> list[str]:
    db, hp, struct = cache["dealbreakers"], cache["honeypots"], cache["structural"]
    concerns: list[str] = []
    if hp["is_honeypot"][i] > 0.5:
        concerns.append("profile has internal inconsistencies (honeypot-flagged)")
    if db["consulting_only_career"][i] > 0.5:
        concerns.append("consulting-only career (JD dealbreaker)")
    if db["closed_source_strength"][i] > 0.0:
        concerns.append("little external validation (no OSS/papers/GitHub)")
    if hp["overclaim_strength"][i] > 0.0:
        concerns.append("some skill claims exceed assessment scores")
    if scores["cv_strength"][i] > 0.05:
        concerns.append("CV/speech-leaning with limited NLP/IR signal")
    days = struct["days_since_last_active"][i]
    if days is not None and days > RECENCY_BUCKET_DAYS[1]:
        concerns.append(f"low recent activity ({int(days)}d since active)")
    notice = io.get_signals(cand).get("notice_period_days")
    if isinstance(notice, (int, float)) and notice > NOTICE_PERIOD_PREFERRED_MAX_DAYS:
        concerns.append(f"notice period {int(notice)}d")
    return concerns


def fetch_records(candidates_path: str, wanted_ids: set[str]) -> dict[str, dict]:
    """Stream the candidate file once, capturing only the records we need for reasoning (top-N)."""
    found: dict[str, dict] = {}
    for cand in io.iter_candidates(candidates_path):
        cid = cand.get("candidate_id")
        if cid in wanted_ids:
            found[cid] = cand
            if len(found) == len(wanted_ids):
                break
    return found


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def write_submission(out_path: str, rows: list[dict]) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.6f}", r["reasoning"]])


def run(candidates_path: str, out_path: str, with_reasoning: bool = True) -> list[dict]:
    t0 = time.time()
    cache = load_cache()
    logger.info("Loaded cache: %d candidates (%.2fs)", len(cache["ids"]), time.time() - t0)

    scores = compute_scores(cache)
    top = select_top(cache, scores)
    logger.info("Scored + selected top %d (%.2fs)", len(top), time.time() - t0)

    ids = cache["ids"]
    top_ids = {ids[i] for i in top}
    records = fetch_records(candidates_path, top_ids) if with_reasoning else {}
    missing = top_ids - set(records)
    if with_reasoning and missing:
        raise RuntimeError(
            f"{len(missing)} top candidate_id(s) not found in {candidates_path} — cache/file "
            f"mismatch. Re-run precompute.py on this exact file. Example missing: {sorted(missing)[:3]}"
        )

    crit = cache["criteria"]
    rows: list[dict] = []
    for rank, i in enumerate(top, start=1):
        cid = ids[i]
        if with_reasoning:
            cand = records[cid]
            reasoning = generate_reasoning(
                cand,
                _top_must_have_ids(crit[i]),
                _concerns(cand, cache, scores, i),
                float(cache["structural"]["days_since_last_active"][i]),
            )
        else:
            reasoning = ""
        rows.append({"candidate_id": cid, "rank": rank, "score": float(scores["final"][i]), "reasoning": reasoning})

    write_submission(out_path, rows)
    logger.info("Wrote %s (%d rows) in %.2fs total", out_path, len(rows), time.time() - t0)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates and write the submission CSV.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl(.gz)")
    parser.add_argument("--out", required=True, help="Output submission CSV path")
    parser.add_argument("--no-reasoning", action="store_true", help="Skip reasoning (faster; debug only)")
    args = parser.parse_args()
    run(args.candidates, args.out, with_reasoning=not args.no_reasoning)


if __name__ == "__main__":
    main()
