"""Offline orchestrator: stream the pool once, build every cache artifact (UPDATED_ARCHITECTURE.md §12 step 1).

Why this file exists
--------------------
Everything expensive (TF-IDF/LSA fit, BM25, structural extraction over 100K records) happens here,
OUTSIDE the 5-minute ranking budget. precompute.py writes cache/ ; rank.py only loads it. This is
the architectural split that makes the timed step trivially fast (§3).

It is explicitly allowed to take as long as it needs and to use more memory than rank.py — only the
ranking step is budgeted (spec §3; README "pre-computation may exceed the 5-minute window").

Run:
    python precompute.py --candidates /path/to/candidates.jsonl

Determinism: the only stochastic component is TruncatedSVD, pinned to RANDOM_SEED. Same input file
=> byte-identical cache.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date

import numpy as np

import io_utils as io
from config import CACHE_DIR, RANDOM_SEED
from features import compute_features
from jd_criteria import MATCHED_CRITERIA
from honeypots import assessment_overclaim_strength, cert_2030_soft_flag, is_honeypot
from hybrid_retrieval import compute_fused_retrieval
from text_match import compute_criteria_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute")

# Feature fields routed to dealbreakers.npz (the rest go to structural_features.npz).
_DEALBREAKER_FIELDS = ("consulting_only_career", "title_chaser", "architect_drift", "closed_source_strength")


def compute_reference_date(path: str) -> date:
    """Pass A: corpus-anchored "now" = max(last_active_date). NEVER datetime.now() (§7 critical bug)."""
    ref: date | None = None
    for cand in io.iter_candidates(path):
        d = io.parse_date(io.get_signals(cand).get("last_active_date"))
        if d is not None and (ref is None or d > ref):
            ref = d
    if ref is None:
        raise ValueError("No parseable last_active_date in the corpus; cannot anchor recency.")
    return ref


def run(candidates_path: str) -> None:
    t_start = time.time()
    logger.info("Pass A: computing corpus-anchored reference date ...")
    reference_date = compute_reference_date(candidates_path)
    logger.info("reference_date = %s", reference_date.isoformat())

    # Pass B: stream once, collecting text views + structural/honeypot features.
    ids: list[str] = []
    entry_texts_flat: list[str] = []
    entry_owner: list[int] = []
    skills_texts: list[str] = []
    summary_texts: list[str] = []
    full_texts: list[str] = []

    feat_cols: dict[str, list[float]] = None  # type: ignore[assignment]
    hp_is: list[float] = []
    hp_overclaim: list[float] = []
    hp_cert2030: list[float] = []

    logger.info("Pass B: extracting text + structural/honeypot features ...")
    for i, cand in enumerate(io.iter_candidates(candidates_path)):
        ids.append(cand["candidate_id"])
        for txt in io.entry_texts(cand):
            entry_texts_flat.append(txt)
            entry_owner.append(i)
        skills_texts.append(io.skills_text(cand))
        summary_texts.append(io.summary_text(cand))
        full_texts.append(io.full_text(cand))

        feats = compute_features(cand, reference_date)
        fd = feats.__dict__
        if feat_cols is None:
            feat_cols = {k: [] for k in fd}
        for k, v in fd.items():
            feat_cols[k].append(v)

        hp_is.append(1.0 if is_honeypot(cand) else 0.0)
        hp_overclaim.append(assessment_overclaim_strength(cand))
        hp_cert2030.append(1.0 if cert_2030_soft_flag(cand) else 0.0)

        if (i + 1) % 20000 == 0:
            logger.info("  ... %d candidates processed (%.1fs)", i + 1, time.time() - t_start)

    n = len(ids)
    n_entries = len(entry_texts_flat)
    logger.info("Pass B complete: %d candidates, %d career entries (%.1fs)",
                n, n_entries, time.time() - t_start)

    # --- Semantic lane (TF-IDF + LSA). Free the per-text lists it consumes afterward. ---
    owner_arr = np.asarray(entry_owner, dtype=np.int64)
    t = time.time()
    criteria_scores = compute_criteria_scores(entry_texts_flat, owner_arr, n, skills_texts, summary_texts)
    logger.info("criteria_scores %s computed (%.1fs)", criteria_scores.shape, time.time() - t)
    del entry_texts_flat, entry_owner, owner_arr, skills_texts, summary_texts

    # --- Lexical lane + RRF fusion. Free full_texts afterward. ---
    t = time.time()
    fused = compute_fused_retrieval(criteria_scores, full_texts)
    logger.info("fused_retrieval computed (%.1fs)", time.time() - t)
    del full_texts

    # --- Persist cache. ---
    io.ensure_cache_dir()
    io.save_candidate_ids(ids)
    io.save_array("criteria_scores", criteria_scores)
    io.save_array("fused_retrieval", fused)
    io.save_reference_date(reference_date)

    structural = {k: np.asarray(v, dtype=np.float32) for k, v in feat_cols.items() if k not in _DEALBREAKER_FIELDS}
    dealbreakers = {k: np.asarray(feat_cols[k], dtype=np.float32) for k in _DEALBREAKER_FIELDS}
    io.save_npz("structural_features", **structural)
    io.save_npz("dealbreakers", **dealbreakers)
    io.save_npz(
        "honeypots",
        is_honeypot=np.asarray(hp_is, dtype=np.float32),
        overclaim_strength=np.asarray(hp_overclaim, dtype=np.float32),
        cert_2030=np.asarray(hp_cert2030, dtype=np.float32),
    )

    manifest = {
        "n_candidates": n,
        "n_career_entries": n_entries,
        "reference_date": reference_date.isoformat(),
        "criteria_columns": [c.id for c in MATCHED_CRITERIA],
        "structural_fields": sorted(structural.keys()),
        "dealbreaker_fields": list(_DEALBREAKER_FIELDS),
        "honeypot_count": int(sum(hp_is)),
        "random_seed": RANDOM_SEED,
        "precompute_seconds": round(time.time() - t_start, 1),
    }
    io.save_manifest(manifest)
    logger.info("Cache written to %s in %.1fs. Honeypots flagged: %d",
                CACHE_DIR, time.time() - t_start, int(sum(hp_is)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the Redrob ranker cache.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl(.gz)")
    args = parser.parse_args()
    run(args.candidates)


if __name__ == "__main__":
    main()
