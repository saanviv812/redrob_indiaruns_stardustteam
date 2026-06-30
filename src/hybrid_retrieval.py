"""Lexical lane (BM25) + Reciprocal Rank Fusion (UPDATED_ARCHITECTURE.md §5).

Why this file exists
--------------------
TF-IDF/LSA (text_match.py) captures fuzzy semantic overlap; BM25 captures exact lexical hits the
latent space can blur. We fuse the two with Reciprocal Rank Fusion (RRF) rather than blending raw
scores, because BM25 is unbounded/corpus-dependent while LSA cosine is bounded-but-compressed —
RRF sidesteps the score-incompatibility problem by operating on ranks, needing no arbitrary
normalization between the lanes.

Resolving §5's open question (RRF output is small positive floats, not in [0,1])
--------------------------------------------------------------------------------
DECISION: normalize the fused RRF score by **rank-percentile** over the candidate pool to land in
[0,1] before rank.py blends it (the 0.85/0.15 blend). Rationale: (a) it is monotonic in the fused
RRF score, so it preserves the fusion's ordering exactly; (b) it is robust to RRF's compressed,
non-linear scale (min-max would let a handful of high-RRF outliers dominate the [0,1] range and
flatten everyone else); (c) it is deterministic and parameter-free. This is a design choice, not
an organizer mandate — defended as above at Stage 5.

Runs only inside precompute.py (offline). rank.py consumes the cached, already-normalized vector.
"""

from __future__ import annotations

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi
from scipy.stats import rankdata

from config import RRF_K
from jd_criteria import MUST_HAVES, criterion_index
from jd_text import JD_FULL_TEXT

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by BM25 corpus and query."""
    return _TOKEN_RE.findall(text.lower())


def compute_bm25_scores(full_texts: list[str]) -> np.ndarray:
    """Raw BM25 score of each candidate's full_text against the JD query (one query, N docs)."""
    corpus = [tokenize(t) for t in full_texts]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(JD_FULL_TEXT))
    return np.asarray(scores, dtype=np.float32)


def _rrf_component(scores: np.ndarray) -> np.ndarray:
    """RRF contribution 1/(k + rank) for one lane. Rank 1 = highest score (ties share avg rank)."""
    # rankdata on -scores: largest score -> smallest rank value (1.0). Average ties (deterministic).
    ranks = rankdata(-scores, method="average")
    return 1.0 / (RRF_K + ranks)


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    """Map values to [0,1] by rank-percentile (resolves §5: stable, monotonic normalization)."""
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    ranks = rankdata(values, method="average")  # 1..n, higher value -> higher rank
    return ((ranks - 1.0) / (n - 1.0)).astype(np.float32)


def compute_fused_retrieval(criteria_scores: np.ndarray, full_texts: list[str]) -> np.ndarray:
    """Fuse the semantic must-have signal with BM25 via RRF; return [0,1] rank-percentile vector.

    semantic_overall = mean of the 4 must-have criterion columns (the JD's "absolutely need" set),
    so the fusion is anchored on the requirements that actually decide fit, not the nice-to-haves.
    """
    idx = criterion_index()
    must_cols = [idx[c.id] for c in MUST_HAVES]
    semantic_overall = criteria_scores[:, must_cols].mean(axis=1)

    bm25_scores = compute_bm25_scores(full_texts)
    logger.info("BM25 scored %d candidates (max=%.3f, mean=%.3f)",
                len(bm25_scores), float(bm25_scores.max()), float(bm25_scores.mean()))

    fused = _rrf_component(semantic_overall) + _rrf_component(bm25_scores)
    return _rank_percentile(fused)
