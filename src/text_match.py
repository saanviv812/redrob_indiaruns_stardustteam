"""Semantic lane: TF-IDF + Truncated SVD (LSA) CriterionMatcher (UPDATED_ARCHITECTURE.md §5).

Why this file exists
--------------------
This is the primary relevance signal. We chose TF-IDF + LSA over transformer embeddings (§1.1):
it reuses the scikit-learn dependency we already need, bundles zero model weights, runs CPU-only,
and is fully deterministic (fixed random_state) — all of which de-risk the Stage-3 sandbox reproduction.

Design (§1.2, §5):
  * Matching happens at the career-ENTRY level, not whole-profile, then we take each candidate's
    best-matching entry per criterion ("max evidence"). This structurally suppresses the
    skills-list-only keyword-stuffing trap, because skills contribute only a minor 0.15 weight.
  * Alias expansion (aliases.py) is applied to entry and summary text and to the criterion queries
    — never to the raw skills list (the §5 guard).
  * Final per-criterion score = 0.75*max_entry + 0.15*skills + 0.10*summary.

Output columns align 1:1 with jd_criteria.MATCHED_CRITERIA (9 scored + 4 text soft-dealbreakers).
This module runs only inside precompute.py (offline, unbudgeted) — never inside rank.py.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from aliases import expand
from config import EVIDENCE_BLEND, RANDOM_SEED, SVD_COMPONENTS, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from jd_criteria import MATCHED_CRITERIA

logger = logging.getLogger(__name__)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize; zero rows stay zero (cosine with them is 0, i.e. 'no evidence')."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def compute_criteria_scores(
    entry_texts_flat: list[str],
    entry_owner: np.ndarray,
    n_candidates: int,
    skills_texts: list[str],
    summary_texts: list[str],
) -> np.ndarray:
    """Score every candidate against every matched criterion via the TF-IDF+LSA semantic lane.

    Parameters
    ----------
    entry_texts_flat : flattened career-entry texts across ALL candidates (alias-expanded here).
    entry_owner      : int array, len == len(entry_texts_flat); candidate row index owning each entry.
    n_candidates     : N (defines output rows).
    skills_texts     : per-candidate skills text (len N); NOT alias-expanded (the §5 guard).
    summary_texts    : per-candidate summary text (len N); alias-expanded here.

    Returns
    -------
    (N, n_criteria) float32 matrix in [0,1], columns ordered as jd_criteria.MATCHED_CRITERIA.
    """
    assert len(entry_owner) == len(entry_texts_flat), "entry_owner must align with entry_texts_flat"
    assert len(skills_texts) == n_candidates and len(summary_texts) == n_candidates

    criterion_texts = [expand(c.text) for c in MATCHED_CRITERIA]
    n_crit = len(criterion_texts)

    # Alias-expand entry + summary text (NOT skills). Empty strings are allowed (zero vectors).
    entries_exp = [expand(t) for t in entry_texts_flat]
    summary_exp = [expand(t) for t in summary_texts]

    # --- Fit TF-IDF on the full candidate corpus + criterion queries (shared vocabulary). ---
    fit_corpus = entries_exp + skills_texts + summary_exp + criterion_texts
    # min_df=2 drops hapax noise across 100K docs, but on a tiny sample (the ≤100-candidate sandbox
    # demo) it can empty the vocabulary — so relax to 1 for small corpora.
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        lowercase=True,
        sublinear_tf=True,        # dampens repeated terms (standard for relevance)
        min_df=2 if len(fit_corpus) >= 200 else 1,
    )
    vectorizer.fit(fit_corpus)
    logger.info("TF-IDF vocabulary: %d terms", len(vectorizer.vocabulary_))

    entry_tfidf = vectorizer.transform(entries_exp)
    skills_tfidf = vectorizer.transform(skills_texts)
    summary_tfidf = vectorizer.transform(summary_exp)
    crit_tfidf = vectorizer.transform(criterion_texts)

    # --- Fit LSA (TruncatedSVD) on the candidate text (entries dominate the latent space). ---
    from scipy.sparse import vstack as sp_vstack

    fit_matrix = sp_vstack([entry_tfidf, skills_tfidf, summary_tfidf])
    # TruncatedSVD needs n_components < min(n_samples, n_features). Cap it so the tiny-sample sandbox
    # demo (≤100 candidates) doesn't crash; full-pool runs are unaffected (cap >> SVD_COMPONENTS there).
    n_comp = max(1, min(SVD_COMPONENTS, fit_matrix.shape[0] - 1, fit_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
    svd.fit(fit_matrix)
    logger.info("LSA explained variance (sum of %d comps): %.3f",
                SVD_COMPONENTS, float(svd.explained_variance_ratio_.sum()))

    crit_lsa = _l2_normalize(svd.transform(crit_tfidf).astype(np.float32))       # (n_crit, k)
    entry_lsa = _l2_normalize(svd.transform(entry_tfidf).astype(np.float32))     # (E, k)
    skills_lsa = _l2_normalize(svd.transform(skills_tfidf).astype(np.float32))   # (N, k)
    summary_lsa = _l2_normalize(svd.transform(summary_tfidf).astype(np.float32)) # (N, k)

    # --- Cosine similarities (dot of normalized vectors), clamped at 0 (no negative "evidence"). ---
    entry_sim = np.maximum(entry_lsa @ crit_lsa.T, 0.0)        # (E, n_crit)
    skills_sim = np.maximum(skills_lsa @ crit_lsa.T, 0.0)      # (N, n_crit)
    summary_sim = np.maximum(summary_lsa @ crit_lsa.T, 0.0)    # (N, n_crit)

    # Max-evidence per candidate over its entries (segment max via np.maximum.at).
    entry_max = np.zeros((n_candidates, n_crit), dtype=np.float32)
    if len(entry_owner):
        np.maximum.at(entry_max, entry_owner, entry_sim.astype(np.float32))

    combined = (
        EVIDENCE_BLEND["entry"] * entry_max
        + EVIDENCE_BLEND["skills"] * skills_sim
        + EVIDENCE_BLEND["summary"] * summary_sim
    ).astype(np.float32)
    np.clip(combined, 0.0, 1.0, out=combined)
    return combined
