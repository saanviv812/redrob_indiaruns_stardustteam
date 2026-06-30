"""Correct, standard ranking metrics — NDCG@k, MAP, P@k (UPDATED_ARCHITECTURE.md §8 rule 3).

Why this file exists
--------------------
The single biggest gap in the competitive field is that almost no one evaluates with a *correct*
metric (one competitor self-reports NDCG@10 = 1.0000, a leakage signal). A self-reported metric
computed wrong is worse than none — it looks like evidence while being false. So this module
implements the exact gain/discount functions the hackathon scores with (submission_spec §4) and is
unit-tested against hand-worked examples in tests/.

Conventions match the spec:
  * graded relevance tiers 0-4, gain = 2^rel - 1 (standard NDCG gain).
  * NDCG@k discount = 1/log2(rank+1), rank starting at 1.
  * P@10 counts tier >= 3 as "relevant" (spec: "Fraction of top-10 that are 'relevant' (tier 3+)").
  * MAP uses the same binary relevance threshold (tier >= 3) — one query (the JD), so MAP == AP.
"""

from __future__ import annotations

import math

RELEVANT_TIER = 3  # spec: "relevant" == tier 3+


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted cumulative gain over the first k items (gain = 2^rel - 1)."""
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2 ** rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_relevances: list[int], k: int) -> float:
    """NDCG@k: DCG of our ordering / DCG of the ideal ordering of the SAME relevance multiset.

    Returns 0.0 if the ideal DCG is 0 (no relevant items among those ranked) — avoids div-by-zero.
    """
    actual = dcg_at_k(ranked_relevances, k)
    ideal = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def precision_at_k(ranked_relevances: list[int], k: int, threshold: int = RELEVANT_TIER) -> float:
    """Fraction of the top-k that are relevant (tier >= threshold)."""
    if k <= 0:
        return 0.0
    top = ranked_relevances[:k]
    return sum(1 for r in top if r >= threshold) / float(k)


def average_precision(ranked_relevances: list[int], total_relevant: int,
                      threshold: int = RELEVANT_TIER) -> float:
    """AP for one query: mean of P@i at each rank i where item i is relevant, divided by R.

    `total_relevant` (R) is the count of relevant items in the FULL ground truth (not just the
    ranked prefix) — using the prefix count would inflate AP, the exact kind of subtle error that
    produces falsely-high MAP. With one query, MAP == this AP.
    """
    if total_relevant <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, rel in enumerate(ranked_relevances, start=1):
        if rel >= threshold:
            hits += 1
            score += hits / i
    return score / total_relevant


def composite_score(ndcg10: float, ndcg50: float, map_score: float, p10: float) -> float:
    """The hackathon's official composite (spec §4): 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10."""
    return 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_score + 0.05 * p10
