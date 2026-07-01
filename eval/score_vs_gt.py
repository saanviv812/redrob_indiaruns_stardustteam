"""Grade our ranking against a provided ground-truth top-100 (the contest metrics).

Unlike eval/evaluate.py (which uses our own inferred anchors), this scores against an EXTERNAL
ground-truth file with per-candidate gold ranks — the real accuracy test. NDCG uses the GLOBAL
ideal ordering (the full GT tier multiset), not a self-ideal, so the numbers are not inflated.

Relevance tiers are derived from GT rank (the GT file is graded by construction):
  rank 1-10 -> tier 4   (top, "directly match JD pillars")
  rank 11-30 -> tier 3  (strong)
  rank 31-60 -> tier 2  (partial fit)
  rank 61-100 -> tier 1 (adjacent / marginal)
  not in GT top-100 -> tier 0

Run from the repo root, e.g.:
  REDROB_CACHE_DIR=cache_dataset_b python eval/score_vs_gt.py --gt ../dataset_b_ground_truth_top100.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
import rank  # noqa: E402


def gt_tier(gt_rank: int) -> int:
    if gt_rank <= 10:
        return 4
    if gt_rank <= 30:
        return 3
    if gt_rank <= 60:
        return 2
    if gt_rank <= 100:
        return 1
    return 0


def _dcg(tiers, k: int) -> float:
    return sum((2 ** t - 1) / math.log2(i + 2) for i, t in enumerate(tiers[:k]))


def load_gt(path: str) -> dict[str, int]:
    """candidate_id -> relevance tier (0-4) from GT rank."""
    tiers: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tiers[row["candidate_id"].strip()] = gt_tier(int(row["rank"]))
    return tiers


def score(gt_path: str) -> dict:
    cache = rank.load_cache()
    scores = rank.compute_scores(cache)
    ids = cache["ids"]
    final = scores["final"]
    order = sorted(range(len(final)), key=lambda i: (-final[i], ids[i]))
    our_ranking = [ids[i] for i in order]            # full descending order
    our_top100 = set(our_ranking[:100])

    gt = load_gt(gt_path)
    gt_top100 = set(gt)
    R = sum(1 for t in gt.values() if t >= 3)        # relevant = tier 3+

    our_tiers = [gt.get(cid, 0) for cid in our_ranking[:100]]
    ideal_tiers = sorted(gt.values(), reverse=True)  # global ideal (the GT's own tier multiset)

    def ndcg(k):
        ideal = _dcg(ideal_tiers, k)
        return _dcg(our_tiers, k) / ideal if ideal > 0 else 0.0

    # MAP over our top-100 (binary rel tier>=3), normalized by total relevant R.
    hits = 0
    ap = 0.0
    for i, cid in enumerate(our_ranking[:100], start=1):
        if gt.get(cid, 0) >= 3:
            hits += 1
            ap += hits / i
    map_score = ap / R if R else 0.0
    p10 = sum(1 for t in our_tiers[:10] if t >= 3) / 10.0

    ndcg10, ndcg50 = ndcg(10), ndcg(50)
    composite = 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_score + 0.05 * p10
    return {
        "overlap_top100": len(our_top100 & gt_top100),
        "overlap_top10": len(set(our_ranking[:10]) & gt_top100),
        "recall_relevant": (sum(1 for c in our_top100 if gt.get(c, 0) >= 3) / R) if R else 0.0,
        "NDCG@10": ndcg10, "NDCG@50": ndcg50, "MAP": map_score, "P@10": p10,
        "composite": composite,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    r = score(args.gt)
    print(f"overlap our-top100 vs GT-top100 : {r['overlap_top100']}/100")
    print(f"overlap our-top10  vs GT-top100 : {r['overlap_top10']}/10")
    print(f"recall of GT relevant (tier3+)  : {r['recall_relevant']:.3f}")
    print(f"NDCG@10={r['NDCG@10']:.4f}  NDCG@50={r['NDCG@50']:.4f}  MAP={r['MAP']:.4f}  P@10={r['P@10']:.4f}")
    print(f"COMPOSITE (contest formula)     : {r['composite']:.4f}")


if __name__ == "__main__":
    main()
