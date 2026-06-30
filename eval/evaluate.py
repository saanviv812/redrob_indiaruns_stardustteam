"""Evaluation harness — Layer-1 anchors + adversarial gross-error checks (UPDATED_ARCHITECTURE.md §8).

Why this file exists
--------------------
Of 26 reviewable competitor repos, only one attempts any evaluation beyond self-validation. This is
our biggest lever. We build it honestly:

  FOUNDATIONAL HONESTY: we have NO ground truth. Every label here is OUR OWN inference, differing
  only in confidence. Nothing below is quoted as ground truth.

This script implements Layer 1 (our own anchors — zero manual labeling) and the gross-error checks
that the anchors enable. Layers 2 (LLM-as-judge) and 3 (human spot-check) are offline/manual and
documented in notebooks/eval_strategy.md; this script is the automatable, reproducible part.

The strongest, LEAST circular signal here is the NEGATIVE class: honeypots and keyword-stuffers are
defined by the JD/spec, not by our ranker, so "our ranker must not rank them high" is a genuine
adversarial check, not an echo of our own scoring. The positive anchors overlap our heuristics
partially (disclosed) and are used only for ordering sanity, never as truth.

Run:
    python eval/evaluate.py --candidates ./candidates.jsonl
(reads cache/ via the src modules; run precompute.py first.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable when run from the repo root.
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

import io_utils as io  # noqa: E402
import rank  # noqa: E402
from metrics import (  # noqa: E402  (eval/metrics.py)
    average_precision, composite_score, ndcg_at_k, precision_at_k,
)

TRAP_TITLES = {
    "business analyst", "hr manager", "mechanical engineer", "accountant", "project manager",
    "customer support", "operations manager", "content writer", "sales executive",
    "civil engineer", "graphic designer", "marketing manager",
}
AIML_TITLE_HINTS = ("machine learning", "ml engineer", "ai engineer", "data scientist",
                    "applied ml", "nlp", "search engineer", "recommendation", "research engineer")
AI_SKILL_HINTS = ("ml", "ai", "nlp", "embedding", "retrieval", "transformer", "llm", "deep learning",
                  "fine-tun", "rag", "ranking", "recommendation", "neural", "pytorch", "tensorflow")

# Per-class caps for a stratified anchored set (oversample the decision boundary; §8 rule 2).
CAPS = {"trap_stuffer": 400, "trap_clean": 200, "background": 400, "weak_pos": 400, "strong_pos": 10**9}


def _ai_skill_count(cand: dict) -> int:
    return sum(1 for s in io.get_skills(cand)
               if any(h in (s.get("name") or "").lower() for h in AI_SKILL_HINTS))


def assign_tier(cand: dict, is_honeypot: bool) -> tuple[str, int] | None:
    """Return (class_name, relevance_tier 0-4) for anchored candidates, or None to skip.

    INFERENCE, not ground truth. Tiers: honeypot/keyword-stuffer 0, off-target 1, background 2,
    weak AI/ML positive 3, strong AI/ML positive 4.
    """
    if is_honeypot:
        return ("honeypot", 0)
    prof = io.get_profile(cand)
    title = (prof.get("current_title") or "").lower()
    yoe = io.safe_float(prof.get("years_of_experience"), 0.0) or 0.0

    if title in TRAP_TITLES:
        return ("trap_stuffer", 0) if _ai_skill_count(cand) >= 3 else ("trap_clean", 1)

    is_aiml = any(h in title for h in AIML_TITLE_HINTS)
    if is_aiml:
        sig = io.get_signals(cand)
        gh = io.safe_float(sig.get("github_activity_score"), -1.0) or -1.0
        assessed_high = any(isinstance(v, (int, float)) and v >= 66 for v in
                            (sig.get("skill_assessment_scores") or {}).values())
        strong = (4.0 <= yoe <= 9.0) and (gh > 60 or assessed_high)
        return ("strong_pos", 4) if strong else ("weak_pos", 3)
    return ("background", 2)


def build_anchors(candidates_path: str) -> dict[str, int]:
    """Stream once; build a stratified, capped anchored label set (id -> tier). Deterministic."""
    cache = rank.load_cache()
    hp = cache["honeypots"]["is_honeypot"]
    idpos = {cid: i for i, cid in enumerate(cache["ids"])}

    counts: dict[str, int] = {}
    anchors: dict[str, int] = {}
    for cand in io.iter_candidates(candidates_path):
        cid = cand.get("candidate_id")
        i = idpos.get(cid)
        is_hp = i is not None and hp[i] > 0.5
        result = assign_tier(cand, is_hp)
        if result is None:
            continue
        cls, tier = result
        cap = CAPS.get(cls, 0) if cls != "honeypot" else 10**9
        if counts.get(cls, 0) >= cap:
            continue
        counts[cls] = counts.get(cls, 0) + 1
        anchors[cid] = tier
    print("Anchored set composition:", {k: counts[k] for k in sorted(counts)})
    return anchors


def evaluate(candidates_path: str) -> None:
    cache = rank.load_cache()
    scores = rank.compute_scores(cache)
    ids = cache["ids"]
    final = scores["final"]
    hp = cache["honeypots"]["is_honeypot"]
    idpos = {cid: i for i, cid in enumerate(ids)}

    # Our full ranking order (desc score, tie-break id asc — same as rank.py).
    order = sorted(range(len(final)), key=lambda i: (-final[i], ids[i]))
    rank_of = {ids[i]: r for r, i in enumerate(order, start=1)}
    top100 = [ids[i] for i in order[:100]]
    top10 = set(top100[:10])

    anchors = build_anchors(candidates_path)

    # --- Gross-error checks (the least-circular, strongest signal). ---
    hp_top100 = sum(1 for c in top100 if hp[idpos[c]] > 0.5)
    hp_top10 = sum(1 for c in top100[:10] if hp[idpos[c]] > 0.5)
    trap_anchor_ids = [c for c, t in anchors.items() if t == 0]
    trap_in_top100 = sum(1 for c in trap_anchor_ids if c in set(top100))
    # percentile rank of negative anchors (1.0 = best/top). Want LOW for honeypots/traps.
    neg_pctl = [1.0 - (rank_of[c] - 1) / (len(ids) - 1) for c in trap_anchor_ids if c in rank_of]
    strong_ids = [c for c, t in anchors.items() if t == 4]
    pos_pctl = [1.0 - (rank_of[c] - 1) / (len(ids) - 1) for c in strong_ids if c in rank_of]

    def med(xs): return sorted(xs)[len(xs) // 2] if xs else float("nan")

    print("\n=== GROSS-ERROR CHECKS (adversarial; honeypots/traps defined by spec, not our ranker) ===")
    print(f"honeypots in top-100: {hp_top100}  (auto-DQ if >10)   top-10: {hp_top10}")
    print(f"negative anchors (honeypot+keyword-stuffer): {len(trap_anchor_ids)};  in our top-100: {trap_in_top100}")
    print(f"median score-percentile  negative anchors: {med(neg_pctl):.3f}  (want LOW)")
    print(f"median score-percentile  strong positives: {med(pos_pctl):.3f}  (want HIGH)")

    # --- NDCG/MAP over the stratified anchored set (ordering quality vs OUR labels). ---
    anchored = [(c, anchors[c]) for c in anchors if c in rank_of]
    anchored.sort(key=lambda ct: rank_of[ct[0]])  # our predicted order
    rels = [t for _, t in anchored]
    total_rel = sum(1 for t in rels if t >= 3)
    n10 = ndcg_at_k(rels, 10); n50 = ndcg_at_k(rels, 50)
    mp = average_precision(rels, total_rel); p10 = precision_at_k(rels, 10)
    print("\n=== NDCG/MAP over the stratified anchored set (OUR inferred labels — NOT ground truth) ===")
    print(f"anchored candidates ranked: {len(anchored)}  (relevant tier3+: {total_rel})")
    print(f"NDCG@10={n10:.4f}  NDCG@50={n50:.4f}  MAP={mp:.4f}  P@10={p10:.4f}")
    print(f"composite (same formula as the contest) = {composite_score(n10, n50, mp, p10):.4f}")
    print("\nCAVEAT (§8): stratified + our-own-labels => a development compass for relative change,")
    print("NOT an unbiased estimate of true NDCG. The hidden Redrob rubric may disagree.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the ranking against Layer-1 anchors.")
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()
    evaluate(args.candidates)


if __name__ == "__main__":
    main()
