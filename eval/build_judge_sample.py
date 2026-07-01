"""Stratified sampler + blind profile extractor for the LLM-as-judge eval (UPDATED_ARCHITECTURE.md §8).

Produces:
  scratchpad/judge_profiles.txt  — SHUFFLED, blind profiles (candidate_id + raw fields only; NO
                                    model score, rank, or stratum). This is what the judge reads.
  scratchpad/judge_key.json      — id -> {stratum, our_rank}. NOT looked at until after labeling.

Fair stratification (fixed seed): spans the decision boundary and the places the model could be
wrong (adjacent/Tier-5, boundary rank 80-150), not just easy wins. Blind judging (§8 rule 1).
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
import io_utils as io  # noqa: E402
import rank  # noqa: E402

OUT = Path("C:/Users/saanv/AppData/Local/Temp/claude/c--Personal-Saanvi-DELL-Personal-SaanviVarma-Hackathosn-INDIA-RUNS-TRACK1/c7ac4723-403e-43e2-a9d9-2614966fdc4b/scratchpad")
# Optional seed override + output suffix (for drawing an independent second sample): argv[2]=seed, argv[3]=suffix
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260702
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""
TRAP = {"business analyst","hr manager","mechanical engineer","accountant","project manager",
        "customer support","operations manager","content writer","sales executive",
        "civil engineer","graphic designer","marketing manager"}
AITITLE = ("machine learning","ml engineer","ml platform","ai engineer","ai/ml","data scientist",
           "applied scientist","applied ml","nlp","deep learning","research engineer",
           "search engineer","ranking engineer","recommendation")
ADJACENT = ("software engineer","backend engineer","data engineer","platform engineer",
            "full stack","fullstack","systems engineer","devops")
AI_SKILL = ("ml","ai","nlp","embedding","retrieval","transformer","llm","deep learning","rag",
            "ranking","recommendation","neural","pytorch","tensorflow","bert","vector")
QUOTA = {"ai_titled":10,"trap_stuffer":8,"trap_clean":5,"honeypot":5,"boundary":12,"adjacent":8}


def ai_skill_count(cand):
    return sum(1 for s in io.get_skills(cand) if any(h in (s.get("name") or "").lower() for h in AI_SKILL))


def stratum(cand, our_rank, is_hp):
    title = (io.get_profile(cand).get("current_title") or "").lower()
    if is_hp:
        return "honeypot"
    if 80 <= our_rank <= 150:
        return "boundary"
    if title in TRAP:
        return "trap_stuffer" if ai_skill_count(cand) >= 3 else "trap_clean"
    if any(k in title for k in AITITLE):
        return "ai_titled"
    if any(k in title for k in ADJACENT):
        return "adjacent"
    return None  # not a stratum we sample from


def compact_profile(cand) -> str:
    p = io.get_profile(cand); s = io.get_signals(cand)
    lines = [f"### {cand['candidate_id']}",
             f"Title: {p.get('current_title')} @ {p.get('current_company')} ({p.get('current_industry')})",
             f"Experience: {p.get('years_of_experience')} yrs | Location: {p.get('location')}, {p.get('country')}",
             f"Summary: {(p.get('summary') or '')[:400]}",
             "Career:"]
    for e in io.get_career(cand):
        lines.append(f"  - {e.get('title')} @ {e.get('company')} [{e.get('industry')}], "
                     f"{e.get('duration_months')}mo{' (current)' if e.get('is_current') else ''}: "
                     f"{(e.get('description') or '')[:220]}")
    sk = [f"{x.get('name')}:{x.get('proficiency')}" for x in io.get_skills(cand)][:14]
    lines.append("Skills: " + ", ".join(sk))
    assess = s.get("skill_assessment_scores") or {}
    assess_str = ", ".join(f"{k}={v:.0f}" for k, v in list(assess.items())[:8]) if assess else "none"
    lines.append(f"Signals: github={s.get('github_activity_score')}, response_rate={s.get('recruiter_response_rate')}, "
                 f"last_active={s.get('last_active_date')}, notice_days={s.get('notice_period_days')}, "
                 f"open_to_work={s.get('open_to_work_flag')}, assessments=[{assess_str}]")
    return "\n".join(lines)


def main():
    cache = rank.load_cache()
    scores = rank.compute_scores(cache)
    ids = cache["ids"]; final = scores["final"]; hp = cache["honeypots"]["is_honeypot"]
    order = sorted(range(len(final)), key=lambda i: (-final[i], ids[i]))
    our_rank = {ids[i]: r for r, i in enumerate(order, 1)}
    idpos = {c: i for i, c in enumerate(ids)}

    # Exclude candidates already in the first sample, so a second sample is truly independent.
    exclude: set[str] = set()
    first_key = OUT / "judge_key.json"
    if SUFFIX and first_key.exists():
        exclude = set(json.loads(first_key.read_text(encoding="utf-8")))

    # Collect the FULL population of each stratum (unbiased), then random-sample the quota from it.
    # (Earlier this capped at the first ~40*quota candidates in file order — a selection bias if the
    #  file is ordered by any property. Collecting all removes that.)
    buckets: dict[str, list[str]] = {k: [] for k in QUOTA}
    for cand in io.iter_candidates(sys.argv[1]):
        cid = cand["candidate_id"]
        if cid in exclude:
            continue
        st = stratum(cand, our_rank[cid], hp[idpos[cid]] > 0.5)
        if st:
            buckets[st].append(cid)

    rng = random.Random(SEED)
    chosen: dict[str, str] = {}   # id -> stratum
    for st, pool in buckets.items():
        pick = rng.sample(pool, min(QUOTA[st], len(pool)))  # uniform random over the whole stratum
        for cid in pick:
            chosen[cid] = st

    # Re-stream to extract profiles for the chosen ids.
    profiles = {}
    for cand in io.iter_candidates(sys.argv[1]):
        if cand["candidate_id"] in chosen:
            profiles[cand["candidate_id"]] = compact_profile(cand)

    shuffled = list(chosen.keys()); rng.shuffle(shuffled)
    OUT.mkdir(parents=True, exist_ok=True)
    prof_name, key_name = f"judge_profiles{SUFFIX}.txt", f"judge_key{SUFFIX}.json"
    with open(OUT / prof_name, "w", encoding="utf-8") as f:
        f.write("# BLIND JUDGING SET — assign each a relevance tier 0-4 vs the JD.\n")
        f.write("# 4=ideal fit, 3=strong, 2=partial/adjacent, 1=weak, 0=not a fit / honeypot / trap.\n\n")
        for cid in shuffled:
            f.write(profiles[cid] + "\n\n")
    key = {cid: {"stratum": chosen[cid], "our_rank": our_rank[cid]} for cid in chosen}
    (OUT / key_name).write_text(json.dumps(key, indent=1), encoding="utf-8")
    from collections import Counter
    print("sample composition:", dict(Counter(chosen.values())), "total:", len(chosen))
    print("wrote", OUT / prof_name)


if __name__ == "__main__":
    main()
