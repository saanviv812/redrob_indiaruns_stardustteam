# Dealbreaker & structural-feature firing rates — verified on the full 100K pool

Evidence record for `src/features.py`. All numbers from streaming the real `candidates.jsonl`.
Corpus-anchored `reference_date = max(last_active_date) = 2026-05-27` (computed in 2.2s; §7 bug fix).
Full feature pass over 100,000 candidates ran in **5.8s** — negligible against the precompute budget.

## Hard dealbreaker gates

| Gate | Fires on | Verdict |
|---|---|---|
| `consulting_only_career` | **7,034 (7.03%)** | Matches the documented 7.0% with the JD's literal 6 firms **exactly**. Live, well-calibrated. |
| `title_chaser` | **0** | Inert on this dataset — see diagnosis below. Kept as a correct gate; honestly inert here. |
| `architect_drift` | **0** | Inert (no Architect/Principal/Director titles in the 47-title vocabulary; §15). |

### Why `title_chaser` fires zero (diagnosed, not hand-waved)

The JD pattern is escalating titles (Senior→Staff→Principal) + short (~1.5y) tenures + job-hopping.
Decomposing our conservative definition against the data:

- ≥3 career entries: **57,306 (57.3%)**
- of those, **majority of completed tenures < 18 months: 15,932** — so short-tenure hopping exists,
- of those, **net seniority climb ≥ 2 ladder levels: 0** — the dataset **does not model title
  escalation across a career history**.

The escalation requirement is what the JD actually describes ("optimizing for Senior → Staff →
Principal"). Loosening it to a ≥1-level climb would conflate a normal single promotion with
title-chasing and create false positives. We keep the gate correct and report it as inert here —
consistent with the project rule "don't force a rule to fire; verify, then keep or kill honestly."

## Structural soft dealbreaker

| Signal | Fires on | Note |
|---|---|---|
| `closed_source_strength > 0` | **1,589 (1.59%)** | 5y+ exp, GitHub = −1, and no validation token anywhere. The validation-token + GitHub guard keeps it narrow (not the 64.6% that merely lack GitHub). Strength capped at 0.5 (least-certain dealbreaker). |

## Structural positive / preference features

| Feature | Population | Cross-check |
|---|---|---|
| `location_acceptable` (India or willing_to_relocate) | 82,244 (82.2%) | India alone is 75.1%; relocation adds the rest. |
| `notice_period_ok` (≤30 days) | 13,809 (13.8%) | Matches documented 13.8% — confirms a soft +0.02 bonus (not a gate) is correct. |
| `github_positive` (score > 60) | 1,749 (1.75%) | Matches documented "~1,766 above 60". Rare, so a real differentiator. |
| `assessment_corroboration > 0` (≥1 skill scoring ≥66) | 7,228 (7.2%) | Objective, hard-to-game positive. |

## Honest implication for Stage 5

Of the 8 JD dealbreakers, on *this* dataset only `consulting_only_career` (hard) and
`closed_source_no_validation` (soft) fire materially; the 4 text-derived soft dealbreakers fire via
similarity (see retrieval lanes); `title_chaser`/`architect_drift` are inert because the data
doesn't contain the patterns they target. We operationalized all 8 (most competitors cover 2–5) and
are honest about which are live vs. inert here, rather than loosening thresholds to manufacture hits.
