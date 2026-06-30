"""The JD as a single query document for the BM25 lexical lane (UPDATED_ARCHITECTURE.md §5).

Why this file exists
--------------------
The BM25 lane scores each candidate's full_text against one query: the JD. We deliberately include
only the *substantive* requirement text (the mandate + must-haves + nice-to-haves), and exclude the
JD's closing "Final note for the participants of the Redrob hackathon" meta-section — that section
is about the contest, not about the candidate profile we're matching, and including it would inject
words like "trap", "dataset", "keyword" into the query.

The text is assembled from the same criterion queries used by the semantic lane (so the two lanes
agree on what they're looking for) plus the JD's own mandate phrasing, then alias-expanded.
"""

from __future__ import annotations

from aliases import expand
from jd_criteria import MUST_HAVES, NICE_TO_HAVES

# The role mandate, in the JD's own words (the substantive "what you'd be doing" + ideal-candidate
# signal), excluding the hackathon meta-section. Kept close to the JD so it is defensible.
_MANDATE = (
    "Senior AI Engineer founding team. Own the intelligence layer: the ranking, retrieval, and "
    "matching systems that decide what recruiters see when they search for candidates. Ship a v2 "
    "ranking system using embeddings, hybrid retrieval, and LLM-based re-ranking. Set up evaluation "
    "infrastructure: offline benchmarks, online A/B testing, recruiter-feedback loops. Drive the "
    "long-term architecture of candidate-JD matching at scale. Applied machine learning at product "
    "companies, not pure services. Shipped at least one end-to-end ranking, search, or "
    "recommendation system to real users at meaningful scale. Strong opinions on retrieval "
    "(hybrid vs dense), evaluation (offline vs online), and LLM integration (fine-tune vs prompt). "
    "Scrappy product-engineering attitude, ships working systems."
)


def _build_jd_full_text() -> str:
    parts = [_MANDATE]
    parts.extend(c.text for c in MUST_HAVES)
    parts.extend(c.text for c in NICE_TO_HAVES)
    return expand(" ".join(parts))


# Assembled once at import (cheap, deterministic).
JD_FULL_TEXT: str = _build_jd_full_text()
