"""The JD checklist — the foundation everything scores against (UPDATED_ARCHITECTURE.md §4).

Why this file exists
--------------------
Every weight and every dealbreaker in this system must trace to a literal sentence in
job_description.docx, because Stage 5 is a live interview where we defend each one. This module
is the single source of truth for that mapping: a `Criterion` carries its matching `text` (the
query the retrieval lanes score against), its `weight`, and its `source` (the verbatim JD sentence).

Three groups:
  * SCORED_CRITERIA      — must-haves + nice-to-haves; drive the qualification score (§7 step 4).
  * SOFT_DEALBREAKERS    — text-derived "do NOT want" patterns; scored by similarity and applied as
                           a soft multiplicative penalty (§7 step 3). closed_source is the 5th soft
                           dealbreaker but is STRUCTURAL (absence of a signal), so it lives in
                           features.py, not here — see HARD_DEALBREAKERS / note below.
  * HARD_DEALBREAKERS    — structural, certain gates (consulting-only, title-chasing, architect
                           drift); detected in features.py, applied as a hard 0.1x gate (§7 step 3).

IMPORTANT (the rule from §4): if you change a weight, change its `source` to the JD sentence that
justifies it. No weight exists without one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
    """One JD requirement. `text` is the query matched against candidate evidence."""

    id: str
    text: str
    category: str           # "must_have" | "nice_to_have" | "soft_dealbreaker"
    weight: float           # 0 for dealbreakers (they gate, they don't add to qualification)
    source: str             # verbatim job_description.docx sentence (Stage-5 defensibility)


# ---------------------------------------------------------------------------
# Must-haves — "Things you absolutely need" (JD). Highest weights.
# strong_python is intentionally 0.6 not 1.0: the JD lists it but it is near-universal among
# the AI/ML target population and a weak discriminator, so weighting it equal to retrieval/eval
# would reward generic engineers. Direction (lower than the three differentiating must-haves) is
# JD-traced; the exact 0.6 is GUESS->CALIBRATE (config-level note in §7 table).
# ---------------------------------------------------------------------------
MUST_HAVES: list[Criterion] = [
    Criterion(
        id="embeddings_retrieval_production",
        text=(
            "production experience embeddings based retrieval systems sentence transformers "
            "openai embeddings bge e5 vector embeddings semantic search deployed to real users "
            "embedding drift index refresh retrieval quality regression dense retrieval"
        ),
        category="must_have",
        weight=1.0,
        source=(
            "Production experience with embeddings-based retrieval systems "
            "(sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users."
        ),
    ),
    Criterion(
        id="vector_db_hybrid_search",
        text=(
            "vector database hybrid search infrastructure pinecone weaviate qdrant milvus "
            "opensearch elasticsearch faiss approximate nearest neighbor ann index operational"
        ),
        category="must_have",
        weight=1.0,
        source=(
            "Production experience with vector databases or hybrid search infrastructure — "
            "Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar."
        ),
    ),
    Criterion(
        id="strong_python",
        text="strong python software engineering code quality production backend services",
        category="must_have",
        weight=0.6,
        source="Strong Python. Yes really, we care about code quality.",
    ),
    Criterion(
        id="ranking_eval_frameworks",
        text=(
            "designing evaluation frameworks for ranking systems ndcg mrr map "
            "offline to online correlation a/b test interpretation relevance metrics ranking quality"
        ),
        category="must_have",
        weight=1.0,
        source=(
            "Hands-on experience designing evaluation frameworks for ranking systems — "
            "NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Nice-to-haves — "Things we'd like you to have but won't reject you for" (JD). Lower weights.
# ---------------------------------------------------------------------------
NICE_TO_HAVES: list[Criterion] = [
    Criterion(
        id="llm_finetuning",
        text="llm fine tuning lora qlora peft instruction tuning supervised fine tuning adapters",
        category="nice_to_have",
        weight=0.5,
        source="LLM fine-tuning experience (LoRA, QLoRA, PEFT).",
    ),
    Criterion(
        id="learning_to_rank",
        text="learning to rank models xgboost gradient boosted trees neural ranking lambdamart ranknet",
        category="nice_to_have",
        weight=0.6,
        source="Experience with learning-to-rank models (XGBoost-based or neural).",
    ),
    Criterion(
        id="hr_tech_background",
        text="hr tech recruiting technology talent marketplace two sided marketplace hiring platform products",
        category="nice_to_have",
        weight=0.3,
        source="Prior exposure to HR-tech, recruiting tech, or marketplace products.",
    ),
    Criterion(
        id="distributed_systems",
        text="distributed systems large scale inference optimization scalability low latency model serving",
        category="nice_to_have",
        weight=0.4,
        source="Background in distributed systems or large-scale inference optimization.",
    ),
    Criterion(
        id="open_source",
        text="open source contributions ai ml space github maintainer published library framework",
        category="nice_to_have",
        weight=0.3,
        source="Open-source contributions in the AI/ML space.",
    ),
]

SCORED_CRITERIA: list[Criterion] = MUST_HAVES + NICE_TO_HAVES

# ---------------------------------------------------------------------------
# Soft (text-derived) dealbreakers — "Things we explicitly do NOT want" that we detect by text
# similarity, applied as a soft penalty scaled by the match strength (§7 step 3). These have
# weight 0 (they never add to the qualification score). closed_source_no_validation is the 5th
# soft dealbreaker but is detected structurally (absence of OSS/papers/talks) in features.py.
#
# Each `text` is a query describing the anti-pattern; a high similarity means the candidate looks
# like the trap. cv_speech_robotics is handled specially in scoring (its penalty is gated by LOW
# NLP/IR relevance — the JD only rejects CV/speech "WITHOUT significant NLP/IR exposure").
# ---------------------------------------------------------------------------
SOFT_DEALBREAKERS: list[Criterion] = [
    Criterion(
        id="pure_research_only",
        text=(
            "academic research lab phd researcher postdoc university published papers "
            "theoretical research pure research no production deployment research scientist"
        ),
        category="soft_dealbreaker",
        weight=0.0,
        source=(
            "If you've spent your career in pure research environments (academic labs, "
            "research-only roles) without any production deployment — we will not move forward."
        ),
    ),
    Criterion(
        id="langchain_tutorial_only",
        text=(
            "langchain openai api wrapper prompt chaining chatgpt wrapper recent project "
            "tutorial demo using langchain to call openai"
        ),
        category="soft_dealbreaker",
        weight=0.0,
        source=(
            "If your 'AI experience' consists primarily of recent (under 12 months) projects using "
            "LangChain to call OpenAI — we will probably not move forward."
        ),
    ),
    Criterion(
        id="cv_speech_robotics_only",
        text=(
            "computer vision image classification object detection segmentation speech recognition "
            "audio processing robotics motion planning control systems perception"
        ),
        category="soft_dealbreaker",
        weight=0.0,
        source=(
            "People whose primary expertise is computer vision, speech, or robotics without "
            "significant NLP/IR exposure."
        ),
    ),
    Criterion(
        id="framework_enthusiast",
        text=(
            "framework tutorial demo hot new framework blog post how i used framework to build "
            "proof of concept hello world side project showcase"
        ),
        category="soft_dealbreaker",
        weight=0.0,
        source=(
            "Framework enthusiasts. If your GitHub is full of LangChain tutorials and your blog "
            "posts are 'How I used [hot framework] to build [demo]' ... We need people who think "
            "about systems, not frameworks."
        ),
    ),
]

# Criteria that get a text-similarity score from the retrieval lanes: scored criteria + the
# text-derived soft dealbreakers. Order here defines the column order of cache/criteria_scores.npy.
MATCHED_CRITERIA: list[Criterion] = SCORED_CRITERIA + SOFT_DEALBREAKERS

# ---------------------------------------------------------------------------
# Hard (structural, certain) dealbreakers — detected in features.py from dates/titles/companies,
# applied as a hard 0.1x gate. Plus the structural soft dealbreaker (closed_source). Listed here
# only for documentation / source provenance; the detection logic lives in features.py.
# ---------------------------------------------------------------------------
HARD_DEALBREAKER_SOURCES = {
    "consulting_only_career": (
        "People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, "
        "Cognizant, Capgemini, etc.) in their entire career."
    ),
    "title_chaser": (
        "Title-chasers. If your career trajectory shows you optimizing for 'Senior' -> 'Staff' -> "
        "'Principal' titles by switching companies every 1.5 years, we're not a fit. "
        "We need someone who plans to be here for 3+ years."
    ),
    "architect_drift": (
        "If you are a senior engineer who hasn't written production code in the last 18 months "
        "because you've moved into 'architecture' or 'tech lead' roles — we will probably not "
        "move forward. This role writes code."
    ),
}

STRUCTURAL_SOFT_DEALBREAKER_SOURCES = {
    "closed_source_no_validation": (
        "People whose work has been entirely on closed-source proprietary systems for 5+ years "
        "without external validation (papers, talks, open-source)."
    ),
}


def criterion_index() -> dict[str, int]:
    """Map criterion id -> column index in the MATCHED_CRITERIA score matrix."""
    return {c.id: i for i, c in enumerate(MATCHED_CRITERIA)}


# Sanity checks — fail at import if the checklist is internally inconsistent.
assert len(MUST_HAVES) == 4, "JD §4 specifies exactly 4 must-haves"
assert len(NICE_TO_HAVES) == 5, "JD §4 specifies exactly 5 nice-to-haves"
assert len(SOFT_DEALBREAKERS) == 4, "4 text-derived soft dealbreakers (closed_source is structural)"
assert len({c.id for c in MATCHED_CRITERIA}) == len(MATCHED_CRITERIA), "criterion ids must be unique"
# 8 named JD dealbreakers total: 3 hard + 1 structural-soft + 4 text-soft.
assert len(HARD_DEALBREAKER_SOURCES) + len(STRUCTURAL_SOFT_DEALBREAKER_SOURCES) + len(SOFT_DEALBREAKERS) == 8
