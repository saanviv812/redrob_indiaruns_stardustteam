"""Static, hand-curated synonym/alias expansion (UPDATED_ARCHITECTURE.md §5).

Why this file exists
--------------------
TF-IDF and BM25 are *lexical* — they don't know "RAG" == "retrieval augmented generation" or that
"vector DB" implies "Pinecone/Milvus/Weaviate/Qdrant". A small, deterministic alias table expanded
into career-entry text, summary text, and the JD-criterion text (BEFORE fitting either lane)
improves recall cheaply, with no model and no network.

THE GUARD (mandatory, §5): never apply this to the standalone skills list. Expanding a Marketing
Manager's stray "ML" tag into a full AI vocabulary makes the keyword-stuffer trap worse on the
lexical lane. Skills text is passed through verbatim.

Each mapping is documented (defensible at Stage 5). Expansion is additive (we append the canonical
phrase, keeping the original token), and applied as whole-word, case-insensitive replacement so we
don't expand substrings inside unrelated words.
"""

from __future__ import annotations

import re

# alias term -> canonical phrase appended when the alias appears. Lowercased keys; whole-word match.
# Kept intentionally small and high-precision: every entry maps a real abbreviation/brand to the
# JD's own vocabulary, so it can only help recall on genuinely relevant candidates.
ALIAS_TABLE: dict[str, str] = {
    # Retrieval / RAG
    "rag": "retrieval augmented generation",
    "ann": "approximate nearest neighbor",
    "knn": "nearest neighbor search",
    "ivf": "inverted file index vector search",
    "hnsw": "approximate nearest neighbor vector index",
    # Vector DBs / search infra -> the JD's hybrid-search vocabulary
    "pinecone": "vector database hybrid search",
    "weaviate": "vector database hybrid search",
    "qdrant": "vector database hybrid search",
    "milvus": "vector database hybrid search",
    "faiss": "vector database approximate nearest neighbor",
    "opensearch": "hybrid search infrastructure",
    "elasticsearch": "hybrid search infrastructure",
    "elastic": "hybrid search infrastructure",
    # Embeddings
    "bge": "embeddings model retrieval",
    "e5": "embeddings model retrieval",
    "sbert": "sentence transformers embeddings",
    "embeddings": "embeddings based retrieval",
    # Recsys / ranking
    "recsys": "recommendation system ranking",
    "ltr": "learning to rank",
    "lambdamart": "learning to rank gradient boosted",
    "ranknet": "neural learning to rank",
    # LLM / fine-tuning
    "llm": "large language model",
    "lora": "llm fine tuning parameter efficient",
    "qlora": "llm fine tuning parameter efficient",
    "peft": "parameter efficient fine tuning",
    "rlhf": "llm fine tuning alignment",
    # Eval
    "ndcg": "ranking evaluation metric",
    "mrr": "ranking evaluation metric",
    "map": "mean average precision ranking evaluation",
    # NLP / IR (relevant to distinguish from CV/speech)
    "nlp": "natural language processing information retrieval",
    "ir": "information retrieval",
    "bm25": "lexical retrieval ranking",
    "tf-idf": "lexical retrieval",
}

# Precompiled whole-word pattern (longest keys first so multi-token aliases win where applicable).
_ALIAS_KEYS = sorted(ALIAS_TABLE.keys(), key=len, reverse=True)
_ALIAS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ALIAS_KEYS) + r")\b",
    flags=re.IGNORECASE,
)


def expand(text: str) -> str:
    """Append canonical phrases for every alias found, preserving the original text.

    Example: "Built a RAG pipeline on Pinecone" ->
             "Built a RAG pipeline on Pinecone retrieval augmented generation vector database
              hybrid search". Idempotent enough for our use (we only expand once, pre-fit).
    """
    if not text:
        return text
    found: list[str] = []
    seen: set[str] = set()
    for match in _ALIAS_PATTERN.finditer(text):
        phrase = ALIAS_TABLE[match.group(0).lower()]
        if phrase not in seen:
            seen.add(phrase)
            found.append(phrase)
    if not found:
        return text
    return text + " " + " ".join(found)
