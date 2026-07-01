"""Sandbox demo for the Redrob Ranker (submission_spec.docx §10.5 — mandatory).

A hosted UI where organizers can verify the ranking system runs reproducibly on a small sample:
upload a candidate JSONL (<=100 lines from candidates.jsonl) -> the FULL pipeline (offline
precompute + CPU-only, no-network ranking) runs -> shows the ranked top candidates with reasoning
and a downloadable CSV. Deploy free on HuggingFace Spaces or Streamlit Cloud (see README).

This app is ONLY the demo wrapper; the ranking logic is unchanged src/rank.py + src/precompute.py.
"""

from __future__ import annotations

import csv
import io as _io
import os
import sys
import tempfile
import time
from pathlib import Path

# Route the pipeline's cache to a temp dir BEFORE importing it (config reads this env at import time).
_CACHE = Path(tempfile.gettempdir()) / "redrob_sandbox_cache"
os.environ.setdefault("REDROB_CACHE_DIR", str(_CACHE))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402
import precompute  # noqa: E402
import rank  # noqa: E402

st.set_page_config(page_title="Redrob Ranker — Sandbox", layout="wide")
st.title("Redrob Ranker — Sandbox Demo")
st.caption(
    "Upload a small candidate sample (JSONL, ≤100 lines from `candidates.jsonl`). "
    "The full pipeline runs end-to-end — **offline precompute + CPU-only ranking, no network calls, "
    "deterministic** — and returns the ranked candidates with 1–2 sentence reasoning."
)

with st.expander("How the ranker works (for reviewers)"):
    st.markdown(
        "- **Retrieval**: TF-IDF + LSA matched at the *career-entry* level (max-evidence per JD criterion) "
        "fused with BM25 via Reciprocal Rank Fusion — resists the JD's keyword-stuffer trap.\n"
        "- **Structural layer**: experience/applied-ML fit, availability (recency, response rate), "
        "GitHub/assessment corroboration, and 8 JD-named dealbreakers.\n"
        "- **Honeypot gate**: two EDA-verified impossibility rules → near-zero multiplier (keeps fakes out of the top).\n"
        "- **Blend** (calibrated against a blind LLM-as-judge): 0.90 qualification / 0.10 retrieval, "
        "then multiplicative gates + tiny additive preferences. Fully deterministic (fixed seed)."
    )

uploaded = st.file_uploader("Candidate sample — .jsonl (one JSON candidate per line)", type=["jsonl"])

if uploaded is None:
    st.info(
        "Waiting for a sample. Create one with:  `head -100 candidates.jsonl > sample.jsonl`  and upload it. "
        "(≤100 candidates keeps it well within the ≤5-min CPU budget.)"
    )
else:
    tmp_in = Path(tempfile.gettempdir()) / "redrob_sandbox_input.jsonl"
    tmp_in.write_bytes(uploaded.getvalue())
    tmp_out = Path(tempfile.gettempdir()) / "redrob_sandbox_out.csv"
    try:
        with st.spinner("Running precompute + ranking (CPU only, no network)…"):
            t0 = time.time()
            precompute.run(str(tmp_in))
            rows = rank.run(str(tmp_in), str(tmp_out))
            elapsed = time.time() - t0
    except Exception as exc:  # surface errors clearly in the demo rather than a blank page
        st.error(f"Pipeline error: {exc}")
    else:
        st.success(
            f"Ranked {len(rows)} candidates in {elapsed:.1f}s — CPU-only, no network, deterministic."
        )
        table = [{"rank": r["rank"], "candidate_id": r["candidate_id"],
                  "score": round(r["score"], 4), "reasoning": r["reasoning"]} for r in rows]
        st.dataframe(table, use_container_width=True, hide_index=True)

        buf = _io.StringIO()
        w = csv.writer(buf)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r["candidate_id"], r["rank"], f"{r['score']:.6f}", r["reasoning"]])
        st.download_button("⬇ Download ranked CSV", buf.getvalue().encode("utf-8"),
                           file_name="sample_ranking.csv", mime="text/csv")
