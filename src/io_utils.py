"""Data loading, field-safe accessors, text building, and cache I/O.

Why this file exists
--------------------
Everything that touches the raw candidate JSON or the on-disk cache goes through here, so
that (a) missing/null fields are handled in exactly one place with documented neutral defaults
(UPDATED_ARCHITECTURE.md §7 "missing-data handling"), and (b) precompute.py and rank.py read/write
cache artifacts through identical helpers, making producer/consumer drift impossible.

Responsibilities
  * Stream candidates from .jsonl or .jsonl.gz (the 100K pool is 487MB — never load it all eagerly).
  * Safe nested getters that distinguish "absent/null" from "present but zero".
  * Build the text views the retrieval lanes consume (entry text, skills text, summary, full_text).
  * Save/load cache arrays + the candidate_id index + reference date + manifest.

No scoring logic lives here — this is pure plumbing, the most-reused and least-clever layer.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from config import CACHE_DIR, CACHE_FILES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming candidate loader
# ---------------------------------------------------------------------------
def _open_maybe_gzip(path: Path) -> io.TextIOBase:
    """Open a .jsonl or .jsonl.gz file as a UTF-8 text stream (validator handles both)."""
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return open(path, mode="rt", encoding="utf-8")


def iter_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield candidate dicts one at a time from a JSONL (optionally gzipped) file.

    Streaming keeps peak memory bounded regardless of pool size. Blank lines are skipped
    (the validator tolerates them, so we must too). A malformed line raises immediately rather
    than being silently dropped — silent data loss is exactly the failure mode the spec warns
    about, and a corrupt pool should fail loudly in precompute, not produce a short ranking.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")
    with _open_maybe_gzip(path) as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no} of {path}: {exc}") from exc


def count_candidates(path: str | Path) -> int:
    """Count non-blank lines without parsing JSON — used for sizing arrays up front."""
    path = Path(path)
    n = 0
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# Field-safe accessors. The schema marks many fields required, but the spec also warns of
# "plain-language Tier 5s" and sparse profiles, and real data has nulls (e.g. end_date). We never
# let a missing value masquerade as a meaningful zero — callers ask for an explicit default.
# ---------------------------------------------------------------------------
def get_profile(cand: dict) -> dict:
    return cand.get("profile") or {}


def get_signals(cand: dict) -> dict:
    return cand.get("redrob_signals") or {}


def get_career(cand: dict) -> list[dict]:
    ch = cand.get("career_history")
    return ch if isinstance(ch, list) else []


def get_skills(cand: dict) -> list[dict]:
    sk = cand.get("skills")
    return sk if isinstance(sk, list) else []


def safe_float(value: Any, default: float | None) -> float | None:
    """Coerce to float; return ``default`` for None/missing/uncoercible. ``default`` may be None
    to signal 'unknown' to a caller that treats unknown differently from any numeric value."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_date(value: Any) -> date | None:
    """Parse an ISO 'YYYY-MM-DD' string to a date; return None for null/blank/malformed.

    Dates drive recency (§7). Returning None (not today's date) for a missing value is deliberate:
    the recency layer treats unknown recency as neutral, never as 'inactive'.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Text views consumed by the retrieval lanes (§5). Kept here so precompute, text_match, and
# hybrid_retrieval all build text identically.
# ---------------------------------------------------------------------------
def _clean(text: Any) -> str:
    return text.strip() if isinstance(text, str) else ""


def entry_text(entry: dict) -> str:
    """Per-career-entry matching text: '{title}. {industry}. {description}' (§5 CriterionMatcher).

    Matching at the entry level (not whole-profile) structurally suppresses the skills-list-only
    keyword-stuffing trap — see §1.2.
    """
    parts = [_clean(entry.get("title")), _clean(entry.get("industry")), _clean(entry.get("description"))]
    return ". ".join(p for p in parts if p)


def entry_texts(cand: dict) -> list[str]:
    """All non-empty career-entry texts for a candidate (the primary evidence lane)."""
    texts = [entry_text(e) for e in get_career(cand)]
    return [t for t in texts if t]


def skills_text(cand: dict) -> str:
    """Space-joined skill names (secondary, deliberately low-weight; never alias-expanded — §5 guard)."""
    return " ".join(_clean(s.get("name")) for s in get_skills(cand) if _clean(s.get("name")))


def summary_text(cand: dict) -> str:
    """Profile summary + headline (tertiary evidence)."""
    prof = get_profile(cand)
    return " ".join(p for p in (_clean(prof.get("headline")), _clean(prof.get("summary"))) if p)


def full_text(cand: dict) -> str:
    """One concatenated document per candidate for the BM25 lexical lane (§5).

    Includes every career entry + skills + summary so BM25 sees the whole profile as one bag.
    """
    parts = entry_texts(cand)
    st = skills_text(cand)
    sm = summary_text(cand)
    if st:
        parts.append(st)
    if sm:
        parts.append(sm)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Cache I/O. precompute.py writes; rank.py reads. All keyed off config.CACHE_FILES.
# ---------------------------------------------------------------------------
def ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _cache_path(key: str) -> Path:
    return CACHE_DIR / CACHE_FILES[key]


def save_candidate_ids(ids: list[str]) -> None:
    """Write the canonical candidate_id ordering (row i here == row i in every .npy/.npz)."""
    ensure_cache_dir()
    path = _cache_path("candidate_ids")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("candidate_id\n")
        for cid in ids:
            fh.write(f"{cid}\n")
    logger.info("Wrote %d candidate_ids -> %s", len(ids), path)


def load_candidate_ids() -> list[str]:
    path = _cache_path("candidate_ids")
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
        assert header == "candidate_id", f"Unexpected header in {path}: {header!r}"
        return [line.strip() for line in fh if line.strip()]


def save_array(key: str, arr: np.ndarray) -> None:
    ensure_cache_dir()
    np.save(_cache_path(key), arr)


def load_array(key: str) -> np.ndarray:
    return np.load(_cache_path(key))


def save_npz(key: str, **arrays: np.ndarray) -> None:
    """Save a named bundle of 1-D feature columns (structural features, dealbreakers, honeypots)."""
    ensure_cache_dir()
    np.savez(_cache_path(key), **arrays)


def load_npz(key: str) -> dict[str, np.ndarray]:
    with np.load(_cache_path(key)) as data:
        return {k: data[k] for k in data.files}


def save_reference_date(ref: date) -> None:
    ensure_cache_dir()
    _cache_path("reference_date").write_text(ref.isoformat(), encoding="utf-8")


def load_reference_date() -> date:
    return date.fromisoformat(_cache_path("reference_date").read_text(encoding="utf-8").strip())


def save_manifest(manifest: dict) -> None:
    ensure_cache_dir()
    _cache_path("manifest").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def load_manifest() -> dict:
    return json.loads(_cache_path("manifest").read_text(encoding="utf-8"))
