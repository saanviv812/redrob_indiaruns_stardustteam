"""Static compliance proof: no network, no GPU, bounded disk (UPDATED_ARCHITECTURE.md §10).

Why this file exists
--------------------
4 of 26 reviewable competitor repos likely fail Stage 3 on exactly these points (2 call hosted LLM
APIs at rank time, 2 are GPU-only). We treat compliance as a *measured* check, never an assertion.
This script greps the source reachable from rank.py for forbidden patterns and sums the cache size,
so the claim "CPU-only, no network, <5GB disk" is evidence, not a promise. Run it in CI / before
every submission.

Exit code 0 = compliant; 1 = a violation was found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
CACHE_DIR = SRC_DIR.parent / "cache"
DISK_BUDGET_GB = 5.0

# Patterns that would indicate a network call or GPU use at ranking time. We scan the whole src/
# tree (rank.py's transitive imports all live here). aliases.py legitimately uses 're'; that's fine.
# NOTE: these match module USAGE (import X / X.method), not bare words — the JD-criterion query
# strings legitimately contain words like "openai" and "langchain" as text to match against, and
# those must NOT be flagged. So every SDK name requires an `import` or attribute-access context.
NETWORK_PATTERNS = [
    r"\brequests\.(get|post|put|delete|patch|head|session)\b",
    r"\bimport\s+httpx\b", r"\bhttpx\.", r"\burllib\.request\b", r"\bimport\s+aiohttp\b",
    r"\bsocket\.socket\b",
    r"\bimport\s+openai\b", r"\bopenai\.", r"\bfrom\s+openai\b",
    r"\bimport\s+anthropic\b", r"\banthropic\.", r"\bfrom\s+anthropic\b",
    r"\bimport\s+cohere\b", r"\bcohere\.", r"\bgoogle\.generativeai\b",
    r"\bimport\s+genai\b", r"\bgenai\.", r"\bimport\s+groq\b", r"\bgroq\.",
    r"\bhuggingface_hub\b", r"\.from_pretrained\b",
]
GPU_PATTERNS = [
    r"\.cuda\(", r"device\s*=\s*['\"]cuda", r"torch\.device\(\s*['\"]cuda",
    r"\bimport torch\b", r"\bimport tensorflow\b", r"\bcupy\b", r"\bonnxruntime_gpu\b",
]


def _scan(patterns: list[str]) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    compiled = [re.compile(p) for p in patterns]
    for py in sorted(SRC_DIR.glob("*.py")):
        if py.name == "compliance_check.py":
            continue  # this file names the patterns as strings; don't flag itself
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for rx in compiled:
                if rx.search(line):
                    hits.append((py.name, lineno, line.strip()))
    return hits


def _cache_size_gb() -> float:
    if not CACHE_DIR.exists():
        return 0.0
    total = sum(p.stat().st_size for p in CACHE_DIR.rglob("*") if p.is_file())
    return total / 1e9


def main() -> int:
    ok = True

    net = _scan(NETWORK_PATTERNS)
    print(f"[network] {'OK — no network usage found' if not net else 'VIOLATION'}")
    for f, ln, txt in net:
        ok = False
        print(f"    {f}:{ln}: {txt}")

    gpu = _scan(GPU_PATTERNS)
    print(f"[gpu]     {'OK — no GPU/CUDA usage found' if not gpu else 'VIOLATION'}")
    for f, ln, txt in gpu:
        ok = False
        print(f"    {f}:{ln}: {txt}")

    size = _cache_size_gb()
    disk_ok = size <= DISK_BUDGET_GB
    ok = ok and disk_ok
    print(f"[disk]    cache = {size:.3f} GB  (budget {DISK_BUDGET_GB} GB) — {'OK' if disk_ok else 'VIOLATION'}")

    print("\nRESULT:", "COMPLIANT" if ok else "NON-COMPLIANT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
