"""Unit + integration tests for the Redrob ranker. Pure stdlib (unittest) — no extra deps.

Run:  python -m unittest discover -s tests -v   (from the repo root)

Covers the deterministic logic the live interview will probe: metric correctness, honeypot rules,
structural features (incl. the recency-anchoring bug), alias expansion, reasoning safety
(no hallucination / length / determinism), and submission invariants over the real cache if present.
"""

from __future__ import annotations

import math
import sys
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

import aliases  # noqa: E402
import features  # noqa: E402
import honeypots  # noqa: E402
import metrics  # noqa: E402
import reasoning  # noqa: E402


def _skill(name, prof="expert", dur=0):
    return {"name": name, "proficiency": prof, "endorsements": 0, "duration_months": dur}


def _candidate(**over):
    """Minimal schema-valid-ish candidate; override any sub-tree via kwargs."""
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {"years_of_experience": 7.0, "current_title": "ML Engineer",
                    "current_company": "Acme", "country": "India", "summary": "", "headline": ""},
        "career_history": [{"company": "Acme", "title": "ML Engineer", "start_date": "2020-01-01",
                            "end_date": None, "duration_months": 60, "is_current": True,
                            "industry": "Tech", "description": "Built retrieval systems."}],
        "skills": [], "redrob_signals": {},
    }
    base.update(over)
    return base


class TestMetrics(unittest.TestCase):
    def test_dcg_known(self):
        # rels [3,2,3], gains [7,3,7], discounts [1, 1/log2(3), 1/log2(4)=0.5]
        expected = 7 / 1 + 3 / math.log2(3) + 7 / 2
        self.assertAlmostEqual(metrics.dcg_at_k([3, 2, 3], 3), expected, places=6)

    def test_ndcg_perfect_and_zero(self):
        self.assertAlmostEqual(metrics.ndcg_at_k([4, 3, 2, 1], 4), 1.0, places=6)
        self.assertEqual(metrics.ndcg_at_k([0, 0, 0], 3), 0.0)

    def test_ndcg_worse_than_ideal(self):
        self.assertLess(metrics.ndcg_at_k([0, 3, 4], 3), 1.0)

    def test_precision_at_k(self):
        self.assertAlmostEqual(metrics.precision_at_k([4, 0, 3, 2], 4), 0.5)  # tiers>=3: items 0,2

    def test_average_precision(self):
        # rels [3,0,3], R=2: hits at ranks 1 (P=1) and 3 (P=2/3) -> (1 + 0.667)/2
        self.assertAlmostEqual(metrics.average_precision([3, 0, 3], 2), (1.0 + 2 / 3) / 2, places=6)

    def test_composite_weights_sum(self):
        self.assertAlmostEqual(metrics.composite_score(1, 1, 1, 1), 1.0, places=6)


class TestHoneypots(unittest.TestCase):
    def test_experience_inflation_fires(self):
        c = _candidate(profile={"years_of_experience": 2.0}, career_history=[
            {"duration_months": 120, "is_current": True}])  # 10y career vs 2y stated
        self.assertTrue(honeypots.experience_inflation_flag(c))

    def test_experience_inflation_clean(self):
        c = _candidate(profile={"years_of_experience": 5.0}, career_history=[
            {"duration_months": 60, "is_current": True}])
        self.assertFalse(honeypots.experience_inflation_flag(c))

    def test_expert_skill_overload(self):
        c = _candidate(skills=[_skill("a"), _skill("b"), _skill("c")])  # 3 expert, 0mo
        self.assertTrue(honeypots.expert_skill_overload_flag(c))
        c2 = _candidate(skills=[_skill("a"), _skill("b")])  # only 2
        self.assertFalse(honeypots.expert_skill_overload_flag(c2))

    def test_missing_yoe_does_not_flag(self):
        c = _candidate(profile={}, career_history=[{"duration_months": 200, "is_current": True}])
        self.assertFalse(honeypots.experience_inflation_flag(c))  # missing data never gates

    def test_overclaim_strength(self):
        c = _candidate(
            skills=[_skill("NLP", "expert", 12)],
            redrob_signals={"skill_assessment_scores": {"NLP": 30.0}})  # claims expert, scores 30
        self.assertGreater(honeypots.assessment_overclaim_strength(c), 0.0)
        # no assessment -> strictly neutral
        c2 = _candidate(skills=[_skill("NLP", "expert", 12)], redrob_signals={})
        self.assertEqual(honeypots.assessment_overclaim_strength(c2), 0.0)


class TestFeatures(unittest.TestCase):
    REF = date(2026, 5, 27)

    def test_experience_band_fit(self):
        self.assertEqual(features.experience_band_fit(7.0), 1.0)
        self.assertEqual(features.experience_band_fit(None), 1.0)  # unknown is neutral
        self.assertLess(features.experience_band_fit(13.0), 1.0)

    def test_consulting_only(self):
        c = _candidate(career_history=[{"company": "Infosys"}, {"company": "TCS"}])
        self.assertTrue(features.consulting_only_career(c))
        c2 = _candidate(career_history=[{"company": "Infosys"}, {"company": "Google"}])
        self.assertFalse(features.consulting_only_career(c2))  # one product role exempts

    def test_recency_anchored_not_wallclock(self):
        # last_active far in the dataset's past relative to REF, but only ~ months — must use REF,
        # not datetime.now() (which would inflate days massively and is non-deterministic).
        c = _candidate(redrob_signals={"last_active_date": "2026-05-01"})
        days = features._days_since_last_active(c, self.REF)
        self.assertEqual(days, 26.0)
        # unknown -> -1.0 (neutral), never a huge wall-clock number
        self.assertEqual(features._days_since_last_active(_candidate(redrob_signals={}), self.REF), -1.0)

    def test_closed_source_guarded(self):
        # 7y, no github, no validation token -> fires
        c = _candidate(profile={"years_of_experience": 7.0},
                       career_history=[{"title": "Engineer", "description": "internal systems"}],
                       skills=[], redrob_signals={"github_activity_score": -1})
        self.assertGreater(features.closed_source_strength(c), 0.0)
        # same but github linked -> does not fire (external validation exists)
        c2 = _candidate(profile={"years_of_experience": 7.0},
                        career_history=[{"title": "Engineer", "description": "internal systems"}],
                        redrob_signals={"github_activity_score": 70})
        self.assertEqual(features.closed_source_strength(c2), 0.0)


class TestAliases(unittest.TestCase):
    def test_whole_word_expansion(self):
        out = aliases.expand("Built a RAG pipeline on Pinecone")
        self.assertIn("retrieval augmented generation", out)
        self.assertIn("vector database", out)
        self.assertTrue(out.startswith("Built a RAG pipeline"))  # original preserved

    def test_no_substring_false_match(self):
        # "rag" must not expand inside "storage"/"dragon"
        self.assertNotIn("retrieval augmented generation", aliases.expand("storage dragon"))


class TestReasoning(unittest.TestCase):
    def test_no_hallucination_and_length(self):
        c = _candidate(profile={"years_of_experience": 6.0, "current_title": "NLP Engineer",
                                "current_company": "Zeta", "summary": "", "headline": ""},
                       skills=[_skill("Embeddings", "expert", 24)],
                       redrob_signals={"recruiter_response_rate": 0.7})
        r = reasoning.generate_reasoning(c, ["embeddings_retrieval_production"], [], 5.0)
        self.assertLessEqual(len(r), 250)
        self.assertIn("Zeta", r)            # company is a real field
        self.assertIn("Embeddings", r)      # skill is a real field

    def test_determinism(self):
        c = _candidate()
        a = reasoning.generate_reasoning(c, ["strong_python"], ["notice period 90d"], 10.0)
        b = reasoning.generate_reasoning(c, ["strong_python"], ["notice period 90d"], 10.0)
        self.assertEqual(a, b)  # deterministic per candidate_id


class TestSubmissionInvariants(unittest.TestCase):
    """Integration test over the REAL cache if precompute.py has been run."""

    @classmethod
    def setUpClass(cls):
        import rank
        from config import CACHE_DIR
        if not (CACHE_DIR / "candidate_ids.csv").exists():
            raise unittest.SkipTest("cache/ not present — run precompute.py first")
        cls.rank = rank
        cls.cache = rank.load_cache()
        cls.scores = rank.compute_scores(cls.cache)
        cls.top = rank.select_top(cls.cache, cls.scores)

    def test_exactly_100_unique(self):
        ids = [self.cache["ids"][i] for i in self.top]
        self.assertEqual(len(ids), 100)
        self.assertEqual(len(set(ids)), 100)

    def test_scores_non_increasing(self):
        s = [self.scores["final"][i] for i in self.top]
        for a, b in zip(s, s[1:]):
            self.assertGreaterEqual(a, b)

    def test_tie_break_candidate_id_ascending(self):
        ids = [self.cache["ids"][i] for i in self.top]
        s = [self.scores["final"][i] for i in self.top]
        for k in range(len(s) - 1):
            if s[k] == s[k + 1]:
                self.assertLess(ids[k], ids[k + 1])

    def test_no_honeypot_in_top100(self):
        hp = self.cache["honeypots"]["is_honeypot"]
        self.assertEqual(sum(hp[i] > 0.5 for i in self.top), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
