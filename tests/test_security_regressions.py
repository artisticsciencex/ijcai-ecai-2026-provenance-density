from __future__ import annotations

import csv
import re
import statistics
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from source_scoring import _reputation_for, score_url  # noqa: E402


class SourceScoringSecurityTests(unittest.TestCase):
    def test_domain_boundaries(self) -> None:
        self.assertEqual(_reputation_for("reuters.com"), 1.0)
        self.assertEqual(_reputation_for("www.reuters.com"), 1.0)
        self.assertEqual(_reputation_for("agency.gov"), 1.0)
        self.assertEqual(_reputation_for("nature.com.attacker.example"), 0.5)
        self.assertEqual(_reputation_for("my.gov.spoof.example.net"), 0.5)
        self.assertEqual(_reputation_for("wikipedia.org.attacker.example"), 0.5)

    def test_empty_keywords_fail_closed(self) -> None:
        score = score_url(
            "https://www.nih.gov/example",
            "A plausible but unrelated snippet.",
            set(),
        )
        self.assertEqual(score, 0.0)

    def test_keyword_matching_uses_token_boundaries(self) -> None:
        score = score_url(
            "https://example.org",
            "CurieSpam is unrelated.",
            {"Curie"},
        )
        self.assertEqual(score, 0.0)


class ReleaseHygieneTests(unittest.TestCase):
    def test_interface_does_not_insert_dynamic_html(self) -> None:
        html = (REPO_ROOT / "interface" / "stimuli_generator.html").read_text()
        self.assertNotIn(".innerHTML", html)

    def test_participant_schema_is_privacy_reduced(self) -> None:
        path = REPO_ROOT / "data" / "user_study_data.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            list(rows[0]),
            [
                "participant_id", "group", "topic", "interface", "veracity",
                "rating", "prior_knowledge",
            ],
        )
        self.assertEqual(len(rows), 243)
        self.assertEqual(len({row["participant_id"] for row in rows}), 81)

        expected_d = {"Control": -0.2087, "Binary": 0.6846, "PDI": 1.8201}
        for interface, expected in expected_d.items():
            true = [
                float(row["rating"])
                for row in rows
                if row["interface"] == interface and row["veracity"] == "True"
            ]
            hallucinated = [
                float(row["rating"])
                for row in rows
                if row["interface"] == interface and row["veracity"] == "Hallucinated"
            ]
            pooled_sd = (
                (
                    (len(true) - 1) * statistics.variance(true)
                    + (len(hallucinated) - 1) * statistics.variance(hallucinated)
                )
                / (len(true) + len(hallucinated) - 2)
            ) ** 0.5
            cohen_d = (statistics.mean(true) - statistics.mean(hallucinated)) / pooled_sd
            self.assertAlmostEqual(cohen_d, expected, places=4)

    def test_no_machine_specific_paths_or_obvious_secrets(self) -> None:
        forbidden_paths = ("/sessions/", "/Users/kiyoshi", ".claude/projects")
        secret_patterns = (
            re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
            re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
            re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
            re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        )
        suffixes = {
            ".py", ".R", ".md", ".txt", ".json", ".yml", ".yaml",
            ".html", ".csv", ".ipynb", ".toml", ".sh",
        }
        problems = []
        for path in REPO_ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix not in suffixes:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(marker in text for marker in forbidden_paths):
                problems.append(f"machine path in {path.relative_to(REPO_ROOT)}")
            if any(pattern.search(text) for pattern in secret_patterns):
                problems.append(f"possible secret in {path.relative_to(REPO_ROOT)}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
