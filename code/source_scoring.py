"""Source-relevance scoring  w(s, c)  =  Reputation(s) · MatchRatio(σ, K_c)^3.

This module is the single place where Section 3.1's "Cubic Contextual Weight"
lives. A reader can read this 60-line file end-to-end and verify it against
Section 3.1 of the paper without touching anything else.

Behaviour is byte-identical to cell 22 of the original audit notebook.

Functions
---------
extract_rare_keywords(text)  → set[str]
score_url(url, snippet, keywords) → float in [0, 1]
"""
from __future__ import annotations
import re
from urllib.parse import urlparse

from config import (
    HIGH_TRUST_DOMAINS, LOW_TRUST_DOMAINS,
    HIGH_TRUST_REPUTATION, LOW_TRUST_REPUTATION, DEFAULT_REPUTATION,
    MATCHRATIO_EXPONENT,
)

# Sentence-leading capitalised tokens like "The", "On", etc. would otherwise
# pollute the rare-keyword set. We treat them as stopwords.
_LEADING_CAP_STOPWORDS: frozenset[str] = frozenset({
    "The", "A", "An", "In", "On", "At", "To",
    "For", "Of", "And", "But", "Or", "Is", "Was", "Are",
})

# Match capitalised tokens of length ≥ 2 — i.e. likely proper nouns / named
# entities. Numbers and dashes are allowed inside the token.
_RARE_KEYWORD_RE = re.compile(r"\b[A-Z][a-zA-Z0-9-]+\b")


def extract_rare_keywords(text: str) -> set[str]:
    """Return the set of rare capitalised keywords in `text`.

    These are the K_c referred to in Section 3.1's MatchRatio formula.
    """
    if not isinstance(text, str):
        return set()
    return {w for w in _RARE_KEYWORD_RE.findall(text)
            if w not in _LEADING_CAP_STOPWORDS}


def _reputation_for(domain: str) -> float:
    """Three-tier reputation prior — Table 1 of the paper."""
    if any(x in domain for x in HIGH_TRUST_DOMAINS):
        return HIGH_TRUST_REPUTATION
    if any(x in domain for x in LOW_TRUST_DOMAINS):
        return LOW_TRUST_REPUTATION
    return DEFAULT_REPUTATION


def score_url(url: str, snippet: str, keywords: set[str]) -> float:
    """Compute  w(s, c)  =  Reputation(s) · MatchRatio(σ, K_c)^3.

    Args
    ----
    url:      The result URL (we use only its registered domain).
    snippet:  The result snippet returned by the search API.
    keywords: K_c — rare keywords of the claim being verified.

    Returns
    -------
    A non-negative float in [0, 1].
    """
    domain = urlparse(url or "").netloc.lower()
    reputation = _reputation_for(domain)

    if not keywords:
        # Empty-keyword fallback. NB: Section 4.4 (Adversarial Robustness)
        # discusses this as a known vulnerability and recommends a
        # conservative default ≤ 0.3 in future iterations.
        relevance = 1.0
    else:
        snip = (snippet or "").lower()
        matches = sum(1 for k in keywords if k.lower() in snip)
        match_ratio = matches / len(keywords)
        relevance = match_ratio ** MATCHRATIO_EXPONENT  # Cubic Penalty

    return reputation * relevance


# ---------------------------------------------------------------------------
# Tiny self-check — run `python source_scoring.py` to verify behaviour.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Cubic Contextual Weight worked example from Section 3.1
    #    "MatchRatio ≈ 0.5 → 0.5^3 = 0.125"
    keys = {"Polonium", "Radium", "Curie"}
    snippet = "Curie discovered Polonium in 1898."  # 2 of 3 keywords (≈ 0.67)
    high_trust_url = "https://en.wikipedia.org/wiki/Marie_Curie"
    score = score_url(high_trust_url, snippet, keys)
    expected = 1.0 * (2/3) ** 3
    assert abs(score - expected) < 1e-9, (score, expected)
    print(f"  ✓ Cubic relevance:  Reputation={1.0}, MatchRatio={2/3:.3f}, "
          f"score={score:.3f}  (expected {expected:.3f})")

    # 2. Empty-keyword fallback
    score_empty = score_url(high_trust_url, snippet, set())
    assert score_empty == 1.0
    print(f"  ✓ Empty-keyword fallback yields reputation × 1.0 = {score_empty}")

    # 3. Reputation tiers
    assert _reputation_for("www.reuters.com") == 1.0
    assert _reputation_for("medium.com") == 0.1
    assert _reputation_for("example.org") == 0.5
    print("  ✓ Reputation tiers (high/low/default) match Table 1.")
