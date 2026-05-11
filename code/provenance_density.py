"""Provenance Density  D(T)  — Equation 1 of the paper.

This module implements the per-claim aggregation of `score_url` into the full
density signal. Behaviour is byte-identical to cell 22 of the original notebook.

Eq. 1 (per the executed code, matching Algorithm 1):

    D(T)  =  (1 − P_int)  ·  tanh( (1/β) · Σ_c ( Σ_s w(s, c) )^λ )

P_int comes from `semantic_entropy.py`; w(s, c) from `source_scoring.py`.
This file owns only the claim-segmentation, retrieval, and aggregation steps.
"""
from __future__ import annotations
import requests
import numpy as np
from openai import OpenAI

from config import (
    BETA, LAMBDA, TOP_K_SEARCH,
    SEGMENTER_MODEL, SEGMENTER_TEMPERATURE,
    get_openai_key, get_serper_key,
)
from source_scoring import extract_rare_keywords, score_url

# Lazily constructed; only created when calculate_density is called.
_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=get_openai_key())
    return _openai_client


# ---------------------------------------------------------------------------
# Claim segmentation
# ---------------------------------------------------------------------------
SEGMENTER_PROMPT = (
    "Break the following text into individual, checkable factual claims. "
    "Return them as a bulleted list. Text: '{text}'"
)


def extract_atomic_claims(text: str) -> list[str]:
    """Use the segmenter LLM to extract atomic factual claims from `text`.

    The segmenter is `gpt-4o-mini` at temperature 0 (Section 3.2). Although it
    shares an architecture with the audited generator, it operates over the
    auditee's frozen output and is therefore a post-hoc parser.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        response = _get_openai_client().chat.completions.create(
            model=SEGMENTER_MODEL,
            temperature=SEGMENTER_TEMPERATURE,
            messages=[{"role": "user", "content": SEGMENTER_PROMPT.format(text=text)}],
        )
        body = response.choices[0].message.content or ""
    except Exception as e:
        print(f"  segmenter error: {e}")
        return []

    claims = [
        line.replace("- ", "").strip()
        for line in body.split("\n")
        if line.strip().startswith("-")
    ]
    return claims


# ---------------------------------------------------------------------------
# Search retrieval
# ---------------------------------------------------------------------------
SERPER_URL = "https://google.serper.dev/search"


def search_google(query: str) -> list[dict]:
    """Top-`TOP_K_SEARCH` organic results from Serper for `query`."""
    headers = {"X-API-KEY": get_serper_key(), "Content-Type": "application/json"}
    try:
        r = requests.post(SERPER_URL, headers=headers,
                          json={"q": query, "num": TOP_K_SEARCH}, timeout=15)
        return r.json().get("organic", []) or []
    except Exception as e:
        print(f"  search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Density aggregation — Equation 1
# ---------------------------------------------------------------------------
def calculate_density(text: str, context_question: str = "") -> float:
    """Compute the density component of Eq. 1 for one audited answer.

    The full D(T) score is `(1 - P_int) * calculate_density(...)`; P_int is
    computed separately by `semantic_entropy.SemanticEntropyCalculator`.

    Aggregation form (matches the code in cell 22):
        density(T)  =  tanh( (1/β) · Σ_c ( Σ_s w(s, c) )^λ )

    Args
    ----
    text:             The audited answer T.
    context_question: The original question Q. Used to disambiguate retrieval
                      for short claims (< 8 tokens), per Algorithm 1 line 4.
    """
    claims = extract_atomic_claims(text)
    if not claims:
        return 0.0

    total_score = 0.0
    for claim in claims:
        if len(claim) < 5:
            continue                              # discard scaffolding-only fragments

        keywords = extract_rare_keywords(claim)

        # Short claims rarely retrieve well in isolation — concatenate the
        # original question to disambiguate.
        if len(claim.split()) < 8:
            query = f"{context_question} {claim}".strip()
        else:
            query = claim

        results = search_google(query)
        # Per-claim source sum, then raised to LAMBDA to favour concentrated
        # corroboration. (See Section 3.1 — Concentration Exponent.)
        claim_score = sum(
            score_url(r.get("link", ""), r.get("snippet", ""), keywords)
            for r in results
        )
        total_score += claim_score ** LAMBDA

    # tanh saturation — bounded in [0, 1)
    return float(np.tanh(total_score / BETA))


# ---------------------------------------------------------------------------
# Self-check that doesn't hit any API (validates Eq. 1 algebra in isolation)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock the inner functions and verify the aggregation formula.
    import provenance_density as pd_mod
    pd_mod.extract_atomic_claims = lambda t: ["claim A", "claim B"]
    # claim A → 2 sources of w=0.5 each → claim_score = 1.0
    # claim B → 1 source of w=0.5      → claim_score = 0.5
    pd_mod.search_google = lambda q: ([{"link": "x", "snippet": "y"}, {"link": "x", "snippet": "y"}]
                                       if "A" in q else [{"link": "x", "snippet": "y"}])
    pd_mod.score_url = lambda u, s, k: 0.5

    expected = float(np.tanh((1.0 ** LAMBDA + 0.5 ** LAMBDA) / BETA))
    got = calculate_density("dummy text", context_question="A")
    # context_question="A" makes BOTH claims retrieve the "A" branch (because
    # query = Q ‖ claim and Q contains 'A'). Adjust the expected math:
    expected = float(np.tanh((1.0 ** LAMBDA + 1.0 ** LAMBDA) / BETA))
    assert abs(got - expected) < 1e-9, (got, expected)
    print(f"  ✓ Aggregation matches Eq. 1: density = {got:.4f}")
