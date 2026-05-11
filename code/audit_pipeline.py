"""Compatibility shim — the audit pipeline has been split into focused modules.

If you have old code that imports from `audit_pipeline`, this file still works:
    from audit_pipeline import calculate_density, score_url, run_audit

For new readers, prefer the per-function modules instead:

    code/
    ├── config.py              Hyperparameters (β, λ, K, domain priors).
    ├── source_scoring.py      w(s, c) = Reputation · MatchRatio³  — Section 3.1.
    ├── provenance_density.py  D(T) = (1−P_int) · tanh((1/β)·Σ(Σw)^λ) — Eq. 1.
    ├── semantic_entropy.py    P_int via DeBERTa-v3-large NLI — Section 3.2.
    ├── dataset.py             TruthfulQA + FreshQA composite — Section 4.
    └── run_audit.py           End-to-end driver. `python run_audit.py` runs N=200.

Each file has a self-check at the bottom: `python <module>.py` exercises the
relevant logic in isolation.
"""
from source_scoring      import score_url, extract_rare_keywords            # noqa: F401
from provenance_density  import (                                           # noqa: F401
    calculate_density, extract_atomic_claims, search_google,
)
from semantic_entropy    import SemanticEntropyCalculator                   # noqa: F401
from dataset             import load_mixed_dataset, FRESH_SAMPLES           # noqa: F401
from run_audit           import run_audit                                   # noqa: F401

if __name__ == "__main__":
    run_audit()
