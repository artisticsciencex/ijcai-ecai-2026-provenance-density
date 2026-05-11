"""Configuration constants for the Provenance Density audit pipeline.

Every numeric value the paper quotes is defined here, so a reader who wants to
verify the implementation against Section 3.1 / Table 1 can do so by looking at
exactly one file. Behavioural code lives in the other modules.

Cross-reference to the paper:
- BETA, LAMBDA  → Section 3.1 (Saturation scalar β, Concentration exponent λ)
- HIGH_TRUST / MID-tier collapse / LOW_TRUST → Section 3.2, Table 1 (Reputation prior)
- TOP_K_SEARCH, NUM_SAMPLES_FOR_PINT, NLI_THRESHOLD, MAX_ANSWER_TOKENS → Table 1
- GENERATOR_MODEL, JUDGE_MODEL, NLI_MODEL_NAME → Section 4 Implementation Details

The behavioural module that uses each constant is named in the comment.
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
GENERATOR_MODEL = "gpt-4o-mini"          # The audited LLM (run_audit.py)
SEGMENTER_MODEL = "gpt-4o-mini"          # Same checkpoint, τ=0 (provenance_density.py)
JUDGE_MODEL     = "gpt-4o"               # GPT-4o-mini's stronger sibling, judge for ROC labelling
                                         # (llm_judge_labeling.py)
NLI_MODEL_NAME  = (
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)                                        # Used by semantic_entropy.py

# ---------------------------------------------------------------------------
# Hyperparameters of D(T)  — must match Table 1 in the paper exactly
# ---------------------------------------------------------------------------
BETA  = 5.0    # Saturation scalar in tanh(total / β)              (provenance_density.py)
LAMBDA = 1.2   # Per-claim concentration exponent: claim_score**λ  (provenance_density.py)
TOP_K_SEARCH = 3                         # Top-k organic results per claim (provenance_density.py)

# Three-tier reputation prior. CUBIC matchratio penalty is applied separately.
HIGH_TRUST_DOMAINS = [
    ".gov", ".edu", "wikipedia.org", "nih.gov",
    "reuters", "apnews", "nature.com",
]
LOW_TRUST_DOMAINS = [
    "reddit.com", "quora.com", "medium.com", "twitter.com",
]
HIGH_TRUST_REPUTATION = 1.0
LOW_TRUST_REPUTATION  = 0.1
DEFAULT_REPUTATION    = 0.5
MATCHRATIO_EXPONENT   = 3                # Cubic Penalty in Section 3.1

# ---------------------------------------------------------------------------
# Internal consistency (P_int)  — see semantic_entropy.py
# ---------------------------------------------------------------------------
NUM_SAMPLES_FOR_PINT = 5
NLI_CONTRADICTION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Audit generation
# ---------------------------------------------------------------------------
GENERATION_TEMPERATURE = 1.0    # Audit sampling — see run_audit.py
SEGMENTER_TEMPERATURE  = 0.0    # Deterministic claim segmentation
MAX_ANSWER_TOKENS = 150         # Cap each audited answer length

# ---------------------------------------------------------------------------
# Dataset sizes  — see dataset.py
# ---------------------------------------------------------------------------
SAMPLE_SIZE_TRUTHFULQA = 150    # Static knowledge sample
SAMPLE_SIZE_FRESH      = 50     # Dynamic / FreshQA sample

# ---------------------------------------------------------------------------
# API keys — read from environment, never hardcoded.
# Set them with `export OPENAI_API_KEY=...` and `export SERPER_API_KEY=...`
# (See README §Quick reproduction guide.)
# ---------------------------------------------------------------------------
def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return key


def get_serper_key() -> str:
    key = os.getenv("SERPER_API_KEY", "")
    if not key:
        raise RuntimeError("SERPER_API_KEY is not set in the environment.")
    return key
