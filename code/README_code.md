# `code/` — module map and verification guide

The audit pipeline is split into **one file per concept** so each piece of the
paper can be cross-checked against exactly one Python module. No file imports
business logic from another that isn't named in its docstring.

## File map

| File | Purpose | Maps to |
|---|---|---|
| `config.py` | All hyperparameters (β = 5.0, λ = 1.2, K = 5, reputation tiers, model identifiers) | Table 1 |
| `source_scoring.py` | `w(s, c) = Reputation(s) · MatchRatio(σ, K_c)³`. Owns the cubic exponent and the three-tier domain prior. | Section 3.1, Table 1 |
| `provenance_density.py` | Atomic-claim segmentation, top-3 search retrieval, per-claim sum, then `tanh((1/β) · Σ_c (Σ_s w)^λ)`. | Section 3.1 (Eq. 1), Algorithm 1 |
| `semantic_entropy.py` | P_int via bidirectional NLI on K stochastic samples. | Section 3.2 |
| `dataset.py` | TruthfulQA (sampled) + FreshQA (verbatim 50 probes) composite. | Section 4 |
| `run_audit.py` | End-to-end driver. `python run_audit.py` reproduces the published audit. | Section 4 |
| `audit_pipeline.py` | Compatibility shim that re-exports the public API of the modules above. | — |
| `demo.ipynb` | Minimal Colab entry-point — installs deps, sets keys via Colab Secrets, calls `run_audit`. **No business logic in the notebook.** | — |
| `parse_tqa.py` | One-off helper to fetch TruthfulQA's `correct_answers` / `incorrect_answers` lists. | Used by `llm_judge_labeling.py` |
| `llm_judge_labeling.py` | GPT-4o-as-judge labelling for ROC / PR analysis. | Section 4.1 |
| `roc_analysis.py` | Computes AUC / AP / 95% CI; renders Figure 2; writes `tqa_roc_metrics.csv`. | Section 4.1 |
| `sensitivity_sweep.py` | Post-hoc β sweep + 13-form aggregation ablation; renders Figure 3. | Section 4.2 |
| `adversarial_probe.py` | Natural adversarial analysis + synthetic stress test. | Section 4.4 |
| `adversarial_figure.py` | Renders Figure 4 from the probe outputs. | Section 4.4 |

## Verifying the implementation against the paper

Each `.py` file has a `__main__` block that runs in **a few seconds** with no
API calls. They exercise the algebra in isolation so a reader can confirm the
implementation matches the paper's claims:

```bash
python source_scoring.py        # cubic relevance, 3-tier reputation prior
python provenance_density.py    # density aggregation matches Eq. 1
python semantic_entropy.py      # P_int aggregation logic
python dataset.py               # bundled FreshQA probes (50 items)
```

To reproduce the published audit ($N = 200$, ~80 min, ~$3 in API calls):

```bash
export OPENAI_API_KEY=...        # rotate yours; never commit
export SERPER_API_KEY=...
python run_audit.py
```

Smaller smoke test (~5 min, ~$0.05):

```bash
python run_audit.py --n-tqa 5 --n-fresh 5 --output demo_results.csv --seed 0
```

## Why the split

Earlier versions of the open-science release shipped the audit as a single
notebook with ~14 iterations of `calculate_density` evolving through the
prototype phase. That made it easy for a careful reader to grab an early cell
and conclude (incorrectly) that the cubic MatchRatio or β were "absent from
the code." This module split eliminates that ambiguity: there is one file per
concept, one definition per function, and the file name names the paper claim
it implements.
