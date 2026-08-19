# Code map and verification guide

| File | Purpose |
|---|---|
| `config.py` | Models, pinned revisions, hyperparameters, and reputation tiers |
| `source_scoring.py` | Domain-boundary validation and cubic contextual source score |
| `provenance_density.py` | Untrusted-text claim segmentation, retrieval, and D(T) trace |
| `semantic_entropy.py` | Bidirectional-NLI internal inconsistency score |
| `dataset.py` | Seeded TruthfulQA sample plus 50 author-created dynamic probes |
| `run_audit.py` | Resume-safe end-to-end audit with provenance metadata |
| `parse_tqa.py` | Export the pinned TruthfulQA reference split |
| `llm_judge_labeling.py` | Reference-grounded judge; dynamic rows fail closed by default |
| `roc_analysis.py` | Cached ROC/PR analysis and figures |
| `sensitivity_sweep.py` | Parameter and aggregation sensitivity analysis |
| `adversarial_probe.py` | Natural analysis and security regression scenarios |
| `user_study_analysis.R` | Paper-specified LMM, LRT, EMMs, Tukey contrasts, effect sizes |

Run the offline implementation checks from the repository root:

```bash
python code/source_scoring.py
python code/provenance_density.py
python code/semantic_entropy.py
python -m compileall -q code
python -m pytest -q
```

`dataset.py` contacts Hugging Face even in its example block. The complete API
audit additionally requires OpenAI and Serper credentials; review the external
data-flow section in the root README before running it.
