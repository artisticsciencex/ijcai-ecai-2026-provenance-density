# Provenance Density: Open-Science Release

This repository contains the data and the code described in the paper
**"Beyond \"Made with AI\": Visualizing Provenance Density to Mitigate the Transparency Penalty"**
by Qing Zhang, Yifei Huang, Juyoung Lee, Thad Starner, and Jun Rekimoto, to appear in
*Proceedings of the IJCAI-ECAI 2026 Special Track on: Human-Centred Artificial Intelligence:
Multidisciplinary Contours and Challenges of Next-Generation AI Research and Applications*,
<https://2026.ijcai.org/ijcai-ecai-2026-call-for-papers-human-centred-ai/>.

The release covers all three empirical components of the paper:
**(1)** the technical audit of the *Provenance Density* metric $D(T)$ on $N=200$ generations
(Section 4); **(2)** the LLM-as-judge labelling and ROC/PR analysis used for camera-ready
revisions (new Section 4.1); **(3)** the within-subjects user experiment with $N=81$
participants (Section 5).

## Repository layout

```
ijcai-ecai-2026-provenance-density/
├── README.md                                   ← this file
├── LICENSE-CODE.txt                            ← MIT, applies to all code
├── LICENSE-DATA.txt                            ← CC-BY-4.0, applies to data + figures
│
├── code/                                       ← see code/README_code.md for the
│   │                                              full module map and verification guide
│   ├── config.py                               ← All hyperparameters (Table 1)
│   ├── source_scoring.py                       ← w(s, c) = Reputation · MatchRatio³ (§3.1)
│   ├── provenance_density.py                   ← D(T) aggregation, Eq. 1
│   ├── semantic_entropy.py                     ← P_int via DeBERTa-v3-large NLI (§3.2)
│   ├── dataset.py                              ← TruthfulQA + FreshQA composite (§4)
│   ├── run_audit.py                            ← End-to-end driver
│   ├── audit_pipeline.py                       ← Compat shim re-exporting the modules
│   ├── demo.ipynb                              ← Minimal Colab demo (no inline logic)
│   ├── llm_judge_labeling.py                   ← GPT-4o judge for ROC labelling (§4.1)
│   ├── parse_tqa.py                            ← TruthfulQA reference fetcher
│   ├── roc_analysis.py                         ← ROC / PR / AUC + Figure 2 (§4.1)
│   ├── sensitivity_sweep.py                    ← β sweep + Figure 3 (§4.2)
│   ├── adversarial_probe.py                    ← Adversarial probe (§4.4)
│   ├── adversarial_figure.py                   ← Figure 4 (§4.4)
│   ├── README_code.md                          ← Per-file map of paper claims → code
│   └── requirements.txt                        ← Python dependencies
│
├── data/
│   ├── user_study_data.csv                     ← N=81 participant ratings, long format (de-identified)
│   ├── consent_protocol.txt                    ← Verbatim consent form shown to all participants
│   ├── experimental_stimuli.json               ← The 6 academic stimuli (3 topics × T/H)
│   ├── audit_full_validation.csv               ← N=200 audit log (clean run)
│   ├── audit_partial_chunks/                   ← Chronological 20-row checkpoints
│   ├── tqa_judge_labels.csv                    ← GPT-4o judge labels for 200 audited rows
│   └── truthfulqa_references.csv               ← TruthfulQA correct/incorrect lookup
│
├── interface/
│   └── stimuli_generator.html                  ← Live WoZ stimulus renderer
│
└── results/
    ├── tqa_roc_metrics.csv                     ← AUC + bootstrap CI by split / detector
    ├── sensitivity_beta_sweep.csv              ← AUC at β ∈ {0.5,1,2,3,5,7,10,15,25}
    ├── sensitivity_beta_ecological.csv         ← M_static vs M_dynamic gap by β
    ├── sensitivity_aggregation_form.csv        ← 13-form aggregation ablation
    ├── adv_natural_rescue.csv                  ← Veto rescue rate on natural adversarials
    ├── adv_synthetic_attacks.csv               ← D(T) inflation under five attacks
    └── figures/                                ← PDF + PNG of every figure in the paper
```

## Quick reproduction guide

### Prerequisites
- Python 3.10–3.12, ~5 GB free disk for the DeBERTa NLI model.
- Optional GPU strongly recommended (T4 or better) for the audit; CPU works but
  is ~10× slower.
- Two API keys (only needed if you re-run the audit and labelling — all cached
  outputs are in `data/`):
  - `OPENAI_API_KEY` — used by the auditee, claim segmenter, and judge.
  - `SERPER_API_KEY` — used by the audit's RAG step (<https://serper.dev>).

```bash
git clone https://github.com/<USER>/ijcai-ecai-2026-provenance-density.git
cd ijcai-ecai-2026-provenance-density
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

### Reproduce the camera-ready figures from cached data (no API calls, ~2 min)

```bash
# Figure 1 + Table 2 (ROC / PR / AUC)
python code/roc_analysis.py

# Figure 3 (β-sensitivity + ecological gap)
python code/sensitivity_sweep.py

# Figure 4 (adversarial probe — natural + synthetic)
python code/adversarial_probe.py
python code/adversarial_figure.py
```

All three scripts read from `data/` and write to `results/`. They reproduce
every number quoted in Section 4 of the paper from cached audit outputs.

### Reproduce the user-study analysis (no API calls)

```bash
# Convert raw .xlsx into long format + run linear mixed model + Tukey HSD
# (See cells 28-30 of code/demo.ipynb)
jupyter nbconvert --to notebook --execute code/demo.ipynb
```

### Re-run the full $N=200$ audit (~80 minutes wall time, ~$3 in API costs)

```bash
export OPENAI_API_KEY="sk-..."     # rotate yours, never commit
export SERPER_API_KEY="..."
cd code && python run_audit.py
```

To smoke-test on 10 questions instead (~5 min, ~$0.05):

```bash
cd code && python run_audit.py --n-tqa 5 --n-fresh 5 --seed 0 \
                              --output ../data/demo_results.csv
```

To verify the implementation against the paper's claims without making any
API calls, each module has a self-check at its bottom:

```bash
cd code
python source_scoring.py        # cubic relevance, 3-tier reputation prior
python provenance_density.py    # density aggregation matches Eq. 1
python semantic_entropy.py      # P_int aggregation logic
python dataset.py               # bundled FreshQA probes (50 items)
```

This regenerates `data/audit_full_validation.csv` and the partial-results
checkpoints. Results will not be byte-identical to ours due to LLM stochasticity,
but the means and ROC / sensitivity outcomes should reproduce within sampling
noise.

### Re-run the GPT-4o judge labelling (~6 min, ~$0.85)

```bash
export OPENAI_API_KEY="sk-..."
python code/llm_judge_labeling.py
```

Resumes automatically; checkpoints to `data/tqa_judge_labels.partial.csv` after
every row. Final output: `data/tqa_judge_labels.csv`.

## Notes on the data

### `data/audit_full_validation.csv`
Columns: `type`, `question`, `answer`, `p_int`, `density`, `final_score`, `latency`.
Each row is one audited generation from gpt-4o-mini at temperature 1.0,
with the metric $D(T) = (1-P_{int}) \cdot \text{density}$ computed against
the top-3 results from Google Serper. NLI consistency was scored with
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.

### `data/tqa_judge_labels.csv`
GPT-4o (temperature 0) was given each `(question, answer)` pair plus the
TruthfulQA dataset's official `correct_answers` and `incorrect_answers` lists
(when the row is from TruthfulQA). For FreshQA rows the judge worked
closed-book. Output schema documented at the top of
`code/llm_judge_labeling.py`. The judge is *generator-independent*: its model
is strictly stronger than the audited gpt-4o-mini, so this is not
self-grading. A 15-case manual spot-check confirmed the judge correctly
catches subtle TruthfulQA adversarials (e.g. "Rosa Parks" → "Claudette
Colvin", fabricated Nobel laureates) and accepts hedged refusals as truthful.

### `data/user_study_data.csv`
243 rows = 81 participants × 3 topic conditions. Columns: `participant_id`,
`group`, `topic`, `interface`, `veracity`, `rating`, `prior_knowledge`,
`age`, `gender`, `ai_usage`. Pseudonymous participant codes (`P_G1_01`, …);
no Prolific IDs, no timestamps, no IP addresses, no free-text comments that
could be used to re-identify a participant.

### `data/consent_protocol.txt`
The verbatim text of the informed-consent form presented to every participant
before they entered the study. We share this for transparency about the
experimental protocol; the raw Prolific-side survey exports are *not* included
in the release because they contain Prolific worker IDs and submission
timestamps which we explicitly promised participants would be stored
separately from their responses and never shared.

## Privacy & ethics

The user study (Section 5) was conducted in accordance with institutional
ethical review board approval. Participants were recruited via Prolific,
provided informed consent, and were compensated in line with fair labor
standards. Only de-identified and aggregated data is included in this
release. No personally identifying information was retained.

## Citation

If you build on this work, please cite the paper:

```bibtex
@inproceedings{zhang2026provenance,
  title     = {Beyond ``Made with AI'': Visualizing Provenance Density
               to Mitigate the Transparency Penalty},
  author    = {Zhang, Qing and Huang, Yifei and Lee, Juyoung and
               Starner, Thad and Rekimoto, Jun},
  booktitle = {Proceedings of the IJCAI-ECAI 2026 Special Track on
               Human-Centred Artificial Intelligence},
  year      = {2026}
}
```

## License

- **Code** (everything under `code/` and any inline `.py` / `.ipynb` snippets):
  MIT License — see `LICENSE-CODE.txt`.
- **Data, figures, README, and other written artifacts** (`data/`, `results/`,
  `interface/`, this README): Creative Commons
  Attribution 4.0 International (CC-BY-4.0) — see `LICENSE-DATA.txt`.

## Contact

Qing Zhang &lt;qzkiyoshi@gmail.com&gt; (corresponding author).
For paper-related queries, please contact the IJCAI-ECAI 2026 HAI Special
Track chairs.
