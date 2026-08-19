# Provenance Density: IJCAI-ECAI 2026 open-science release

Code, cached results, experimental materials, and a privacy-reduced participant
dataset for **“Beyond ‘Made with AI’: Visualizing Provenance Density to
Mitigate the Transparency Penalty.”** The paper appears in the IJCAI-ECAI 2026
Special Track on Human-Centred AI.

- [Official accepted-paper listing](https://2026.ijcai.org/accepted-papers/?ijtrack=special-track-on-human-centred-ai)
- [Paper PDF](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2026/HC13.pdf)

## What is included

```text
code/       audit, labeling, cached-data analysis, and user-study scripts
data/       cached audit data, study materials, and pseudonymized ratings
results/    tabular outputs and paper figures
interface/  Wizard-of-Oz stimulus renderer
```

See `code/README_code.md` for the file-to-method map. The 50 records called
`Dynamic_FreshQA` are author-created, FreshQA-style probes; they are not an
official FreshQA redistribution. TruthfulQA attribution is in
`THIRD_PARTY_NOTICES.md`.

## Reproduce cached analyses

Python 3.10–3.12 is supported. From the repository root:

```bash
git clone https://github.com/artisticsciencex/ijcai-ecai-2026-provenance-density.git
cd ijcai-ecai-2026-provenance-density
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-lock.txt

python code/roc_analysis.py
python code/sensitivity_sweep.py
python code/adversarial_probe.py
python code/adversarial_figure.py
```

The scripts resolve paths from their own locations, so they can be launched
from any working directory. They read `data/` and write `results/`.

### User-study model

The released long-form data contains 81 participants and 243 observations.
The paper's mixed-effects specification is implemented directly as
`rating ~ interface * veracity + (1 | participant_id)`:

```bash
Rscript code/install_r_dependencies.R
Rscript code/user_study_analysis.R
```

This writes the likelihood-ratio interaction test, estimated marginal means,
Tukey-adjusted contrasts, cell summaries, and discernment effect sizes to
`results/`. Exact age, gender, and AI-use fields are not needed by this model
and were removed from the public file after a disclosure-risk review.

## Re-run the API audit

The cached results require no credentials. A new audit sends data to external
services and incurs cost. Store credentials in an environment variable or a
secret manager; never put them in source code, notebooks, chat, or screenshots.

```bash
export OPENAI_API_KEY="your-key"
export SERPER_API_KEY="your-key"
python code/run_audit.py --n-tqa 5 --n-fresh 5 --seed 0 \
  --output data/demo_results.csv
```

Omit the size flags for the N=200 run. New runs record all stochastic samples,
provider response metadata, retrieval snippets and URLs, timestamps, the
dataset seed, and pinned Hugging Face revisions. A partial CSV is updated after
each successful row so an interrupted run can resume. These additions improve
traceability; provider-side models, search indexes, and API behavior can still
change, so byte-identical reproduction is not promised.

Pinned resources:

- Generator/segmenter: `gpt-4o-mini-2024-07-18`
- Judge: `gpt-4o-2024-11-20`
- NLI revision: `b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7`
- TruthfulQA revision: `741b8276f2d1982aa3d5b832d3ee81ed3b896490`

### External data flow

- OpenAI receives audit questions and generated answers; the segmenter also
  receives the answer being decomposed. Judge reruns receive question, answer,
  and reference-answer fields.
- Serper receives claim-derived search queries and returns result URLs and
  snippets. Do not audit confidential, personal, or embargoed text without an
  approved data-processing basis.
- Hugging Face is contacted to download the pinned TruthfulQA dataset and NLI
  model. Participant data is not sent by these scripts.

Review the providers' current privacy and retention terms before running the
pipeline, including the [OpenAI API data-usage policy](https://platform.openai.com/docs/guides/your-data)
and [Serper privacy policy](https://serper.dev/privacy-policy).

## Judge-label limitations

`data/tqa_judge_labels.csv` is a previously generated artifact. TruthfulQA rows
were judged with curated upstream references. The 50 dynamic rows were judged
closed-book and are **exploratory labels, not independently verified ground
truth**; do not use their metrics as confirmatory evidence.

Future runs of `code/llm_judge_labeling.py` fail closed for dynamic rows unless
`data/dynamic_probe_references.csv` supplies curated references. The explicit
`--allow-closed-book-dynamic` option exists only for exploratory analysis and
adds a label-status field to new output. Static TruthfulQA analysis is the
reference-grounded evaluation.

## Privacy and research ethics

`data/user_study_data.csv` is participant-level and pseudonymized; it is not
anonymous or aggregated. Participant codes are retained only to model repeated
measurements. Direct platform identifiers, timestamps, IP addresses, free text,
and demographic quasi-identifiers are not included. See `data/PRIVACY.md` and
`HUMAN-DATA-USE-NOTICE.txt` before using the file.

The consent text is provided for protocol transparency. This repository does
not publish raw recruitment-platform exports. Maintainers should add the exact
institutional review identifier to this README when it is appropriate and
verified; no approval identifier is inferred here.

## Security

No API key, access token, or private key is required to inspect cached results.
Automated tests cover domain-boundary spoofing, empty-keyword fail-closed
behavior, public-data schema, and absolute-path leakage. Please report a
security issue privately as described in `SECURITY.md`.

## Licensing

This is a multi-license repository:

- Project code: MIT (`LICENSE-CODE.txt`).
- Non-human project data, figures, and documentation: CC BY 4.0, with the
  exclusions listed in `LICENSE-DATA.txt`.
- Human-participant ratings: `HUMAN-DATA-USE-NOTICE.txt`.
- TruthfulQA-derived questions and references, including fields embedded in
  mixed audit CSV rows: upstream Apache 2.0 terms and attribution in
  `LICENSE-TRUTHFULQA-APACHE-2.0.txt` and `THIRD_PARTY_NOTICES.md`.

The root `LICENSE` is an overview; it does not replace the component terms.

## Citation

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

Contact: Qing Zhang (`qzkiyoshi@gmail.com`).
