# Third-party notices

## TruthfulQA

`data/truthfulqa_references.csv` is a structured redistribution of the
TruthfulQA generation dataset by Stephanie Lin, Jacob Hilton, and Owain Evans.
TruthfulQA question text also appears in rows marked `Static_TruthfulQA` in
`data/audit_full_validation.csv` and `data/tqa_judge_labels.csv`.
The upstream project is available at:

<https://github.com/sylinrl/TruthfulQA>

TruthfulQA is licensed under the Apache License, Version 2.0. The redistributed
question/reference material remains under that license and is not relicensed
under this repository's CC BY 4.0 terms. Project-generated model answers,
metrics, and judge fields in the mixed CSV files are separately licensed as
described in `LICENSE-DATA.txt`. See `LICENSE-TRUTHFULQA-APACHE-2.0.txt`.

The CSV changes representation and column names for local lookup. It does not
claim endorsement by the upstream authors.

## FreshQA terminology

The 50 dynamic questions in `code/dataset.py` are author-created,
FreshQA-style probes retained from the project's original audit notebook. The
official FreshQA dataset is not redistributed in this repository. FreshQA is
cited as methodological inspiration:

<https://github.com/freshllms/freshqa>
