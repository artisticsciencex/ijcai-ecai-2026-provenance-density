"""Export the pinned TruthfulQA generation split to a reference CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from config import TRUTHFULQA_DATASET_REVISION


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "truthfulqa_references.csv"


def export_truthfulqa(output: Path = DEFAULT_OUTPUT) -> Path:
    dataset = load_dataset(
        "truthfulqa/truthful_qa",
        "generation",
        split="validation",
        revision=TRUTHFULQA_DATASET_REVISION,
    )
    rows = []
    for index, item in enumerate(dataset):
        rows.append({
            "idx": index,
            "type": item.get("type", ""),
            "category": item.get("category", ""),
            "question": item["question"],
            "best_answer": item.get("best_answer", ""),
            "correct_answers": json.dumps(item.get("correct_answers", []), ensure_ascii=False),
            "incorrect_answers": json.dumps(item.get("incorrect_answers", []), ensure_ascii=False),
            "source": item.get("source", ""),
            "dataset_id": "truthfulqa/truthful_qa",
            "dataset_revision": TRUTHFULQA_DATASET_REVISION,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} rows to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_truthfulqa(args.output)


if __name__ == "__main__":
    main()
