"""End-to-end driver for the technical audit (Section 4 of the paper).

This module orchestrates the four other modules:
    config.py             — hyperparameters
    dataset.py            — composite TruthfulQA + FreshQA loader
    semantic_entropy.py   — P_int via DeBERTa-v3-large NLI
    source_scoring.py     — w(s, c) = Reputation · MatchRatio³
    provenance_density.py — D(T) aggregation (Eq. 1 / Algorithm 1)

Outputs a CSV with the score columns plus stochastic samples, provider response
metadata, a per-claim retrieval trace, timestamps, seed, and pinned revisions.

Usage
-----
    export OPENAI_API_KEY=...
    export SERPER_API_KEY=...
    python run_audit.py                  # full N=200 run to data/audit_rerun.csv
    python run_audit.py --n-tqa 5 --n-fresh 5 --output demo.csv  # smoke test
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import pandas as pd
from openai import OpenAI

from config import (
    GENERATOR_MODEL, GENERATION_TEMPERATURE, MAX_ANSWER_TOKENS,
    NUM_SAMPLES_FOR_PINT,
    SAMPLE_SIZE_TRUTHFULQA, SAMPLE_SIZE_FRESH,
    NLI_MODEL_NAME, NLI_MODEL_REVISION, TRUTHFULQA_DATASET_REVISION,
    get_openai_key,
)
from dataset import load_mixed_dataset
from semantic_entropy import SemanticEntropyCalculator
from provenance_density import calculate_density_with_trace


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "data" / "audit_rerun.csv"


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Replace a CSV only after a complete same-directory temporary write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_audit(
    n_tqa: int = SAMPLE_SIZE_TRUTHFULQA,
    n_fresh: int = SAMPLE_SIZE_FRESH,
    output: str | Path = DEFAULT_OUTPUT,
    seed: int | None = 0,
    checkpoint_every: int = 20,
    resume: bool = True,
) -> pd.DataFrame:
    """Run the audit and return the per-row results as a DataFrame.

    The CSV is also written to disk and intermediate checkpoints are saved
    every `checkpoint_every` rows so a long run can recover from interruption.
    """
    output_path = Path(output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=get_openai_key())
    nli = SemanticEntropyCalculator()
    dataset = load_mixed_dataset(n_truthfulqa=n_tqa, n_fresh=n_fresh, seed=seed)

    print(f"\n🚀 STARTING AUDIT  (N = {len(dataset)})  on {nli.device}\n")
    rows: list[dict] = []
    if resume and partial_path.exists():
        rows = pd.read_csv(partial_path).to_dict("records")
        print(f"Resuming from {partial_path} with {len(rows)} completed rows.")
    done_questions = {row["question"] for row in rows}
    run_started_utc = datetime.now(timezone.utc).isoformat()
    t_total = time.time()

    for i, item in enumerate(dataset, start=1):
        t_row = time.time()
        question = item["question"]
        if question in done_questions:
            continue
        print(f"[{i}/{len(dataset)}] ({item['type']})\n  Q: {question}")

        # 1. Sample K stochastic generations from the auditee (τ=1)
        try:
            responses = [
                client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": question}],
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=MAX_ANSWER_TOKENS,
                )
                for _ in range(NUM_SAMPLES_FOR_PINT)
            ]
            samples = [response.choices[0].message.content or "" for response in responses]
        except Exception as exc:
            if rows:
                _write_csv_atomic(pd.DataFrame(rows), partial_path)
            raise RuntimeError(
                f"Generation failed for row {i} ({type(exc).__name__}); partial results were saved."
            ) from None

        # 2. P_int from bidirectional NLI on the K samples
        p_int = nli.calculate_p_int(samples)

        # 3. Density on the first sample (the audited answer)
        density, provenance_trace = calculate_density_with_trace(
            samples[0], context_question=question
        )

        # 4. Compose D(T) per Eq. 1
        final = (1.0 - p_int) * density
        elapsed = time.time() - t_row
        print(f"  → D(T) = {final:.3f}   (P_int = {p_int:.2f},  density = {density:.2f},  t = {elapsed:.1f}s)")

        rows.append({
            "type": item["type"], "question": question, "answer": samples[0],
            "p_int": p_int, "density": density, "final_score": final,
            "latency": elapsed,
            "stochastic_samples": json.dumps(samples, ensure_ascii=False),
            "generation_metadata": json.dumps([
                {
                    "model": getattr(response, "model", None),
                    "created": getattr(response, "created", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "request_id": getattr(response, "_request_id", None),
                }
                for response in responses
            ], ensure_ascii=False),
            "provenance_trace": json.dumps(provenance_trace, ensure_ascii=False),
            "audit_started_utc": run_started_utc,
            "dataset_seed": seed,
            "truthfulqa_revision": TRUTHFULQA_DATASET_REVISION,
            "nli_model": NLI_MODEL_NAME,
            "nli_revision": NLI_MODEL_REVISION,
        })
        done_questions.add(question)

        # Resume file updated atomically after every successful row.
        _write_csv_atomic(pd.DataFrame(rows), partial_path)

        # Resume-friendly checkpoint
        if i % checkpoint_every == 0:
            checkpoint = output_path.with_name(
                f"{output_path.stem}.checkpoint-{i}{output_path.suffix}"
            )
            _write_csv_atomic(pd.DataFrame(rows), checkpoint)

    df = pd.DataFrame(rows)
    _write_csv_atomic(df, output_path)
    if partial_path.exists():
        partial_path.unlink()
    print(f"\n✅ done in {(time.time() - t_total) / 60:.1f} minutes")
    print(f"Saved {output_path}")
    print(df.groupby("type")[["final_score", "p_int", "density"]].mean().round(3))
    return df


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tqa",   type=int, default=SAMPLE_SIZE_TRUTHFULQA)
    ap.add_argument("--n-fresh", type=int, default=SAMPLE_SIZE_FRESH)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    run_audit(
        n_tqa=args.n_tqa,
        n_fresh=args.n_fresh,
        output=args.output,
        seed=args.seed,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    _cli()
