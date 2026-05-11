"""End-to-end driver for the technical audit (Section 4 of the paper).

This module orchestrates the four other modules:
    config.py             — hyperparameters
    dataset.py            — composite TruthfulQA + FreshQA loader
    semantic_entropy.py   — P_int via DeBERTa-v3-large NLI
    source_scoring.py     — w(s, c) = Reputation · MatchRatio³
    provenance_density.py — D(T) aggregation (Eq. 1 / Algorithm 1)

Outputs `provenance_density_full_validation.csv` with per-row columns:
    type, question, answer, p_int, density, final_score, latency

Usage
-----
    export OPENAI_API_KEY=...
    export SERPER_API_KEY=...
    python run_audit.py                  # full N=200 run
    python run_audit.py --n-tqa 5 --n-fresh 5 --output demo.csv  # smoke test
"""
from __future__ import annotations
import argparse
import time
import pandas as pd
from openai import OpenAI

from config import (
    GENERATOR_MODEL, GENERATION_TEMPERATURE, MAX_ANSWER_TOKENS,
    NUM_SAMPLES_FOR_PINT,
    SAMPLE_SIZE_TRUTHFULQA, SAMPLE_SIZE_FRESH,
    get_openai_key,
)
from dataset import load_mixed_dataset
from semantic_entropy import SemanticEntropyCalculator
from provenance_density import calculate_density


def run_audit(
    n_tqa: int = SAMPLE_SIZE_TRUTHFULQA,
    n_fresh: int = SAMPLE_SIZE_FRESH,
    output: str = "provenance_density_full_validation.csv",
    seed: int | None = None,
    checkpoint_every: int = 20,
) -> pd.DataFrame:
    """Run the audit and return the per-row results as a DataFrame.

    The CSV is also written to disk and intermediate checkpoints are saved
    every `checkpoint_every` rows so a long run can recover from interruption.
    """
    client = OpenAI(api_key=get_openai_key())
    nli = SemanticEntropyCalculator()
    dataset = load_mixed_dataset(n_truthfulqa=n_tqa, n_fresh=n_fresh, seed=seed)

    print(f"\n🚀 STARTING AUDIT  (N = {len(dataset)})  on {nli.device}\n")
    rows: list[dict] = []
    t_total = time.time()

    for i, item in enumerate(dataset, start=1):
        t_row = time.time()
        question = item["question"]
        print(f"[{i}/{len(dataset)}] ({item['type']})\n  Q: {question}")

        # 1. Sample K stochastic generations from the auditee (τ=1)
        try:
            samples = [
                client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": question}],
                    temperature=GENERATION_TEMPERATURE,
                    max_tokens=MAX_ANSWER_TOKENS,
                ).choices[0].message.content
                for _ in range(NUM_SAMPLES_FOR_PINT)
            ]
        except Exception as e:
            print(f"  generator error: {e}")
            continue

        # 2. P_int from bidirectional NLI on the K samples
        p_int = nli.calculate_p_int(samples)

        # 3. Density on the first sample (the audited answer)
        density = calculate_density(samples[0], context_question=question)

        # 4. Compose D(T) per Eq. 1
        final = (1.0 - p_int) * density
        elapsed = time.time() - t_row
        print(f"  → D(T) = {final:.3f}   (P_int = {p_int:.2f},  density = {density:.2f},  t = {elapsed:.1f}s)")

        rows.append({
            "type": item["type"], "question": question, "answer": samples[0],
            "p_int": p_int, "density": density, "final_score": final,
            "latency": elapsed,
        })

        # Resume-friendly checkpoint
        if i % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(f"partial_results_{i}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"\n✅ done in {(time.time() - t_total) / 60:.1f} minutes")
    print(f"Saved {output}")
    print(df.groupby("type")[["final_score", "p_int", "density"]].mean().round(3))
    return df


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tqa",   type=int, default=SAMPLE_SIZE_TRUTHFULQA)
    ap.add_argument("--n-fresh", type=int, default=SAMPLE_SIZE_FRESH)
    ap.add_argument("--output",  type=str, default="provenance_density_full_validation.csv")
    ap.add_argument("--seed",    type=int, default=None)
    args = ap.parse_args()
    run_audit(n_tqa=args.n_tqa, n_fresh=args.n_fresh, output=args.output, seed=args.seed)


if __name__ == "__main__":
    _cli()
