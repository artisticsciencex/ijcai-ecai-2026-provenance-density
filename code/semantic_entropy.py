"""Internal consistency  P_int  via bidirectional NLI on K stochastic samples.

This module implements the "Consistency Veto" (1 − P_int) of Eq. 1.
Behaviour is byte-identical to cell 22 of the original notebook.

Method (Section 3.2 + Farquhar et al., 2024):
    1. Sample K = NUM_SAMPLES_FOR_PINT generations from the auditee at τ = 1.
    2. For every unordered pair (i, j), check NLI both ways
       (i entails j AND j entails i, i.e. neither direction is a contradiction
       at the NLI_CONTRADICTION_THRESHOLD).
    3. Define ρ_consistent as the fraction of pairs that are bidirectionally
       non-contradictory, and P_int = 1 − ρ_consistent.

P_int is high when the model's K samples disagree (model is "confused"),
low when they agree.
"""
from __future__ import annotations
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import NLI_MODEL_NAME, NLI_CONTRADICTION_THRESHOLD


class SemanticEntropyCalculator:
    """Compute P_int from a list of stochastic samples."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NLI model: {NLI_MODEL_NAME} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL_NAME, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Locate the index of the "contradiction" label dynamically — different
        # NLI checkpoints order their classes differently.
        self.label_map = self.model.config.label2id
        self.contradiction_id = -1
        for label, idx in self.label_map.items():
            if label.lower() == "contradiction":
                self.contradiction_id = idx
                break
        if self.contradiction_id == -1:
            self.contradiction_id = 2  # safe fallback for MoritzLaurer DeBERTa

    def check_implication(self, premise: str, hypothesis: str) -> bool:
        """Return True iff the model considers `premise` to NOT contradict
        `hypothesis` at the configured contradiction threshold."""
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            contradiction_score = float(probs[0][self.contradiction_id].item())
        return contradiction_score < NLI_CONTRADICTION_THRESHOLD

    def calculate_p_int(self, samples: list[str]) -> float:
        """Return P_int = 1 − ρ_consistent over all unordered pairs of samples."""
        if not samples or len(samples) < 2:
            return 0.0
        n = len(samples)
        consistent = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if (self.check_implication(samples[i], samples[j])
                        and self.check_implication(samples[j], samples[i])):
                    consistent += 1
        if total == 0:
            return 0.0
        return 1.0 - consistent / total


if __name__ == "__main__":
    # Smoke-test only the aggregation logic (no model load).
    class _Stub:
        def __init__(self, agree=True): self.agree = agree
        def check_implication(self, p, h): return self.agree
    SemanticEntropyCalculator.check_implication = _Stub(agree=True).check_implication
    sec = SemanticEntropyCalculator.__new__(SemanticEntropyCalculator)
    sec.check_implication = _Stub(agree=True).check_implication
    p_consistent = sec.calculate_p_int(["a", "a", "a"])
    assert p_consistent == 0.0
    sec.check_implication = _Stub(agree=False).check_implication
    p_inconsistent = sec.calculate_p_int(["a", "b"])
    assert p_inconsistent == 1.0
    print("  ✓ P_int aggregation matches Section 3.2 (consistent → 0, inconsistent → 1).")
