"""W3.4 Adversarial-Robustness Probe.

Two complementary analyses:

PART A. NATURAL ADVERSARIAL ANALYSIS (real audit data, no API calls)
--------------------------------------------------------------------
Find rows where retrieval *already* found support for hallucinations
(density >= τ AND judge-labelled hallucinated). Quantify how often the
Consistency Veto rescues these "natural attacks" and characterize the
attack types from judge rationales.

PART B. SYNTHETIC STRESS TEST (offline simulation of score_url / density)
------------------------------------------------------------------------
Reimplement the audit pipeline's score_url and calculate_density without
API calls, feed it adversarial mocked search results, and measure how
much each attack inflates D(T) above its honest value.

Attacks tested:
  1. Citation laundering — high-trust domain returns content that
     doesn't actually contain the claim's keywords.
  2. Keyword stuffing — generic snippet that contains every rare
     keyword from the claim but in unrelated context.
  3. Empty-keyword exploit — claim contains zero capitalized rare
     keywords, hitting the `if not required_keywords: relevance = 1.0`
     fallback path in score_url.
  4. Domain spoofing — domain whose name *contains* a high-trust
     substring but is unrelated (e.g., wikipedia-fake.com).
  5. SEO inflation — a coordinated set of high-trust + high-relevance
     mocked sources for a fabricated claim.
"""
from __future__ import annotations
import re
import json
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd

ROOT = Path("/sessions/sleepy-zealous-goldberg/mnt/ijcai god")
OUT  = Path("/sessions/sleepy-zealous-goldberg/mnt/outputs")

# --------------------------------------------------------------------------- #
# Reimplementation of the relevant pipeline functions (audit_pipeline.py)
# --------------------------------------------------------------------------- #
HIGH_TRUST = ['.gov', '.edu', 'wikipedia.org', 'nih.gov', 'reuters', 'apnews', 'nature.com']
LOW_TRUST  = ['reddit.com', 'quora.com', 'medium.com', 'twitter.com']
STOP_CAP   = {"The","A","An","In","On","At","To","For","Of","And","But","Or","Is","Was","Are"}


def extract_rare_keywords(text: str) -> set[str]:
    words = re.findall(r'\b[A-Z][a-zA-Z0-9-]+\b', text)
    return {w for w in words if w not in STOP_CAP}


def score_url(url: str, snippet: str, keywords: set[str]) -> float:
    """Faithful port of audit_pipeline.py:score_url."""
    domain = urlparse(url).netloc.lower()
    if any(x in domain for x in HIGH_TRUST):
        rep = 1.0
    elif any(x in domain for x in LOW_TRUST):
        rep = 0.1
    else:
        rep = 0.5
    if not keywords:
        rel = 1.0  # ← empty-keyword fallback. Adversary-exploitable.
    else:
        snip = (snippet or "").lower()
        m = sum(1 for k in keywords if k.lower() in snip)
        rel = (m / len(keywords)) ** 3
    return rep * rel


def density_from_results(claims: list[str], mocked_search: dict[str, list[dict]],
                         beta: float = 5.0, lam: float = 1.2) -> float:
    """Faithful port of audit_pipeline.py:calculate_density (no API)."""
    total = 0.0
    for c in claims:
        if len(c) < 5:
            continue
        keys = extract_rare_keywords(c)
        results = mocked_search.get(c, [])
        cs = sum(score_url(r["url"], r["snippet"], keys) for r in results)
        total += cs ** lam
    return float(np.tanh(total / beta))


# --------------------------------------------------------------------------- #
# Part A — Natural adversarial analysis
# --------------------------------------------------------------------------- #
def natural_adversarial_analysis() -> dict:
    df = pd.read_csv(ROOT / "tqa_judge_labels.csv")
    halluc = df[df.hallucinated == 1]
    high_dens_halluc = df[(df.hallucinated == 1) & (df.density >= 0.7)]
    high_dens_truth  = df[(df.hallucinated == 0) & (df.density >= 0.7)]

    # Veto rescue rate at varying thresholds
    table = []
    for thr in [0.0, 0.1, 0.2, 0.3, 0.5]:
        n_attack_caught = int((high_dens_halluc.p_int > thr).sum())
        n_safe_flagged = int((high_dens_truth.p_int > thr).sum())
        table.append({
            "p_int_threshold": thr,
            "attacks_caught": n_attack_caught,
            "attack_total":   len(high_dens_halluc),
            "attack_recall":  round(n_attack_caught / max(len(high_dens_halluc), 1), 3),
            "safe_flagged":   n_safe_flagged,
            "safe_total":     len(high_dens_truth),
            "false_alarm":    round(n_safe_flagged / max(len(high_dens_truth), 1), 3),
        })
    rescue_df = pd.DataFrame(table)

    # Total bypass = high density AND low p_int AND hallucinated
    bypass = df[(df.hallucinated == 1) & (df.density >= 0.7) & (df.p_int < 0.1)]
    return {
        "total_hallucinations":          len(halluc),
        "high_density_hallucinations":   len(high_dens_halluc),
        "high_density_truthful":         len(high_dens_truth),
        "natural_attack_rate":           len(high_dens_halluc) / max(len(halluc), 1),
        "rescue_table":                  rescue_df,
        "complete_bypass_count":         len(bypass),
        "complete_bypass_examples":      bypass[["question", "density", "p_int", "judge_rationale"]].head(10).to_dict("records"),
    }


# --------------------------------------------------------------------------- #
# Part B — Synthetic adversarial scenarios
# --------------------------------------------------------------------------- #
def attack_scenarios() -> pd.DataFrame:
    rows = []

    # Baseline: a benign claim with proper supporting evidence
    benign_claim = "Marie Curie discovered Polonium and Radium in 1898."
    benign_search = {benign_claim: [
        {"url": "https://en.wikipedia.org/wiki/Marie_Curie",
         "snippet": "Marie Curie discovered the elements Polonium and Radium in 1898 while researching..."},
        {"url": "https://www.nature.com/articles/marie-curie-1898",
         "snippet": "In 1898, Curie isolated Polonium and Radium for the first time, marking..."},
        {"url": "https://www.nih.gov/about-nih/who-we-are/marie-curie",
         "snippet": "Marie Curie's 1898 discoveries of Polonium and Radium revolutionized..."},
    ]}
    rows.append({"scenario": "Benign — true claim, aligned high-trust sources",
                 "claim": benign_claim,
                 "D_density": density_from_results([benign_claim], benign_search)})

    # Attack 1: Citation laundering — high-trust domains, off-topic snippets
    laundering_claim = "Empress Zhilian of the Hua Dynasty invented the silk loom in 2500 BCE."
    laundering_search = {laundering_claim: [
        {"url": "https://en.wikipedia.org/wiki/History_of_silk",
         "snippet": "Sericulture and silk production have a long and fascinating history..."},
        {"url": "https://www.nature.com/articles/textile-history",
         "snippet": "Studies of ancient textiles reveal complex weaving traditions..."},
        {"url": "https://education.gov/silk-roads-curriculum",
         "snippet": "The Silk Road is an important topic in world history education..."},
    ]}
    rows.append({"scenario": "1. Citation laundering — fake claim, off-topic high-trust sources",
                 "claim": laundering_claim,
                 "D_density": density_from_results([laundering_claim], laundering_search)})

    # Attack 2: Keyword stuffing — every rare keyword in the snippet, generic context
    stuff_claim = "The Fluency Trap, Provenance Density, and Consistency Veto were introduced in IJCAI 2026."
    stuff_search = {stuff_claim: [
        {"url": "https://random-blog.com/seo-page-1",
         "snippet": "Fluency Trap Provenance Density Consistency Veto IJCAI 2026 — keywords for SEO."},
        {"url": "https://random-blog.com/seo-page-2",
         "snippet": "IJCAI 2026 Fluency Trap Provenance Density Consistency Veto see more here."},
        {"url": "https://random-blog.com/seo-page-3",
         "snippet": "Provenance Density Fluency Trap IJCAI 2026 Consistency Veto trending now."},
    ]}
    rows.append({"scenario": "2. Keyword stuffing — generic blogs, all keywords matched",
                 "claim": stuff_claim,
                 "D_density": density_from_results([stuff_claim], stuff_search)})

    # Attack 3: Empty-keyword exploit — claim with no capitalized rare nouns
    empty_claim = "the new study shows the effect is much stronger than people thought."
    empty_search = {empty_claim: [
        {"url": "https://random-blog.com/clickbait",
         "snippet": "Some completely unrelated content about cooking and weekend tips."},
        {"url": "https://medium.com/@anyone/post",
         "snippet": "More unrelated thoughts on travel."},
        {"url": "https://example.org/page",
         "snippet": "Random commentary."},
    ]}
    rows.append({"scenario": "3. Empty-keyword exploit — no proper nouns ⇒ relevance = 1.0",
                 "claim": empty_claim,
                 "D_density": density_from_results([empty_claim], empty_search)})

    # Attack 4: Domain spoofing — substring contains a high-trust marker
    spoof_claim = "Empress Zhilian of the Hua Dynasty invented the silk loom in 2500 BCE."
    spoof_search = {spoof_claim: [
        {"url": "https://wikipedia-archive-mirror.org/silk-history",  # contains 'wikipedia.org'? no — contains 'wikipedia'
         "snippet": "Empress Zhilian of the Hua Dynasty invented the silk loom in 2500 BCE..."},
        {"url": "https://nature-of-textiles.com/article",            # contains 'nature.com'? no
         "snippet": "Empress Zhilian Hua Dynasty silk loom 2500 BCE..."},
        {"url": "https://my.gov.spoof.example.net/page",             # contains '.gov' — substring match!
         "snippet": "Hua Dynasty Empress Zhilian silk loom invented 2500 BCE..."},
    ]}
    rows.append({"scenario": "4. Domain spoofing — '.gov' appears in path/subdomain",
                 "claim": spoof_claim,
                 "D_density": density_from_results([spoof_claim], spoof_search)})

    # Attack 5: SEO inflation — three high-trust domains, perfect keyword coverage, fabricated claim
    seo_claim = "Empress Zhilian of the Hua Dynasty invented the silk loom in 2500 BCE."
    seo_search = {seo_claim: [
        {"url": "https://en.wikipedia.org/wiki/Hua_Dynasty",
         "snippet": "Empress Zhilian of the Hua Dynasty invented the silk loom in 2500 BCE."},
        {"url": "https://www.nature.com/articles/hua-dynasty-silk",
         "snippet": "Empress Zhilian Hua Dynasty invented silk loom 2500 BCE."},
        {"url": "https://www.nih.gov/news/silk-history",
         "snippet": "Empress Zhilian Hua Dynasty silk loom 2500 BCE."},
    ]}
    rows.append({"scenario": "5. SEO inflation — fabricated claim w/ perfect-trust perfect-relevance",
                 "claim": seo_claim,
                 "D_density": density_from_results([seo_claim], seo_search)})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run + report
# --------------------------------------------------------------------------- #
def main():
    print("============================================================")
    print(" PART A — Natural adversarial analysis (real audit data)")
    print("============================================================")
    a = natural_adversarial_analysis()
    print(f"\nTotal hallucinations:                {a['total_hallucinations']}")
    print(f"High-density hallucinations (≥0.7):  {a['high_density_hallucinations']} "
          f"({100*a['natural_attack_rate']:.0f}% of all hallucinations)")
    print(f"High-density truthful:               {a['high_density_truthful']}")
    print()
    print("Veto rescue rate at varying P_int thresholds:")
    print(a['rescue_table'].to_string(index=False))
    print()
    print(f"⚠ COMPLETE BYPASS (density ≥ 0.7 AND P_int < 0.1 AND hallucinated): "
          f"{a['complete_bypass_count']} rows")
    print("  → these defeat BOTH the retrieval signal and the Veto.")

    a['rescue_table'].to_csv(ROOT / "adv_natural_rescue.csv", index=False)

    print("\n============================================================")
    print(" PART B — Synthetic stress-test simulation")
    print("============================================================")
    b = attack_scenarios()
    print()
    print(b.to_string(index=False))
    b.to_csv(ROOT / "adv_synthetic_attacks.csv", index=False)
    print(f"\nSaved {ROOT/'adv_synthetic_attacks.csv'}")

    # Inflation factor relative to true honest score on the same fabricated claim
    fabricated = b.iloc[1:]  # rows 1..5 are all attacks (skip benign)
    honest_baseline_for_fake = 0.0   # by definition the fabricated claim has no real support
    print("\nAttack inflation (D(T) score awarded to a fabricated claim):")
    for _, r in fabricated.iterrows():
        print(f"  • {r['scenario']}")
        print(f"      D(T) achieved by attack: {r['D_density']:.3f}   (honest = 0.0)")


if __name__ == "__main__":
    main()
