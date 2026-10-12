#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Contrastive embedding dataset builder for Track A RAG retriever fine-tuning.

Inspired by OTel's contrastive embedding training approach:
  - Anchor: the scenario's context description + symptom summary (from answer label)
  - Positive: a different scenario that shares the same answer code(s) (same fault family)
  - Hard negatives (×3): scenarios from adjacent but distinct fault families

The resulting triplets are used by train_embedding.py to fine-tune
all-MiniLM-L6-v2 with CoSENTLoss so that the RAG retriever retrieves
same-fault-type examples rather than superficially similar ones.

Output:
    finetune/embedding_train.json   (~1500 triplets)

Usage:
    cd "Track A"
    python finetune/prepare_embedding_data.py
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Fault family definitions
# Groups answer codes by root-cause family so hard negatives come from
# adjacent but distinct fault types (harder than random negatives).
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from answer code to fault family name.
# Derived from the 22 option labels (handover, coverage, interference, load,
# missing-neighbor, other). Scenarios whose answers span multiple families
# are assigned to the family of the first code.
FAULT_FAMILIES = {
    # Late / incorrect handover (A3 offset tuning)
    "late_handover": {
        "C1",   # Add neighbor relationship (often co-occurs with late HO)
        "C9",   # Decrease A3 Offset threshold
        "C10",  # Increase A3 Offset threshold
        "C11",  # Decrease A3 hysteresis
        "C12",  # Increase A3 hysteresis
        "C13",  # Decrease time-to-trigger
        "C14",  # Increase time-to-trigger
    },
    # Coverage / power
    "coverage": {
        "C2",   # Decrease transmission power of serving cell
        "C3",   # Increase transmission power of serving cell
        "C4",   # Decrease antenna tilt of serving cell (lift beam)
        "C5",   # Increase antenna tilt of serving cell (downtilt)
    },
    # Interference
    "interference": {
        "C6",   # Decrease transmission power of interfering cell
        "C7",   # Increase transmission power of interfering cell
        "C8",   # Decrease antenna tilt of interfering cell
        "C16",  # Increase antenna tilt of interfering cell
    },
    # Inter-frequency load / offload
    "inter_freq": {
        "C15",  # Decrease CovInterFreq A2 threshold
        "C17",  # Increase CovInterFreq A2 threshold
        "C18",  # Decrease CovInterFreq A5 threshold 1
        "C19",  # Increase CovInterFreq A5 threshold 1
        "C20",  # Decrease CovInterFreq A5 threshold 2
        "C21",  # Increase CovInterFreq A5 threshold 2
    },
    # Test / backhaul
    "other": {
        "C22",  # Check test server / backhaul
    },
}

# Reverse map: code → family name
CODE_TO_FAMILY: Dict[str, str] = {}
for _family, _codes in FAULT_FAMILIES.items():
    for _code in _codes:
        CODE_TO_FAMILY[_code] = _family

FAMILY_NAMES = list(FAULT_FAMILIES.keys())


def primary_family(answer: str) -> str:
    """Return the fault family of the first answer code."""
    first_code = answer.split("|")[0].strip()
    return CODE_TO_FAMILY.get(first_code, "other")


def build_anchor_text(example: dict) -> str:
    """
    Construct a short, information-dense anchor string from a scenario.

    Uses the context description and the answer option labels (the fault names)
    rather than raw RF numbers so the embedding captures semantic fault type.
    """
    ctx = example.get("context", {})
    description = ctx.get("description", "").strip()

    task = example.get("task", {})
    options = task.get("options", [])
    answer = example.get("answer", "")

    # Collect label text for each answer code
    answer_labels = []
    for code in answer.split("|"):
        code = code.strip()
        lbl = next((o["label"] for o in options if o["id"] == code), code)
        answer_labels.append(f"{code}: {lbl}")

    wireless = ctx.get("wireless_network_information", {})
    scenario_type = wireless.get("mobility_scenario", "")
    num_bs = wireless.get("num_base_stations", "")

    parts = []
    if description:
        parts.append(description[:300])  # cap to avoid very long anchors
    if scenario_type:
        parts.append(f"Mobility scenario: {scenario_type}")
    if num_bs:
        parts.append(f"Base stations: {num_bs}")
    if answer_labels:
        parts.append("Fault: " + "; ".join(answer_labels))

    return " | ".join(parts)


def sample_negative(
    current_family: str,
    family_to_indices: Dict[str, List[int]],
    examples: List[dict],
    rng: random.Random,
    exclude_idx: int,
    n: int = 3,
) -> List[str]:
    """
    Sample n hard negatives from families adjacent to current_family.
    Adjacent = different families in priority order; random fallback if needed.
    """
    # Priority: other families first (adjacent fault types), then same family
    other_families = [f for f in FAMILY_NAMES if f != current_family]
    rng.shuffle(other_families)
    candidate_indices: List[int] = []
    for fam in other_families:
        candidate_indices.extend(family_to_indices.get(fam, []))
    # Fallback: same family (different scenario)
    same_fam = [i for i in family_to_indices.get(current_family, []) if i != exclude_idx]
    candidate_indices.extend(same_fam)

    # Remove the anchor scenario itself
    candidate_indices = [i for i in candidate_indices if i != exclude_idx]

    chosen = rng.sample(candidate_indices, min(n, len(candidate_indices)))
    return [build_anchor_text(examples[i]) for i in chosen]


def build_triplets(examples: List[dict], rng: random.Random) -> List[dict]:
    """
    Build anchor/positive/negative triplets from the training examples.

    Rules:
    - Positive: same fault family, different scenario
    - Hard negatives: different fault families (adjacent first)
    - Skip anchor groups too small to form a positive pair (< 2 members)
    """
    # Group scenario indices by fault family
    family_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        fam = primary_family(ex.get("answer", ""))
        family_to_indices[fam].append(i)

    triplets = []
    for i, ex in enumerate(examples):
        fam = primary_family(ex.get("answer", ""))
        same_fam_indices = [j for j in family_to_indices[fam] if j != i]

        if not same_fam_indices:
            continue  # skip — no positive available

        anchor_text = build_anchor_text(ex)
        if not anchor_text.strip():
            continue

        positive_idx = rng.choice(same_fam_indices)
        positive_text = build_anchor_text(examples[positive_idx])
        if not positive_text.strip():
            continue

        negatives = sample_negative(fam, family_to_indices, examples, rng, i, n=3)
        if not negatives:
            continue

        record: dict = {"anchor": anchor_text, "positive": positive_text}
        for k, neg_text in enumerate(negatives, start=1):
            record[f"negative_{k}"] = neg_text

        triplets.append(record)

    return triplets


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    track_a_dir = os.path.dirname(script_dir)
    train_json = os.path.join(track_a_dir, "data", "Phase_1", "train.json")
    out_path = os.path.join(script_dir, "embedding_train.json")

    print(f"Loading training data from {train_json}...")
    with open(train_json, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples.")

    rng = random.Random(42)
    rng.shuffle(examples)

    print("Building contrastive triplets...")
    triplets = build_triplets(examples, rng)
    print(f"Generated {len(triplets)} triplets.")

    # Print family distribution for transparency
    family_counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        fam = primary_family(ex.get("answer", ""))
        family_counts[fam] += 1
    print("\nFault family distribution in training data:")
    for fam, count in sorted(family_counts.items()):
        print(f"  {fam:20s}: {count} scenarios")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(triplets)} triplets to {out_path}")
    print("Next: run  python finetune/train_embedding.py  to fine-tune the retriever.")


if __name__ == "__main__":
    main()
