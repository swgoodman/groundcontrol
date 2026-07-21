"""Injection-canary experiment: detection versus number of poisoned passages.

    uv run python scripts/run_canary.py

Compares two ways of using the same scorer on the same data:

    whole-context   concatenate the retrieved set and ask "is the claim supported?"
    canary          score each passage separately and look for internal disagreement

Reports detection rate at a fixed false-positive budget on clean sets, plus how far the
top-supporting passage identifies the injected one, swept across k poisoned of n.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from groundcontrol import canary
from groundcontrol.data import injection
from groundcontrol.scorers.finetuned import Finetuned

MODEL = "artifacts/groundcontrol-deberta-v3-base-v1-local"
TEMPERATURE = 1.7456764875286837
N_PASSAGES = 5
N_SETS = 240
TARGET_FPR = 0.10


def evaluate(scorer, sets) -> dict:
    poisoned, clean, localized = [], [], []
    whole_ctx_poisoned, whole_ctx_clean = [], []

    for s in sets:
        r = canary.run(scorer, s)
        if s.poisoned:
            poisoned.append(r.conflict)
            whole_ctx_poisoned.append(r.whole_context_supported)
            localized.append(r.localizes_attack())
        else:
            clean.append(r.conflict)
            whole_ctx_clean.append(r.whole_context_supported)

    poisoned_arr, clean_arr = np.array(poisoned), np.array(clean)

    # Threshold set on clean sets alone, so the false-positive budget is honoured
    # without ever looking at attacked traffic.
    threshold = float(np.quantile(clean_arr, 1 - TARGET_FPR)) if clean_arr.size else 1.0

    return {
        "n_poisoned_sets": len(poisoned),
        "n_clean_sets": len(clean),
        "threshold": threshold,
        "canary_detection_rate": float((poisoned_arr > threshold).mean()),
        "canary_false_positive_rate": float((clean_arr > threshold).mean()),
        "mean_conflict_poisoned": float(poisoned_arr.mean()),
        "mean_conflict_clean": float(clean_arr.mean()),
        "localization_rate": float(np.mean(localized)) if localized else 0.0,
        # The baseline: a standard groundedness check calls the poisoned answer
        # supported, because the attacker's text is inside the context it reads.
        "whole_context_flags_attack": float(1 - np.mean(whole_ctx_poisoned)),
        "whole_context_flags_clean": float(1 - np.mean(whole_ctx_clean)),
    }


def main() -> None:
    scorer = Finetuned(MODEL, name="groundcontrol", temperature=TEMPERATURE)

    # Two conditions. Varying k alone does not test what the canary depends on: the
    # builder keeps the refuting passage, so a trusted contradiction survives even at
    # k=4. `keep_refuting=False` models the attacker displacing the true evidence out
    # of the retrieved set, which is the condition that should actually blind it.
    conditions = [(k, True) for k in (1, 2, 3, 4)] + [(k, False) for k in (1, 2)]

    results = {}
    for k, keep in conditions:
        sets = injection.build(
            n_poisoned=k,
            n_passages=N_PASSAGES,
            n_sets=N_SETS,
            split="validation",
            allow_majority=True,
            keep_refuting=keep,
        )
        name = f"k={k}_of_{N_PASSAGES}" + ("" if keep else "_evidence_displaced")
        print(f"{name}: {len(sets)} sets ({sum(s.poisoned for s in sets)} poisoned)")
        results[name] = evaluate(scorer, sets)
        results[name]["majority_poisoned"] = k * 2 >= N_PASSAGES
        results[name]["trusted_evidence_retrieved"] = keep
        print(json.dumps(results[name], indent=2))

    out = Path("reports/canary_sweep.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
