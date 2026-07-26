"""Injection-canary experiment: detection versus number of poisoned passages.

    uv run python scripts/run_canary.py

Compares two ways of using the same scorer on the same data:

    whole-context   concatenate the retrieved set and ask "is the claim supported?"
    canary          score each passage separately and look for internal disagreement

Both are held to the same false-positive budget, each threshold fitted on clean sets
alone. Reading one detector at a fitted budget and the other at its own default cutoff
would hand the first more licence to alarm and call the difference a result. Every rate
carries a bootstrap interval, and AUROC beside it, because the fitted threshold sits in
a region where small moves cost a lot of detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from groundcontrol import canary
from groundcontrol.data import injection
from groundcontrol.eval import detection
from groundcontrol.scorers.finetuned import Finetuned

MODEL = "artifacts/groundcontrol-deberta-v3-base-v1-local"
TEMPERATURE = 1.7456764875286837
N_PASSAGES = 5
# 800 poisoned sets per condition and 800 clean controls scored once. The binding
# constraint is the threshold, not the attack sample: it is the 90th percentile of the
# clean scores, so it rests on the top tenth of the clean controls and needs enough of
# them to hold still. Clean controls are the cheap half to buy — they are identical
# across every condition, so they are scored once and shared.
N_SETS = 1600
TARGET_FPR = 0.10


def score(scorer, sets) -> dict[str, list]:
    """Run the canary over a list of sets and keep the raw per-set scores."""
    conflict, joined, localized, default_flag = [], [], [], []
    for s in sets:
        r = canary.run(scorer, s)
        conflict.append(r.conflict)
        joined.append(r.whole_context_score)
        default_flag.append(not r.whole_context_supported)
        if s.poisoned:
            localized.append(r.localizes_attack())
    return {
        "conflict": conflict,
        "joined": joined,
        "localizes": localized,
        "default_flags": default_flag,
    }


def evaluate(poisoned: dict, clean: dict) -> dict:
    conflict_poisoned, joined_poisoned = poisoned["conflict"], poisoned["joined"]
    conflict_clean, joined_clean = clean["conflict"], clean["joined"]
    localized, default_flag_poisoned = poisoned["localizes"], poisoned["default_flags"]
    default_flag_clean = clean["default_flags"]

    comparison = detection.evaluate(
        [
            # Conflict is high when the set disagrees with itself.
            detection.DetectorScores(
                "canary", conflict_poisoned, conflict_clean, higher_is_attack=True
            ),
            # The standard check reads P(supported) on the joined context, so it flags
            # the *low* tail. Given the same budget, not its own 0.5 default.
            detection.DetectorScores(
                "whole_context", joined_poisoned, joined_clean, higher_is_attack=False
            ),
        ],
        target_fpr=TARGET_FPR,
        reference="canary",
    )

    n_localized = int(np.sum(localized))
    localization_ci = detection.proportion_ci(n_localized, len(localized)) if localized else None

    return {
        "n_poisoned_sets": len(conflict_poisoned),
        "n_clean_sets": len(conflict_clean),
        "matched_fpr": comparison.to_dict(),
        "mean_conflict_poisoned": float(np.mean(conflict_poisoned)),
        "mean_conflict_clean": float(np.mean(conflict_clean)),
        "localization_rate": n_localized / len(localized) if localized else 0.0,
        "localization_ci": [localization_ci.lo, localization_ci.hi] if localization_ci else None,
        # Kept for continuity with the earlier reports: what the standard check does at
        # its own 0.5 cutoff, which is *not* the matched-budget number above.
        "whole_context_default_threshold": {
            "flags_attack": float(np.mean(default_flag_poisoned)),
            "flags_clean": float(np.mean(default_flag_clean)),
        },
    }


def build(k: int, keep_refuting: bool):
    return injection.build(
        n_poisoned=k,
        n_passages=N_PASSAGES,
        n_sets=N_SETS,
        split="validation",
        allow_majority=True,
        keep_refuting=keep_refuting,
    )


SCORES_PATH = Path("reports/canary_scores.json")

# Two conditions. Varying k alone does not test what the canary depends on: the builder
# keeps the refuting passage, so a trusted contradiction survives even at k=4.
# `keep_refuting=False` models the attacker displacing the true evidence out of the
# retrieved set, which is the condition that should actually blind it.
CONDITIONS = [(k, True) for k in (1, 2, 3, 4)] + [(k, False) for k in (1, 2)]


def condition_name(k: int, keep_refuting: bool) -> str:
    return f"k={k}_of_{N_PASSAGES}" + ("" if keep_refuting else "_evidence_displaced")


def collect_scores() -> dict:
    """Score every condition and return the raw per-set arrays."""
    scorer = Finetuned(MODEL, name="groundcontrol", temperature=TEMPERATURE)

    # Clean controls are built from SUPPORTS claims and never carry a payload, so they
    # are byte-identical across every condition. Scoring them once instead of six times
    # is what pays for having enough of them to pin the threshold.
    clean_sets = [s for s in build(*CONDITIONS[0]) if not s.poisoned]
    print(f"clean controls: {len(clean_sets)} sets, scored once for every condition")
    clean = score(scorer, clean_sets)
    raw = {"clean": {k: v for k, v in clean.items() if k != "localizes"}}

    for k, keep in CONDITIONS:
        sets = build(k, keep)

        # The reuse above is only sound while that holds; assert rather than trust it.
        assert [s.passages for s in sets if not s.poisoned] == [s.passages for s in clean_sets], (
            "clean controls differ across conditions; they can no longer be shared"
        )

        name = condition_name(k, keep)
        poisoned_sets = [s for s in sets if s.poisoned]
        print(f"\n{name}: {len(poisoned_sets)} poisoned sets")
        raw[name] = score(scorer, poisoned_sets)

    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Re-analysis is the common case once the scores exist. Where the threshold goes,
    # what budget to spend, whether to lead with a rate or with AUROC — every one of
    # those is a question about these arrays, not about the model, and re-scoring to
    # answer them would cost half an hour and change nothing.
    parser.add_argument(
        "--from-scores",
        action="store_true",
        help=f"re-analyse {SCORES_PATH} instead of re-running the model",
    )
    args = parser.parse_args()

    if args.from_scores:
        raw = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        print(f"re-analysing {SCORES_PATH} ({len(raw['clean']['conflict'])} clean sets)")
    else:
        raw = collect_scores()
        SCORES_PATH.write_text(json.dumps(raw), encoding="utf-8")

    results = {}
    for k, keep in CONDITIONS:
        name = condition_name(k, keep)
        results[name] = evaluate(raw[name], raw["clean"])
        results[name]["majority_poisoned"] = k * 2 >= N_PASSAGES
        results[name]["trusted_evidence_retrieved"] = keep
        _print_block(name, results[name])

    out = Path("reports/canary_sweep.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out} and {SCORES_PATH}")


def _print_block(name: str, r: dict) -> None:
    m = r["matched_fpr"]
    print(f"\n{'=' * 78}\n{name}   budget={m['target_fpr']:.0%} FPR")
    print(
        f"  {'detector':<16}{'detects':>9}{'95% CI':>18}{'AUROC':>8}{'95% CI':>16}{'threshold':>11}"
    )
    for d in m["detections"].values():
        ci = f"[{d['detection_ci']['lo']:.3f}, {d['detection_ci']['hi']:.3f}]"
        auroc_ci = f"[{d['auroc_ci']['lo']:.3f}, {d['auroc_ci']['hi']:.3f}]"
        print(
            f"  {d['name']:<16}{d['detection_rate']:>9.3f}{ci:>18}"
            f"{d['auroc']:>8.3f}{auroc_ci:>16}{d['threshold']:>11.3f}"
        )
    for edge in m["edges"].values():
        diff_ci = edge["difference_ci"]
        print(
            f"  canary - {edge['name']}: {edge['difference']:+.3f} "
            f"[{diff_ci['lo']:+.3f}, {diff_ci['hi']:+.3f}]"
        )
    lo, hi = r["localization_ci"]
    print(f"  localization: {r['localization_rate']:.3f} [{lo:.3f}, {hi:.3f}]")
    default = r["whole_context_default_threshold"]
    print(
        f"  (whole-context at its own 0.5 cutoff: flags {default['flags_attack']:.3f} "
        f"of attacks at {default['flags_clean']:.3f} FPR — a different operating point)"
    )


if __name__ == "__main__":
    main()
