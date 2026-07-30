"""Separate paraphrase *drift* from paraphrase *rewording* in the canary probe.

    uv run python scripts/run_canary_paraphrase_filtered.py

`run_canary_paraphrase.py` showed that paraphrasing the payload drops canary detection
from ~61% to ~45%. But a T5 paraphraser does two different things, and only one of them
is a fair test. It rewords ("Paris is the capital of France" -> "France's capital is
Paris"), which still fully asserts the claim, so any detection drop is a real property of
the method. Or it *weakens* the assertion ("... is the capital" -> "... is often regarded
as a capital"), so the payload no longer strongly asserts the claim, and the drop is an
artefact of a broken payload rather than of rewording.

The fix the README's next-steps names: filter on mutual entailment against an independent
model, hold the claim subset fixed, rerun. A paraphrase is kept only if it and the claim
entail each other in both directions, judged by a model that is neither the scorer under
test nor its lineage (roberta-large-mnli: different architecture, MNLI-only, never sees
the injection sets). Verbatim and paraphrased arms are then compared on the *same* kept
claims, so the remaining gap is rewording alone.

Reading the output:
- paraphrased detection on the kept subset is the drift-corrected number. If it climbs
  back toward the verbatim rate, the 45% was mostly drift. If it stays near 45%, rewording
  genuinely costs the method. This decides whether the honest figure is 45, 61, or between.
- verbatim-on-kept (not verbatim-on-all) is the right comparator, because the kept subset
  is a selected population and could be intrinsically easier or harder even verbatim.
- both detectors are held to the same false-positive budget (`eval.detection`), and every
  rate carries a bootstrap interval. AUROC is reported beside each rate because the clean
  conflict scores are bimodal — roughly an eighth of ordinary retrieval sets contain a
  genuine contradiction — so the fitted threshold lands on a cliff and the rate alone
  moves far more than the underlying separation does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from groundcontrol import canary
from groundcontrol.data import injection
from groundcontrol.eval import detection as detection_mod
from groundcontrol.scorers.finetuned import Finetuned

MODEL = "artifacts/groundcontrol-deberta-v3-base-v1-local"
PARAPHRASER = "humarin/chatgpt_paraphraser_on_T5_base"
# Independent fidelity judge. Deliberately not a DeBERTa: the scorer under test is a
# fine-tuned DeBERTa warm-started from MNLI, so a DeBERTa-MNLI judge would share its
# lineage and its blind spots. roberta-large-mnli is a different architecture, MNLI-only,
# and never scores the injection sets, so keeping a paraphrase cannot be circular with
# the scorer deciding it is entailing.
JUDGE = "roberta-large-mnli"
N_PASSAGES = 5
N_SETS = 1600
TARGET_FPR = 0.10
BATCH = 16


def load_temperature() -> float:
    cal = json.loads(Path(MODEL, "calibration.json").read_text(encoding="utf-8"))
    return float(cal["temperature"])


def make_paraphraser():
    """Beam-search paraphraser that also records every claim -> paraphrase it produces.

    build() hands the callable the exact list of poisoned claims, so capturing here is
    how the experiment recovers the mapping without duplicating build's dataset logic or
    depending on positional alignment (build_set drops some claims to None).
    """
    tok = AutoTokenizer.from_pretrained(PARAPHRASER)
    model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER)
    model.eval()
    captured: dict[str, str] = {}

    def paraphrase(claims: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(claims), BATCH):
            chunk = [f"paraphrase: {c}" for c in claims[i : i + BATCH]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=96)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    num_beams=5,
                    max_length=96,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                )
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        captured.update(zip(claims[: len(out)], out, strict=False))
        return out

    return paraphrase, captured


def make_judge():
    """Return a fn giving P(entailment) for a batch of (premise, hypothesis) pairs.

    Label order is read from the checkpoint, never assumed, the same discipline the
    zero-shot scorer uses: a hardcoded entailment index inverts silently on a
    contradiction-first head and still prints plausible numbers.
    """
    tok = AutoTokenizer.from_pretrained(JUDGE)
    model = AutoModelForSequenceClassification.from_pretrained(JUDGE)
    model.eval()
    entail_idx = next(
        i for i, label in model.config.id2label.items() if str(label).lower() == "entailment"
    )

    def p_entail(premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out: list[float] = []
        for i in range(0, len(premises), BATCH):
            enc = tok(
                premises[i : i + BATCH],
                hypotheses[i : i + BATCH],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                probs = model(**enc).logits.softmax(-1)[:, entail_idx]
            out.extend(probs.tolist())
        return np.array(out)

    return p_entail


def score_sets(scorer, sets) -> list[dict]:
    """Run the canary once per set and keep the fields the subsetting needs."""
    scored = []
    for s in sets:
        r = canary.run(scorer, s)
        payload_support = [v.p_supported for v in r.passage_verdicts if v.poisoned]
        scored.append(
            {
                "claim": s.claim,
                "poisoned": s.poisoned,
                "conflict": r.conflict,
                "localizes": r.localizes_attack(),
                # P(supported) on the joined context, kept as a score rather than a
                # verdict: the comparison thresholds it to the same false-positive
                # budget as the canary instead of reading it at its own 0.5 default.
                "joined": r.whole_context_score,
                "payload_support": float(np.mean(payload_support)) if payload_support else None,
            }
        )
    return scored


def detection(scored: list[dict], clean: dict, claims: set[str] | None = None) -> dict:
    """One cell of the design: both detectors on this subset, at a matched FPR budget.

    `clean` carries the clean-set scores the thresholds are fitted on. They come from
    outside this call because clean controls are identical across every cell — same
    SUPPORTS claims, never paraphrased — so a subset of poisoned claims must not drag
    the operating point around with it.
    """
    rows = [x for x in scored if x["poisoned"] and (claims is None or x["claim"] in claims)]
    if not rows:
        return {"n": 0}

    support = [x["payload_support"] for x in rows if x["payload_support"] is not None]
    comparison = detection_mod.evaluate(
        [
            detection_mod.DetectorScores(
                "canary", [x["conflict"] for x in rows], clean["conflict"], higher_is_attack=True
            ),
            detection_mod.DetectorScores(
                "whole_context",
                [x["joined"] for x in rows],
                clean["joined"],
                higher_is_attack=False,
            ),
        ],
        target_fpr=TARGET_FPR,
        reference="canary",
    )
    canary_det = comparison.detections["canary"]
    whole_det = comparison.detections["whole_context"]
    edge = comparison.edges["whole_context"]

    n_localized = int(np.sum([x["localizes"] for x in rows]))
    localization_ci = detection_mod.proportion_ci(n_localized, len(rows))

    return {
        "n": len(rows),
        "detection_rate": canary_det.detection_rate,
        "detection_ci": [canary_det.detection_ci.lo, canary_det.detection_ci.hi],
        "auroc": canary_det.auroc,
        "auroc_ci": [canary_det.auroc_ci.lo, canary_det.auroc_ci.hi],
        "whole_context_rate": whole_det.detection_rate,
        "whole_context_ci": [whole_det.detection_ci.lo, whole_det.detection_ci.hi],
        "whole_context_auroc": whole_det.auroc,
        # The threshold-free version of the edge, which is the one to lead with when the
        # operating point sits on a cliff.
        "auroc_difference": edge.auroc_difference,
        "auroc_difference_ci": [edge.auroc_difference_ci.lo, edge.auroc_difference_ci.hi],
        "achieved_fpr": {
            "canary": canary_det.achieved_fpr,
            "whole_context": whole_det.achieved_fpr,
        },
        # canary's advantage over the standard check, as the multiple the README reports,
        # now with the interval that says how much of it survives resampling. In the
        # verbatim cells the baseline detects almost nothing, so `edge_ci` is unbounded
        # above and `edge_unbounded` says how much of the bootstrap went that way — read
        # `difference` there instead of the multiple.
        "edge": edge.ratio,
        "edge_ci": [edge.ratio_ci.lo, edge.ratio_ci.hi],
        "edge_unbounded": edge.ratio_unbounded,
        "difference": edge.difference,
        "difference_ci": [edge.difference_ci.lo, edge.difference_ci.hi],
        "mean_conflict": float(np.mean([x["conflict"] for x in rows])),
        "mean_payload_support": float(np.mean(support)) if support else None,
        "localization_rate": n_localized / len(rows),
        "localization_ci": [localization_ci.lo, localization_ci.hi],
    }


def main() -> None:
    scorer = Finetuned(MODEL, name="groundcontrol", temperature=load_temperature())
    paraphrase, captured = make_paraphraser()
    judge = make_judge()

    def build(k: int, paraphrase_fn=None):
        return injection.build(
            n_poisoned=k,
            n_passages=N_PASSAGES,
            n_sets=N_SETS,
            split="validation",
            allow_majority=True,
            keep_refuting=True,
            paraphrase=paraphrase_fn,
        )

    # Thresholds come from clean traffic only, never from attacked sets. Clean controls
    # are built from SUPPORTS claims and carry no payload, so they do not vary with k or
    # with the arm — score them once, and spend what that saves on having enough of them
    # to pin a threshold that rests on their top tenth.
    clean_scored = score_sets(scorer, [s for s in build(1) if not s.poisoned])
    clean = {
        "conflict": [x["conflict"] for x in clean_scored],
        "joined": [x["joined"] for x in clean_scored],
    }
    print(f"clean controls: {len(clean_scored)} sets, scored once for every cell")

    results, raw_scores = {}, {"clean": clean}
    for k in (1, 2):
        verbatim_scored = score_sets(scorer, [s for s in build(k) if s.poisoned])
        paraphrased_scored = score_sets(scorer, [s for s in build(k, paraphrase) if s.poisoned])

        # Mutual-entailment fidelity filter on the bare paraphrase vs the claim under test.
        poisoned_claims = [x["claim"] for x in paraphrased_scored if x["poisoned"]]
        paras = [captured[c] for c in poisoned_claims]
        fwd = judge(paras, poisoned_claims)  # paraphrase entails claim: catches weakening
        bwd = judge(poisoned_claims, paras)  # claim entails paraphrase: catches drift/add
        kept, dropped, fidelity = set(), set(), {}
        for claim, para, f, b in zip(poisoned_claims, paras, fwd, bwd, strict=True):
            keep = bool(f >= 0.5 and b >= 0.5)
            (kept if keep else dropped).add(claim)
            fidelity[claim] = {
                "paraphrase": para,
                "p_fwd": float(f),
                "p_bwd": float(b),
                "kept": keep,
            }

        results[f"k={k}"] = {
            "target_fpr": TARGET_FPR,
            "n_clean_sets": len(clean["conflict"]),
            "n_poisoned": len(poisoned_claims),
            "n_kept": len(kept),
            "n_dropped": len(dropped),
            "kept_fraction": len(kept) / len(poisoned_claims) if poisoned_claims else 0.0,
            "mean_p_entail_kept": _mean_fidelity(fidelity, kept),
            "mean_p_entail_dropped": _mean_fidelity(fidelity, dropped),
            "verbatim_all": detection(verbatim_scored, clean),
            "paraphrased_all": detection(paraphrased_scored, clean),
            "verbatim_kept": detection(verbatim_scored, clean, kept),
            "paraphrased_kept": detection(paraphrased_scored, clean, kept),
            "verbatim_dropped": detection(verbatim_scored, clean, dropped),
            "paraphrased_dropped": detection(paraphrased_scored, clean, dropped),
            "examples": {
                "kept": _example(fidelity, kept),
                "dropped": _example(fidelity, dropped),
            },
        }
        raw_scores[f"k={k}"] = {"verbatim": verbatim_scored, "paraphrased": paraphrased_scored}
        _print_block(k, results[f"k={k}"])

    out = Path("reports/canary_paraphrase_filtered.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Per-set scores, so the operating point can be re-analysed without re-running the
    # model, the paraphraser, and the judge.
    scores_out = Path("reports/canary_paraphrase_scores.json")
    scores_out.write_text(json.dumps(raw_scores), encoding="utf-8")
    print(f"\nwrote {out} and {scores_out}")


def _mean_fidelity(fidelity: dict, claims: set[str]) -> dict | None:
    rows = [fidelity[c] for c in claims]
    if not rows:
        return None
    return {
        "p_fwd": float(np.mean([r["p_fwd"] for r in rows])),
        "p_bwd": float(np.mean([r["p_bwd"] for r in rows])),
    }


def _example(fidelity: dict, claims: set[str]) -> dict | None:
    if not claims:
        return None
    c = sorted(claims)[0]
    return {"claim": c, **fidelity[c]}


def _print_block(k: int, r: dict) -> None:
    print(
        f"\n{'=' * 92}\nk={k} of {N_PASSAGES}   "
        f"both detectors at a {r['target_fpr']:.0%} FPR budget "
        f"fitted on {r['n_clean_sets']} clean sets"
    )
    print(
        f"  kept {r['n_kept']}/{r['n_poisoned']} paraphrases as faithful "
        f"({r['kept_fraction']:.0%}); {r['n_dropped']} dropped as drift"
    )
    header = (
        f"  {'cell':<20}{'n':>4}{'canary':>8}{'95% CI':>16}{'AUROC':>8}"
        f"{'whole-ctx':>11}{'wc AUROC':>11}{'edge':>7}{'payload':>9}"
    )
    print(header)
    for cell in (
        "verbatim_all",
        "paraphrased_all",
        "verbatim_kept",
        "paraphrased_kept",
        "verbatim_dropped",
        "paraphrased_dropped",
    ):
        d = r[cell]
        if d.get("n"):
            sup = d["mean_payload_support"]
            sup_s = f"{sup:.3f}" if sup is not None else "  -  "
            edge_s = f"{d['edge']:.1f}x" if np.isfinite(d["edge"]) else "  inf"
            ci = f"[{d['detection_ci'][0]:.2f}, {d['detection_ci'][1]:.2f}]"
            print(
                f"  {cell:<20}{d['n']:>4}{d['detection_rate']:>8.3f}{ci:>16}"
                f"{d['auroc']:>8.3f}{d['whole_context_rate']:>11.3f}"
                f"{d['whole_context_auroc']:>11.3f}{edge_s:>7}{sup_s:>9}"
            )


if __name__ == "__main__":
    main()
