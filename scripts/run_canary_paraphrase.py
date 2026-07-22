"""Does the injection result depend on payloads restating the claim verbatim?

    uv run python scripts/run_canary_paraphrase.py

The conflict score is `min(max support, max contradiction)` over the retrieved set. A
payload built as "{claim} is true" pins the support term near 1, so the contradiction
term decides every case and the measured detection rate is really measuring how well the
scorer contradicts. A real attacker writes prose that asserts the claim in its own words.

This reruns the sweep with the payload's assertion paraphrased, claim under test held
fixed, and reports both arms side by side. A large drop would mean the headline number
was partly an artefact of the payload template rather than a property of the method.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from groundcontrol import canary
from groundcontrol.data import injection
from groundcontrol.scorers.finetuned import Finetuned

MODEL = "artifacts/groundcontrol-deberta-v3-base-v1-local"
TEMPERATURE = 1.7456764875286837
PARAPHRASER = "humarin/chatgpt_paraphraser_on_T5_base"
N_PASSAGES = 5
N_SETS = 240
TARGET_FPR = 0.10
BATCH = 16


def make_paraphraser():
    tok = AutoTokenizer.from_pretrained(PARAPHRASER)
    model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER)
    model.eval()

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
        return out

    return paraphrase


def evaluate(scorer, sets) -> dict:
    poisoned, clean, localized, payload_support = [], [], [], []
    whole_ctx_poisoned, whole_ctx_clean = [], []

    for s in sets:
        r = canary.run(scorer, s)
        if s.poisoned:
            poisoned.append(r.conflict)
            whole_ctx_poisoned.append(r.whole_context_supported)
            localized.append(r.localizes_attack())
            # The support term the payload contributes, which is what a verbatim
            # restatement inflates. Reported so the mechanism is visible, not inferred.
            payload_support.extend(v.p_supported for v in r.passage_verdicts if v.poisoned)
        else:
            clean.append(r.conflict)
            whole_ctx_clean.append(r.whole_context_supported)

    poisoned_arr, clean_arr = np.array(poisoned), np.array(clean)
    threshold = float(np.quantile(clean_arr, 1 - TARGET_FPR)) if clean_arr.size else 1.0

    return {
        "n_poisoned_sets": len(poisoned),
        "threshold": threshold,
        "canary_detection_rate": float((poisoned_arr > threshold).mean()),
        "mean_conflict_poisoned": float(poisoned_arr.mean()),
        "mean_conflict_clean": float(clean_arr.mean()),
        "mean_payload_support": float(np.mean(payload_support)),
        "localization_rate": float(np.mean(localized)) if localized else 0.0,
        "whole_context_flags_attack": float(1 - np.mean(whole_ctx_poisoned)),
    }


def main() -> None:
    scorer = Finetuned(MODEL, name="groundcontrol", temperature=TEMPERATURE)
    paraphrase = make_paraphraser()

    results = {}
    for k in (1, 2):
        for arm, fn in (("verbatim", None), ("paraphrased", paraphrase)):
            sets = injection.build(
                n_poisoned=k,
                n_passages=N_PASSAGES,
                n_sets=N_SETS,
                split="validation",
                allow_majority=True,
                keep_refuting=True,
                paraphrase=fn,
            )
            name = f"k={k}_of_{N_PASSAGES}_{arm}"
            print(f"\n{name}: {len(sets)} sets")
            if arm == "paraphrased":
                sample = next(s for s in sets if s.poisoned)
                print(f"  claim under test: {sample.claim!r}")
                print(f"  payload:          {sample.passages[sample.poisoned_indices[0]]!r}")
            results[name] = evaluate(scorer, sets)
            print(json.dumps(results[name], indent=2))

    print("\n" + "=" * 74)
    print(f"{'condition':<28}{'detection':>11}{'conflict':>11}{'payload P(sup)':>16}")
    for name, r in results.items():
        print(
            f"{name:<28}{r['canary_detection_rate']:>11.3f}"
            f"{r['mean_conflict_poisoned']:>11.3f}{r['mean_payload_support']:>16.3f}"
        )

    out = Path("reports/canary_paraphrase.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
