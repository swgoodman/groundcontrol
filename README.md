# groundcontrol 📡

[![CI](https://github.com/swgoodman/groundcontrol/actions/workflows/ci.yml/badge.svg)](https://github.com/swgoodman/groundcontrol/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> Can a groundedness check be small enough to leave on, and double as a prompt injection canary?

Groundedness checks in popular evaluation frameworks like [RAGAS](https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_faithfulness.py) and [DeepEval](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/faithfulness/faithfulness.py) help validate LLM responses are in line with provided source context (e.g. scraped web pages, files users upload, anything the system might retrieve to augment the original query's context). 

These checks use an LLM judge, making them slow and expensive. So they typically run offline, against test sets or sampled traffic rather than on every live request. The evaluation concatenates the retrieved passages into one block and asks whether the answer's claims are supported by it. 

That raises two questions. Can the check get fast and cheap enough that we can run it synchronously in the request path, protecting users from ungrounded responses? And if it's already reading every passage on every request, can the same check catch a prompt injection as it happens?

## Solution Explored

To detect injection, the check has to verify each passage individually. Concatenated, the block is trusted as a whole, so an injection borrows the credibility of the accurate sources around it. Scored alone with methods like [RobustRAG](https://arxiv.org/abs/2405.15556) and [SummaC](https://aclanthology.org/2022.tacl-1.10/), it's one voice out of five, and its fight with the real evidence becomes visible. What keeps that fix shelved is cost: five times the calls is expensive for an LLM judge. But what if a small encoder could get the job done?

So that's the solution explored here: fine-tune one small entailment model (DeBERTa-v3, 184M, runs on CPU) to score each passage on its own, and read the disagreement it surfaces as an injection canary. The model can't judge what's true. It only notices when the evidence can't all be true at once. If we can make the check small enough to leave on, the defense comes for free.

## Result

The canary score is `min(strongest support, strongest contradiction)` across the set. It spikes when one passage fully supports the claim and another swears it's false - which is exactly what an injection next to real evidence looks like. 

I put the existing solution and hypothesis head-to-head: same data for both, and each was allowed to false-alarm on 10% of clean traffic. Each row draws on 800 attacked and 800 clean sets, where a set is the five passages retrieved for one query:

| poison in set | whole-context AUROC | canary AUROC          | canary detects @10% FPR | canary localizes |
| ------------- | ------------------- | --------------------- | ----------------------- | ---------------- |
| 1 of 5        | 0.740               | **0.869** [.850,.888] | 47.9%                   | 98.6%            |
| 2 of 5        | 0.678               | **0.883** [.865,.902] | 63.6%                   | 99.4%            |
| 3 of 5        | 0.620               | **0.886** [.868,.905] | 69.0%                   | 99.5%            |
| 4 of 5        | 0.575               | **0.886** [.868,.904] | 71.0%                   | 99.6%            |

> [!NOTE]
> - AUROC — pick one attacked set and one clean set at random. This is how often the attacked one scores higher. 0.5 is a coin flip. 1.0 is perfect. Brackets are 95% confidence intervals.
> - Detects @10% FPR — the share of attacks caught when the alarm is only allowed to go off on 10% of clean traffic.
> - Localizes — when a set really is attacked, how often the top-scoring passage is the poisoned one.
> - Why AUROC is the headline — "attacked scores higher than clean" holds up on new data. The exact threshold doesn't: tuned for a 10% false-alarm rate, it actually fires on anywhere from 6% to 15% of clean traffic it hasn't seen. So AUROC is the reliable number, and the detection column is an estimate.

### Four findings that shape the result

- The canary needs a witness
  - The advantage requires one honest passage that contradicts the lie to survive retrieval. Remove it and the edge over the joined check is +0.000 AUROC [-0.031, +0.034]. The canary still finds the payload (99.4%), it just has nothing to disagree with it. For this detection to work as is, the injection can't crowd out every honest source; at least one contradicting passage has to make it into the set.
- Wording matters
  - The table above uses verbatim payloads, which flatter the canary. Paraphrase the payload, but still assert the claim, and detection drops from 47.6% to 31.3%, AUROC 0.867 to 0.850. Localization is consistent.
- The 3-way head is load-bearing
  - The model was fine-tuned with a three-class natural language inference (NLI) classification head, scoring each passage as supported, contradicted, or neutral. That choice is what makes the canary work. "Doesn't support" is not "contradicts", as most passages are just off-topic. Read contradiction as `1 - P(supported)` and nothing is detected. Read it from the head's contradicted class and detection goes 0% to 61%.
- Efficiency is untested
  - The cost comparison against an LLM judge (e.g. [Granite Guardian](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)) never ran. The fine-tune does beat its zero-shot starting point on each quality axis, though one column is in-sample ([`reports/v1_comparison.md`](reports/v1_comparison.md)).

Where I'd like to explore next: real attack payloads (PoisonedRAG, BIPIA), a real retriever instead of simulated displacement, comparison with other checkers (HHEM, MiniCheck, Granite Guardian), and localization over multi-step agent behavior.

## Running it

```bash
uv sync --all-extras         # torch, transformers
uv run pytest

uv run python scripts/run_eval.py configs/phase0_smoke.yaml       # baseline leaderboard
uv run python scripts/run_eval.py configs/v1_comparison.yaml      # fine-tuned vs zero-shot
uv run python scripts/run_train.py configs/train_v1_local.yaml    # fine-tune
uv run python scripts/run_canary.py                               # injection sweep
uv run python scripts/run_canary.py --from-scores                 # re-analyse, no model
uv run python scripts/run_canary_paraphrase.py                    # payload-wording probe
uv run python scripts/run_canary_paraphrase_filtered.py           # drift-filtered probe
```

- `phase0_smoke.yaml` runs end to end against public checkpoints.
- The canary scripts need the fine-tuned checkpoint in `artifacts/`, which isn't in this repo. Run `run_train.py` first to reproduce the Result section.
- AggreFact is gated (auto-approved): accept the terms and export `HF_TOKEN`.

## License

MIT, for the code in this repo. It does not extend to the model: any released weights
inherit the terms of the DeBERTa-v3 base and the training corpora (RAGTruth, FEVER,
HaluEval, AggreFact, MNLI), some of which are research or non-commercial only. Curated
datasets and any published weights are licensed separately, under whichever of those
terms is strictest.
