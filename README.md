# groundcheck

> **Status: Phase 0, pre-release.** Interfaces and metrics are in place. No results yet.
> Every number below is a placeholder until it is measured. Working name, likely to change.

A small, calibrated groundedness checker and the eval harness around it.

Give it a source passage and a claim, and it tells you whether the claim is actually
supported by the source, with a calibrated confidence. Run it over a RAG pipeline's
output to catch hallucination. Run it per-passage over a retrieved set and it also
flags **prompt injection**, because an injected passage makes the retrieved evidence
contradict itself. Same check, two failure modes.

It runs on CPU, needs no access to the generating model's internals, and drops into CI.

## Why this exists

RAG fails in two halves. Retrieval can put the wrong evidence in front of the model,
and the model can fail to stay faithful to the evidence it got. **This measures the
second half.** That boundary is deliberate and stated up front: if the retriever
fetches a wrong-but-plausible passage and the generator faithfully echoes it, a
groundedness check *passes* while the answer is *wrong*. This tool certifies
faithfulness to the retrieved evidence, not truth about the world.

Most teams check that second half by eyeballing outputs or by asking a large model to
judge, which is slow, expensive, opaque, and itself uncalibrated. The bet here is that
a small fine-tuned scorer can do the job at a fraction of the cost and latency, and be
better calibrated while doing it. That is a hypothesis this repo is built to test, not
a result it claims.

## Design commitments

These are load-bearing, and they are why the repo is shaped the way it is.

**Protocols, not inheritance.** A scorer is anything with `score`, `score_batch`, and
`efficiency`. A dataset is anything that yields `Example`s. Adding either is one file
and one decorator.

**Metrics live in one module.** Training-time eval and harness-time eval both call
`groundcheck.eval.metrics`, so a training log and a leaderboard row cannot disagree.

**Calibration is a first-class output**, not a footnote. ECE and reliability curves sit
next to accuracy. You cannot oversee what you cannot measure, and an uncalibrated
confidence is not a measurement.

**Efficiency is a first-class output too.** Footprint, memory, latency, throughput, and
cost per 1k are reported next to accuracy. "Small enough to run anywhere" has to be a
number.

**Latency is never measured on an unlabeled machine.** `BenchmarkDevice` requires an
explicit label, so a number measured on a developer laptop can never be quietly
published as commodity-CPU latency. See `groundcheck/device.py`.

**Single-class slices report NaN, not a flattering number.** A smoke slice that happens
to be all-supported yields NaN balanced accuracy rather than a value that means nothing.

## The label convention

Stated once, because it is the most corruptible thing in the project.

Internally the model predicts **3 ways**: `supported`, `contradicted`, `neutral`. This
fits the MNLI warm-start and preserves the contradiction signal that the injection
canary depends on.

For reporting it collapses to **binary**: `supported` is True, and both `contradicted`
and `neutral` are False. The **class of interest is not-supported**, because that is the
rare and costly one. Headline precision, recall, F1, and PR-AUC are all reported on it.

## Install

Requires Python 3.12 or newer. Local development runs 3.14.

```bash
uv sync --extra dev          # core + test tooling
uv sync --all-extras         # adds torch, transformers, gradio
uv run pytest
```

## Layout

```
groundcheck/
├── data/base.py       Example schema + Dataset protocol
├── scorers/base.py    Verdict + Scorer protocol + EfficiencyProfile
├── eval/metrics.py    balanced acc, P/R/F1 + PR-AUC on not-supported, ECE, reliability
├── device.py          compute device (inferred) vs benchmark device (declared)
└── registry.py        name -> factory, for config-driven runs
```

## Roadmap

- [x] **Phase 0a** Interfaces, metrics, device discipline, CI
- [ ] **Phase 0b** Dataset adapters (RAGTruth, HaluEval, FEVER), zero-shot NLI baseline, runner, report
- [ ] **Phase 1** Fine-tuned scorer, calibration, INT8 variant, comparators, demo, CI-gate example
- [ ] **Phase 1.5** Adversarial and injection-induced curated datasets, released openly
- [ ] **Phase 2** Agent-trajectory extension: per-step drift localization

## License

Apache-2.0. Curated datasets, when released, are licensed separately.
