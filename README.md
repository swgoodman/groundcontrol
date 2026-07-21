# groundcheck

A small, calibrated groundedness checker for RAG, and the eval harness around it.

Give it a source passage and a claim, and it tells you whether the claim is actually
supported by the source, with a calibrated confidence. Runs on CPU, needs no access to
the generating model's internals, and drops into CI.

> **Status: pre-release.** The scorer is trained and evaluated; the comparator suite,
> demo, and published weights are not done yet. Every number below is measured and
> reproducible from this repo. Limitations are stated in full further down, and they
> are load-bearing rather than boilerplate.

## The result

The standard way to check groundedness in a RAG pipeline is to concatenate the
retrieved passages and ask whether the answer is supported. Against **indirect prompt
injection**, that check does not merely underperform. It fails by construction: the
attacker's text is inside the context it reads, so it returns *supported*.

Calling the same scorer **per passage** and looking for internal disagreement catches
most of those attacks, with no retraining and no model internals.

| poisoned passages | whole-context check | per-passage canary | localizes the poison |
|---|---|---|---|
| 1 of 5 | 1.7% | **60.8%** | 97.5% |
| 2 of 5 | 1.7% | **78.3%** | 98.3% |
| 3 of 5 | 0.0% | **80.0%** | 98.3% |
| 4 of 5 | 0.0% | **81.7%** | 98.3% |

Detection rate at a 10% false-positive budget, set on clean traffic only. 240 sets per
condition, half poisoned.

### The failure mode is retrieval, not attack volume

Detection *rises* as the attacker takes more of the retrieved set, because more poison
means a stronger support signal against unchanged contradicting evidence. Flooding the
context makes an attacker more visible, not less.

What actually blinds the canary is removing the trusted evidence from retrieval:

| condition | detection |
|---|---|
| 1 of 5 poisoned, true evidence retrieved | 60.8% |
| 1 of 5 poisoned, **true evidence displaced** | **25.8%** |

So the precondition is "did retrieval return at least one trusted contradicting
passage," not "what fraction is poisoned." That is a useful thing for a deployer to
know, because it moves the defensive burden from detecting adversarial text, which is
hard, to retrieval diversity and redundancy, which is ordinary engineering.

## The scorer

DeBERTa-v3-base (184M), fine-tuned on 45,404 decontaminated examples from RAGTruth,
FEVER, and HaluEval, with a 3-way head and temperature calibration.

| | LLM-AggreFact | | RAGTruth (in-domain) | |
|---|---|---|---|---|
| | zero-shot | fine-tuned | zero-shot | fine-tuned |
| balanced accuracy | 0.591 | **0.695** | 0.575 | **0.807** |
| F1 (not-supported) | 0.422 | **0.508** | 0.494 | **0.740** |
| ECE | 0.474 | **0.181** | 0.275 | **0.019** |
| gate lift | 0.45 | **0.63** | 0.37 | **0.84** |

LLM-AggreFact is the leaderboard-comparable surface. RAGTruth is in-domain and
therefore flatters the fine-tuned model; both are shown rather than only the favourable
one.

Per-corpus across AggreFact's eleven constituent datasets: accuracy improved on 8 of
11, calibration on 10 of 11. Lfqa, which was never trained on, gained +0.213 balanced
accuracy, close to in-domain RAGTruth's +0.222. The two news-summarization sets
regressed (AggreFact-CNN −0.112, AggreFact-XSum −0.049).

### Why calibration is reported as a first-class number

A scorer is only useful as a *gate* if its confidence means something. You auto-accept
above a threshold and review the rest, and that only pays if the low-confidence pile is
genuinely enriched for errors.

Fitting a temperature is the standard post-hoc fix, and where it lands is diagnostic.
The zero-shot baseline's fit **ran to the edge of the search range and kept pushing**,
which is the optimizer saying the best thing to do with this model's confidence is
discard it. The fine-tuned scorer fits at **T = 1.75**, comfortably interior: mildly
overconfident, straightforwardly correctable.

That difference does not show up in an accuracy column, and it decides whether the
thing can be deployed as a gate at all.

## Limitations

- **The injections are constructed, not captured.** Template payloads over FEVER
  claims. The claim is "grounding drift detects injection-induced inconsistency under
  minority poisoning, on synthetic cases." It is not "this defeats PoisonedRAG."
- **Calibration does not transfer cleanly across claim granularity.** The same corpus
  scores ECE 0.019 at response level and 0.201 at sentence level, because temperature
  was fitted on response-level validation data.
- **AggreFact is 56% RAGTruth**, which is in the training mix (disjoint documents, but
  the same task and annotation scheme). The aggregate is not a zero-shot number, and
  the per-corpus breakdown is the honest view.
- **Accuracy is not state of the art.** MiniCheck-class systems report higher balanced
  accuracy on AggreFact. The contributions here are the injection result, the
  calibration discipline, and the harness.
- **No efficiency numbers yet.** Latency has to be measured on a named reference
  machine, and a developer laptop is not one.

## Design commitments

**Protocols, not inheritance.** A scorer is anything with `score`, `score_batch`, and
`efficiency`. Adding one is a single file plus a registry line.

**Metrics live in one module.** Training-time eval and harness-time eval both call
`groundcheck.eval.metrics`, so a training log and a leaderboard row cannot disagree
about what balanced accuracy means.

**Contamination is measured at the document level, not the pair level.** Aggregators
re-derive claims: AggreFact decomposes RAGTruth responses into sentence-level claims
and reformats the passages, so no `(context, claim)` pair survives intact even where
the underlying documents are shared. Pair matching found 0 overlaps across 46,072
training rows; document matching found 668. RAGTruth, HaluEval, and AggreFact-CNN all
derive from CNN/DailyMail.

**Latency is never measured on an unlabeled machine.** `BenchmarkDevice` refuses to
construct without an explicit label, so a laptop number cannot be published as
commodity-CPU latency.

**Sampling is a seeded shuffle, never the head.** Published datasets are ordered by
corpus. AggreFact's first 300 test rows are 95% one class.

**Single-class slices report NaN**, not a flattering number.

## Labels

The model predicts **3 ways** internally: `supported`, `contradicted`, `neutral`. This
fits the MNLI warm start and preserves the contradiction signal.

For reporting it collapses to **binary**, where `supported` is true and both other
labels are false. The **class of interest is not-supported**, the rare and costly one,
and headline precision, recall, F1, and PR-AUC are all reported on it.

The 3-way head is not cosmetic. The injection canary needs `P(contradicted)`
specifically: a passage that simply never mentions the claim is *neutral*, not
contradicting, and an early version of the canary that used `1 - P(supported)` scored
**0% detection** because every ordinary retrieved set looked maximally conflicted.
Reading contradiction from the 3-way head took it to 61%. The binary collapse leaves
accuracy untouched and destroys the security signal.

Sources differ in how much they know. Rows whose dataset records only "hallucinated,"
without saying whether it contradicts or merely invents, are tagged
`label3_source="coarse"` and supervise the collapsed decision only, via a masked loss.
They never teach a contradiction no annotator assigned.

## Install

```bash
uv sync                      # core + dev tooling
uv sync --all-extras         # adds torch, transformers, gradio
uv run pytest                # 167 tests
uv run pytest -m network     # live dataset schema checks
```

Developed and tested on Python 3.12, the supported floor. LLM-AggreFact is gated;
`uv run hf auth login` with a read token after accepting its terms.

## Reproduce

```bash
uv run python scripts/run_eval.py configs/phase0_smoke.yaml   # baseline leaderboard
uv run python scripts/run_train.py configs/train_v1_local.yaml  # fine-tune
uv run python scripts/run_canary.py                             # injection sweep
```

## Layout

```
groundcheck/
├── data/           Example schema, adapters, decontamination, injection-set builder
├── scorers/        Scorer protocol, zero-shot NLI baseline, fine-tuned scorer
├── eval/           metrics · calibration-aware reporting · efficiency · risk-coverage
├── canary.py       per-passage conflict detection
├── calibration.py  temperature scaling, with saturation detection
├── losses.py       3-way objective with binary supervision for coarse labels
└── device.py       inferred compute device vs declared benchmark device
```

## Roadmap

- [x] Interfaces, metrics, device discipline, CI
- [x] Dataset adapters, decontamination, zero-shot baseline, runner, report
- [x] Fine-tuned scorer, calibration, risk-coverage gate analysis
- [x] Minimal injection canary and detection sweep
- [ ] Comparators (HHEM, MiniCheck, Granite Guardian), published weights, demo
- [ ] Efficiency profile on a named reference machine, INT8 variant
- [ ] Real attack payloads (PoisonedRAG, BIPIA) and a released dataset
- [ ] Agent-trajectory extension: per-step drift localization

## License

Apache-2.0. Curated datasets, when released, are licensed separately.
