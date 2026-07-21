"""Runner, efficiency, report, and config, exercised with a stub scorer.

No model is loaded here. The harness contract is that it works with *any* conforming
scorer, so testing it against a real encoder would only make the tests slow and the
failures ambiguous.
"""

from __future__ import annotations

import json

import pytest

from groundcontrol.config import ExperimentConfig
from groundcontrol.data.base import Example
from groundcontrol.device import BenchmarkDevice
from groundcontrol.eval import report
from groundcontrol.eval.efficiency import measure
from groundcontrol.eval.runner import run
from groundcontrol.scorers.base import EfficiencyProfile, Verdict


class StubScorer:
    """Returns a preset score per example, so expected metrics are computable by hand."""

    name = "stub"
    training_corpora = ()

    def __init__(self, scores: list[float] | None = None, constant: float = 0.5):
        self.scores = scores
        self.constant = constant
        self.calls = 0

    def score(self, context: str, claim: str) -> Verdict:
        return Verdict(supported=self.constant >= 0.5, score=self.constant)

    def score_batch(self, items: list[Example]) -> list[Verdict]:
        out = []
        for _ in items:
            p = self.scores[self.calls] if self.scores else self.constant
            self.calls += 1
            out.append(Verdict(supported=p >= 0.5, score=p))
        return out

    def efficiency(self) -> EfficiencyProfile:
        return EfficiencyProfile(hosted=False, params_m=1.0, size_mb=4.0, cost_per_1k_usd=0.0)


def _examples(labels: list[str]) -> list[Example]:
    return [
        Example(context=f"context {i}", claim=f"claim {i}", label=label, meta={"id": str(i)})
        for i, label in enumerate(labels)
    ]


def _device() -> BenchmarkDevice:
    return BenchmarkDevice(device="cpu", label="test machine")


# --- runner -----------------------------------------------------------------------


def test_run_produces_metrics_from_scorer_output():
    examples = _examples(["supported", "supported", "neutral", "contradicted"])
    scorer = StubScorer(scores=[0.9, 0.8, 0.1, 0.2])
    result = run(scorer, examples, dataset_name="stubset", split="test")

    assert result.metrics.n == 4
    assert result.metrics.balanced_acc == pytest.approx(1.0)
    assert result.scorer == "stub"
    assert result.dataset == "stubset"


def test_run_records_the_class_balance_and_coarse_label_rate():
    # Both belong next to the score: a leaderboard row is unreadable without knowing
    # how skewed the slice was, and how much of the 3-way signal was trustworthy.
    examples = _examples(["supported", "neutral"])
    examples[1].meta["label3_source"] = "coarse"
    result = run(StubScorer(), examples, dataset_name="stubset", split="test")

    assert result.notes["supported_rate"] == 0.5
    assert result.notes["coarse_label3_rate"] == 0.5


def test_run_rejects_a_scorer_that_drops_examples():
    class Truncating(StubScorer):
        def score_batch(self, items):
            return super().score_batch(items)[:-1]

    with pytest.raises(ValueError, match="verdicts"):
        run(Truncating(), _examples(["supported", "neutral"]), dataset_name="s", split="test")


def test_run_requires_examples():
    with pytest.raises(ValueError, match="zero examples"):
        run(StubScorer(), [], dataset_name="s", split="test")


def test_efficiency_is_skipped_unless_a_benchmark_device_is_given():
    # Timing without a named machine would produce an uncomparable number, so the
    # default is to produce none at all.
    examples = _examples(["supported", "neutral"])
    assert run(StubScorer(), examples, dataset_name="s", split="test").efficiency is None
    result = run(StubScorer(), examples, dataset_name="s", split="test", benchmark_device=_device())
    assert result.efficiency.device_label == "test machine"


# --- efficiency -------------------------------------------------------------------


def test_measure_reports_latency_and_carries_the_static_profile():
    eff = measure(StubScorer(), _examples(["supported"] * 20), _device(), warmup=1)
    assert eff.latency_ms_p50 >= 0
    assert eff.latency_ms_p95 >= eff.latency_ms_p50
    assert eff.throughput_qps > 0
    assert eff.size_mb == 4.0 and eff.hosted is False


def test_measure_discards_warmup_batches():
    scorer = StubScorer()
    measure(scorer, _examples(["supported"] * 10), _device(), batch_size=1, warmup=3)
    # 3 warmup + 10 timed: warmup runs, it just does not count toward the timings.
    assert scorer.calls == 13


def test_measure_requires_examples():
    with pytest.raises(ValueError, match="zero examples"):
        measure(StubScorer(), [], _device())


# --- report -----------------------------------------------------------------------


def test_markdown_renders_a_row_per_result_sorted_by_accuracy():
    weak = run(StubScorer(scores=[0.9, 0.9]), _examples(["supported", "neutral"]), "a", "test")
    strong = run(StubScorer(scores=[0.9, 0.1]), _examples(["supported", "neutral"]), "b", "test")
    md = report.to_markdown([weak, strong])

    assert md.index("| stub | b |") < md.index("| stub | a |")


def test_markdown_flags_contamination():
    class FeverTrained(StubScorer):
        training_corpora = ("mnli", "fever")

    examples = _examples(["supported", "neutral"])
    for e in examples:
        e.meta["dataset"] = "fever"

    result = run(FeverTrained(), examples, "fever", "validation")
    md = report.to_markdown([result])
    assert "## Contamination" in md
    assert "not comparable to a zero-shot entry" in md


def test_contamination_is_detected_through_an_aggregated_benchmark():
    # The trap this exists to close. A RAGTruth-trained scorer evaluated on AggreFact
    # looks leaderboard-comparable and is not, because AggreFact redistributes
    # RAGTruth. Nothing hardcodes that relationship: the examples report their own
    # upstream corpus, so the overlap is observed rather than looked up.
    class RagTruthTrained(StubScorer):
        training_corpora = ("mnli", "ragtruth")

    examples = _examples(["supported", "neutral"])
    for e in examples:
        e.meta.update({"dataset": "aggrefact", "source_dataset": "RAGTruth"})

    result = run(RagTruthTrained(), examples, "aggrefact", "test")

    assert result.notes["upstream_corpora_inside_this_eval_set"] == ["RAGTruth"]
    assert "ragtruth" in report.to_markdown([result]).lower()
    assert "## Contamination" in report.to_markdown([result])


def test_no_contamination_warning_when_the_corpora_are_disjoint():
    class MnliOnly(StubScorer):
        training_corpora = ("mnli",)

    examples = _examples(["supported", "neutral"])
    for e in examples:
        e.meta.update({"dataset": "aggrefact", "source_dataset": "ExpertQA"})

    assert "## Contamination" not in report.to_markdown(
        [run(MnliOnly(), examples, "aggrefact", "test")]
    )


def test_report_surfaces_measured_decontamination():
    result = run(StubScorer(), _examples(["supported", "neutral"]), "aggrefact", "test")
    result.notes["decontamination"] = {"n_removed": 42}

    assert "42 training examples appeared" in report.to_markdown([result])


def test_markdown_reports_undefined_metrics_as_not_available():
    # A single-class slice has no meaningful balanced accuracy. It must read as
    # unavailable rather than as a number.
    result = run(StubScorer(), _examples(["supported", "supported"]), "s", "test")
    assert "n/a" in report.to_markdown([result])


def test_markdown_handles_no_results():
    assert "_No results._" in report.to_markdown([])


def test_write_emits_markdown_and_machine_readable_json(tmp_path):
    result = run(StubScorer(scores=[0.9, 0.1]), _examples(["supported", "neutral"]), "s", "test")
    paths = report.write([result], tmp_path, stem="run1")

    assert paths["markdown"].read_text().startswith("# ")
    payload = json.loads(paths["json"].read_text())
    assert payload[0]["metrics"]["n"] == 2
    assert payload[0]["runtime"]["compute_device"]


# --- config -----------------------------------------------------------------------


def test_config_round_trips_from_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: demo\n"
        "scorers:\n  - name: nli-zeroshot\n    args:\n      batch_size: 8\n"
        "datasets:\n  - name: fever\n    split: validation\n    limit: 10\n"
        "benchmark_device_label: test machine\n"
    )
    cfg = ExperimentConfig.from_yaml(path)

    assert cfg.scorers[0].args == {"batch_size": 8}
    assert (cfg.datasets[0].split, cfg.datasets[0].limit) == ("validation", 10)
    assert cfg.benchmark_device == "cpu"


def test_config_requires_the_core_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        ExperimentConfig.from_dict({"name": "x"})


def test_shipped_smoke_config_is_valid():
    cfg = ExperimentConfig.from_yaml("configs/phase0_smoke.yaml")
    assert cfg.benchmark_device_label
    assert {d.name for d in cfg.datasets} == {"aggrefact", "ragtruth", "fever"}


def test_smoke_config_does_not_report_halueval():
    # HaluEval remains a training source, but its QA answers are bare spans rather
    # than propositions, so reporting it would measure claim form, not groundedness.
    cfg = ExperimentConfig.from_yaml("configs/phase0_smoke.yaml")
    assert "halueval" not in {d.name for d in cfg.datasets}


def test_report_includes_the_gate_section():
    strong = run(
        StubScorer(scores=[0.95, 0.9, 0.85, 0.1, 0.05]),
        _examples(["supported"] * 3 + ["neutral"] * 2),
        "s",
        "test",
    )
    md = report.to_markdown([strong])

    assert "## As a gate" in md
    assert "max coverage under 1% risk" in md


def test_gate_columns_reach_the_leaderboard_row():
    result = run(StubScorer(scores=[0.9, 0.1]), _examples(["supported", "neutral"]), "s", "test")
    row = result.to_row()

    assert row["aurc"] >= 0
    assert row["gate_lift"] == pytest.approx(1.0)
    assert row["cov_at_1pct"] == pytest.approx(0.5)
