"""Config and data preparation, tested without torch.

The training loop itself is not unit-tested — it is HuggingFace's, and asserting that
loss decreases is slow and tells you little. What is tested here is everything that
decides *what the model sees*: the mix, the label encoding, and the decontamination
that keeps the benchmark honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from groundcontrol.data.base import Example
from groundcontrol.train import (
    LABEL_TO_INDEX,
    SourceSpec,
    TrainConfig,
    build_compute_metrics,
    encode_labels,
    prepare_training_data,
)


def _ex(label="supported", coarse=False, context="a passage", claim="a claim"):
    meta = {"label3_source": "coarse" if coarse else "native"}
    return Example(context=context, claim=claim, label=label, meta=meta)


# --- config -----------------------------------------------------------------------


def test_config_round_trips_from_yaml(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "run_name: demo\n"
        "epochs: 2\n"
        "train_sources:\n  - name: ragtruth\n    split: train\n    limit: 50\n"
        "eval_sources:\n  - name: fever\n    split: validation\n"
        "decontaminate_against:\n  - name: aggrefact\n    split: test\n"
    )
    cfg = TrainConfig.from_yaml(path)

    assert cfg.run_name == "demo"
    assert cfg.epochs == 2
    assert cfg.train_sources[0] == SourceSpec("ragtruth", "train", 50, {})
    assert cfg.decontaminate_against[0].name == "aggrefact"


def test_config_requires_a_run_name():
    with pytest.raises(ValueError, match="run_name"):
        TrainConfig.from_dict({"epochs": 1})


def test_shipped_training_config_is_valid_and_holds_out_the_benchmark():
    cfg = TrainConfig.from_yaml("configs/train_v1.yaml")
    trained_on = {s.name for s in cfg.train_sources}

    assert "aggrefact" not in trained_on, "the benchmark must never be trained on"
    assert "aggrefact" in {s.name for s in cfg.decontaminate_against}
    # An mnli-only warm start; the fever-anli variant has already seen an eval corpus.
    assert "fever-anli" not in cfg.base_model


# --- label encoding ---------------------------------------------------------------


def test_labels_encode_to_the_loss_module_ordering():
    labels, _ = encode_labels([_ex("supported"), _ex("contradicted"), _ex("neutral")])
    assert list(labels) == [
        LABEL_TO_INDEX["supported"],
        LABEL_TO_INDEX["contradicted"],
        LABEL_TO_INDEX["neutral"],
    ]


def test_coarse_mask_marks_only_untrustworthy_three_way_labels():
    _, coarse = encode_labels([_ex(coarse=False), _ex("neutral", coarse=True), _ex()])
    assert list(coarse) == [False, True, False]


def test_examples_without_provenance_are_treated_as_reliable():
    # Absent metadata should not silently downgrade supervision.
    example = Example(context="c", claim="q", label="supported")
    assert encode_labels([example])[1].tolist() == [False]


# --- metrics wiring ---------------------------------------------------------------


def test_training_metrics_come_from_the_harness_module():
    # A perfect 3-way prediction should read as perfect on the collapsed decision.
    logits = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
    labels = np.array([0, 1, 2])

    result = build_compute_metrics()((logits, labels))

    assert result["balanced_acc"] == pytest.approx(1.0)
    assert result["f1_notsup"] == pytest.approx(1.0)
    assert set(result) == {"balanced_acc", "f1_notsup", "pr_auc_notsup", "ece"}


def test_temperature_changes_calibration_but_not_decisions():
    # Only holds because metrics marginalize to binary *before* scaling. Applying
    # temperature to the 3-vector and reading the supported column would let that
    # probability cross 0.5, silently moving decisions.
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 3)) * 4
    labels = rng.integers(0, 3, size=200)

    sharp = build_compute_metrics(1.0)((logits, labels))
    soft = build_compute_metrics(20.0)((logits, labels))

    assert soft["ece"] != sharp["ece"]
    assert soft["balanced_acc"] == pytest.approx(sharp["balanced_acc"])
    assert soft["f1_notsup"] == pytest.approx(sharp["f1_notsup"])


# --- decontamination wiring -------------------------------------------------------


def test_preparation_removes_training_rows_sharing_an_evaluation_document(monkeypatch):
    shared = "A passage that both the training mix and the benchmark happen to use. " * 3
    train = [_ex(context=shared, claim="train claim"), _ex(context="unrelated " * 20)]
    held_out = [_ex(context=shared, claim="a re-derived sentence")]

    sources = {"train": train, "held": held_out}
    monkeypatch.setattr(
        "groundcontrol.train.load_sources",
        lambda specs: sources[specs[0].name],
    )

    cfg = TrainConfig(
        run_name="t",
        train_sources=[SourceSpec("train", "train")],
        decontaminate_against=[SourceSpec("held", "test")],
    )
    kept, provenance = prepare_training_data(cfg)

    assert len(kept) == 1
    assert provenance["n_removed_shared_documents"] == 1
    assert provenance["n_after"] == 1


def test_preparation_refuses_an_empty_mix(monkeypatch):
    monkeypatch.setattr("groundcontrol.train.load_sources", lambda specs: [])
    cfg = TrainConfig(run_name="t", train_sources=[SourceSpec("nothing", "train")])

    with pytest.raises(ValueError, match="no training examples"):
        prepare_training_data(cfg)
