"""Fine-tune a groundedness scorer.

Lives in the package rather than in notebook cells so it can be tested, and so the
notebook stays a thin driver that clones, installs, and calls `train()`. The same
arrangement is what lets a Colab run and a local run be the same code.

Evaluation during training calls `groundcheck.eval.metrics`, the module the harness
uses, so a training log and a leaderboard row cannot disagree about what balanced
accuracy means.

torch and transformers are imported inside functions: the package must stay importable
without them, and the config and data-prep logic below is tested in an environment that
has neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groundcheck.data.base import Example
from groundcheck.data.decontaminate import decontaminate, drop_shared_documents
from groundcheck.losses import LABEL_ORDER, SUPPORTED
from groundcheck.registry import get_dataset

LABEL_TO_INDEX = {name: i for i, name in enumerate(LABEL_ORDER)}


@dataclass(slots=True)
class SourceSpec:
    name: str
    split: str
    limit: int | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainConfig:
    run_name: str
    base_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli"
    train_sources: list[SourceSpec] = field(default_factory=list)
    eval_sources: list[SourceSpec] = field(default_factory=list)
    # Held out from training and never trained on. Overlap against it is measured and
    # removed, so a benchmark number stays comparable.
    decontaminate_against: list[SourceSpec] = field(default_factory=list)

    max_length: int = 512
    learning_rate: float = 2e-5
    epochs: float = 3.0
    batch_size: int = 16
    eval_batch_size: int = 32
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 0
    fp16: bool = False

    # Step-based rather than epoch-based, so a long run is observable while it runs.
    # Epoch-end-only reporting means no signal until half the job is spent.
    logging_steps: int = 50
    eval_steps: int = 500
    save_total_limit: int = 2

    output_dir: str = "artifacts"
    hub_repo_id: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, raw: dict) -> TrainConfig:
        if "run_name" not in raw:
            raise ValueError("config is missing required key: run_name")

        def specs(key: str) -> list[SourceSpec]:
            return [
                SourceSpec(
                    name=s["name"],
                    split=s.get("split", "train"),
                    limit=s.get("limit"),
                    args=s.get("args", {}),
                )
                for s in raw.get(key, [])
            ]

        known = {f for f in cls.__slots__} - {
            "train_sources",
            "eval_sources",
            "decontaminate_against",
        }
        scalars = {k: v for k, v in raw.items() if k in known}
        return cls(
            train_sources=specs("train_sources"),
            eval_sources=specs("eval_sources"),
            decontaminate_against=specs("decontaminate_against"),
            **scalars,
        )


def load_sources(specs: list[SourceSpec]) -> list[Example]:
    examples: list[Example] = []
    for spec in specs:
        dataset = get_dataset(spec.name, **spec.args)
        examples.extend(dataset.load(spec.split, limit=spec.limit))
    return examples


def encode_labels(examples: list[Example]) -> tuple[np.ndarray, np.ndarray]:
    """Label indices plus the mask marking rows whose 3-way label is untrustworthy."""
    labels = np.array([LABEL_TO_INDEX[e.label] for e in examples], dtype=np.int64)
    coarse = np.array([e.meta.get("label3_source") == "coarse" for e in examples], dtype=bool)
    return labels, coarse


def prepare_training_data(config: TrainConfig) -> tuple[list[Example], dict]:
    """Load the mix and strip anything overlapping the held-out benchmark.

    Both checks run. Pair matching removes exact reuse; document matching catches the
    case where an aggregator re-derived claims from a passage we also train on, which
    pair matching cannot see and would silently report as clean.
    """
    train = load_sources(config.train_sources)
    if not train:
        raise ValueError("no training examples; check train_sources in the config")

    provenance: dict[str, Any] = {"n_loaded": len(train)}

    if config.decontaminate_against:
        held_out = load_sources(config.decontaminate_against)
        train, report = decontaminate(train, held_out)
        train, n_shared_docs = drop_shared_documents(train, held_out)
        provenance.update(
            {
                "n_removed_pairs": report.n_removed,
                "n_removed_shared_documents": n_shared_docs,
                "n_after": len(train),
                "summary": report.summary(),
            }
        )

    return train, provenance


def build_compute_metrics(temperature: float = 1.0):
    """Training-time metrics, computed by the same module the harness uses."""
    from groundcheck.calibration import collapse_to_binary_logits, softmax

    def compute_metrics(eval_pred):
        from groundcheck.eval import metrics as metrics_mod

        logits, label_ids = eval_pred
        binary = collapse_to_binary_logits(np.asarray(logits), supported_index=SUPPORTED)
        p_supported = softmax(binary, temperature)[:, 1]
        y_true = np.asarray(label_ids) == SUPPORTED

        result = metrics_mod.compute(y_true, p_supported)
        return {
            "balanced_acc": result.balanced_acc,
            "f1_notsup": result.f1_notsup,
            "pr_auc_notsup": result.pr_auc_notsup,
            "ece": result.ece,
        }

    return compute_metrics


def tokenize(
    examples: list[Example], tokenizer, max_length: int, max_claim_tokens: int | None = None
):
    """Encode (context, claim) pairs, spending the window on the context first.

    The claim gets a reserved budget and the context takes the rest. Without the
    reservation, `only_first` raises whenever a claim alone exceeds the window, which
    RAGTruth hits because its claims are whole model responses rather than sentences.

    Claims are pre-truncated to that budget, so a long claim degrades predictably
    instead of the encoding failing or silently discarding the evidence.
    """
    claims = [e.claim for e in examples]
    budget = max_claim_tokens or max_length // 2

    lengths = tokenizer(claims, add_special_tokens=False)["input_ids"]
    if any(len(ids) > budget for ids in lengths):
        claims = tokenizer.batch_decode(
            tokenizer(claims, truncation=True, max_length=budget, add_special_tokens=False)[
                "input_ids"
            ],
            skip_special_tokens=True,
        )

    return tokenizer(
        [e.context for e in examples],
        claims,
        truncation="only_first",
        max_length=max_length,
        padding=False,
    )


def count_truncated_claims(examples: list[Example], tokenizer, budget: int) -> int:
    encoded = tokenizer([e.claim for e in examples], add_special_tokens=False)["input_ids"]
    return sum(1 for ids in encoded if len(ids) > budget)


def build_dataset(examples: list[Example], tokenizer, max_length: int):
    from datasets import Dataset

    encoded = tokenize(examples, tokenizer, max_length)
    labels, coarse = encode_labels(examples)
    return Dataset.from_dict(
        {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "coarse": coarse,
        }
    )


def _make_trainer_class():
    """Build the Trainer subclass at call time, so importing this module needs no torch."""
    from transformers import Trainer

    from groundcheck.losses import masked_three_way_loss

    class GroundcheckTrainer(Trainer):
        class_weights = None

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            coarse = inputs.pop("coarse")
            outputs = model(**inputs)
            loss = masked_three_way_loss(
                outputs.logits,
                labels,
                coarse,
                class_weights=(
                    self.class_weights.to(outputs.logits.device)
                    if self.class_weights is not None
                    else None
                ),
            )
            return (loss, outputs) if return_outputs else loss

    return GroundcheckTrainer


def _collator(tokenizer):
    """Pad dynamically while carrying `coarse` through, which the default collator drops."""
    from transformers import DataCollatorWithPadding

    base = DataCollatorWithPadding(tokenizer)

    def collate(features):
        import torch

        coarse = torch.tensor([f.pop("coarse") for f in features], dtype=torch.bool)
        batch = base(features)
        batch["coarse"] = coarse
        return batch

    return collate


def fit_temperature_on(model, dataset, tokenizer, config: TrainConfig) -> tuple[float, dict]:
    """Fit temperature on validation logits and report what it did to calibration."""
    import torch
    from transformers import TrainingArguments

    from groundcheck.calibration import collapse_to_binary_logits, fit_temperature, softmax
    from groundcheck.eval import metrics as metrics_mod

    trainer_cls = _make_trainer_class()
    trainer = trainer_cls(
        model=model,
        args=TrainingArguments(
            output_dir=str(Path(config.output_dir) / config.run_name / "predict"),
            per_device_eval_batch_size=config.eval_batch_size,
            remove_unused_columns=False,
            report_to=[],
        ),
        data_collator=_collator(tokenizer),
    )
    with torch.inference_mode():
        output = trainer.predict(dataset)

    logits = np.asarray(output.predictions)
    labels = np.asarray(dataset["labels"])
    binary = (labels == SUPPORTED).astype(int)

    collapsed = collapse_to_binary_logits(logits, supported_index=SUPPORTED)
    temperature = fit_temperature(collapsed, binary)

    before = metrics_mod.compute(binary.astype(bool), softmax(collapsed, 1.0)[:, 1])
    after = metrics_mod.compute(binary.astype(bool), softmax(collapsed, temperature)[:, 1])

    return temperature, {
        "temperature": temperature,
        "ece_before": before.ece,
        "ece_after": after.ece,
        "balanced_acc": after.balanced_acc,
    }


def train(config: TrainConfig, push_to_hub: bool = False) -> dict:
    """Run the fine-tune end to end and write artifacts. Returns a run summary."""
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        set_seed,
    )

    from groundcheck.device import describe_runtime, resolve_compute_device
    from groundcheck.losses import class_weights_from_labels

    set_seed(config.seed)
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples, provenance = prepare_training_data(config)
    eval_examples = load_sources(config.eval_sources)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    train_ds = build_dataset(train_examples, tokenizer, config.max_length)
    eval_ds = build_dataset(eval_examples, tokenizer, config.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=3,
        id2label=dict(enumerate(LABEL_ORDER)),
        label2id=LABEL_TO_INDEX,
        ignore_mismatched_sizes=True,
    )

    labels, _ = encode_labels(train_examples)
    trainer_cls = _make_trainer_class()
    trainer_cls.class_weights = class_weights_from_labels(torch.tensor(labels))

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_notsup",
        greater_is_better=True,
        seed=config.seed,
        fp16=config.fp16,
        report_to=[],
        # Trainer drops columns the model's forward does not accept, which would take
        # `coarse` with it and silently supervise every row as if its 3-way label were
        # reliable.
        remove_unused_columns=False,
    )

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_collator(tokenizer),
        compute_metrics=build_compute_metrics(),
    )
    trainer.train()

    temperature, calibration = fit_temperature_on(model, eval_ds, tokenizer, config)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    if push_to_hub and config.hub_repo_id:
        model.push_to_hub(config.hub_repo_id)
        tokenizer.push_to_hub(config.hub_repo_id)

    return {
        "run_name": config.run_name,
        "n_train": len(train_examples),
        "n_eval": len(eval_examples),
        "coarse_rate": float(encode_labels(train_examples)[1].mean()),
        "provenance": provenance,
        "calibration": calibration,
        "runtime": describe_runtime(),
        "compute_device": resolve_compute_device(),
        "output_dir": str(output_dir),
    }
