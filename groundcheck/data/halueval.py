"""HaluEval adapter.

Source: `pminervini/HaluEval`. Each native row is a *pair*: one source passage with a
correct response and a hallucinated one. Every row therefore yields two examples, one
supported and one not, which is why HaluEval is close to balanced by construction while
RAGTruth is not.

Two things about this source need stating plainly, because both affect what the labels
mean.

**The 3-way label is coarse.** HaluEval marks a response as hallucinated without
recording whether it contradicts the passage or merely invents detail the passage never
covers. Those rows get "neutral", which asserts only "not supported", and are tagged
`label3_source="coarse"` so training can mask the 3-way loss on them. The binary label
is exact regardless. Inventing a "contradicted" label here would feed the injection
canary a distinction no annotator ever made.

**The `general` config is excluded.** It holds ChatGPT responses to user queries with a
yes/no hallucination flag and no source passage. Groundedness is defined relative to
evidence, so a row with no evidence is not a groundedness example, whatever its flag
says. The other three configs all carry a passage.

The two halves of a pair are assigned to the same split, since scoring a model on a
hallucinated answer whose correct twin it trained on would leak.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from groundcheck.data.base import Example
from groundcheck.data.splits import assign_split

REPO = "pminervini/HaluEval"


@dataclass(frozen=True)
class _ConfigSpec:
    context_field: str
    right_field: str
    wrong_field: str
    task: str
    query_field: str | None = None


# `general` is deliberately absent: it has no source passage. See the module docstring.
CONFIGS: dict[str, _ConfigSpec] = {
    "qa": _ConfigSpec("knowledge", "right_answer", "hallucinated_answer", "qa", "question"),
    "summarization": _ConfigSpec(
        "document", "right_summary", "hallucinated_summary", "summarization"
    ),
    "dialogue": _ConfigSpec(
        "knowledge", "right_response", "hallucinated_response", "dialogue", "dialogue_history"
    ),
}

# `qa` is available but off by default: its correct answers are bare spans
# ("Arthur's Magazine") while its hallucinated answers are full sentences. A noun
# phrase asserts nothing, so an entailment scorer rates correct answers *below*
# hallucinated ones and the label ends up tracking claim form rather than truth.
# Converting question plus answer into a declarative claim would fix it and is a
# curation experiment, not a v1 dependency. Summarization and dialogue carry
# sentence-length responses on both sides and are unaffected.
DEFAULT_CONFIGS: tuple[str, ...] = ("summarization", "dialogue")


def to_examples(row: dict, config: str, index: int, split: str | None = None) -> list[Example]:
    """Convert one native row into its supported and not-supported halves."""
    spec = CONFIGS[config]
    context = (row.get(spec.context_field) or "").strip()
    if not context:
        return []

    base_id = f"halueval-{config}-{index}"
    resolved = split if split is not None else assign_split(base_id)

    out: list[Example] = []
    for suffix, field, label in (
        ("pos", spec.right_field, "supported"),
        ("neg", spec.wrong_field, "neutral"),
    ):
        claim = (row.get(field) or "").strip()
        if not claim:
            continue
        out.append(
            Example(
                context=context,
                claim=claim,
                label=label,
                meta={
                    "dataset": "halueval",
                    "domain": spec.task,
                    "id": f"{base_id}-{suffix}",
                    "task": spec.task,
                    "granularity": "answer",
                    # Only the hallucinated half is coarse; a correct response is
                    # unambiguously supported.
                    "label3_source": "native" if label == "supported" else "coarse",
                    "split": resolved,
                    "config": config,
                    "pair_id": base_id,
                    "query": row.get(spec.query_field) if spec.query_field else None,
                },
            )
        )
    return out


class HaluEval:
    name = "halueval"
    domain = "mixed"

    def __init__(
        self,
        configs: tuple[str, ...] = DEFAULT_CONFIGS,
        repo: str = REPO,
    ):
        unknown = set(configs) - set(CONFIGS)
        if unknown:
            raise ValueError(
                f"unsupported HaluEval configs {sorted(unknown)}; "
                f"available: {sorted(CONFIGS)} ('general' has no source passage)"
            )
        self.configs = configs
        self.repo = repo

    def load(self, split: str, limit: int | None = None) -> list[Example]:
        """Load a deterministic split. HaluEval ships one undivided pool of 10k rows."""
        if split not in {"train", "validation", "test", "all"}:
            raise ValueError(f"halueval splits are train/validation/test/all, got {split!r}")

        out: list[Example] = []
        for config in self.configs:
            raw = load_dataset(self.repo, config, split="data")
            for i, row in enumerate(raw):
                examples = to_examples(row, config, i)
                if split != "all":
                    examples = [e for e in examples if e.meta["split"] == split]
                out.extend(examples)
                if limit is not None and len(out) >= limit:
                    return out[:limit]
        return out
