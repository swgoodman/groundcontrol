"""Live checks against the gated LLM-AggreFact dataset.

Requires accepting the terms on the dataset page and a read token
(`uv run hf auth login`). Skipped when no token is present rather than failed, since
absence of credentials is not a defect.

Run with `uv run pytest -m network`.
"""

from __future__ import annotations

import pytest

from groundcheck.data.aggrefact import AggreFact
from groundcheck.data.base import LABELS_3

pytestmark = pytest.mark.network


def _skip_without_access():
    from huggingface_hub import get_token

    if not get_token():
        pytest.skip("no HuggingFace token; run `uv run hf auth login`")


def test_aggrefact_live_schema():
    _skip_without_access()
    examples = AggreFact().load("test", limit=300)

    assert examples
    for ex in examples:
        assert ex.context.strip() and ex.claim.strip()
        assert ex.label in LABELS_3
        assert ex.meta["source_dataset"]
    # A benchmark that was all one class would make balanced accuracy meaningless.
    supported = sum(e.supported for e in examples)
    assert 0.1 < supported / len(examples) < 0.9


def test_aggrefact_aggregates_several_upstream_corpora():
    _skip_without_access()
    corpora = {e.meta["source_dataset"] for e in AggreFact().load("test", limit=2000)}
    assert len(corpora) > 1


def test_label_polarity_is_not_inverted():
    """An inverted 0/1 mapping would put an NLI baseline symmetrically below chance.

    This is the check that the documented `SUPPORTED_LABEL = 1` assumption is right.
    Cheap to state, and the failure it catches is otherwise invisible: every number
    downstream would look plausible and be backwards.
    """
    _skip_without_access()
    import numpy as np

    from groundcheck.scorers.nli_zeroshot import NLIZeroShot

    examples = AggreFact().load("test", limit=300)
    verdicts = NLIZeroShot().score_batch(examples)

    p = np.array([v.score for v in verdicts])
    y = np.array([e.supported for e in examples])
    assert p[y].mean() > p[~y].mean(), (
        "entailment probability is higher for unsupported claims than supported ones, "
        "which means SUPPORTED_LABEL is inverted"
    )


def test_source_dataset_filter():
    _skip_without_access()
    all_corpora = {e.meta["source_dataset"] for e in AggreFact().load("test", limit=2000)}
    one = sorted(all_corpora)[0]

    filtered = AggreFact(source_datasets=(one,)).load("test", limit=100)
    assert {e.meta["source_dataset"] for e in filtered} == {one}
