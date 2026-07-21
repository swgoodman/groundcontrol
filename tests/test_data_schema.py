import pytest

from groundcontrol.data.base import Example, is_supported


def test_label_collapse_polarity():
    # The single most corruptible convention in the project: which way "supported" runs.
    assert is_supported("supported") is True
    assert is_supported("contradicted") is False
    assert is_supported("neutral") is False


def test_unknown_label_rejected():
    with pytest.raises(ValueError, match="unknown label"):
        is_supported("SUPPORTS")  # FEVER's native label, must be harmonized first


def test_example_exposes_binary_view():
    ex = Example(context="Revenue was $4.2M.", claim="Revenue was $4.2M.", label="supported")
    assert ex.supported is True
    assert Example(context="c", claim="q", label="neutral").supported is False


def test_example_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown label"):
        Example(context="c", claim="q", label="hallucinated")
    with pytest.raises(ValueError, match="context must be non-empty"):
        Example(context="", claim="q", label="supported")
    with pytest.raises(ValueError, match="claim must be non-empty"):
        Example(context="c", claim="", label="supported")


def test_meta_defaults_are_not_shared():
    a = Example(context="c", claim="q", label="supported")
    b = Example(context="c", claim="q", label="supported")
    a.meta["dataset"] = "fever"
    assert b.meta == {}
