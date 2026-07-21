"""The core package must not drag in optional dependencies.

`datasets`, `torch`, and `transformers` are optional extras. Someone dropping the CI
gate into their pipeline needs the scorer and the metrics, not the dataset loaders, and
CI runs the hermetic suite without any of them installed.

The registry keeps this true by declaring components as import paths and resolving them
only when one is constructed, so listing names is free and only `get_*` pays.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

OPTIONAL = ("datasets", "torch", "transformers", "gradio")


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True
    )


def test_importing_the_core_does_not_import_optional_dependencies():
    result = _run(f"""
        import sys

        import groundcontrol
        import groundcontrol.data
        import groundcontrol.device
        import groundcontrol.registry
        from groundcontrol.eval import metrics

        leaked = [name for name in {OPTIONAL!r} if name in sys.modules]
        assert not leaked, f"core import pulled in optional dependencies: {{leaked}}"
    """)
    assert result.returncode == 0, result.stderr


def test_listing_components_does_not_import_their_backends():
    # The point of the lazy registry: a config can name a dataset, and the harness can
    # report what is available, without paying for datasets or torch.
    result = _run(f"""
        import sys
        from groundcontrol.registry import available_datasets, available_scorers

        assert set(available_datasets()) >= {{"ragtruth", "fever", "halueval"}}
        available_scorers()

        leaked = [name for name in {OPTIONAL!r} if name in sys.modules]
        assert not leaked, f"listing imported backends: {{leaked}}"
    """)
    assert result.returncode == 0, result.stderr


def test_getting_a_dataset_imports_its_backend():
    # The other half of the contract: resolution is deferred, not skipped.
    result = _run("""
        import sys
        from groundcontrol.registry import get_dataset

        assert "datasets" not in sys.modules
        get_dataset("fever")
        assert "datasets" in sys.modules
    """)
    assert result.returncode == 0, result.stderr
