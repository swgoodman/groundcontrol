import pytest

from groundcontrol.device import BenchmarkDevice, describe_runtime, resolve_compute_device


def test_compute_device_is_resolvable_and_overridable():
    assert resolve_compute_device() in {"cuda", "mps", "cpu"}
    assert resolve_compute_device(prefer="cpu") == "cpu"


def test_benchmark_device_demands_an_explicit_label():
    # An unlabeled latency number is not comparable across machines, and the whole
    # efficiency thesis rests on the number being comparable.
    with pytest.raises(ValueError, match="explicit label"):
        BenchmarkDevice(device="cpu", label="")


def test_local_cpu_is_labeled():
    d = BenchmarkDevice.local_cpu()
    assert d.device == "cpu" and d.label


def test_runtime_provenance_records_the_compute_device():
    info = describe_runtime()
    assert {"python", "platform", "machine", "compute_device"} <= info.keys()
