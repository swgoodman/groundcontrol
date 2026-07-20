"""Render results as a leaderboard.

The table ranks on quality *and* cost, because those are the two axes a deployment
decision actually turns on. Rows are sorted by balanced accuracy, with the cost columns
sitting alongside rather than in a separate table, so a scorer that wins on accuracy and
loses on latency cannot be read as a straightforward win.

The report also flags contamination: a scorer trained on a corpus it is being evaluated
against is not comparable to a zero-shot entry, and that fact belongs next to the number
rather than in a caveat someone has to remember.
"""

from __future__ import annotations

import json
from pathlib import Path

from groundcheck.eval.runner import RunResult

# Which evaluation datasets are compromised by which training corpora.
_CONTAMINATION = {
    "fever": {"fever"},
    "ragtruth": {"ragtruth"},
    "halueval": {"halueval"},
}


def contamination_warnings(results: list[RunResult]) -> list[str]:
    warnings: list[str] = []
    for r in results:
        trained_on = set(r.notes.get("training_corpora", ()))
        overlap = trained_on & _CONTAMINATION.get(r.dataset, set())
        if overlap:
            warnings.append(
                f"`{r.scorer}` was trained on {sorted(overlap)}, so its `{r.dataset}` "
                f"result is in-domain and not comparable to a zero-shot entry."
            )
    return warnings


def _fmt(value, spec: str = ".3f") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return format(value, spec)
    return str(value)


def to_markdown(results: list[RunResult], title: str = "Leaderboard") -> str:
    if not results:
        return f"# {title}\n\n_No results._\n"

    rows = sorted(
        (r.to_row() for r in results),
        key=lambda r: (r["balanced_acc"] != r["balanced_acc"], -r["balanced_acc"]),
    )

    header = (
        "| scorer | dataset | n | bal acc | F1 (not-sup) | PR-AUC | ECE "
        "| size MB | device | p50 ms | qps | $/1k |"
    )
    sep = "|" + "---|" * 12
    lines = [f"# {title}", "", header, sep]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["scorer"],
                    r["dataset"],
                    str(r["n"]),
                    _fmt(r["balanced_acc"]),
                    _fmt(r["f1_notsup"]),
                    _fmt(r["pr_auc_notsup"]),
                    _fmt(r["ece"]),
                    _fmt(r.get("size_mb"), ".0f"),
                    _fmt(r.get("device")),
                    _fmt(r.get("latency_ms_p50"), ".1f"),
                    _fmt(r.get("throughput_qps"), ".1f"),
                    _fmt(r.get("cost_per_1k_usd"), ".4f"),
                ]
            )
            + " |"
        )

    warnings = contamination_warnings(results)
    if warnings:
        lines += ["", "## Contamination", ""] + [f"- {w}" for w in warnings]

    lines += ["", "## What these columns mean", ""]
    lines += [
        "- **bal acc** — balanced accuracy, so the majority-supported class cannot",
        "  carry the score.",
        "- **F1 (not-sup)** — on the not-supported class, the rare and costly one. A",
        "  scorer that never flags anything scores zero here and still looks fine on",
        "  plain accuracy.",
        "- **ECE** — expected calibration error. Lower is better. A confident wrong",
        "  answer is worse than an uncertain one, and only this column shows it.",
        "- **p50 ms / $ per 1k** — measured on the stated device. Latency from one",
        "  machine says nothing about another, which is why the device is named.",
        "- **n/a** — undefined for this slice, most often a single-class split.",
    ]

    calibration = [r for r in results if r.metrics.reliability]
    if calibration:
        lines += ["", "## Calibration", ""]
        for r in calibration:
            lines += [
                f"### {r.scorer} on {r.dataset}",
                "",
                "| confidence | n | accuracy |",
                "|---|---|---|",
            ]
            for b in r.metrics.reliability:
                if b.count:
                    lines.append(f"| {b.lower:.1f}–{b.upper:.1f} | {b.count} | {b.accuracy:.3f} |")
            lines.append("")

    return "\n".join(lines) + "\n"


def write(results: list[RunResult], out_dir: Path, stem: str = "leaderboard") -> dict[str, Path]:
    """Write markdown for humans and JSON for regression tracking."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"

    md_path.write_text(to_markdown(results), encoding="utf-8")
    json_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str), encoding="utf-8"
    )
    return {"markdown": md_path, "json": json_path}
