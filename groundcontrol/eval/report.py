"""Render results as a leaderboard.

The table ranks on quality *and* cost, because those are the two axes a deployment
decision actually turns on. Rows are sorted by balanced accuracy, with the cost columns
sitting alongside rather than in a separate table, so a scorer that wins on accuracy and
loses on latency cannot be read as a straightforward win.

The report also flags two ways a row can be honest about its inputs and still mislead.
*Contamination*: a scorer trained on a corpus it is being evaluated against is not
comparable to a zero-shot entry. *In-sample calibration*: a scorer evaluated on the split
that fitted its temperature and picked its checkpoint is reporting a fit, not a
measurement, and its ECE and gate columns will flatter it. Both belong next to the number
rather than in a caveat someone has to remember.
"""

from __future__ import annotations

import json
from pathlib import Path

from groundcontrol.eval.runner import RunResult


def contamination_warnings(results: list[RunResult]) -> list[str]:
    """Flag results whose scorer trained on a corpus present in the evaluation data.

    The corpora are those the run actually loaded, reported by the examples themselves,
    rather than a lookup table kept in sync by hand. That matters for aggregated
    benchmarks: AggreFact redistributes RAGTruth, so a RAGTruth-trained scorer is not
    comparable to the zero-shot entries beside it, and nothing here needs to know that
    in advance to say so.
    """
    warnings: list[str] = []
    for r in results:
        trained_on = {c.lower() for c in r.notes.get("corpora_the_scorer_trained_on", ())}
        present = {c.lower() for c in r.notes.get("upstream_corpora_inside_this_eval_set", ())}
        overlap = trained_on & present

        if overlap:
            warnings.append(
                f"`{r.scorer}` trained on {sorted(overlap)}, which `{r.dataset}` "
                f"contains, so this result is in-domain and not comparable to a "
                f"zero-shot entry."
            )

        removed = r.notes.get("decontamination", {}).get("n_removed")
        if removed:
            warnings.append(
                f"`{r.scorer}` on `{r.dataset}`: {removed} training examples appeared "
                f"in this evaluation set and were removed before training."
            )
    return warnings


def in_sample_warnings(results: list[RunResult]) -> list[str]:
    """Flag rows evaluated on the split that fitted the scorer.

    Three things were chosen on that split: the temperature, the best checkpoint, and
    the early-stopping point. ECE is the column this ruins outright — a temperature
    fitted to minimize NLL on a set will look calibrated on that set by construction —
    and the gate columns inherit it, since risk-coverage ranks on the same scores. The
    warning names the column rather than the row, so nobody reads it as a general
    disclaimer and keeps quoting the number.
    """
    warnings: list[str] = []
    for r in results:
        fitted_on = r.notes.get("calibration_fitted_on")
        if fitted_on and str(fitted_on).lower() == r.dataset.lower():
            warnings.append(
                f"`{r.scorer}` on `{r.dataset}`: temperature was fitted and the "
                f"checkpoint selected on this split, so **ECE ({r.metrics.ece:.3f}) is "
                f"in-sample**, as are the gate columns. Read the held-out row for a "
                f"calibration number, not this one."
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

    in_sample = in_sample_warnings(results)
    if in_sample:
        lines += ["", "## In-sample calibration", ""] + [f"- {w}" for w in in_sample]

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

    gated = [r for r in results if r.gate]
    if gated:
        lines += [
            "",
            "## As a gate",
            "",
            "Auto-accept the answers a scorer is most confident are grounded, review the",
            "rest. **Risk** is how many ungrounded answers still reach a user at that",
            "coverage. The row to beat is *no information*: a scorer whose confidence",
            "means nothing holds risk flat at the base rate, making the gate equivalent",
            "to reviewing a random sample.",
            "",
            "| scorer | dataset | base rate | risk @50% | risk @80% | risk @100% "
            "| max coverage under 1% risk | lift |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in gated:
            g = r.gate
            lines.append(
                f"| {r.scorer} | {r.dataset} | {g.base_rate:.3f} "
                f"| {g.risk_at_coverage(0.5):.3f} | {g.risk_at_coverage(0.8):.3f} "
                f"| {g.risk_at_coverage(1.0):.3f} | {g.coverage_at_risk(0.01):.1%} "
                f"| {g.lift_over_random:.2f} |"
            )
        lines += [
            "",
            "**lift** places the scorer between no-information (0.00) and a perfect",
            "ranking (1.00). Normalized against what is achievable rather than against",
            "zero, since past the grounded fraction even flawless ordering must start",
            "admitting ungrounded answers.",
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


def write(
    results: list[RunResult],
    out_dir: Path,
    stem: str = "leaderboard",
    title: str | None = None,
) -> dict[str, Path]:
    """Write markdown for humans and JSON for regression tracking.

    `title` defaults to the file stem so the written report is headed by the run that
    produced it, rather than by a generic word that makes two runs indistinguishable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"

    md_path.write_text(to_markdown(results, title=title or stem), encoding="utf-8")
    json_path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2, default=str), encoding="utf-8"
    )
    return {"markdown": md_path, "json": json_path}
