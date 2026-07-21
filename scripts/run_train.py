"""Run a training config.

uv run python scripts/run_train.py configs/train_v1_local.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundcontrol.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--push", action="store_true", help="push to the Hub when done")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    summary = train(config, push_to_hub=args.push)

    out = Path(summary["output_dir"]) / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
