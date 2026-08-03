"""Command-line entry point for the toxic-comment baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a responsible toxic-comment TF-IDF baseline."
    )
    parser.add_argument("dataset", help="Path to a CSV, XLSX, XLS, or JSON dataset.")
    parser.add_argument("--output-dir", default="artifacts/run")
    parser.add_argument("--text-column", default="comment_text")
    parser.add_argument("--target-column", default="toxic")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_experiment(
        args.dataset,
        args.output_dir,
        text_column=args.text_column,
        target_column=args.target_column,
        test_size=args.test_size,
        max_features=args.max_features,
        random_state=args.random_state,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
