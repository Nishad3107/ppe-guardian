#!/usr/bin/env python3
"""Validate PPE dataset, print class distribution, and generate datasets/ppe/data.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

from ppe_pipeline_common import (
    assert_dataset_quality,
    print_distribution,
    print_json,
    validate_ppe_dataset,
    write_data_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PPE dataset and create YOLO data.yaml")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/ppe_final"),
        help="Root dataset directory (default: datasets/ppe_final)",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=20.0,
        help="Abort if max/min class count ratio exceeds this threshold (default: 20.0)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Optional output path for data.yaml (default: <dataset-root>/data.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = validate_ppe_dataset(args.dataset_root, imbalance_threshold=args.imbalance_threshold)
    print_distribution(summary)

    data_yaml = write_data_yaml(args.dataset_root, output_path=args.data_yaml)
    print(f"Generated data.yaml: {data_yaml}")
    print_json("Validation summary:", summary)
    assert_dataset_quality(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
