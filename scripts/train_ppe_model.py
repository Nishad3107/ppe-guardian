#!/usr/bin/env python3
"""Create reproducible PPE training pipeline and optionally execute training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ppe_pipeline_common import (
    assert_dataset_quality,
    print_distribution,
    validate_ppe_dataset,
    write_data_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset and run YOLO PPE training")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/ppe_final"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imbalance-threshold", type=float, default=20.0)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute training command. Without --run, only prints the exact command.",
    )
    return parser.parse_args()


def _format_multiline_command(command: list[str]) -> str:
    head = " ".join(command[:3])
    tail = command[3:]
    if not tail:
        return head
    formatted_tail = " \\\n    ".join(tail)
    return f"{head} \\\n    {formatted_tail}"


def main() -> int:
    args = parse_args()
    summary = validate_ppe_dataset(args.dataset_root, imbalance_threshold=args.imbalance_threshold)
    print_distribution(summary)

    data_yaml = write_data_yaml(args.dataset_root)
    print(f"Using data config: {data_yaml}")

    python_dir = Path(sys.executable).parent
    yolo_candidate = python_dir / "yolo"
    if not yolo_candidate.exists():
        yolo_candidate = python_dir / "yolo.exe"
    yolo_cmd = str(yolo_candidate) if yolo_candidate.exists() else "yolo"

    command = [
        yolo_cmd,
        "detect",
        "train",
        f"model={args.model}",
        f"data={data_yaml.as_posix()}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
    ]

    print("\nTraining command:")
    print(_format_multiline_command(command))
    assert_dataset_quality(summary)

    if not args.run:
        print("\nDry-run mode. Re-run with --run to start training.")
        return 0

    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
