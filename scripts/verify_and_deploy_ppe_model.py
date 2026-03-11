#!/usr/bin/env python3
"""Verify trained PPE model classes, deploy to models/ppe.pt, and run inference smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from ppe_pipeline_common import (
    deploy_model,
    get_sample_frame,
    load_model_names,
    print_json,
    run_inference,
    validate_model_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify best.pt and deploy as models/ppe.pt")
    parser.add_argument(
        "--best-model",
        type=Path,
        default=Path("runs/detect/train/weights/best.pt"),
        help="Path to trained best.pt",
    )
    parser.add_argument(
        "--deploy-model",
        type=Path,
        default=Path("models/ppe.pt"),
        help="Deployment target path",
    )
    parser.add_argument(
        "--sample-image",
        type=Path,
        default=None,
        help="Optional image path for inference smoke test",
    )
    parser.add_argument(
        "--sample-video",
        type=Path,
        default=Path("videos/test-video.mp4"),
        help="Fallback video for sample frame extraction",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Inference confidence threshold")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    best_names = load_model_names(args.best_model)
    print_json("best.pt model.names:", {str(k): v for k, v in best_names.items()})

    integrity = validate_model_names(best_names)
    print_json("best.pt class integrity:", integrity)
    if not integrity["ok"]:
        raise RuntimeError(
            "Trained model missing required classes: "
            + ", ".join(integrity["missing"])
        )

    deploy_model(args.best_model, args.deploy_model)
    print(f"Deployed model to: {args.deploy_model}")

    deployed_names = load_model_names(args.deploy_model)
    deployed_integrity = validate_model_names(deployed_names)
    print_json("deployed model class integrity:", deployed_integrity)
    if not deployed_integrity["ok"]:
        raise RuntimeError("Deployment verification failed: deployed model classes are invalid")

    frame, source = get_sample_frame(sample_image=args.sample_image, sample_video=args.sample_video)
    inference = run_inference(args.deploy_model, frame, conf=args.conf)
    print_json(f"Inference smoke test ({source}):", inference)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
