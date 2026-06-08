#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

import cv2

MPL_CONFIG_DIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run regression evaluation against expected PPE Guardian outputs.")
    parser.add_argument("--video", required=True, help="Video path to evaluate.")
    parser.add_argument("--expected", required=True, help="Expected output JSON path.")
    return parser


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video)
    expected_path = Path(args.expected)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")
    if not expected_path.exists():
        raise SystemExit(f"Expected file not found: {expected_path}")

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    frame_expectations = {int(item["frame"]): item for item in expected.get("frames", [])}
    if not frame_expectations:
        raise SystemExit("Expected file does not define any frames.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")

    job = app._create_job("regression", "REGRESSION", str(video_path))
    job.source = "REGRESSION"
    job.source_path = str(video_path)
    job.source_ready = True

    sorted_frames = sorted(frame_expectations)
    max_frame = sorted_frames[-1]
    current_frame = 0
    predictions: dict[int, dict[str, int | float]] = {}
    try:
        while current_frame < max_frame:
            ok, frame = capture.read()
            if not ok:
                break
            current_frame += 1
            result = app._process_frame(job, frame, record_violations=False)
            if current_frame in frame_expectations:
                predictions[current_frame] = {
                    "violating_tracks": int(result["violating_tracks"]),
                    "people_detected": int(result["people_detected"]),
                }
    finally:
        capture.release()

    tp = 0
    fp = 0
    fn = 0
    details: list[dict[str, int | bool]] = []
    for frame_number in sorted_frames:
        expected_frame = frame_expectations[frame_number]
        predicted = predictions.get(frame_number, {"violating_tracks": 0, "people_detected": 0})
        expected_violations = int(expected_frame.get("violating_tracks", 0))
        predicted_violations = int(predicted["violating_tracks"])
        tp += min(expected_violations, predicted_violations)
        fp += max(predicted_violations - expected_violations, 0)
        fn += max(expected_violations - predicted_violations, 0)
        details.append(
            {
                "frame": frame_number,
                "expected_violating_tracks": expected_violations,
                "predicted_violating_tracks": predicted_violations,
                "match": expected_violations == predicted_violations,
            }
        )

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    summary = {
        "video": str(video_path),
        "expected": str(expected_path),
        "model_integrity_ok": app.MODEL_INTEGRITY_STATUS["ok"],
        "frames_evaluated": len(sorted_frames),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "frames": details,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
