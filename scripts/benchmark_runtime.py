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
    parser = argparse.ArgumentParser(description="Benchmark PPE Guardian runtime performance on a video.")
    parser.add_argument(
        "--video",
        default=str(app.RECORDED_VIDEO_PATH),
        help="Video path to benchmark.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup frames to discard from summary stats.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")

    job = app._create_job("benchmark", "BENCHMARK", str(video_path))
    job.source = "BENCHMARK"
    job.source_path = str(video_path)
    job.source_ready = True

    results: list[dict[str, float | int | bool]] = []
    processed = 0
    try:
        while processed < max(args.frames, 1):
            ok, frame = capture.read()
            if not ok:
                break
            result = app._process_frame(job, frame, record_violations=False)
            results.append(
                {
                    "fps": float(result["fps"]),
                    "person_inference_ms": float(result["person_inference_ms"]),
                    "ppe_inference_ms": float(result["ppe_inference_ms"]),
                    "people_detected": int(result["people_detected"]),
                    "violating_tracks": int(result["violating_tracks"]),
                    "ppe_inference_ok": bool(result["ppe_inference_ok"]),
                }
            )
            processed += 1
    finally:
        capture.release()

    warmup = min(max(args.warmup, 0), len(results))
    sample = results[warmup:] if warmup < len(results) else results
    if not sample:
        raise SystemExit("No frames were processed.")

    summary = {
        "video": str(video_path),
        "frames_processed": len(results),
        "warmup_frames": warmup,
        "model_integrity_ok": app.MODEL_INTEGRITY_STATUS["ok"],
        "average_fps": round(sum(item["fps"] for item in sample) / len(sample), 2),
        "average_person_inference_ms": round(
            sum(item["person_inference_ms"] for item in sample) / len(sample), 2
        ),
        "average_ppe_inference_ms": round(
            sum(item["ppe_inference_ms"] for item in sample) / len(sample), 2
        ),
        "peak_people_detected": max(item["people_detected"] for item in sample),
        "frames_with_violations": sum(1 for item in sample if item["violating_tracks"] > 0),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
