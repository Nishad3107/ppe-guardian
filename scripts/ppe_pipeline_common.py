#!/usr/bin/env python3
"""Shared utilities for PPE dataset validation, training, and model verification."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import cv2

PPE_CLASS_NAMES = [
    "helmet",
    "mask",
    "glasses",
    "boots",
]
EXPECTED_CLASS_IDS = set(range(len(PPE_CLASS_NAMES)))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def normalize_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def validate_dataset_structure(dataset_root: Path) -> None:
    required_dirs = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [str(path) for path in required_dirs if not path.exists() or not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Dataset structure invalid. Missing required directories: "
            + ", ".join(missing)
        )


def _find_image_for_label(images_dir: Path, label_file: Path) -> Optional[Path]:
    stem = label_file.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_label_file(path: Path) -> list[int]:
    class_ids: list[int] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line_no, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            raise RuntimeError(f"Invalid label format in {path}:{line_no}. Expected YOLO row with >=5 values.")
        try:
            class_id = int(float(parts[0]))
        except ValueError as exc:
            raise RuntimeError(f"Invalid class id in {path}:{line_no}: {parts[0]}") from exc
        if class_id not in EXPECTED_CLASS_IDS:
            raise RuntimeError(
                f"Unexpected class id {class_id} in {path}:{line_no}. "
                f"Expected ids: {sorted(EXPECTED_CLASS_IDS)}"
            )
        class_ids.append(class_id)
    return class_ids


def validate_ppe_dataset(dataset_root: Path, imbalance_threshold: float) -> dict:
    validate_dataset_structure(dataset_root)

    total_counts: Counter[int] = Counter()
    split_counts: dict[str, Counter[int]] = {"train": Counter(), "val": Counter()}
    label_files_by_split: dict[str, int] = {}
    image_files_by_split: dict[str, int] = {}
    missing_images: list[str] = []

    for split in ("train", "val"):
        labels_dir = dataset_root / "labels" / split
        images_dir = dataset_root / "images" / split
        label_files = sorted(labels_dir.rglob("*.txt"))
        image_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

        if not label_files:
            raise RuntimeError(f"No label files found in {labels_dir}")
        if not image_files:
            raise RuntimeError(f"No image files found in {images_dir}")

        label_files_by_split[split] = len(label_files)
        image_files_by_split[split] = len(image_files)

        for label_path in label_files:
            if _find_image_for_label(images_dir, label_path) is None:
                missing_images.append(str(label_path))
            class_ids = _parse_label_file(label_path)
            for class_id in class_ids:
                total_counts[class_id] += 1
                split_counts[split][class_id] += 1

    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise RuntimeError(f"Missing matching image files for labels: {preview}")

    missing_classes = [name for idx, name in enumerate(PPE_CLASS_NAMES) if total_counts[idx] == 0]

    non_zero_counts = [total_counts[idx] for idx in range(len(PPE_CLASS_NAMES)) if total_counts[idx] > 0]
    if not non_zero_counts:
        raise RuntimeError("No labeled objects found in dataset labels.")
    min_count = min(non_zero_counts)
    max_count = max(non_zero_counts)
    imbalance_ratio = max_count / max(min_count, 1)

    return {
        "dataset_root": str(dataset_root),
        "label_files_by_split": label_files_by_split,
        "image_files_by_split": image_files_by_split,
        "class_distribution": {PPE_CLASS_NAMES[idx]: int(total_counts[idx]) for idx in range(len(PPE_CLASS_NAMES))},
        "train_distribution": {PPE_CLASS_NAMES[idx]: int(split_counts["train"][idx]) for idx in range(len(PPE_CLASS_NAMES))},
        "val_distribution": {PPE_CLASS_NAMES[idx]: int(split_counts["val"][idx]) for idx in range(len(PPE_CLASS_NAMES))},
        "missing_classes": missing_classes,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_threshold": imbalance_threshold,
        "imbalance_severe": imbalance_ratio > imbalance_threshold,
    }


def write_data_yaml(dataset_root: Path, output_path: Optional[Path] = None) -> Path:
    if output_path is None:
        output_path = dataset_root / "data.yaml"

    names_rows = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(PPE_CLASS_NAMES))
    content = (
        f"path: {dataset_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(PPE_CLASS_NAMES)}\n"
        "names:\n"
        f"{names_rows}\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def print_distribution(summary: dict) -> None:
    print("Dataset class distribution:")
    for class_name in PPE_CLASS_NAMES:
        total = summary["class_distribution"][class_name]
        train_count = summary["train_distribution"][class_name]
        val_count = summary["val_distribution"][class_name]
        print(f"  {class_name:12s} total={total:6d} train={train_count:6d} val={val_count:6d}")
    print(f"Class imbalance ratio (max/min): {summary['imbalance_ratio']:.2f}")


def assert_dataset_quality(summary: dict) -> None:
    missing_classes = summary.get("missing_classes", [])
    if missing_classes:
        raise RuntimeError(
            "Dataset class coverage invalid. Missing classes: "
            + ", ".join(missing_classes)
        )
    if summary.get("imbalance_severe"):
        raise RuntimeError(
            "Severe class imbalance detected. "
            f"max/min ratio={summary['imbalance_ratio']:.2f}, "
            f"threshold={summary['imbalance_threshold']:.2f}"
        )


def load_model_names(model_path: Path) -> dict[int, str]:
    from ultralytics import YOLO

    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    model = YOLO(str(model_path))
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise RuntimeError(f"Unexpected model.names format for {model_path}: {type(names)}")


def validate_model_names(names: dict[int, str]) -> dict:
    actual = {normalize_name(name) for _, name in sorted(names.items())}
    expected = {normalize_name(name) for name in PPE_CLASS_NAMES}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "ok": len(missing) == 0,
        "expected": sorted(expected),
        "actual": sorted(actual),
        "missing": missing,
        "unexpected": unexpected,
    }


def deploy_model(source_model: Path, deploy_model: Path) -> None:
    deploy_model.parent.mkdir(parents=True, exist_ok=True)
    temp_path = deploy_model.with_suffix(deploy_model.suffix + ".tmp")
    shutil.copy2(source_model, temp_path)
    temp_path.replace(deploy_model)


def get_sample_frame(sample_image: Optional[Path], sample_video: Path) -> tuple[Any, str]:
    if sample_image is not None:
        image = cv2.imread(str(sample_image))
        if image is None:
            raise RuntimeError(f"Could not read sample image: {sample_image}")
        return image, f"image:{sample_image}"

    if not sample_video.exists():
        raise RuntimeError(f"Sample video not found: {sample_video}")
    cap = cv2.VideoCapture(str(sample_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open sample video: {sample_video}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from sample video: {sample_video}")
    return frame, f"video:{sample_video}"


def run_inference(model_path: Path, frame, conf: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    result = model(frame, conf=conf, verbose=False)[0]
    names = load_model_names(model_path)

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"class_{class_id}")
            score = float(box.conf[0]) if box.conf is not None else 0.0
            detections.append({"class_id": class_id, "class_name": class_name, "confidence": score})

    class_counter = Counter(det["class_name"] for det in detections)
    return {
        "num_detections": len(detections),
        "class_counts": dict(class_counter),
        "detections": detections,
    }


def print_json(title: str, payload: dict) -> None:
    print(title)
    print(json.dumps(payload, indent=2))
