#!/usr/bin/env python3
"""Controlled remap+merge pipeline for PPE datasets."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ppe_pipeline_common import PPE_CLASS_NAMES, assert_dataset_quality, validate_ppe_dataset, write_data_yaml

TARGET_ID_BY_CLASS = {name: idx for idx, name in enumerate(PPE_CLASS_NAMES)}

# Required semantic remap table from specification.
SEMANTIC_REMAP_TABLE = {
    "helmet_dataset": {"helmet": 0},
    "mask_dataset": {"mask": 1},
    "glasses_dataset": {"glasses": 2},
    "boots_dataset": {"boots": 3},
}

# Source roots and source-id mappings discovered from current assets.
# Unknown/untrusted IDs are intentionally ignored and logged for manual review.
SOURCE_CONFIG = {
    "helmet_dataset": {
        "root": Path("datasets/helmet_dataset"),
        "split_candidates": ("train",),
        "source_to_target": {0: TARGET_ID_BY_CLASS["helmet"]},
        "ignored_source_ids": set(),
    },
    "mask_dataset": {
        "root": Path("datasets/mask/mask-detection"),
        "split_candidates": ("train", "valid"),
        "source_to_target": {
            0: TARGET_ID_BY_CLASS["mask"],
        },
        "ignored_source_ids": {1},
    },
    "glasses_dataset": {
        "root": Path("datasets/glasses"),
        "split_candidates": ("train", "valid"),
        "source_to_target": {
            0: TARGET_ID_BY_CLASS["glasses"],
        },
        "ignored_source_ids": {1, 2, 3},
    },
    "boots_dataset": {
        "root": Path("datasets/boots/datasets/boots/boots-detection"),
        "split_candidates": ("train", "valid"),
        "source_to_target": {
            0: TARGET_ID_BY_CLASS["boots"],
            3: TARGET_ID_BY_CLASS["boots"],      # safety_shoe alias
        },
        "ignored_source_ids": {1, 2},
    },
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _split_paths(root: Path, split: str) -> tuple[Path, Path] | tuple[None, None]:
    first = (root / "images" / split, root / "labels" / split)
    second = (root / split / "images", root / split / "labels")
    if first[0].exists() and first[1].exists():
        return first
    if second[0].exists() and second[1].exists():
        return second
    return (None, None)


def _find_image(images_dir: Path, label_path: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _hashed_name(dataset_key: str, split: str, source_stem: str) -> str:
    digest = hashlib.sha1(f"{dataset_key}:{split}:{source_stem}".encode("utf-8")).hexdigest()[:12]
    clean_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_stem).strip("_")
    return f"{dataset_key}_{clean_stem}_{digest}"


def _remap_label_file(
    source_label: Path,
    dest_label: Path,
    source_to_target: dict[int, int],
    ignored_source_ids: set[int],
    unknown_counter: Counter[int],
    target_counter: Counter[int],
) -> bool:
    wrote_any = False
    remapped_rows: list[str] = []
    for line_no, raw in enumerate(source_label.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 5:
            raise RuntimeError(f"Malformed YOLO label row in {source_label}:{line_no}")
        try:
            source_id = int(float(parts[0]))
        except ValueError as exc:
            raise RuntimeError(f"Invalid class id in {source_label}:{line_no}: {parts[0]}") from exc

        if source_id in ignored_source_ids:
            continue
        if source_id not in source_to_target:
            unknown_counter[source_id] += 1
            continue

        target_id = source_to_target[source_id]
        target_counter[target_id] += 1
        remapped_rows.append(" ".join([str(target_id)] + parts[1:]))
        wrote_any = True

    if wrote_any:
        dest_label.write_text("\n".join(remapped_rows) + "\n", encoding="utf-8")
    return wrote_any


def _class_image_counts(labels_root: Path) -> Counter[int]:
    image_counts: Counter[int] = Counter()
    label_files = list(labels_root.rglob("*.txt"))
    for label_path in label_files:
        classes_in_image = set()
        for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = raw.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) < 1:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            classes_in_image.add(class_id)
        for class_id in classes_in_image:
            image_counts[class_id] += 1
    return image_counts


def _materialize_image(src: Path, dst: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError as exc:
            # Cross-device links can be safely copied. Space errors should bubble up.
            if exc.errno == errno.EXDEV:
                shutil.copy2(src, dst)
                return
            raise
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def _augmentation_targets(label_counts: Counter[int], imbalance_threshold: float) -> list[dict[str, object]]:
    non_zero = [count for count in label_counts.values() if count > 0]
    if not non_zero:
        return []
    max_count = max(non_zero)
    targets = []
    for class_id, class_name in enumerate(PPE_CLASS_NAMES):
        count = label_counts[class_id]
        if count == 0:
            targets.append({"class": class_name, "reason": "missing", "suggestion": "collect+annotate new samples"})
            continue
        ratio = max_count / count
        if ratio >= imbalance_threshold:
            targets.append(
                {
                    "class": class_name,
                    "reason": f"underrepresented (max/class={ratio:.2f})",
                    "suggestion": "targeted augmentation + data collection",
                }
            )
    return targets


def merge_ppe_dataset(
    output_root: Path,
    imbalance_threshold: float,
    force_replace: bool,
    image_materialization: str,
    max_files_per_dataset_split: int,
) -> dict:
    temp_root = output_root.parent / f"{output_root.name}_tmp_merge"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    (temp_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (temp_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (temp_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (temp_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    summary = {
        "semantic_remap_table": SEMANTIC_REMAP_TABLE,
        "datasets": {},
        "unknown_source_ids": {},
        "class_distribution": {},
        "image_distribution": {},
        "warnings": [],
        "augmentation_targets": [],
        "ready_for_training": False,
        "temp_root": str(temp_root),
    }

    label_counter: Counter[int] = Counter()
    unknown_counter_all: defaultdict[str, Counter[int]] = defaultdict(Counter)

    for dataset_key, cfg in SOURCE_CONFIG.items():
        root = cfg["root"]
        source_to_target = cfg["source_to_target"]
        ignored_ids = cfg["ignored_source_ids"]
        split_candidates = cfg["split_candidates"]

        dataset_report = {
            "root": str(root),
            "exists": root.exists(),
            "processed_label_files": 0,
            "copied_images": 0,
            "dropped_empty_labels": 0,
            "splits_used": [],
        }
        if not root.exists():
            summary["warnings"].append(f"{dataset_key}: source root missing: {root}")
            summary["datasets"][dataset_key] = dataset_report
            continue

        for source_split in split_candidates:
            images_dir, labels_dir = _split_paths(root, source_split)
            if images_dir is None or labels_dir is None:
                continue
            target_split = "train" if source_split == "train" else "val"
            dataset_report["splits_used"].append({"source_split": source_split, "target_split": target_split})
            selected_in_split = 0

            for source_label in sorted(labels_dir.rglob("*.txt")):
                if max_files_per_dataset_split > 0 and selected_in_split >= max_files_per_dataset_split:
                    break
                source_image = _find_image(images_dir, source_label)
                if source_image is None:
                    summary["warnings"].append(f"{dataset_key}: missing image for {source_label}")
                    continue

                merged_stem = _hashed_name(dataset_key, target_split, source_label.stem)
                dest_label = temp_root / "labels" / target_split / f"{merged_stem}.txt"
                dest_image = temp_root / "images" / target_split / f"{merged_stem}{source_image.suffix.lower()}"

                wrote_label = _remap_label_file(
                    source_label=source_label,
                    dest_label=dest_label,
                    source_to_target=source_to_target,
                    ignored_source_ids=ignored_ids,
                    unknown_counter=unknown_counter_all[dataset_key],
                    target_counter=label_counter,
                )
                dataset_report["processed_label_files"] += 1
                if not wrote_label:
                    dataset_report["dropped_empty_labels"] += 1
                    continue

                _materialize_image(source_image, dest_image, image_materialization)
                dataset_report["copied_images"] += 1
                selected_in_split += 1

        summary["datasets"][dataset_key] = dataset_report

    for dataset_key, unknown_counter in unknown_counter_all.items():
        if unknown_counter:
            summary["unknown_source_ids"][dataset_key] = {str(k): int(v) for k, v in sorted(unknown_counter.items())}
            summary["warnings"].append(
                f"{dataset_key}: encountered unknown source IDs {sorted(unknown_counter)}; rows dropped"
            )

    image_counter = _class_image_counts(temp_root / "labels")
    summary["class_distribution"] = {name: int(label_counter[idx]) for idx, name in enumerate(PPE_CLASS_NAMES)}
    summary["image_distribution"] = {name: int(image_counter[idx]) for idx, name in enumerate(PPE_CLASS_NAMES)}

    non_zero = [label_counter[idx] for idx in range(len(PPE_CLASS_NAMES)) if label_counter[idx] > 0]
    imbalance_ratio = (max(non_zero) / min(non_zero)) if non_zero else 0.0
    summary["imbalance_ratio"] = imbalance_ratio
    summary["imbalance_threshold"] = imbalance_threshold
    summary["severe_imbalance"] = bool(non_zero and imbalance_ratio > imbalance_threshold)
    summary["augmentation_targets"] = _augmentation_targets(label_counter, imbalance_threshold)

    write_data_yaml(temp_root)
    quality_error = None
    try:
        merged_quality = validate_ppe_dataset(temp_root, imbalance_threshold=imbalance_threshold)
        assert_dataset_quality(merged_quality)
        summary["ready_for_training"] = True
    except Exception as exc:
        quality_error = str(exc)
        summary["ready_for_training"] = False
        summary["warnings"].append(f"final_validation_failed: {quality_error}")

    report_path = output_root.parent / "ppe_merge_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report_path"] = str(report_path)

    if summary["ready_for_training"]:
        if output_root.exists() and force_replace:
            shutil.rmtree(output_root)
        if output_root.exists() and not force_replace:
            raise RuntimeError(
                f"{output_root} already exists. Re-run with --force-replace to replace it."
            )
        temp_root.replace(output_root)
        write_data_yaml(output_root)
    return summary


def print_balance_table(summary: dict) -> None:
    print("Per-class distribution:")
    for class_name in PPE_CLASS_NAMES:
        labels = summary["class_distribution"].get(class_name, 0)
        images = summary["image_distribution"].get(class_name, 0)
        print(f"  {class_name:12s} images={images:6d} labels={labels:6d}")
    print(f"Imbalance ratio (max/min labels): {summary.get('imbalance_ratio', 0.0):.2f}")
    if summary.get("severe_imbalance"):
        print("WARNING: Severe imbalance detected.")
    if summary.get("augmentation_targets"):
        print("Suggested augmentation targets:")
        for item in summary["augmentation_targets"]:
            print(f"  - {item['class']}: {item['reason']} -> {item['suggestion']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled PPE dataset merge with safe class remapping")
    parser.add_argument("--output-root", type=Path, default=Path("datasets/ppe_final"))
    parser.add_argument("--imbalance-threshold", type=float, default=20.0)
    parser.add_argument("--force-replace", action="store_true")
    parser.add_argument(
        "--image-materialization",
        choices=("hardlink", "copy", "symlink"),
        default="symlink",
        help="How to materialize merged images (default: symlink).",
    )
    parser.add_argument(
        "--max-files-per-dataset-split",
        type=int,
        default=0,
        help="Cap selected samples per source dataset split to control storage (default: 0 = unlimited).",
    )
    args = parser.parse_args()

    summary = merge_ppe_dataset(
        output_root=args.output_root,
        imbalance_threshold=args.imbalance_threshold,
        force_replace=args.force_replace,
        image_materialization=args.image_materialization,
        max_files_per_dataset_split=args.max_files_per_dataset_split,
    )
    print_balance_table(summary)
    print(f"Merge report: {summary.get('report_path')}")
    print(json.dumps(summary, indent=2))

    if not summary["ready_for_training"]:
        print("NOT READY FOR TRAINING: merge completed to temp dataset but validation failed.")
        return 1
    print("READY FOR TRAINING: merged dataset validated and promoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
