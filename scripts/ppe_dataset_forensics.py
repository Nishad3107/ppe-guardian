#!/usr/bin/env python3
"""Forensic scan of PPE source datasets and mapping consistency."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    root: Path
    expected_semantics: tuple[str, ...]


DATASET_SPECS = [
    DatasetSpec("helmet_dataset", Path("datasets/helmet_dataset"), ("helmet", "no_helmet")),
    DatasetSpec("mask_dataset", Path("datasets/mask/mask-detection"), ("mask", "no_mask")),
    DatasetSpec("glasses_dataset", Path("datasets/glasses"), ("glasses", "no_glasses")),
    DatasetSpec("boots_dataset", Path("datasets/boots/datasets/boots/boots-detection"), ("boots", "no_boots")),
]

LABEL_SPLITS = ("train", "val", "valid", "test")
NEGATIVE_TOKENS = ("no_", "without", "not_", "not-", "missing", "absent")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _parse_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        return {}
    text = data_yaml.read_text(encoding="utf-8", errors="ignore")

    list_match = re.search(r"names:\s*(\[[^\]]*\])", text, re.S)
    if list_match:
        try:
            values = ast.literal_eval(list_match.group(1))
            if isinstance(values, list):
                return {idx: str(name) for idx, name in enumerate(values)}
        except (SyntaxError, ValueError):
            pass

    result: dict[int, str] = {}
    in_names = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if not in_names:
            continue
        if not stripped:
            continue
        match = re.match(r"^(\d+)\s*:\s*(.+)$", stripped)
        if match:
            result[int(match.group(1))] = match.group(2).strip().strip("'\"")
            continue
        if not raw.startswith(" "):
            break
    return result


def _collect_split_paths(root: Path) -> dict[str, dict[str, Path]]:
    split_paths: dict[str, dict[str, Path]] = {}
    for split in LABEL_SPLITS:
        options = [
            {"images": root / "images" / split, "labels": root / "labels" / split},
            {"images": root / split / "images", "labels": root / split / "labels"},
        ]
        for candidate in options:
            if candidate["images"].exists() and candidate["labels"].exists():
                split_paths[split] = candidate
                break
    return split_paths


def _label_files_for_dataset(root: Path) -> list[Path]:
    label_files: list[Path] = []
    for labels_root in root.rglob("labels"):
        if labels_root.is_dir():
            label_files.extend(sorted(labels_root.rglob("*.txt")))
    return sorted(set(label_files))


def _scan_label_ids(label_files: list[Path]) -> tuple[Counter[int], list[str]]:
    counts: Counter[int] = Counter()
    errors: list[str] = []
    for path in label_files:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            text = raw.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) < 5:
                errors.append(f"{path}:{line_no}: malformed row")
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                errors.append(f"{path}:{line_no}: invalid class id '{parts[0]}'")
                continue
            counts[class_id] += 1
    return counts, errors


def _detect_negative_classes(name_map: dict[int, str]) -> list[str]:
    negatives = []
    for _, value in sorted(name_map.items()):
        norm = _normalize(value)
        if any(tok in norm for tok in NEGATIVE_TOKENS):
            negatives.append(value)
    return negatives


def run_forensics() -> dict:
    report = {"datasets": {}, "collisions": {}, "merge_risks": {}}
    id_name_usage: defaultdict[int, set[str]] = defaultdict(set)
    base_name_to_dataset: defaultdict[str, set[str]] = defaultdict(set)

    for spec in DATASET_SPECS:
        dataset_info: dict[str, object] = {
            "root": str(spec.root),
            "exists": spec.root.exists(),
            "expected_semantics": list(spec.expected_semantics),
        }
        if not spec.root.exists():
            report["datasets"][spec.key] = dataset_info
            continue

        names_map = _parse_names(spec.root / "data.yaml")
        label_files = _label_files_for_dataset(spec.root)
        class_counts, errors = _scan_label_ids(label_files)
        split_paths = _collect_split_paths(spec.root)

        split_summary = {}
        for split, paths in split_paths.items():
            img_count = sum(1 for p in paths["images"].rglob("*") if p.is_file())
            lbl_count = sum(1 for p in paths["labels"].rglob("*.txt"))
            split_summary[split] = {
                "images_dir": str(paths["images"]),
                "labels_dir": str(paths["labels"]),
                "image_files": img_count,
                "label_files": lbl_count,
            }

        for class_id, class_name in names_map.items():
            id_name_usage[class_id].add(_normalize(class_name))

        for label_path in label_files:
            base_name_to_dataset[label_path.name].add(spec.key)

        ids_without_mapping = sorted(class_id for class_id in class_counts if class_id not in names_map)
        mapped_but_absent = sorted(class_id for class_id in names_map if class_counts[class_id] == 0)

        dataset_info.update(
            {
                "data_yaml": str(spec.root / "data.yaml"),
                "class_id_to_name": {str(k): v for k, v in sorted(names_map.items())},
                "unique_class_ids_found": sorted(class_counts),
                "class_id_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
                "negative_classes_in_names": _detect_negative_classes(names_map),
                "split_summary": split_summary,
                "missing_val_split": not ("val" in split_paths or "valid" in split_paths),
                "label_parse_errors": errors[:50],
                "label_parse_error_count": len(errors),
                "ids_without_name_mapping": ids_without_mapping,
                "mapped_ids_with_zero_labels": mapped_but_absent,
            }
        )
        report["datasets"][spec.key] = dataset_info

    collisions = {}
    for class_id, names in sorted(id_name_usage.items()):
        if len(names) > 1:
            collisions[str(class_id)] = sorted(names)
    report["collisions"]["class_id_name_collisions"] = collisions

    overwrite_risks = {
        label_name: sorted(dataset_keys)
        for label_name, dataset_keys in base_name_to_dataset.items()
        if len(dataset_keys) > 1
    }
    report["merge_risks"]["label_filename_overwrite_risk_count"] = len(overwrite_risks)
    report["merge_risks"]["label_filename_overwrite_examples"] = dict(list(sorted(overwrite_risks.items())[:30]))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PPE dataset forensic scanner")
    parser.add_argument("--json-out", type=Path, default=Path("datasets/ppe_forensics_report.json"))
    args = parser.parse_args()

    report = run_forensics()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved forensic report: {args.json_out}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
