import ast
import atexit
import datetime
import hashlib
import json
import logging
import os
import re
import signal
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for
from ultralytics import YOLO

app = Flask(__name__)


def _load_local_env_file(path: Path) -> None:
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        logging.getLogger("ppe_guardian").warning("Failed to read local env file %s: %s", path, exc)
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip("\"'"))

PPE_ITEMS = ("helmet", "mask", "glasses", "boots")
EXPECTED_PPE_CLASSES = (
    "helmet",
    "mask",
    "glasses",
    "boots",
)
EXPECTED_PPE_CLASS_TO_ITEM = {name: name for name in PPE_ITEMS}

LOGGER = logging.getLogger("ppe_guardian")
if not LOGGER.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

_load_local_env_file(Path(".env.local"))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PERSON_MODEL_PATH = Path(os.getenv("PERSON_MODEL_PATH", "yolov8n.pt"))
PPE_MODEL_PATH = Path(os.getenv("PPE_MODEL_PATH", "models/ppe.pt"))
PPE_FALLBACK_MODEL_PATH = Path(os.getenv("PPE_FALLBACK_MODEL_PATH", "yolov8n.pt"))
RECORDED_VIDEO_PATH = Path(os.getenv("RECORDED_VIDEO_PATH", "videos/test-video.mp4"))
VIOLATIONS_LOG_PATH = Path(os.getenv("VIOLATIONS_LOG_PATH", "violations.log"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/ppe_guardian.db"))
APP_ENV = os.getenv("APP_ENV", os.getenv("FLASK_ENV", "development")).strip().lower()
IS_PRODUCTION = APP_ENV == "production"
REQUIRE_VALID_PPE_MODEL = _env_bool("REQUIRE_VALID_PPE_MODEL", False)
AUTO_OPEN_SOURCE = os.getenv("AUTO_OPEN_SOURCE", "recorded").strip().lower()
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1").strip()
SERVER_PORT = _env_int("SERVER_PORT", 8000)
UPLOAD_RETENTION_DAYS = max(1, _env_int("UPLOAD_RETENTION_DAYS", 7))

PERSON_CONFIDENCE_THRESHOLD = _env_float("PERSON_CONFIDENCE_THRESHOLD", 0.45)
PPE_CONFIDENCE_THRESHOLD = _env_float("PPE_CONFIDENCE_THRESHOLD", 0.35)
PPE_ASSOCIATION_IOU_THRESHOLD = _env_float("PPE_ASSOCIATION_IOU_THRESHOLD", 0.30)
PPE_CONTAINMENT_THRESHOLD = _env_float("PPE_CONTAINMENT_THRESHOLD", 0.60)
PPE_ASSOCIATION_MARGIN = _env_float("PPE_ASSOCIATION_MARGIN", 0.05)
TARGET_MAX_FPS = _env_float("TARGET_MAX_FPS", 15.0)
FPS_OVERLAY_ENABLED = _env_bool("FPS_OVERLAY_ENABLED", False)

PPE_REFRESH_FRAMES = max(1, int(_env_float("PPE_REFRESH_FRAMES", 1)))
TEMPORAL_WINDOW_SIZE = max(1, _env_int("TEMPORAL_WINDOW_SIZE", 6))
TEMPORAL_CONFIRM_FRAMES = max(1, min(_env_int("TEMPORAL_CONFIRM_FRAMES", 3), TEMPORAL_WINDOW_SIZE))
VIOLATION_DEDUP_SECONDS = _env_float("VIOLATION_DEDUP_SECONDS", 5.0)
TRACK_STALE_FRAME_LIMIT = int(_env_float("TRACK_STALE_FRAME_LIMIT", 180))
VIOLATION_CACHE_TTL_SECONDS = _env_float("VIOLATION_CACHE_TTL_SECONDS", 300.0)
TRACK_HASH_TIME_WINDOW_SECONDS = _env_int("TRACK_HASH_TIME_WINDOW_SECONDS", 30)
TRACK_HASH_POINTS = max(3, _env_int("TRACK_HASH_POINTS", 10))
MAX_INFERENCE_TIME_SECONDS = _env_float("MAX_INFERENCE_TIME_SECONDS", 0.08)
PERF_WARNING_COOLDOWN_SECONDS = _env_float("PERF_WARNING_COOLDOWN_SECONDS", 3.0)

PERSON_MODEL = YOLO(str(PERSON_MODEL_PATH))


def _ensure_models_dir() -> None:
    try:
        PPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Failed to create models directory: %s", exc)


def _load_ppe_model() -> tuple[Optional[YOLO], str, Optional[str], Optional[str]]:
    _ensure_models_dir()

    if PPE_MODEL_PATH.exists():
        try:
            model = YOLO(str(PPE_MODEL_PATH))
            LOGGER.info("Loaded PPE model from %s", PPE_MODEL_PATH)
            return model, "primary", str(PPE_MODEL_PATH), None
        except Exception as exc:
            LOGGER.exception("Failed to load PPE model from %s", PPE_MODEL_PATH)
            return None, "primary_error", str(PPE_MODEL_PATH), f"load_error:{exc}"

    LOGGER.warning("PPE model not found at %s, attempting fallback model", PPE_MODEL_PATH)
    try:
        model = YOLO(str(PPE_FALLBACK_MODEL_PATH))
        LOGGER.info("Loaded fallback PPE model from %s", PPE_FALLBACK_MODEL_PATH)
        return model, "fallback", str(PPE_FALLBACK_MODEL_PATH), None
    except Exception as exc:
        LOGGER.exception("Failed to load fallback model from %s", PPE_FALLBACK_MODEL_PATH)
        return None, "fallback_error", str(PPE_FALLBACK_MODEL_PATH), f"load_error:{exc}"


PPE_MODEL, PPE_MODEL_SOURCE, PPE_MODEL_USED_PATH, PPE_MODEL_LOAD_ERROR = _load_ppe_model()


def _names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: str(v) for idx, v in enumerate(names)}
    return {}


def _normalize_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class ModelIntegrityError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]):
        super().__init__(message)
        self.details = details


def _log_structured(level: int, event: str, payload: dict[str, Any]) -> None:
    LOGGER.log(level, json.dumps({"event": event, **payload}, separators=(",", ":"), default=str))


def _model_class_names(model: Optional[YOLO]) -> list[str]:
    if model is None:
        return []
    names = _names_to_dict(getattr(model, "names", {}))
    return [_normalize_class_name(name) for _, name in sorted(names.items())]


def validate_model_classes(model: Optional[YOLO], expected_classes: tuple[str, ...]) -> dict[str, Any]:
    expected_norm = {_normalize_class_name(name) for name in expected_classes}
    actual_norm = set(_model_class_names(model))
    missing = sorted(expected_norm - actual_norm)
    unexpected = sorted(actual_norm - expected_norm)

    result = {
        "ok": len(missing) == 0,
        "expected_classes": sorted(expected_norm),
        "actual_classes": sorted(actual_norm),
        "missing_classes": missing,
        "unexpected_classes": unexpected,
        "model_path": PPE_MODEL_USED_PATH or str(PPE_MODEL_PATH),
        "model_source": PPE_MODEL_SOURCE,
    }
    if not result["ok"]:
        raise ModelIntegrityError("PPE model class integrity check failed", result)
    return result


def _build_ppe_class_maps(
    model: Optional[YOLO],
) -> tuple[dict[str, set[int]], dict[int, str]]:
    item_to_class_ids = {item: set() for item in PPE_ITEMS}
    class_item_map: dict[int, str] = {}
    if model is None:
        return item_to_class_ids, class_item_map

    names = _names_to_dict(getattr(model, "names", {}))
    for class_id, class_name in names.items():
        normalized = _normalize_class_name(class_name)
        if normalized not in EXPECTED_PPE_CLASS_TO_ITEM:
            continue
        item = EXPECTED_PPE_CLASS_TO_ITEM[normalized]
        item_to_class_ids[item].add(class_id)
        class_item_map[class_id] = item

    return item_to_class_ids, class_item_map


def _supported_ppe_items(class_map: dict[str, set[int]]) -> list[str]:
    return sorted(item for item, class_ids in class_map.items() if class_ids)


MODEL_INTEGRITY_STATUS: dict[str, Any] = {
    "ok": False,
    "expected_classes": sorted(_normalize_class_name(c) for c in EXPECTED_PPE_CLASSES),
    "actual_classes": [],
    "missing_classes": [],
    "unexpected_classes": [],
    "model_path": PPE_MODEL_USED_PATH or str(PPE_MODEL_PATH),
    "model_source": PPE_MODEL_SOURCE,
    "error": None,
}
PPE_CLASS_MAP, PPE_CLASS_ITEM_MAP = _build_ppe_class_maps(PPE_MODEL)
PPE_DETECTION_ENABLED = PPE_MODEL is not None
LAST_PERF_WARNING_BY_STAGE: dict[str, float] = {}
SHUTDOWN_EVENT = threading.Event()
JOBS_LOCK = threading.Lock()
JOBS: dict[str, "SourceJob"] = {}
ACTIVE_JOB_ID: Optional[str] = None


@dataclass
class SourceJob:
    job_id: str
    requested_source: str
    requested_path: str
    source: str
    source_path: str
    source_ready: bool = False
    source_error: Optional[str] = None
    latest_frame_jpeg: Optional[bytes] = None
    latest_frame_seq: int = 0
    latest_frame_time: Optional[str] = None
    people_detected: int = 0
    active_tracks: int = 0
    violating_tracks: int = 0
    total_logged_violations: int = 0
    last_violation: Optional[dict[str, Any]] = None
    fps: float = 0.0
    tracker: DeepSort = field(default_factory=lambda: DeepSort(max_age=30, embedder=None))
    track_cache: dict[int, dict[str, Any]] = field(default_factory=dict)
    last_violation_by_key: dict[str, float] = field(default_factory=dict)
    frame_count: int = 0
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.lock)

DATASET_CACHE = {"last_scan": 0.0, "summary": []}
DATASET_LOCK = threading.Lock()
DATASET_SCAN_INTERVAL_SECONDS = 60
LOG_COUNT_LOCK = threading.Lock()
DB_LOCK = threading.Lock()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _ensure_database_dir() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _db_connection() -> sqlite3.Connection:
    _ensure_database_dir()
    conn = sqlite3.connect(DATABASE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    with DB_LOCK:
        with _db_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    worker_id TEXT,
                    violation_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    camera_source TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    missing_json TEXT NOT NULL,
                    status_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_violations_timestamp
                    ON violations(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_violations_job_id
                    ON violations(job_id, timestamp DESC);

                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    source_status TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_uploads_job_id
                    ON uploads(job_id);
                CREATE INDEX IF NOT EXISTS idx_uploads_expires_at
                    ON uploads(expires_at);
                """
            )


def _count_existing_log_entries() -> int:
    with DB_LOCK:
        with _db_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM violations").fetchone()
    return int(row["count"]) if row is not None else 0


GLOBAL_LOG_COUNT = 0


def _record_upload(job_id: str, original_filename: str, stored_path: str) -> None:
    uploaded_at = datetime.datetime.now(datetime.UTC)
    expires_at = uploaded_at + datetime.timedelta(days=UPLOAD_RETENTION_DAYS)
    with DB_LOCK:
        with _db_connection() as conn:
            conn.execute(
                """
                INSERT INTO uploads (job_id, original_filename, stored_path, uploaded_at, expires_at, source_status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    original_filename,
                    stored_path,
                    uploaded_at.isoformat(),
                    expires_at.isoformat(),
                    "uploaded",
                ),
            )


def _update_upload_status(job_id: str, source_status: str) -> None:
    with DB_LOCK:
        with _db_connection() as conn:
            conn.execute(
                "UPDATE uploads SET source_status = ? WHERE job_id = ?",
                (source_status, job_id),
            )


def _database_health_snapshot() -> dict[str, Any]:
    with DB_LOCK:
        with _db_connection() as conn:
            violation_count = conn.execute("SELECT COUNT(*) AS count FROM violations").fetchone()["count"]
            upload_count = conn.execute("SELECT COUNT(*) AS count FROM uploads").fetchone()["count"]
    return {
        "path": str(DATABASE_PATH),
        "ok": True,
        "violations": int(violation_count),
        "uploads": int(upload_count),
    }


def _job_base_metrics(job: SourceJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "source": job.source,
        "source_path": job.source_path,
        "frame_time": job.latest_frame_time,
        "people_detected": job.people_detected,
        "active_tracks": job.active_tracks,
        "violating_tracks": job.violating_tracks,
        "total_logged_violations": GLOBAL_LOG_COUNT,
        "last_violation": job.last_violation,
        "fps": job.fps,
        "ppe_supported_items": _supported_ppe_items(PPE_CLASS_MAP) if PPE_DETECTION_ENABLED else [],
        "ppe_model_path": MODEL_INTEGRITY_STATUS.get("model_path", PPE_MODEL_USED_PATH or str(PPE_MODEL_PATH)),
        "ppe_model_source": MODEL_INTEGRITY_STATUS.get("model_source", PPE_MODEL_SOURCE),
        "model_integrity_ok": MODEL_INTEGRITY_STATUS["ok"],
        "model_missing_classes": MODEL_INTEGRITY_STATUS.get("missing_classes", []),
        "ppe_detection_enabled": PPE_DETECTION_ENABLED,
        "fps_overlay_enabled": FPS_OVERLAY_ENABLED,
        "source_ready": job.source_ready,
        "source_error": job.source_error,
    }


def _empty_metrics(job_id: Optional[str] = None) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "source": "UNINITIALIZED",
        "source_path": "",
        "frame_time": None,
        "people_detected": 0,
        "active_tracks": 0,
        "violating_tracks": 0,
        "total_logged_violations": 0,
        "last_violation": None,
        "fps": 0.0,
        "ppe_supported_items": _supported_ppe_items(PPE_CLASS_MAP) if PPE_DETECTION_ENABLED else [],
        "ppe_model_path": MODEL_INTEGRITY_STATUS.get("model_path", PPE_MODEL_USED_PATH or str(PPE_MODEL_PATH)),
        "ppe_model_source": MODEL_INTEGRITY_STATUS.get("model_source", PPE_MODEL_SOURCE),
        "model_integrity_ok": MODEL_INTEGRITY_STATUS["ok"],
        "model_missing_classes": MODEL_INTEGRITY_STATUS.get("missing_classes", []),
        "ppe_detection_enabled": PPE_DETECTION_ENABLED,
        "fps_overlay_enabled": FPS_OVERLAY_ENABLED,
        "source_ready": False,
        "source_error": "job_not_found",
    }


def _snapshot_metrics(job_id: Optional[str] = None) -> dict[str, Any]:
    job = _resolve_job(job_id)
    if job is None:
        return _empty_metrics(job_id)
    with job.lock:
        return dict(_job_base_metrics(job))


def _active_jobs_snapshot() -> list[dict[str, Any]]:
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    snapshot: list[dict[str, Any]] = []
    for job in jobs:
        with job.lock:
            snapshot.append(
                {
                    "job_id": job.job_id,
                    "source": job.source,
                    "source_path": job.source_path,
                    "source_ready": job.source_ready,
                    "source_error": job.source_error,
                    "last_frame_time": job.latest_frame_time,
                    "fps": job.fps,
                    "active_tracks": job.active_tracks,
                    "thread_alive": bool(job.thread and job.thread.is_alive()),
                }
            )
    return snapshot


def run_startup_validation(fail_on_error: bool = False) -> dict[str, Any]:
    global MODEL_INTEGRITY_STATUS, PPE_CLASS_MAP, PPE_CLASS_ITEM_MAP, PPE_DETECTION_ENABLED

    if PPE_MODEL is None:
        MODEL_INTEGRITY_STATUS = {
            "ok": False,
            "expected_classes": sorted(_normalize_class_name(c) for c in EXPECTED_PPE_CLASSES),
            "actual_classes": [],
            "missing_classes": sorted(_normalize_class_name(c) for c in EXPECTED_PPE_CLASSES),
            "unexpected_classes": [],
            "model_path": PPE_MODEL_USED_PATH or str(PPE_MODEL_PATH),
            "model_source": PPE_MODEL_SOURCE,
            "error": PPE_MODEL_LOAD_ERROR or "model_file_not_found",
        }
        _log_structured(logging.ERROR, "model_integrity_failure", MODEL_INTEGRITY_STATUS)
    else:
        try:
            MODEL_INTEGRITY_STATUS = validate_model_classes(PPE_MODEL, EXPECTED_PPE_CLASSES)
            _log_structured(logging.INFO, "model_integrity_ok", MODEL_INTEGRITY_STATUS)
        except ModelIntegrityError as exc:
            MODEL_INTEGRITY_STATUS = dict(exc.details)
            MODEL_INTEGRITY_STATUS["ok"] = False
            MODEL_INTEGRITY_STATUS["error"] = MODEL_INTEGRITY_STATUS.get("error") or "class_mismatch"
            _log_structured(logging.ERROR, "model_integrity_failure", MODEL_INTEGRITY_STATUS)

    PPE_CLASS_MAP, PPE_CLASS_ITEM_MAP = _build_ppe_class_maps(PPE_MODEL)
    PPE_DETECTION_ENABLED = MODEL_INTEGRITY_STATUS["ok"] and PPE_MODEL is not None

    if fail_on_error and not MODEL_INTEGRITY_STATUS["ok"]:
        raise SystemExit("Startup aborted due to model integrity failure")
    return MODEL_INTEGRITY_STATUS


init_database()
GLOBAL_LOG_COUNT = _count_existing_log_entries()
run_startup_validation(fail_on_error=False)
if not MODEL_INTEGRITY_STATUS["ok"]:
    actual = ", ".join(MODEL_INTEGRITY_STATUS.get("actual_classes", [])) or "none"
    expected = ", ".join(MODEL_INTEGRITY_STATUS.get("expected_classes", []))
    LOGGER.warning(
        "PPE detection disabled. Expected classes [%s] but model at %s provides [%s].",
        expected,
        MODEL_INTEGRITY_STATUS.get("model_path", PPE_MODEL_PATH),
        actual,
    )
    if "person" in MODEL_INTEGRITY_STATUS.get("actual_classes", []):
        LOGGER.warning(
            "The configured PPE model looks like a general COCO model. Replace models/ppe.pt with a PPE-trained model."
        )
if (
    REQUIRE_VALID_PPE_MODEL
    and not MODEL_INTEGRITY_STATUS["ok"]
    and MODEL_INTEGRITY_STATUS.get("model_source") == "primary"
):
    raise SystemExit("Startup blocked: PPE model integrity validation failed")


def _safe_crop(frame, x1: int, y1: int, x2: int, y2: int):
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _simple_embed(crop) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros(64, dtype=np.float32)

    resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm <= 1e-12:
        return np.zeros(64, dtype=np.float32)
    return hist / norm


def _build_embeddings(frame, detections) -> list[np.ndarray]:
    embeds: list[np.ndarray] = []
    for det in detections:
        x, y, w, h = det[0]
        crop = _safe_crop(frame, int(x), int(y), int(x + w), int(y + h))
        embeds.append(_simple_embed(crop))
    return embeds


def _read_classes_from_data_yaml(path: Path) -> list[str]:
    classes: list[str] = []
    if not path.exists():
        return classes

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return classes

    in_names_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("names:"):
            payload = stripped.split("names:", 1)[1].strip()
            if payload.startswith("[") and payload.endswith("]"):
                try:
                    values = ast.literal_eval(payload)
                    if isinstance(values, list):
                        classes = [str(v) for v in values]
                        return classes
                except (ValueError, SyntaxError):
                    pass
            in_names_block = True
            continue

        if in_names_block:
            if stripped.startswith("-"):
                classes.append(stripped.lstrip("-").strip().strip("'\""))
                continue
            match = re.match(r"^\d+\s*:\s*(.+)$", stripped)
            if match:
                classes.append(match.group(1).strip().strip("'\""))
                continue
            break

    return classes


def _count_files(path: Path, allowed_ext: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed_ext
    )


def _is_subpath(path: Path, parent: Path) -> bool:
    path_s = str(path)
    parent_s = str(parent)
    return path_s == parent_s or path_s.startswith(parent_s + os.sep)


def _find_dataset_roots(base: Path) -> list[Path]:
    roots: list[Path] = []
    for root, dirs, _ in os.walk(base):
        split_dirs = {"train", "valid", "test"}
        if split_dirs.intersection(set(dirs)):
            roots.append(Path(root))

    unique: list[Path] = []
    for root in sorted(roots, key=lambda p: len(str(p))):
        if not any(_is_subpath(root, existing) for existing in unique):
            unique.append(root)
    return unique


def scan_datasets() -> list[dict[str, Any]]:
    base = Path("datasets")
    if not base.exists():
        return []

    datasets: list[dict[str, Any]] = []
    for root in _find_dataset_roots(base):
        splits = {}
        for split in ("train", "valid", "test"):
            split_dir = root / split
            if not split_dir.exists():
                continue

            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            image_count = _count_files(images_dir if images_dir.exists() else split_dir, IMAGE_EXTENSIONS)
            label_count = _count_files(labels_dir if labels_dir.exists() else split_dir, {".txt"})

            splits[split] = {
                "images": image_count,
                "labels": label_count,
            }

        if not splits:
            continue

        datasets.append(
            {
                "name": str(root.relative_to(base)),
                "classes": _read_classes_from_data_yaml(root / "data.yaml"),
                "splits": splits,
            }
        )

    return datasets


def get_dataset_summary(force: bool = False) -> list[dict[str, Any]]:
    with DATASET_LOCK:
        now = time.time()
        if force or (now - DATASET_CACHE["last_scan"] > DATASET_SCAN_INTERVAL_SECONDS):
            DATASET_CACHE["summary"] = scan_datasets()
            DATASET_CACHE["last_scan"] = now
        return DATASET_CACHE["summary"]


def _handle_shutdown(signum=None, frame=None) -> None:
    SHUTDOWN_EVENT.set()
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    for job in jobs:
        job.stop_event.set()


def _source_label(source: str) -> str:
    labels = {
        "CAMERA": "Live Camera",
        "CAMERA_FALLBACK": "Camera Fallback Video",
        "RECORDED_VIDEO": "Recorded Video",
        "UPLOADED_VIDEO": "Uploaded Video",
    }
    return labels.get(source, source.replace("_", " ").title())


def _resolve_job(job_id: Optional[str] = None) -> Optional[SourceJob]:
    with JOBS_LOCK:
        if job_id and job_id in JOBS:
            return JOBS[job_id]
        if ACTIVE_JOB_ID and ACTIVE_JOB_ID in JOBS:
            return JOBS[ACTIVE_JOB_ID]
        return next(iter(JOBS.values()), None)


def _set_active_job(job_id: str) -> None:
    global ACTIVE_JOB_ID
    with JOBS_LOCK:
        if job_id in JOBS:
            ACTIVE_JOB_ID = job_id


def _job_stream_urls(job: SourceJob) -> dict[str, str]:
    return {
        "feed_url": url_for("video_feed_job", job_id=job.job_id),
        "metrics_url": url_for("api_metrics", job_id=job.job_id),
        "dashboard_url": url_for("dashboard", job_id=job.job_id),
        "logs_url": url_for("api_logs", job_id=job.job_id),
    }


def _await_job_start(job: SourceJob, timeout_seconds: float = 1.5) -> None:
    deadline = time.time() + timeout_seconds
    with job.condition:
        while (
            job.latest_frame_jpeg is None
            and job.source_error is None
            and not job.source_ready
            and time.time() < deadline
        ):
            remaining = max(deadline - time.time(), 0.0)
            if remaining <= 0:
                break
            job.condition.wait(timeout=remaining)


def _create_job(job_id: str, source: str, path: str) -> SourceJob:
    return SourceJob(
        job_id=job_id,
        requested_source=source,
        requested_path=path,
        source=source,
        source_path=path,
        total_logged_violations=_count_existing_log_entries(),
    )


def _open_capture_for_job(job: SourceJob) -> tuple[Optional[cv2.VideoCapture], str, str, Optional[str]]:
    selected_source = job.requested_source
    selected_path = job.requested_path

    if job.requested_source == "CAMERA":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap, "CAMERA", "0", None
        cap.release()
        if RECORDED_VIDEO_PATH.exists():
            cap = cv2.VideoCapture(str(RECORDED_VIDEO_PATH))
            if cap.isOpened():
                return cap, "CAMERA_FALLBACK", str(RECORDED_VIDEO_PATH), None
        return None, selected_source, "0", "camera_unavailable"

    if not selected_path:
        return None, selected_source, selected_path, "missing_source_path"

    cap = cv2.VideoCapture(selected_path)
    if not cap.isOpened():
        return None, selected_source, selected_path, "source_unavailable"
    return cap, selected_source, selected_path, None


def _ensure_job(source: str, path: Optional[str] = None) -> SourceJob:
    if source == "CAMERA":
        job_id = "camera"
        requested_path = "0"
    elif source == "RECORDED_VIDEO":
        job_id = "recorded"
        requested_path = path or str(RECORDED_VIDEO_PATH)
    elif source == "UPLOADED_VIDEO":
        job_id = f"upload-{uuid4().hex[:12]}"
        requested_path = path or ""
    else:
        raise ValueError(f"Unsupported source: {source}")

    with JOBS_LOCK:
        existing = JOBS.get(job_id)
        if existing and existing.thread and existing.thread.is_alive():
            return existing
        job = _create_job(job_id=job_id, source=source, path=requested_path)
        JOBS[job_id] = job

    thread = threading.Thread(target=_job_worker, args=(job,), daemon=True, name=f"job-{job_id}")
    job.thread = thread
    thread.start()
    _set_active_job(job.job_id)
    return job


def _ensure_uploaded_job(path: str) -> SourceJob:
    job = _create_job(job_id=f"upload-{uuid4().hex[:12]}", source="UPLOADED_VIDEO", path=path)
    with JOBS_LOCK:
        JOBS[job.job_id] = job
    thread = threading.Thread(target=_job_worker, args=(job,), daemon=True, name=f"job-{job.job_id}")
    job.thread = thread
    thread.start()
    _set_active_job(job.job_id)
    return job


def _ensure_uploaded_job_with_metadata(path: str, original_filename: str) -> SourceJob:
    job = _ensure_uploaded_job(path)
    _record_upload(job.job_id, original_filename, path)
    return job


def ensure_default_job() -> Optional[SourceJob]:
    existing = _resolve_job()
    if existing is not None:
        return existing
    if AUTO_OPEN_SOURCE == "camera":
        return _ensure_job("CAMERA")
    return _ensure_job("RECORDED_VIDEO", str(RECORDED_VIDEO_PATH))


def _bbox_intersection_area(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    inter = _bbox_intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_containment_ratio(inner_box: tuple[int, int, int, int], outer_box: tuple[int, int, int, int]) -> float:
    inter = _bbox_intersection_area(inner_box, outer_box)
    inner_area = max(_bbox_area(inner_box), 1)
    return inter / inner_area


def _bbox_center_inside(inner_box: tuple[int, int, int, int], outer_box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = inner_box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _monitor_inference_time(stage: str, duration_seconds: float, source: str) -> None:
    if duration_seconds <= MAX_INFERENCE_TIME_SECONDS:
        return
    now = time.time()
    last = LAST_PERF_WARNING_BY_STAGE.get(stage, 0.0)
    if now - last < PERF_WARNING_COOLDOWN_SECONDS:
        return
    LAST_PERF_WARNING_BY_STAGE[stage] = now
    _log_structured(
        logging.WARNING,
        "inference_slow",
        {
            "stage": stage,
            "duration_ms": round(duration_seconds * 1000.0, 2),
            "threshold_ms": round(MAX_INFERENCE_TIME_SECONDS * 1000.0, 2),
            "source": source,
        },
    )


def _run_person_detection(frame) -> list[tuple[int, int, int, int]]:
    person_boxes: list[tuple[int, int, int, int]] = []
    started = time.perf_counter()
    try:
        result = PERSON_MODEL(frame, conf=PERSON_CONFIDENCE_THRESHOLD, verbose=False)[0]
    except Exception:
        return person_boxes
    finally:
        _monitor_inference_time("person_detection", time.perf_counter() - started, "PERSON_MODEL")

    names = _names_to_dict(getattr(PERSON_MODEL, "names", {}))
    if result.boxes is None:
        return person_boxes

    for box in result.boxes:
        cls_id = int(box.cls[0])
        if names.get(cls_id, "").lower() != "person":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_boxes.append((x1, y1, x2, y2))
    return person_boxes


def _run_ppe_detection(frame) -> Optional[list[tuple[int, tuple[int, int, int, int], float]]]:
    if not PPE_DETECTION_ENABLED or PPE_MODEL is None:
        return None

    started = time.perf_counter()
    try:
        result = PPE_MODEL(frame, conf=PPE_CONFIDENCE_THRESHOLD, verbose=False)[0]
    except Exception:
        return None
    finally:
        _monitor_inference_time("ppe_detection", time.perf_counter() - started, "PPE_MODEL")

    detections: list[tuple[int, tuple[int, int, int, int], float]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        detections.append((cls_id, (x1, y1, x2, y2), conf))
    return detections


def _select_track_for_ppe_detection(
    detection_box: tuple[int, int, int, int],
    track_boxes: dict[int, tuple[int, int, int, int]],
) -> Optional[int]:
    scores: list[tuple[int, float, float, bool]] = []
    for track_id, person_box in track_boxes.items():
        iou = _bbox_iou(detection_box, person_box)
        containment = _bbox_containment_ratio(detection_box, person_box)
        center_inside = _bbox_center_inside(detection_box, person_box)
        scores.append((track_id, iou, containment, center_inside))

    if not scores:
        return None

    by_iou = sorted(scores, key=lambda s: s[1], reverse=True)
    top_iou_track, top_iou, top_containment, top_inside = by_iou[0]
    second_iou = by_iou[1][1] if len(by_iou) > 1 else 0.0
    if (
        top_iou >= PPE_ASSOCIATION_IOU_THRESHOLD
        and (top_iou - second_iou) >= PPE_ASSOCIATION_MARGIN
    ):
        return top_iou_track

    containment_candidates = [s for s in scores if s[3] or s[2] >= PPE_CONTAINMENT_THRESHOLD]
    if not containment_candidates:
        return None
    by_containment = sorted(containment_candidates, key=lambda s: (s[2], s[1]), reverse=True)
    top_contain_track, top_iou, top_containment, _ = by_containment[0]
    second_containment = by_containment[1][2] if len(by_containment) > 1 else 0.0
    if (
        top_containment >= PPE_CONTAINMENT_THRESHOLD
        and (top_containment - second_containment) >= PPE_ASSOCIATION_MARGIN
    ):
        return top_contain_track
    return None


def _associate_ppe_to_tracks(
    track_boxes: dict[int, tuple[int, int, int, int]],
    ppe_detections: list[tuple[int, tuple[int, int, int, int], float]],
) -> dict[int, dict[str, dict[str, float | bool]]]:
    evidence_by_track = {
        track_id: {
            item: {"detected": False, "confidence": 0.0}
            for item in PPE_ITEMS
        }
        for track_id in track_boxes
    }
    if not track_boxes:
        return evidence_by_track

    for cls_id, ppe_box, conf in sorted(ppe_detections, key=lambda d: d[2], reverse=True):
        item = PPE_CLASS_ITEM_MAP.get(cls_id)
        if item is None:
            continue

        matched_track_id = _select_track_for_ppe_detection(ppe_box, track_boxes)
        if matched_track_id is None:
            continue

        current_conf = evidence_by_track[matched_track_id][item]["confidence"] or 0.0
        if conf >= current_conf:
            evidence_by_track[matched_track_id][item] = {
                "detected": True,
                "confidence": round(float(conf), 4),
            }

    return evidence_by_track


def _new_track_state(frame_number: int) -> dict[str, Any]:
    return {
        "last_frame": frame_number,
        "status": {item: False for item in PPE_ITEMS},
        "confidence": {item: 0.0 for item in PPE_ITEMS},
        "history": {item: deque(maxlen=TEMPORAL_WINDOW_SIZE) for item in PPE_ITEMS},
        "trajectory": deque(maxlen=TRACK_HASH_POINTS),
        "worker_id": "",
    }


def _empty_frame_evidence() -> dict[str, dict[str, Any]]:
    return {
        item: {"detected": None, "confidence": 0.0}
        for item in PPE_ITEMS
    }


def _update_worker_id(
    track_state: dict[str, Any], track_box: tuple[int, int, int, int], now_ts: float, source: str
) -> str:
    x1, y1, x2, y2 = track_box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    trajectory = track_state.get("trajectory")
    if not isinstance(trajectory, deque):
        trajectory = deque(maxlen=TRACK_HASH_POINTS)
        track_state["trajectory"] = trajectory
    trajectory.append((cx // 8, cy // 8))

    time_bucket = int(now_ts // max(TRACK_HASH_TIME_WINDOW_SECONDS, 1))
    payload = {
        "source": source,
        "bucket": time_bucket,
        "trajectory": list(trajectory),
    }
    stable_hash = hashlib.sha1(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()[:12]
    worker_id = f"worker_{stable_hash}"
    track_state["worker_id"] = worker_id
    return worker_id


def _apply_temporal_filter(
    track_state: dict[str, Any],
    frame_evidence: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, float]]:
    history = track_state.get("history")
    if not isinstance(history, dict):
        history = {item: deque(maxlen=TEMPORAL_WINDOW_SIZE) for item in PPE_ITEMS}
        track_state["history"] = history

    smoothed_status = dict(track_state.get("status", {item: False for item in PPE_ITEMS}))
    smoothed_conf = dict(track_state.get("confidence", {item: 0.0 for item in PPE_ITEMS}))

    for item in PPE_ITEMS:
        item_history = history.get(item)
        if not isinstance(item_history, deque):
            item_history = deque(maxlen=TEMPORAL_WINDOW_SIZE)
            history[item] = item_history

        detected = frame_evidence[item]["detected"]
        conf = float(frame_evidence[item]["confidence"] or 0.0)
        if detected is not None:
            item_history.append(bool(detected))

        positive_count = sum(1 for v in item_history if v is True)
        negative_count = sum(1 for v in item_history if v is False)

        if positive_count >= TEMPORAL_CONFIRM_FRAMES and positive_count >= negative_count:
            smoothed_status[item] = True
            smoothed_conf[item] = max(smoothed_conf[item], conf)
        elif negative_count >= TEMPORAL_CONFIRM_FRAMES and negative_count > positive_count:
            smoothed_status[item] = False
            smoothed_conf[item] = max(smoothed_conf[item] * 0.95, conf)
        elif detected is not None:
            smoothed_conf[item] = max(smoothed_conf[item] * 0.95, conf)

    track_state["status"] = smoothed_status
    track_state["confidence"] = smoothed_conf
    return smoothed_status, smoothed_conf


def _build_violation_fingerprint(source: str, worker_id: str, missing_items: list[str]) -> str:
    violation_type = ",".join(sorted(f"missing_{item}" for item in missing_items))
    return f"{source}|{worker_id}|{violation_type}"


def _violation_confidence(missing_items: list[str], item_conf: dict[str, float]) -> float:
    if not missing_items:
        return 0.0
    confidences = [float(item_conf.get(item, 0.0)) for item in missing_items]
    return round(sum(confidences) / max(len(confidences), 1), 4)


def _cleanup_stale_state(job: SourceJob, frame_number: int, active_track_ids: set[int]) -> None:
    stale_track_ids = [
        track_id
        for track_id, state in job.track_cache.items()
        if (track_id not in active_track_ids) and (frame_number - int(state.get("last_frame", 0)) > TRACK_STALE_FRAME_LIMIT)
    ]
    for track_id in stale_track_ids:
        job.track_cache.pop(track_id, None)

    now = time.time()
    stale_violation_keys = [
        key for key, ts in job.last_violation_by_key.items() if now - ts > VIOLATION_CACHE_TTL_SECONDS
    ]
    for key in stale_violation_keys:
        job.last_violation_by_key.pop(key, None)


def _apply_frame_rate_limit(frame_started_at: float) -> None:
    if TARGET_MAX_FPS <= 0:
        return
    frame_budget = 1.0 / TARGET_MAX_FPS
    elapsed = time.perf_counter() - frame_started_at
    if elapsed < frame_budget:
        time.sleep(frame_budget - elapsed)


def write_violation(
    job: SourceJob,
    track_id: int,
    worker_id: str,
    missing_items: list[str],
    ppe_status: dict[str, bool],
    confidence: float,
) -> None:
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    violation_type = ",".join(sorted(f"missing_{item}" for item in missing_items))
    payload = {
        "job_id": job.job_id,
        "worker_id": worker_id,
        "violation_type": violation_type,
        "confidence": round(float(confidence), 4),
        "timestamp": timestamp,
        "camera_source": job.source,
        "source": job.source,
        "source_path": job.source_path,
        "id": int(track_id),
        "missing": missing_items,
        "status": ppe_status,
    }

    with DB_LOCK:
        with _db_connection() as conn:
            conn.execute(
                """
                INSERT INTO violations (
                    job_id,
                    worker_id,
                    violation_type,
                    confidence,
                    timestamp,
                    camera_source,
                    source,
                    source_path,
                    track_id,
                    missing_json,
                    status_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["job_id"],
                    payload["worker_id"],
                    payload["violation_type"],
                    payload["confidence"],
                    payload["timestamp"],
                    payload["camera_source"],
                    payload["source"],
                    payload["source_path"],
                    payload["id"],
                    json.dumps(payload["missing"], separators=(",", ":")),
                    json.dumps(payload["status"], separators=(",", ":")),
                ),
            )

    _log_structured(logging.INFO, "violation_logged", payload)

    global GLOBAL_LOG_COUNT
    with LOG_COUNT_LOCK:
        GLOBAL_LOG_COUNT += 1
    with job.lock:
        job.last_violation = payload


def _violation_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        missing = json.loads(row["missing_json"]) if row["missing_json"] else []
    except json.JSONDecodeError:
        missing = []

    try:
        status = json.loads(row["status_json"]) if row["status_json"] else {}
    except json.JSONDecodeError:
        status = {}

    return {
        "job_id": row["job_id"],
        "worker_id": row["worker_id"],
        "violation_type": row["violation_type"],
        "confidence": row["confidence"],
        "timestamp": row["timestamp"],
        "source": row["source"],
        "camera_source": row["camera_source"],
        "source_path": row["source_path"],
        "id": row["track_id"],
        "missing": missing,
        "status": status,
    }


def get_recent_violations(limit: int = 100, job_id: Optional[str] = None) -> list[dict[str, Any]]:
    query = """
        SELECT job_id, worker_id, violation_type, confidence, timestamp, camera_source, source, source_path,
               track_id, missing_json, status_json
        FROM violations
    """
    params: list[Any] = []
    if job_id is not None:
        query += " WHERE job_id = ?"
        params.append(job_id)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with DB_LOCK:
        with _db_connection() as conn:
            rows = conn.execute(query, params).fetchall()
    return [_violation_from_row(row) for row in rows]


def _job_worker(job: SourceJob) -> None:
    capture, actual_source, actual_path, open_error = _open_capture_for_job(job)
    with job.condition:
        job.source = actual_source
        job.source_path = actual_path
        job.source_ready = capture is not None and open_error is None
        job.source_error = open_error
        job.condition.notify_all()

    if job.requested_source == "UPLOADED_VIDEO":
        _update_upload_status(job.job_id, "ready" if capture is not None and open_error is None else (open_error or "source_open_failed"))

    if capture is None:
        return

    try:
        while not SHUTDOWN_EVENT.is_set() and not job.stop_event.is_set():
            frame_started_at = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if job.source in {"RECORDED_VIDEO", "UPLOADED_VIDEO", "CAMERA_FALLBACK"}:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                time.sleep(0.05)
                continue

            with job.lock:
                job.frame_count += 1
                frame_number = job.frame_count

            person_boxes = _run_person_detection(frame)
            detections = [
                ([x1, y1, x2 - x1, y2 - y1], 0.9, "person")
                for (x1, y1, x2, y2) in person_boxes
            ]
            embeds = _build_embeddings(frame, detections)
            tracks = job.tracker.update_tracks(detections, embeds=embeds)

            ppe_detections = _run_ppe_detection(frame)
            ppe_inference_ok = ppe_detections is not None

            confirmed_track_boxes: dict[int, tuple[int, int, int, int]] = {}
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = int(track.track_id)
                confirmed_track_boxes[track_id] = tuple(map(int, track.to_ltrb()))

            evidence_from_ppe = {}
            if ppe_inference_ok:
                evidence_from_ppe = _associate_ppe_to_tracks(confirmed_track_boxes, ppe_detections or [])

            detected_people = len(person_boxes)
            active_tracks = 0
            violating_tracks = 0
            active_track_ids: set[int] = set()
            now_ts = time.time()

            for track_id, (x1, y1, x2, y2) in confirmed_track_boxes.items():
                active_tracks += 1
                active_track_ids.add(track_id)
                track_state = job.track_cache.get(track_id)
                if track_state is None:
                    track_state = _new_track_state(frame_number)

                should_refresh = frame_number - int(track_state.get("last_frame", 0)) >= PPE_REFRESH_FRAMES

                frame_evidence = _empty_frame_evidence()
                if should_refresh and ppe_inference_ok:
                    frame_evidence = evidence_from_ppe.get(track_id, _empty_frame_evidence())
                smoothed_status, smoothed_confidence = _apply_temporal_filter(track_state, frame_evidence)
                worker_id = _update_worker_id(track_state, (x1, y1, x2, y2), now_ts, job.source)

                track_state["last_frame"] = frame_number
                job.track_cache[track_id] = track_state

                if ppe_inference_ok:
                    ppe_status = smoothed_status
                    ppe_confidence = smoothed_confidence
                else:
                    ppe_status = dict(track_state.get("status", {item: False for item in PPE_ITEMS}))
                    ppe_confidence = dict(track_state.get("confidence", {item: 0.0 for item in PPE_ITEMS}))

                missing_items = [item for item, present in ppe_status.items() if not present]
                is_violation = bool(missing_items)
                if is_violation:
                    violating_tracks += 1

                box_color = (0, 0, 255) if is_violation else (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

                text_y = y1 + 20
                for item in PPE_ITEMS:
                    present = ppe_status[item]
                    color = (0, 200, 0) if present else (0, 0, 255)
                    cv2.putText(
                        frame,
                        f"{item.upper()}: {'OK' if present else 'NO'}",
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    text_y += 18

                if is_violation:
                    fingerprint = _build_violation_fingerprint(job.source, worker_id, missing_items)
                    now = time.time()
                    last_time = job.last_violation_by_key.get(fingerprint, 0.0)
                    if now - last_time >= VIOLATION_DEDUP_SECONDS:
                        job.last_violation_by_key[fingerprint] = now
                        write_violation(
                            job=job,
                            track_id=track_id,
                            worker_id=worker_id,
                            missing_items=missing_items,
                            ppe_status=ppe_status,
                            confidence=_violation_confidence(missing_items, ppe_confidence),
                        )

            _cleanup_stale_state(job, frame_number, active_track_ids)

            elapsed = max(time.perf_counter() - frame_started_at, 1e-6)
            fps_value = round(1.0 / elapsed, 2)
            if FPS_OVERLAY_ENABLED:
                cv2.putText(
                    frame,
                    f"FPS: {fps_value}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 230, 0),
                    2,
                )

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            with job.condition:
                job.people_detected = detected_people
                job.active_tracks = active_tracks
                job.violating_tracks = violating_tracks
                job.fps = fps_value
                job.latest_frame_time = datetime.datetime.now(datetime.UTC).isoformat()
                job.latest_frame_jpeg = buffer.tobytes()
                job.latest_frame_seq += 1
                job.condition.notify_all()

            _apply_frame_rate_limit(frame_started_at)
    finally:
        capture.release()


def generate_frames(job_id: str):
    job = _resolve_job(job_id)
    if job is None:
        return

    last_seq = -1
    while not SHUTDOWN_EVENT.is_set() and not job.stop_event.is_set():
        with job.condition:
            if job.latest_frame_jpeg is None or job.latest_frame_seq == last_seq:
                job.condition.wait(timeout=0.5)
            if job.latest_frame_jpeg is None:
                continue
            last_seq = job.latest_frame_seq
            jpg = job.latest_frame_jpeg

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


@app.route("/camera")
def camera():
    job = _ensure_job("CAMERA")
    _await_job_start(job)
    return _render_job_camera(job)


@app.route("/recorded")
def recorded():
    job = _ensure_job("RECORDED_VIDEO", str(RECORDED_VIDEO_PATH))
    _await_job_start(job)
    return _render_job_camera(job)


@app.route("/upload", methods=["GET", "POST"])
def upload_video():
    if request.method == "GET":
        return render_template("upload.html")

    uploaded = request.files.get("video")
    if not uploaded or uploaded.filename is None:
        return redirect(url_for("upload_video"))

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return redirect(url_for("upload_video"))

    Path("videos").mkdir(exist_ok=True)
    save_path = Path("videos") / f"uploaded_{int(time.time())}{suffix}"
    uploaded.save(save_path)

    job = _ensure_uploaded_job_with_metadata(str(save_path), uploaded.filename)
    _await_job_start(job)
    return _render_job_camera(job)


def _render_job_camera(job: SourceJob):
    urls = _job_stream_urls(job)
    with job.lock:
        source = _source_label(job.source)
        source_path = job.source_path
        source_ready = job.source_ready
    status_code = 200 if source_ready else 503
    return render_template(
        "camera.html",
        source=source,
        source_path=source_path,
        feed_url=urls["feed_url"],
        metrics_url=urls["metrics_url"],
        dashboard_url=urls["dashboard_url"],
        job_id=job.job_id,
    ), status_code


@app.route("/jobs/<job_id>")
def view_job(job_id: str):
    job = _resolve_job(job_id)
    if job is None:
        return redirect(url_for("dashboard"))
    _set_active_job(job.job_id)
    return _render_job_camera(job)


@app.route("/video_feed")
def video_feed():
    job = ensure_default_job()
    if job is None:
        return Response(status=503)
    return Response(generate_frames(job.job_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/<job_id>")
def video_feed_job(job_id: str):
    job = _resolve_job(job_id)
    if job is None:
        return Response(status=404)
    return Response(generate_frames(job.job_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/dashboard")
def dashboard():
    requested_job_id = request.args.get("job_id")
    job = _resolve_job(requested_job_id)
    if job is None:
        job = ensure_default_job()
    if job is None:
        return render_template(
            "dashboard.html",
            job_id="",
            feed_url=url_for("video_feed"),
            metrics_url=url_for("api_metrics"),
            logs_url=url_for("api_logs"),
        )
    urls = _job_stream_urls(job)
    return render_template(
        "dashboard.html",
        job_id=job.job_id,
        feed_url=urls["feed_url"],
        metrics_url=urls["metrics_url"],
        logs_url=urls["logs_url"],
        camera_view_url=url_for("view_job", job_id=job.job_id),
    )


@app.route("/api/metrics")
def api_metrics():
    job_id = request.args.get("job_id")
    return jsonify(_snapshot_metrics(job_id))


@app.route("/api/logs")
def api_logs():
    limit = request.args.get("limit", default=80, type=int)
    limit = max(1, min(limit, 500))
    job_id = request.args.get("job_id")
    return jsonify({"logs": get_recent_violations(limit=limit, job_id=job_id)})


@app.route("/api/health")
def api_health():
    active_job = _resolve_job(request.args.get("job_id"))
    metrics = _snapshot_metrics(active_job.job_id if active_job is not None else None)
    db_health = _database_health_snapshot()
    return jsonify(
        {
            "status": "ok" if db_health["ok"] else "degraded",
            "database": db_health,
            "model": {
                "integrity_ok": MODEL_INTEGRITY_STATUS["ok"],
                "model_path": MODEL_INTEGRITY_STATUS.get("model_path", PPE_MODEL_USED_PATH or str(PPE_MODEL_PATH)),
                "model_source": MODEL_INTEGRITY_STATUS.get("model_source", PPE_MODEL_SOURCE),
                "missing_classes": MODEL_INTEGRITY_STATUS.get("missing_classes", []),
                "actual_classes": MODEL_INTEGRITY_STATUS.get("actual_classes", []),
            },
            "active_job": metrics,
            "jobs": _active_jobs_snapshot(),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    )


@app.route("/api/datasets")
def api_datasets():
    force = request.args.get("refresh", "0") == "1"
    return jsonify({"datasets": get_dataset_summary(force=force)})


if __name__ == "__main__":
    atexit.register(_handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    run_startup_validation(fail_on_error=REQUIRE_VALID_PPE_MODEL)
    get_dataset_summary(force=True)
    ensure_default_job()
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
