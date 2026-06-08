import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeBox:
    def __init__(self, cls_id=0, xyxy=(0, 0, 10, 10), conf=0.9):
        self.cls = [cls_id]
        self.xyxy = [xyxy]
        self.conf = [conf]


class FakeYOLO:
    def __init__(self, path: str):
        self.path = path
        lowered = Path(path).name.lower()
        if lowered == "ppe-valid.pt":
            self.names = {0: "helmet", 1: "mask", 2: "glasses", 3: "boots"}
        elif lowered == "ppe.pt":
            self.names = {0: "person", 1: "car"}
        else:
            self.names = {0: "person"}

    def __call__(self, _frame, conf=0.0, verbose=False):
        return [types.SimpleNamespace(boxes=None)]


class FakeVideoCapture:
    specs = {}

    def __init__(self, source):
        self.source = source
        spec = self.specs.get(source, self.specs.get(str(source), {}))
        self.opened = bool(spec.get("opened", False))

    def isOpened(self):
        return self.opened

    def release(self):
        return None

    def read(self):
        return False, None

    def set(self, *_args, **_kwargs):
        return True


class FakeDeepSort:
    def __init__(self, *args, **kwargs):
        pass

    def update_tracks(self, detections, embeds=None):
        return []


def install_fake_modules():
    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_POS_FRAMES = 1
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.VideoCapture = FakeVideoCapture
    cv2.rectangle = lambda *args, **kwargs: None
    cv2.putText = lambda *args, **kwargs: None
    cv2.imencode = lambda *args, **kwargs: (True, b"jpeg")
    cv2.resize = lambda frame, size, interpolation=None: frame
    cv2.cvtColor = lambda frame, code: frame
    cv2.calcHist = lambda *args, **kwargs: [0.0] * 64
    cv2.COLOR_BGR2HSV = 40
    cv2.INTER_AREA = 3
    sys.modules["cv2"] = cv2

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    sys.modules["ultralytics"] = ultralytics

    deep_sort_pkg = types.ModuleType("deep_sort_realtime")
    deep_sort_tracker = types.ModuleType("deep_sort_realtime.deepsort_tracker")
    deep_sort_tracker.DeepSort = FakeDeepSort
    sys.modules["deep_sort_realtime"] = deep_sort_pkg
    sys.modules["deep_sort_realtime.deepsort_tracker"] = deep_sort_tracker


class AppModuleTestCase(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.base)

        (self.base / "models").mkdir(parents=True, exist_ok=True)
        (self.base / "videos").mkdir(parents=True, exist_ok=True)
        (self.base / "data").mkdir(parents=True, exist_ok=True)
        (self.base / "models" / "ppe.pt").write_bytes(b"fake")
        (self.base / "videos" / "test-video.mp4").write_bytes(b"video")
        (self.base / "yolov8n.pt").write_bytes(b"person")

        self.env_backup = os.environ.copy()
        os.environ.update(
            {
                "PERSON_MODEL_PATH": str(self.base / "yolov8n.pt"),
                "PPE_MODEL_PATH": str(self.base / "models" / "ppe.pt"),
                "RECORDED_VIDEO_PATH": str(self.base / "videos" / "test-video.mp4"),
                "DATABASE_PATH": str(self.base / "data" / "ppe_guardian.db"),
                "AUTH_REQUIRED": "0",
                "SECRET_KEY": "test-secret",
            }
        )

        self.module_backup = {
            name: sys.modules.get(name)
            for name in ("app", "cv2", "ultralytics", "deep_sort_realtime", "deep_sort_realtime.deepsort_tracker")
        }
        for name in self.module_backup:
            sys.modules.pop(name, None)
        install_fake_modules()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        self.app_module = importlib.import_module("app")
        FakeVideoCapture.specs = {}

    def tearDown(self):
        os.chdir(self.old_cwd)
        sys.modules.pop("app", None)
        for name in ("cv2", "ultralytics", "deep_sort_realtime", "deep_sort_realtime.deepsort_tracker"):
            sys.modules.pop(name, None)
        for name, module in self.module_backup.items():
            if module is not None:
                sys.modules[name] = module
        os.environ.clear()
        os.environ.update(self.env_backup)
        self.tempdir.cleanup()

    def test_run_startup_validation_disables_invalid_ppe_model(self):
        status = self.app_module.run_startup_validation(fail_on_error=False)
        self.assertFalse(status["ok"])
        self.assertIn("helmet", status["missing_classes"])
        self.assertFalse(self.app_module.PPE_DETECTION_ENABLED)

    def test_build_startup_checklist_reports_existing_paths(self):
        checklist = self.app_module.build_startup_checklist()
        self.assertTrue(checklist["ok"])
        self.assertTrue(checklist["checks"]["person_model"]["ok"])
        self.assertTrue(checklist["checks"]["ppe_model"]["exists"])
        self.assertTrue(checklist["checks"]["recorded_video"]["exists"])

    def test_open_capture_for_job_falls_back_to_recorded_video(self):
        FakeVideoCapture.specs = {
            0: {"opened": False},
            str(self.app_module.RECORDED_VIDEO_PATH): {"opened": True},
        }
        job = self.app_module._create_job("camera", "CAMERA", "0")
        capture, source, source_path, error = self.app_module._open_capture_for_job(job)
        self.assertIsNotNone(capture)
        self.assertEqual(source, "CAMERA_FALLBACK")
        self.assertEqual(source_path, str(self.app_module.RECORDED_VIDEO_PATH))
        self.assertIsNone(error)

    def test_upload_route_rejects_invalid_extension(self):
        client = self.app_module.app.test_client()
        response = client.post(
            "/upload",
            data={"video": (io.BytesIO(b"bad"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/upload"))

    def test_upload_route_accepts_valid_video(self):
        client = self.app_module.app.test_client()
        job = self.app_module._create_job("upload-test", "UPLOADED_VIDEO", str(self.base / "videos" / "uploaded.mp4"))
        job.source = "UPLOADED_VIDEO"
        job.source_path = str(self.base / "videos" / "uploaded.mp4")
        job.source_ready = True
        with mock.patch.object(self.app_module, "_ensure_uploaded_job_with_metadata", return_value=job), mock.patch.object(
            self.app_module, "_await_job_start", return_value=None
        ):
            response = client.post(
                "/upload",
                data={"video": (io.BytesIO(b"video"), "clip.mp4")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"upload-test", response.data)

    def test_metrics_route_returns_job_snapshot(self):
        client = self.app_module.app.test_client()
        job = self.app_module._create_job("recorded", "RECORDED_VIDEO", str(self.app_module.RECORDED_VIDEO_PATH))
        job.source = "RECORDED_VIDEO"
        job.source_path = str(self.app_module.RECORDED_VIDEO_PATH)
        job.source_ready = True
        job.people_detected = 3
        job.active_tracks = 2
        with self.app_module.JOBS_LOCK:
            self.app_module.JOBS[job.job_id] = job
        response = client.get("/api/metrics", query_string={"job_id": job.job_id})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["job_id"], "recorded")
        self.assertEqual(payload["people_detected"], 3)
        self.assertEqual(payload["active_tracks"], 2)

    def test_violation_row_parsing_tolerates_invalid_json(self):
        parsed = self.app_module._violation_from_row(
            {
                "job_id": "job-1",
                "worker_id": "worker-1",
                "violation_type": "missing_helmet",
                "confidence": 0.9,
                "timestamp": "2026-06-08T00:00:00+00:00",
                "source": "RECORDED_VIDEO",
                "camera_source": "RECORDED_VIDEO",
                "source_path": "videos/test-video.mp4",
                "track_id": 4,
                "missing_json": "{bad json",
                "status_json": "{bad json",
            }
        )
        self.assertEqual(parsed["missing"], [])
        self.assertEqual(parsed["status"], {})


if __name__ == "__main__":
    unittest.main()
