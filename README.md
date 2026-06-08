# Marketwise – PPE Helmet Safety Monitoring System

## Overview
Marketwise is a real-time computer vision system that monitors helmet compliance.
It detects people, assigns unique IDs using tracking, and checks helmet presence
from the head region. Violations are highlighted and logged with timestamps.

## Features
- Real-time person detection
- Unique ID tracking (DeepSORT)
- Helmet compliance check
- Violation logging
- Live web dashboard (Flask)

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- DeepSORT
- Flask

## How It Works
1. Detect people in each frame
2. Track each person with a unique ID
3. Crop head region from each person
4. Detect helmet presence
5. If helmet not detected → mark violation and log it

## Run Locally
```bash
npm run dev
```

Fallback:
```bash
python app.py
```

## Local Config
```bash
cp .env.example .env.local
```

Use `.env.local` for machine-specific settings such as `SERVER_PORT`,
`AUTO_OPEN_SOURCE`, or `REQUIRE_VALID_PPE_MODEL`. The file is ignored by git.

## Persistence
The app now stores violations and upload metadata in SQLite at `data/ppe_guardian.db`
by default. You can override the location with `DATABASE_PATH` in `.env.local`.

Health and observability:
```bash
curl http://127.0.0.1:8000/api/health
```

## Security Controls
- Uploads are limited by `MAX_UPLOAD_SIZE_MB`
- Unsupported upload mime types and extensions are rejected
- Expired uploaded videos are cleaned up automatically using `UPLOAD_RETENTION_DAYS`
- `/upload`, `/dashboard`, `/jobs/*`, `/api/logs`, `/api/datasets`, and `/api/health` require login

Default local credentials:
```bash
username: admin
password: admin123
```

Override them in `.env.local` before using this outside local development.

## Production WSGI
Use a production WSGI server instead of `python app.py`:
```bash
gunicorn --bind 0.0.0.0:8000 wsgi:application
```

## Phase 5 Verification
Phase 5 moves PPE detection onto person upper-body crops and adds repeatable runtime
and regression checks. When the PPE model is invalid or disabled, the app now shows
an unknown PPE state instead of generating false violations.

Runtime benchmark:
```bash
./venv/bin/python scripts/benchmark_runtime.py --video videos/test-video.mp4 --frames 120 --warmup 5
```

Regression evaluation:
```bash
./venv/bin/python scripts/evaluate_regression.py --video videos/test-video.mp4 --expected evaluation/test-video.expected.json
```
