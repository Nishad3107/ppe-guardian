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
