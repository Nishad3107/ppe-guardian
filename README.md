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
python app.py
