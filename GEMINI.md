# Project Overview

This project is a real-time computer vision system for monitoring personal protective equipment (PPE) compliance, specifically focusing on helmet safety. It uses a combination of object detection and tracking to identify individuals in a video stream, track them, and then check for the presence of a helmet.

The system is built primarily in Python and leverages several key libraries:

*   **YOLOv8 (Ultralytics):** For real-time object detection. Two models are used: a general person detection model (`yolov8n.pt`) and a custom-trained model for PPE detection (`models/ppe.pt`).
*   **DeepSORT:** For tracking detected individuals across video frames, assigning a unique ID to each person.
*   **OpenCV:** For video capture and image processing.
*   **Flask:** To provide a web-based dashboard for live monitoring and viewing violation logs.

The application can process video from either a live camera feed or a pre-recorded video file. When a person is detected without a helmet, a violation is recorded in a `violations.log` file, and the violation is visually highlighted in the video feed.

# Building and Running

## Dependencies

The project's dependencies are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

## Running the Application

The main application is a Flask web server. To start the server, run:

```bash
python app.py
```

This will start the web server on `http://0.0.0.0:8000`. You can access the application by opening this URL in a web browser.

The application has the following endpoints:

*   `/`: The main index page.
*   `/camera`: Starts the video feed from the default camera.
*   `/recorded`: Starts the video feed from a pre-recorded video file (`videos/recorded.mp4`).
*   `/dashboard`: Displays a log of all recorded violations.
*   `/video_feed`: The video feed itself.

# Development Conventions

*   **Models:** The YOLOv8 models are stored in the root directory (`yolov8n.pt`) and the `models/` directory. The `models/ppe.pt` model is used for PPE detection.
*   **Datasets:** The `datasets/` directory contains the data used to train the PPE detection models.
*   **Violation Logging:** Violations are logged to the `violations.log` file in the root directory. Each line in the log file represents a single violation and includes a timestamp, the source of the video, the ID of the person, and the type of PPE violation.
*   **Testing:** There are several `test_*.py` files that appear to be for testing individual components of the system, such as `test_helmet.py`, `test_mask.py`, and `test_person.py`.
*   **Standalone Scripts:** `ppe_detect.py` appears to be a standalone script for testing the PPE detection model, and not directly used by the main application.
