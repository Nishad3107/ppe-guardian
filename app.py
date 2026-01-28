import cv2
import datetime
import os
from flask import Flask, render_template, Response
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)

person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("models/ppe.pt")

tracker = DeepSort(max_age=30)

cap = None
SOURCE = "CAMERA"

def open_video_stream(source="CAMERA"):
    global cap, SOURCE
    if cap:
        cap.release()
    if source == "CAMERA":
        cap = cv2.VideoCapture(0)
        SOURCE = "CAMERA"
    else:
        cap = cv2.VideoCapture(source)
        SOURCE = source

def check_ppe(person_crop):
    status = {"helmet": False, "mask": False, "glasses": False, "boots": False}
    if person_crop is None or person_crop.size == 0:
        return status

    results = ppe_model(person_crop, conf=0.35)[0]
    if results.boxes is None:
        return status

    for box in results.boxes:
        cls = ppe_model.names[int(box.cls[0])]
        if cls in status:
            status[cls] = True
    return status

def generate_frames():
    global cap

    while True:
        if cap is None:
            continue

        success, frame = cap.read()
        if not success:
            break

        detections = []
        persons = person_model(frame, conf=0.5)[0]

        if persons.boxes:
            for box in persons.boxes:
                if person_model.names[int(box.cls[0])] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            crop = frame[y1:y2, x1:x2]
            ppe = check_ppe(crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            y = y1 + 20
            violation = False

            for k, v in ppe.items():
                txt = f"{k.upper()}: {'OK' if v else 'NO'}"
                col = (0, 255, 0) if v else (0, 0, 255)
                cv2.putText(frame, txt, (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                y += 18
                if not v:
                    violation = True

            if violation:
                with open("violations.log", "a") as f:
                    f.write(
                        f"{datetime.datetime.now()} | SOURCE={SOURCE} | ID={tid} | {ppe}\n"
                    )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    videos = [v for v in os.listdir("videos") if v.endswith(".mp4")]
    return render_template("index.html", videos=videos)

@app.route("/camera")
def camera():
    open_video_stream("CAMERA")
    return render_template("camera.html")

@app.route("/recorded/<filename>")
def recorded(filename):
    open_video_stream(f"videos/{filename}")
    return render_template("camera.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/dashboard")
def dashboard():
    if not os.path.exists("violations.log"):
        return render_template("dashboard.html", logs=[])
    with open("violations.log", "r") as f:
        logs = f.readlines()
    return render_template("dashboard.html", logs=logs)

if __name__ == "__main__":
    open_video_stream()
    app.run(host="0.0.0.0", port=8000, debug=False)