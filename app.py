import cv2
import datetime
from flask import Flask, render_template, Response
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

person_model   = YOLO("yolov8n.pt")
helmet_model   = YOLO("models/helmet.pt")
mask_model     = YOLO("models/mask.pt")
glasses_model  = YOLO("models/glasses.pt")
boots_model    = YOLO("models/boots.pt")


app = Flask(__name__)


person_model = YOLO("yolov8n.pt")
helmet_model = YOLO("models/helmet.pt")

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)

def has_helmet(person_crop):
    if person_crop.size == 0:
        return False

    h, _, _ = person_crop.shape
    head_crop = person_crop[0:int(h * 0.25), :]

    results = helmet_model(head_crop, conf=0.7)
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            return True
    return False



def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        detections = []
        results = person_model(frame, conf=0.5)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if person_model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())

            person_crop = frame[y1:y2, x1:x2]
            helmet_present = has_helmet(person_crop)

            if helmet_present:
                color = (0, 255, 0)
                status = "HELMET OK"
            else:
                color = (0, 0, 255)
                status = "NO HELMET"

                with open("violations.log", "a") as f:
                    f.write(
                        f"ID {track_id} | NO HELMET | {datetime.datetime.now()}\n"
                    )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID {track_id} - {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
