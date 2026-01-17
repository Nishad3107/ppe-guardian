import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

# ================= CONFIG =================
SOURCE_MODE = "video"          # "camera" or "video"
VIDEO_PATH = "videos/test-video.mp4"
FRAME_SKIP = 2
HISTORY_LEN = 15
THRESHOLD = 0.6
# =========================================

# ================= MODELS =================
person_model  = YOLO("yolov8n.pt")
helmet_model  = YOLO("models/helmet.pt")
mask_model    = YOLO("models/mask.pt")
glasses_model = YOLO("models/glasses.pt")
boots_model   = YOLO("models/boots.pt")

# ================= APP ====================
app = Flask(__name__)
tracker = DeepSort(max_age=30)

# ================= VIDEO ==================
def get_capture():
    cap = cv2.VideoCapture(VIDEO_PATH if SOURCE_MODE=="video" else 0)
    if not cap.isOpened():
        raise RuntimeError("Camera / Video not found")
    return cap

cap = get_capture()

# ============ TEMPORAL MEMORY =============
ppe_history = {}
frame_count = 0

def stable(history):
    if len(history) == 0:
        return False
    return sum(history) / len(history) >= THRESHOLD

# ================= STREAM =================
def generate_frames():
    global frame_count

    while True:
        ret, frame = cap.read()
        if not ret:
            if SOURCE_MODE=="video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_count += 1

        # -------- PERSON DETECTION --------
        detections = []
        persons = person_model(frame, conf=0.5)

        for r in persons:
            for b in r.boxes:
                if person_model.names[int(b.cls[0])] == "person":
                    x1,y1,x2,y2 = map(int,b.xyxy[0])
                    detections.append(([x1,y1,x2-x1,y2-y1],0.9,"person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            x1,y1,x2,y2 = map(int,t.to_ltrb())
            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                continue

            h = person.shape[0]
            head  = person[0:int(h*0.3), :]
            face  = person[int(h*0.25):int(h*0.55), :]
            boots = person[int(h*0.65):h, :]

            if frame_count % FRAME_SKIP == 0:
                helmet_raw  = any(len(r.boxes)>0 for r in helmet_model(head,  conf=0.4))
                mask_raw    = any(len(r.boxes)>0 for r in mask_model(face,    conf=0.4))
                glasses_raw = any(len(r.boxes)>0 for r in glasses_model(face, conf=0.4))
                boots_raw   = any(len(r.boxes)>0 for r in boots_model(boots,  conf=0.4))

                ppe_history.setdefault(tid,{
                    "helmet":deque(maxlen=HISTORY_LEN),
                    "mask":deque(maxlen=HISTORY_LEN),
                    "glasses":deque(maxlen=HISTORY_LEN),
                    "boots":deque(maxlen=HISTORY_LEN)
                })

                ppe_history[tid]["helmet"].append(helmet_raw)
                ppe_history[tid]["mask"].append(mask_raw)
                ppe_history[tid]["glasses"].append(glasses_raw)
                ppe_history[tid]["boots"].append(boots_raw)

            hist = ppe_history[tid]

            helmet_ok  = stable(hist["helmet"])
            mask_ok    = stable(hist["mask"])
            glasses_ok = stable(hist["glasses"])
            boots_ok   = stable(hist["boots"])

            # -------- TEXT & COLORS --------
            def status(txt, ok):
                return (f"{txt}: OK", (0,255,0)) if ok else (f"{txt}: NO", (0,0,255))

            labels = [
                status("Helmet", helmet_ok),
                status("Mask", mask_ok),
                status("Glasses", glasses_ok),
                status("Boots", boots_ok)
            ]

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)

            y_offset = y1 - 10
            for text,color in labels:
                cv2.putText(frame,
                            text,
                            (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)
                y_offset -= 20

            cv2.putText(frame,
                        f"ID {tid}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ================= MAIN ===================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)