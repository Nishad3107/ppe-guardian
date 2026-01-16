import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], 0.9, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue

        track_id = t.track_id
        x1, y1, x2, y2 = map(int, t.to_ltrb())

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(frame, f"ID {track_id}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,0,0), 2)

    cv2.imshow("Person Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
