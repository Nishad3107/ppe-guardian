import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

print("✅ PPE Detection Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Color logic
            color = (0, 255, 0)
            if "no_" in label:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("PPE Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
