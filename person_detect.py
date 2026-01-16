import cv2
from ultralytics import YOLO

# YOLO model load
model = YOLO("yolov8n.pt")

# Camera open
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("✅ Camera + YOLO ready")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, conf=0.5)

    # Results draw
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Sirf PERSON detect karo
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "PERSON", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
