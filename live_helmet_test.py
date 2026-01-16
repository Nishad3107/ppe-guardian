import cv2
from ultralytics import YOLO

model = YOLO("models/helmet.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Live Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
