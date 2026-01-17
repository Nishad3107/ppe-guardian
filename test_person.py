import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videos/test-video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, conf=0.5)
    for r in results:
        for b in r.boxes:
            if model.names[int(b.cls[0])] == "person":
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,"PERSON",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("Person Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
