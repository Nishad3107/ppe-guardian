import cv2
from ultralytics import YOLO

helmet = YOLO("models/helmet.pt")
cap = cv2.VideoCapture("videos/test-video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    h = frame.shape[0]
    head = frame[0:int(h*0.35), :]

    res = helmet(head, conf=0.3)
    if any(len(r.boxes) > 0 for r in res):
        cv2.putText(frame,"HELMET DETECTED",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.imshow("Helmet Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
