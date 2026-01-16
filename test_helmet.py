from ultralytics import YOLO
import cv2

model = YOLO("models/helmet.pt")

img = cv2.imread("test_helmet.png")
h, w, _ = frame.shape
head_crop = frame[0:int(h*0.35), :]   # top 35% only

results = model(head_crop, conf=0.6)

for r in results:
    for box in r.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imshow("Helmet Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
