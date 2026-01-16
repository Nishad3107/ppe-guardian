import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

def is_inside(ppe, person):
    px1, py1, px2, py2 = person
    x1, y1, x2, y2 = ppe
    return x1 > px1 and y1 > py1 and x2 < px2 and y2 < py2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    persons = []
    helmets = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person":
                persons.append((x1,y1,x2,y2))
            if label == "hat":  # helmet approx
                helmets.append((x1,y1,x2,y2))

    for p in persons:
        has_helmet = False
        for h in helmets:
            if is_inside(h, p):
                has_helmet = True

        color = (0,255,0) if has_helmet else (0,0,255)
        status = "HELMET OK" if has_helmet else "NO HELMET"

        cv2.rectangle(frame, (p[0],p[1]), (p[2],p[3]), color, 2)
        cv2.putText(frame, status, (p[0], p[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Association Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
