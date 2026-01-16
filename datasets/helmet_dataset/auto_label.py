from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")  # pretrained COCO model

IMAGE_DIR = "images/train"
LABEL_DIR = "labels/train"

os.makedirs(LABEL_DIR, exist_ok=True)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    results = model(img, conf=0.4)

    label_path = os.path.join(LABEL_DIR, img_name.replace(".png", ".txt").replace(".jpg", ".txt"))

    with open(label_path, "w") as f:
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]

                # COCO has "person" and "hat" types; we treat "person" with head region as helmet proxy
                if name in ["person", "hat"]:
                    x1, y1, x2, y2 = box.xyxy[0]
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    # class 0 = helmet (proxy)
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
