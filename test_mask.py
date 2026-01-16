from ultralytics import YOLO
import cv2

model = YOLO("models/mask.pt")

img = cv2.imread("test_mask.jpg")
results = model(img, conf=0.4)

annotated = results[0].plot()
cv2.imshow("Mask Test", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()
