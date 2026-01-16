import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Camera opened")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Not geting Frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
