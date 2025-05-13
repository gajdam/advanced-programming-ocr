import cv2
import numpy as np

# task 1:
canvas1 = np.zeros((300, 300, 3), dtype="uint8")
h, w = canvas1.shape[:2]
center = (w // 2, h // 2)
blue = (255, 0, 0)
cv2.line(canvas1, center, (w - 1, h - 1), blue, 2)
cv2.imshow("Task1", canvas1)
cv2.waitKey(0)

# task 2:
canvas2 = np.zeros((400, 400, 3), dtype="uint8")

# 2a
green = (0, 255, 0)
cv2.rectangle(canvas2, (0, 0), (100, 50), green, -1)

# 2b
red = (0, 0, 255)
start_pt = (400 - 100, 400 - 50)
end_pt = (399, 399)
cv2.rectangle(canvas2, start_pt, end_pt, red, 3)

cv2.imshow("task2", canvas2)
cv2.waitKey(0)

# task 3
canvas3 = np.zeros((300, 300, 3), dtype="uint8")

# 3a
cv2.circle(canvas3, (40, 40), 40, blue, 2)

# 3b
cv2.circle(canvas3, (canvas3.shape[1] // 2, canvas3.shape[0] // 2), 60, red, 2)

cv2.imshow("Task3", canvas3)
cv2.waitKey(0)

# task 4
canvas4 = np.zeros((300, 300, 3), dtype="uint8")

sq_size = 100
top_left = ((300 - sq_size) // 2, (300 - sq_size) // 2)
bottom_right = ((300 + sq_size) // 2, (300 + sq_size) // 2)
cv2.rectangle(canvas4, top_left, bottom_right, green, 2)

center = (300 // 2, 300 // 2)
cv2.circle(canvas4, center, 30, blue, 2)

cv2.imshow("Task4", canvas4)
cv2.waitKey(0)

# task 5
canvas5 = np.zeros((300, 300, 3), dtype="uint8")
center = (canvas5.shape[1] // 2, canvas5.shape[0] // 2)

half = 10
while half <= 150:
    tl = (center[0] - half, center[1] - half)
    br = (center[0] + half, center[1] + half)
    cv2.rectangle(canvas5, tl, br, red, 1)
    half += 10

cv2.imshow("Task5", canvas5)
cv2.waitKey(0)

# task 6
profile = cv2.imread("profile.jpg")

gray = cv2.cvtColor(profile, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for x, y, w, h in faces:

    roi_gray = gray[y: y + h, x: x + w]
    roi_color = profile[y: y + h, x: x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for ex, ey, ew, eh in eyes:
        center_eye = (x + ex + ew // 2, y + ey + eh // 2)
        radius = int(max(ew, eh) / 2)
        cv2.circle(profile, center_eye, radius, (0, 0, 255), -1)

    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
    for sx, sy, sw, sh in smiles:
        cv2.rectangle(
            profile, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), -1
        )

    center_face = (x + w // 2, y + h // 2)
    radius_face = int(0.6 * max(w, h))
    cv2.circle(profile, center_face, radius_face, (255, 0, 0), 2)
    break

cv2.imshow("Task6", profile)
cv2.waitKey(0)
cv2.destroyAllWindows()
