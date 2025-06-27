import cv2
import numpy as np

face_image = cv2.imread("face.jpg")

# ex 1
mask = np.zeros(face_image.shape[:2], dtype="uint8")

center = (face_image.shape[1] // 2, face_image.shape[0] // 2)
axes = (80, 100)
cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

masked_face = cv2.bitwise_and(face_image, face_image, mask=mask)

cv2.imshow("Original Face", face_image)
cv2.imshow("Masked Face", masked_face)
cv2.waitKey(0)

# ex 2
face_image2 = cv2.imread("face.jpg")
mask2 = np.ones(face_image2.shape, dtype="uint8") * 255

cv2.rectangle(mask2, (120, 100), (220, 140), (0, 0, 0), -1)
hidden_eyes = cv2.bitwise_and(face_image2, mask2)

cv2.imshow("Face with Eyes Hidden", hidden_eyes)
cv2.waitKey(0)

# ex 3
color_image = cv2.imread("color_image.jpg")  # np. kwiaty, samochód
hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)

extracted = cv2.bitwise_and(color_image, color_image, mask=mask_red)

cv2.imshow("Original Color Image", color_image)
cv2.imshow("Red Color Extracted", extracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
