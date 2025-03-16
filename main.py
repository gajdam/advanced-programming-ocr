import cv2
import numpy as np
import imutils

image = cv2.imread('OIP.jpg')
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)

# ex 1
M = cv2.getRotationMatrix2D((cX, cY), 45, 1)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated 45", rotated)
cv2.waitKey(0)

# ex 2
M = cv2.getRotationMatrix2D((cX, cY), -90, 1)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated -90", rotated)
cv2.waitKey(0)

# ex 3
M = cv2.getRotationMatrix2D((0,0), 30, 1)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated by counter", rotated)
cv2.waitKey(0)

# ex 4
# alfa = int(input("Kąt: "))
# M = cv2.getRotationMatrix2D((cX, cY), alfa, 1)
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow(f"Rotated by {alfa}", rotated)
# cv2.waitKey(0)

# ex 5
rotated = imutils.rotate(image, 180)
cv2.imshow("Rotated 180", rotated)
cv2.waitKey(0)

# ex 6
rotated = imutils.rotate_bound(image, -33)
cv2.imshow("Rotated bound -33", rotated)
cv2.waitKey(0)

# ex 7
rotated_im = imutils.rotate(image, 60)
M = cv2.getRotationMatrix2D((cX, cY), 60, 1)
rotated_m = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated_m - 60", rotated_m)
cv2.imshow("Rotated_im - 60", rotated_im)
cv2.waitKey(0)

# ex 8
rotated = imutils.rotate(image, 30)
rotated = imutils.rotate(rotated, 30)
rotated = imutils.rotate(rotated, 30)

roated2 = imutils.rotate(image, 90)
cv2.imshow("Rotated - 30x3", rotated)
cv2.imshow("Rotated - 90", roated2)
cv2.waitKey(0)

# ex 9
rotated = imutils.rotate(rotated, 75)
cv2.imwrite("rotated_output.jpg", rotated)

# ex 10
for i in range(0, 375, 15):
    rotated = imutils.rotate(image, i)
    cv2.imshow(f"Rotated Image", rotated)
    cv2.waitKey(500)

