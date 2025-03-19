import cv2
import imutils
import numpy as np

image = cv2.imread('tank.jpg')
cv2.imshow('Original', image)

# ex 1
M = np.float32([[1, 0, 30], [0, 1, 40]]) # ujemne w lewo i górę
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('shifted', shifted)

# ex 2
M = np.float32([[1, 0, -20], [0, 1, -50]]) # ujemne w lewo i górę
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('shifted2', shifted)

# ex 3

(h, w) = image.shape[:2]
M = np.float32([[1, 0, w // 2 + 1], [0, 1, h // 2 + 1]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('shifted3', shifted)

# ex 4
shifted = imutils.translate(image, 50, 100)
cv2.imshow('shifted4', shifted)

# ex 5

x = int(input("X: "))
y = int(input("Y: "))

shifted = imutils.translate(image, x, y)
cv2.imshow('shifted5', shifted)

cv2.waitKey(0)
cv2.destroyAllWindows()
