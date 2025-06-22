import cv2
import imutils

image = cv2.imread('tank.jpg')

# ex 1
flipped_hor = cv2.flip(image, 1)
cv2.imshow('flipped', flipped_hor)
cv2.waitKey(0)

# ex 2
flipped_vert = cv2.flip(image, 0)
cv2.imshow('flipped', flipped_vert)
cv2.waitKey(0)

# ex 3
flipped_both = cv2.flip(image, -1)
cv2.imshow('flipped', flipped_both)
cv2.waitKey(0)

# ex 4
cv2.imshow('original', image)
cv2.imshow('flipped_hor', flipped_hor)
cv2.imshow('flipped_vert', flipped_vert)
cv2.imshow('flipped_both', flipped_both)
cv2.waitKey(0)

# ex 5
(h, w) = image.shape[:2]

startX, endX = w // 3, 2 * w // 3
startY, endY = h // 3, 2 * h // 3
roi = image[startY:endY, startX:endX]

flipped_roi = cv2.flip(roi, 1)
image[startY:endY, startX:endX] = flipped_roi
cv2.imshow("Flipped ROI Image", image)
cv2.waitKey(0)

# ex 6

flip_mode = int(input("Podaj sposób odbicia (0 – pionowe, 1 – poziome, -1 – oba): "))

flipped = cv2.flip(image, flip_mode)

cv2.imshow("Flipped Image", flipped)
cv2.waitKey(0)

cv2.destroyAllWindows()