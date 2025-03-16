import cv2
import imutils

image = cv2.imread('OIP.jpg')

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

