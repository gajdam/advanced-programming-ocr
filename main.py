import cv2
import numpy as np

image = cv2.imread("tank.jpg")
B, G, R = cv2.split(image)

# ex 1
cv2.imshow("Blue Channel", B)
cv2.imshow("Green Channel", G)
cv2.imshow("Red Channel", R)

cv2.imwrite("blue_channel.jpg", B)
cv2.imwrite("green_channel.jpg", G)
cv2.imwrite("red_channel.jpg", R)
cv2.waitKey(0)

# ex 2
image = cv2.imread("tank.jpg")
B, G, R = cv2.split(image)

cv2.imshow("Channel B", B)
cv2.imshow("Channel G", G)
cv2.imshow("Channel R", R)
cv2.waitKey(0)

# ex 3
RBG = cv2.merge([R, B, G])
G_zero = cv2.merge([B, np.zeros_like(G), R])

cv2.imshow("RBG Order", RBG)
cv2.imshow("Green Channel Zeroed", G_zero)
cv2.waitKey(0)

# ex 4
R_boosted = cv2.add(R, 50)
boosted_image = cv2.merge([B, G, R_boosted])

cv2.imshow("Red Boosted", boosted_image)
cv2.waitKey(0)

# ex 5
image = cv2.imread("tank.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = (0, 100, 100)
upper_red = (10, 255, 255)
mask = cv2.inRange(hsv, lower_red, upper_red)

B, G, R = cv2.split(image)
R = cv2.add(R, 50)
merged = cv2.merge([B, G, R])
selective_red = cv2.bitwise_and(merged, merged, mask=mask)

cv2.imshow("Selective Red Enhancement", selective_red)
cv2.waitKey(0)

# ex 6
logo = cv2.imread("opencv_logo.png")
B, G, R = cv2.split(logo)

swapped = cv2.merge([R, G, B])
no_green = cv2.merge([B, np.zeros_like(G), R])

cv2.imshow("Original Logo", logo)
cv2.imshow("Swapped B <-> R", swapped)
cv2.imshow("No Green", no_green)
cv2.waitKey(0)
cv2.destroyAllWindows()