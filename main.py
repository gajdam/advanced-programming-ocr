import numpy as np
import cv2


# ex 1
circle = np.zeros((300, 300), dtype="uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)

triangle = np.zeros((300, 300), dtype="uint8")
points = np.array([[150, 25], [275, 275], [25, 275]])

cv2.drawContours(triangle, [points], 0, 255, -1)
cv2.imshow("Triangle", triangle)

# XOR
bitwiseXor = cv2.bitwise_xor(triangle, circle)
cv2.imshow("XOR", bitwiseXor)

# AND
bitwiseAnd = cv2.bitwise_and(triangle, circle)
cv2.imshow("AND", bitwiseAnd)

# NOT
bitwiseNot = cv2.bitwise_not(triangle)
cv2.imshow("NOT", bitwiseNot)

# OR
bitwiseOr = cv2.bitwise_or(triangle, circle)
cv2.imshow("OR", bitwiseOr)


# ex 2
image1 = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(image1, (50, 50), (250, 250), 255, -1)

image2 = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(image2, (60, 60), (250, 250), 255, -1)
xor_result = cv2.bitwise_xor(image1, image2)

cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.imshow("XOR Result", xor_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
if __name__ == "__main__":
    print("Hello world!")
