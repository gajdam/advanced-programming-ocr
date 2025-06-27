import cv2
import numpy as np

image = cv2.imread("tank.jpg")

# ex 1
added_np = image + 50

M = np.ones(image.shape, dtype="uint8") * 50
added_cv = cv2.add(image, M)

cv2.imshow("Original", image)
cv2.imshow("NumPy Addition", added_np)
cv2.imshow("OpenCV Addition", added_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ex 2
bright_np = image + 150
M150 = np.ones(image.shape, dtype="uint8") * 150
bright_cv = cv2.add(image, M150)

cv2.imshow("NumPy 150+", bright_np)
cv2.imshow("OpenCV 150+", bright_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ex 3
dark_np = image - 80
M80 = np.ones(image.shape, dtype="uint8") * 80
dark_cv = cv2.subtract(image, M80)

cv2.imshow("NumPy -80", dark_np)
cv2.imshow("OpenCV -80", dark_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ex 4
(B, G, R) = cv2.split(image)
R = cv2.add(R, 30)
G = cv2.subtract(G, 20)
B = cv2.add(B, 10)
filtered = cv2.merge([B, G, R])

cv2.imshow("Instagram Filter", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ex 5
image1 = cv2.imread("tank.jpg")
image2 = cv2.imread("tank1.jpg")

if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

diff = cv2.absdiff(image1, image2)

cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.imshow("Difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()