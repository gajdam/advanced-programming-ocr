import cv2
import numpy as np
import imutils

image = cv2.imread('tank.jpg')
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)

# ex 1
resized = cv2.resize(image, (cX, cY), interpolation=cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)

# ex 2
w2 = w * 2
h2 = h * 2
resized = cv2.resize(image, (w2, h2), interpolation=cv2.INTER_LINEAR)
cv2.imshow("resized", resized)
cv2.waitKey(0)

# ex 3
resized = cv2.resize(image, (200, 300), interpolation=cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)

# ex 4
w3 = w * 3
h3 = h * 3
resized_linear = cv2.resize(image, (w3, h3), interpolation=cv2.INTER_LINEAR)
resized_nearest = cv2.resize(image, (w3, h3), interpolation=cv2.INTER_NEAREST)
resized_cubic = cv2.resize(image, (w3, h3), interpolation=cv2.INTER_CUBIC)
resized_lanczos4 = cv2.resize(image, (w3, h3), interpolation=cv2.INTER_LANCZOS4)
cv2.imshow("resized_linear", resized_linear)
cv2.imshow("resized_nearest", resized_nearest)
cv2.imshow("resized_cubic", resized_cubic)
cv2.imshow("resized_lanczos4", resized_lanczos4)
cv2.waitKey(0)

# ex 5
resized = imutils.resize(image, width=500)
cv2.imshow("resized width", resized)
cv2.waitKey(0)

# ex 6
resized = imutils.resize(image, height=400)
cv2.imshow("resized height", resized)
cv2.waitKey(0)

# ex 7
w4 = w // 5
h4 = h // 5

resized = cv2.resize(image, (w4, h4), interpolation=cv2.INTER_LINEAR)
cv2.imshow("resized", resized)
cv2.waitKey(0)

# ex 8
new_dim = (w // 5, h // 5)
resized_area = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Task8", resized_area)
cv2.waitKey(0)

# ex 9
for scale in np.arange(1.0, 3.01, 0.2):
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(f"Task9 - {int(scale*100)}%", resized)
    cv2.waitKey(500)

# ex 10
resized_800 = imutils.resize(image, width=800)
cv2.imwrite('resized_output.jpg', resized_800)

cv2.destroyAllWindows()
