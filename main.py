import cv2

image = cv2.imread('OIP.jpg')
image_gray = cv2.imread('OIP.jpg', cv2.IMREAD_GRAYSCALE)

print(f"color image channels: {len(image.shape[:3])}")
print(f"gray image channels: {len(image_gray.shape[:3])}")

cv2.imwrite("OIP_gray.jpg", image_gray)

cv2.imshow('image_color', image)
cv2.resizeWindow('image_color', 500, 500)

cv2.imshow('image_gray', image_gray)
cv2.namedWindow('image_gray', cv2.WINDOW_NORMAL)

cv2.waitKey(0)
cv2.destroyAllWindows()
