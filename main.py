import cv2

image = cv2.imread("tank.jpg")
(h, w) = image.shape[:2]

# ex 1
roi = image[0:100, 0:100]  # Lewy górny róg 100x100
cv2.imshow("ROI (100x100)", roi)
cv2.waitKey(0)

# ex 2
lower_half = image[h // 2 : h, :]
cv2.imshow("Dolna połowa", lower_half)
cv2.waitKey(0)

# ex 3
right_half = image[:, w // 2 : w]
cv2.imshow("Prawa połowa", right_half)
cv2.waitKey(0)

# ex 4
startX = int(input("Podaj startX: "))
endX = int(input("Podaj endX: "))
startY = int(input("Podaj startY: "))
endY = int(input("Podaj endY: "))

roi = image[startY:endY, startX:endX]
cv2.imshow("Wybrany ROI", roi)
cv2.waitKey(0)

# ex 5
face_roi = image[100:300, 150:350]  # przybliżony obszar twarzy
cv2.imshow("Twarz", face_roi)
cv2.waitKey(0)

# ex 6
fragment = image[50:150, 50:150]
image[200:300, 200:300] = fragment
cv2.imshow("Obraz z wklejonym fragmentem", image)
cv2.waitKey(0)

# ex 7
cell_h = h // 3
cell_w = w // 3

for i in range(3):
    for j in range(3):
        cell = image[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
        cv2.imshow(f"Part ({i},{j})", cell)
        cv2.waitKey(0)

# ex 8
roi_width = 100

for x in range(0, w - roi_width, 10):
    roi = image[0:h, x : x + roi_width]
    cv2.imshow("Przesuwające ROI", roi)
    key = cv2.waitKey(100)
    if key == ord("q"):
        break

# ex 9
cropped = image[0:300, 0:300]
cv2.imwrite("cropped_image.jpg", cropped)
print("Zapisano jako cropped_image.jpg")

cv2.destroyAllWindows()
