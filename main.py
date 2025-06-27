import cv2
import numpy as np
import imutils
#x 1
image = cv2.imread("kostka.jpg")
resized = imutils.resize(image, width=300)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

for thresh_val in [100, 140, 180]:
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    cv2.imshow(f"Threshold {thresh_val}", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ex 2
image = cv2.imread("kostka.jpg")
resized = imutils.resize(image, width=300)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

modes = [cv2.RETR_EXTERNAL, cv2.RETR_TREE, cv2.RETR_LIST]
mode_names = ["RETR_EXTERNAL", "RETR_TREE", "RETR_LIST"]

for mode, name in zip(modes, mode_names):
    image_copy = resized.copy()
    cnts = cv2.findContours(thresh.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(image_copy, cnts, -1, (0, 0, 255), 2)
    cv2.imshow(f"Contours - {name}", image_copy)
cv2.waitKey(0)

# ex 3
for scale in [1.0, 0.75, 0.5]:
    resized = imutils.resize(image, width=int(image.shape[1] * scale))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    output = resized.copy()
    cv2.drawContours(output, cnts, -1, (0, 0, 255), 2)
    cv2.imshow(f"Contours at scale {scale}", output)
cv2.waitKey(0)

# ex 4
image = cv2.imread("kostka.jpg")
resized = imutils.resize(image, width=300)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

for i, c in enumerate(cnts):
    x, y, w, h = cv2.boundingRect(c)
    roi = resized[y:y+h, x:x+w]
    cv2.putText(resized, str(i+1), (x + w//2, y + h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite(f"kostka_{i+1:02d}.png", roi)

cv2.imshow("Numbered Bricks", resized)
cv2.waitKey(0)

# ex 5
image = cv2.imread("kostka.jpg")
resized = imutils.resize(image, width=300)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(resized, f"{w}x{h}px", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Brick Sizes", resized)
cv2.waitKey(0)

# ex 6
image = cv2.imread("kostka.jpg")
resized = imutils.resize(image, width=300)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

filtered = [c for c in cnts if 500 < cv2.contourArea(c) < 5000]
output = resized.copy()
cv2.drawContours(output, filtered, -1, (0, 255, 255), 2)

cv2.imshow("Filtered Contours", output)
cv2.waitKey(0)

# ex 7
sizes = []
for c in filtered:
    _, _, w, h = cv2.boundingRect(c)
    sizes.append((w, h))

widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
count = len(sizes)

print("Liczba wykrytych kostek:", count)
print("Średnia szerokość:", np.mean(widths))
print("Średnia wysokość:", np.mean(heights))
print("Minimalny rozmiar:", min(widths), "x", min(heights))
print("Maksymalny rozmiar:", max(widths), "x", max(heights))

cv2.destroyAllWindows()
