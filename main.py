import cv2
import numpy as np
import imutils

# ex 1
image = cv2.imread("fanta_bottle.jpg")
template = cv2.imread("fanta_logo.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
(h, w) = template.shape[:2]
cv2.rectangle(image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)

print("Współrzędne wykrycia:", max_loc)
print("Wartość dopasowania:", max_val)
cv2.imshow("Detected Logo", image)
cv2.waitKey(0)

# ex 2
for angle in [30, 45]:
    rotated = imutils.rotate(image_gray, angle)
    result = cv2.matchTemplate(rotated, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"Kąt: {angle} stopni, maxVal: {max_val}")

# ex 3
for scale in [0.5, 1.5]:
    scaled = cv2.resize(image_gray, (0, 0), fx=scale, fy=scale)
    result = cv2.matchTemplate(scaled, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"Skala: {scale}, maxVal: {max_val}")

# ex 4
methods = [
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED
]
method_names = [
    "TM_CCOEFF", "TM_CCOEFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_SQDIFF", "TM_SQDIFF_NORMED"
]

for method, name in zip(methods, method_names):
    result = cv2.matchTemplate(image_gray, template_gray, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else max_loc
    detected = image.copy()
    cv2.rectangle(detected, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 2)
    print(f"{name} => maxVal: {max_val if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else min_val}")
    cv2.imshow(f"Method: {name}", detected)
    cv2.waitKey(0)

# ex 5
screenshot = cv2.imread("interface.png")
template = cv2.imread("icon.png")
screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)
h, w = template.shape[:2]
cv2.rectangle(screenshot, max_loc, (max_loc[0] + w, max_loc[1] + h), (255, 0, 0), 2)
cv2.imshow("Icon Detection", screenshot)
cv2.waitKey(0)

# ex 6
image = cv2.imread("lego_scene.jpg")
template = cv2.imread("lego_block.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
thresh = 0.8
locs = np.where(result >= thresh)

for pt in zip(*locs[::-1]):
    cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 255), 2)

cv2.imshow("Detected Multiple Templates", image)
cv2.waitKey(0)

# ex 7
image = cv2.imread("scene.jpg")
template = cv2.imread("object.jpg")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    roi = image_gray[y:y+h, x:x+w]
    if roi.shape[0] >= template_gray.shape[0] and roi.shape[1] >= template_gray.shape[1]:
        res = cv2.matchTemplate(roi, template_gray, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if val > 0.7:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Contours + Matching", image)
cv2.waitKey(0)
cv2.destroyAllWindows()