import cv2

image = cv2.imread("tank.jpg")

# ex 1
(b, g, r) = image[0, 0]
print("Pixel at (0,0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

# ex 2
cv2.imshow("Original", image)
image[-10, -10] = (0, 0, 255)
cv2.imshow("New image", image)
cv2.waitKey(0)

# # ex 3
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
(b, g, r) = image[cX, cY]

print(f"Pixel at ({cX},{cY}) - Red: {r}, Green: {g}, Blue: {b}")

# # ex 4
x = int(input("X: "))
y = int(input("Y: "))

if x > w or y > h:
    print("wartosci wychodza za obszar zdjecia")
else:
    image[x, y] = (0, 0, 0)
    cv2.imshow("Image after user changes", image)
    cv2.waitKey(0)

# ex 5
first_part = image[:cY, :cX]
second_part = image[:cY, cX:]
third_part = image[cY:, :cX]
fourth_part = image[cY:, cX:]
cv2.imshow("first", first_part)
cv2.imshow("second", second_part)
cv2.imshow("third", third_part)
cv2.imshow("fourth", fourth_part)

image[:cY, :cX] = (255, 0, 0)
cv2.imshow("image", image)
cv2.waitKey(0)

# ex 6
image[cY - 50: cY + 50, cX - 50: cX + 50] = (0, 0, 255)
cv2.imshow("image with red", image)
cv2.waitKey(0)

# ex 7
part_height = h // 3
part_width = w // 3

parts = []
for i in range(3):
    row = []
    for j in range(3):
        row.append(
            image[
                i * part_height: (i + 1) * part_height,
                j * part_width: (j + 1) * part_width,
            ]
        )
    parts.append(row)

center_crop = parts[1][1]

cv2.imshow("center crop", center_crop)
cv2.waitKey(0)


image = cv2.imread("tank.jpg")
(h, w) = image.shape[:2]

# ex8
before_row = image.copy()
cv2.imshow("Before – wiersz 100", before_row)

if 0 <= 100 < h:
    image[100, :] = (0, 255, 0)
else:
    print("Obraz ma mniej niż 101 wierszy!")

cv2.imshow("After – wiersz 100 na zielono", image)
cv2.waitKey(0)

# ex9
image = cv2.imread("tank.jpg")
before_rect = image.copy()
cv2.imshow("Before – prostokat", before_rect)

x1, y1, x2, y2 = 50, 50, 100, 100
if x2 < w and y2 < h:
    image[y1:y2, x1:x2] = (255, 255, 255)
else:
    print("Zakres prostokąta wykracza poza wymiary obrazu!")

cv2.imshow("After – prostokat (50,50)-(100,100) na biało", image)
cv2.waitKey(0)

# ex10
image = cv2.imread("tank.jpg")

pts = [(50, 50), (200, 200)]
vals = []
for x, y in pts:
    if x < w and y < h:
        b, g, r = image[y, x]
        vals.append((r, g, b))
        print(f"Pixel ({x},{y}) – R: {r}, G: {g}, B: {b}")
    else:
        print(f"Punkt ({x},{y}) poza obszarem obrazu!")

if len(vals) == 2:
    diff = tuple(vals[0][i] - vals[1][i] for i in range(3))
    print(f"Różnice (R,G,B) = ({diff[0]}, {diff[1]}, {diff[2]})")


image = cv2.imread("tank.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

x_max, y_max = maxLoc
b_max, g_max, r_max = image[y_max, x_max]

print(f"Najjaśniejszy piksel w ({x_max},{y_max}) = {maxVal}")
print(f"(BGR): ({b_max}, {g_max}, {r_max})")
cv2.destroyAllWindows()
