import cv2

image = cv2.imread('1.jpg')

# ex 1
(b,g,r) = image[0,0]
print("Pixel at (0,0) - Red: {}, Green: {}, Blue: {}".format(r,g,b))

# ex 2
# cv2.imshow("Original", image)
# image[-10, -10] = (0, 0, 255)
# cv2.imshow("New image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # ex 3
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
(b,g,r) = image[cX, cY]

# print(f"Pixel at ({cX},{cY}) - Red: {r}, Green: {g}, Blue: {b}")
#
# # ex 4
# x = int(input("X: "))
# y = int(input("Y: "))
#
# if x > w or y > h:
#     print("wartosci wychodza za obszar zdjecia")
# else:
#     image[x, y] = (0, 0, 0)
#     cv2.imshow("Image after user changes", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# ex 5
# first_part = image[:cY, :cX]
# second_part = image[:cY, cX:]
# third_part = image[cY:, :cX]
# fourth_part = image[cY:, cX:]
# cv2.imshow("first", first_part)
# cv2.imshow("second", second_part)
# cv2.imshow("third", third_part)
# cv2.imshow("fourth", fourth_part)

# image[:cY, :cX] = (255, 0,0)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ex 6
image[cY-50:cY + 50,cX-50:cX+50 ] = (0,0,255)
cv2.imshow("image with red", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ex 7
part_height = h // 3
part_width = w // 3

parts = []
for i in range(3):
    row = []
    for j in range(3):
        row.append(image[i * part_height:(i + 1) * part_height, j * part_width:(j + 1) * part_width])
    parts.append(row)

center_crop = parts[1][1]

cv2.imshow("center crop", center_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()