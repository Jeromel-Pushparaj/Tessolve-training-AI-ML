import cv2

img = cv2.imread("tigger.jpg")
cv2.imshow("Tigger", img)

cv2.waitKey(10000)