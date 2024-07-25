import cv2
import imutils

img = cv2.imread('D:\\playground\\Tessolve trainning\\imgs\\tigger.jpg')
rs = imutils.resize(img, width=50)
cv2.imshow('oriimg', img)
cv2.imshow('rs.jpeg',rs)

cv2.imwrite('rs.jpg',rs)

cv2.waitKey(10000)
cv2.destroyAllWindows()
