import cv2
image = cv2.imread("imgs\\parrot.jpg")
g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', image)
cv2.imshow('Gray', g)
# cv2.imwrite('graynew.jpg', g)
cv2.waitKey(10000)

print(image.shape)
