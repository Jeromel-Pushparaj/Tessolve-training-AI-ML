import cv2
image = cv2.imread("ex.jpg")
g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', image)
cv2.imshow('Gray', g)
cv2.imwrite('graynew.jpg', g)

print(image.shape)
