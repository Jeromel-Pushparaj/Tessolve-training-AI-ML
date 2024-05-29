import cv2
img = cv2.imread("ex1.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresholdImg = cv2.threshold(grayImg,160,255,cv2.THRESH_BINARY)[1]

cv2.imshow("origi", img)
cv2.imshow("Threshold.jpg",thresholdImg)
