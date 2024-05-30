import cv2
img = cv2.imread("imgs//tigger.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresholdImg = cv2.threshold(grayImg,130,225,cv2.THRESH_BINARY)[1]

cv2.imshow("origi", img)
cv2.imshow("Threshold.jpg",thresholdImg)

cv2.waitKey(10000)
cv2.destroyAllWindows()
