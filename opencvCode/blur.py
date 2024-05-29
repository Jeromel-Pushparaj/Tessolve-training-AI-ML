import cv2
img=cv2.imread('ex1.jpg')
gm=cv2.GaussianBlur(img,(41,41),0)
gm1=cv2.GaussianBlur(img,(21,21),0)
cv2.imshow("org",img)
cv2.imshow("gm",gm)
cv2.imshow("gm1",gm)
