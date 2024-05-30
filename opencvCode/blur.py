import cv2
img=cv2.imread('D:\\playground\\Tessolve trainning\\imgs\\car image.jpg')
gm=cv2.GaussianBlur(img,(41,41),0)
gm1=cv2.GaussianBlur(img,(11,11),0)

cv2.imshow("org",img)
cv2.imshow("gm",gm)
cv2.imshow("gm1",gm)
cv2.waitKey(10000)
