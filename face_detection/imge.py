import cv2
 
img = cv2.imread('D:\\playground\\Tessolve trainning\\imgs\\parrot.jpg')

cv2.imshow('show',img)
 
 
cv2.imwrite('photo.jpg',img)
 
 
cv2.waitKey(10000) 
cv2.destroyAllWindows()