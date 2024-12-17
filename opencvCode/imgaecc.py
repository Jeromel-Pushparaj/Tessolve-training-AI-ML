import cv2
img =cv2.imread(r"D:\playground\Tessolvetrainning\imgs\car image.jpg")
cv2.imshow('car',img)
cv2.imwrite('photo.jpg',img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
