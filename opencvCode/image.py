import cv2

img = cv2.imread(r"D:\playground\Tessolvetrainning\imgs\tigger.jpg")
cv2.imshow("Tigger", img)

cv2.waitKey(10000)
cv2.destroyAllWindows()