import cv2
alg = "D:\\playground\\Tessolve trainning\\face_detection\\haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg)
cam  = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y), (x+w,y+h), (235,20,0),5)
        center_x = int(x + w // 2)
        center_y = int(y + h // 2)
        radius = int(min(w // 2, h // 2))
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 10), 4)
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()