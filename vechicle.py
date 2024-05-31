import cv2
haar_cascade = 'cars.xml'
	
cap = cv2.VideoCapture('videoplayback.mp4') 
car_cascade = cv2.CascadeClassifier(haar_cascade)

ret, frames = cap.read() 
		
 
gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
		

cars = car_cascade.detectMultiScale(gray, 1.1, 1)

for (x,y,w,h) in cars: 
	cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
	

cv2.imshow('video', frames)

while True: 

	ret, frames = cap.read() 
		
 
	gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
		

	cars = car_cascade.detectMultiScale(gray, 1.1, 1) 
		

	for (x,y,w,h) in cars: 
		cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
	

	cv2.imshow('video', frames) 
		

	if cv2.waitKey(33) == 27: 
		break
	

cv2.destroyAllWindows()
