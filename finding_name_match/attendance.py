import cv2, numpy, os
from datetime import datetime

# Path to Haar Cascade XML file
haar_file = 'D:\\playground\\Tessolve trainning\\finding_name_match\\haarcascade_frontalface_default.xml'
datasets = 'D:\\playground\\Tessolve trainning\\finding_name_match\\datasets'

# Initialize variables
(images, labels, names, id) = ([], [], {}, 0)

# Load images and labels
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            try:
                images.append(cv2.imread(path, 0))
                labels.append(int(id))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        id += 1

# Convert images and labels to NumPy arrays
(width, height) = (130, 100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# Create and train the model
try:
    model = cv2.face.FisherFaceRecognizer_create()
    model.train(images, labels)
except Exception as e:
    print(f"Error creating or training the model: {e}")
    exit()

# Load the Haar Cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(haar_file)
except Exception as e:
    print(f"Error loading Haar Cascade classifier: {e}")
    exit()

# Function to mark attendance
def markAttendance(name):
    try:
        # Check if the file exists, and create it if it does not
        if not os.path.isfile('attendance.csv'):
            with open('attendance.csv', 'w') as f:
                f.write('Name,DateTime\n')  # Write headers to the file

        with open('attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
    except Exception as e:
        print(f"Error marking attendance: {e}")

# Initialize webcam
try:
    webcam = cv2.VideoCapture(0)
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit()

cnt = 0
while True:
    (_, im) = webcam.read()
    if im is None:
        print("Error capturing image from webcam")
        break
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        # Predict the face
        try:
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1] < 800:
                name = names[prediction[0]]
                cv2.putText(im, f'{name} - {prediction[1]:.0f}', (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
                print(name)
                markAttendance(name)
                cnt = 0
            else:
                cnt += 1
                cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                if cnt > 100:
                    print("Unknown Person")
                    cv2.imwrite("unknown.jpg", im)
                    cnt = 0
        except Exception as e:
            print(f"Error during face prediction: {e}")
    
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # Press 'ESC' to exit
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()