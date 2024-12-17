from fer import FER
import matplotlib.pyplot as plt 
img = plt.imread("happy.jpg")
detector = FER(mtcnn=True)
print(detector.detect_emotions(img))
plt.imshow(img)