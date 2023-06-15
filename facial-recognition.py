from deepface import DeepFace
import matplotlib.pyplot as plt

#The path of the images
image1_path = 'photos/image1.jpg'
image2_path = 'photos/image2.jpg'

#Detects the images using Deepface and their paths
image1 = DeepFace.detectFace(img_path=image1_path, detector_backend="opencv")
image2 = DeepFace.detectFace(img_path=image2_path, detector_backend="opencv")

#Analyzes the image for dominant emotion, gender, and race
analyzedImage = DeepFace.analyze(img_path=image2_path)
print(analyzedImage)

#Prints the image in the interactive window
plt.imshow(image2)
