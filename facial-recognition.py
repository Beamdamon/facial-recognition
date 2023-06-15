from deepface import DeepFace
import matplotlib.pyplot as plt

image1_path = 'photos/image1.jpg'
image2_path = 'photos/image2.jpg'

image1 = DeepFace.detectFace(img_path=image1_path, detector_backend="opencv")
image2 = DeepFace.detectFace(img_path=image2_path, detector_backend="opencv")

analyzedImage = DeepFace.analyze(img_path=image2_path)
print(analyzedImage)

plt.imshow(image2)
