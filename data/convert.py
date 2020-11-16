import cv2
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def detect_face(file,filename, images_delete):
    directory = r'C:\Users\Rob\Documents\College\ECE 588\FinalProject588\data\cropped'
    img = cv2.imread(file)
    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(color, 1.1, 10)


    if type(faces) == tuple:
        print("No face detected!")
        images_delete.append(filename)

    else:
        for (x, y, w, h) in faces:
            cropped = img[y:y+h, x:x+w]
            print(filename)
            cv2.imwrite(directory + '\\' + filename, cropped)
    return images_delete


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
images_delete = []
directory = r'C:\Users\Rob\Documents\College\ECE 588\FinalProject588\data\raw'

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        images_delete = detect_face(os.path.join(directory, filename), filename, images_delete)
    else:
        images_delete.append(filename)

print(images_delete)