import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

if (len(sys.argv) != 2):
    print("Please enter onne image file")
    exit(1)
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

file = sys.argv[1]

# Read the input image
img = cv2.imread(file)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 10)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()

# Change BGR to RGB
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Make the Mask
mask = np.zeros(img2.shape[:2], np.uint8)
mask[:,:] = 0
rectangle = cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
masked_img = cv2.bitwise_and(img2,img2,mask = rectangle) #mask

# Calculate histogram with mask and without mask
# Check third argument for mask
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],mask,[256],[0,256])
    plt.subplot(224), plt.plot(histr,color = col)
    plt.xlim([0,256])
    
#hist_full = cv2.calcHist([img2],[0],None,[256],[0,256])

plt.subplot(221), plt.imshow(img2, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')

plt.show()