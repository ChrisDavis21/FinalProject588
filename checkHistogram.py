# takes input image and outputs average of histogram comparisons

import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.cluster import KMeans

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
#for (x, y, w, h) in faces:
   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
   #print(1)
x = faces[0][0]
y =faces[0][1]
w = faces[0][2]
h = faces[0][3]
# Display the output
cv2.waitKey()

# Change BGR to RGB
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
anewh = int(h/2)
anewy = int(y+anewh)
xscalefactor = int(0.1*(x+w)) #Reduces Horizontal Window by 20%

# Make the Mask
mask = np.zeros(img2.shape[:2], np.uint8)
mask[:,:] = 0
rectangle = cv2.rectangle(mask,(x+xscalefactor,y),(x+w-xscalefactor,y+(anewh)),(255,255,255),-1)
masked_img = cv2.bitwise_and(img2,img2,mask = rectangle) #mask

# Calculate histogram with mask and without mask
# Check third argument for mask
hist1r = cv2.calcHist([img2],[0],mask,[256],[0,256])
hist1g = cv2.calcHist([img2],[1],mask,[256],[0,256])
hist1b = cv2.calcHist([img2],[2],mask,[256],[0,256])
    
  
mask2 = np.zeros(img2.shape[:2], np.uint8)
mask2[:,:] = 0
rectangle2 = cv2.rectangle(mask2,(x+xscalefactor,anewy),(x+w-xscalefactor,y+h),(255,255,255),-1)
masked_img2 = cv2.bitwise_and(img2,img2,mask = rectangle2) #mask

# Calculate histogram with mask and without mask
# Check third argument for mask

hist2r = cv2.calcHist([img2],[0],mask2,[256],[0,256])
hist2g = cv2.calcHist([img2],[1],mask2,[256],[0,256])
hist2b = cv2.calcHist([img2],[2],mask2,[256],[0,256])

# CompareHistograms
redVal = cv2.compareHist(hist1r, hist2r, cv2.HISTCMP_CORREL)
greenVal = cv2.compareHist(hist1g, hist2g, cv2.HISTCMP_CORREL)  
blueVal = cv2.compareHist(hist1b, hist2b, cv2.HISTCMP_CORREL)    
#print(redVal)
#print(greenVal)
#print(blueVal)
#print( )
mean = (redVal + greenVal + blueVal) / 3

print(mean)
