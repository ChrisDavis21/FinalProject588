import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Input data
file = sys.argv[1]
img = cv2.imread(file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 10)

x = faces[0][0]
y =faces[0][1]
w = faces[0][2]
h = faces[0][3]
# Display the output
cv2.waitKey()

# Change BGR to RGB
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
xscalefactor = int(0.15*(w)) #Reduces Horizontal Window by 20%

#print(img2.shape)
# Make the Mask
mask = np.zeros(img2.shape[:2], np.uint8)
mask[:,:] = 0
rectangle = cv2.rectangle(mask,(x+xscalefactor,y),(x+w-xscalefactor,y+h),(255,255,255),-1)
masked_img = cv2.bitwise_and(img2,img2,mask = rectangle) #mask
#print(x+xscalefactor)
#print(x+w-xscalefactor)
#print(y)
#rint(y+h)

# maskSLIC result
m_slic = segmentation.slic(img, n_segments = 2, compactness = 10, mask=mask, start_label=1, convert2lab=1)
#print(np.matrix(m_slic))
a = np.array(m_slic)
#print(a.shape)
unique, counts = np.unique(a, return_counts=True)
#print(dict(zip(unique, counts)))

count = 0
topTotal = 0
bottomTotal = 0
half = h/2
half = round(half)

for i in range(x+xscalefactor, x+w-xscalefactor):
    for j in range(y, y+half):
        topTotal+=a[j,i]
        count+=1


topAvg = topTotal / count
#print(topAvg, topTotal, count)
count = 0

for i in range(x+xscalefactor, x+w-xscalefactor):
    for j in range(y+half, y+h):
        bottomTotal += a[j,i]
        count+=1

bottomAvg = bottomTotal / count
#print(bottomAvg, bottomTotal, count)
print(abs(topAvg-bottomAvg))











