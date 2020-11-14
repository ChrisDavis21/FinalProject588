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
   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
   print(1)

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
color = ('r','g','b')
for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],mask,[256],[0,256])
    plt.subplot(324), plt.plot(histr,color = col)
    plt.xlim([0,256])
    
    
    
mask2 = np.zeros(img2.shape[:2], np.uint8)
mask2[:,:] = 0
rectangle2 = cv2.rectangle(mask2,(x+xscalefactor,anewy),(x+w-xscalefactor,y+h),(255,255,255),-1)
masked_img2 = cv2.bitwise_and(img2,img2,mask = rectangle2) #mask

# Calculate histogram with mask and without mask
# Check third argument for mask
color = ('r','g','b')
for i,col in enumerate(color):
    histr2 = cv2.calcHist([img2],[i],mask2,[256],[0,256])
    plt.subplot(326), plt.plot(histr2,color = col)
    plt.xlim([0,256])

    
#hist_full = cv2.calcHist([img2],[0],None,[256],[0,256])

plt.subplot(321), plt.imshow(img2)
plt.subplot(322), plt.imshow(mask,'gray')
plt.subplot(323), plt.imshow(masked_img, 'gray')
plt.subplot(325), plt.imshow(masked_img2, 'gray')
plt.show()