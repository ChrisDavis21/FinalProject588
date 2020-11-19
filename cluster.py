import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
#np.set_printoptions(threshold=sys.maxsize)
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

# Draw rectangle around the faces
for (x, y, w, h) in faces:
   # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
   print(1)

# Display the output
cv2.waitKey()

# Change BGR to RGB
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
xscalefactor = int(0.15*(w)) #Reduces Horizontal Window by 20%
yscale = int(0.15*(h))
# Make the Mask
mask = np.zeros(img2.shape[:2], np.uint8)
mask[:,:] = 0
rectangle = cv2.rectangle(mask,(x+xscalefactor,y),(x+w-xscalefactor,y+h),(255,255,255),-1)
masked_img = cv2.bitwise_and(img2,img2,mask = rectangle) #mask
cropped = img[y:y+h, x+xscalefactor:x+w-xscalefactor]
# SLIC result
slic = segmentation.slic(cropped, compactness = 20, start_label=1)

# maskSLIC result
m_slic = segmentation.slic(cropped, n_segments = 2, compactness = 0.01, start_label=1, convert2lab=1)
print(np.matrix(m_slic))
a = np.array(m_slic)
unique, counts = np.unique(a, return_counts=True)
print(dict(zip(unique, counts)))

# Display result
fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
ax1, ax2, ax3, ax4 = ax_arr.ravel()

ax1.imshow(cropped)
ax1.set_title("Origin image")

ax2.imshow(mask, cmap="gray")
ax2.set_title("Mask")

ax3.imshow(segmentation.mark_boundaries(cropped, slic))
#ax3.contour(mask, colors='red', linewidths=1)
ax3.set_title("SLIC")

ax4.imshow(segmentation.mark_boundaries(cropped, m_slic))
#ax4.contour(mask, colors='red', linewidths=1)
ax4.set_title("maskSLIC")

for ax in ax_arr.ravel():
    ax.set_axis_off()

plt.tight_layout()
plt.show()