from skimage.feature import canny
from skimage.morphology import closing
from PIL import Image
import skimage.io
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread("data/test3.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
colors = ('b','g','r')
faces = face_cascade.detectMultiScale(gray, 1.1, 10)

for (x, y, w, h) in faces:
    mask = np.zeros(img.shape[:2], np.uint8)
    mask_top = np.zeros(img.shape[:2], np.uint8)
    mask_bottom = np.zeros(img.shape[:2], np.uint8)
    mask_top[:, :] = 0
    mask_bottom[:, :] = 0
    midPoint = int(y + h/2)
    rectangle = cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    top = cv2.rectangle(mask_top, (x, y), (x + w, midPoint), (255, 255, 255), -1)
    bottom = cv2.rectangle(mask_bottom, (x, midPoint), (x + w, y + h), (255, 255, 255), -1)
    masked_top = cv2.bitwise_and(gray,gray,mask = top)
    masked_bottom = cv2.bitwise_and(gray,gray,mask = bottom)
    masked = cv2.bitwise_and(gray,gray,mask = rectangle)
    # hist = cv2.calcHist([img],[0],mask,[256],[0,256])
    for i,col in enumerate(colors):
        histr = cv2.calcHist([img],[i],top,[256],[0,256])
        plt.subplot(222), plt.plot(histr,color = col)
        plt.xlim([0,256])

    for i,col in enumerate(colors):
        histr = cv2.calcHist([img],[i],bottom,[256],[0,256])
        plt.subplot(224), plt.plot(histr,color = col)
        plt.xlim([0,256]) 
    plt.subplot(221), plt.imshow(masked, 'gray')
    # im = Image.fromarray(masked)
    # im.save('foobar.jpg')
    # plt.subplot(221), plt.imshow(masked_top,'gray')
    # plt.subplot(223), plt.imshow(masked_bottom, 'gray')
    
    # plt.show()




# img = skimage.io.imread("data/test3.jpg", as_gray=True)
fig, ax = plt.subplots(1, 1, figsize=(20,20))
ax.imshow(masked,'gray')
ax.set_axis_off()
# plt.show()

edges = canny(masked)

close = closing(edges)
print(close)
# self.edges = cv2.Canny(self.gray_img, canny_low_thresh, canny_high_thresh, apertureSize=canny_kernel_size)

# detect lines with hough transform
edges = cv2.Canny(masked,180,200)

minLineLength = 30
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
# lines = cv2.HoughLines(edges,1,np.pi/180,15)
if lines is None:
    lines = []
print(lines.shape)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(masked,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('hough',masked)
cv2.waitKey(0)
# self.lines_hough = self._generate_hough_lines(lines)

# return self.lines_hough 

fig, ax = plt.subplots(1, 1, figsize=(20,20))
ax.imshow(close,'gray')
ax.set_axis_off()
# plt.show()