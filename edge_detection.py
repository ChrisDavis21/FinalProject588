import cv2
import numpy as np  
import os
from scipy.ndimage.measurements import label

def detect(file, filename):
    #load image as grayscale
    img = cv2.imread(file,0)
    # detect lines with hough transform
    edges = cv2.Canny(img,180,200)

    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength=minLineLength,maxLineGap=maxLineGap)
    if lines is None:
        print("no lines found, assuming no mask")
        return False
    line_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            if abs(y2-y1) < 20:
                cv2.line(line_img,(x1,y1),(x2,y2),(255,255,255),2)

    kernel = np.ones((15,15))
    # do a morphologic close
    line_img = cv2.morphologyEx(line_img,cv2.MORPH_CLOSE, kernel)
    line_img[line_img!=0] = 1

    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(line_img, structure)
    indices = np.indices(line_img.shape).T[:,:,[1, 0]]

    largest_line = 0
    foo = indices[labeled == 1]
    for i in range(1, len(np.unique(labeled))):
        idxes = indices[labeled == i]
        min_ = np.min(idxes[:, 1])
        max_ = np.max(idxes[:, 1])
        length = max_ - min_ 
        if length > largest_line:
            largest_line = length


    percent = largest_line / img.shape[1] 
    if percent > 0.8:
        print("mask found!")
        return True
    else:
        print("no mask found!")
        return False


directory = r'C:\Users\Rob\Documents\College\ECE 588\FinalProject588\data\cropped'

misclassify = []
for filename in os.listdir(directory):
    res = detect(os.path.join(directory, filename), filename)
    mask = filename.startswith("mask")
    if mask and not res or not mask and res:
        misclassify.append(filename)

accuracy = 1 - (len(misclassify) / len(os.listdir(directory)))
print("Accuracy of", accuracy)